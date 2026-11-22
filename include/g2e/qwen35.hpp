// Qwen3.5/3.8 hybrid text model (GDN + gated attention) on g2e graphs.
//
// Layers are launched in blocks (default 4 = the model's GDN,GDN,GDN,attention period): one recipe per
// (block layer pattern, token count T, attention context bucket) is shared by every block with that pattern;
// weights, caches and states are bound by address at launch.
//
// Residual stream: every layer consumes (res, delta) and produces (res', delta'); the fused add+RMSNorm+FP8-quant
// kernel computes s = res + delta at the start of each layer and after the mixer.
// GDN (T == 1): spec_conv + gdn_packed_decode kernels, recurrent state updated in place ([HV, DV, DK] f32).
// GDN (T > 1): reference-grade unrolled recurrence (prefill; replaced in M5).
// Attention: head-major BF16 K/V cache per layer [KV, max_context, D]; each head's prefix of `bucket` rows is
// bound as its own tensor, the new rows are written by the graph straight into the cache.
#pragma once

#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace g2e {

struct Qwen35Config {
    uint32_t layers = 0, hidden = 0, intermediate = 0, vocab = 0;
    uint32_t heads = 0, kv_heads = 0, head_dim = 0, rotary_dim = 0;
    double rope_theta = 0;
    uint32_t lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_kernel = 0;
    float eps = 1e-6f;
    std::vector<uint32_t> mrope_section;  // multimodal RoPE (t, h, w) sections, interleaved; empty: text only
    std::vector<bool> attention;  // per layer: true = full attention, false = GDN
    static Qwen35Config from_pack(const WeightPack& pack);
    uint32_t conv_channels() const { return 2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim; }
};

class Qwen35 {
public:
    Qwen35(const Device& device, const Stream& stream, const WeightPack& weights, uint32_t max_context,
           uint32_t block_layers = 4);
    ~Qwen35();
    const Qwen35Config& config() const { return cfg_; }
    uint32_t position() const { return pos_; }
    void reset();
    // Sequence slots (several requests resident at once, run one at a time): every slot owns its KV caches (its
    // own capacity, <= max_context), GDN states and position; select_slot() swaps them in (no copies). Slot 0 is
    // the one built by the constructor. No verification may be pending when switching.
    uint32_t add_slot(uint32_t capacity);
    void select_slot(uint32_t slot);
    uint32_t slot() const { return active_slot_; }
    uint32_t slots() const { return uint32_t(seqs_.size()); }
    uint32_t capacity() const { return cap_; }  // context rows of the selected slot
    // dynamic KV caches: set the selected slot's capacity (a value of round_capacity, <= the max context; growing
    // copies the rows held, layer by layer; 0 frees its caches and resets the slot)
    void set_capacity(uint32_t capacity);
    // the capacity to allocate for `rows` (>= rows): an attention bucket (no new recipe shapes), at least 16384
    uint32_t round_capacity(uint32_t rows) const;
    uint64_t kv_bytes_for(uint32_t capacity) const;  // the KV caches of one slot of that capacity
    uint64_t slot_kv(uint32_t slot) const;           // allocated KV bytes of a slot
    uint64_t kv_total() const;                       // of every slot
    // forget a verification that will not be accepted (an error between verify() and accept(); the states were
    // not touched, the rows past the position are masked)
    void clear_pending() {
        if (pending_T_ && pending_total_) --pending_total_;
        pending_T_ = 0;
        pending_tree_N_ = 0;
    }
    // Prefix cache support (g2e/prefix_cache.hpp). All copies are queued on the model's stream (ordered with the
    // forwards); host buffers must stay alive until the stream passed them.
    // KV rows [r0, r0 + n) of every attention layer, host layout [slot][k|v][head][n][D] bf16
    uint64_t kv_row_bytes() const;
    void save_kv(uint32_t r0, uint32_t n, const HostBuffer& dst, uint64_t off) const;
    void load_kv(uint32_t r0, uint32_t n, const HostBuffer& src, uint64_t off);
    // the recurrent states (GDN conv + ssm, current sides); load_state also sets the position (after reset())
    uint64_t state_bytes() const;
    void save_state(const HostBuffer& dst, uint64_t off) const;
    void load_state(const HostBuffer& src, uint64_t off, uint32_t pos);
    const Stream& stream() const { return stream_; }
    // Appends `tokens` at the current position and returns the logits of the last one. With `dump`, the residual
    // stream (res + delta, BF16 [T, hidden]) after every block is appended (per layer when block_layers == 1).
    std::vector<float> forward(std::span<const int32_t> tokens, std::vector<std::vector<uint16_t>>* dump = nullptr,
                               bool logits = true);
    // A short piece (2..15 tokens) in one forward: a 16-row verification (the rows past it padded with its last
    // token) whose accept() commits exactly its rows; the logits of its last row (when wanted) from the verified
    // rows. Replaces the power-of-two pieces (up to 4 forwards, each reading every weight and the whole context).
    std::vector<float> forward_short(std::span<const int32_t> tokens, bool logits = true);
    // A non-power-of-two prefill piece (17..511 tokens) in one forward padded to T_pad (32..512): GDN rows >= n
    // get zero beta/decay so recurrent state S is mathematically exact, conv_out is gathered at [n..n+KC-2], and
    // head_recipe reads row n - 1. Eliminates multiple tail forwards on agent turns and unaligned prompts.
    std::vector<float> forward_padded(std::span<const int32_t> tokens, uint32_t T_pad, bool logits = true);
    // Speculative verification: run `tokens` (anchor + drafts) at the current position and return each row's
    // greedy next token. Nothing is committed except scratch; accept(a) then commits rows 0..a (a+1 tokens):
    // the GDN states are recomputed for the accepted prefix, the conv windows re-gathered, pos += a + 1.
    // `launched` (optional) runs once all of the verification is queued, before waiting for its results: work
    // queued there (the drafter's context append) overlaps the host's wait.
    std::vector<int32_t> verify(std::span<const int32_t> tokens, const std::function<void()>& launched = {});
    void accept(uint32_t accepted);
    // Batched verification: slots[b] verifies blocks[b] (8 or 16 tokens, the same for all: anchor + drafts) at its own position, all
    // in one forward (the dense layers read their weights once for every slot; attention runs per slot over its
    // own cache, the GDN recurrences per slot from its own states). Returns each block's greedy targets. Every
    // slot then has a pending verification: select it and accept(a), in any order. Slot b's captured features
    // are rows b * kSpecMaxT .. of features(). 2 <= slots <= kMaxBatch, distinct, no verification pending.
    static constexpr uint32_t kMaxBatch = 8;
    std::vector<std::vector<int32_t>> verify_batch(std::span<const uint32_t> slots,
                                                   std::span<const std::vector<int32_t>> blocks,
                                                   const std::function<void()>& launched = {});
    // verify_batch with the sampling head: per slot the VerifyTop of its rows at its own 1 / temperature (inv_temps[b];
    // a greedy slot passes 1 and takes each row's top value). The pending verifications as verify_batch's.
    struct VerifyTop;  // below
    std::vector<VerifyTop> verify_batch_topk(std::span<const uint32_t> slots, std::span<const std::vector<int32_t>> blocks,
                                             std::span<const float> inv_temps, const std::function<void()>& launched = {});
    // Batched prefill: slots[b] appends pieces[b] (T tokens each, the same T for all, B * T <= 2048, T a multiple of
    // 64) at its own position. The dense layers once for all rows; attention per slot over its own cache; the GDN
    // recurrences of all slots as one chunked delta rule (the slots' states as extra heads: the serial inter-chunk
    // chain, ~40% of a prefill piece's MME time, is shared). Returns per slot its last row's logits when
    // want_logits[b]. Slot b's captured features are rows b * T .. of features(). prefill_batchable() first.
    std::vector<std::vector<float>> prefill_batch(std::span<const uint32_t> slots, std::span<const std::vector<int32_t>> pieces,
                                                  std::span<const uint8_t> want_logits);
    bool prefill_batchable(uint32_t slot, uint32_t T) const;  // the slot's context fits the batched attention
    static constexpr uint32_t kPrefillBatchMaxRows = 16384;   // its attention rows (acore buckets) at most
    // accept() of several slots' pending batched verifications in one launch (slots[b] keeps accepted[b] + 1 rows)
    void accept_batch(std::span<const uint32_t> slots, std::span<const uint32_t> accepted);
    // Sampled verification: per row the kTopK largest temperature-scaled logits (ids, values) and the full-vocabulary
    // softmax statistics (max, sum of exp(x - max)), enough for exact probabilities of those candidates.
    static constexpr uint32_t kTopK = 64;
    static constexpr uint32_t kTopkChunk = 128;  // Ops::topk2 chunk of the verification top-k
    static constexpr uint64_t kAcceptStage = 512u << 10;  // staging offset of accept()'s controls
    static constexpr uint64_t kIdsStage = 768u << 10;     // token ids of a forward (ids_ layout)
    static constexpr uint64_t kTempStage = kIdsStage + (128u << 10);  // verify_topk's 1/T
    struct VerifyTop {
        std::vector<int32_t> ids;     // [T, kTopK]
        std::vector<float> values;    // [T, kTopK] scaled logits
        std::vector<float> max, sum;  // [T]
    };
    VerifyTop verify_topk(std::span<const int32_t> tokens, float inv_temp, const std::function<void()>& launched = {});
    // Tree speculative verification: verify an N-node prefix tree (N <= 64, topological order: parent[0] = -1,
    // 0 <= parent[i] < i, depth[i] <= 15) sharing the current slot's prefix KV & GDN state in a single pass.
    struct TreeSpec {
        std::vector<int32_t> tokens;    // [N] tokens[0] = anchor
        std::vector<int32_t> parent;    // [N] parent[0] = -1, 0 <= parent[i] < i
        std::vector<uint32_t> depth;    // [N] depth[0] = 0, depth[i] = depth[parent[i]] + 1
    };
    std::vector<int32_t> verify_tree(const TreeSpec& tree);
    VerifyTop verify_tree_topk(const TreeSpec& tree, float inv_temp);
    // Commit winning root-to-node path `win_nodes` (1..16 nodes, win_nodes[0] = 0) of the pending tree verification:
    // updates GDN states, GDN conv windows, attention KV caches, and gathers captured features into tree_features().
    void accept_tree(std::span<const int32_t> win_nodes);
    const DeviceBuffer& tree_features() const { return tree_feats_; }
    uint64_t tree_feature_stride() const { return uint64_t(kSpecMaxT) * cfg_.hidden * 2; }
    bool tree_supported() const;
    // Whole prompt: chunks of 64 tokens, then the remainder as 32/16/8/4/2/1-token pieces. Returns the last logits.
    // `on_chunk(pos, n)` runs after every chunk (the captured features of those n rows are in features()).
    // Build (or load from the recipe cache) every recipe a request up to max_context can need: prefill pieces,
    // decode, verification, for every attention context bucket. `progress(done, total)` after each.
    void precompile(const std::function<void(size_t, size_t)>& progress = {});
    // An image in a prompt: its placeholder tokens start at `start` (relative to the prefill's tokens), one per
    // merged patch of the grid (LLM grid: patches / merge), and its features (vision encoder output, BF16
    // [grid_h * grid_w, hidden]) replace their embeddings. Positions are the 3D (t, h, w) multimodal RoPE ones;
    // text after an image continues from start + max(grid_h, grid_w).
    struct ImageSpan {
        uint32_t start = 0, grid_h = 0, grid_w = 0;
        const DeviceBuffer* feats = nullptr;
        uint64_t offset = 0;
        uint32_t tokens() const { return grid_h * grid_w; }
    };
    std::vector<float> prefill_images(std::span<const int32_t> tokens, std::span<const ImageSpan> images,
                                      const std::function<void(uint32_t, uint32_t)>& on_chunk = {});
    std::vector<float> prefill(std::span<const int32_t> tokens,
                               const std::function<void(uint32_t, uint32_t)>& on_chunk = {});
    // Capture the residual stream after these layers for every row of every forward/verify:
    // feature k of row r at features() + k * feature_stride() + r * hidden * 2 (BF16).
    void set_capture(std::vector<uint32_t> layers);
    const DeviceBuffer& features() const { return feats_; }
    uint64_t feature_stride() const { return uint64_t(kMaxT) * cfg_.hidden * 2; }
    double last_device_ms() const { return last_ms_; }

    // Debug: the block containing `layer` gets extra outputs (dbg_*) for that layer's intermediates.
    void set_debug_layer(int layer) { debug_layer_ = layer; }
    void set_debug_points(std::vector<std::string> points) { debug_points_ = std::move(points); }
    void set_layer_limit(uint32_t n) { layer_limit_ = n; }  // run only the first n layers (multiple of the block)
    struct DebugTensor { std::vector<uint64_t> shape; std::vector<float> values; };
    const std::map<std::string, DebugTensor>& debug_tensors() const { return debug_out_; }
    void set_profile(bool on) { profile_ = on; }
    struct Profile { double blocks_ms = 0, head_ms = 0, commit_ms = 0; std::map<std::string, double> by_recipe; };
    const Profile& last_profile() const { return prof_; }

private:
    struct Pair { TensorId res, delta; };
    Recipe& block_recipe(uint32_t first_layer, uint32_t T, uint32_t bucket);
    // layers in the launch starting at `first`: block_, or with one-layer blocks a run of up to G2E_GDN_GROUP (3)
    // consecutive GDN layers in one launch (fewer launches; attention layers stay alone: their split launches)
    uint32_t group_at(uint32_t first, uint32_t nlayers) const;
    uint32_t launch_n_ = 0;  // block_recipe's layer count while building / binding (0: block_)
    Recipe& head_recipe();
    Recipe& commit_recipe(uint32_t T);
    Recipe& spec_commit_recipe(uint32_t T, bool raw_qk);
    Recipe& spec_commit_batch_recipe(uint32_t B, uint32_t T, bool raw_qk);
    bool wy_geometry() const;  // the GDN head geometry of the WY verification kernels
    // FP8 KV caches (G2E_KV_FP8=1): E4M3 with unit scales (as vLLM's uncalibrated FP8 KV), half the bytes. Written
    // as FP8 by the attention launches; the verification / decode attention multiplies FP8 q / K and FP8 P (x128) /
    // V on the MME, the FP8 prefill SDPA takes them as they are, the short-context paths convert to BF16.
    bool kv8_ = false;
    DType kv_dt() const { return kv8_ ? DType::F8 : DType::BF16; }
    uint64_t kv_es() const { return kv8_ ? 1 : 2; }  // bytes per KV element
    TensorId kv_to_f8(Ops& o, TensorId bf16, float mult, const std::string& name);  // E4M3 codes of x * mult
    TensorId kv_in_bf16(Ops& o, const std::string& name, uint64_t rows);         // a cache input, as BF16
    TensorId kv_scores(Ops& o, TensorId q, TensorId kc, const std::string& name);  // q [R,D] x K [Sb,D]^T (BF16)
    TensorId kv_pv(Ops& o, TensorId p, TensorId vc3, const std::string& name);     // P [NC,R,C] x V [NC,C,D] -> F32 (x128 if FP8)
    Recipe& verify_head_recipe(uint32_t T);
    Recipe& verify_topk_recipe(uint32_t T);
    Recipe& capture_recipe(uint32_t T);
    Recipe& embed_recipe(uint32_t T);
    bool spec_wy(uint32_t T) const;
    bool attn_batched() const;
    bool attn_sdpa(uint32_t Sb) const;
    // Long contexts: an attention layer as two launches. The first writes its new K/V rows straight into the
    // caches (and q, the gate and the residual into scratch), the second runs SDPA over cache rows [0, bucket)
    // (no copy of the cache) and the MLP. Used when bucket_for(pos + T) takes SDPA and the block is one layer.
    bool attn_split(uint32_t T) const;
    Recipe& attn_pre_recipe(uint32_t T);
    Recipe& attn_post_recipe(uint32_t T, uint32_t Sb);
    // Long-context SRAM-tiled FP8 prefill SDPA (Pn = Sb - W > 8192): reusable O(1) chunk recipes (6144/4096/8192)
    // + one Sb-independent tail+MLP recipe, keeping SDPA tiles inside 48 MiB SRAM (~1.55x faster at 65K-262K).
    bool fp8_chunked(uint32_t T, uint32_t Sb) const;
    Recipe& attn_fp8_chunk_recipe(uint32_t T, uint32_t C, bool is_first);
    Recipe& attn_post_chunked_tail_recipe(uint32_t T);
    void ensure_fp8_chunk_bufs();
    // batched verification (verify_batch): B slots x kSpecMaxT rows
    Recipe& gdn_batch_recipe(uint32_t B, uint32_t T);
    Recipe& apre_batch_recipe(uint32_t B, uint32_t T);
    Recipe& acore_recipe(uint32_t T, uint32_t Sb);  // one slot's attention: q [T, NH, D] over its cache -> [T, NH * D]
    Recipe& apost_batch_recipe(uint32_t B, uint32_t T);
    Recipe& gdn_prefill_batch_recipe(uint32_t B, uint32_t T);
    Recipe& verify_topkb_recipe(uint32_t R);  // verify_topk_recipe with 1 / temperature per row
    struct BatchOut {
        std::vector<std::vector<int32_t>> targets;
        std::vector<VerifyTop> tops;
    };
    BatchOut batch_verify(std::span<const uint32_t> slots, std::span<const std::vector<int32_t>> blocks,
                          std::span<const float> inv_temps, const std::function<void()>& launched);
    DeviceBuffer vtopb_, tempb_;  // batched sampling head: values / ids / max / sum (kVtb*), 1 / temperature per row
    static constexpr uint64_t kVtbIds = 32768, kVtbMax = 65536, kVtbSum = 66048, kVtbEnd = 66560;
    static uint32_t core_rows(uint32_t rows);  // acore's context rows: 1024, 2048, then multiples of 2048
    // every slot's attention in one launch (rows padded to a common bucket: core_bucket); 0 = none fits
    Recipe& acore_batch_recipe(uint32_t B, uint32_t T, uint32_t Sb);
    static uint32_t core_bucket(uint32_t rows);  // 1024, 2048, 3072, 4096, 6144, 8192, 12288, ... (x1.5 / x2 steps)
    DeviceBuffer att_;  // batched attention outputs [kMaxBatch * kSpecMaxT, NH * D]
    DeviceBuffer apre_q_, apre_g_, apre_r_;  // scratch between the two launches
    std::unique_ptr<DeviceBuffer> fp8_q8_, fp8_o_[2], fp8_m_[2], fp8_l_[2];  // lazy ping-pong accumulators for chunked FP8 SDPA
    bool fused_gate_quant() const;
    bool gated_kernel() const;
    uint32_t gdn_chunk() const;
    void run_layers(std::span<const int32_t> tokens, bool spec, std::vector<std::vector<uint16_t>>* dump,
                    uint32_t valid_n = 0);
    // field-major stash: field f of GDN layer s at base[f] + s * size[f] (fields q, k, v, g, b, seq)
    struct StashLayout { uint64_t base[6], size[6]; };
    StashLayout stash_layout(uint32_t T, bool wy) const;  // wy: X = [k; q] in field 0 (the WY verification)  // scratch K/V rows of every attention layer -> caches
    Pair gdn_layer(Ops& o, uint32_t j, uint32_t T, Pair in, uint32_t B = 1);  // B > 1: batched verification
    Pair attention_layer(Ops& o, uint32_t j, uint32_t T, uint32_t bucket, Pair in);
    Pair mlp(Ops& o, uint32_t j, TensorId y, TensorId s);
    Pair proj_mlp(Ops& o, uint32_t j, TensorId codes, TensorId scale, TensorId w, TensorId ws, TensorId s,
                  const std::string& proj_name, bool fold_out_dq);
    // Chunked gated delta rule over C (power of two, >= 8) rows; state [HV, DV, DK]; returns core [C, HV, DV] f32.
    // q,k [C,HK,DK], v [C,HV,DV], g/beta [C,HV] f32, state [HV,DV,DK]; HK/HV may cover several layers at once.
    std::pair<TensorId, TensorId> gdn_chunk(Ops& o, const std::string& prefix, uint32_t C, uint64_t HK, uint64_t HV,
                                            TensorId state, TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta);
    std::map<std::string, TensorId> build_inputs_;  // persistent inputs shared within one recipe build
    void dbg(Ops& o, uint32_t j, const std::string& name, TensorId t);
    uint32_t bucket_for(uint32_t rows) const;
    static constexpr uint32_t kAttnFine = 2048, kAttnStep = 2048;  // fine attention rows (attn_rows)
    uint32_t attn_rows(uint32_t rows, uint32_t T) const;
  public:
    // stderr: the device memory held, the KV cache, the largest workspaces among the compiled recipes
    void memory_report() const;
  private:
    bool fp8_prefill(uint32_t T, uint32_t Sb) const;
    uint32_t fp8_tail(uint32_t T) const;
    void prepare_aux();

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    Qwen35Config cfg_;
    uint32_t max_ctx_, block_, pos_ = 0;
    uint32_t cap_ = 0;  // the selected slot's context rows (KV stride); max_ctx_ sizes the shared tables
    struct Seq {  // a slot's state while it is not selected
        std::vector<DeviceBuffer> k, v;
        DeviceBuffer conv[2], ssm[2];
        std::vector<int> conv_side, ssm_side;
        uint32_t pos = 0, cap = 0;
        int64_t rope_delta = 0;
        uint32_t pending_T = 0, stash_b = 0;
        bool pending_wy = false;
    };
    std::vector<Seq> seqs_;      // seqs_[active_slot_] is empty: its state is in the members
    uint32_t active_slot_ = 0;
    void swap_seq(Seq& s);
    static constexpr uint32_t kMaxT = 4096;     // tokens per forward (prefill chunk)
    // ids_ layout: token ids [kMaxT] i32 at 0, pos - 1 (f32) at kIdsPosm1, last visible rows [T] i32 at kIdsLast
    static constexpr uint64_t kIdsPosm1 = uint64_t(kMaxT) * 4, kIdsPosm1H1 = kIdsPosm1 + 128, kIdsLast = kIdsPosm1H1 + 128;
    static constexpr uint64_t kIdsRopePos = kIdsLast + uint64_t(kMaxT) * 4;  // batched forwards: every row's RoPE position [rows] i32
    static constexpr uint64_t kIdsPadMask = kIdsRopePos + uint64_t(kMaxT) * 4;  // padded prefill: [T, 1] f32 mask (1 valid, 0 pad)
    static constexpr uint64_t kIdsConvRows = kIdsPadMask + uint64_t(kMaxT) * 4; // padded prefill: [KC - 1] i32 gather indices
    static constexpr uint32_t kSpecMaxT = 16;   // verification width
    std::map<std::string, std::unique_ptr<Recipe>> recipes_;
    std::unique_ptr<Recipe> head_;
    // kv_scratch_: [attention layers, 2, KV, T, D] new rows of this step; spec_*: verify stash and commit controls
    DeviceBuffer kv_scratch_, spec_stash_, spec_ctl_, argmax_, feats_, vtop_;
    std::vector<uint32_t> capture_;
    uint32_t pending_T_ = 0;
    uint32_t stash_b_ = 0;        // stash / control region of the selected slot's pending verification
    bool pending_wy_ = false;     // its stash layout (WY: X = [k; q])
    uint32_t pending_total_ = 0;  // slots with a pending verification (all of them must accept first)
    static constexpr uint64_t kBestOff = 8192;  // argmax_: cand8 [rows, kHeadCand] at 0, best [rows] here
    DeviceBuffer ids_;                     // token ids, pos - 1, last visible rows (kIdsPosm1, kIdsLast)
    // multimodal RoPE: the rotary position of cache row p is p + rope_delta_ for text; a forward with image
    // tokens takes explicit rows (rope_rows_, uploaded into rope_cs_: cos [kMaxT, R2], then sin)
    int64_t rope_delta_ = 0;
    bool rope_explicit_ = false;
    std::vector<float> rope_rows_;
    DeviceBuffer rope_cs_;
    std::vector<uint32_t> mrope_section_;  // interleaved (t, h, w) frequency sections
    struct Inject { const DeviceBuffer* src; uint64_t src_offset; uint32_t row, rows; };
    std::vector<Inject> inject_;  // rows of the next forward's embeddings to overwrite (image features)
    std::unique_ptr<Event> done_ev_;       // after a verification's result download
    // greedy verification head: lm_head quantised per row to FP8 picks kHeadCand candidates per row, their BF16
    // rows give the exact logits (G2E_HEAD_BF16=1: the full BF16 product)
    static constexpr uint32_t kHeadCand = 8;
    DeviceBuffer lm8_, lm8s_;
    bool lm8_on_ = false;
    void quantize_lm_head();
    std::unique_ptr<Event> stage_ev_;      // after run_layers' uploads (the next one rewrites those regions)
    bool zeroed_ = false;  // rows of the last verify, 0 when nothing is pending
    int cur_ = 0;             // residual-stream side holding the last forward's output
    DeviceBuffer res_[2], delta_[2], zero_, logits_, consts_, aux_;
    // RoPE tables for every position [max_ctx + kMaxT, rotary/2] f32: a forward binds the rows of its positions
    DeviceBuffer cos_tab_, sin_tab_;
    TensorId mask_t_ = kNoTensor;  // the attention mask of the recipe being built (attn_mask)
    TensorId attn_mask(Ops& o, uint32_t T, uint32_t Sb);
    std::map<std::string, uint64_t> aux_off_;               // host-prepared weights in aux_
    std::vector<DeviceBuffer> k_cache_, v_cache_;          // per attention layer, [KV, max_ctx, D]
    // GDN conv windows [NG, KC-1, C] bf16 and recurrent states [NG, HV, DV, DK] f32, ping-pong sides
    DeviceBuffer conv_all_[2], ssm_all_[2];
    uint64_t conv_bytes_ = 0, ssm_bytes_ = 0;
    uint32_t num_gdn_ = 0;
    std::vector<int> conv_side_, ssm_side_;
    std::vector<int> slot_;                                // layer -> index within its kind
    HostBuffer staging_;
    double last_ms_ = 0;
    int debug_layer_ = -1;
    int building_debug_j_ = -1;
    TensorId idx_t_ = kNoTensor, acc_t_ = kNoTensor;
    bool building_spec_ = false;  // block-shared i32 constants while building
    bool building_padded_ = false;  // masked GDN prefill recipe (pad_mask + conv_rows)
    std::vector<std::string> debug_points_;
    uint32_t layer_limit_ = 0;
    std::map<std::string, DeviceBuffer> dbg_bufs_;
    std::unique_ptr<DeviceBuffer> trace_buf_;  // G2E_TRACE
    std::map<std::string, DebugTensor> debug_out_;
    bool profile_ = false;
    Profile prof_;
    std::vector<std::unique_ptr<Event>> prof_ev_;
    // Tree speculative verification (N <= 64 unique tree nodes, 1x S0, 1x prefix KV read)
    Recipe& gdn_tree_recipe(uint32_t N);
    Recipe& atree_post_recipe(uint32_t N, uint32_t W, bool has_prefix);
    Recipe& tree_commit_recipe(uint32_t N);
    StashLayout tree_stash_layout(uint32_t N) const;
    void ensure_tree_bufs(uint32_t N);
    void run_tree_layers(const TreeSpec& tree);
    static constexpr uint64_t kTreeStage = 2u << 20;
    static constexpr uint64_t kTreeCommitStage = 3u << 20;
    static constexpr uint64_t kTreeOffParent = 0;
    static constexpr uint64_t kTreeOffConvIdx = 256;
    static constexpr uint64_t kTreeOffTailPos = 1280;
    static constexpr uint64_t kTreeOffMask = 1408;
    static constexpr uint64_t kTreeOffPathCtl = 17792;
    static constexpr uint64_t kTreeOffWin16 = 17920;
    static constexpr uint64_t kTreeOffConvRows = 18048;
    static constexpr uint64_t kTreeCtlBytes = 20480;
    static constexpr uint64_t kDTreeOffIds = 0;
    static constexpr uint64_t kDTreeOffParent = 256;
    static constexpr uint64_t kDTreeOffConvIdx = 512;
    static constexpr uint64_t kDTreeOffPosInt = 1536;
    static constexpr uint64_t kDTreeOffRopePos = 1664;
    static constexpr uint64_t kDTreeFwdBytes = 1920;
    static constexpr uint64_t kDTreeOffCommitBase = 2048;
    static constexpr uint64_t kDTreeOffPathCtl = 2048;
    static constexpr uint64_t kDTreeOffConvRows = 2176;
    static constexpr uint64_t kDTreeOffCommitPos = 2304;
    static constexpr uint64_t kDTreeCommitBytes = 384;
    static constexpr uint64_t kDTreeCtlBytes = 4096;
    DeviceBuffer tree_ctl_, tree_direct_ctl_, tree_feats_;
    uint32_t pending_tree_N_ = 0;
    bool last_tree_direct_ = false;
    double last_tree_direct_ms_ = 0.0;

    struct DirectShared;
    struct DirectBatchState;
    struct DirectTreeState;
    std::unique_ptr<DirectShared> direct_shared_;
    std::unique_ptr<DirectBatchState> direct_b8_;
    std::unique_ptr<DirectBatchState> direct_b1_;
    std::unique_ptr<DirectTreeState> direct_tree_;
    bool last_used_direct_b1_ = false;
    bool ensure_direct_shared(uint32_t nl);
    bool ensure_direct_plan(std::unique_ptr<DirectBatchState>& dst, uint32_t B, std::array<uint32_t, 8> slot_caps);
    bool ensure_direct_b8(std::array<uint32_t, 8> slot_caps);
    DirectBatchState* ensure_direct_b1(uint32_t cur_pos);
    DirectTreeState* ensure_direct_tree(uint32_t N);
    void clear_direct_attn_caches();
};

}  // namespace g2e
