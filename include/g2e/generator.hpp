// Request execution on one Qwen35 (+ optional DSpine drafter): prefix reuse across requests, greedy speculative
// decoding with the drafter, sampled decoding token by token, stop tokens, max_tokens, cancellation.
#pragma once

#include <chrono>
#include <memory>
#include "g2e/dspine.hpp"
#include "g2e/qwen35.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace g2e {

struct SamplingParams {
    float temperature = 1.0f;  // <= 1e-5: greedy
    float top_p = 1.0f;
    int top_k = 0;             // 0: off
    float min_p = 0.0f;
    int64_t seed = -1;
    uint32_t max_tokens = 16;
    std::vector<int32_t> stop_token_ids;  // extra stop tokens (reported as stop_reason)
    bool ignore_eos = false;
    bool greedy() const { return temperature <= 1e-5f; }
};

struct GenerationResult {
    std::vector<int32_t> tokens;  // generated tokens (a stop token included)
    std::string finish = "length";  // "stop" | "length" | "abort"
    int32_t stop_token = -1;        // the extra stop token that ended it (-1: none or an EOS token)
    uint32_t prompt_tokens = 0, reused_tokens = 0;
    double prompt_ms = 0, decode_ms = 0;
    uint32_t rounds = 0, drafts_offered = 0, drafts_accepted = 0;
};

class Generator {
public:
    Generator(Qwen35& model, DSpine* drafter, std::vector<int32_t> eos_ids, uint32_t max_context);
    // `on_tokens(new_tokens)` streams committed output; returning false cancels (finish "abort").
    // `images`: image placeholder spans of the prompt (their features already encoded); a prompt with images is
    // never matched against the cached prefix (placeholders of different images are the same tokens).
    GenerationResult generate(const std::vector<int32_t>& prompt, const SamplingParams& params,
                              const std::function<bool(std::span<const int32_t>)>& on_tokens,
                              std::span<const Qwen35::ImageSpan> images = {});
    uint32_t max_context() const { return max_ctx_; }
    void reset();
    // Step-wise execution (several generators, one per model slot, interleaved by a scheduler): start() takes
    // a request, every step() runs one unit (a prefill segment of at most `prefill_tokens` tokens, or one decoding
    // round) on this generator's slot and returns true once the request is finished; take_result() then.
    // generate() = start() + step() until done.
    void set_slot(uint32_t slot) { slot_ = slot; }
    uint32_t slot_index() const { return slot_; }
    void start(const std::vector<int32_t>& prompt, const SamplingParams& params,
               std::function<bool(std::span<const int32_t>)> on_tokens, std::span<const Qwen35::ImageSpan> images = {});
    bool step(uint32_t prefill_tokens = 0xffffffffu);
    bool busy() const { return task_ != nullptr; }
    bool prefilling() const { return task_ && task_->phase != Task::Decode; }
    GenerationResult take_result();
    // drop the running request after an error (the slot starts over with the next one)
    void abort();
    // Batched decoding: one speculative round of several generators (one per slot, all on the same model and
    // drafter) with their blocks verified in one forward (Qwen35::verify_batch). Generators that cannot take part
    // (see batchable(), or no room for a round) run their own step. Returns per generator whether it finished.
    static std::vector<bool> step_batch(std::span<Generator* const> gens);
    // decoding a text request with the drafter (the rounds step_batch can batch: greedy and sampled)
    bool batchable() const;
    // dynamic KV: before the slot's caches grow, reserve(slot, extra bytes) makes room (false: no room); requests
    // then take capacity as they need it, up to max_context()
    void set_kv_reserve(std::function<bool(uint32_t, uint64_t)> reserve) { reserve_ = std::move(reserve); }
    // free the slot's KV caches (idle generator; its state is in the prefix cache when there is one)
    void release();
    // the tokens whose state this generator's slot holds (prefix matching when assigning requests to slots)
    const std::vector<int32_t>& history() const { return history_; }
    // states of earlier prefixes in host memory / on disk (g2e/prefix_cache.hpp); null: only the last history
    void set_prefix_cache(class PrefixCache* cache) { cache_ = cache; }
    // when true, finish() does not save the prefix cache inline; the caller invokes save_prefix() after notifying
    // the client (so SSE [DONE] is never delayed by host KV copies)
    void set_defer_prefix_save(bool defer) { defer_prefix_save_ = defer; }
    void save_prefix();
    // prompt positions where the next generate() also takes prefix-cache snapshots (message boundaries: a later
    // request with the same history but another last message, or a thinking model's next turn, resumes there)
    void set_snapshot_points(std::vector<uint32_t> points) { snap_points_ = std::move(points); }

private:
    struct Task {
        enum Phase { Begin, Prefill, Decode } phase = Begin;
        std::vector<int32_t> prompt;
        SamplingParams params;
        std::function<bool(std::span<const int32_t>)> on_tokens;
        std::vector<Qwen35::ImageSpan> images;
        GenerationResult r;
        uint32_t max_new = 0;
        size_t at = 0;                  // next prompt position to prefill
        std::vector<uint32_t> points;   // snapshot points still ahead
        int32_t anchor = -1;
        bool done = false;
        std::chrono::steady_clock::time_point t0, t1;
    };
    std::unique_ptr<Task> task_;
    double round_ema_ = 4.0;  // recent tokens per speculative round (step_batch's width choice)
    uint32_t slot_ = 0;
    void select();
    std::function<bool(uint32_t, uint64_t)> reserve_;
    bool ensure(uint32_t need, uint32_t want);  // capacity >= need (want: rows worth reserving now)
    uint32_t cap() const { return model_.capacity(); }
    void begin(Task& t);              // prefix reuse / restore, snapshot points
    void prefill_segment(Task& t, uint32_t budget);
    void first_token(Task& t, const std::vector<float>& logits);
    void decode_round(Task& t);
    bool room_for_round(Task& t);  // grows the KV caches for a round; false: finished (no room)
    bool stop_token(const Task& t, int32_t x) const;
    // after a verification: commit the accepted rows, pre-draft the next round, deliver the tokens
    void complete_round(Task& t, uint32_t pos, const std::vector<int32_t>& drafts, const std::vector<int32_t>& fresh,
                        uint32_t a, bool defer_draft = false);
    // complete_round without the model's commit and the delivery: the rows to commit; `keep` the tokens to deliver
    uint32_t plan_round(Task& t, uint32_t pos, const std::vector<int32_t>& drafts, const std::vector<int32_t>& fresh,
                        uint32_t a, bool defer_draft, uint32_t width, std::vector<int32_t>& keep);
    // the next drafts of several generators (next_.anchor / pos / sampled set): greedy chains or sampled tables,
    // batched (draft_batch_full) when there are several; next_ valid after
    static void predraft_many(std::span<Generator* const> gens);
    // sampled rounds in parts (sampled_round; step_batch verifies several slots' blocks at once in between):
    // the draft chain sampled from the drafter's transition table, then the rejection sampling against the target
    // rows (`width` rows: the anchor's and width - 1 drafts')
    void sample_chain(const DSpine::SampledDraft& d, const SamplingParams& p, std::vector<int32_t>& drafts,
                      std::vector<std::vector<double>>& q_rows);
    void resolve_sampled(const Qwen35::VerifyTop& v, const SamplingParams& p, const DSpine::SampledDraft& d,
                         const std::vector<int32_t>& drafts, const std::vector<std::vector<double>>& q_rows, uint32_t width,
                         std::vector<int32_t>& fresh, uint32_t& accepted);
    void deliver_keep(Task& t, const std::vector<int32_t>& keep);
    void finish(Task& t);
    class PrefixCache* cache_ = nullptr;
    bool defer_prefix_save_ = false;
    std::vector<uint32_t> snap_points_;
    int32_t sample(const std::vector<float>& logits, const SamplingParams& p);
    // the verification's rows into the drafter's context, unmarked (queued while the verification runs)
    void pre_append(uint32_t pos);
    // The next round's draft, made before the commit of this round is queued: the drafter does not depend on the
    // target's commit, so the commit then runs while the host prepares the next verification.
    struct NextDraft {
        bool valid = false, sampled = false;
        bool want = false;  // step_batch: a draft at (anchor, pos) to make with the others' (draft_batch)
        uint32_t pos = 0;
        int32_t anchor = -1;
        std::vector<int32_t> drafts;
        DSpine::SampledDraft table;
    };
    NextDraft next_;
    std::vector<int32_t> take_draft(int32_t anchor, uint32_t pos);
    DSpine::SampledDraft take_table(int32_t anchor, uint32_t pos);
    // one speculative round with sampling (dspine_mlx serial_sample_chain + sampled_verify); returns accepted drafts
    // and the bonus token in `fresh`, the number of accepted drafts in `accepted`
    void sampled_round(int32_t anchor, uint32_t pos, const SamplingParams& p, std::vector<int32_t>& drafts,
                       std::vector<int32_t>& fresh, uint32_t& accepted);
    // Tree speculative decoding round (Sequoia best-first tree from DSpine::draft_table, greedy or lossless sampled)
    void tree_round(Task& t, uint32_t pos, uint32_t tree_nodes);
    void complete_tree_round(Task& t, uint32_t pos, const std::vector<int32_t>& win,
                             const std::vector<int32_t>& fresh);
    double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
    std::vector<float> prefill_from(const std::vector<int32_t>& prompt, GenerationResult& r,
                                    std::span<const Qwen35::ImageSpan> images);
    bool history_images_ = false;  // the cached prefix holds image rows

    Qwen35& model_;
    DSpine* drafter_;
    std::vector<int32_t> eos_;
    uint32_t max_ctx_;
    std::vector<int32_t> history_;  // tokens whose state is in the caches (positions 0 .. pos-1)
    std::mt19937_64 rng_;
};

}  // namespace g2e
