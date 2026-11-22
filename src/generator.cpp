#include "g2e/generator.hpp"
#include "g2e/prefix_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace g2e {

namespace {
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
int32_t argmax(const std::vector<float>& v) { return int32_t(std::max_element(v.begin(), v.end()) - v.begin()); }

// propose.filter_probs: joint top-k / top-p + min-p over one row; ties rank the lower lane first.
std::vector<double> filter_probs(const std::vector<double>& p, int top_k, double top_p, double min_p, std::vector<bool>* keep_out) {
    const size_t K = p.size();
    std::vector<bool> keep(K);
    double mx = 0;
    for (double x : p) mx = std::max(mx, x);
    double total = 0;
    for (size_t i = 0; i < K; ++i) {
        size_t rank = 0;
        double before = 0;
        for (size_t j = 0; j < K; ++j)
            if (p[j] > p[i] || (p[j] == p[i] && j < i)) { ++rank; before += p[j]; }
        keep[i] = int(rank) < top_k && before <= top_p && (min_p <= 0 || p[i] >= min_p * mx);
        if (keep[i]) total += p[i];
    }
    std::vector<double> q(K, 0.0);
    for (size_t i = 0; i < K; ++i) if (keep[i] && total > 0) q[i] = p[i] / total;
    if (keep_out) *keep_out = keep;
    return q;
}

// Draw from a discrete distribution (unnormalized weights) with one uniform.
size_t draw(const std::vector<double>& w, double u) {
    double total = 0;
    for (double x : w) total += x;
    double t = u * total;
    for (size_t i = 0; i < w.size(); ++i) {
        t -= w[i];
        if (t < 0 && w[i] > 0) return i;
    }
    for (size_t i = w.size(); i-- > 0;) if (w[i] > 0) return i;
    return 0;
}
}  // namespace

Generator::Generator(Qwen35& model, DSpine* drafter, std::vector<int32_t> eos_ids, uint32_t max_context)
    : model_(model), drafter_(drafter), eos_(std::move(eos_ids)), max_ctx_(max_context), rng_(std::random_device{}()) {}

void Generator::reset() {
    select();  // this generator's slot
    model_.reset();
    if (drafter_) drafter_->reset();
    history_.clear();
    history_images_ = false;
}

// Reuse the cached state when the prompt extends what the caches hold (the usual multi-turn case: the previous
// prompt, its answer, the new turn); otherwise start over. At least one token is always read (its logits are needed).
std::vector<float> Generator::prefill_from(const std::vector<int32_t>& prompt, GenerationResult& r,
                                          std::span<const Qwen35::ImageSpan> images) {
    const bool extends = images.empty() && !history_images_ && !history_.empty() && history_.size() < prompt.size() &&
                         std::equal(history_.begin(), history_.end(), prompt.begin()) && model_.position() == history_.size();
    if (!extends) {
        reset();
        // the longest cached prefix (host memory / disk) instead of prefilling it again
        if (cache_ && images.empty()) {
            const uint32_t got = cache_->restore(prompt);
            if (got) history_.assign(prompt.begin(), prompt.begin() + got);
        }
    }
    const size_t h = history_.size();
    r.reused_tokens = uint32_t(h);
    const std::span<const int32_t> tail(prompt.data() + h, prompt.size() - h);
    std::vector<float> logits;
    std::vector<uint32_t> points;
    if (cache_ && images.empty())
        for (uint32_t p : snap_points_)
            if (p > h && p < prompt.size() && p >= PrefixCache::kMinTokens) points.push_back(p);
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    auto append = [&](uint32_t at, uint32_t n) {
        const uint32_t end = at + n;
        if (drafter_) {
            uint32_t keep = at;
            const uint32_t W = drafter_->window();
            if (W) {
                uint32_t next_stop = uint32_t(prompt.size());
                if (cache_ && images.empty()) {
                    const uint32_t snap = ((end + PrefixCache::kSnapEvery - 1) / PrefixCache::kSnapEvery) * PrefixCache::kSnapEvery;
                    if (snap < next_stop) next_stop = snap;
                }
                for (uint32_t p : points)
                    if (p >= end && p < next_stop) next_stop = p;
                const uint32_t win_lo = next_stop > W ? next_stop - W : 0u;
                keep = std::max(at, std::min(end, win_lo));
            }
            if (keep < end)
                drafter_->append_context(model_.features(), model_.feature_stride(), end - keep, keep, UINT32_MAX, keep - at);
        }
        // a snapshot every kSnapEvery tokens of a long prompt: a shared document resumes there for other questions
        if (cache_ && images.empty() && end % PrefixCache::kSnapEvery == 0 && end < prompt.size())
            cache_->snapshot(prompt, end);
    };
    if (!images.empty()) {
        logits = model_.prefill_images(tail, images, append);  // (h == 0: images reset above)
    } else {
        // prefill up to each snapshot point (a message boundary), snapshot, go on
        size_t at = h;
        for (uint32_t p : points) {
            model_.prefill(std::span<const int32_t>(prompt.data() + at, p - at), append);
            cache_->snapshot(prompt, p);
            at = p;
        }
        logits = model_.prefill(std::span<const int32_t>(prompt.data() + at, prompt.size() - at), append);
    }
    snap_points_.clear();
    history_ = prompt;
    history_images_ = !images.empty();
    return logits;
}

int32_t Generator::sample(const std::vector<float>& logits, const SamplingParams& p) {
    if (p.greedy()) return argmax(logits);
    const size_t V = logits.size();
    std::vector<int32_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    size_t n = V;
    if (p.top_k > 0 && size_t(p.top_k) < V) {
        n = size_t(p.top_k);
        std::nth_element(idx.begin(), idx.begin() + long(n), idx.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    }
    std::sort(idx.begin(), idx.begin() + long(n), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    const double inv_t = 1.0 / double(p.temperature);
    const double mx = logits[idx[0]];
    std::vector<double> pr(n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += pr[i] = std::exp((double(logits[idx[i]]) - mx) * inv_t);
    for (auto& x : pr) x /= sum;
    size_t keep = n;
    if (p.top_p < 1.0f) {  // smallest prefix whose mass reaches top_p (at least one token)
        double cum = 0;
        for (size_t i = 0; i < n; ++i) {
            cum += pr[i];
            if (cum >= double(p.top_p)) { keep = i + 1; break; }
        }
    }
    if (p.min_p > 0.0f) {
        const double floor = double(p.min_p) * pr[0];
        size_t k = 1;
        while (k < keep && pr[k] >= floor) ++k;
        keep = k;
    }
    double total = 0;
    for (size_t i = 0; i < keep; ++i) total += pr[i];
    std::uniform_real_distribution<double> u(0.0, total);
    double x = u(rng_);
    for (size_t i = 0; i < keep; ++i) {
        x -= pr[i];
        if (x <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int32_t> Generator::take_draft(int32_t anchor, uint32_t pos) {
    if (next_.valid && !next_.sampled && next_.pos == pos && next_.anchor == anchor) {
        next_.valid = false;
        return std::move(next_.drafts);
    }
    next_.valid = false;
    return drafter_->draft(anchor, pos);
}

DSpine::SampledDraft Generator::take_table(int32_t anchor, uint32_t pos) {
    if (next_.valid && next_.sampled && next_.pos == pos && next_.anchor == anchor) {
        next_.valid = false;
        return std::move(next_.table);
    }
    next_.valid = false;
    return drafter_->draft_table(anchor, pos);
}

void Generator::pre_append(uint32_t pos) {
    drafter_->append_context(model_.features(), model_.feature_stride(), DSpine::kBlock, pos, 0);
}

void Generator::sample_chain(const DSpine::SampledDraft& d, const SamplingParams& p, std::vector<int32_t>& drafts,
                             std::vector<std::vector<double>>& q_rows) {
    const uint32_t W = DSpine::kBlock, N = W - 1, K = W;
    const double inv_t = 1.0 / std::max(double(p.temperature), 1e-5);
    // sample each row from the transition logits of the previous row's choice
    const int chain_k = p.top_k <= 0 ? int(K) : std::min(p.top_k, int(K));
    q_rows.assign(N, {});
    drafts.assign(N, 0);
    size_t sender = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const float* row = &d.table[(size_t(i) * K + sender) * K];
        double mx = -1e300;
        for (uint32_t k = 0; k < K; ++k) mx = std::max(mx, double(row[k]) * inv_t);
        std::vector<double> pr(K);
        double s = 0;
        for (uint32_t k = 0; k < K; ++k) s += pr[k] = std::exp(double(row[k]) * inv_t - mx);
        for (auto& x : pr) x /= s;
        std::vector<bool> keep;
        q_rows[i] = filter_probs(pr, chain_k, p.top_p, p.min_p, &keep);
        const size_t choice = draw(q_rows[i], uniform());
        drafts[i] = d.cand[size_t(i) * K + choice];
        sender = choice;
    }
}

void Generator::resolve_sampled(const Qwen35::VerifyTop& v, const SamplingParams& p, const DSpine::SampledDraft& d,
                                const std::vector<int32_t>& drafts, const std::vector<std::vector<double>>& q_rows,
                                uint32_t width, std::vector<int32_t>& fresh, uint32_t& accepted) {
    const uint32_t K = DSpine::kBlock, N = width - 1;
    // ---- target distributions (top-64 candidates with exact full-vocabulary probabilities), filtered
    const uint32_t C = Qwen35::kTopK;
    const int target_k = p.top_k <= 0 ? int(C) : std::min(p.top_k, int(C));
    std::vector<std::vector<std::pair<int32_t, double>>> prow(width);  // filtered target probability per candidate id
    for (uint32_t t = 0; t < width; ++t) {
        std::vector<size_t> order(C);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const float va = v.values[t * C + a], vb = v.values[t * C + b];
            return va > vb || (va == vb && v.ids[t * C + a] < v.ids[t * C + b]);
        });
        const double mx = v.max[t], sum = v.sum[t];
        double before = 0, kept = 0;
        const double pmax = std::exp(double(v.values[t * C + order[0]]) - mx) / sum;
        for (size_t r = 0; r < C; ++r) {
            const size_t j = order[r];
            const double pf = std::exp(double(v.values[t * C + j]) - mx) / sum;
            const bool keep = int(r) < target_k && before <= double(p.top_p) && (p.min_p <= 0 || pf >= double(p.min_p) * pmax);
            if (keep) { prow[t].push_back({v.ids[t * C + j], pf}); kept += pf; }
            before += pf;
        }
        if (prow[t].empty()) {
            prow[t].push_back({v.ids[t * C + order[0]], 1.0});
        } else {
            for (auto& [id, pr] : prow[t]) pr /= kept;
        }
    }
    auto p_of = [&](uint32_t t, int32_t id) {
        for (const auto& [i, pr] : prow[t]) if (i == id) return pr;
        return 0.0;
    };
    // ---- chain rejection sampling (propose.sampled_verify)
    uint32_t a = 0;
    while (a < N) {
        const double pd = p_of(a, drafts[a]);
        double qd = 0;
        for (uint32_t k = 0; k < K; ++k) if (d.cand[size_t(a) * K + k] == drafts[a]) qd += q_rows[a][k];
        if (uniform() < std::min(1.0, pd / std::max(qd, 1e-30))) ++a;
        else break;
    }
    int32_t bonus;
    if (a == N) {
        std::vector<double> w;
        for (const auto& e : prow[N]) w.push_back(e.second);
        bonus = prow[N][draw(w, uniform())].first;
    } else {
        std::vector<double> w;
        for (const auto& [id, pr] : prow[a]) {
            double q = 0;
            for (uint32_t k = 0; k < K; ++k) if (d.cand[size_t(a) * K + k] == id) q += q_rows[a][k];
            w.push_back(std::max(pr - q, 0.0));
        }
        double total = 0;
        for (double x : w) total += x;
        if (total <= 0) { w.clear(); for (const auto& e : prow[a]) w.push_back(e.second); }
        bonus = prow[a][draw(w, uniform())].first;
    }
    fresh.assign(drafts.begin(), drafts.begin() + a);
    fresh.push_back(bonus);
    accepted = a;
}

void Generator::sampled_round(int32_t anchor, uint32_t pos, const SamplingParams& p, std::vector<int32_t>& drafts,
                              std::vector<int32_t>& fresh, uint32_t& accepted) {
    const uint32_t W = DSpine::kBlock;
    const double inv_t = 1.0 / std::max(double(p.temperature), 1e-5);
    const auto d = take_table(anchor, pos);
    std::vector<std::vector<double>> q_rows;
    sample_chain(d, p, drafts, q_rows);
    std::vector<int32_t> block{anchor};
    block.insert(block.end(), drafts.begin(), drafts.end());
    const auto v = model_.verify_topk(block, float(inv_t), [&] { pre_append(pos); });
    resolve_sampled(v, p, d, drafts, q_rows, W, fresh, accepted);
}

GenerationResult Generator::generate(const std::vector<int32_t>& prompt, const SamplingParams& params,
                                     const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                     std::span<const Qwen35::ImageSpan> images) {
    start(prompt, params, on_tokens, images);
    while (!step()) {}
    return take_result();
}

void Generator::select() {
    model_.select_slot(slot_);
    if (drafter_) drafter_->select_slot(slot_);
}

void Generator::start(const std::vector<int32_t>& prompt, const SamplingParams& params,
                      std::function<bool(std::span<const int32_t>)> on_tokens, std::span<const Qwen35::ImageSpan> images) {
    if (task_) throw std::logic_error("generator: a request is running");
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    round_ema_ = 4.0;
    auto t = std::make_unique<Task>();
    t->prompt = prompt;
    t->params = params;
    t->on_tokens = std::move(on_tokens);
    t->images.assign(images.begin(), images.end());
    t->r.prompt_tokens = uint32_t(prompt.size());
    task_ = std::move(t);
}

void Generator::release() {
    if (task_) throw std::logic_error("generator: release while a request is running");
    select();
    model_.set_capacity(0);  // also resets the slot
    if (drafter_) drafter_->reset();
    history_.clear();
    history_images_ = false;
    next_.valid = false;
}

bool Generator::ensure(uint32_t need, uint32_t want) {
    if (need <= cap()) return true;
    if (!reserve_ || need > max_ctx_) return false;
    for (uint32_t c : {model_.round_capacity(std::max(need, want)), model_.round_capacity(need)}) {
        if (c < need) continue;
        const uint64_t have = model_.kv_bytes_for(cap()), extra = model_.kv_bytes_for(c) - have;
        const bool ok = reserve_(slot_, extra);
        select();  // making room may have switched slots
        if (!ok) continue;
        model_.set_capacity(c);
        return true;
    }
    return false;
}

void Generator::abort() {
    task_.reset();
    next_.valid = false;
    select();
    model_.clear_pending();
    history_.clear();  // the next request resets the slot
    history_images_ = false;
}

GenerationResult Generator::take_result() {
    if (!task_ || !task_->done) throw std::logic_error("generator: no finished request");
    auto r = std::move(task_->r);
    task_.reset();
    return r;
}

bool Generator::step(uint32_t prefill_tokens) {
    if (!task_) throw std::logic_error("generator: step without a request");
    Task& t = *task_;
    if (t.done) return true;
    select();
    if (t.phase == Task::Begin) {
        if (t.prompt.size() + 1 > max_ctx_) throw std::invalid_argument("prompt longer than the context");
        if (t.params.seed >= 0) rng_.seed(uint64_t(t.params.seed));
        t.max_new = std::min<uint32_t>(t.params.max_tokens, max_ctx_ - uint32_t(t.prompt.size()));
        next_.valid = false;
        t.t0 = std::chrono::steady_clock::now();
        begin(t);
        t.phase = Task::Prefill;
        if (!t.images.empty()) {  // image prompts in one go (explicit multimodal RoPE rows over the whole tail)
            GenerationResult& r = t.r;
            auto logits = prefill_from(t.prompt, r, t.images);
            first_token(t, logits);
            return t.done;
        }
    }
    if (t.phase == Task::Prefill) {
        prefill_segment(t, prefill_tokens);
        return t.done;
    }
    decode_round(t);
    return t.done;
}

void Generator::begin(Task& t) {
    const auto& prompt = t.prompt;
    const bool extends = t.images.empty() && !history_images_ && !history_.empty() && history_.size() < prompt.size() &&
                         std::equal(history_.begin(), history_.end(), prompt.begin()) && model_.position() == history_.size();
    // KV capacity: the prompt and a first round now, room for some answer while at it (it grows later as needed)
    const uint32_t W = DSpine::kBlock, P = uint32_t(prompt.size());
    const uint32_t need = std::min(max_ctx_, P + W + 1);
    const uint32_t want = std::min(max_ctx_, P + std::min<uint32_t>(t.max_new, 2048) + W + 1);
    if (!extends) {
        reset();
        if (reserve_ && cap() > 2 * model_.round_capacity(want)) model_.set_capacity(model_.round_capacity(want));
    }
    if (!ensure(need, want)) throw std::runtime_error("not enough device memory for the KV cache of this request");
    if (!t.images.empty()) return;  // prefill_from handles image prompts
    if (!extends) {
        if (cache_) {  // the longest cached prefix (host memory / disk) instead of prefilling it again
            const uint32_t got = cache_->restore(prompt);
            if (got) history_.assign(prompt.begin(), prompt.begin() + got);
        }
    }
    t.at = history_.size();
    t.r.reused_tokens = uint32_t(t.at);
    if (cache_ && prompt.size() > t.at) {
        const size_t new_chunks = (prompt.size() - t.at) / PrefixCache::kChunk + 1;
        if (new_chunks > 1) cache_->prepare_chunks(new_chunks);
    }
    t.points.clear();
    if (cache_)
        for (uint32_t p : snap_points_)
            // not near the start of the incremental turn or near the end (the assistant header): splitting a short
            // multi-turn tail (< 256 tokens from t.at) doubles the number of forward passes and adds a synchronous
            // snapshot copy in the middle of TTFT, while the full turn is saved at the end of the request anyway
            if (p >= t.at + 256 && p + 64 <= prompt.size() && p >= PrefixCache::kMinTokens) t.points.push_back(p);
    std::sort(t.points.begin(), t.points.end());
    t.points.erase(std::unique(t.points.begin(), t.points.end()), t.points.end());
    snap_points_.clear();
    history_images_ = false;
}

void Generator::prefill_segment(Task& t, uint32_t budget) {
    const auto& prompt = t.prompt;
    auto append = [&](uint32_t at, uint32_t n) {
        const uint32_t end = at + n;  // a snapshot every kSnapEvery tokens of a long prompt
        if (drafter_) {
            uint32_t keep = at;
            const uint32_t W = drafter_->window();
            if (W) {
                uint32_t next_stop = uint32_t(prompt.size());
                if (cache_ && t.images.empty()) {
                    const uint32_t snap = ((end + PrefixCache::kSnapEvery - 1) / PrefixCache::kSnapEvery) * PrefixCache::kSnapEvery;
                    if (snap < next_stop) next_stop = snap;
                }
                for (uint32_t p : t.points)
                    if (p >= end && p < next_stop) next_stop = p;
                const uint32_t win_lo = next_stop > W ? next_stop - W : 0u;
                keep = std::max(at, std::min(end, win_lo));
            }
            if (keep < end)
                drafter_->append_context(model_.features(), model_.feature_stride(), end - keep, keep, UINT32_MAX, keep - at);
        }
        if (cache_ && end % PrefixCache::kSnapEvery == 0 && end < prompt.size()) cache_->snapshot(prompt, end);
    };
    size_t end = prompt.size();
    if (!t.points.empty()) end = std::min<size_t>(end, t.points.front());
    if (budget && end - t.at > budget) end = t.at + budget;
    const auto logits = model_.prefill(std::span<const int32_t>(prompt.data() + t.at, end - t.at), append);
    t.at = end;
    history_.assign(prompt.begin(), prompt.begin() + long(end));
    if (!t.points.empty() && t.points.front() == end) {
        static const bool plog = std::getenv("G2E_PREFILL_LOG") != nullptr;
        const auto s0 = std::chrono::steady_clock::now();
        cache_->snapshot(prompt, uint32_t(end));
        if (plog) {
            model_.stream().synchronize();
            std::fprintf(stderr, "[prefill] snapshot at %zu: %.1f ms\n", end, ms_since(s0));
        }
        t.points.erase(t.points.begin());
    }
    if (end == prompt.size()) first_token(t, logits);
}

void Generator::first_token(Task& t, const std::vector<float>& logits) {
    GenerationResult& r = t.r;
    r.prompt_ms = ms_since(t.t0);
    t.t1 = std::chrono::steady_clock::now();
    history_ = t.prompt;
    history_images_ = !t.images.empty();
    t.phase = Task::Decode;
    t.anchor = sample(logits, t.params);  // generated, not yet in the caches
    r.tokens.push_back(t.anchor);
    if (!t.on_tokens(std::span<const int32_t>(&t.anchor, 1))) {
        r.finish = "abort";
        finish(t);
        return;
    }
    const auto& params = t.params;
    const bool extra = std::find(params.stop_token_ids.begin(), params.stop_token_ids.end(), t.anchor) != params.stop_token_ids.end();
    const bool eos = !params.ignore_eos && std::find(eos_.begin(), eos_.end(), t.anchor) != eos_.end();
    if (extra) { r.finish = "stop"; r.stop_token = t.anchor; }
    else if (eos) r.finish = "stop";
    else if (r.tokens.size() >= t.max_new) r.finish = "length";
    if (extra || eos || r.tokens.size() >= t.max_new) finish(t);
}

void Generator::save_prefix() {
    if (!cache_ || history_images_) return;
    select();
    if (model_.position() != history_.size()) return;
    const auto c0 = std::chrono::steady_clock::now();
    const double a0 = cache_->stats().alloc_ms;
    cache_->save(history_);
    const double ms = ms_since(c0), am = cache_->stats().alloc_ms - a0;
    if (ms > 20)
        std::fprintf(stderr, "[g2e] slot %u: prefix cache save of %zu tokens %.0f ms (pinned allocations %.0f ms)\n",
                     slot_, history_.size(), ms, am);
}

void Generator::finish(Task& t) {
    t.r.decode_ms = t.phase == Task::Decode ? ms_since(t.t1) : 0.0;
    t.done = true;
    // the conversation so far (prompt + answer whose state is in the caches) for a later request of the same history
    if (!defer_prefix_save_) save_prefix();
}

bool Generator::stop_token(const Task& t, int32_t x) const {
    const auto& p = t.params;
    return (!p.ignore_eos && std::find(eos_.begin(), eos_.end(), x) != eos_.end()) ||
           std::find(p.stop_token_ids.begin(), p.stop_token_ids.end(), x) != p.stop_token_ids.end();
}

bool Generator::room_for_round(Task& t) {
    const uint32_t W = DSpine::kBlock, p = model_.position();
    if (p + W > cap() && cap() < max_ctx_ &&
        !ensure(std::min(max_ctx_, p + W + 1), uint32_t(std::min<uint64_t>(max_ctx_, uint64_t(p) + std::max(p / 2, 16384u))))) {
        std::fprintf(stderr, "[g2e] slot %u: no device memory to grow the KV cache past %u rows\n", slot_, cap());
        t.r.finish = "length";
        finish(t);
        return false;
    }
    return true;
}

void Generator::deliver_keep(Task& t, const std::vector<int32_t>& keep) {
    GenerationResult& r = t.r;
    r.tokens.insert(r.tokens.end(), keep.begin(), keep.end());
    if (!t.on_tokens(std::span<const int32_t>(keep))) {
        r.finish = "abort";
        finish(t);
        return;
    }
    const auto& extra = t.params.stop_token_ids;
    for (auto x : keep) {
        if (std::find(extra.begin(), extra.end(), x) != extra.end()) {
            r.finish = "stop";
            r.stop_token = x;
            finish(t);
            return;
        }
        if (stop_token(t, x)) {  // an EOS
            r.finish = "stop";
            finish(t);
            return;
        }
    }
    if (r.tokens.size() >= t.max_new) {
        r.finish = "length";
        finish(t);
        return;
    }
    t.anchor = keep.back();
}

void Generator::complete_round(Task& t, uint32_t pos, const std::vector<int32_t>& drafts, const std::vector<int32_t>& fresh,
                               uint32_t a, bool defer_draft) {
    std::vector<int32_t> keep;
    // the commit is queued before the next round's draft (it does not depend on it): the device runs both without
    // waiting for the host in between (G2E_ACCEPT_LATE=1: draft first, as before)
    static const bool late = std::getenv("G2E_ACCEPT_LATE") != nullptr;
    const uint32_t committed = plan_round(t, pos, drafts, fresh, a, defer_draft || !late, DSpine::kBlock, keep);
    model_.accept(committed);
    if (next_.want && !defer_draft) {
        if (next_.sampled) next_.table = drafter_->draft_table(next_.anchor, next_.pos);
        else next_.drafts = drafter_->draft(next_.anchor, next_.pos);
        next_.valid = true;
        next_.want = false;
    }
    deliver_keep(t, keep);
}

uint32_t Generator::plan_round(Task& t, uint32_t pos, const std::vector<int32_t>& drafts, const std::vector<int32_t>& fresh,
                               uint32_t a, bool defer_draft, uint32_t width, std::vector<int32_t>& keep) {
    GenerationResult& r = t.r;
    const uint32_t W = DSpine::kBlock;
    // stop at the first stop token or at max_tokens, and commit only up to there: the last kept token stays
    // pending (not in the caches), exactly as when the round ends normally
    size_t c = fresh.size() - 1;
    for (size_t k = 0; k < fresh.size(); ++k)
        if (stop_token(t, fresh[k])) { c = k; break; }
    c = std::min<size_t>(c, t.max_new - r.tokens.size() - 1);
    const uint32_t committed = uint32_t(std::min<size_t>(a, c));
    drafter_->mark(pos, committed + 1);  // its rows were appended while the verification ran
    {   // draft the next round now when it will run speculatively (see NextDraft)
        const int32_t next_anchor = fresh[c];
        const uint32_t next_pos = pos + committed + 1;
        const bool stops = c + 1 < fresh.size() || stop_token(t, fresh[c]);
        if (!stops && r.tokens.size() + c + 1 < t.max_new && next_pos + W <= max_ctx_) {
            next_.valid = true;
            next_.sampled = !t.params.greedy();
            next_.pos = next_pos;
            next_.anchor = next_anchor;
            if (defer_draft) {  // made by the caller with the other slots' (step_batch)
                next_.valid = false;
                next_.want = true;
            } else if (next_.sampled) {
                next_.table = drafter_->draft_table(next_anchor, next_pos);
            } else {
                next_.drafts = drafter_->draft(next_anchor, next_pos);
            }
        }
    }
    history_.push_back(t.anchor);
    history_.insert(history_.end(), drafts.begin(), drafts.begin() + committed);
    ++r.rounds;
    r.drafts_offered += width - 1;
    r.drafts_accepted += committed;
    round_ema_ = 0.8 * round_ema_ + 0.2 * double(a + 1);
    keep.assign(fresh.begin(), fresh.begin() + long(c) + 1);
    return committed;
}

namespace {

// Deterministic Sequoia-style best-first prefix tree from DSpine's [15, 16, 16] transition table.
// Zero target RNG is touched, so verifying and drawing directly from target distributions along the tree
// is 100% mathematically exact and lossless for all (temperature, top_p, top_k, min_p).
Qwen35::TreeSpec build_best_first_tree(int32_t anchor, const DSpine::SampledDraft& d, uint32_t max_nodes,
                                       float draft_temp, uint32_t& valid_nodes) {
    const uint32_t W = DSpine::kBlock, K = W;
    const double inv_t = 1.0 / std::max(double(draft_temp), 1e-5);
    auto row_logp = [&](uint32_t depth, uint32_t sender) {
        const float* row = &d.table[(size_t(depth) * K + sender) * K];
        double mx = -1e300;
        for (uint32_t k = 0; k < K; ++k) mx = std::max(mx, double(row[k]) * inv_t);
        double s = 0;
        std::array<double, 16> lp{};
        for (uint32_t k = 0; k < K; ++k) { lp[k] = double(row[k]) * inv_t - mx; s += std::exp(lp[k]); }
        const double lse = std::log(s);
        for (uint32_t k = 0; k < K; ++k) lp[k] -= lse;
        return lp;
    };

    Qwen35::TreeSpec ts;
    ts.tokens.reserve(max_nodes);
    ts.parent.reserve(max_nodes);
    ts.depth.reserve(max_nodes);
    ts.tokens.push_back(anchor);
    ts.parent.push_back(-1);
    ts.depth.push_back(0);

    struct Cand {
        double logp;
        int32_t tok;
        int32_t parent;
        uint32_t depth;
        uint32_t sender;
        bool operator<(const Cand& o) const { return logp < o.logp; }
    };
    std::priority_queue<Cand> pq;
    auto push_children = [&](int32_t p_idx, uint32_t p_depth, uint32_t p_sender, double p_logp) {
        if (p_depth >= W - 1) return;
        const auto lp = row_logp(p_depth, p_sender);
        for (uint32_t k = 0; k < K; ++k)
            pq.push({p_logp + lp[k], d.cand[size_t(p_depth) * K + k], p_idx, p_depth + 1, k});
    };
    push_children(0, 0, 0, 0.0);

    while (ts.tokens.size() < max_nodes && !pq.empty()) {
        const Cand c = pq.top();
        pq.pop();
        bool dup = false;
        for (size_t i = 1; i < ts.tokens.size(); ++i)
            if (ts.parent[i] == c.parent && ts.tokens[i] == c.tok) { dup = true; break; }
        if (dup) continue;
        const int32_t idx = int32_t(ts.tokens.size());
        ts.tokens.push_back(c.tok);
        ts.parent.push_back(c.parent);
        ts.depth.push_back(c.depth);
        push_children(idx, c.depth, c.sender, c.logp);
    }
    valid_nodes = uint32_t(ts.tokens.size());
    while (ts.tokens.size() < max_nodes) {
        ts.tokens.push_back(anchor);
        ts.parent.push_back(0);
        ts.depth.push_back(1);
    }
    return ts;
}

}  // namespace

void Generator::complete_tree_round(Task& t, uint32_t pos, const std::vector<int32_t>& win,
                                    const std::vector<int32_t>& fresh) {
    GenerationResult& r = t.r;
    const uint32_t W = DSpine::kBlock;
    const uint32_t a = uint32_t(win.size() - 1);
    size_t c = fresh.size() - 1;
    for (size_t k = 0; k < fresh.size(); ++k)
        if (stop_token(t, fresh[k])) { c = k; break; }
    c = std::min<size_t>(c, t.max_new - r.tokens.size() - 1);
    const uint32_t committed = uint32_t(std::min<size_t>(a, c));

    // Commit truncated winning path [win[0..committed]] into target KV/GDN + gather drafter features
    model_.accept_tree(std::span<const int32_t>(win.data(), committed + 1));
    drafter_->append_context(model_.tree_features(), model_.tree_feature_stride(), committed + 1, pos, UINT32_MAX, 0);

    // Pre-draft next round's transition table immediately while device stream runs append_context
    const int32_t next_anchor = fresh[c];
    const uint32_t next_pos = pos + committed + 1;
    const bool stops = c + 1 < fresh.size() || stop_token(t, fresh[c]);
    if (!stops && r.tokens.size() + c + 1 < t.max_new && next_pos + W <= max_ctx_) {
        next_.valid = true;
        next_.sampled = true;
        next_.want = false;
        next_.pos = next_pos;
        next_.anchor = next_anchor;
        next_.table = drafter_->draft_table(next_anchor, next_pos);
    }

    history_.push_back(t.anchor);
    history_.insert(history_.end(), fresh.begin(), fresh.begin() + long(committed));
    ++r.rounds;
    r.drafts_offered += W - 1;
    r.drafts_accepted += committed;
    round_ema_ = 0.8 * round_ema_ + 0.2 * double(a + 1);
    std::vector<int32_t> keep(fresh.begin(), fresh.begin() + long(c) + 1);
    deliver_keep(t, keep);
}

void Generator::tree_round(Task& t, uint32_t pos, uint32_t tree_nodes) {
    const SamplingParams& p = t.params;
    const uint32_t W = DSpine::kBlock;
    static const float base_temp = [] {
        const char* e = std::getenv("G2E_TREE_TEMP");
        return e ? float(std::atof(e)) : 1.5f;
    }();
    const float draft_temp = p.greedy() ? base_temp : std::max(base_temp, p.temperature);
    const auto d = take_table(t.anchor, pos);
    uint32_t valid_nodes = 0;
    const auto ts = build_best_first_tree(t.anchor, d, tree_nodes, draft_temp, valid_nodes);

    std::vector<int32_t> win = {0};
    std::vector<int32_t> fresh;
    win.reserve(W);
    fresh.reserve(W);

    if (p.greedy()) {
        const auto tgt = model_.verify_tree(ts);
        int32_t cur = 0;
        while (win.size() < W) {
            const int32_t want = tgt[size_t(cur)];
            int32_t child = -1;
            for (uint32_t i = 1; i < valid_nodes; ++i) {
                if (ts.parent[i] == cur && ts.tokens[i] == want) { child = int32_t(i); break; }
            }
            fresh.push_back(want);
            if (child < 0) break;
            win.push_back(child);
            cur = child;
        }
        if (fresh.size() < win.size()) fresh.push_back(tgt[size_t(cur)]);
    } else {
        const double inv_t = 1.0 / std::max(double(p.temperature), 1e-5);
        const auto v = model_.verify_tree_topk(ts, float(inv_t));
        const uint32_t C = Qwen35::kTopK;
        const int target_k = p.top_k <= 0 ? int(C) : std::min(p.top_k, int(C));
        auto sample_node = [&](int32_t node) -> int32_t {
            const size_t t_idx = size_t(node);
            std::array<size_t, Qwen35::kTopK> order{};
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                const float va = v.values[t_idx * C + a], vb = v.values[t_idx * C + b];
                return va > vb || (va == vb && v.ids[t_idx * C + a] < v.ids[t_idx * C + b]);
            });
            const double mx = v.max[t_idx], sum = v.sum[t_idx];
            double before = 0, kept = 0;
            const double pmax = std::exp(double(v.values[t_idx * C + order[0]]) - mx) / sum;
            std::vector<int32_t> cand_ids;
            std::vector<double> cand_w;
            cand_ids.reserve(C);
            cand_w.reserve(C);
            for (size_t r = 0; r < C; ++r) {
                const size_t j = order[r];
                const double pf = std::exp(double(v.values[t_idx * C + j]) - mx) / sum;
                const bool keep = int(r) < target_k && before <= double(p.top_p) &&
                                  (p.min_p <= 0 || pf >= double(p.min_p) * pmax);
                if (keep) { cand_ids.push_back(v.ids[t_idx * C + j]); cand_w.push_back(pf); kept += pf; }
                before += pf;
            }
            if (cand_ids.empty()) return v.ids[t_idx * C + order[0]];
            for (double& w : cand_w) w /= kept;
            return cand_ids[draw(cand_w, uniform())];
        };

        int32_t cur = 0;
        while (true) {
            const int32_t y = sample_node(cur);
            fresh.push_back(y);
            if (win.size() >= W) break;
            int32_t child = -1;
            for (uint32_t i = 1; i < valid_nodes; ++i) {
                if (ts.parent[i] == cur && ts.tokens[i] == y) { child = int32_t(i); break; }
            }
            if (child < 0) break;
            win.push_back(child);
            cur = child;
        }
    }
    complete_tree_round(t, pos, win, fresh);
}

void Generator::decode_round(Task& t) {
    GenerationResult& r = t.r;
    const SamplingParams& params = t.params;
    const uint32_t W = DSpine::kBlock;
    if (!room_for_round(t)) return;
    const uint32_t pos = model_.position();
    if (drafter_ && pos + W <= cap()) {
        static const uint32_t tree_nodes = [] {
            const char* e = std::getenv("G2E_TREE_NODES");
            return e ? uint32_t(std::atoi(e)) : 24u;
        }();
        static const uint32_t tree_max_pos = [] {
            const char* e = std::getenv("G2E_TREE_MAX_POS");
            return e ? uint32_t(std::atoi(e)) : 0u;
        }();
        if (tree_nodes >= 2 && model_.tree_supported() && (tree_max_pos == 0 || pos < tree_max_pos)) {
            tree_round(t, pos, tree_nodes);
            return;
        }
        std::vector<int32_t> drafts, fresh;
        uint32_t a = 0;
        if (params.greedy()) {
            drafts = take_draft(t.anchor, pos);
            std::vector<int32_t> block{t.anchor};
            block.insert(block.end(), drafts.begin(), drafts.end());
            const auto target = model_.verify(block, [&] { pre_append(pos); });
            while (a < W - 1 && drafts[a] == target[a]) ++a;
            fresh.assign(drafts.begin(), drafts.begin() + a);
            fresh.push_back(target[a]);
        } else {
            sampled_round(t.anchor, pos, params, drafts, fresh, a);
        }
        complete_round(t, pos, drafts, fresh, a);
        return;
    }
    if (pos + 1 > cap()) {
        r.finish = "length";
        finish(t);
        return;
    }
    const auto logits = model_.forward(std::span<const int32_t>(&t.anchor, 1));
    history_.push_back(t.anchor);
    deliver_keep(t, {sample(logits, params)});
}

void Generator::predraft_many(std::span<Generator* const> gens) {
    for (size_t k = 0; k < gens.size(); k += DSpine::kMaxBatch) {  // groups of up to kMaxBatch
        const size_t n = std::min<size_t>(DSpine::kMaxBatch, gens.size() - k);
        if (n == 1) {
            Generator& g = *gens[k];
            g.select();
            if (g.next_.sampled) g.next_.table = g.drafter_->draft_table(g.next_.anchor, g.next_.pos);
            else g.next_.drafts = g.drafter_->draft(g.next_.anchor, g.next_.pos);
            g.next_.valid = true;
            g.next_.want = false;
            continue;
        }
        std::vector<uint32_t> slots, pos;
        std::vector<int32_t> anchors;
        bool tables = false;
        for (size_t m = 0; m < n; ++m) {
            const Generator& g = *gens[k + m];
            slots.push_back(g.slot_);
            anchors.push_back(g.next_.anchor);
            pos.push_back(g.next_.pos);
            tables = tables || g.next_.sampled;
        }
        auto part = gens[k]->drafter_->draft_batch_full(slots, anchors, pos, tables);
        for (size_t m = 0; m < n; ++m) {
            Generator& g = *gens[k + m];
            if (g.next_.sampled) g.next_.table = std::move(part[m].table);
            else g.next_.drafts = std::move(part[m].drafts);
            g.next_.valid = true;
            g.next_.want = false;
        }
    }
}

bool Generator::batchable() const {
    return task_ && !task_->done && task_->phase == Task::Decode && drafter_ && !history_images_;
}

std::vector<bool> Generator::step_batch(std::span<Generator* const> gens) {
    const uint32_t W = DSpine::kBlock;
    std::vector<bool> done(gens.size(), false);
    std::vector<size_t> in;  // the generators of the batch
    for (size_t k = 0; k < gens.size(); ++k) {
        Generator& g = *gens[k];
        if (!g.batchable()) {
            done[k] = g.step();
            continue;
        }
        g.select();
        if (!g.room_for_round(*g.task_)) {
            done[k] = true;
            continue;
        }
        if (g.model_.position() + W > g.cap()) {  // at the end of the context: a plain step
            done[k] = g.step();
            continue;
        }
        in.push_back(k);
    }
    if (in.size() < 2) {
        for (size_t k : in) done[k] = gens[k]->step();
        return done;
    }
    Qwen35& model = gens[in[0]]->model_;
    // batches of at least G2E_BATCH_W8 slots (default 5; 0: never) whose recent rounds averaged fewer than
    // G2E_BATCH_W8_EMA tokens (default 5.5) verify 8 rows per slot instead of 16: the batched forward is compute
    // bound there and the rows past the usual acceptance mostly wasted. (At width 8 the average is capped at 8, so a
    // well-accepting batch climbs back over the threshold.) Off by default: on TokenSpeed (6-9 tokens per round at
    // width 16) the capped average stayed under the threshold, the batches stuck at width 8 and lost more tokens per
    // round than the ~19% cheaper verification saved.
    static const uint32_t w8_from = [] {
        const char* e = std::getenv("G2E_BATCH_W8");
        return e ? uint32_t(std::atoi(e)) : 0u;  // off by default: see below
    }();
    static const double w8_ema = [] {
        const char* e = std::getenv("G2E_BATCH_W8_EMA");
        return e ? std::atof(e) : 5.5;
    }();
    double ema = 0;
    for (size_t k : in) ema += gens[k]->round_ema_;
    ema /= double(in.size());
    const uint32_t Wv = w8_from && in.size() >= w8_from && ema < w8_ema ? 8 : W;
    static const bool prof = std::getenv("G2E_BATCH_PROF") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = in.size();
    std::vector<uint32_t> slots(n), pos(n);
    std::vector<std::vector<int32_t>> blocks(n), drafts(n);
    std::vector<bool> sampled(n);
    std::vector<DSpine::SampledDraft> tables(n);
    std::vector<std::vector<std::vector<double>>> q_rows(n);
    bool any_sampled = false;
    {   // this round's drafts: the ones made last round, the missing ones in one batched draft
        std::vector<Generator*> miss;
        for (size_t b = 0; b < n; ++b) {
            Generator& g = *gens[in[b]];
            g.select();
            const uint32_t p = g.model_.position();
            const int32_t anchor = g.task_->anchor;
            sampled[b] = !g.task_->params.greedy();
            any_sampled = any_sampled || sampled[b];
            slots[b] = g.slot_;
            pos[b] = p;
            if (!(g.next_.valid && g.next_.sampled == sampled[b] && g.next_.pos == p && g.next_.anchor == anchor)) {
                g.next_.valid = g.next_.want = false;
                g.next_.sampled = sampled[b];
                g.next_.pos = p;
                g.next_.anchor = anchor;
                miss.push_back(&g);
            }
        }
        if (!miss.empty()) predraft_many(miss);
        for (size_t b = 0; b < n; ++b) {
            Generator& g = *gens[in[b]];
            if (sampled[b]) {
                tables[b] = std::move(g.next_.table);
                g.sample_chain(tables[b], g.task_->params, drafts[b], q_rows[b]);
            } else {
                drafts[b] = std::move(g.next_.drafts);
            }
            g.next_.valid = g.next_.want = false;
            blocks[b].push_back(g.task_->anchor);
            blocks[b].insert(blocks[b].end(), drafts[b].begin(), drafts[b].begin() + (Wv - 1));
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    auto append = [&] {  // the drafters' context appends overlap the wait (one batched launch for all slots)
        gens[in[0]]->drafter_->append_context_batch(slots, model.features(), model.feature_stride(), Wv, pos, 0);
    };
    std::vector<std::vector<int32_t>> targets;
    std::vector<Qwen35::VerifyTop> tops;
    if (any_sampled) {  // the sampling head for all (a greedy slot at temperature 1 takes each row's top value)
        std::vector<float> inv_t(n);
        for (size_t b = 0; b < n; ++b)
            inv_t[b] = sampled[b] ? float(1.0 / std::max(double(gens[in[b]]->task_->params.temperature), 1e-5)) : 1.f;
        tops = model.verify_batch_topk(slots, blocks, inv_t, append);
    } else {
        targets = model.verify_batch(slots, blocks, append);
    }
    const auto t2 = std::chrono::steady_clock::now();
    uint32_t accepted = 0;
    std::vector<uint32_t> committed(n);
    std::vector<std::vector<int32_t>> keep(n);
    for (size_t b = 0; b < n; ++b) {
        Generator& g = *gens[in[b]];
        g.select();
        uint32_t a = 0;
        std::vector<int32_t> fresh;
        if (sampled[b]) {
            g.resolve_sampled(tops[b], g.task_->params, tables[b], drafts[b], q_rows[b], Wv, fresh, a);
        } else {
            std::vector<int32_t> tg = any_sampled ? std::vector<int32_t>(Wv) : targets[b];
            if (any_sampled) {
                const uint32_t C = Qwen35::kTopK;
                for (uint32_t t = 0; t < Wv; ++t) {
                    uint32_t best = 0;
                    for (uint32_t c = 1; c < C; ++c) {
                        const float vc = tops[b].values[t * C + c], vb = tops[b].values[t * C + best];
                        if (vc > vb || (vc == vb && tops[b].ids[t * C + c] < tops[b].ids[t * C + best])) best = c;
                    }
                    tg[t] = tops[b].ids[t * C + best];
                }
            }
            while (a < Wv - 1 && drafts[b][a] == tg[a]) ++a;
            fresh.assign(drafts[b].begin(), drafts[b].begin() + a);
            fresh.push_back(tg[a]);
        }
        committed[b] = g.plan_round(*g.task_, pos[b], drafts[b], fresh, a, true, Wv, keep[b]);
        accepted += a + 1;
    }
    model.accept_batch(slots, committed);  // every slot's commit in one launch
    for (size_t b = 0; b < n; ++b) {
        Generator& g = *gens[in[b]];
        g.select();
        g.deliver_keep(*g.task_, keep[b]);
        done[in[b]] = g.task_->done;
    }
    {   // the next round's drafts, batched
        std::vector<Generator*> want;
        for (size_t k : in)
            if (gens[k]->next_.want) want.push_back(gens[k]);
        if (!want.empty()) predraft_many(want);
    }
    if (prof) {  // every 50 batches: one batch's parts, and the averages incl. the time between batches
        static int n = 0;
        static double in_ms = 0, gap_ms = 0, toks = 0;
        static std::chrono::steady_clock::time_point prev_end{};
        if (prev_end.time_since_epoch().count()) gap_ms += std::chrono::duration<double, std::milli>(t0 - prev_end).count();
        in_ms += ms_since(t0);
        toks += accepted;
        if (++n % 50 == 0) {
            std::fprintf(stderr, "[g2e] batch %zu x %u (ema %.1f): drafts %.1f ms, verify %.1f ms (device %.1f), commit + next drafts %.1f ms; %u tokens"
                                 " | last 50: %.1f ms in batches + %.1f ms between, %.1f tokens per batch -> %.0f tok/s\n",
                         in.size(), Wv, ema, ms_since(t0) - ms_since(t1), ms_since(t1) - ms_since(t2), model.last_device_ms(), ms_since(t2),
                         accepted, in_ms / 50, gap_ms / 50, toks / 50, toks / (in_ms + gap_ms) * 1e3);
            in_ms = gap_ms = toks = 0;
        }
        prev_end = std::chrono::steady_clock::now();
    }
    return done;
}

}  // namespace g2e
