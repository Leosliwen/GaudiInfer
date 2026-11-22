// Image decoding (stb_image) and the Qwen2-VL image processor (smart resize, bicubic, normalise, patchify).
#include "g2e/vision.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "stb/stb_image.h"

namespace g2e {
namespace {

// Python's round(): to nearest, ties to even
double py_round(double x) { return std::nearbyint(x); }

// transformers smart_resize: multiples of `factor`, pixel count within [min_pixels, max_pixels], aspect kept
void smart_resize(int height, int width, int factor, uint64_t min_pixels, uint64_t max_pixels, int& out_h, int& out_w) {
    if (double(std::max(height, width)) / double(std::min(height, width)) > 200)
        throw std::invalid_argument("image aspect ratio must be smaller than 200");
    double h_bar = py_round(double(height) / factor) * factor, w_bar = py_round(double(width) / factor) * factor;
    if (h_bar * w_bar > double(max_pixels)) {
        const double beta = std::sqrt(double(height) * width / double(max_pixels));
        h_bar = std::max<double>(factor, std::floor(height / beta / factor) * factor);
        w_bar = std::max<double>(factor, std::floor(width / beta / factor) * factor);
    } else if (h_bar * w_bar < double(min_pixels)) {
        const double beta = std::sqrt(double(min_pixels) / (double(height) * width));
        h_bar = std::ceil(height * beta / factor) * factor;
        w_bar = std::ceil(width * beta / factor) * factor;
    }
    out_h = int(h_bar);
    out_w = int(w_bar);
}

double bicubic(double x) {  // a = -0.5 (PIL)
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

// PIL ImagingResample coefficients for one axis (antialiased: the filter widens by the scale when shrinking),
// as 22-bit fixed point for 8-bit images
struct Coeffs {
    int ksize = 0;
    std::vector<int> start, size;
    std::vector<int32_t> k;  // [out, ksize]
};
Coeffs coefficients(int in, int out) {
    constexpr int kBits = 32 - 8 - 2;
    const double scale = double(in) / out, filterscale = std::max(scale, 1.0), support = 2.0 * filterscale;
    Coeffs c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.start.resize(out);
    c.size.resize(out);
    c.k.assign(size_t(out) * c.ksize, 0);
    std::vector<double> w(c.ksize);
    for (int i = 0; i < out; ++i) {
        const double center = (i + 0.5) * scale;
        int xmin = int(center - support + 0.5), xmax = int(center + support + 0.5);
        xmin = std::max(xmin, 0);
        xmax = std::min(xmax, in) - xmin;
        double total = 0;
        for (int x = 0; x < xmax; ++x) total += w[x] = bicubic((x + xmin - center + 0.5) / filterscale);
        for (int x = 0; x < xmax; ++x) {
            const double v = total != 0 ? w[x] / total : 0;
            c.k[size_t(i) * c.ksize + x] = int32_t(v < 0 ? std::trunc(v * (1 << kBits) - 0.5) : std::trunc(v * (1 << kBits) + 0.5));
        }
        c.start[i] = xmin;
        c.size[i] = xmax;
    }
    return c;
}

uint8_t clip8(int64_t v) {
    constexpr int kBits = 32 - 8 - 2;
    v = (v + (int64_t(1) << (kBits - 1))) >> kBits;
    return uint8_t(std::clamp<int64_t>(v, 0, 255));
}

// RGB8 [h][w][3] -> [oh][ow][3]: horizontal pass, then vertical (PIL's order), 8-bit intermediate
std::vector<uint8_t> resize_rgb(const uint8_t* src, int w, int h, int ow, int oh) {
    std::vector<uint8_t> mid(size_t(h) * ow * 3), dst(size_t(oh) * ow * 3);
    const auto ch = coefficients(w, ow), cv = coefficients(h, oh);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t acc = 0;
                for (int k = 0; k < ch.size[x]; ++k)
                    acc += int64_t(src[(size_t(y) * w + ch.start[x] + k) * 3 + c]) * ch.k[size_t(x) * ch.ksize + k];
                mid[(size_t(y) * ow + x) * 3 + c] = clip8(acc);
            }
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t acc = 0;
                for (int k = 0; k < cv.size[y]; ++k)
                    acc += int64_t(mid[(size_t(cv.start[y] + k) * ow + x) * 3 + c]) * cv.k[size_t(y) * cv.ksize + k];
                dst[(size_t(y) * ow + x) * 3 + c] = clip8(acc);
            }
    return dst;
}

}  // namespace

namespace {

bool is_webp(const uint8_t* data, size_t size) {
    return data && size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

std::vector<uint8_t> decode_webp(const uint8_t* data, size_t size, int& width, int& height) {
    using DecodeRGBFn = uint8_t* (*)(const uint8_t*, size_t, int*, int*);
    using FreeFn = void (*)(void*);
    static DecodeRGBFn decode_rgb = nullptr;
    static FreeFn webp_free = nullptr;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        void* handle = dlopen("libwebp.so.7", RTLD_LAZY);
        if (!handle) handle = dlopen("libwebp.so", RTLD_LAZY);
        if (handle) {
            decode_rgb = reinterpret_cast<DecodeRGBFn>(dlsym(handle, "WebPDecodeRGB"));
            webp_free = reinterpret_cast<FreeFn>(dlsym(handle, "WebPFree"));
        }
    });
    if (!decode_rgb || !webp_free) {
        throw std::invalid_argument("cannot decode the image: WebP image requires libwebp (libwebp.so.7)");
    }
    uint8_t* px = decode_rgb(data, size, &width, &height);
    if (!px || width <= 0 || height <= 0) {
        if (px) webp_free(px);
        throw std::invalid_argument("cannot decode the image: invalid or corrupt WebP");
    }
    std::vector<uint8_t> rgb(px, px + size_t(width) * size_t(height) * 3);
    webp_free(px);
    return rgb;
}

}  // namespace

std::vector<uint8_t> decode_image(const uint8_t* data, size_t size, int& width, int& height) {
    if (is_webp(data, size)) return decode_webp(data, size, width, height);
    int channels = 0;
    stbi_uc* px = stbi_load_from_memory(data, int(size), &width, &height, &channels, 3);
    if (!px) throw std::invalid_argument(std::string("cannot decode the image: ") + stbi_failure_reason());
    std::vector<uint8_t> rgb(px, px + size_t(width) * height * 3);
    stbi_image_free(px);
    return rgb;
}

VisionImage preprocess_image(const uint8_t* rgb, int width, int height, uint64_t min_pixels, uint64_t max_pixels,
                             uint32_t patch, uint32_t merge) {
    int rh = 0, rw = 0;
    smart_resize(height, width, int(patch * merge), min_pixels, max_pixels, rh, rw);
    const auto img = (rh == height && rw == width) ? std::vector<uint8_t>(rgb, rgb + size_t(width) * height * 3)
                                                   : resize_rgb(rgb, width, height, rw, rh);
    VisionImage out;
    out.grid_h = uint32_t(rh) / patch;
    out.grid_w = uint32_t(rw) / patch;
    const uint64_t P = patch, K = 3 * 2 * P * P, N = uint64_t(out.grid_h) * out.grid_w;
    out.patches.resize(N * K);
    const uint32_t bw = out.grid_w / merge;
    for (uint64_t p = 0; p < N; ++p) {
        // processor order: merge blocks row-major, the block's patches row-major
        const uint64_t blk = p / (merge * merge), in = p % (merge * merge);
        const uint64_t row = (blk / bw) * merge + in / merge, col = (blk % bw) * merge + in % merge;
        float* dst = &out.patches[p * K];
        for (uint64_t c = 0; c < 3; ++c)
            for (uint64_t t = 0; t < 2; ++t)  // the temporal copy of a still image
                for (uint64_t y = 0; y < P; ++y)
                    for (uint64_t x = 0; x < P; ++x) {
                        const uint8_t v = img[((row * P + y) * uint64_t(rw) + col * P + x) * 3 + c];
                        // rescale 1/255, normalise (mean 0.5, std 0.5)
                        dst[((c * 2 + t) * P + y) * P + x] = (float(v) * float(1.0 / 255.0) - 0.5f) / 0.5f;
                    }
    }
    return out;
}

}  // namespace g2e
