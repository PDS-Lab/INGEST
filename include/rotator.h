#pragma once

#include "ffht.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <memory>

#include <myutils/random.h>

constexpr size_t fht_kac_default_niter = 2;

namespace detail {
inline void kacs_walk_legacy(size_t n, float const *input, float *output) {
  for (size_t i = 0; i < n / 2; ++i) {
    float x = input[i];
    float y = input[i + (n / 2)];
    output[i] = x + y;
    output[i + (n / 2)] = x - y;
  }
}

#if defined(__AVX512F__)
inline void kacs_walk_avx512(size_t n, float const *input, float *output) {
  n /= 2;
  size_t i = n;
  __m512 a, b;
  do {
    if (i < 16) [[unlikely]] {
      __mmask16 mask = (1U << i) - 1;
      a = _mm512_maskz_loadu_ps(mask, input);
      b = _mm512_maskz_loadu_ps(mask, input + n);
      __m512 x = _mm512_add_ps(a, b);
      __m512 y = _mm512_sub_ps(a, b);
      _mm512_mask_storeu_ps(output, mask, x);
      _mm512_mask_storeu_ps(output + n, mask, y);
      i = 0;
    } else [[likely]] {
      a = _mm512_loadu_ps(input);
      b = _mm512_loadu_ps(input + n);
      __m512 x = _mm512_add_ps(a, b);
      __m512 y = _mm512_sub_ps(a, b);
      _mm512_storeu_ps(output, x);
      _mm512_storeu_ps(output + n, y);
      input += 16, output += 16, i -= 16;
    }
  } while (i);
}
#endif
} 

inline void kacs_walk(size_t n, float const *input, float *output) {
#if defined(__AVX512F__)
  detail::kacs_walk_avx512(n, input, output);
#else
  detail::kacs_walk_legacy(n, input, output);
#endif
}

namespace detail {
inline void flip_scale_legacy(size_t n, const float *input, float *output,
                              const void *flip, float scale) {
  const uint8_t *flip_u8 = (const uint8_t *)flip;
  for (size_t i = 0; i < n; ++i)
    output[i] = input[i] * scale * ((flip_u8[i / 8] & (1 << (i % 8))) ? -1 : 1);
}

#if defined(__AVX512F__)
inline void flip_scale_avx512(size_t n, const float *input, float *output,
                              const void *flip, float scale) {
  auto vscale = _mm512_set1_ps(scale);
  auto vsign = _mm512_set1_ps(-.0f);
  const uint16_t *flip_u16 = (const uint16_t *)flip;
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      auto x = _mm512_maskz_loadu_ps(mask, input);
      x = _mm512_mask_xor_ps(x, *flip_u16, x, vsign);
      x = _mm512_mul_ps(x, vscale);
      _mm512_mask_storeu_ps(output, mask, x);
      n = 0;
    } else [[likely]] {
      auto x = _mm512_loadu_ps(input);
      x = _mm512_mask_xor_ps(x, *flip_u16, x, vsign);
      x = _mm512_mul_ps(x, vscale);
      _mm512_storeu_ps(output, x);
      input += 16, output += 16, n -= 16, flip_u16++;
    }
  } while (n);
}
#endif
} 

inline void flip_scale(size_t n, const float *input, float *output,
                       const void *flip, float scale) {
#if defined(__AVX512F__)
  detail::flip_scale_avx512(n, input, output, flip, scale);
#else
  detail::flip_scale_legacy(n, input, output, flip, scale);
#endif
}

namespace detail {
inline void vec_scale_legacy(size_t n, const float *input, float *output,
                             float scale) {
  for (size_t i = 0; i < n; ++i)
    output[i] = input[i] * scale;
}

#if defined(__AVX512F__)
inline void vec_scale_avx512(size_t n, const float *input, float *output,
                             float scale) {
  auto vscale = _mm512_set1_ps(scale);
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      auto x = _mm512_maskz_loadu_ps(mask, input);
      x = _mm512_mul_ps(x, vscale);
      _mm512_mask_storeu_ps(output, mask, x);
      n = 0;
    } else [[likely]] {
      auto x = _mm512_loadu_ps(input);
      x = _mm512_mul_ps(x, vscale);
      _mm512_storeu_ps(output, x);
      input += 16, output += 16, n -= 16;
    }
  } while (n);
}
#endif
} 

inline void vec_scale(size_t n, const float *input, float *output,
                      float scale) {
#if defined(__AVX512F__)
  detail::vec_scale_avx512(n, input, output, scale);
#else
  detail::vec_scale_legacy(n, input, output, scale);
#endif
}

namespace detail {
inline void vec_flip_legacy(size_t n, const float *input, float *output,
                            const void *flip) {
  const uint8_t *flip_u8 = (const uint8_t *)flip;
  for (size_t i = 0; i < n; ++i)
    output[i] = input[i] * ((flip_u8[i / 8] & (1 << (i % 8))) ? -1 : 1);
}

#if defined(__AVX512F__)
inline void vec_flip_avx512(size_t n, const float *input, float *output,
                            const void *flip) {
  auto vsign = _mm512_set1_ps(-.0f);
  const uint16_t *flip_u16 = (const uint16_t *)flip;
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      auto x = _mm512_maskz_loadu_ps(mask, input);
      x = _mm512_mask_xor_ps(x, *flip_u16, x, vsign);
      _mm512_mask_storeu_ps(output, mask, x);
      n = 0;
    } else [[likely]] {
      auto x = _mm512_loadu_ps(input);
      x = _mm512_mask_xor_ps(x, *flip_u16, x, vsign);
      _mm512_storeu_ps(output, x);
      input += 16, output += 16, n -= 16, flip_u16++;
    }
  } while (n);
}
#endif
} 

inline void vec_flip(size_t n, const float *input, float *output,
                     const void *flip) {
#if defined(__AVX512F__)
  detail::vec_flip_avx512(n, input, output, flip);
#else
  detail::vec_flip_legacy(n, input, output, flip);
#endif
}

constexpr static float kac_rescale_tbl[] = {
    0.7071067811865475,   0.5,       0.35355339059327373,  0.25,
    0.17677669529663687,  0.125,     0.08838834764831843,  0.0625,
    0.044194173824159216, 0.03125,   0.022097086912079608, 0.015625,
    0.011048543456039804, 0.0078125, 0.005524271728019902, 0.00390625,
};

constexpr size_t
fht_kac_rotate_flip_size(size_t dim, size_t niter = fht_kac_default_niter) {
  return niter * (1 << (std::bit_width(dim) - 1)) / 8;
}

constexpr float fht_kac_rotate_fac(size_t dim) {
  return 1.f / std::sqrt((float)(1 << (std::bit_width(dim) - 1)));
}

template <size_t niter = fht_kac_default_niter, bool post_scale = true>
  requires(niter > 0 && niter <= 16)
inline void fht_kac_rotate(size_t dim, float fac, const void *flip,
                           const float *input, float *output) {
  const size_t log2dim = std::bit_width(dim) - 1;
  const size_t trunc_dim = 1 << log2dim;
  const size_t start = dim - trunc_dim;
  const ffht::fht_float_func_t fht_func = ffht::fht_float_tbl[log2dim - 1];
  const uint8_t *flip_u8 = (const uint8_t *)flip;
  for (size_t i = 0; i < niter; ++i) {
    float *ptr = output + (i % 2 == 0 ? 0 : start);
    if (i == 0) {
      flip_scale(trunc_dim, input, output, flip_u8 + i * trunc_dim / 8, fac);
      std::copy(input + trunc_dim, input + dim, output + trunc_dim);
      std::fill(output + dim, output + dim, 0);
    } else
      flip_scale(trunc_dim, ptr, ptr, flip_u8 + i * trunc_dim / 8, fac);
    fht_func(ptr);
    if (dim != trunc_dim)
      kacs_walk(dim, output, output);
  }
  if (post_scale && dim != trunc_dim)
    vec_scale(dim, output, output, kac_rescale_tbl[niter - 1]);
}

template <size_t niter = fht_kac_default_niter, bool post_scale = true>
struct FhtKacRotator {
  uint32_t dim_;
  uint32_t trunc_dim_;
  float fac_;
  float rescale_;
  std::unique_ptr<uint8_t[]> flip_;
  ffht::fht_float_func_t fht_func_;
  FhtKacRotator() = default;
  FhtKacRotator(auto &&...args) { init(std::forward<decltype(args)>(args)...); }
  void init_params(size_t dim) {
    dim_ = dim;
    const size_t log2dim = std::bit_width(dim_) - 1;
    trunc_dim_ = 1 << log2dim;
    fac_ = 1.f / std::sqrt((float)trunc_dim_);
    rescale_ = 1.f / sqrtf(1u << niter);
    fht_func_ = ffht::fht_float_tbl[log2dim - 1];
    flip_ = std::make_unique<uint8_t[]>(niter * trunc_dim_ / 8);
  }
  void init_flip(RandomEngine auto &rng) {
    rng.fill(flip_.get(), niter * trunc_dim_ / 8);
  }
  void init_flip(void *flip) {
    memcpy(flip_.get(), flip, niter * trunc_dim_ / 8);
  }
  void init(size_t dim, RandomEngine auto &rng) {
    init_params(dim);
    init_flip(rng);
  }
  void init(size_t dim) { init(dim, get_default_random_engine()); }
  void init(size_t dim, void *flip) {
    init_params(dim);
    init_flip(flip);
  }
  constexpr float rescale() const { return rescale_; }
  void rotate(const float *input, float *output) const {
    const size_t start = dim_ - trunc_dim_;
    for (size_t i = 0; i < niter; ++i) {
      float *ptr = output + (i % 2 == 0 ? 0 : start);
      if (i == 0) {
        flip_scale(trunc_dim_, input, output, &flip_[i * trunc_dim_ / 8], fac_);
        std::copy(input + trunc_dim_, input + dim_, output + trunc_dim_);
        std::fill(output + dim_, output + dim_, 0);
      } else
        flip_scale(trunc_dim_, ptr, ptr, &flip_[i * trunc_dim_ / 8], fac_);
      fht_func_(ptr);
      if (dim_ != trunc_dim_)
        kacs_walk(dim_, output, output);
    }
    if (post_scale && dim_ != trunc_dim_)
      vec_scale(dim_, output, output, rescale());
  }
};

std::unique_ptr<float[]> init_proj_rand(const size_t dim, const size_t proj_dim,
                                        uint64_t seed = 0);
std::unique_ptr<float[]> init_proj_pca(const size_t dim, const size_t proj_dim,
                                       size_t nsample, const float *data,
                                       uint64_t seed = 0);
float *apply_proj(const size_t dim, const size_t proj_dim, const float *input,
                  float *output, float *P);
float *apply_proj(const size_t dim, const size_t proj_dim, const int8_t *input,
                  float *output, float *P);
float *apply_proj(const size_t dim, const size_t proj_dim, const uint8_t *input,
                  float *output, float *P);
