#pragma once

#include <myutils/print.h> // for formatter

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include <immintrin.h>

namespace detail {
inline float l2sqr_impl_legacy(const float *x, const float *y, const size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  return sum;
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t l2sqr_impl_legacy(const T *x, const T *y, const size_t n) {
  uint32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    int16_t diff = (int16_t)x[i] - (int16_t)y[i];
    sum += diff * diff;
  }
  return sum;
}

inline float ip_impl_legacy(const float *x, const float *y, const size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += x[i] * y[i];
  return sum;
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t ip_impl_legacy(const T *x, const T *y, const size_t n) {
  uint32_t sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += (int16_t)x[i] * (int16_t)y[i];
  return sum;
}

inline float normsqr_impl_legacy(const float *x, const size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += x[i] * x[i];
  return sum;
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t normsqr_impl_legacy(const T *x, const size_t n) {
  uint32_t sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += (int16_t)(x[i]) * (int16_t)(x[i]);
  return sum;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline float l2sqr_impl_avx512(const float *x, const float *y, size_t n) {
  __m512 sum = _mm512_setzero_ps();
  __m512 a, b;
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      a = _mm512_maskz_loadu_ps(mask, x);
      b = _mm512_maskz_loadu_ps(mask, y);
      n = 0;
    } else [[likely]] {
      a = _mm512_loadu_ps(x);
      b = _mm512_loadu_ps(y);
      x += 16, y += 16, n -= 16;
    }
    __m512 diff = _mm512_sub_ps(a, b);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  } while (n);
  return _mm512_reduce_add_ps(sum);
}

inline float ip_impl_avx512(const float *x, const float *y, size_t n) {
  __m512 sum = _mm512_setzero_ps();
  __m512 a, b;
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      a = _mm512_maskz_loadu_ps(mask, x);
      b = _mm512_maskz_loadu_ps(mask, y);
      n = 0;
    } else [[likely]] {
      a = _mm512_loadu_ps(x);
      b = _mm512_loadu_ps(y);
      x += 16, y += 16, n -= 16;
    }
    sum = _mm512_fmadd_ps(a, b, sum);
  } while (n);
  return _mm512_reduce_add_ps(sum);
}

inline float normsqr_impl_avx512(const float *x, size_t n) {
  __m512 sum = _mm512_setzero_ps();
  __m512 a;
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      a = _mm512_maskz_loadu_ps(mask, x);
      n = 0;
    } else [[likely]] {
      a = _mm512_loadu_ps(x);
      x += 16, n -= 16;
    }
    sum = _mm512_fmadd_ps(a, a, sum);
  } while (n);
  return _mm512_reduce_add_ps(sum);
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t l2sqr_impl_avx512(const T *x, const T *y, size_t n) {
  __m512i sum = _mm512_setzero_si512();
  __m256i a_i8, b_i8;
  do {
    if (n < 32) [[unlikely]] {
      __mmask32 mask = (1U << n) - 1;
      a_i8 = _mm256_maskz_loadu_epi8(mask, x);
      b_i8 = _mm256_maskz_loadu_epi8(mask, y);
      n = 0;
    } else [[likely]] {
      a_i8 = _mm256_loadu_si256((__m256i const *)x);
      b_i8 = _mm256_loadu_si256((__m256i const *)y);
      x += 32, y += 32, n -= 32;
    }
    __m512i diff;
    if constexpr (std::is_same_v<T, uint8_t>)
      diff = _mm512_sub_epi16(_mm512_cvtepu8_epi16(a_i8),
                              _mm512_cvtepu8_epi16(b_i8));
    else
      diff = _mm512_sub_epi16(_mm512_cvtepi8_epi16(a_i8),
                              _mm512_cvtepi8_epi16(b_i8));
#if defined(__AVX512VNNI__)
    sum = _mm512_dpwssd_epi32(sum, diff, diff);
#else
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(diff, diff));
#endif
  } while (n);
  return _mm512_reduce_add_epi32(sum);
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t ip_impl_avx512(const T *x, const T *y, size_t n) {
  __m512i sum = _mm512_setzero_si512();
  __m256i a_i8, b_i8;
  __m512i a_i16, b_i16;
  do {
    if (n < 32) [[unlikely]] {
      __mmask32 mask = (1U << n) - 1;
      a_i8 = _mm256_maskz_loadu_epi8(mask, x);
      b_i8 = _mm256_maskz_loadu_epi8(mask, y);
      n = 0;
    } else [[likely]] {
      a_i8 = _mm256_loadu_si256((__m256i const *)x);
      b_i8 = _mm256_loadu_si256((__m256i const *)y);
      x += 32, y += 32, n -= 32;
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
      a_i16 = _mm512_cvtepu8_epi16(a_i8);
      b_i16 = _mm512_cvtepu8_epi16(b_i8);
    } else {
      a_i16 = _mm512_cvtepi8_epi16(a_i8);
      b_i16 = _mm512_cvtepi8_epi16(b_i8);
    }
#if defined(__AVX512VNNI__)
    sum = _mm512_dpwssd_epi32(sum, a_i16, b_i16);
#else
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(a_i16, b_i16));
#endif
  } while (n);
  return _mm512_reduce_add_epi32(sum);
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t normsqr_impl_avx512(const T *x, size_t n) {
  __m512i sum = _mm512_setzero_si512();
  __m256i a_i8, b_i8;
  do {
    if (n < 32) [[unlikely]] {
      __mmask32 mask = (1U << n) - 1;
      a_i8 = _mm256_maskz_loadu_epi8(mask, x);
      n = 0;
    } else [[likely]] {
      a_i8 = _mm256_loadu_si256((__m256i const *)x);
      x += 32, n -= 32;
    }
    __m512i d;
    if constexpr (std::is_same_v<T, uint8_t>)
      d = _mm512_cvtepu8_epi16(a_i8);
    else
      d = _mm512_cvtepi8_epi16(a_i8);
#if defined(__AVX512VNNI__)
    sum = _mm512_dpwssd_epi32(sum, d, d);
#else
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(d, d));
#endif
  } while (n);
  return _mm512_reduce_add_epi32(sum);
}
#endif

} // namespace detail

#if !__has_builtin(__builtin_assume)
#define __builtin_assume(x)                                                    \
  if (!(x))                                                                    \
    __builtin_unreachable();
#endif

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
           std::is_same_v<T, float>
inline auto l2sqr(const T *x, const T *y, size_t n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  return detail::l2sqr_impl_avx512(x, y, n);
#else
  return detail::l2sqr_impl_legacy(x, y, n);
#endif
}
inline float l2sqr_align(const float *x, const float *y, size_t n) {
  __builtin_assume(n % 16 == 0);
  return l2sqr(x, y, n);
}
template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t l2sqr_align(const T *x, const T *y, size_t n) {
  __builtin_assume(n % 64 == 0);
  return l2sqr(x, y, n);
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
           std::is_same_v<T, float>
inline auto inner_product(const T *x, const T *y, size_t n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  return detail::ip_impl_avx512(x, y, n);
#else
  return detail::ip_impl_legacy(x, y, n);
#endif
}
inline float inner_product_align(const float *x, const float *y, size_t n) {
  __builtin_assume(n % 16 == 0);
  return inner_product(x, y, n);
}
template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t inner_product_align(const T *x, const T *y, size_t n) {
  __builtin_assume(n % 64 == 0);
  return inner_product(x, y, n);
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
           std::is_same_v<T, float>
inline auto normsqr(const T *x, size_t n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  return detail::normsqr_impl_avx512(x, n);
#else
  return detail::normsqr_impl_legacy(x, n);
#endif
}
inline float normsqr_align(const float *x, size_t n) {
  __builtin_assume(n % 16 == 0);
  return normsqr(x, n);
}
template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline uint32_t normsqr_align(const T *x, size_t n) {
  __builtin_assume(n % 64 == 0);
  return normsqr(x, n);
}

namespace detail {
template <typename T>
inline std::tuple<T, T> minmax_impl_legacy(const T *x, const size_t n) {
  T maxv = std::numeric_limits<T>::lowest();
  T minv = std::numeric_limits<T>::max();
  for (size_t i = 0; i < n; ++i) {
    maxv = std::max(maxv, x[i]);
    minv = std::min(minv, x[i]);
  }
  return std::make_tuple(minv, maxv);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline std::tuple<float, float> minmax_impl_avx512(const float *x, size_t n) {
  __m512 v;
  __m512 maxv = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  __m512 minv = _mm512_set1_ps(std::numeric_limits<float>::max());
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      v = _mm512_maskz_loadu_ps(mask, x);
      maxv = _mm512_mask_max_ps(maxv, mask, maxv, v);
      minv = _mm512_mask_min_ps(minv, mask, minv, v);
      n = 0;
    } else [[likely]] {
      v = _mm512_loadu_ps(x);
      maxv = _mm512_max_ps(maxv, v);
      minv = _mm512_min_ps(minv, v);
      x += 16;
      n -= 16;
    }
  } while (n);
  return std::make_tuple(_mm512_reduce_min_ps(minv),
                         _mm512_reduce_max_ps(maxv));
}

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline std::tuple<T, T> minmax_impl_avx512(const T *x, size_t n) {
  return detail::minmax_impl_legacy(x, n); // TODO: impl for u8 i8
}
#endif
} // namespace detail

template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
           std::is_same_v<T, float>
inline std::tuple<T, T> minmax(const T *x, size_t n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  return detail::minmax_impl_avx512(x, n);
#else
  return detail::minmax_impl_legacy(x, n);
#endif
}
inline std::tuple<float, float> minmax_align(const float *x, size_t n) {
  __builtin_assume(n % 16 == 0);
  return minmax(x, n);
}
template <typename T>
  requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>
inline std::tuple<T, T> minmax_align(const T *x, size_t n) {
  __builtin_assume(n % 64 == 0);
  return minmax(x, n);
}

namespace detail {
#if defined(__AVX512F__) && defined(__AVX512BW__)
inline void quantize_impl_avx512(size_t n, float lo, float delta,
                                 const float *from, uint8_t *to) {
  float inv_delta = 1.f / delta;
  auto vinv_delta = _mm512_set1_ps(inv_delta);
  auto vneg_lo_inv_delta = _mm512_set1_ps(-lo * inv_delta);
  do {
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      auto cur = _mm512_maskz_loadu_ps(mask, from);
      cur = _mm512_fmadd_ps(cur, vinv_delta, vneg_lo_inv_delta);
      auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
      _mm_mask_storeu_epi8(to, mask, i8);
      n = 0;
    } else [[likely]] {
      auto cur = _mm512_loadu_ps(from);
      cur = _mm512_fmadd_ps(cur, vinv_delta, vneg_lo_inv_delta);
      auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
      _mm_storeu_epi8(to, i8);
      from += 16, to += 16, n -= 16;
    }
  } while (n);
}
#endif

inline void quantize_impl_legacy(size_t n, float lo, float delta,
                                 const float *from, uint8_t *to) {
  for (size_t i = 0; i < n; ++i)
    to[i] = std::lroundf((from[i] - lo) / delta);
}
} // namespace detail

inline void quantize(size_t n, float lo, float delta, const float *from,
                     uint8_t *to) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  detail::quantize_impl_avx512(n, lo, delta, from, to);
#else
  detail::quantize_impl_legacy(n, lo, delta, from, to);
#endif
}

inline void quantize_align(size_t n, float lo, float delta, const float *from,
                           uint8_t *to) {
  __builtin_assume(n % 16 == 0);
  quantize(n, lo, delta, from, to);
}

namespace detail {
inline uint64_t xor_cksum_legacy(void *ptr, size_t size, uint64_t cksum = 0) {
  auto u64_ptr = (uint64_t *)ptr;
  for (size_t i = 0; i < size / sizeof(uint64_t); ++i)
    cksum ^= u64_ptr[i];
  size_t remaining = size % 8;
  if (remaining > 0) {
    uint64_t last_block = 0;
    uint8_t *remaining_data =
        (uint8_t *)ptr + (size / sizeof(uint64_t)) * sizeof(uint64_t);
    for (size_t i = 0; i < remaining; ++i)
      last_block |= static_cast<uint64_t>(remaining_data[i]) << (i * 8);
    cksum ^= last_block;
  }
  return cksum;
}
#if defined(__AVX512F__) && defined(__AVX512BW__)
inline uint64_t xor_cksum_avx512(void *ptr, size_t size, uint64_t cksum = 0) {
  auto u8_ptr = (uint8_t *)ptr;
  __m512i v_cksum = _mm512_setzero_si512();
  do {
    __m512i cur;
    if (size < 64) [[unlikely]] {
      __mmask64 mask = (1ULL << size) - 1;
      cur = _mm512_maskz_loadu_epi8(mask, u8_ptr);
      size = 0;
    } else [[likely]] {
      cur = _mm512_loadu_epi8(u8_ptr);
      u8_ptr += 64, size -= 64;
    }
    v_cksum = _mm512_xor_epi64(v_cksum, cur);
  } while (size);

  __m256i low = _mm512_castsi512_si256(v_cksum);
  __m256i high = _mm512_extracti64x4_epi64(v_cksum, 1);
  __m256i xor256 = _mm256_xor_si256(low, high);

  __m128i low128 = _mm256_castsi256_si128(xor256);
  __m128i high128 = _mm256_extracti128_si256(xor256, 1);
  __m128i xor128 = _mm_xor_si128(low128, high128);

  uint64_t result = _mm_extract_epi64(xor128, 0) ^ _mm_extract_epi64(xor128, 1);

  return result ^ cksum;
}
#endif
} // namespace detail

struct RecallResult {
  uint32_t at;
  float recall;
};
template <> struct std::formatter<RecallResult> {
  std::formatter<float> sub_fmt;
  std::string prefix = "recall@";
  std::string suffix = ":";
  constexpr auto parse(std::format_parse_context &__pc) {
    const auto __last = __pc.end();
    auto __first = __pc.begin();
    auto __finished = [&] { return __first == __last || *__first == '}'; };

    // Parse prefix
    if (!__finished() && *__first != ':')
      prefix.clear();
    while (!__finished() && *__first != ':') {
      if (*__first == '/') {
        __first++;
        if (__finished())
          throw std::format_error("Invalid format");
      }
      prefix += *__first++;
    }
    if (!__finished() && *__first == ':')
      __first++;

    // Parse suffix
    if (!__finished() && *__first != ':') {
      suffix.clear();
      while (!__finished() && *__first != ':') {
        if (*__first == '/') {
          __first++;
          if (__finished())
            throw std::format_error("Invalid format");
        }
        suffix += *__first++;
      }
      if (!__finished() && *__first == ':')
        __first++;
    }

    __pc.advance_to(__first);
    return sub_fmt.parse(__pc);
  }
  auto format(auto &r, std::format_context &ctx) const {
    std::format_to(ctx.out(), "{}{}{}", prefix, r.at, suffix);
    return sub_fmt.format(r.recall, ctx);
  }
};
inline auto calc_recall(uint32_t nq, uint32_t nt, std::integral auto const *gt,
                        uint32_t K, std::integral auto const *I,
                        uint32_t nrecall_at,
                        std::integral auto const *recall_at) {
  std::vector<uint32_t> cnters(nrecall_at, 0);
  for (uint32_t q = 0; q < nq; ++q) {
    auto gt_q = gt + (size_t)q * nt;
    auto I_q = I + (size_t)q * K;
    for (uint32_t i = 0; i < K; ++i)
      for (uint32_t j = 0; j < K; ++j)
        if (I_q[i] == gt_q[j])
          for (uint32_t r = 0; r < nrecall_at; ++r)
            if (uint32_t at = recall_at[r]; at <= K && j < at && i < at)
              cnters[r]++;
  }
  std::vector<RecallResult> recall_list;
  for (uint32_t i = 0; i < nrecall_at /* && recall_at[i] < K*/; ++i)
    recall_list.emplace_back(recall_at[i],
                             1. * cnters[i] / (nq * recall_at[i]));
  return recall_list;
}

struct L2sqr {
  template <typename T>
    requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
             std::is_same_v<T, float>
  constexpr auto operator()(const T *x, const T *y, size_t n) const {
    return l2sqr(x, y, n);
  }
};

struct InnerProduct {
  template <typename T>
    requires std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
             std::is_same_v<T, float>
  constexpr auto operator()(const T *x, const T *y, size_t n) const {
    return inner_product(x, y, n);
  }
};

enum class DistFunc { L2, IP };

template <typename dist_t, typename idx_t> struct NeighborImpl {
  dist_t dist{std::numeric_limits<dist_t>::max()};
  idx_t id{std::numeric_limits<idx_t>::max()};
  auto operator==(const NeighborImpl &other) const { return id == other.id; }
  auto operator<=>(const NeighborImpl &other) const {
    return dist <=> other.dist;
  }
  operator idx_t() const { return id; }
};

template <typename dist_t, typename idx_t>
struct std::formatter<NeighborImpl<dist_t, idx_t>> : EmptyParser {
  auto format(const NeighborImpl<dist_t, idx_t> &n, auto &ctx) const {
    return std::format_to(ctx.out(), "({}, {})", n.dist, n.id);
  }
};
