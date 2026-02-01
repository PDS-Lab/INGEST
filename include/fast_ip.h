#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include <myutils/utils.h>

constexpr size_t fast_ip_batch_size = 32;

constexpr size_t get_lut_size(size_t dim4, size_t pad_dim = 16) {
  return upper_align(dim4, pad_dim) / 4 * 16;
}

inline void pack_lut(size_t dim4, const auto *__restrict q,
                     auto *__restrict lut, size_t pad_dim = 16) {
  internal_assert(dim4 % 4 == 0);
  size_t i = 0;
  for (; i < dim4; i += 4) {
    
    lut[0] = 0;                          
    lut[1] = q[0];                       
    lut[2] = q[1];                       
    lut[3] = q[1] + q[0];                
    lut[4] = q[2];                       
    lut[5] = q[2] + q[0];                
    lut[6] = q[2] + q[1];                
    lut[7] = q[2] + q[1] + q[0];         
    lut[8] = q[3];                       
    lut[9] = q[3] + q[0];                
    lut[10] = q[3] + q[1];               
    lut[11] = q[3] + q[1] + q[0];        
    lut[12] = q[3] + q[2];               
    lut[13] = q[3] + q[2] + q[0];        
    lut[14] = q[3] + q[2] + q[1];        
    lut[15] = q[3] + q[2] + q[1] + q[0]; 
    q += 4;
    lut += 16;
  }
  for (; i < upper_align(dim4, pad_dim); i += 4) {
    for (size_t j = 0; j < 16; ++j)
      lut[j] = 0;
    lut += 16;
  }
}





inline void pack_raw_code_impl(size_t dim4, const auto *__restrict xb,
                               uint8_t *__restrict code) {
  internal_assert(dim4 % 4 == 0);
  auto get_u4 = [&xb](uint32_t i4) {
    return (xb[i4 + 3] > 0 ? 8 : 0) | (xb[i4 + 2] > 0 ? 4 : 0) |
           (xb[i4 + 1] > 0 ? 2 : 0) | (xb[i4 + 0] > 0 ? 1 : 0);
  };
  for (size_t j = 0; j < dim4; j += 4) {
    for (size_t k = 0; k < 32 / 4; ++k) {
      uint8_t lo1 = get_u4((k)*dim4 + j);
      uint8_t hi1 = get_u4((k + 32 / 2) * dim4 + j);
      *code++ = lo1 | (hi1 << 4);
      uint8_t lo2 = get_u4((k + 32 / 4) * dim4 + j);
      uint8_t hi2 = get_u4((k + 32 / 2 + 32 / 4) * dim4 + j);
      *code++ = lo2 | (hi2 << 4);
    }
  }
}

namespace detail {

template <typename CodeGetter>
  requires std::is_invocable_r_v<const uint8_t *, CodeGetter, size_t>
inline void pack_separate_code_impl_legacy(size_t dim4,
                                           CodeGetter &&code_getter,
                                           uint8_t *__restrict code_packed) {
  internal_assert(dim4 % 4 == 0);
  auto get_u4 = [](const uint8_t *__restrict c, uint32_t j4) {
    return (c[j4 / 8] >> (j4 % 8)) & 0xf;
  };
  for (size_t j = 0; j < dim4; j += 4) {
    for (size_t k = 0; k < 32 / 4; ++k) {
      uint8_t lo1 = get_u4(code_getter(k), j);
      uint8_t hi1 = get_u4(code_getter(k + 16), j);
      *code_packed++ = lo1 | (hi1 << 4);
      uint8_t lo2 = get_u4(code_getter(k + 8), j);
      uint8_t hi2 = get_u4(code_getter(k + 24), j);
      *code_packed++ = lo2 | (hi2 << 4);
    }
  }
}

inline void pack_code_impl_legacy(size_t dim4, const uint8_t *__restrict code,
                                  uint8_t *__restrict code_packed) {
  pack_separate_code_impl_legacy(
      dim4, [&](size_t i) { return code + i * dim4 / 8; }, code_packed);
}

#define interleave16x16_u8(bit, from, to, stride, epi)                         \
  for (size_t i = 0; i < 16 / stride; ++i)                                     \
    for (size_t j = 0; j < stride / 2; ++j) {                                  \
      size_t base = i * stride;                                                \
      to[base + 2 * j] = _mm##bit##_unpacklo_epi##epi(                         \
          from[base + j], from[base + j + stride / 2]);                        \
      to[base + 2 * j + 1] = _mm##bit##_unpackhi_epi##epi(                     \
          from[base + j], from[base + j + stride / 2]);                        \
    }

#ifdef __AVX2__
template <typename CodeGetter>
  requires std::is_invocable_r_v<const uint8_t *, CodeGetter, size_t>
inline void pack_separate_code_impl_avx2(size_t dim8, CodeGetter &&code_getter,
                                         uint8_t *__restrict code_packed) {
  const __m256i lo_mask_256 = _mm256_set1_epi8(0x0f);
  const __m256i hi_mask_256 = _mm256_set1_epi8(0xf0);
  internal_assert(dim8 % 8 == 0);
  __m256i regs[2][16];
  const size_t stride = dim8 / 8;
  __m128i *pcode_128 = (__m128i *)code_packed;
  for (size_t d = 0; d < dim8; d += 64) {
    
    for (size_t i = 0; i < 16; ++i) {
      uint64_t a = *(uint64_t *)(code_getter(i + 00) + d / 8);
      uint64_t b = *(uint64_t *)(code_getter(i + 16) + d / 8);
      __m256i tmp = _mm256_cvtepu8_epi16(_mm_set_epi64x(b, a));
      __m256i lo = _mm256_and_si256(tmp, lo_mask_256);
      __m256i hi = _mm256_slli_epi16(_mm256_and_si256(tmp, hi_mask_256), 4);
      regs[0][i] = _mm256_or_si256(lo, hi);
    }
    
    interleave16x16_u8(256, regs[0], regs[1], 2, 8);
    interleave16x16_u8(256, regs[1], regs[0], 4, 16);
    interleave16x16_u8(256, regs[0], regs[1], 8, 32);
    interleave16x16_u8(256, regs[1], regs[0], 16, 8);
    
    
    const size_t nlane = std::min(dim8 - d, 64ul) / 4;
    for (size_t i = 0; i < nlane; ++i) {
      __m128i lo = _mm256_castsi256_si128(regs[0][i]);
      __m128i hi = _mm256_extracti128_si256(regs[0][i], 1);
      __m128i merged = _mm_or_si128(_mm_slli_epi16(hi, 4), lo);
      _mm_storeu_si128(pcode_128++, merged);
    }
  }
}
inline void pack_code_impl_avx2(size_t dim8, const uint8_t *__restrict code,
                                uint8_t *__restrict code_packed) {
  const size_t stride = dim8 / 8;
  pack_separate_code_impl_avx2(
      dim8, [&](size_t i) { return code + i * stride; }, code_packed);
}
inline void pack_separate_code_impl_avx2(size_t dim8,
                                         const uint8_t **__restrict code,
                                         uint8_t *__restrict code_packed) {
  const size_t stride = dim8 / 8;
  pack_separate_code_impl_avx2(
      dim8, [&](size_t i) { return code[i]; }, code_packed);
}
#endif

#ifdef __SSE2__

inline void pack_code_impl_sse2(size_t dim8, const uint8_t *__restrict code,
                                uint8_t *__restrict code_packed) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i lo_mask_16 = _mm_set1_epi16(0x000f);
  const __m128i hi_mask_16 = _mm_set1_epi16(0x00f0);
  internal_assert(dim8 % 8 == 0);
  __m128i regs_lo[2][16];
  __m128i regs_hi[2][16];
  const size_t stride = dim8 / 8;
  __m128i *pcode_128 = (__m128i *)code_packed;
  for (size_t d = 0; d < dim8; d += 64) {
    for (size_t i = 0; i < 16; ++i) {
      uint64_t a = *(uint64_t *)(code + (i + 00) * stride + d / 8);
      uint64_t b = *(uint64_t *)(code + (i + 16) * stride + d / 8);
      __m128i v = _mm_set_epi64x((long long)b, (long long)a);
      __m128i lo16 = _mm_unpacklo_epi8(v, zero);
      __m128i hi16 = _mm_unpackhi_epi8(v, zero);
      __m128i lo_comb =
          _mm_or_si128(_mm_and_si128(lo16, lo_mask_16),
                       _mm_slli_epi16(_mm_and_si128(lo16, hi_mask_16), 4));
      __m128i hi_comb =
          _mm_or_si128(_mm_and_si128(hi16, lo_mask_16),
                       _mm_slli_epi16(_mm_and_si128(hi16, hi_mask_16), 4));
      regs_lo[0][i] = lo_comb;
      regs_hi[0][i] = hi_comb;
    }
    interleave16x16_u8(, regs_lo[0], regs_lo[1], 2, 8);
    interleave16x16_u8(, regs_lo[1], regs_lo[0], 4, 16);
    interleave16x16_u8(, regs_lo[0], regs_lo[1], 8, 32);
    interleave16x16_u8(, regs_lo[1], regs_lo[0], 16, 8);
    interleave16x16_u8(, regs_hi[0], regs_hi[1], 2, 8);
    interleave16x16_u8(, regs_hi[1], regs_hi[0], 4, 16);
    interleave16x16_u8(, regs_hi[0], regs_hi[1], 8, 32);
    interleave16x16_u8(, regs_hi[1], regs_hi[0], 16, 8);
    const size_t nlane = std::min(dim8 - d, 64ul) / 4;
    for (size_t i = 0; i < nlane; ++i) {
      __m128i merged =
          _mm_or_si128(_mm_slli_epi16(regs_hi[0][i], 4), regs_lo[0][i]);
      _mm_storeu_si128(pcode_128++, merged);
    }
  }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)


inline void pack_code_impl_avx512
    [[deprecated]] (size_t dim64, const uint8_t *__restrict code,
                    uint8_t *__restrict code_packed) {
  const __m512i lo_mask_512 = _mm512_set1_epi8(0x0f);
  const __m512i hi_mask_512 = _mm512_set1_epi8(0xf0);
  internal_assert(dim64 % 8 == 0);
  __m512i regs[2][16];
  const size_t stride = dim64 / 8;
  
  
  for (size_t d = 0; d < dim64; d += 64) {
    for (size_t i = 0; i < 16; ++i) {
      uint64_t a0 = *(uint64_t *)(code + (i + 00) * stride + d / 8);
      uint64_t b0 = *(uint64_t *)(code + (i + 16) * stride + d / 8);
      uint64_t a1 = *(uint64_t *)(code + (i + 32) * stride + d / 8);
      uint64_t b1 = *(uint64_t *)(code + (i + 48) * stride + d / 8);
      __m512i tmp = _mm512_cvtepu8_epi16(_mm256_set_epi64x(b1, a1, b0, a0));
      __m512i lo = _mm512_and_si512(tmp, lo_mask_512);
      __m512i hi = _mm512_slli_epi16(_mm512_and_si512(tmp, hi_mask_512), 4);
      regs[0][i] = _mm512_or_si512(lo, hi);
    }
    interleave16x16_u8(512, regs[0], regs[1], 2, 8);
    interleave16x16_u8(512, regs[1], regs[0], 4, 16);
    interleave16x16_u8(512, regs[0], regs[1], 8, 32);
    interleave16x16_u8(512, regs[1], regs[0], 16, 8);
    for (size_t i = 0; i < 16; ++i) {
      __m128i lo0 = _mm512_extracti32x4_epi32(regs[0][i], 0);
      __m128i hi0 = _mm512_extracti32x4_epi32(regs[0][i], 1);
      __m128i lo1 = _mm512_extracti32x4_epi32(regs[0][i], 2);
      __m128i hi1 = _mm512_extracti32x4_epi32(regs[0][i], 3);
      __m128i merged0 = _mm_or_si128(_mm_slli_epi16(hi0, 4), lo0);
      __m128i merged1 = _mm_or_si128(_mm_slli_epi16(hi1, 4), lo1);
      
      
      _mm_storeu_si128((__m128i *)(code_packed + (d / 4 + i) * 16), merged0);
      _mm_storeu_si128((__m128i *)(code_packed + (d / 4 + i) * 16 + 512),
                       merged1);
    }
  }
}
#endif
} 

template <typename CodeGetter>
  requires std::is_invocable_r_v<const uint8_t *, CodeGetter, size_t>
inline void pack_separate_code_impl(size_t dim8, CodeGetter &&code_getter,
                                    uint8_t *__restrict code_packed) {
#ifdef __AVX2__
  detail::pack_separate_code_impl_avx2(dim8, code_getter, code_packed);


#else
#warning "AVX2/SSE2 not supported, fallback to legacy implementation"
  detail::pack_separate_code_impl_legacy(dim8, code_getter, code_packed);
#endif
}



inline void pack_code_impl(size_t dim, const uint8_t *__restrict code,
                           uint8_t *__restrict code_packed) {
#ifdef __AVX2__
  detail::pack_code_impl_avx2(dim, code, code_packed);
#elif defined(__SSE2__)
  detail::pack_code_impl_sse2(dim, code, code_packed);
#else
#warning "AVX2/SSE2 not supported, fallback to legacy implementation"
  detail::pack_code_impl_legacy(dim, code, code_packed);
#endif
}

namespace detail {
#ifdef __SSSE3__
inline void fast_ip_impl_128(size_t dim4, const void *__restrict code,
                             const void *__restrict lut,
                             uint16_t *__restrict res) {
  const __m128i lo_mask_128 = _mm_set1_epi8(0x0f);
  auto code_128 = (__m128i *)code;
  auto lut_128 = (__m128i *)lut;
  auto accu0 = _mm_setzero_si128();
  auto accu1 = _mm_setzero_si128();
  auto accu2 = _mm_setzero_si128();
  auto accu3 = _mm_setzero_si128();
  for (size_t j = 0; j < dim4; j += 4) {
    auto c = _mm_loadu_si128(code_128++);
    auto tbl = _mm_loadu_si128(lut_128++);
    auto lo = _mm_and_si128(c, lo_mask_128);
    auto hi = _mm_and_si128(_mm_srli_epi16(c, 4), lo_mask_128);
    auto res_lo = _mm_shuffle_epi8(tbl, lo);
    auto res_hi = _mm_shuffle_epi8(tbl, hi);
    accu0 = _mm_add_epi16(accu0, res_lo);
    accu1 = _mm_add_epi16(accu1, _mm_srli_epi16(res_lo, 8));
    accu2 = _mm_add_epi16(accu2, res_hi);
    accu3 = _mm_add_epi16(accu3, _mm_srli_epi16(res_hi, 8));
  }
  accu0 = _mm_sub_epi16(accu0, _mm_slli_epi16(accu1, 8));
  accu2 = _mm_sub_epi16(accu2, _mm_slli_epi16(accu3, 8));
  _mm_storeu_si128((__m128i *)(res + 0), accu0);
  _mm_storeu_si128((__m128i *)(res + 8), accu1);
  _mm_storeu_si128((__m128i *)(res + 16), accu2);
  _mm_storeu_si128((__m128i *)(res + 24), accu3);
}
#endif

#ifdef __AVX2__
inline void fast_ip_impl_256(size_t dim8, const void *__restrict code,
                             const void *__restrict lut,
                             uint16_t *__restrict res) {
  const __m256i lo_mask_256 = _mm256_set1_epi8(0x0f);
  auto code_256 = (__m256i *)code;
  auto lut_256 = (__m256i *)lut;
  auto accu0 = _mm256_setzero_si256();
  auto accu1 = _mm256_setzero_si256();
  auto accu2 = _mm256_setzero_si256();
  auto accu3 = _mm256_setzero_si256();
  for (size_t j = 0; j < dim8; j += 8) {
    auto c = _mm256_loadu_si256(code_256++);
    auto tbl = _mm256_loadu_si256(lut_256++);
    auto lo = _mm256_and_si256(c, lo_mask_256);
    auto hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), lo_mask_256);
    auto res_lo = _mm256_shuffle_epi8(tbl, lo);
    auto res_hi = _mm256_shuffle_epi8(tbl, hi);
    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
  }
  accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
  accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
  auto dis0 = _mm256_add_epi16(_mm256_blend_epi32(accu0, accu1, 0xF0),
                               _mm256_permute2x128_si256(accu0, accu1, 0x21));
  auto dis1 = _mm256_add_epi16(_mm256_blend_epi32(accu2, accu3, 0xF0),
                               _mm256_permute2x128_si256(accu2, accu3, 0x21));
  _mm256_storeu_si256((__m256i *)(res + 0), dis0);
  _mm256_storeu_si256((__m256i *)(res + 16), dis1);
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline void fast_ip_impl_512(size_t dim16, const void *__restrict code,
                             const void *__restrict lut,
                             uint16_t *__restrict res) {
  const __m512i lo_mask_512 = _mm512_set1_epi8(0x0f);
  auto code_512 = (__m512i *)code;
  auto lut_512 = (__m512i *)lut;
  auto accu0 = _mm512_setzero_si512();
  auto accu1 = _mm512_setzero_si512();
  auto accu2 = _mm512_setzero_si512();
  auto accu3 = _mm512_setzero_si512();
  for (size_t j = 0; j < dim16; j += 16) {
    auto c = _mm512_loadu_si512(code_512++);
    auto tbl = _mm512_loadu_si512(lut_512++);
    auto lo = _mm512_and_si512(c, lo_mask_512);
    auto hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask_512);
    auto res_lo = _mm512_shuffle_epi8(tbl, lo);
    auto res_hi = _mm512_shuffle_epi8(tbl, hi);
    accu0 = _mm512_add_epi16(accu0, res_lo);
    accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
    accu2 = _mm512_add_epi16(accu2, res_hi);
    accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
  }
  accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
  accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));
  __m512i ret1 =
      _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu0, accu1),
                       _mm512_shuffle_i64x2(accu0, accu1, 0b01001110));
  __m512i ret2 =
      _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu2, accu3),
                       _mm512_shuffle_i64x2(accu2, accu3, 0b01001110));
  _mm512_storeu_si512(
      (__m512i *)res,
      _mm512_add_epi16(_mm512_shuffle_i64x2(ret1, ret2, 0b10001000),
                       _mm512_shuffle_i64x2(ret1, ret2, 0b11011101)));
}
#endif
} 


inline void fast_ip_impl(size_t dim, const void *__restrict code,
                         const void *__restrict lut, uint16_t *__restrict res) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  detail::fast_ip_impl_512(dim, code, lut, res);
#elif defined(__AVX2__)
  detail::fast_ip_impl_256(dim, code, lut, res);
#elif defined(__SSSE3__)
  detail::fast_ip_impl_128(dim, code, lut, res);
#else
#error "SSSE3, AVX2 or AVX512 not supported"
#endif
}

inline uint16_t fast_ip_one(size_t dim, const void *__restrict code,
                            const void *__restrict lut) {
  auto code_u8 = (const uint8_t *)code;
  auto lut_u8 = (const uint8_t *)lut;
  uint16_t accu_res = 0;
  for (size_t i = 0; i < dim / 8; ++i) {
    auto lo = code_u8[i] & 0xf;
    auto hi = code_u8[i] >> 4;
    auto lut_lo = lut_u8 + i * 32;
    auto lut_hi = lut_u8 + i * 32 + 16;
    accu_res += (uint16_t)lut_lo[lo] + (uint16_t)lut_hi[hi];
  }
  return accu_res;
}
