#pragma once
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <tuple>

#include "fast_ip.h"
#include "index_utils.h"

constexpr float kRabitqEpsilon = 1.9f;

namespace detail {
#if defined(__AVX512F__) && defined(__AVX512BW__)
template <DistFunc metric = DistFunc::L2, bool with_err = false>
  requires(metric == DistFunc::L2 || metric == DistFunc::IP)
inline auto rabitq_impl_avx512(size_t dim, const float *o, const float *c,
                               void *code) {
  const __m512 vhalf = _mm512_set1_ps(0.5f);
  const __m512 vnhalf = _mm512_set1_ps(-0.5f);
  
  auto code_u16 = (uint16_t *)code;
  __m512 vsum_co_co = _mm512_setzero_ps();
  __m512 vsum_co_sco = _mm512_setzero_ps();
  __m512 vsum_c_sco = _mm512_setzero_ps();
  __m512 vsum_co_c = _mm512_setzero_ps();
  size_t n = dim;
  do {
    __m512 vo, vc, vco, vsco;
    __mmask16 sign_vco;
    if (n < 16) [[unlikely]] {
      __mmask16 mask = (1U << n) - 1;
      vo = _mm512_maskz_loadu_ps(mask, o);
      vc = _mm512_maskz_loadu_ps(mask, c);
      vco = _mm512_sub_ps(vo, vc);
      
      sign_vco = _mm512_cmp_ps_mask(vco, _mm512_setzero_ps(), _CMP_GT_OQ);
      vsco = _mm512_mask_blend_ps(sign_vco, vnhalf, vhalf);
      vsco = _mm512_maskz_mov_ps(mask, vsco);
      if (n <= 8) {
        auto code_u8 = (uint8_t *)code_u16;
        *code_u8 = sign_vco;
      } else
        *code_u16 = sign_vco;
      n = 0;
    } else [[likely]] {
      vo = _mm512_loadu_ps(o);
      vc = _mm512_loadu_ps(c);
      vco = _mm512_sub_ps(vo, vc);
      
      sign_vco = _mm512_cmp_ps_mask(vco, _mm512_setzero_ps(), _CMP_GT_OQ);
      vsco = _mm512_mask_blend_ps(sign_vco, vnhalf, vhalf);
      *code_u16++ = sign_vco;
      o += 16, c += 16, n -= 16;
    }
    vsum_co_co = _mm512_fmadd_ps(vco, vco, vsum_co_co);
    vsum_co_sco = _mm512_fmadd_ps(vco, vsco, vsum_co_sco);
    vsum_c_sco = _mm512_fmadd_ps(vc, vsco, vsum_c_sco);
    if constexpr (metric == DistFunc::IP)
      vsum_co_c = _mm512_fmadd_ps(vco, vc, vsum_co_c);
  } while (n);
  float oc_sqr = _mm512_reduce_add_ps(vsum_co_co);
  float ip_co_sco = _mm512_reduce_add_ps(vsum_co_sco);
  float ip_c_sco = _mm512_reduce_add_ps(vsum_c_sco);
  float ip_co_c = _mm512_reduce_add_ps(vsum_co_c);

  float tmp_error = 0;
  if constexpr (with_err)
    tmp_error =
        std::sqrt(oc_sqr) * kRabitqEpsilon *
        std::sqrt((((oc_sqr * dim * .25) / (ip_co_sco * ip_co_sco)) - 1) /
                  (dim - 1));
  if constexpr (metric == DistFunc::L2) {
    float fa = oc_sqr + 2 * oc_sqr * ip_c_sco / ip_co_sco;
    float fr = -2 * oc_sqr / ip_co_sco;
    float fe = 2 * tmp_error;
    return std::make_tuple(fa, fr, fe);
  } else if constexpr (metric == DistFunc::IP) {
    float fa = 1 - ip_co_c + oc_sqr * ip_c_sco / ip_co_sco;
    float fr = -oc_sqr / ip_co_sco;
    float fe = 1 * tmp_error;
    return std::make_tuple(fa, fr, fe);
  }
}
#endif

template <DistFunc metric = DistFunc::L2, bool with_err = false>
  requires(metric == DistFunc::L2)
inline auto rabitq_impl_legacy(size_t dim, const float *o, const float *c,
                               void *code) {
  auto code_u8 = (uint8_t *)code;
  float oc_sqr = 0;
  float ip_resi_sdiff = 0;
  float ip_cent_sdiff = 0;
  float ip_diff_cent = 0;
  for (size_t i = 0; i < dim; ++i) {
    auto diff = o[i] - c[i];
    oc_sqr += diff * diff;
    float sign_diff = diff > 0 ? .5 : -.5;
    ip_resi_sdiff += diff * sign_diff;
    ip_cent_sdiff += c[i] * sign_diff;
    if constexpr (metric == DistFunc::IP)
      ip_diff_cent += diff * c[i];
    if (i % 8 == 0)
      code_u8[i / 8] = 0;
    code_u8[i / 8] |= (diff > 0 ? 1 : 0) << (i % 8);
  }
  float tmp_error = 0;
  if constexpr (with_err)
    tmp_error =
        std::sqrt(oc_sqr) * kRabitqEpsilon *
        std::sqrt(
            (((oc_sqr * dim * .25) / (ip_resi_sdiff * ip_resi_sdiff)) - 1) /
            (dim - 1));
  if constexpr (metric == DistFunc::L2) {
    float fa = oc_sqr + 2 * oc_sqr * ip_cent_sdiff / ip_resi_sdiff;
    float fr = -2 * oc_sqr / ip_resi_sdiff;
    float fe = 2 * tmp_error;
    return std::make_tuple(fa, fr, fe);
  } else if constexpr (metric == DistFunc::IP) {
    float fa = 1 - ip_diff_cent + oc_sqr * ip_cent_sdiff / ip_resi_sdiff;
    float fr = -oc_sqr / ip_resi_sdiff;
    float fe = 1 * tmp_error;
    return std::make_tuple(fa, fr, fe);
  }
}
} 


template <DistFunc metric = DistFunc::L2, bool with_err = false>
  requires(metric == DistFunc::L2)
inline auto rabitq_impl(size_t dim, const float *o, const float *c,
                        void *code) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  return detail::rabitq_impl_avx512<metric, with_err>(dim, o, c, code);
#else
  return detail::rabitq_impl_legacy<metric, with_err>(dim, o, c, code);
#endif
}

inline auto prepare_query(size_t dim, const float *q, uint8_t *qlut,
                          float *lut) {
  const size_t lut_size = get_lut_size(dim);
  pack_lut(dim, q, lut);
  auto [lo, hi] = minmax(lut, lut_size);
  float delta = (hi - lo) / 255;
  float offset = lo * lut_size / 16 + -.5f * std::accumulate(q, q + dim, 0.f);
  quantize(lut_size, lo, delta, lut, qlut);
  return std::make_tuple(delta, offset);
}
