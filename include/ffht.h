#pragma once

namespace ffht {

using fht_float_func_t = void (*)(float *);
extern fht_float_func_t fht_float_tbl[];

inline void dumb_fht(float *buf, int log_n) {
  int n = 1 << log_n;
  for (int i = 0; i < log_n; ++i) {
    int s1 = 1 << i;
    int s2 = s1 << 1;
    for (int j = 0; j < n; j += s2) {
      for (int k = 0; k < s1; ++k) {
        float u = buf[j + k];
        float v = buf[j + k + s1];
        buf[j + k] = u + v;
        buf[j + k + s1] = u - v;
      }
    }
  }
}

inline int fht_float [[maybe_unused]] (float *buf, int log_n) {
  if (log_n <= 0) [[unlikely]]
    return -1;
  if (log_n > 30) [[unlikely]]
    dumb_fht(buf, log_n);
  else
    fht_float_tbl[log_n - 1](buf);
  return 0;
}
} // namespace ffht