#pragma once

#include "conn_mgr.h"
#include "define.h"
#include "fast_ip.h"
#include "index_utils.h"
#include "rabitq.h"
#include <lcoro/rdma_worker.h>
#include <myutils/get_clock.h>
#include <myutils/heap.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include <ankerl/unordered_dense.h>

struct NbEntryIdHash {
  using is_avalanching = void;
  auto operator()(const NbEntry &x) const noexcept -> uint64_t {
    return ankerl::unordered_dense::detail::wyhash::hash(x.id());
  }
};
using BaseNeighborType = NeighborImpl<float, NbEntry>;

auto prune_vamana(auto *src, uint32_t *src_idxs, size_t nidxs,
                  uint32_t *dst_idxs, size_t M, const auto &dist_func,
                  float alpha = prune_alpha, bool saturate = false,
                  bool *need_fail_back = nullptr) {
  std::stable_sort( 
      src_idxs, src_idxs + nidxs,
      [&](const auto &a, const auto &b) { return src[a] < src[b]; });

  size_t pos = 0;
  std::vector<float> occlude_factors(nidxs, 0.0f);
  float cur_alpha = 1.0f;
  while (cur_alpha <= alpha && pos < M) {
    for (size_t i = 0; i < nidxs && pos < M; ++i) {
      auto &cur = src[src_idxs[i]];
      auto &cur_fac = occlude_factors[i];
      if (cur_fac > cur_alpha)
        continue;
      dst_idxs[pos++] = src_idxs[i];
      cur_fac = std::numeric_limits<float>::max(); 
      for (size_t j = i + 1; j < nidxs; ++j) {
        auto &other = src[src_idxs[j]];
        auto &other_fac = occlude_factors[j];
        if (other_fac > cur_alpha)
          continue;
        float djk = dist_func(cur, other);
        
        other_fac = djk == 0 ? std::numeric_limits<float>::max() / 2
                             : std::max(other_fac, other.dist / djk);
      }
    }
    if (need_fail_back && cur_alpha == 1.0f && pos == M)
      *need_fail_back = true;
    cur_alpha *= 1.2f;
  }
  size_t npruned = pos;
  if (saturate && pos < M) {
    for (size_t i = 0; i < nidxs && pos < M; ++i) {
      auto &cur = src[src_idxs[i]];
      auto &cur_fac = occlude_factors[i];
      if (cur_fac == std::numeric_limits<float>::max())
        continue;
      dst_idxs[pos++] = src_idxs[i];
    }
  }
  return std::make_tuple(npruned, pos);
}

auto prune_vamana(auto *src, size_t nsrc, uint32_t *dst_idxs, size_t M,
                  const auto &dist_func, float alpha = prune_alpha,
                  bool saturate = false, bool *need_fail_back = nullptr) {
  std::vector<uint32_t> src_idxs(nsrc);
  std::iota(src_idxs.begin(), src_idxs.end(), 0);
  return prune_vamana(src, src_idxs.data(), nsrc, dst_idxs, M, dist_func, alpha,
                      saturate, need_fail_back);
}

template <typename NeighborType> struct CandidateHeap {
  NeighborType *retset{nullptr};
  std::reverse_iterator<NeighborType *> cands;
  uint32_t ncand{0};
  uint32_t nret{0};
  uint32_t L{0};
  uint32_t max_L{0};
  ~CandidateHeap() { free(retset); }
  constexpr auto data() { return retset; }
  constexpr auto operator[](size_t i) { return retset[i]; }
  constexpr auto has_candidates() const { return ncand > 0; }
  void set_L(uint32_t L, bool force = false) {
    if (L > max_L || force) {
      max_L = L;
      retset = (NeighborType *)realloc(retset, sizeof(NeighborType) * max_L);
    }
    this->L = L;
    cands = std::make_reverse_iterator(retset + L);
  }
  void clear() { ncand = nret = 0; }
  constexpr void push_back(const NeighborType &n) {
    internal_assert(ncand < L);
    cands[ncand++] = n;
  }
  constexpr void init() {
    internal_assert(ncand <= L);
    minmax_heap::heapify(cands, ncand);
  }
  constexpr bool push_cand(const NeighborType &n) {
    internal_assert(ncand + nret <= L);
    if (ncand + nret < L) [[unlikely]] {
      
      ++ncand;
      minmax_heap::heap_up(cands, ncand, n, ncand);
      return true;
    } else if (ncand == L || (ncand > 0 && cands[0] > retset[0])) [[likely]] {
      
      if (n < cands[0]) {
        
        minmax_heap::heap_down_impl(cands, ncand, n, 1, std::greater{});
        return true;
      }
    } else if (n < retset[0]) [[unlikely]] {
      
      --nret;
      binary_heap::heap_down(retset, nret, retset[nret]);
      ++ncand;
      minmax_heap::heap_up(cands, ncand, n, ncand);
      return true;
    }
    return false;
  }
  constexpr auto max_cand_idx() const {
    if (ncand >= 3) [[likely]]
      return cands[1] < cands[2] ? 1 : 2;
    else
      return ncand == 1 ? 0 : (cands[0] < cands[1] ? 0 : 1);
  }
  constexpr auto max_cand() const { return cands[max_cand_idx()]; }
  constexpr auto pop_cand() {
    internal_assert(ncand > 0);
    internal_assert(ncand + nret <= L);
    uint32_t best = max_cand_idx();
    auto res = cands[best];
    
    --ncand;
    minmax_heap::heap_down_impl(cands, ncand, cands[ncand], best + 1,
                                std::less{});
    return res;
  }
  constexpr auto max_res() const { return retset[0]; }
  constexpr void push_res(const NeighborType &n) {
    internal_assert(ncand + nret < L); 
    ++nret;
    binary_heap::heap_up(retset, nret, n, nret);
  }
  constexpr void sort_result() { std::sort_heap(retset, retset + nret); }
};

struct Trace {
  uint64_t n_hops;                  
  uint64_t n_checks;                
  uint64_t read_bytes;              
  uint64_t n_enq_cand;              
  uint64_t n_enq_res;               
  uint64_t n_fwd_core;              
  uint64_t n_append_try;            
  uint64_t n_append_success;        
  uint64_t n_prune_cand;            
  uint64_t n_prune_reads;           
  uint64_t n_prune_is_visited;      
  uint64_t n_modify_slot_try;       
  uint64_t n_modify_slot_success;   
  uint64_t search_io_total_cycle;   
  uint64_t search_lock_total_cycle; 
  uint64_t search_total_cycle;      
  uint64_t insert_io_r_total_cycle; 
  uint64_t insert_io_w_total_cycle; 
  uint64_t insert_lock_total_cycle; 
  uint64_t insert_total_cycle;      
  uint64_t search_time;             
  uint64_t write_fwd_time;          
  uint64_t add_rlink_time;          
  uint64_t fail_back_cnt;
  uint64_t fail_back_check;
  void reset() { memset(this, 0, sizeof(*this)); }
  std::string search_trace_str() const {
    return std::format(
        "n_hops:{},n_checks:{},read_bytes:{},n_enq_cand:{},n_enq_res:{},"
        "s_io_total_cycle:{},s_lock_total_cycle:{},s_total_cycle:{}",
        n_hops, n_checks, read_bytes, n_enq_cand, n_enq_res,
        search_io_total_cycle, search_lock_total_cycle, search_total_cycle);
  }
  std::string insert_trace_str() const {
    return std::format("n_fwd_core:{},n_append_try:{},n_append_success:{},"
                       "n_prune_cand:{},n_prune_reads:{},n_prune_is_visited:{},"
                       "n_modify_slot_try:{},n_modify_slot_success:{},"
                       "i_io_r_total_cycle:{},i_io_w_total_cycle:{},"
                       "i_lock_total_cycle:{},i_total_cycle:{},fail_back_cnt:{}"
                       ",fail_back_check:{}",
                       n_fwd_core, n_append_try, n_append_success, n_prune_cand,
                       n_prune_reads, n_prune_is_visited, n_modify_slot_try,
                       n_modify_slot_success, insert_io_r_total_cycle,
                       insert_io_w_total_cycle, insert_lock_total_cycle,
                       insert_total_cycle, fail_back_cnt, fail_back_check);
  }
};
template <> struct std::formatter<Trace> : EmptyParser {
  auto format(const Trace &trace, auto &ctx) const {
    return std::format_to(
        ctx.out(),
        "Trace({},{},search_time:{},write_fwd_time:{},add_rlink_time:{})",
        trace.search_trace_str(), trace.insert_trace_str(), trace.search_time,
        trace.write_fwd_time, trace.add_rlink_time);
  }
};

struct RLock {
  std::vector<MapInfo> map_infos;
  void init(ConnManager &conn_mgr, size_t tid,
            const std::vector<std::string> &lock_cfgs) {
    for (auto &lock_cfg : lock_cfgs) {
      auto pos = lock_cfg.find_last_of(':');
      auto uid = std::stoi(lock_cfg.substr(pos + 1));
      auto server_addr = lock_cfg.substr(0, pos);
      auto &conn = conn_mgr.get_or_create_conn(server_addr, tid);
      auto rmr = conn->query_rmr_sync(uid);
      map_infos.emplace_back(conn, rmr.addr, rmr.rkey);
    }
  }
  
  void sync_reset(void *lbuf, uint32_t lkey) {
    memset(lbuf, 0, raw_lock_tbl_size);
    for (auto &[conn, addr, rkey] : map_infos)
      conn->sync_write(addr, rkey, lbuf, raw_lock_tbl_size, lkey);
  }

  Task<bool> lock_cas(size_t i, void *lbuf, uint32_t lkey) {
    const size_t lock_idx = i % map_infos.size();
    const size_t lock_off = i / map_infos.size();
    auto [conn, addr, rkey] = map_infos[lock_idx];
    const uint64_t lock_addr =
        addr +
        lock_off % (raw_lock_tbl_size / sizeof(uint64_t)) * sizeof(uint64_t);
    co_await conn->cas(lock_addr, rkey, lbuf, lkey, 0, 1);
    co_return *((uint64_t *)lbuf) == 0;
  }
  Task<bool> lock_mask_cas(size_t i, void *lbuf, uint32_t lkey) {
    const size_t lock_idx = i % map_infos.size();
    const size_t lock_off = i / map_infos.size();
    auto [conn, addr, rkey] = map_infos[lock_idx];
    const size_t lock_norm_bit = lock_off % (raw_lock_tbl_size * 8);
    const size_t word_idx = lock_norm_bit / 64; 
    const size_t bit_idx = lock_norm_bit % 64;
    const uint64_t lock_addr = addr + word_idx * sizeof(uint64_t);
    const uint64_t mask = 1ull << bit_idx;
    co_await conn->cas_masked(lock_addr, rkey, lbuf, lkey, 0, mask, mask, mask);
    co_return (*((uint64_t *)lbuf) & mask) == 0;
  }
  auto unlock_write(size_t i, void *, uint32_t) {
    const size_t lock_idx = i % map_infos.size();
    const size_t lock_off = i / map_infos.size();
    auto [conn, addr, rkey] = map_infos[lock_idx];
    const uint64_t lock_addr =
        addr +
        lock_off % (raw_lock_tbl_size / sizeof(uint64_t)) * sizeof(uint64_t);
    uint64_t zero = 0;
    return conn->write(lock_addr, rkey, &zero, sizeof(uint64_t), 0);
  }
  Task<> unlock_mask_cas(size_t i, void *lbuf, uint32_t lkey) {
    const size_t lock_idx = i % map_infos.size();
    const size_t lock_off = i / map_infos.size();
    auto [conn, addr, rkey] = map_infos[lock_idx];
    const size_t lock_norm_bit = lock_off % (raw_lock_tbl_size * 8);
    const size_t word_idx = lock_norm_bit / 64; 
    const size_t bit_idx = lock_norm_bit % 64;
    const uint64_t lock_addr = addr + word_idx * sizeof(uint64_t);
    const uint64_t mask = 1ull << bit_idx;
    co_await conn->cas_masked(lock_addr, rkey, lbuf, lkey, mask, 0, mask, mask);
    expect_eq(*((uint64_t *)lbuf) & mask, mask);
  }
};

struct alignas(cache_line_size) CoroContext {
  constexpr static bool enable_trace = true;
  constexpr static bool enable_r_print = false;
  constexpr static bool enable_fail_back = false;
  constexpr static bool force_lock = true;
  constexpr static size_t max_lock_retry = 1000000; 
  uint16_t tid;
  uint16_t cid;
  uint32_t scratch_top;
  uint8_t *scratch;
  size_t scratch_size;
  uint32_t scratch_lkey;
  CandidateHeap<BaseNeighborType> retset; 
  ankerl::unordered_dense::map<NbEntry, uint8_t *, NbEntryIdHash> visited;

  void reset() {
    visited.clear();
    retset.clear();
    scratch_top = 0;
  }
  uint8_t *alloc_scratch(size_t size, size_t align = cache_line_size) {
    auto ptr = scratch + scratch_top;
    scratch_top = upper_align(scratch_top + size, align);
    expect_le(scratch_top, scratch_size);
    return ptr;
  }
  uint8_t *get_tmp_buf(size_t size) { 
    expect_le(scratch_top + size, scratch_size);
    return scratch + scratch_top;
  }

  auto get_item_helper(size_t i, ConnManager &conn_mgr, size_t cfg_id) {
    auto &cfg = conn_mgr.cfgs[cfg_id];
    auto &map_info = conn_mgr.map_infos[cfg_id];
    return get_item(cfg, map_info, conn_mgr.max_nthreads, i, tid);
  }

  Task<uint8_t *> read(size_t i, ConnManager &conn_mgr, size_t cfg_id,
                       size_t size, size_t alloc_size = 0,
                       size_t align = cache_line_size) {
    auto lbuf = alloc_scratch(alloc_size == 0 ? size : alloc_size, align);
    auto [conn, addr, rkey] = get_item_helper(i, conn_mgr, cfg_id);
    co_await conn->read(addr, rkey, lbuf, size, conn_mgr.scratch_mr->lkey);
    co_return lbuf;
  }

  Task<int> write_slot(GraphConfig &cfg, rdma::Conn &conn, uint64_t raddr,
                       uint32_t rkey, size_t slot_idx, NbEntry new_nbe,
                       float *new_vec, float *cur_vec, uint8_t *code_lbuf,
                       Trace *trace) {
    uint64_t tmp_start_cycles, tmp_end_cycles;
    auto rslot_addr = raddr + cfg.nbrs_offset() + slot_idx * sizeof(NbEntry);
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    
    co_await conn->write(rslot_addr, rkey, &new_nbe, sizeof(NbEntry), 0);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->insert_io_w_total_cycle += tmp_end_cycles - tmp_start_cycles;
    }
    if (!new_vec || !cur_vec) 
      co_return 0;
    
    
    auto [bcode, f_add, f_rescale] = cfg.get_code(code_lbuf, 0);
    std::tie(*f_add, *f_rescale, std::ignore) =
        rabitq_impl(cfg.dim(), new_vec, cur_vec, bcode);
    auto rcode_addr = raddr + cfg.codes_offset() + slot_idx * cfg.code_size();
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    co_await conn->write(rcode_addr, rkey, code_lbuf, cfg.code_size(),
                         scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->insert_io_w_total_cycle += tmp_end_cycles - tmp_start_cycles;
    }
    co_return 0;
  }

  Task<int> modify_slot(GraphConfig &cfg, rdma::Conn &conn, uint64_t raddr,
                        uint32_t rkey, size_t slot_idx, NbEntry *lslot,
                        NbEntry old_nbe, NbEntry new_nbe, float *new_vec,
                        float *cur_vec, uint8_t *code_lbuf, Trace *trace) {
    uint64_t tmp_start_cycles, tmp_end_cycles;
    auto rslot_addr = raddr + cfg.nbrs_offset() + slot_idx * sizeof(NbEntry);
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    co_await conn->cas(rslot_addr, rkey, lslot, scratch_lkey, old_nbe._raw,
                       new_nbe._raw);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->insert_io_w_total_cycle += tmp_end_cycles - tmp_start_cycles;
    }
    if (*lslot != old_nbe) 
      co_return 2;
    if (!new_vec || !cur_vec) 
      co_return 0;
    
    
    auto [bcode, f_add, f_rescale] = cfg.get_code(code_lbuf, 0);
    std::tie(*f_add, *f_rescale, std::ignore) =
        rabitq_impl(cfg.dim(), new_vec, cur_vec, bcode);
    auto rcode_addr = raddr + cfg.codes_offset() + slot_idx * cfg.code_size();
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    co_await conn->write(rcode_addr, rkey, code_lbuf, cfg.code_size(),
                         scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->insert_io_w_total_cycle += tmp_end_cycles - tmp_start_cycles;
    }
    co_return 0;
  }

  Task<bool> lock(size_t i, RLock *rlock, void *lbuf, uint32_t lkey) {
    if constexpr (!force_lock) {
      bool locked = false;
      do {
        locked = co_await rlock->lock_mask_cas(i, lbuf, lkey);
      } while (!locked);
      co_return true;
    } else {
      for (size_t j = 0; j < max_lock_retry; ++j)
        if (co_await rlock->lock_mask_cas(i, lbuf, lkey))
          co_return true;
      co_return true;
    }
  }
  auto unlock(size_t i, RLock *rlock, void *lbuf, uint32_t lkey) {
    return rlock->unlock_mask_cas(i, lbuf, lkey);
  }

  Task<uint8_t *> raw_search(const float *q, uint32_t L, bool rotate,
                             auto is_del_func, ConnManager &conn_mgr,
                             size_t cfg_id, RLock *rlock, Trace *trace) {
    uint64_t search_start_cycle, search_end_cycle;
    uint64_t tmp_start_cycle, tmp_end_cycle;
    if constexpr (enable_trace) {
      search_start_cycle = get_cycles();
    }
    auto &cfg = conn_mgr.cfgs[cfg_id];
    size_t dim = cfg.dim();
    retset.set_L(L);
    reset();

    auto rq = (float *)alloc_scratch(cfg.raw_item_size());
    if (rotate)
      cfg.rotate(q, rq);
    else
      memmove(rq, q, cfg.vec_size());
    auto tmp_lock_buf = (uint64_t *)(rq + cfg.dim());

    
    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    visited[NbEntry(cfg.ep())] = co_await read(
        cfg.ep(), conn_mgr, cfg_id, cfg.vec_size(), cfg.raw_item_size());
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->search_io_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    if constexpr (enable_trace) {
      trace->n_hops++;
      trace->n_checks++;
      trace->read_bytes += cfg.vec_size();
    }
    retset.push_back(BaseNeighborType{
        l2sqr(rq, cfg.get_vec(visited[NbEntry(cfg.ep())]), dim),
        NbEntry(cfg.ep())});
    retset.init();
    for (size_t it = 0; it < max_search_iters && retset.has_candidates();
         ++it) {
      auto cand = retset.pop_cand();
      auto cand_nbe = cand.id;
      auto cand_item = visited[cand_nbe];
      if (!is_del_func(cand_nbe)) {
        retset.push_res(cand);
        if constexpr (enable_trace) {
          trace->n_enq_res++;
        }
      }
      
      auto nbrs = cfg.get_nbrs(cand_item);
      if (rlock) {
        if constexpr (enable_trace) {
          tmp_start_cycle = get_cycles();
        }
        auto locked =
            co_await lock(cand_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
        expect(locked);
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->search_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
        }
      }
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      auto [conn, addr, rkey] =
          get_item_helper(cand_nbe.id(), conn_mgr, cfg_id);
      co_await conn->read(addr + cfg.nbrs_offset(), rkey, nbrs, cfg.nbrs_size(),
                          scratch_lkey);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->search_io_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
      if (rlock) {
        if constexpr (enable_trace) {
          tmp_start_cycle = get_cycles();
        }
        co_await unlock(cand_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->search_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
        }
      }
      if constexpr (enable_trace) {
        trace->n_hops++;
        trace->read_bytes += cfg.nbrs_size();
      }

      for (size_t i = 0; i < cfg.M(); ++i) {
        auto nnbe = nbrs[i];
        if (!NbEntry::is_valid(nnbe) ||
            nnbe.prune_flag() == PruneFlag::REDUNDANT)
          continue;
        if (visited.contains(nnbe))
          continue;
        
        if constexpr (enable_trace) {
          tmp_start_cycle = get_cycles();
        }
        visited[nnbe] = co_await read(nnbe.id(), conn_mgr, cfg_id,
                                      cfg.vec_size(), cfg.raw_item_size());
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->search_io_total_cycle += tmp_end_cycle - tmp_start_cycle;
          trace->n_checks++;
          trace->read_bytes += cfg.vec_size();
        }
        auto dd = l2sqr(rq, cfg.get_vec(visited[nnbe]), dim);
        bool added = retset.push_cand(BaseNeighborType{dd, nnbe});
        if constexpr (enable_trace) {
          trace->n_enq_cand += added ? 1 : 0;
        }
      }
    }
    if constexpr (enable_trace) {
      trace->n_checks += visited.size();
    }
    retset.sort_result();
    if constexpr (enable_trace) {
      search_end_cycle = get_cycles();
      trace->search_total_cycle += search_end_cycle - search_start_cycle;
    }
    co_return (uint8_t *) rq;
  }

  Task<int> _unused_raw_add_rlink_ordered(NbEntry new_nbe,
                                          BaseNeighborType cur_nb,
                                          ConnManager &conn_mgr, size_t cfg_id,
                                          RLock *rlock, uint64_t *tmp_lock_buf,
                                          Trace *trace) {
    uint64_t tmp_start_cycle, tmp_end_cycle;
    auto &cfg = conn_mgr.cfgs[cfg_id];
    auto cur_nbe = cur_nb.id;
    auto cur_dist = cur_nb.dist;
    auto cur_item = visited[cur_nbe]; 
    expect(cur_item != nullptr);
    expect(rlock != nullptr);
    auto cur_nbrs = cfg.get_nbrs(cur_item);
    auto [conn, raddr, rkey] = get_item_helper(cur_nbe.id(), conn_mgr, cfg_id);
    
    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    auto locked =
        co_await lock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
    expect(locked);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    
    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    co_await conn->read(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                        cfg.nbrs_size(), scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto nn = cur_nbrs[j]; 
      if (!NbEntry::is_valid(nn) || nn.prune_flag() == PruneFlag::REDUNDANT) {
        if constexpr (enable_trace) {
          tmp_start_cycle = get_cycles();
        }
        
        co_await conn->write(raddr + cfg.nbrs_offset() + j * sizeof(NbEntry),
                             rkey, &new_nbe, sizeof(NbEntry), 0);
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->insert_io_w_total_cycle += tmp_end_cycle - tmp_start_cycle;
          trace->n_append_try++;
          trace->n_append_success++;
          tmp_start_cycle = get_cycles();
        }
        co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
        }
        co_return 0; 
      }
    }
    
    std::vector<BaseNeighborType> cand_nbrs(cfg.M() + 1);
    std::vector<uint32_t> cand_idxs;
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto &nn = cur_nbrs[j];
      BaseNeighborType nbr;
      if (!visited.contains(nn) || visited[nn] == nullptr) {
        if constexpr (enable_trace) {
          trace->n_prune_reads++;
          tmp_start_cycle = get_cycles();
        }
        visited[nn] =
            co_await read(nn.id(), conn_mgr, cfg_id,
                          cfg.vec_size()); 
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
        }
      }
      nbr.id = cur_nbrs[j];
      nbr.dist =
          l2sqr(cfg.get_vec(visited[nbr.id]), cfg.get_vec(cur_item), cfg.dim());
      cand_nbrs[j] = nbr;
      cand_idxs.push_back(j);
    }
    cand_nbrs[cfg.M()] = BaseNeighborType{cur_dist, new_nbe};
    cand_idxs.push_back(cfg.M());
    if constexpr (enable_trace) {
      trace->n_prune_cand += cand_idxs.size();
    }
    std::vector<uint32_t> bwd_idxs(cand_idxs.size());
    auto [n_bwd_core, n_bwd_ret] = prune_vamana(
        cand_nbrs.data(), cand_idxs.data(), cand_idxs.size(), bwd_idxs.data(),
        cand_idxs.size(),
        [&](const auto &a, const auto &b) {
          return l2sqr(cfg.get_vec(visited[a.id]), cfg.get_vec(visited[b.id]),
                       cfg.dim());
        },
        prune_alpha, false);
    for (size_t j = 0; j < cfg.M(); ++j) {
      cur_nbrs[j] =
          j < n_bwd_core
              ? cand_nbrs[bwd_idxs[j]].id.with_prune_flag(PruneFlag::CORE)
              : NbEntry::invalid();
    }
    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    co_await conn->write(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                         cfg.nbrs_size(), scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_io_w_total_cycle += tmp_end_cycle - tmp_start_cycle;
      tmp_start_cycle = get_cycles();
    }
    co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    co_return 0;
  }

  
  Task<int> raw_add_rlink(NbEntry new_nbe, BaseNeighborType cur_nb,
                          bool full_trim, ConnManager &conn_mgr, size_t cfg_id,
                          RLock *rlock, uint64_t *tmp_lock_buf, Trace *trace) {
    uint64_t tmp_start_cycle, tmp_end_cycle;
    auto &cfg = conn_mgr.cfgs[cfg_id];
    auto cur_nbe = cur_nb.id;
    auto cur_dist = cur_nb.dist;
    auto cur_item = visited[cur_nbe]; 
    expect(cur_item != nullptr);
    auto cur_nbrs = cfg.get_nbrs(cur_item);
    auto [conn, raddr, rkey] = get_item_helper(cur_nbe.id(), conn_mgr, cfg_id);
    if (rlock) {
      
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      auto locked =
          co_await lock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
      expect(locked);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
      
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      co_await conn->read(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                          cfg.nbrs_size(), scratch_lkey);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
    }
    
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto nn = cur_nbrs[j]; 
      if (!NbEntry::is_valid(nn) || nn.prune_flag() == PruneFlag::REDUNDANT) {
        if (rlock) {
          if constexpr (enable_trace) {
            tmp_start_cycle = get_cycles();
          }
          
          co_await conn->write(raddr + cfg.nbrs_offset() + j * sizeof(NbEntry),
                               rkey, &new_nbe, sizeof(NbEntry), 0);
          if constexpr (enable_trace) {
            tmp_end_cycle = get_cycles();
            trace->insert_io_w_total_cycle += tmp_end_cycle - tmp_start_cycle;
            trace->n_append_try++;
            trace->n_append_success++;
            tmp_start_cycle = get_cycles();
          }
          co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
          if constexpr (enable_trace) {
            tmp_end_cycle = get_cycles();
            trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
          }
          co_return 0; 
        }
        int rc =
            co_await modify_slot(cfg, conn, raddr, rkey, j, cur_nbrs + j, nn,
                                 new_nbe, nullptr, nullptr, nullptr, trace);
        if constexpr (enable_trace) {
          trace->n_append_try++;
          trace->n_append_success += rc == 0 ? 1 : 0;
        }
        if (rc == 0)
          co_return 0; 
        
      }
    }
    
    std::vector<BaseNeighborType> cand_nbrs(cfg.M() + 1);
    std::vector<uint32_t> cand_idxs;
    
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto &nn = cur_nbrs[j];
      BaseNeighborType nbr;
      if (!visited.contains(nn) || visited[nn] == nullptr) {
        if (full_trim) [[unlikely]] 
        {
          if constexpr (enable_trace) {
            trace->n_prune_reads++;
            
            tmp_start_cycle = get_cycles();
          }
          visited[nn] =
              co_await read(nn.id(), conn_mgr, cfg_id,
                            cfg.vec_size()); 
          if constexpr (enable_trace) {
            tmp_end_cycle = get_cycles();
            trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
          }
        } else
          continue; 
      }
      nbr.id = cur_nbrs[j];
      nbr.dist =
          l2sqr(cfg.get_vec(visited[nbr.id]), cfg.get_vec(cur_item), cfg.dim());
      cand_nbrs[j] = nbr;
      cand_idxs.push_back(j);
    }
    cand_nbrs[cfg.M()] = BaseNeighborType{cur_dist, new_nbe};
    cand_idxs.push_back(cfg.M());
    if constexpr (enable_trace) {
      trace->n_prune_cand += cand_idxs.size();
    }
    std::vector<uint32_t> bwd_idxs(cand_idxs.size());
    auto [n_bwd_core, n_bwd_ret] = prune_vamana(
        cand_nbrs.data(), cand_idxs.data(), cand_idxs.size(), bwd_idxs.data(),
        cand_idxs.size(),
        [&](const auto &a, const auto &b) {
          return l2sqr(cfg.get_vec(visited[a.id]), cfg.get_vec(visited[b.id]),
                       cfg.dim());
        },
        prune_alpha, true);
    if (rlock) {
      for (size_t j = 0; j < cfg.M(); ++j) {
        cur_nbrs[j] =
            j < n_bwd_core
                ? cand_nbrs[bwd_idxs[j]].id.with_prune_flag(PruneFlag::CORE)
                : NbEntry::invalid();
      }
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      co_await conn->write(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                           cfg.nbrs_size(), scratch_lkey);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_io_w_total_cycle += tmp_end_cycle - tmp_start_cycle;
        tmp_start_cycle = get_cycles();
      }
      co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
    } else {
      for (size_t j = 0; j < bwd_idxs.size() - 1; ++j) {
        auto slot_idx = bwd_idxs[j];
        auto old_nbe = cand_nbrs[slot_idx].id;
        auto tmp_nbe = cand_nbrs[slot_idx].id;
        if (tmp_nbe == new_nbe) {
          slot_idx = bwd_idxs.back(); 
          old_nbe = cand_nbrs[slot_idx].id;
        } else if (j < n_bwd_core) {
          if (tmp_nbe.prune_flag() == PruneFlag::CORE)
            continue;
          tmp_nbe.set_prune_flag(PruneFlag::CORE);
        } else {
          if (tmp_nbe.prune_flag() == PruneFlag::REDUNDANT)
            continue;
          tmp_nbe.set_prune_flag(PruneFlag::REDUNDANT);
        }
        int rc = co_await modify_slot(
            cfg, conn, raddr, rkey, slot_idx, cur_nbrs + slot_idx, old_nbe,
            tmp_nbe, nullptr, nullptr, nullptr, trace); 
        
        
        if constexpr (enable_trace) {
          trace->n_modify_slot_try++;
          trace->n_modify_slot_success += rc == 0 ? 1 : 0;
        }
      }
    }
    co_return 0;
  }

  Task<> raw_insert(NbEntry new_nbe, const float *vec, uint32_t L, bool rotate,
                    bool full_trim, auto is_del_func, ConnManager &conn_mgr,
                    size_t cfg_id, RLock *rlock, bool with_search_lock,
                    Trace *trace) {
    uint64_t insert_start_cycles, insert_end_cycles;
    if constexpr (enable_trace) {
      insert_start_cycles = get_cycles();
    }
    auto &cfg = conn_mgr.cfgs[cfg_id];
    
    uint64_t tmp_start_cycles, tmp_end_cycles;
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    auto new_item =
        co_await raw_search(vec, L, rotate, is_del_func, conn_mgr, cfg_id,
                            with_search_lock ? rlock : nullptr, trace);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->search_time = tmp_end_cycles - tmp_start_cycles;
    }
    visited[new_nbe] = new_item;
    auto new_vec = cfg.get_vec(new_item);   
    auto new_nbrs = cfg.get_nbrs(new_item); 
    auto dist_func = [&](const auto &a, const auto &b) {
      return l2sqr(cfg.get_vec(visited[a.id]), cfg.get_vec(visited[b.id]),
                   cfg.dim());
    };
    
    std::vector<uint32_t> fwd_idxs(cfg.M());
    std::vector<uint32_t> src_idxs(retset.nret);
    std::iota(src_idxs.begin(), src_idxs.end(), 0);
    auto [n_fwd_core, n_fwd_ret] =
        prune_vamana(retset.data(), src_idxs.data(), src_idxs.size(),
                     fwd_idxs.data(), cfg.M(), dist_func, prune_alpha, false);
    if constexpr (enable_trace) {
      trace->n_fwd_core += n_fwd_core;
    }
    for (size_t i = 0; i < cfg.M(); ++i)
      new_nbrs[i] =
          i < n_fwd_core
              ? retset[fwd_idxs[i]].id.with_prune_flag(PruneFlag::CORE)
              : NbEntry::invalid();

    
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    auto [conn, addr, rkey] = get_item_helper(new_nbe.id(), conn_mgr, cfg_id);
    co_await conn->write(addr, rkey, new_item, cfg.raw_item_size(),
                         scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->write_fwd_time = tmp_end_cycles - tmp_start_cycles;
      trace->insert_io_w_total_cycle += tmp_end_cycles - tmp_start_cycles;
    }

    
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    new_nbe.set_prune_flag(PruneFlag::PENDING);
    auto tmp_lock_buf = (uint64_t *)alloc_scratch(sizeof(uint64_t));
    for (size_t i = 0; i < n_fwd_core; ++i)
      co_await raw_add_rlink(new_nbe, retset[fwd_idxs[i]], full_trim, conn_mgr,
                             cfg_id, rlock, tmp_lock_buf, trace);
    if constexpr (enable_trace) {
      insert_end_cycles = tmp_end_cycles = get_cycles();
      trace->add_rlink_time = tmp_end_cycles - tmp_start_cycles;
      trace->insert_total_cycle += insert_end_cycles - insert_start_cycles;
    }
  }

  Task<uint8_t *> search(const float *q, uint32_t L, bool rotate,
                         auto is_del_func, ConnManager &conn_mgr, size_t cfg_id,
                         RLock *rlock, Trace *trace) {
    uint64_t search_start_cycles, search_end_cycles;
    uint64_t tmp_start_cycles, tmp_end_cycles;
    if constexpr (enable_trace) {
      search_start_cycles = get_cycles();
    }
    auto &cfg = conn_mgr.cfgs[cfg_id];
    const size_t dim = cfg.dim();

    retset.set_L(L);
    reset();
    auto rq = (float *)alloc_scratch(cfg.item_size());
    if (rotate)
      cfg.rotate(q, rq);
    else
      memmove(rq, q, cfg.vec_size());
    auto qlut = (uint8_t *)(rq + dim); 
    auto tmp_lut = (float *)get_tmp_buf(get_lut_size(dim) * sizeof(float));
    auto [delta, offset] = prepare_query(dim, rq, qlut, tmp_lut);

    void *tmp_lock_buf = nullptr;
    if (rlock)
      tmp_lock_buf = alloc_scratch(sizeof(uint64_t));

    retset.push_back({std::numeric_limits<float>::max(), NbEntry(cfg.ep())});
    retset.init();
    for (size_t it = 0; it <= max_search_iters && retset.has_candidates();
         ++it) {
      auto cand = retset.pop_cand();
      auto [_d, cand_nbe] = cand;
      if (rlock) {
        if constexpr (enable_trace) {
          tmp_start_cycles = get_cycles();
        }
        auto locked =
            co_await lock(cand_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
        expect(locked);
        if constexpr (enable_trace) {
          tmp_end_cycles = get_cycles();
          trace->search_lock_total_cycle += tmp_end_cycles - tmp_start_cycles;
        }
      }
      if constexpr (enable_trace) {
        tmp_start_cycles = get_cycles();
      }
      auto cur_item = visited[cand_nbe] =
          co_await read(cand_nbe.id(), conn_mgr, cfg_id, cfg.item_size());
      if constexpr (enable_trace) {
        tmp_end_cycles = get_cycles();
        trace->search_io_total_cycle += tmp_end_cycles - tmp_start_cycles;
        trace->n_hops++;
        trace->read_bytes += cfg.item_size();
      }
      if (rlock) {
        if constexpr (enable_trace) {
          tmp_start_cycles = get_cycles();
        }
        co_await unlock(cand_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
        if constexpr (enable_trace) {
          tmp_end_cycles = get_cycles();
          trace->search_lock_total_cycle += tmp_end_cycles - tmp_start_cycles;
        }
      }
      
      
      
      auto g_add = cand.dist = l2sqr(rq, cfg.get_vec(cur_item), dim);
      if (!is_del_func(cand_nbe)) {
        retset.push_res(cand);
        if constexpr (enable_trace) {
          trace->n_enq_res++;
        }
      }
      auto nbrs = cfg.get_nbrs(cur_item);
      auto codes = cfg.get_codes(cur_item);
      auto tmp_pcode =
          (uint8_t *)get_tmp_buf(cfg.code_size() * fast_ip_batch_size);
      for (size_t i = 0; i < cfg.M(); i += fast_ip_batch_size) {
        const size_t batch_size = std::min(fast_ip_batch_size, cfg.M() - i);
        uint16_t accu_res[fast_ip_batch_size];
        pack_separate_code_impl(
            dim, [&](size_t j) { return codes + (i + j) * cfg.code_size(); },
            tmp_pcode);
        fast_ip_impl(dim, tmp_pcode, qlut, accu_res);
        
        for (size_t j = 0; j < batch_size; ++j) {
          auto nnbe = nbrs[i + j];
          if (!NbEntry::is_valid(nnbe) ||
              !visited.insert({nnbe, nullptr}).second)
            continue;
          
          
          auto [bcode, f_add, f_rescale] = cfg.get_code(codes, i + j);
          auto est_dist =
              *f_add + g_add + *f_rescale * (delta * accu_res[j] + offset);
          bool added = retset.push_cand({est_dist, nnbe});
          if constexpr (enable_trace) {
            trace->n_enq_cand += added ? 1 : 0;
          }
        }
      }
    }
    if constexpr (enable_trace) {
      trace->n_checks += visited.size();
    }
    retset.sort_result();
    if constexpr (enable_trace) {
      search_end_cycles = get_cycles();
      trace->search_total_cycle += search_end_cycles - search_start_cycles;
    }
    co_return (uint8_t *) rq;
  }

  Task<int> add_rlink_ordered(NbEntry new_nbe, float *new_vec,
                              BaseNeighborType cur_nb, ConnManager &conn_mgr,
                              size_t cfg_id, RLock *rlock,
                              uint64_t *tmp_lock_buf, Trace *trace) {
    uint64_t tmp_start_cycle, tmp_end_cycle;
    auto &cfg = conn_mgr.cfgs[cfg_id];
    auto cur_nbe = cur_nb.id;
    auto cur_dist = cur_nb.dist;
    auto cur_item = visited[cur_nbe]; 
    expect(cur_item != nullptr);
    expect(rlock != nullptr);
    auto cur_nbrs = cfg.get_nbrs(cur_item);
    auto cur_codes = cfg.get_codes(cur_item);
    auto [conn, raddr, rkey] = get_item_helper(cur_nbe.id(), conn_mgr, cfg_id);
    
    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    auto locked =
        co_await lock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
    expect(locked);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    
    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    co_await conn->read(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                        cfg.nbrs_size(), scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto nn = cur_nbrs[j]; 
      if (!NbEntry::is_valid(nn) || nn.prune_flag() == PruneFlag::REDUNDANT) {
        cur_nbrs[j] = new_nbe;
        auto [bcode, f_add, f_rescale] = cfg.get_code(cur_codes, j);
        std::tie(*f_add, *f_rescale, std::ignore) =
            rabitq_impl(cfg.dim(), new_vec, cfg.get_vec(cur_item), bcode);
        auto rcode_addr = raddr + cfg.codes_offset() + j * cfg.code_size();
        if constexpr (enable_trace) {
          tmp_start_cycle = get_cycles();
        }
        
        co_await conn->write(raddr + cfg.nbrs_offset() + j * sizeof(NbEntry),
                             rkey, cur_nbrs + j, sizeof(NbEntry), scratch_lkey);
        co_await conn->write(rcode_addr, rkey, bcode, cfg.code_size(),
                             scratch_lkey);
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->insert_io_w_total_cycle += tmp_end_cycle - tmp_start_cycle;
          trace->n_append_try++;
          trace->n_append_success++;
          tmp_start_cycle = get_cycles();
        }
        co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
        }
        co_return 0; 
      }
    }
    
    std::vector<BaseNeighborType> cand_nbrs(cfg.M() + 1);
    std::vector<uint32_t> cand_idxs;
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto &nn = cur_nbrs[j];
      BaseNeighborType nbr;
      if (!visited.contains(nn) || visited[nn] == nullptr) {
        if constexpr (enable_trace) {
          trace->n_prune_reads++;
          tmp_start_cycle = get_cycles();
        }
        visited[nn] =
            co_await read(nn.id(), conn_mgr, cfg_id,
                          cfg.vec_size()); 
        if constexpr (enable_trace) {
          tmp_end_cycle = get_cycles();
          trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
        }
      }
      nbr.id = cur_nbrs[j];
      nbr.dist =
          l2sqr(cfg.get_vec(visited[nbr.id]), cfg.get_vec(cur_item), cfg.dim());
      cand_nbrs[j] = nbr;
      cand_idxs.push_back(j);
    }
    cand_nbrs[cfg.M()] = BaseNeighborType{cur_dist, new_nbe};
    cand_idxs.push_back(cfg.M());
    if constexpr (enable_trace) {
      trace->n_prune_cand += cand_idxs.size();
    }
    std::vector<uint32_t> bwd_idxs(cand_idxs.size());
    auto [n_bwd_core, n_bwd_ret] = prune_vamana(
        cand_nbrs.data(), cand_idxs.data(), cand_idxs.size(), bwd_idxs.data(),
        cand_idxs.size(),
        [&](const auto &a, const auto &b) {
          return l2sqr(cfg.get_vec(visited[a.id]), cfg.get_vec(visited[b.id]),
                       cfg.dim());
        },
        prune_alpha, true);
    for (size_t j = 0; j < cfg.M(); ++j) {
      cur_nbrs[j] = cand_nbrs[bwd_idxs[j]].id.with_prune_flag(
          j < n_bwd_core ? PruneFlag::CORE : PruneFlag::REDUNDANT);
      auto [bcode, f_add, f_rescale] = cfg.get_code(cur_codes, j);
      std::tie(*f_add, *f_rescale, std::ignore) = rabitq_impl(
          cfg.dim(), cfg.get_vec(visited[cand_nbrs[bwd_idxs[j]].id]),
          cfg.get_vec(cur_item), bcode);
    }

    if constexpr (enable_trace) {
      tmp_start_cycle = get_cycles();
    }
    co_await conn->write(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                         cfg.nbrs_size() + cfg.M() * cfg.code_size(),
                         scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_io_w_total_cycle += tmp_end_cycle - tmp_start_cycle;
      tmp_start_cycle = get_cycles();
    }
    co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycle = get_cycles();
      trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
    }
    co_return 0;
  }

  
  Task<int> add_rlink(NbEntry new_nbe, float *new_vec, BaseNeighborType cur_nb,
                      bool full_trim, ConnManager &conn_mgr, size_t cfg_id,
                      RLock *rlock, uint64_t *tmp_lock_buf, Trace *trace) {
    uint64_t tmp_start_cycle, tmp_end_cycle;
    auto &cfg = conn_mgr.cfgs[cfg_id];
    auto cur_nbe = cur_nb.id;
    auto cur_dist = cur_nb.dist;
    auto cur_item = visited[cur_nbe]; 
    expect(cur_item != nullptr);
    auto cur_nbrs = cfg.get_nbrs(cur_item);
    auto cur_codes = cfg.get_codes(cur_item);
    auto [conn, raddr, rkey] = get_item_helper(cur_nbe.id(), conn_mgr, cfg_id);
    if (rlock) {
      
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      auto locked =
          co_await lock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
      expect(locked);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
      
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      co_await conn->read(raddr + cfg.nbrs_offset(), rkey, cur_nbrs,
                          cfg.nbrs_size(), scratch_lkey);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
    }
    
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto nn = cur_nbrs[j]; 
      if (nn.prune_flag() == PruneFlag::REDUNDANT) {
        auto code_lbuf = cur_codes + j * cfg.code_size();
        if (rlock) {
          co_await write_slot(cfg, conn, raddr, rkey, j, new_nbe, new_vec,
                              cfg.get_vec(cur_item), code_lbuf, trace);
          if constexpr (enable_trace) {
            trace->n_append_try++;
            trace->n_append_success++;
            tmp_start_cycle = get_cycles();
          }
          co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
          if constexpr (enable_trace) {
            tmp_end_cycle = get_cycles();
            trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
          }
          co_return 0; 
        }
        int rc = co_await modify_slot(cfg, conn, raddr, rkey, j, cur_nbrs + j,
                                      nn, new_nbe, new_vec,
                                      cfg.get_vec(cur_item), code_lbuf, trace);
        if constexpr (enable_trace) {
          trace->n_append_try++;
          trace->n_append_success += rc == 0 ? 1 : 0;
        }
        if (rc == 0)
          co_return 0; 
        
      }
    }
    
    std::vector<BaseNeighborType> cand_nbrs(cfg.M() + 1);
    std::vector<uint32_t> cand_idxs;
    float max_dist = 0;
    for (size_t j = 0; j < cfg.M(); ++j) {
      auto &nn = cur_nbrs[j];
      BaseNeighborType nbr;
      if (!visited.contains(nn) || visited[nn] == nullptr) {
        if (full_trim) [[unlikely]] 
        {
          if constexpr (enable_trace) {
            trace->n_prune_reads++;
            tmp_start_cycle = get_cycles();
          }
          visited[nn] =
              co_await read(nn.id(), conn_mgr, cfg_id,
                            cfg.vec_size()); 
          if constexpr (enable_trace) {
            tmp_end_cycle = get_cycles();
            trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
          }
        } else
          continue; 
      }
      nbr.id = cur_nbrs[j];
      nbr.dist =
          l2sqr(cfg.get_vec(visited[nbr.id]), cfg.get_vec(cur_item), cfg.dim());
      if constexpr (enable_r_print)
        max_dist = std::max(max_dist, nbr.dist);
      cand_nbrs[j] = nbr;
      cand_idxs.push_back(j);
    }
    if constexpr (enable_r_print)
      println("Rp: {}", max_dist);
    cand_nbrs[cfg.M()] = BaseNeighborType{cur_dist, new_nbe};
    cand_idxs.push_back(cfg.M());
    if constexpr (enable_trace) {
      trace->n_prune_cand += cand_idxs.size();
    }
    std::vector<uint32_t> bwd_idxs(cand_idxs.size());
    bool need_fail_back = false;
    auto [n_bwd_core, n_bwd_ret] = prune_vamana(
        cand_nbrs.data(), cand_idxs.data(), cand_idxs.size(), bwd_idxs.data(),
        cand_idxs.size(),
        [&](const auto &a, const auto &b) {
          return l2sqr(cfg.get_vec(visited[a.id]), cfg.get_vec(visited[b.id]),
                       cfg.dim());
        },
        prune_alpha, true, &need_fail_back);
    expect_eq(n_bwd_ret, cand_idxs.size());

    if constexpr (enable_fail_back) {
      if (!full_trim && need_fail_back) {
        
        for (size_t j = 0; j < cfg.M(); ++j) {
          auto &nn = cur_nbrs[j];
          if (!visited.contains(nn) || visited[nn] == nullptr) {
            if constexpr (enable_trace) {
              trace->n_prune_reads++;
              tmp_start_cycle = get_cycles();
            }
            visited[nn] =
                co_await read(nn.id(), conn_mgr, cfg_id,
                              cfg.vec_size()); 
            if constexpr (enable_trace) {
              tmp_end_cycle = get_cycles();
              trace->insert_io_r_total_cycle += tmp_end_cycle - tmp_start_cycle;
            }
            BaseNeighborType nbr;
            nbr.id = cur_nbrs[j];
            nbr.dist = l2sqr(cfg.get_vec(visited[nbr.id]),
                             cfg.get_vec(cur_item), cfg.dim());
            cand_nbrs[j] = nbr;
          }
        }

        bwd_idxs.resize(cand_nbrs.size());

        std::tie(n_bwd_core, n_bwd_ret) = prune_vamana(
            cand_nbrs.data(), cand_nbrs.size(), bwd_idxs.data(),
            cand_nbrs.size(),
            [&](const auto &a, const auto &b) {
              return l2sqr(cfg.get_vec(visited[a.id]),
                           cfg.get_vec(visited[b.id]), cfg.dim());
            },
            prune_alpha, true);
        expect_eq(n_bwd_ret, cand_nbrs.size());
      }
    }
    if constexpr (enable_trace) {
      if (need_fail_back)
        trace->fail_back_cnt++;
      trace->fail_back_check++;
    }
    for (size_t j = 0; j < n_bwd_ret - 1; ++j) {
      auto slot_idx = bwd_idxs[j];
      auto old_nbe = cand_nbrs[slot_idx].id;
      auto tmp_nbe = cand_nbrs[slot_idx].id;
      if (tmp_nbe == new_nbe) {
        slot_idx = bwd_idxs.back(); 
        old_nbe = cand_nbrs[slot_idx].id;
      } else if (j < n_bwd_core) {
        if (tmp_nbe.prune_flag() == PruneFlag::CORE)
          continue;
        tmp_nbe.set_prune_flag(PruneFlag::CORE);
      } else {
        if (tmp_nbe.prune_flag() == PruneFlag::REDUNDANT)
          continue;
        tmp_nbe.set_prune_flag(PruneFlag::REDUNDANT);
      }
      if (rlock) {
        co_await write_slot(
            cfg, conn, raddr, rkey, slot_idx, tmp_nbe,
            tmp_nbe == new_nbe ? new_vec : nullptr,
            tmp_nbe == new_nbe ? cfg.get_vec(cur_item) : nullptr,
            tmp_nbe == new_nbe ? cur_codes + slot_idx * cfg.code_size()
                               : nullptr,
            trace);
        if constexpr (enable_trace) {
          trace->n_modify_slot_try++;
          trace->n_modify_slot_success++;
        }
      } else {
        int rc = co_await modify_slot(
            cfg, conn, raddr, rkey, slot_idx, cur_nbrs + slot_idx, old_nbe,
            tmp_nbe, tmp_nbe == new_nbe ? new_vec : nullptr,
            tmp_nbe == new_nbe ? cfg.get_vec(cur_item) : nullptr,
            tmp_nbe == new_nbe ? cur_codes + slot_idx * cfg.code_size()
                               : nullptr,
            trace);
        
        
        if constexpr (enable_trace) {
          trace->n_modify_slot_try++;
          trace->n_modify_slot_success += rc == 0 ? 1 : 0;
        }
      }
    }
    if (rlock) {
      if constexpr (enable_trace) {
        tmp_start_cycle = get_cycles();
      }
      co_await unlock(cur_nbe.id(), rlock, tmp_lock_buf, scratch_lkey);
      if constexpr (enable_trace) {
        tmp_end_cycle = get_cycles();
        trace->insert_lock_total_cycle += tmp_end_cycle - tmp_start_cycle;
      }
    }
    co_return 0;
  }

  Task<> insert(NbEntry new_nbe, const float *vec, bool rotate, uint32_t L,
                bool full_trim, bool ordered, auto is_del_func,
                ConnManager &conn_mgr, size_t cfg_id, RLock *rlock,
                bool with_search_lock, Trace *trace) {
    uint64_t insert_start_cycles, insert_end_cycles;
    if constexpr (enable_trace) {
      insert_start_cycles = get_cycles();
    }
    auto &cfg = conn_mgr.cfgs[cfg_id];
    
    uint64_t tmp_start_cycles, tmp_end_cycles;
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    auto new_item =
        co_await search(vec, L, rotate, is_del_func, conn_mgr, cfg_id,
                        with_search_lock ? rlock : nullptr, trace);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->search_time = tmp_end_cycles - tmp_start_cycles;
    }
    visited[new_nbe] = new_item;
    auto new_vec = cfg.get_vec(new_item); 
    auto new_nbrs = cfg.get_nbrs(new_item);
    auto new_codes = cfg.get_codes(new_item);
    auto dist_func = [&](const auto &a, const auto &b) {
      return l2sqr(cfg.get_vec(visited[a.id]), cfg.get_vec(visited[b.id]),
                   cfg.dim());
    };
    
    std::vector<uint32_t> fwd_idxs(cfg.M());
    std::vector<uint32_t> src_idxs(retset.nret);
    std::iota(src_idxs.begin(), src_idxs.end(), 0);
    auto [n_fwd_core, n_fwd_ret] =
        prune_vamana(retset.data(), src_idxs.data(), src_idxs.size(),
                     fwd_idxs.data(), cfg.M(), dist_func, prune_alpha, true);
    expect_eq(n_fwd_ret, cfg.M());
    if constexpr (enable_trace) {
      trace->n_fwd_core += n_fwd_core;
    }
    for (size_t i = 0; i < cfg.M(); ++i)
      new_nbrs[i] = retset[fwd_idxs[i]].id.with_prune_flag(
          i < n_fwd_core ? PruneFlag::CORE : PruneFlag::REDUNDANT);
    
    for (size_t i = 0; i < cfg.M(); ++i) {
      auto cur = retset[fwd_idxs[i]].id;
      auto [bcode, f_add, f_rescale] = cfg.get_code(new_codes, i);
      std::tie(*f_add, *f_rescale, std::ignore) =
          rabitq_impl(cfg.dim(), cfg.get_vec(visited[cur]), new_vec, bcode);
    }

    
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    auto [conn, addr, rkey] = get_item_helper(new_nbe.id(), conn_mgr, cfg_id);
    co_await conn->write(addr, rkey, new_item, cfg.item_size(), scratch_lkey);
    if constexpr (enable_trace) {
      tmp_end_cycles = get_cycles();
      trace->write_fwd_time = tmp_end_cycles - tmp_start_cycles;
      trace->insert_io_w_total_cycle += tmp_end_cycles - tmp_start_cycles;
    }
    if constexpr (enable_r_print)
      println("==>Rq: {}", retset[retset.nret - 1].dist);

    
    if constexpr (enable_trace) {
      tmp_start_cycles = get_cycles();
    }
    new_nbe.set_prune_flag(PruneFlag::PENDING);
    auto tmp_lock_buf = (uint64_t *)alloc_scratch(sizeof(uint64_t));
    for (size_t i = 0; i < n_fwd_core; ++i)
      if (ordered)
        co_await add_rlink_ordered(new_nbe, new_vec, retset[fwd_idxs[i]],
                                   conn_mgr, cfg_id, rlock, tmp_lock_buf,
                                   trace);
      else
        co_await add_rlink(new_nbe, new_vec, retset[fwd_idxs[i]], full_trim,
                           conn_mgr, cfg_id, rlock, tmp_lock_buf, trace);
    if constexpr (enable_trace) {
      insert_end_cycles = tmp_end_cycles = get_cycles();
      trace->add_rlink_time = tmp_end_cycles - tmp_start_cycles;
      trace->insert_total_cycle += insert_end_cycles - insert_start_cycles;
    }
  }

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  Task<> fix_insert(NbEntry new_nbe, const float *vec, bool rotate, uint32_t L,
                    auto is_del_func, ConnManager &conn_mgr, size_t cfg_id,
                    Trace *trace) {
    auto &cfg = conn_mgr.cfgs[cfg_id];
    auto new_item = alloc_scratch(cfg.item_size());
    auto new_vec = (float *)new_item;
    if (rotate)
      cfg.rotate(vec, new_vec);
    else
      memmove(new_vec, vec, cfg.vec_size());
    auto old_item =
        co_await read(new_nbe.id(), conn_mgr, cfg_id, cfg.item_size());
    auto old_nbrs = cfg.get_nbrs(old_item);
    auto old_codes = cfg.get_codes(old_item);
    auto new_nbrs = cfg.get_nbrs(new_item);
    auto new_codes = cfg.get_codes(new_item);
    bool vec_match = memcpy(old_item, new_item, cfg.vec_size()) == 0;
    bool nbrs_vaild = true;
    for (size_t i = 0; i < cfg.M(); ++i)
      if (!NbEntry::is_valid(old_nbrs[i])) {
        nbrs_vaild = false;
        break;
      }
    
    if (!vec_match || !nbrs_vaild)
      co_return co_await insert(new_nbe, vec, rotate, L, false, false,
                                is_del_func, conn_mgr, cfg_id, nullptr, false,
                                trace);
    
    for (size_t i = 0; i < cfg.M(); ++i) {
      auto cur = old_nbrs[i];
      new_nbrs[i] = cur;
      auto cur_item = read(cur.id(), conn_mgr, cfg_id, cfg.item_size());
      visited[cur] = cur_item;
      auto [bcode, f_add, f_rescale] = cfg.get_code(new_codes, i);
      std::tie(*f_add, *f_rescale, std::ignore) =
          rabitq_impl(cfg.dim(), cfg.get_vec(cur_item), new_vec, bcode);
    }
    
    auto [conn, addr, rkey] = get_item_helper(new_nbe.id(), conn_mgr, cfg_id);
    co_await conn->write(addr + cfg.codes_offset(), rkey, new_codes,
                         cfg.code_size() * cfg.M(), scratch_lkey);
    
    for (size_t i = 0; i < cfg.M(); ++i) {
      auto cur = old_nbrs[i];
      BaseNeighborType cur_nb = {
          l2sqr(new_vec, cfg.get_vec(visited[cur]), cfg.dim()), cur};
      co_await add_rlink(new_nbe, new_vec, cur_nb, true, conn_mgr, cfg_id,
                         nullptr, nullptr, trace);
    }
  }

  Task<bool> check_item(NbEntry check_nbe, ConnManager &conn_mgr,
                        size_t cfg_id) {
    auto &cfg = conn_mgr.cfgs[cfg_id];
    reset();
    auto bitcmp = [](const uint8_t *a, const uint8_t *b, size_t size) {
      size_t cnt = 0;
      for (size_t i = 0; i < size; ++i)
        cnt += __builtin_popcount(a[i] ^ b[i]);
      return cnt;
    };
    auto item = visited[check_nbe] =
        co_await read(check_nbe.id(), conn_mgr, cfg_id, cfg.item_size());
    auto check_vec = cfg.get_vec(item);
    auto check_nbrs = cfg.get_nbrs(item);
    auto check_codes = cfg.get_codes(item);
    auto ref_code = alloc_scratch(cfg.code_size());
    auto [ref_bcode, ref_f_add, ref_f_rescale] = cfg.get_code(ref_code, 0);
    for (size_t i = 0; i < cfg.M(); ++i) {
      auto oth_nbe = check_nbrs[i];
      if (!NbEntry::is_valid(oth_nbe))
        continue;
      auto oth_vec = (float *)co_await read(oth_nbe.id(), conn_mgr, cfg_id,
                                            cfg.vec_size());
      auto [bcode, f_add, f_rescale] = cfg.get_code(check_codes, i);
      std::tie(*ref_f_add, *ref_f_rescale, std::ignore) =
          rabitq_impl(cfg.dim(), oth_vec, check_vec, ref_bcode);
      auto biterr = bitcmp(bcode, ref_bcode, cfg.dim() / 8);
      if (biterr > 0 || std::abs(*f_add - *ref_f_add) > 1e-6 ||
          std::abs(*f_rescale - *ref_f_rescale) > 1e-6)
        co_return true;
    }
    co_return false;
  }

  Task<> gen_code(NbEntry cur_nbe, ConnManager &conn_mgr, size_t cfg_id) {
    auto &cfg = conn_mgr.cfgs[cfg_id];
    reset();
    auto item = co_await read(cur_nbe.id(), conn_mgr, cfg_id,
                              cfg.raw_item_size(), cfg.item_size());
    auto cur_vec = cfg.get_vec(item);
    auto cur_nbrs = cfg.get_nbrs(item);
    auto cur_codes = cfg.get_codes(item);
    for (size_t i = 0; i < cfg.M(); ++i) {
      auto oth_nbe = cur_nbrs[i];
      if (!NbEntry::is_valid(oth_nbe))
        continue;
      auto oth_vec = (float *)co_await read(oth_nbe.id(), conn_mgr, cfg_id,
                                            cfg.vec_size());
      auto [bcode, f_add, f_rescale] = cfg.get_code(cur_codes, i);
      std::tie(*f_add, *f_rescale, std::ignore) =
          rabitq_impl(cfg.dim(), oth_vec, cur_vec, bcode);
    }
    auto [conn, addr, rkey] = get_item_helper(cur_nbe.id(), conn_mgr, cfg_id);
    co_await conn->write(addr + cfg.codes_offset(), rkey, cur_codes,
                         cfg.code_size() * cfg.M(), scratch_lkey);
    co_return;
  }
};
