#include "algo.h"
#include "conn_mgr.h"
#include "define.h"
#include "sock_barrier.h"

#include <chrono>
#include <lcoro/rdma_worker.h>
#include <myutils/func.h>
#include <myutils/io.h>
#include <myutils/random.h>

#include <argparse/argparse.hpp>
#include <thread>

int main(int argc, char **argv) {
  size_t start_nvecs;
  size_t end_nvecs;
  size_t step_nvecs;
  std::string base_path;
  std::string query_path;
  std::string gt_path;
  std::string global_barrier_addr;
  
  std::string graph_cfg;
  std::string lock_cfgs;
  size_t client_rank;
  size_t nclients;
  int max_threads = 16;
  int max_coros = 2;
  size_t L_insert = 200;
  size_t sel = 0;
  size_t hard_interval = 0;
  size_t gt_stride = 1'000'000;
  
  constexpr std::tuple<bool, bool, bool, bool> sel2bools[] = {
      {true, true, false, true},   {false, true, false, true},
      {false, false, false, true}, {false, false, false, false},
      {true, false, false, true},
  };
  const size_t cfg_id = 0;

  std::string dev_name;
  int rdma_dev_port = 1;
  int rdma_gid_index = 1;
  std::string arg_cpu_list =
      std::format("0-{}", std::thread::hardware_concurrency() - 1);
  {
    argparse::ArgumentParser program("client");
    program.add_argument("start-nvecs").required().store_into(start_nvecs);
    program.add_argument("end-nvecs").required().store_into(end_nvecs);
    program.add_argument("step-nvecs").required().store_into(step_nvecs);
    program.add_argument("base-path").required().store_into(base_path);
    program.add_argument("query-path").required().store_into(query_path);
    program.add_argument("gt-path").required().store_into(gt_path);
    program.add_argument("global-barrier-addr")
        .required()
        .store_into(global_barrier_addr);
    program.add_argument("graph-cfg").required().store_into(graph_cfg);
    program.add_argument("lock-cfgs").required().store_into(lock_cfgs);
    program.add_argument("client-rank").required().store_into(client_rank);
    program.add_argument("nclients").required().store_into(nclients);
    program.add_argument("-t", "--max-threads")
        .default_value(max_threads)
        .store_into(max_threads);
    program.add_argument("-c", "--max-coros")
        .default_value(max_coros)
        .store_into(max_coros);
    program.add_argument("-b", "--bind-cpu-list")
        .default_value(arg_cpu_list)
        .store_into(arg_cpu_list);
    program.add_argument("-l", "--L-insert")
        .default_value(L_insert)
        .store_into(L_insert);
    program.add_argument("-s", "--sel").default_value(sel).store_into(sel);
    program.add_argument("-H", "--hard-interval")
        .default_value(hard_interval)
        .store_into(hard_interval);
    program.add_argument("-g", "--gt-stride")
        .default_value(gt_stride)
        .store_into(gt_stride);
    program.add_argument("-d", "--dev-name")
        .default_value(dev_name)
        .store_into(dev_name);
    program.add_argument("--dev-port")
        .default_value(rdma_dev_port)
        .store_into(rdma_dev_port);
    program.add_argument("--gid-index")
        .default_value(rdma_gid_index)
        .store_into(rdma_gid_index);

    try {
      program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
      std::cerr << err.what() << std::endl;
      std::cerr << program;
      return 1;
    }
  }
  const auto [full_trim, ordered, with_search_lock, with_insert_lock] =
      sel2bools[sel];
  expect_lt(client_rank, nclients);

  eprintln("barrier_addr: {}, client_rank: {}, nclients: {}, bind_cpu_list: {}",
           global_barrier_addr, client_rank, nclients, arg_cpu_list);

  auto cpus = parse_cpu_list(arg_cpu_list);
  expect(std::all_of(cpus.begin(), cpus.end(), [](int cpu) {
    return cpu < std::thread::hardware_concurrency();
  }));
  expect_lt(max_threads, cpus.size());

  rdma::Device dev(dev_name, rdma_gid_index, rdma_dev_port);
  ConnManager conn_mgr(dev, max_threads, max_coros);
  auto &cfg = conn_mgr.cfgs.emplace_back();
  cfg.init_from_str(graph_cfg);
  eprintln("graph cfg: {}", cfg);
  conn_mgr.connect_mem_servers();
  eprintln("mem servers connected");

  std::vector<RLock> rlocks(max_threads);
  std::vector<std::string> lock_cfgs_vec;
  for (auto lock_cfg : std::ranges::split_view(lock_cfgs, ','))
    lock_cfgs_vec.emplace_back(lock_cfg.begin(), lock_cfg.end());
  for (size_t tid = 0; tid < max_threads; ++tid)
    rlocks[tid].init(conn_mgr, tid, lock_cfgs_vec);
  rlocks[0].sync_reset(conn_mgr.scratch_mr->addr, conn_mgr.scratch_mr->lkey);
  eprintln("rlocks initialized");

  std::vector<CoroContext> c_ctxs(max_threads * max_coros);
  for (size_t tid = 0; tid < max_threads; ++tid)
    for (size_t cid = 0; cid < max_coros; ++cid) {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      c_ctx.tid = tid;
      c_ctx.cid = cid;
      c_ctx.scratch = conn_mgr.get_scratch(tid, cid);
      c_ctx.scratch_size = conn_mgr.scratch_size;
      c_ctx.scratch_lkey = conn_mgr.scratch_mr->lkey;
    }
  SockBarrier g_barrier(global_barrier_addr);

  const size_t global_ninsert = end_nvecs - start_nvecs;
  const size_t nbatches = upper_div(global_ninsert, step_nvecs);
  const size_t max_client_batch_size = upper_div(step_nvecs, nclients);
  const size_t dim = cfg.dim();
  MMap base_mmap(base_path);
  const auto xb = base_mmap.ptr<float>();
  auto vec = [&](size_t i) { return xb + i * dim; };
  MMap query_mmap(query_path);
  const auto xq = query_mmap.ptr<float>();
  auto query = [&](size_t i) { return xq + i * dim; };
  const size_t nq_real = query_mmap.size_ / sizeof(float) / dim;
  MMap gt_mmap(gt_path);
  const auto gt_base = gt_mmap.ptr<uint32_t>();
  const size_t ngt = 100;
  const size_t gt_bs = gt_mmap.size_ / sizeof(uint32_t) / nq_real / ngt;
  expect_eq(nq_real * ngt * gt_bs * sizeof(uint32_t), gt_mmap.size_);
  auto gt = [&](size_t cur_nvecs) -> const uint32_t * {
    if (gt_base == nullptr || cur_nvecs % gt_stride != 0 ||
        cur_nvecs / gt_stride > gt_bs)
      return nullptr;
    return gt_base + (cur_nvecs / gt_stride - 1) * nq_real * ngt;
  };
  auto access_xb = [&](size_t from, size_t to) {
    float tmp = 0;
    for (size_t i = from; i < to; ++i)
      tmp += vec(i)[0];
    
    asm volatile("" : "+m"(tmp));
    return tmp;
  };

  const size_t nq_max = 2000;
  const size_t nq_min = 100;
  const size_t nq_cap = std::min(nq_max, nq_real);
  const size_t nq_floor = std::min(nq_min, nq_cap);
  auto get_nq = [&](size_t nth, size_t L) {
#if DEBUG
    size_t est_qps = 100000 / L * nth;
    return std::clamp(upper_align(est_qps, 100), nq_floor, nq_cap);
#else
    return nq_cap;
#endif
  };
  auto get_raw_nq = [&](size_t nth, size_t L) {
#if DEBUG
    size_t est_qps = 20000 / L * nth;
    return std::clamp(upper_align(est_qps, 100), nq_floor, nq_cap);
#else
    return nq_cap;
#endif
  };

  eprintln("wait for barrier");
  g_barrier.wait();

  constexpr size_t Ls[] = {20,  30,  40,  60,  80,  100, 120, 140,
                           160, 180, 200, 240, 280, 320, 360, 400};
  constexpr size_t recall_ats[] = {1, 5, 10, 20, 50, 100};
  constexpr size_t K = 100;
  std::vector<uint32_t> res(nq_max * K);
  std::atomic_size_t q_alloc = 0;
  std::atomic_size_t i_alloc = 0;
  std::vector<Trace> traces(max_threads);
  auto reset_traces = [&]() {
    for (auto &trace : traces)
      trace.reset();
  };
  auto sum_traces = [&]() {
    Trace sum;
    sum.reset();
    uint64_t *p_sum = reinterpret_cast<uint64_t *>(&sum);
    for (auto &trace : traces) {
      uint64_t *p_trace = reinterpret_cast<uint64_t *>(&trace);
      for (size_t i = 0; i < sizeof(Trace) / sizeof(uint64_t); ++i)
        p_sum[i] += p_trace[i];
    }
    return sum;
  };
  std::chrono::high_resolution_clock::time_point ts, te;

  std::vector<uint64_t> insert_times(max_client_batch_size);
  std::vector<uint64_t> search_times(std::max(max_client_batch_size, nq_max));

  Barrier b(max_threads * max_coros);
  std::unique_ptr<Barrier> b_dyn;

  println("#*# start test #*#");
  auto thmain = [&](size_t tid) {
    bind_core(cpus[tid]);
    auto &cli = conn_mgr.clients[tid];
    auto &rlock = rlocks[tid];
    auto do_raw_search = [&](size_t cid, const std::string &prefix,
                             size_t nthreads, bool with_lock,
                             const uint32_t *gt, auto is_del_func,
                             Barrier *b_dyn) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      if (b_dyn)
        co_await *b_dyn;
      for (size_t L : Ls) {
        const size_t nq = get_raw_nq(nthreads, L);
        c_ctx.retset.set_L(L);
        if (tid == 0 && cid == 0) {
          q_alloc = 0;
          std::fill(res.begin(), res.end(),
                    std::numeric_limits<uint32_t>::max());
        }
        for (size_t i = 0; i < 20; ++i)
          co_await c_ctx.raw_search(query(randu64(nq)), L, true, is_del_func,
                                    conn_mgr, cfg_id,
                                    with_lock ? &rlock : nullptr, &traces[tid]);
        if (b_dyn)
          co_await *b_dyn;
        if (tid == 0 && cid == 0) {
          reset_traces();
          ts = std::chrono::high_resolution_clock::now();
        }
        if (b_dyn)
          co_await *b_dyn;
        while (true) {
          auto qid = q_alloc.fetch_add(1);
          if (qid >= nq)
            break;
          auto single_ts = std::chrono::high_resolution_clock::now();
          co_await c_ctx.raw_search(query(qid), L, true, is_del_func, conn_mgr,
                                    cfg_id, with_lock ? &rlock : nullptr,
                                    &traces[tid]);
          auto single_te = std::chrono::high_resolution_clock::now();
          auto single_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               single_te - single_ts)
                               .count();
          search_times[qid] = single_ns;
          for (size_t i = 0; i < K; ++i)
            res[qid * K + i] = i < std::min(K, L)
                                   ? c_ctx.retset[i].id.id()
                                   : std::numeric_limits<uint32_t>::max();
        }
        if (b_dyn)
          co_await *b_dyn;
        if (tid == 0 && cid == 0) {
          te = std::chrono::high_resolution_clock::now();
          auto dur_s =
              std::chrono::duration_cast<std::chrono::duration<double>>(te - ts)
                  .count();
          auto qps = nq * 1.0 / dur_s;
          std::sort(search_times.data(), search_times.data() + nq);
          uint64_t total_time = std::accumulate(search_times.data(),
                                                search_times.data() + nq, 0ull);
          auto avg_search_ms = 1. * total_time / nq / 1000000;
          auto p50_search_ms = 1. * search_times[nq / 2] / 1000000;
          auto p90_search_ms = 1. * search_times[nq * 9 / 10] / 1000000;
          auto p99_search_ms = 1. * search_times[nq * 99 / 100] / 1000000;
          auto trace = sum_traces();
          print("{}$L:{},qps:{},avg_ms:{},p50_ms:{},p90_ms:{},p99_ms:{},{}",
                prefix, L, qps, avg_search_ms, p50_search_ms, p90_search_ms,
                p99_search_ms, trace.search_trace_str());
          if (gt) {
            auto recall = calc_recall(nq, ngt, gt, K, res.data(),
                                      std::size(recall_ats), recall_ats);
            println(",{}", recall);
          } else
            println("");
        }
      }
      co_return 0;
    };
    auto do_raw_insert =
        [&](size_t cid, const std::string &prefix, size_t from, size_t to,
            size_t L, auto is_del_func, bool full_trim, bool with_search_lock,
            bool with_insert_lock, Barrier *b_dyn) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      if (to < from)
        to = from;
      const size_t nins = to - from;
      if (b_dyn)
        co_await *b_dyn;
      if (tid == 0 && cid == 0) {
        i_alloc = from;
        reset_traces();
        ts = std::chrono::high_resolution_clock::now();
      }
      if (b_dyn)
        co_await *b_dyn;
      while (true) {
        auto id = i_alloc.fetch_add(1);
        if (id >= to)
          break;
        auto single_ts = std::chrono::high_resolution_clock::now();
        co_await c_ctx.raw_insert(NbEntry(id), vec(id), L, true, full_trim,
                                  is_del_func, conn_mgr, cfg_id,
                                  with_insert_lock ? &rlock : nullptr,
                                  with_search_lock, &traces[tid]);
        auto single_te = std::chrono::high_resolution_clock::now();
        auto single_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             single_te - single_ts)
                             .count();
        insert_times[id - from] = single_ns;
        search_times[id - from] = traces[tid].search_time;
      }
      if (b_dyn)
        co_await *b_dyn;
      if (tid == 0 && cid == 0) {
        te = std::chrono::high_resolution_clock::now();
        auto dur_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(te - ts)
                .count();
        double avg_insert_ms = 0, p50_insert_ms = 0, p90_insert_ms = 0,
               p99_insert_ms = 0;
        double avg_search_ms = 0, p50_search_ms = 0, p90_search_ms = 0,
               p99_search_ms = 0;
        double ips = 0;
        if (nins > 0 && dur_s > 0) {
          std::sort(insert_times.data(), insert_times.data() + nins);
          uint64_t total_insert_time = std::accumulate(
              insert_times.data(), insert_times.data() + nins, 0ull);
          avg_insert_ms = 1. * total_insert_time / nins / 1000000;
          p50_insert_ms = 1. * insert_times[nins / 2] / 1000000;
          p90_insert_ms = 1. * insert_times[nins * 9 / 10] / 1000000;
          p99_insert_ms = 1. * insert_times[nins * 99 / 100] / 1000000;
          std::sort(search_times.data(), search_times.data() + nins);
          uint64_t total_search_time = std::accumulate(
              search_times.data(), search_times.data() + nins, 0ull);
          avg_search_ms = 1. * total_search_time / nins / 1000000;
          p50_search_ms = 1. * search_times[nins / 2] / 1000000;
          p90_search_ms = 1. * search_times[nins * 9 / 10] / 1000000;
          p99_search_ms = 1. * search_times[nins * 99 / 100] / 1000000;
          ips = nins * 1.0 / dur_s;
        }
        auto trace = sum_traces();
        println("{}$ips:{},avg_insert_ms:{},p50_insert_ms:{},p90_insert_ms:{},"
                "p99_insert_ms:{},avg_search_ms:{},p50_search_ms:{},p90_search_"
                "ms:{},p99_search_ms:{},{},{}",
                prefix, ips, avg_insert_ms, p50_insert_ms, p90_insert_ms,
                p99_insert_ms, avg_search_ms, p50_search_ms, p90_search_ms,
                p99_search_ms, trace.search_trace_str(),
                trace.insert_trace_str());
      }
      co_return 0;
    };
    auto do_search = [&](size_t cid, const std::string &prefix, size_t nthreads,
                         bool with_lock, const uint32_t *gt, auto is_del_func,
                         Barrier *b_dyn) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      if (b_dyn)
        co_await *b_dyn;
      for (size_t L : Ls) {
        const size_t nq = get_nq(nthreads, L);
        c_ctx.retset.set_L(L);
        if (tid == 0 && cid == 0) {
          q_alloc = 0;
          std::fill(res.begin(), res.end(),
                    std::numeric_limits<uint32_t>::max());
        }
        for (size_t i = 0; i < 20; ++i)
          co_await c_ctx.search(query(randu64(nq)), L, true, is_del_func,
                                conn_mgr, cfg_id, with_lock ? &rlock : nullptr,
                                &traces[tid]);
        if (b_dyn)
          co_await *b_dyn;
        if (tid == 0 && cid == 0) {
          reset_traces();
          ts = std::chrono::high_resolution_clock::now();
        }
        if (b_dyn)
          co_await *b_dyn;
        while (true) {
          auto qid = q_alloc.fetch_add(1);
          if (qid >= nq)
            break;
          auto single_ts = std::chrono::high_resolution_clock::now();
          co_await c_ctx.search(query(qid), L, true, is_del_func, conn_mgr,
                                cfg_id, with_lock ? &rlock : nullptr,
                                &traces[tid]);
          auto single_te = std::chrono::high_resolution_clock::now();
          auto single_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               single_te - single_ts)
                               .count();
          search_times[qid] = single_ns;
          for (size_t i = 0; i < K; ++i)
            res[qid * K + i] = i < std::min(K, L)
                                   ? c_ctx.retset[i].id.id()
                                   : std::numeric_limits<uint32_t>::max();
        }
        if (b_dyn)
          co_await *b_dyn;
        if (tid == 0 && cid == 0) {
          te = std::chrono::high_resolution_clock::now();
          auto dur_s =
              std::chrono::duration_cast<std::chrono::duration<double>>(te - ts)
                  .count();
          auto qps = nq * 1.0 / dur_s;
          std::sort(search_times.data(), search_times.data() + nq);
          uint64_t total_time = std::accumulate(search_times.data(),
                                                search_times.data() + nq, 0ull);
          auto avg_search_ms = 1. * total_time / nq / 1000000;
          auto p50_search_ms = 1. * search_times[nq / 2] / 1000000;
          auto p90_search_ms = 1. * search_times[nq * 9 / 10] / 1000000;
          auto p99_search_ms = 1. * search_times[nq * 99 / 100] / 1000000;
          auto trace = sum_traces();
          print("{}$L:{},qps:{},avg_ms:{},p50_ms:{},p90_ms:{},p99_ms:{},{}",
                prefix, L, qps, avg_search_ms, p50_search_ms, p90_search_ms,
                p99_search_ms, trace.search_trace_str());
          if (gt) {
            auto recall = calc_recall(nq, ngt, gt, K, res.data(),
                                      std::size(recall_ats), recall_ats);
            println(",{}", recall);
          } else
            println("");
        }
      }
      co_return 0;
    };
    auto do_insert = [&](size_t cid, const std::string &prefix, size_t from,
                         size_t to, size_t L, auto is_del_func, bool full_trim,
                         bool ordered, bool with_search_lock,
                         bool with_insert_lock, Barrier *b_dyn) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      if (to < from)
        to = from;
      const size_t nins = to - from;
      if (b_dyn)
        co_await *b_dyn;
      if (tid == 0 && cid == 0) {
        i_alloc = from;
        reset_traces();
        ts = std::chrono::high_resolution_clock::now();
      }
      if (b_dyn)
        co_await *b_dyn;
      while (true) {
        auto id = i_alloc.fetch_add(1);
        if (id >= to)
          break;
        auto single_ts = std::chrono::high_resolution_clock::now();
        co_await c_ctx.insert(NbEntry(id), vec(id), true, L, full_trim, ordered,
                              is_del_func, conn_mgr, cfg_id,
                              with_insert_lock ? &rlock : nullptr,
                              with_search_lock, &traces[tid]);
        auto single_te = std::chrono::high_resolution_clock::now();
        auto single_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             single_te - single_ts)
                             .count();
        insert_times[id - from] = single_ns;
        search_times[id - from] = traces[tid].search_time;
      }
      if (b_dyn)
        co_await *b_dyn;
      if (tid == 0 && cid == 0) {
        te = std::chrono::high_resolution_clock::now();
        auto dur_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(te - ts)
                .count();
        double avg_insert_ms = 0, p50_insert_ms = 0, p90_insert_ms = 0,
               p99_insert_ms = 0;
        double avg_search_ms = 0, p50_search_ms = 0, p90_search_ms = 0,
               p99_search_ms = 0;
        double ips = 0;
        if (nins > 0 && dur_s > 0) {
          std::sort(insert_times.data(), insert_times.data() + nins);
          uint64_t total_insert_time = std::accumulate(
              insert_times.data(), insert_times.data() + nins, 0ull);
          avg_insert_ms = 1. * total_insert_time / nins / 1000000;
          p50_insert_ms = 1. * insert_times[nins / 2] / 1000000;
          p90_insert_ms = 1. * insert_times[nins * 9 / 10] / 1000000;
          p99_insert_ms = 1. * insert_times[nins * 99 / 100] / 1000000;
          std::sort(search_times.data(), search_times.data() + nins);
          uint64_t total_search_time = std::accumulate(
              search_times.data(), search_times.data() + nins, 0ull);
          avg_search_ms = 1. * total_search_time / nins / 1000000;
          p50_search_ms = 1. * search_times[nins / 2] / 1000000;
          p90_search_ms = 1. * search_times[nins * 9 / 10] / 1000000;
          p99_search_ms = 1. * search_times[nins * 99 / 100] / 1000000;
          ips = nins * 1.0 / dur_s;
        }
        auto trace = sum_traces();
        println("{}$ips:{},avg_insert_ms:{},p50_insert_ms:{},p90_insert_ms:{},"
                "p99_insert_ms:{},avg_search_ms:{},p50_search_ms:{},p90_search_"
                "ms:{},p99_search_ms:{},{},{}",
                prefix, ips, avg_insert_ms, p50_insert_ms, p90_insert_ms,
                p99_insert_ms, avg_search_ms, p50_search_ms, p90_search_ms,
                p99_search_ms, trace.search_trace_str(),
                trace.insert_trace_str());
      }
      co_return 0;
    };
    auto do_easy_search = [&](size_t cid, size_t cur_nvecs) -> Task<int> {
      if (tid == 0 && cid == 0)
        g_barrier.wait();
      co_await b;
      if (tid == 0 && cid == 0)
        co_await do_search(
            cid, std::format("search,{},1t1c", cur_nvecs), 1, false,
            gt(cur_nvecs), [](auto &&) { return false; }, nullptr);
      co_await b;
      if (tid == 0 && cid == 0)
        g_barrier.wait();
      co_await b;
      co_await do_search(
          cid,
          std::format("search,{},{}t{}c", cur_nvecs, max_threads, max_coros),
          max_threads, false, gt(cur_nvecs), [](auto &&) { return false; }, &b);
      co_return 0;
    };
    auto do_easy_raw_search = [&](size_t cid, size_t cur_nvecs) -> Task<int> {
      if (tid == 0 && cid == 0)
        g_barrier.wait();
      co_await b;
      if (tid == 0 && cid == 0)
        co_await do_raw_search(
            cid, std::format("search,{},1t1c", cur_nvecs), 1, false,
            gt(cur_nvecs), [](auto &&) { return false; }, nullptr);
      co_await b;
      if (tid == 0 && cid == 0)
        g_barrier.wait();
      co_await b;
      co_await do_raw_search(
          cid,
          std::format("search,{},{}t{}c", cur_nvecs, max_threads, max_coros),
          max_threads, false, gt(cur_nvecs), [](auto &&) { return false; }, &b);
      co_return 0;
    };
    auto do_hard_search = [&](size_t cid, size_t cur_nvecs) -> Task<int> {
      for (size_t nth : {1, 2, 4, 8, 16})
        for (size_t nco : {1, 2}) {
          if (nth > max_threads || nco > max_coros)
            continue;
          if (tid == 0 && cid == 0) {
            g_barrier.wait();
            b_dyn = std::make_unique<Barrier>(nth * nco);
          }
          co_await b;
          if (tid < nth && cid < nco)
            co_await do_search(
                cid, std::format("search,{},{}t{}c", cur_nvecs, nth, nco), nth,
                false, gt(cur_nvecs), [](auto &&) { return false; },
                b_dyn.get());
        }
      co_return 0;
    };
    auto do_hard_raw_search = [&](size_t cid, size_t cur_nvecs) -> Task<int> {
      for (size_t nth : {1, 2, 4, 8, 16})
        for (size_t nco : {1, 2}) {
          if (nth > max_threads || nco > max_coros)
            continue;
          if (tid == 0 && cid == 0) {
            g_barrier.wait();
            b_dyn = std::make_unique<Barrier>(nth * nco);
          }
          co_await b;
          if (tid < nth && cid < nco)
            co_await do_raw_search(
                cid, std::format("raw_search,{},{}t{}c", cur_nvecs, nth, nco),
                nth, false, gt(cur_nvecs), [](auto &&) { return false; },
                b_dyn.get());
        }
      co_return 0;
    };
    auto comain = [&](size_t cid) -> Task<int> {
      for (size_t i = 0; i < nbatches; ++i) {
        size_t cur_nvecs = start_nvecs + i * step_nvecs;
        size_t cur_to = std::min(cur_nvecs + step_nvecs, end_nvecs);
        size_t real_batch_size = std::min(step_nvecs, cur_to - cur_nvecs);
        size_t client_batch_size = upper_div(real_batch_size, nclients);
        size_t client_from = cur_nvecs + client_batch_size * client_rank;
        size_t client_to = std::min(client_from + client_batch_size, cur_to);
        const size_t assigned =
            client_to > client_from ? (client_to - client_from) : 0;
        size_t nins_delta =
            std::min<size_t>({assigned, client_batch_size / 2, 1000ul});
        if ((hard_interval != 0) &&
            (cur_nvecs % hard_interval == 0 || cur_nvecs == start_nvecs))
          co_await do_hard_search(cid, cur_nvecs);
        else
          co_await do_easy_search(cid, cur_nvecs);

        if (tid == 0 && cid == 0)
          g_barrier.wait();
        access_xb(client_from, client_to);
        co_await b;
        if (tid == 0 && cid == 0)
          co_await do_insert(
              cid, std::format("insert,{},{},1t1c", cur_nvecs, cur_to),
              client_from, client_from + nins_delta, L_insert,
              [](auto &&) { return false; }, full_trim, ordered,
              with_search_lock, with_insert_lock, nullptr);
        if (tid == 0 && cid == 0)
          g_barrier.wait();
        co_await b;
        co_await do_insert(
            cid,
            std::format("insert,{},{},{}t{}c", cur_nvecs, cur_to, max_threads,
                        max_coros),
            client_from + nins_delta, client_to, L_insert,
            [](auto &&) { return false; }, full_trim, ordered, with_search_lock,
            with_insert_lock, &b);
      }
      if (hard_interval != 0)
        co_await do_hard_search(cid, upper_align(end_nvecs, gt_stride));
      else
        co_await do_easy_search(cid, upper_align(end_nvecs, gt_stride));
      co_return 0;
    };
    EventLoop el(*cli);
    std::vector<Task<int>> tasks;
    for (size_t cid = 0; cid < max_coros; ++cid) {
      tasks.push_back(comain(cid));
    }
    el.start(gather(tasks));
  };
  std::vector<std::thread> threads;
  for (size_t tid = 0; tid < max_threads; ++tid)
    threads.emplace_back(thmain, tid);
  for (auto &thread : threads)
    thread.join();
  g_barrier.wait();

  return 0;
}