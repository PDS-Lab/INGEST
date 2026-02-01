#include "algo.h"
#include "conn_mgr.h"
#include "define.h"
#include "sock_barrier.h"

#include <atomic>
#include <chrono>
#include <lcoro/rdma_worker.h>
#include <myutils/func.h>
#include <myutils/io.h>
#include <myutils/random.h>

#include <argparse/argparse.hpp>
#include <thread>

int main(int argc, char **argv) {
  size_t start_nvecs;
  std::string base_path;
  std::string query_path;
  std::string gt_path;
  std::string global_barrier_addr;
  std::string meta_cfg;
  
  std::string graph_cfg;
  std::string lock_cfgs;
  size_t client_rank;
  size_t nclients;
  int insert_threads = 8;
  int search_threads = 8;
  int max_coros = 2;
  size_t L_insert = 200;
  size_t gt_stride = 1'000'000;
  int sel = 3; 
  
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
    program.add_argument("base-path").required().store_into(base_path);
    program.add_argument("query-path").required().store_into(query_path);
    program.add_argument("gt-path").required().store_into(gt_path);
    program.add_argument("global-barrier-addr")
        .required()
        .store_into(global_barrier_addr);
    program.add_argument("meta-cfg").required().store_into(meta_cfg);
    program.add_argument("graph-cfg").required().store_into(graph_cfg);
    program.add_argument("lock-cfgs").required().store_into(lock_cfgs);
    program.add_argument("client-rank").required().store_into(client_rank);
    program.add_argument("nclients").required().store_into(nclients);
    program.add_argument("insert-threads")
        .required()
        .store_into(insert_threads);
    program.add_argument("search-threads")
        .required()
        .store_into(search_threads);
    program.add_argument("-c", "--max-coros")
        .default_value(max_coros)
        .store_into(max_coros);
    program.add_argument("-b", "--bind-cpu-list")
        .default_value(arg_cpu_list)
        .store_into(arg_cpu_list);
    program.add_argument("-l", "--L-insert")
        .default_value(L_insert)
        .store_into(L_insert);
    program.add_argument("-g", "--gt-stride")
        .default_value(gt_stride)
        .store_into(gt_stride);
    program.add_argument("-s", "--sel").default_value(sel).store_into(sel);
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
  expect_lt(client_rank, nclients);

  eprintln("barrier_addr: {}, client_rank: {}, nclients: {}, bind_cpu_list: {}",
           global_barrier_addr, client_rank, nclients, arg_cpu_list);

  const size_t max_threads = insert_threads + search_threads;

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

  std::vector<rdma::Conn> meta_conns(max_threads);
  auto meta_cfg_pos = meta_cfg.find_last_of(':');
  auto meta_server_addr = meta_cfg.substr(0, meta_cfg_pos);
  auto meta_mem_id = std::stoi(meta_cfg.substr(meta_cfg_pos + 1));
  for (size_t tid = 0; tid < max_threads; ++tid)
    meta_conns[tid] =
        std::move(conn_mgr.get_or_create_conn(meta_server_addr, tid));
  auto meta_rmr = meta_conns[0]->query_rmr_sync(meta_mem_id);
  expect_eq(meta_rmr.uid, meta_mem_id);
  eprintln("meta servers connected");
  if (client_rank == 0)
    meta_conns[0]->sync_write(meta_rmr.addr, meta_rmr.rkey, &start_nvecs,
                              sizeof(uint64_t));

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
  const size_t max_insert = 2'000'000;
  const size_t mini_batch = 500;
  access_xb(start_nvecs, start_nvecs + max_insert);

  const size_t nq = std::min(nq_real, 2000ul);

  eprintln("wait for barrier");
  g_barrier.wait();

  constexpr size_t Ls[] = {20,  30,  40,  60,  80,  100, 120, 140,
                           160, 180, 200, 240, 280, 320, 360, 400};
  constexpr size_t recall_ats[] = {1, 5, 10, 20, 50, 100};
  constexpr size_t K = 100;
  std::vector<uint32_t> res(nq * K);
  std::vector<Trace> traces(max_threads);
  for (auto &trace : traces)
    trace.reset();
  std::chrono::high_resolution_clock::time_point ts, te;

  std::vector<uint64_t> s_times(nq);
  
  
  std::vector<std::vector<uint64_t>> is_times_by_tid(max_threads);
  std::vector<std::vector<uint64_t>> ii_times_by_tid(max_threads);
  std::atomic_size_t q_alloc = 0;

  Barrier b(max_threads * max_coros);
  std::unique_ptr<Barrier> b_dyn =
      std::make_unique<Barrier>(search_threads * max_coros);
  auto is_del_func = [&](auto &&) { return false; };
  std::atomic_bool stop_flag = false;

  std::atomic_flag i_lock{ATOMIC_FLAG_INIT};
  uint64_t i_alloc = 0;
  uint64_t i_to = 0;

  println("#*# start test #*#");
  auto thmain = [&](size_t tid) {
    auto lock = [&]() -> Task<> {
      while (i_lock.test_and_set(std::memory_order_acquire))
        co_await BaseEventLoop::get_current()->yield();
      co_return;
    };
    auto unlock = [&]() { i_lock.clear(std::memory_order_release); };
    auto alloc_next_id = [&](uint64_t *scratch,
                             CoroContext &c_ctx) -> Task<uint64_t> {
      co_await lock();
      if (i_alloc >= i_to) {
        co_await meta_conns[tid]->faa(meta_rmr.addr, meta_rmr.rkey, scratch,
                                      c_ctx.scratch_lkey, mini_batch);
        const uint64_t new_from = *scratch;
        i_alloc = new_from;
        i_to = new_from + mini_batch;
      }
      auto got = i_alloc++;
      unlock();
      co_return got;
    };

    bind_core(cpus[tid]);
    auto &cli = conn_mgr.clients[tid];
    auto &rlock = rlocks[tid];
    auto do_raw_search = [&](size_t cid, const std::string &prefix,
                             bool with_lock, const uint32_t *gt,
                             Barrier *b_dyn) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      if (b_dyn)
        co_await *b_dyn;
      for (size_t L : Ls) {
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
          for (size_t i = 0; i < search_threads; ++i)
            traces[i].reset();
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
          s_times[qid] = single_ns;
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
          std::sort(s_times.data(), s_times.data() + nq);
          uint64_t total_time =
              std::accumulate(s_times.data(), s_times.data() + nq, 0ull);
          auto avg_search_ms = 1. * total_time / nq / 1000000;
          auto p50_search_ms = 1. * s_times[nq / 2] / 1000000;
          auto p90_search_ms = 1. * s_times[nq * 9 / 10] / 1000000;
          auto p99_search_ms = 1. * s_times[nq * 99 / 100] / 1000000;
          Trace trace;
          trace.reset();
          uint64_t *p_sum = reinterpret_cast<uint64_t *>(&trace);
          for (size_t i = 0; i < search_threads; ++i) {
            uint64_t *p_trace = reinterpret_cast<uint64_t *>(&traces[i]);
            for (size_t j = 0; j < sizeof(Trace) / sizeof(uint64_t); ++j)
              p_sum[j] += p_trace[j];
          }
          print(
              "{}$L:{},qps:{},avg_ms:{},p50_ms:{},p90_ms:{},p99_ms:{},nq:{},{}",
              prefix, L, qps, avg_search_ms, p50_search_ms, p90_search_ms,
              p99_search_ms, nq, trace.search_trace_str());
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
    auto do_raw_insert = [&](size_t cid, const std::string &prefix, size_t L,
                             bool full_trim, bool with_search_lock,
                             bool with_insert_lock) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      uint64_t *scratch = (uint64_t *)conn_mgr.get_scratch(tid, cid);
      size_t id;
      while (!stop_flag) {
        auto id = co_await alloc_next_id(scratch, c_ctx);
        if (stop_flag || id >= start_nvecs + max_insert)
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
        ii_times_by_tid[tid].emplace_back(single_ns);
        is_times_by_tid[tid].emplace_back(traces[tid].search_time);
      }
      co_return 0;
    };
    auto do_search = [&](size_t cid, const std::string &prefix, bool with_lock,
                         const uint32_t *gt, Barrier *b_dyn) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      if (b_dyn)
        co_await *b_dyn;
      for (size_t L : Ls) {
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
          for (size_t i = 0; i < search_threads; ++i)
            traces[i].reset();
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
          s_times[qid] = single_ns;
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
          std::sort(s_times.data(), s_times.data() + nq);
          uint64_t total_time =
              std::accumulate(s_times.data(), s_times.data() + nq, 0ull);
          auto avg_search_ms = 1. * total_time / nq / 1000000;
          auto p50_search_ms = 1. * s_times[nq / 2] / 1000000;
          auto p90_search_ms = 1. * s_times[nq * 9 / 10] / 1000000;
          auto p99_search_ms = 1. * s_times[nq * 99 / 100] / 1000000;
          Trace trace;
          trace.reset();
          uint64_t *p_sum = reinterpret_cast<uint64_t *>(&trace);
          for (size_t i = 0; i < search_threads; ++i) {
            uint64_t *p_trace = reinterpret_cast<uint64_t *>(&traces[i]);
            for (size_t j = 0; j < sizeof(Trace) / sizeof(uint64_t); ++j)
              p_sum[j] += p_trace[j];
          }
          print(
              "{}$L:{},qps:{},avg_ms:{},p50_ms:{},p90_ms:{},p99_ms:{},nq:{},{}",
              prefix, L, qps, avg_search_ms, p50_search_ms, p90_search_ms,
              p99_search_ms, nq, trace.search_trace_str());
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
    auto do_insert = [&](size_t cid, const std::string &prefix, size_t L,
                         bool full_trim, bool ordered, bool with_search_lock,
                         bool with_insert_lock) -> Task<int> {
      auto &c_ctx = c_ctxs[tid * max_coros + cid];
      uint64_t *scratch = (uint64_t *)conn_mgr.get_scratch(tid, cid);
      size_t id;
      while (!stop_flag) {
        auto id = co_await alloc_next_id(scratch, c_ctx);
        if (stop_flag || id >= start_nvecs + max_insert)
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
        ii_times_by_tid[tid].emplace_back(single_ns);
        is_times_by_tid[tid].emplace_back(traces[tid].search_time);
      }
      co_return 0;
    };
    auto comain = [&](size_t cid) -> Task<int> {
      if (tid < search_threads) {
        for (size_t nloop = 0; nloop < 3; ++nloop) {
          if (sel == -1)
            co_await do_raw_search(cid,
                                   std::format("search,{},{}t{}c,loop{}",
                                               start_nvecs, search_threads,
                                               max_coros, nloop),
                                   false, gt(start_nvecs), b_dyn.get());
          else {
            const auto [full_trim, ordered, with_search_lock,
                        with_insert_lock] = sel2bools[sel];
            co_await do_search(cid,
                               std::format("search,{},{}t{}c,loop{}",
                                           start_nvecs, search_threads,
                                           max_coros, nloop),
                               false, gt(start_nvecs), b_dyn.get());
          }
        }
        stop_flag = true;
      } else {
        if (sel == -1)
          co_await do_raw_insert(cid,
                                 std::format("insert,{},{}t{}c", start_nvecs,
                                             insert_threads, max_coros),
                                 L_insert, true, false, true);
        else {
          const auto [full_trim, ordered, with_search_lock, with_insert_lock] =
              sel2bools[sel];
          co_await do_insert(cid,
                             std::format("insert,{},{}t{}c", start_nvecs,
                                         insert_threads, max_coros),
                             L_insert, full_trim, ordered, with_search_lock,
                             with_insert_lock);
        }
      }
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
  auto t1 = std::chrono::high_resolution_clock::now();
  for (size_t tid = 0; tid < max_threads; ++tid)
    threads.emplace_back(thmain, tid);
  for (auto &thread : threads)
    thread.join();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dur_s =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();

  std::vector<uint64_t> ii_times;
  std::vector<uint64_t> is_times;
  for (size_t tid = search_threads; tid < max_threads; ++tid) {
    ii_times.insert(ii_times.end(), ii_times_by_tid[tid].begin(),
                    ii_times_by_tid[tid].end());
    is_times.insert(is_times.end(), is_times_by_tid[tid].begin(),
                    is_times_by_tid[tid].end());
  }

  double ips = 0.0;
  double avg_insert_ms = 0.0, p50_insert_ms = 0.0, p90_insert_ms = 0.0,
         p99_insert_ms = 0.0;
  double avg_search_ms = 0.0, p50_search_ms = 0.0, p90_search_ms = 0.0,
         p99_search_ms = 0.0;
  if (!ii_times.empty() && dur_s > 0) {
    ips = ii_times.size() * 1.0 / dur_s;
    std::sort(ii_times.begin(), ii_times.end());
    avg_insert_ms = 1. *
                    std::accumulate(ii_times.begin(), ii_times.end(), 0ull) /
                    ii_times.size() / 1000000;
    p50_insert_ms = 1. * ii_times[ii_times.size() / 2] / 1000000;
    p90_insert_ms = 1. * ii_times[ii_times.size() * 9 / 10] / 1000000;
    p99_insert_ms = 1. * ii_times[ii_times.size() * 99 / 100] / 1000000;
  }
  if (!is_times.empty()) {
    std::sort(is_times.begin(), is_times.end());
    avg_search_ms = 1. *
                    std::accumulate(is_times.begin(), is_times.end(), 0ull) /
                    is_times.size() / 1000000;
    p50_search_ms = 1. * is_times[is_times.size() / 2] / 1000000;
    p90_search_ms = 1. * is_times[is_times.size() * 9 / 10] / 1000000;
    p99_search_ms = 1. * is_times[is_times.size() * 99 / 100] / 1000000;
  }

  Trace trace;
  trace.reset();
  uint64_t *p_sum = reinterpret_cast<uint64_t *>(&trace);
  for (size_t i = search_threads; i < max_threads; ++i) {
    uint64_t *p_trace = reinterpret_cast<uint64_t *>(&traces[i]);
    for (size_t j = 0; j < sizeof(Trace) / sizeof(uint64_t); ++j)
      p_sum[j] += p_trace[j];
  }
  println("insert,{},{}t{}c,$ips:{},avg_insert_ms:{},p50_insert_ms:{},"
          "p90_insert_ms:{},p99_insert_ms:{},avg_search_ms:{},p50_search_ms:{},"
          "p90_search_ms:{},p99_search_ms:{},ni:{},{},{}",
          start_nvecs, insert_threads, max_coros, ips, avg_insert_ms,
          p50_insert_ms, p90_insert_ms, p99_insert_ms, avg_search_ms,
          p50_search_ms, p90_search_ms, p99_search_ms, ii_times.size(),
          trace.search_trace_str(), trace.insert_trace_str());

  return 0;
}