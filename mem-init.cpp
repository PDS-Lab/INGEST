#include "conn_mgr.h"
#include "define.h"
// #include "index_io.h"
#include <myutils/io.h>
#include <myutils/utils.h>
#include <lcoro/rdma_worker.h>
#include <fstream>

#include <argparse/argparse.hpp>

using namespace rdma;

constexpr auto parse_to_map(const std::string &path) {
  std::ifstream infile(path);
  std::unordered_map<std::string, std::string> data_map;
  std::string line;
  auto trim = [](const std::string &str) -> std::string {
    const std::string whitespace = " \t\n\r\f\v";
    size_t first = str.find_first_not_of(whitespace);
    if (std::string::npos == first)
      return "";
    size_t last = str.find_last_not_of(whitespace);
    return str.substr(first, (last - first + 1));
  };
  while (std::getline(infile, line)) {
    size_t colon_pos = line.find(':');
    if (colon_pos != std::string::npos) {
      std::string key = line.substr(0, colon_pos);
      std::string value = line.substr(colon_pos + 1);
      key = trim(key);
      value = trim(value);
      if (!key.empty())
        data_map[key] = value;
    }
  }
  return data_map;
}

int main(int argc, char **argv) {
  std::string dev_name;
  int rdma_gid_index = 1;
  int rdma_dev_port = 1;
  size_t max_nthreads = 16;
  size_t max_ncoros = 2;

  std::string index_path;
  std::string arg_mem_blk_size;
  std::vector<std::pair<std::string, uint32_t>> arg_mem_servers; // IP:PORT:UID
  {
    argparse::ArgumentParser program("pub");
    // program.add_argument("base-path").required().store_into(base_path);
    program.add_argument("index-path").required().store_into(index_path);
    program.add_argument("mem-blk-size")
        .required()
        .store_into(arg_mem_blk_size);
    program.add_argument("mem-servers")
        .nargs(argparse::nargs_pattern::at_least_one)
        .action([&](const std::string &value) {
          auto pos = value.find_last_of(':');
          if (pos == std::string::npos)
            throw std::invalid_argument("Invalid mem list: " + value);
          std::string server_addr = value.substr(0, pos);
          uint32_t uid = std::stoul(value.substr(pos + 1));
          arg_mem_servers.emplace_back(server_addr, uid);
        });

    program.add_argument("-d", "--dev-name")
        .default_value(dev_name)
        .store_into(dev_name);
    program.add_argument("-g", "--gid-index")
        .default_value(rdma_gid_index)
        .store_into(rdma_gid_index);
    program.add_argument("--rdma-port")
        .default_value(rdma_dev_port)
        .store_into(rdma_dev_port);
    program.add_argument("-t", "--max-nthreads")
        .default_value(max_nthreads)
        .store_into(max_nthreads);
    program.add_argument("-c", "--max-ncoros")
        .default_value(max_ncoros)
        .store_into(max_ncoros);
    try {
      program.parse_args(argc, argv);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      std::cerr << program << std::endl;
      return 1;
    }
  }
  eprintln("dev_name: {}, rdma_gid_index: {}, rdma_dev_port: {}", dev_name,
           rdma_gid_index, rdma_dev_port);
  eprintln("max_nthreads: {}, max_ncoros: {}", max_nthreads, max_ncoros);
  eprintln("index_path: {}", index_path);
  eprintln("mem_blk_size: {}", arg_mem_blk_size);
  for (auto [server_addr, uid] : arg_mem_servers)
    eprintln("  server_addr: {}, uid: {}", server_addr, uid);
  auto index_cfg_meta = parse_to_map(std::format("{}.meta", index_path));
  const auto dim = std::stoull(index_cfg_meta.at("dim"));
  const auto nvecs = std::stoull(index_cfg_meta.at("nvecs"));
  const auto M = std::stoull(index_cfg_meta.at("M"));
  const auto ep = std::stoull(index_cfg_meta.at("ep"));
  const auto flip_seed = std::stoull(index_cfg_meta.at("flip_seed"));
  eprintln("dim: {}, nvecs: {}, M: {}, ep: {}, flip_seed: {}", dim, nvecs, M,
           ep, flip_seed);

  MMap index_mmap(index_path);
  auto index_ptr = index_mmap.ptr<uint8_t>();

  Device dev(dev_name, rdma_gid_index, rdma_dev_port);
  ConnManager conn_mgr(dev, max_nthreads, max_ncoros);
  auto &cfg = conn_mgr.cfgs.emplace_back();
  std::vector<MemCfg> mem_cfgs;
  for (auto [server_addr, uid] : arg_mem_servers)
    mem_cfgs.emplace_back(uid, server_addr);

  const size_t mem_blk_size = parse_size_string(arg_mem_blk_size, 1024);
  cfg.init(dim, M, ep, mem_blk_size, mem_cfgs, flip_seed);
  conn_mgr.connect_mem_servers();
  eprintln("cfg: {}", cfg);
  // println("{},{},{},{},{}", dim, M, ep, mem_blk_size, flip_seed);
  println("{}", cfg["cmd"]);

  auto map_info = conn_mgr.map_infos[0];

  Barrier b(max_nthreads * max_ncoros);
  std::atomic_size_t task_alloc{0};
  auto thmain = [&](size_t tid) {
    auto &cli = conn_mgr.clients[tid];
    std::vector<float> rc(dim);
    std::vector<float> ro(dim);
    auto comain = [&](size_t cid) -> Task<int> {
      while (true) {
        auto i = task_alloc.fetch_add(1);
        if (i >= nvecs)
          break;
        auto [conn, addr, rkey] =
            get_item(cfg, map_info, conn_mgr.max_nthreads, i, tid);
        auto scratch = conn_mgr.get_scratch(tid, cid);
        memcpy(scratch, index_ptr + i * cfg.item_size(), cfg.item_size());
        co_await conn->write(addr, rkey, scratch, cfg.item_size(),
                             conn_mgr.scratch_mr->lkey);
      }
      co_return 0;
    };

    EventLoop el(*cli);
    std::vector<Task<int>> tasks;
    for (size_t cid = 0; cid < max_ncoros; ++cid)
      tasks.push_back(comain(cid));
    el.start(gather(tasks));
  };
  std::vector<std::thread> threads;
  for (size_t tid = 0; tid < max_nthreads; ++tid)
    threads.emplace_back(thmain, tid);
  for (auto &thread : threads)
    thread.join();

  return 0;
}