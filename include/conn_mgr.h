#pragma once

#include "define.h"

#include <lcoro/rdma_worker.h>

struct MapInfo {
  rdma::Conn &conn;
  uintptr_t addr;
  uint32_t rkey;
};

inline std::string normalize_server_addr(const rdma::Device &dev,
                                         const std::string &server_addr) {
  std::string ip;
  int port;
  if (auto pos = server_addr.find(':'); pos != std::string::npos) {
    ip = server_addr.substr(0, pos);
    auto port_str = server_addr.substr(pos + 1);
    port = port_str.empty() ? rdma::default_listen_port : std::stoi(port_str);
  } else {
    ip = server_addr;
    port = rdma::default_listen_port;
  }
  if (ip.empty())
    ip = dev.get_ipstr();
  return std::format("{}:{}", ip, port);
}

struct ConnManager {
  rdma::Device &dev;
  ibv_mr *scratch_mr{nullptr};
  size_t scratch_size{0};
  std::vector<GraphConfig> cfgs;
  std::vector<std::vector<MapInfo>> map_infos; 

  size_t max_nthreads{0};
  size_t max_ncoros{0};
  
  std::vector<std::unique_ptr<rdma::Client>> clients;
  
  std::vector<std::map<std::string, rdma::Conn>> conns;

  ConnManager(rdma::Device &dev, size_t max_nthreads, size_t max_ncoros,
              size_t scratch_size = default_scratch_size)
      : dev(dev), max_nthreads(max_nthreads), max_ncoros(max_ncoros),
        scratch_size(scratch_size) {
    clients.resize(max_nthreads);
    conns.resize(max_nthreads);
    scratch_mr = dev.create_mr(scratch_size * max_nthreads * max_ncoros);
    for (size_t i = 0; i < max_nthreads; ++i)
      clients[i] = std::make_unique<rdma::Client>(dev);
  }
  rdma::Conn &get_or_create_conn(const std::string &server_addr, size_t tid = 0,
                                 rdma::QPCap cap = {}) {
    auto normalized_server_addr = normalize_server_addr(dev, server_addr);
    expect_lt(tid, max_nthreads);
    if (!conns[tid].contains(normalized_server_addr))
      conns[tid][normalized_server_addr] =
          clients[tid]->connect(normalized_server_addr, cap, tid);
    return conns[tid][normalized_server_addr];
  }
  void connect_mem_servers() {
    map_infos.resize(cfgs.size());
    std::map<std::tuple<std::string, uint32_t>, rdma::RemoteMR> rmr_map;
    auto get_rmr = [&](const std::string &server_addr, uint32_t uid,
                       rdma::Conn &conn) {
      auto key = std::make_tuple(server_addr, uid);
      if (rmr_map.contains(key))
        return rmr_map[key];
      auto rmr = conn->query_rmr_sync(uid);
      expect_eq(rmr.uid, uid);
      rmr_map[key] = rmr;
      return rmr;
    };
    for (size_t cfg_id = 0; cfg_id < cfgs.size(); ++cfg_id) {
      auto &cfg_map_info = map_infos[cfg_id];
      for (auto &mem_cfg : cfgs[cfg_id].mem_cfgs()) {
        for (size_t i = 0; i < max_nthreads; ++i) {
          auto normalized_server_addr =
              normalize_server_addr(dev, mem_cfg.server_addr);
          auto &conn = get_or_create_conn(
              normalized_server_addr, i,
              {.max_send_wr = (uint32_t)(max_ncoros * max_req_per_coro)});
          auto rmr = get_rmr(normalized_server_addr, mem_cfg.uid, conn);
          expect_le(cfgs[cfg_id].seg_nvecs() * cfgs[cfg_id].stride(), rmr.len);
          cfg_map_info.emplace_back(conn, rmr.addr, rmr.rkey);
        }
      }
    }
  }
  auto get_scratch(size_t tid, size_t cid) {
    return (uint8_t *)scratch_mr->addr +
           (tid * max_ncoros + cid) * scratch_size;
  }
};

inline auto get_item(GraphConfig &cfg, std::vector<MapInfo> &map_info,
                     size_t max_nthreads, size_t i, size_t tid) {
  auto [id, off] = id2addr(i, cfg.batch_size());
  id = id * max_nthreads + tid;
  expect_lt(id, map_info.size());
  auto &conn = map_info[id].conn;
  auto addr = map_info[id].addr + off * cfg.stride();
  return std::make_tuple(std::ref(conn), addr, map_info[id].rkey);
}
