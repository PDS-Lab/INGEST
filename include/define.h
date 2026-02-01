#pragma once

#include "rotator.h"
#include <myutils/memory.h>
#include <myutils/random.h>
#include <myutils/utils.h>
#include <ranges>

constexpr float prune_alpha = 1.2f;
constexpr size_t rotator_niter = 2;
constexpr size_t group_size = 4;
constexpr size_t align = 64;
constexpr size_t default_scratch_size = 128ull << 20;
constexpr size_t max_req_per_coro = 4;
constexpr size_t max_search_iters = std::numeric_limits<size_t>::max();
constexpr size_t raw_lock_tbl_size = 65536;

enum class PruneFlag : uint8_t {
  CORE = 0,     
  PENDING = 1,  
  REDUNDANT = 2 
};

struct NbEntry {
  static constexpr size_t id_bits = 56;
  static constexpr size_t prune_flag_bits = 8;
  static constexpr uint64_t id_shift = 0;
  static constexpr uint64_t prune_flag_shift = id_bits;
  static constexpr uint64_t id_mask = ((1ull << id_bits) - 1) << id_shift;
  static constexpr uint64_t prune_flag_mask = ((1ull << prune_flag_bits) - 1)
                                              << prune_flag_shift;
  uint64_t _raw{std::numeric_limits<uint64_t>::max()};
  explicit constexpr NbEntry(uint64_t id = 0,
                             PruneFlag prune_flag = PruneFlag::CORE) {
    set(id, prune_flag);
  }
  constexpr static NbEntry invalid() {
    return NbEntry(id_mask, PruneFlag::REDUNDANT);
  }
  constexpr static bool is_valid(NbEntry nbe) { return nbe.id() != id_mask; }
  constexpr void set(uint64_t id, PruneFlag prune_flag = PruneFlag::CORE) {
    internal_assert(id < (1ull << id_bits));
    const uint64_t prune_flag_value = static_cast<uint8_t>(prune_flag);
    internal_assert(prune_flag_value < (1ull << prune_flag_bits));
    _raw = (prune_flag_value << prune_flag_shift) | (id << id_shift);
  }
  constexpr void set_id(uint64_t id) {
    internal_assert(id < (1ull << id_bits));
    _raw = (_raw & ~id_mask) | (id << id_shift);
  }
  constexpr void set_prune_flag(PruneFlag prune_flag) {
    const uint64_t prune_flag_value = static_cast<uint8_t>(prune_flag);
    internal_assert(prune_flag_value < (1ull << prune_flag_bits));
    _raw = (_raw & ~prune_flag_mask) | (prune_flag_value << prune_flag_shift);
  }
  constexpr uint64_t id() const { return (_raw & id_mask) >> id_shift; }
  constexpr PruneFlag prune_flag() const {
    return static_cast<PruneFlag>((_raw & prune_flag_mask) >> prune_flag_shift);
  }
  constexpr NbEntry with_prune_flag(PruneFlag prune_flag) const {
    return NbEntry(id(), prune_flag);
  }
};
constexpr auto operator<=>(const NbEntry &a, const NbEntry &b) {
  return a.id() <=> b.id();
}
constexpr auto operator==(const NbEntry &a, const NbEntry &b) {
  return a.id() == b.id();
}
template <> struct std::formatter<NbEntry> : EmptyParser {
  auto format(const NbEntry &id_ver, auto &ctx) const {
    return std::format_to(ctx.out(), "({},{})", id_ver.id(),
                          static_cast<uint8_t>(id_ver.prune_flag()));
  };
};

struct Meta {
  size_t cur_nvecs{0};
};

struct MemCfg {
  uint32_t uid;
  std::string server_addr; 
};
template <> struct std::formatter<MemCfg> : EmptyParser {
  auto format(const MemCfg &mem_cfg, auto &ctx) const {
    return std::format_to(ctx.out(), "MemCfg(uid: {}, server_addr: {})",
                          mem_cfg.uid, mem_cfg.server_addr);
  };
};

constexpr auto id2addr(size_t i, size_t batch_size) {
  const size_t bid = i / batch_size;
  const size_t boff = i % batch_size;
  const size_t gid = boff % group_size;
  const size_t goff = boff / group_size;
  const size_t id = (bid * group_size) + gid;
  return std::make_tuple(id, goff);
}

constexpr size_t parse_size_string(std::string_view size_str, size_t base) {
  constexpr double eps = 1e-6;
  if (size_str.empty())
    return 0;

  size_t pos = 0;
  while (pos < size_str.length() &&
         (std::isdigit(size_str[pos]) || size_str[pos] == '.'))
    ++pos;

  double value = std::stod(std::string(size_str.substr(0, pos)));

  if (pos < size_str.length()) {
    char unit = std::tolower(size_str[pos]);
    switch (unit) {
    case 'k': 
      value *= base;
      break;
    case 'm': 
      value *= base * base;
      break;
    case 'g': 
    case 'b': 
      value *= base * base * base;
      break;
    default:
      throw std::invalid_argument("Invalid size unit: " +
                                  std::string(1, size_str[pos]));
    }
  }

  return static_cast<size_t>(value + eps);
}

struct GraphConfig {
  uint32_t _dim;
  uint32_t _M;
  uint64_t _ep;

  uint32_t _batch_size;   
  uint32_t _item_size;    
  uint32_t _stride;       
  uint32_t _nbrs_offset;  
  uint32_t _codes_offset; 
  uint32_t _code_size;    
  float _rot_fac;
  std::vector<uint8_t> _flip;

  std::vector<MemCfg> _mem_cfgs;
  std::map<std::string, std::string> _extra_info;

  constexpr size_t dim() const { return _dim; }
  constexpr size_t M() const { return _M; }
  constexpr uint64_t ep() const { return _ep; }
  constexpr size_t batch_size() const { return _batch_size; }
  constexpr size_t item_size() const { return _item_size; }
  constexpr size_t raw_item_size() const { return vec_size() + nbrs_size(); }
  constexpr size_t stride() const { return _stride; }
  constexpr size_t nbrs_offset() const { return _nbrs_offset; }
  constexpr size_t codes_offset() const { return _codes_offset; }
  constexpr size_t code_size() const { return _code_size; }
  constexpr float rot_fac() const { return _rot_fac; }
  auto &operator[](const std::string &key) { return _extra_info[key]; }
  const auto &mem_cfgs() const { return _mem_cfgs; }
  auto &mem_cfgs() { return _mem_cfgs; }
  const auto &extra_info() const { return _extra_info; }
  auto &extra_info() { return _extra_info; }

  static constexpr size_t vec_size(uint32_t dim) { return sizeof(float) * dim; }
  static constexpr size_t nbrs_size(uint32_t M) { return sizeof(NbEntry) * M; }
  static constexpr size_t code_size(uint32_t dim) {
    return dim / 8 + sizeof(float) * 2;
  }
  static constexpr size_t item_size(uint32_t dim, uint32_t M) {
    return vec_size(dim) + nbrs_size(M) + code_size(dim) * M + sizeof(uint16_t);
  }
  static constexpr size_t stride(uint32_t dim, uint32_t M) {
    return upper_align(item_size(dim, M), align);
  }

  void init(uint32_t dim, uint32_t M, uint64_t ep, size_t mem_blk_size,
            size_t flip_seed = 0) {
    std::vector<MemCfg> tmp_mem_cfgs;
    init(dim, M, ep, mem_blk_size, tmp_mem_cfgs, flip_seed);
  }
  void init(uint32_t dim, uint32_t M, uint64_t ep, size_t mem_blk_size,
            std::vector<MemCfg> &mem_cfgs, size_t flip_seed = 0) {
    std::map<std::string, std::string> tmp_extra_info;
    init(dim, M, ep, mem_blk_size, mem_cfgs, tmp_extra_info, flip_seed);
  }
  void init(uint32_t dim, uint32_t M, uint64_t ep, size_t mem_blk_size,
            std::vector<MemCfg> &mem_cfgs,
            std::map<std::string, std::string> &extra_info,
            size_t flip_seed = 0) {
    expect(is_times_of(dim, 16));
    this->_dim = dim;
    this->_M = M;
    this->_ep = ep;
    this->_mem_cfgs.swap(mem_cfgs);
    this->_extra_info.swap(extra_info);

    _code_size = bcode_size() + fac_size();
    _item_size = vec_size() + nbrs_size() + _code_size * M;
    _stride = upper_align(vec_size() + nbrs_size() + _code_size * M, align);
    _nbrs_offset = vec_size();
    _codes_offset = _nbrs_offset + nbrs_size();
    _rot_fac = 1.f / std::sqrt((float)(1 << (std::bit_width(dim) - 1)));
    _flip.resize(fht_kac_rotate_flip_size(dim, rotator_niter));
    Splitmix64(flip_seed).fill(_flip.data(), _flip.size());

    expect_eq(_mem_cfgs.size() % group_size, 0);
    _batch_size = group_size * (mem_blk_size / stride());
    _extra_info["cmd"] =
        std::format("{},{},{},{},{}", dim, M, ep, mem_blk_size, flip_seed);
    for (auto &mem_cfg : _mem_cfgs)
      _extra_info["cmd"] +=
          std::format(",{}:{}", mem_cfg.server_addr, mem_cfg.uid);
  }
  void init_from_str(std::string_view cfg_str) {
    std::vector<std::string> graph_cfg_parts;
    for (auto part : std::ranges::split_view(cfg_str, ','))
      graph_cfg_parts.emplace_back(part.begin(), part.end());
    expect_ge(graph_cfg_parts.size(), 5 + group_size);
    expect(is_times_of(graph_cfg_parts.size() - 5, group_size));
    auto dim = std::stoul(graph_cfg_parts[0]);
    auto M = std::stoul(graph_cfg_parts[1]);
    auto ep = std::stoul(graph_cfg_parts[2]);
    auto mem_blk_size = parse_size_string(graph_cfg_parts[3], 1024);
    auto flip_seed = std::stoul(graph_cfg_parts[4]);
    std::vector<MemCfg> mem_cfgs;
    for (size_t i = 5; i < graph_cfg_parts.size(); ++i) {
      auto pos = graph_cfg_parts[i].find_last_of(':');
      expect_ne(pos, std::string::npos);
      auto server_addr = graph_cfg_parts[i].substr(0, pos);
      auto uid = std::stoul(graph_cfg_parts[i].substr(pos + 1));
      mem_cfgs.emplace_back(uid, server_addr);
    }
    init(dim, M, ep, mem_blk_size, mem_cfgs, flip_seed);
  }
  auto id2addr(size_t i) { return ::id2addr(i, batch_size()); }
  void rotate(const float *vec, float *rotated_vec) {
    fht_kac_rotate(dim(), rot_fac(), _flip.data(), vec, rotated_vec);
  }
  float *get_vec(void *item) { return (float *)item; }
  NbEntry *get_nbrs(void *item) {
    return (NbEntry *)((uint8_t *)item + nbrs_offset());
  }
  uint8_t *get_codes(void *item) {
    return (uint8_t *)((uint8_t *)item + codes_offset());
  }
  auto get_code(void *codes, size_t j) {
    auto bcode = (uint8_t *)((uint8_t *)codes + j * code_size());
    auto f_add = (float *)(bcode + dim() / 8);
    auto f_rescale = f_add + 1;
    return std::make_tuple(bcode, f_add, f_rescale);
  }

  constexpr size_t vec_size() const { return sizeof(float) * dim(); }
  constexpr size_t nbrs_size() const { return sizeof(NbEntry) * M(); }
  constexpr size_t bcode_size() const { return dim() / 8; }
  constexpr size_t fac_size() const { return sizeof(float) * 2; }
  constexpr size_t seg_nvecs() const { return batch_size() / group_size; }
  constexpr size_t max_nvecs() const { return seg_nvecs() * mem_cfgs().size(); }
  constexpr std::string cmd_str() { return extra_info()["cmd"]; }
};

template <> struct std::formatter<GraphConfig> : EmptyParser {
  auto format(const GraphConfig &cfg, auto &ctx) const {
    return std::format_to(
        ctx.out(),
        "GlobalConfig(dim: {}, M: {}, ep: {}, seg_nvecs: {}/{}, "
        "mem_info: [{}], extra_info: {{{}}})",
        cfg.dim(), cfg.M(), cfg.ep(), cfg.seg_nvecs(), cfg.max_nvecs(),
        cfg.mem_cfgs(), cfg.extra_info());
  };
};
