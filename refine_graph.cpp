// #include "index_io.h"
#include "define.h"
#include <myutils/io.h>
#include <myutils/utils.h>

#include "myutils/random.h"
#include "rabitq.h"

#include <argparse/argparse.hpp>

#include <omp.h>

struct BaseParser {
  uint32_t nvecs;
  uint32_t M;
  uint32_t ep;
  virtual std::tuple<uint32_t, uint32_t *> nbrs(uint32_t i) const = 0;
};

struct SQGParser : BaseParser {
  MMap in_mmap;
  uint32_t *edges;
  SQGParser(const std::string &input_path,
            uint32_t ep = std::numeric_limits<uint32_t>::max()) {
    in_mmap.map(input_path);
    edges = in_mmap.ptr<uint32_t>();
    M = edges[0];
    this->ep = ep;
    this->nvecs = in_mmap.size_ / (M + 1) / sizeof(uint32_t);
  }
  std::tuple<uint32_t, uint32_t *> nbrs(uint32_t i) const {
    return {M, edges + i * (M + 1) + 1};
  }
};

inline auto load_edges_parallel(void *edges_base, size_t size, uint32_t M,
                                bool dedup, uint32_t nthread) {
  expect(is_times_of(size, sizeof(uint32_t)));
  auto ptr_u32 = (uint32_t *)edges_base;
  auto validate = [M](uint32_t *ptr, size_t ntest = 10) {
    while (ntest-- > 0) {
      if (*ptr > M)
        return false;
      ptr += 1 + *ptr;
    }
    return true;
  };
  expect(validate(ptr_u32));
  size_t bs = upper_div(size / sizeof(uint32_t), nthread);
  std::vector<uint32_t *> starts(nthread + 1); // estimated batch start
  for (size_t i = 0; i < nthread; ++i) {
    auto start = ptr_u32 + i * bs;
    while (!validate(start))
      start++;
    starts[i] = start;
  }
  starts[nthread] = (uint32_t *)((char *)edges_base + size);
  std::vector<size_t> cnt(nthread);
#pragma omp parallel num_threads(nthread)
  {
    auto tid = omp_get_thread_num();
    auto ptr = starts[tid];
    auto end = starts[tid + 1];
    size_t tmp = 0;
    while (ptr < end) {
      auto nn = *ptr;
      ptr += 1 + nn;
      ++tmp;
    }
    cnt[tid] = tmp;
  }
  std::vector<size_t> offsets(nthread + 1, 0);
  std::partial_sum(cnt.begin(), cnt.end(), offsets.begin() + 1);
  size_t nvecs = offsets.back();

  std::unique_ptr<uint32_t[]> graph(new uint32_t[nvecs * (M + 1)]);
#pragma omp parallel num_threads(nthread)
  {
    auto tid = omp_get_thread_num();
    auto ptr = starts[tid];
    auto end = starts[tid + 1];
    auto res = graph.get() + offsets[tid] * (M + 1);
    while (ptr < end) {
      auto nn = *ptr;
      std::copy(ptr + 1, ptr + nn + 1, res + 1);
      if (dedup) {
        std::sort(res + 1, res + nn + 1);
        *res = std::unique(res + 1, res + nn + 1) - (res + 1);
      } else
        *res = nn;
      res += M + 1;
      ptr += 1 + nn;
    }
  }
  return std::make_tuple(nvecs, std::move(graph));
}

inline auto
load_vamana(const std::string &index_path, bool dedup = true,
            uint32_t nthread = std::thread::hardware_concurrency()) {
  struct VamanaHeader {
    uint64_t index_size;
    uint32_t M;
    uint32_t ep;
    uint64_t num_frozen_points;
  };
  MMap index_mmap(index_path);
  auto hdr = index_mmap.ptr<VamanaHeader>();
  expect_eq(hdr->index_size, index_mmap.size_);
  auto [nvecs, graph] = load_edges_parallel(
      hdr + 1, index_mmap.size_ - sizeof(VamanaHeader), hdr->M, dedup, nthread);
  return std::make_tuple(nvecs, hdr->M, std::vector<uint32_t>{hdr->ep},
                         std::move(graph));
}

struct VamanaParser : BaseParser {
  std::unique_ptr<uint32_t[]> edges;
  VamanaParser(const std::string &input_path) {
    auto [nvecs, M, ep, edges] = load_vamana(input_path);
    this->nvecs = nvecs;
    this->M = M;
    this->ep = ep[0];
    this->edges = std::move(edges);
  }
  std::tuple<uint32_t, uint32_t *> nbrs(uint32_t i) const {
    auto nnbrs_ptr = edges.get() + i * (M + 1);
    return {*nnbrs_ptr, nnbrs_ptr + 1};
  }
};

uint32_t find_ep(const auto *vecs, size_t nvecs, size_t dim) {
  std::vector<double> tmp_sum_vec(dim, 0);
  auto tmp_sum = tmp_sum_vec.data();
#pragma omp parallel for reduction(+ : tmp_sum[0 : dim])
  for (size_t i = 0; i < nvecs; ++i)
    for (size_t j = 0; j < dim; ++j)
      tmp_sum[j] += vecs[i][j];
  std::vector<float> centroid(dim);
  for (size_t j = 0; j < dim; ++j)
    centroid[j] = tmp_sum[j] / nvecs;
  float min_dist = std::numeric_limits<float>::max();
  uint32_t min_id = std::numeric_limits<uint32_t>::max();
#pragma omp parallel
  {
    float local_min_dist = std::numeric_limits<float>::max();
    uint32_t local_min_id = std::numeric_limits<uint32_t>::max();
#pragma omp for nowait
    for (size_t i = 0; i < nvecs; ++i) {
      auto dist = l2sqr(vecs[i], centroid.data(), dim);
      if (dist < local_min_dist) {
        local_min_dist = dist;
        local_min_id = i;
      }
    }
#pragma omp critical
    {
      if (local_min_dist < min_dist ||
          (local_min_dist == min_dist && local_min_id < min_id)) {
        min_dist = local_min_dist;
        min_id = local_min_id;
      }
    }
  }
  return min_id;
}

using NeighborType = NeighborImpl<float, uint32_t>;

auto trim_rng(std::vector<NeighborType> &src, size_t M, const auto &dist_func) {
  std::vector<int> idxs(src.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  std::sort(idxs.begin(), idxs.end(),
            [&](const auto &a, const auto &b) { return src[a] < src[b]; });
  std::vector<uint32_t> res;
  res.reserve(M);
  for (size_t i = 0; i < src.size() && res.size() < M; ++i) {
    auto &cur = src[idxs[i]];
    bool occlude = false;
    for (size_t j = 0; j < res.size(); ++j)
      if (dist_func(cur.id, src[res[j]].id) < cur.dist) {
        occlude = true;
        break;
      }
    if (!occlude) {
      res.push_back(idxs[i]);
      idxs[i] = -1; // mark as used
    }
  }
  size_t ntrimmed = res.size();
  for (size_t i = 0; i < src.size() && res.size() < M; ++i)
    if (idxs[i] >= 0)
      res.push_back(idxs[i]);
  return std::make_tuple(ntrimmed, res);
}

auto trim_vamana(auto *src, uint32_t *src_idxs, size_t nidxs,
                 uint32_t *dst_idxs, size_t M, const auto &dist_func,
                 float alpha = 1.2f, bool saturate = false) {
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
      cur_fac = std::numeric_limits<float>::max(); // mark as used
      for (size_t j = i + 1; j < nidxs; ++j) {
        auto &other = src[src_idxs[j]];
        auto &other_fac = occlude_factors[j];
        if (other_fac > cur_alpha)
          continue;
        float djk = dist_func(cur, other);
        // 防止距离为0被误认为已经使用，除以2
        other_fac = djk == 0 ? std::numeric_limits<float>::max() / 2
                             : std::max(other_fac, other.dist / djk);
      }
    }
    cur_alpha *= 1.2f;
  }
  size_t ntrimmed = pos;
  if (saturate && pos < M) {
    for (size_t i = 0; i < nidxs && pos < M; ++i) {
      auto &cur = src[src_idxs[i]];
      auto &cur_fac = occlude_factors[i];
      if (cur_fac == std::numeric_limits<float>::max())
        continue;
      dst_idxs[pos++] = src_idxs[i];
    }
  }
  return std::make_tuple(ntrimmed, pos);
}

auto trim_vamana(auto *src, size_t nsrc, uint32_t *dst_idxs, size_t M,
                 const auto &dist_func, float alpha = 1.2f,
                 bool saturate = false) {
  std::vector<uint32_t> src_idxs(nsrc);
  std::iota(src_idxs.begin(), src_idxs.end(), 0);
  return trim_vamana(src, src_idxs.data(), nsrc, dst_idxs, M, dist_func, alpha,
                     saturate);
}

const std::string vaild_input_types[] = {"sqg", "vamana"};

int main(int argc, char *argv[]) {
  size_t dim;
  size_t nvecs;
  std::string base_path;
  std::string input_path;
  std::string output_prefix;
  // size_t arg_dim = 0;
  // size_t arg_max_nvecs = 0;
  std::string input_type;
  uint32_t ep = std::numeric_limits<uint32_t>::max();

  uint64_t flip_seed = 2025;

  {
    argparse::ArgumentParser parser("refine_graph");
    parser.add_argument("dim").required().store_into(dim);
    parser.add_argument("nvecs").required().store_into(nvecs);
    parser.add_argument("base_path").required().store_into(base_path);
    parser.add_argument("input_path").required().store_into(input_path);
    parser.add_argument("output_prefix").required().store_into(output_prefix);
    parser.add_argument("-t", "--input-type")
        .default_value(input_type)
        .store_into(input_type);
    parser.add_argument("-e", "--ep").default_value(ep).store_into(ep);
    // parser.add_argument("-d", "--dim")
    //     .default_value(arg_dim)
    //     .store_into(arg_dim);
    // parser.add_argument("-n", "--max-nvecs")
    //     .default_value(arg_max_nvecs)
    //     .store_into(arg_max_nvecs);
    parser.add_argument("-f", "--flip-seed")
        .default_value(flip_seed)
        .store_into(flip_seed);
    try {
      parser.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
      std::cerr << err.what() << std::endl;
      std::cerr << parser;
      return 1;
    }
  }

  if (input_type.empty()) {
    std::filesystem::path input_path_obj(input_path);
    input_type = input_path_obj.extension().string().substr(1);
  }
  BaseParser *parser = nullptr;
  if (input_type == "sqg")
    parser = new SQGParser(input_path, ep);
  else if (input_type == "vamana" || input_type == "diskann")
    parser = new VamanaParser(input_path);
  else
    throw std::runtime_error("Invalid input type: " + input_type);

  // Dataset ds(arg_dim, arg_max_nvecs, base_path);

  // const size_t dim = ds.dim();
  const size_t M = parser->M;
  // const size_t nvecs = parser->nvecs;
  // expect_le(nvecs, ds.nvecs());
  MMap base_mmap(base_path);
  auto xb = base_mmap.ptr<float>();
  auto vec = [&](size_t i) { return xb + i * dim; };

  GraphConfig cfg;
  cfg.init(dim, M, parser->ep, 0, flip_seed);
  println("flip({}): {::02x}", cfg.rot_fac(), cfg._flip);

  const size_t item_size = cfg.item_size();

  println("nvecs: {}, M: {}, dim: {}, item_size: {}", nvecs, M, dim, item_size);

  FILE *fout = fopen(std::format("{}.meta", output_prefix).c_str(), "w");
  println(fout, "dim: {}", dim);
  println(fout, "nvecs: {}", nvecs);
  println(fout, "M: {}", M);
  println(fout, "ep: {}", parser->ep);
  println(fout, "flip_seed: {}", flip_seed);
  println(fout, "item_size: {}", item_size);
  println(fout, "ds_path: {}", base_path);
  fclose(fout);

  MMap out_mmap(output_prefix, nvecs * item_size, true);
  auto out_ptr = out_mmap.ptr<uint8_t>();

  size_t core_cnt = 0;
  size_t redundant_cnt = 0;
#pragma omp parallel reduction(+ : core_cnt, redundant_cnt)
  {
    std::vector<NeighborType> tmp_nbrs;
#pragma omp for
    for (size_t i = 0; i < nvecs; ++i) {
      auto item = out_ptr + i * item_size;
      // vec
      cfg.rotate(vec(i), cfg.get_vec(item));
      // nbrs
      auto [nnbrs, src_nbrs] = parser->nbrs(i);
      auto dst_nbrs = cfg.get_nbrs(item);
      if (1) {
        tmp_nbrs.resize(nnbrs);
        for (size_t j = 0; j < nnbrs; ++j)
          tmp_nbrs[j] = {l2sqr(vec(i), vec(src_nbrs[j]), dim), src_nbrs[j]};
        std::vector<uint32_t> dst_idxs(M);
        auto [ntrimmed, nret] = trim_vamana(
            tmp_nbrs.data(), nnbrs, dst_idxs.data(), M,
            [&](uint32_t a, uint32_t b) { return l2sqr(vec(a), vec(b), dim); },
            1.2f, true);
        for (size_t j = 0; j < ntrimmed; ++j)
          dst_nbrs[j].set(tmp_nbrs[dst_idxs[j]].id, PruneFlag::CORE);
        core_cnt += ntrimmed;
        for (size_t j = ntrimmed; j < std::min(nret, M); ++j)
          dst_nbrs[j].set(tmp_nbrs[dst_idxs[j]].id, PruneFlag::REDUNDANT);
        redundant_cnt += M - ntrimmed;
        for (size_t j = nret; j < M; ++j)
          dst_nbrs[j].set(randu64(nvecs), PruneFlag::REDUNDANT);
      } else {
        for (size_t j = 0; j < nnbrs; ++j)
          dst_nbrs[j].set(src_nbrs[j], PruneFlag::CORE);
        for (size_t j = nnbrs; j < M; ++j)
          dst_nbrs[j].set(randu64(nvecs), PruneFlag::REDUNDANT);
      }
    }
  }

  println("core_cnt: {}, redundant_cnt: {}", core_cnt, redundant_cnt);

#pragma omp parallel
  {
#pragma omp for
    for (size_t i = 0; i < nvecs; ++i) {
      auto cur_item = out_ptr + i * item_size;
      auto cur_vec = cfg.get_vec(cur_item);
      auto cur_nbrs = cfg.get_nbrs(cur_item);
      auto codes = cfg.get_codes(cur_item);
      for (size_t j = 0; j < M; ++j) {
        if (!NbEntry::is_valid(cur_nbrs[j]))
          continue;
        auto nbr_item = out_ptr + cur_nbrs[j].id() * item_size;
        auto nbr_vec = cfg.get_vec(nbr_item);
        auto [bcode, f_add, f_recale] = cfg.get_code(codes, j);
        std::tie(*f_add, *f_recale, std::ignore) =
            rabitq_impl(dim, nbr_vec, cur_vec, bcode);
      }
    }
  }
}