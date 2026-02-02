// #include "index_io.h"
#include "index_utils.h"
#include <myutils/heap.h>
#include <myutils/io.h>
#include <myutils/utils.h>

#include <argparse/argparse.hpp>
#include <limits>
#include <thread>

int main(int argc, char *argv[]) {
  size_t dim = 0;
  std::string base_path;
  std::string query_path;
  std::string gt_path;
  size_t gt_begin = 0;
  size_t gt_end = 0;
  size_t gt_step = 1000000;
  size_t ngt = 100;
  size_t max_mem_gb = 64;
  size_t nthread = std::thread::hardware_concurrency();
  {
    argparse::ArgumentParser parser("mkgt");
    parser.add_argument("dim").required().store_into(dim);
    parser.add_argument("base_path").required().store_into(base_path);
    parser.add_argument("query_path").required().store_into(query_path);
    parser.add_argument("gt_path").required().store_into(gt_path);
    parser.add_argument("-b", "--gt-begin")
        .default_value(gt_begin)
        .store_into(gt_begin);
    parser.add_argument("-e", "--gt-end")
        .default_value(gt_end)
        .store_into(gt_end);
    parser.add_argument("-s", "--gt-step")
        .default_value(gt_step)
        .store_into(gt_step);
    parser.add_argument("-n", "--ngt").default_value(ngt).store_into(ngt);
    parser.add_argument("-t", "--nthread")
        .default_value(nthread)
        .store_into(nthread);
    parser.add_argument("-m", "--max-mem-gb")
        .default_value(max_mem_gb)
        .store_into(max_mem_gb);

    try {
      parser.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
      std::cerr << err.what() << std::endl;
      std::cerr << parser;
      return 1;
    }
  }
  if (nthread == 0)
    nthread = 1;
  expect_gt(gt_step, 0);
  expect_lt(gt_begin, gt_end);
  expect_ge(gt_begin, ngt);
  const size_t gt_num = upper_div(gt_end - gt_begin, gt_step) + 1;
  println("gt_num: {}", gt_num);

  MMap base_mmap(base_path);
  const auto xb = base_mmap.ptr<float>();
  auto vec = [&](size_t i) { return xb + i * dim; };
  MMap query_mmap(query_path);
  const auto xq = query_mmap.ptr<float>();
  auto query = [&](size_t i) { return xq + i * dim; };
  const size_t nq = query_mmap.size_ / sizeof(float) / dim;

  MMap gt_mmap(gt_path, gt_num * nq * ngt * sizeof(uint32_t), true);
  std::vector<size_t> gt_ats(gt_num);
  for (size_t i = 0; i < gt_num; ++i)
    gt_ats[i] = gt_begin + i * gt_step;
  gt_ats.back() = gt_end;
  auto gt_base = gt_mmap.ptr<uint32_t>();

  std::vector<NeighborImpl<float, uint32_t>> retsets(nq * ngt);
#pragma omp parallel for num_threads(nthread)
  for (size_t i = 0; i < retsets.size(); ++i)
    retsets[i] = {std::numeric_limits<float>::max(),
                  std::numeric_limits<uint32_t>::max()};

  size_t max_blk_size =
      (max_mem_gb * 1024 * 1024 * 1024 -
       retsets.size() * sizeof(NeighborImpl<float, uint32_t>)) /
      dim / sizeof(float);
  max_blk_size = lower_align(max_blk_size, 1000000);
  expect_gt(max_blk_size, 1000000);
  println("max_blk_size: {}", max_blk_size);
  size_t cur = 0;
  for (size_t gid = 0; gid < gt_ats.size(); ++gid) {
    println("gt_at: {}", gt_ats[gid]);
    do {
      size_t batch_size = std::min(gt_ats[gid] - cur, max_blk_size);
#pragma omp parallel for schedule(dynamic) num_threads(nthread)
      for (size_t q = 0; q < nq; ++q) {
        auto retset = retsets.data() + q * ngt;
        for (size_t i = 0; i < batch_size; ++i) {
          auto d = l2sqr(query(q), vec(cur + i), dim);
          if (d < retset[0].dist)
            binary_heap::heap_down(retset, ngt, {d, (uint32_t)(cur + i)});
        }
      }
      cur += batch_size;
    } while (cur < gt_ats[gid]);
    auto gt = gt_base + gid * nq * ngt;
#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t q = 0; q < nq; ++q) {
      auto retset = retsets.data() + q * ngt;
      std::sort(retset, retset + ngt);
      for (size_t k = 0; k < ngt; ++k)
        gt[q * ngt + k] = retset[k].id;
      binary_heap::heapify(retset, ngt);
    }
  }

  return 0;
}