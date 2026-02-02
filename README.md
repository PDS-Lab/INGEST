# INGEST

# Note on RDMA dependency.
The lcoro library used by our CMake configuration contains unpublished RDMA optimizations that are not part of this paper. The techniques and evaluation in this work do not rely on RDMA-specific optimizations. To reproduce the system, users should replace the RDMA-related components with their own RDMA implementation. The key algorithms presented in this paper are implemented in algo.h.

## Prepare

### Data Preparation

* `base.fbin` / `query.fbin`: Vectors stored contiguously, **without metadata**.
* `gt.u32bin`: Concatenation of ground truth results under the current index size, with shape:
  *(number of query vectors) × (k exact nearest neighbors per query) × (number of test batches + 1)*.

Generate `gt.u32bin`:

```
./build/Release/mkgt.exe dim base_path query_path gt_path -b gt-begin -e gt-end -s gt-step
```

### Build Initial Graph (based on: [https://github.com/microsoft/DiskANN](https://github.com/microsoft/DiskANN))

```
  auto index_build_params = diskann::IndexWriteParametersBuilder(L_build, M)
                                .with_filter_list_size(0)
                                .with_alpha(alpha)
                                .with_saturate_graph(saturate_graph)
                                .with_num_threads(nthread)
                                .build();
  auto filter_params = diskann::IndexFilterParamsBuilder()
                           .with_universal_label("")
                           .with_label_file("")
                           .with_save_path_prefix(index_prefix)
                           .build();

  auto config =
      diskann::IndexConfigBuilder()
          .with_metric(diskann::Metric::L2)
          .with_dimension(dim)
          .with_max_points(nvecs)
          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
          .with_data_type("float")
          .with_label_type("uint32")
          .is_dynamic_index(false)
          .with_index_write_params(index_build_params)
          .is_enable_tags(false)
          .is_use_opq(false)
          .is_pq_dist_build(false)
          .with_num_pq_chunks(0)
          .build();

  auto index_factory = diskann::IndexFactory(config);
  auto index = index_factory.create_instance();

  std::vector<uint32_t> tags;
  index->build(ds.vec(start_nvecs), nvecs, tags);
  index->save(index_prefix.c_str());
```

### Preprocess Initial Graph

```
./build/Release/refine_graph.exe dim nvecs base_path input_path output_prefix
```

---

## Build

Adjust the following parameters as needed.

### `define.h`

```
constexpr float prune_alpha = 1.2f;
constexpr size_t rotator_niter = 2;
constexpr size_t group_size = 4;
constexpr size_t align = 64;
constexpr size_t default_scratch_size = 128ull << 20;
constexpr size_t max_req_per_coro = 4;
constexpr size_t max_search_iters = std::numeric_limits<size_t>::max();
constexpr size_t raw_lock_tbl_size = 65536;
```

### `algo.h`

```
struct alignas(cache_line_size) CoroContext {
  constexpr static bool enable_trace = true;
  constexpr static bool enable_r_print = false;
  constexpr static bool enable_fail_back = false;
  constexpr static bool force_lock = true;
  constexpr static size_t max_lock_retry = 1000000; 
```

### Build Commands

```
cmake -B build -G "Ninja Multi-Config" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --config Release
```

---

## Run

### On the memory node:

```
./build/Release/mem-server.exe -p port -l lock-mem-id [-m meta-mem-id] 1:80g
```

### Prepare the execution script and modify `testcase` according to your setup:

Example:

```
from autotest_impl import run_test, run_all, TestCase

MNS = [] # mem-server-ip:port
CNS = [] # ip
THIS_IP = ""

MN_CFG = ",".join([f"{s}:1" for s in MNS])
LOCK_CFG = ",".join([f"{s}:2" for s in MNS])

test_plan = []

for num_cn in [1, 2, 3, 4]:
    for s in [-1, 0, 2, 3]:
        tc = TestCase(
            f"laion10m-g4-{num_cn}-s{s}",
            1,
            CNS[:num_cn],
            f"./bins/main-init-g4.exe /data/bins/refined-graph/laion10m.vamana 64g {MN_CFG.replace(","," ")}",
            f"./bench-scale-g4.exe 10000000 ./nfs/laion/base.fbin ./nfs/laion/query.fbin ./nfs/laion/gt.1-maxM.u32bin {THIS_IP}:23333 768,32,3522045,68719476736,2025,{MN_CFG} {LOCK_CFG} {{rank}} {num_cn} -t 16 -c 2 -b $(cat cpu.list) -s {s}",
        )
        test_plan.append(tc)

run_all(test_plan, remote_log_dir="./logs", print_only=False)
```
