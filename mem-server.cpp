#include <myutils/io.h>

#include <lcoro/rdma_worker.h>

#include <argparse/argparse.hpp>

// Parse size string like "1.5G", "512M", "1024K" to bytes
size_t parse_size_string(const std::string &size_str) {
  if (size_str.empty())
    throw std::invalid_argument("Empty size string");

  size_t pos = 0;
  while (pos < size_str.length() &&
         (std::isdigit(size_str[pos]) || size_str[pos] == '.'))
    ++pos;

  double value = std::stod(size_str.substr(0, pos));

  if (pos < size_str.length()) {
    char unit = std::tolower(size_str[pos]);
    switch (unit) {
    case 'k':
      value *= 1024;
      break;
    case 'm':
      value *= 1024 * 1024;
      break;
    case 'g':
      value *= 1024ULL * 1024 * 1024;
      break;
    default:
      throw std::invalid_argument("Invalid size unit: " +
                                  std::string(1, size_str[pos]));
    }
  }

  return static_cast<size_t>(value);
}

// uid:size[:path]
auto parse_memcfg(const std::string &memcfg) {
  size_t first_colon = memcfg.find(':');
  if (first_colon == std::string::npos)
    throw std::invalid_argument("Invalid memcfg: " + memcfg);

  std::string uid = memcfg.substr(0, first_colon);
  std::string remaining = memcfg.substr(first_colon + 1);

  size_t second_colon = remaining.find(':');
  std::string size_str;
  std::string path;

  if (second_colon == std::string::npos) {
    size_str = remaining;
    path = "";
  } else {
    size_str = remaining.substr(0, second_colon);
    path = remaining.substr(second_colon + 1);
  }

  size_t size = parse_size_string(size_str);

  return std::make_tuple(std::stoul(uid), size, path);
}

using namespace rdma;
int main(int argc, char **argv) {
  constexpr size_t lock_mem_size = 65536;
  std::string dev_name;
  int rdma_dev_port = 1;
  int rdma_gid_index = 1;
  int port = default_listen_port;
  bool unlink_path = false;
  std::string shm_prefix;
  size_t lock_mem_id = 0;
  size_t meta_mem_id = 0;
  std::vector<std::string> memcfgs;
  argparse::ArgumentParser program("server");
  program.add_argument("-d", "--dev-name")
      .default_value(dev_name)
      .store_into(dev_name);
  program.add_argument("-r", "--rdma-dev-port")
      .default_value(rdma_dev_port)
      .store_into(rdma_dev_port);
  program.add_argument("-g", "--rdma-gid-index")
      .default_value(rdma_gid_index)
      .store_into(rdma_gid_index);
  program.add_argument("-p", "--port").default_value(port).store_into(port);
  program.add_argument("-s", "--shm-prefix")
      .default_value(shm_prefix)
      .store_into(shm_prefix);
  program.add_argument("-u", "--unlink-path")
      .implicit_value(true)
      .store_into(unlink_path);
  program.add_argument("-l", "--lock-mem-id")
      .default_value(lock_mem_id)
      .store_into(lock_mem_id);
  program.add_argument("-m", "--meta-mem-id")
      .default_value(meta_mem_id)
      .store_into(meta_mem_id);
  program.add_argument("mem_cfgs")
      .nargs(argparse::nargs_pattern::at_least_one)
      .store_into(memcfgs);
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  println("dev_name: {}, port: {}, shm_prefix: {}, unlink_path: {}", dev_name,
          port, shm_prefix, unlink_path);
  Device dev(dev_name, rdma_gid_index, rdma_dev_port);
  if (lock_mem_id != 0) {
    auto mr = dev.reg_dmmr(lock_mem_id, lock_mem_size);
    char zero_buf[lock_mem_size] = {0};
    ibv_memcpy_to_dm((ibv_dm *)mr->addr, 0, zero_buf, lock_mem_size);
    println("registered lock mem id: {}, size: {}", lock_mem_id, lock_mem_size);
  }
  if (meta_mem_id != 0) {
    auto mr = dev.reg_dmmr(meta_mem_id, 64);
    char zero_buf[64] = {0};
    ibv_memcpy_to_dm((ibv_dm *)mr->addr, 0, zero_buf, 64);
    println("registered meta mem id: {}, size: {}", meta_mem_id, 64);
  }
  std::vector<Mem> mems;
  for (size_t i = 0; i < memcfgs.size(); ++i) {
    auto [uid, siz, path] = parse_memcfg(memcfgs[i]);
    auto mem = mems.emplace_back(alloc_shm(
        shm_prefix.empty() ? "" : std::format("{}_{}", shm_prefix, uid), siz));
    println("registering MR {} with size {}", uid, siz);
    if (!path.empty()) {
      MMap mmap(path);
      size_t size = mmap.size_;
      void *ptr = mmap.ptr<void>();
      expect_le(mmap.size_, mem.size);
      madvise(ptr, size, MADV_SEQUENTIAL);
      memcpy(mem.ptr, ptr, size);
    }
    auto tmp = dev.create_mr(mem.size, mem.ptr);
    dev.reg_mr(uid, tmp);
  }
  ibv_mr mr;
  Server ser(dev);
  ser.start_serve(port);
  println("server started");
  getchar();
  println("stopping server");
  ser.stop_serve();
  println("server stopped");
  for (auto &mem : mems)
    free_shm(mem, unlink_path);
  return 0;
}