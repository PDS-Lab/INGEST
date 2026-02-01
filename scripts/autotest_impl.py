from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fabric import ThreadingGroup as Group

# 当 stdout/stderr 被重定向到管道/文件（例如 tee）时，Python 默认可能块缓冲，导致日志“憋很久才刷”。
# 这里在 import 时尽量启用行缓冲/写穿透，确保 print() 实时落盘。
def _ensure_unbuffered_stdio() -> None:
    import sys

    for s in (sys.stdout, sys.stderr):
        try:
            # TextIOWrapper.reconfigure (Py3.7+)
            if hasattr(s, "reconfigure"):
                s.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            # 某些环境下（例如被替换的 IO 对象）可能不支持 reconfigure；忽略即可
            pass


_ensure_unbuffered_stdio()

# barrier 服务监听地址（通常 0.0.0.0 即可）
BARRIER_LISTEN_HOST = "0.0.0.0"

DEFAULT_BARRIER_PORT = 23333
DEFAULT_WORKDIR = "00-test"
DEFAULT_LOG_DIR = "./autorun_res"

# 轮询间隔（秒）
POLL_INTERVAL_S = 10


@dataclass(frozen=True)
class TestCase:
    name: str
    repeat: int
    cns: list[str]
    init_cmd: str
    bench_cmd: str

###############################################################################
# 实现区
###############################################################################

def _sh_single_quote(s: str) -> str:
    # 安全地把字符串作为一个 shell 参数（单引号包裹）
    return "'" + s.replace("'", "'\"'\"'") + "'"

def _fmt_dur(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h:
        return f"{h}h{m:02d}m{ss:02d}s"
    if m:
        return f"{m}m{ss:02d}s"
    return f"{ss}s"


def _run_local_init(init_cmd: str) -> str:
    p = subprocess.run(
        init_cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class BarrierServer(threading.Thread):
    """
    Python 版 barrier-master.cxx：
    - 接受 num_workers 个 TCP 连接
    - 每一轮等待每个连接发送 1 byte
    - 收齐后给每个连接回写 1 byte
    """

    def __init__(self, host: str, port: int, num_workers: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event()
        self._exc: Optional[BaseException] = None

    def stop(self) -> None:
        self._stop_evt.set()

    def wait_ready(self, timeout_s: float = 10.0) -> bool:
        return self._ready_evt.wait(timeout=timeout_s)

    def raise_if_failed(self) -> None:
        if self._exc is not None:
            raise self._exc

    def run(self) -> None:
        import socket

        listen_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listen_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_fd.bind((self.host, self.port))
            listen_fd.listen(self.num_workers)
            listen_fd.settimeout(1.0)
            self._ready_evt.set()

            clients: list[socket.socket] = []
            while len(clients) < self.num_workers and not self._stop_evt.is_set():
                try:
                    c, _ = listen_fd.accept()
                except socket.timeout:
                    continue
                c.settimeout(1.0)
                clients.append(c)
                print(
                    f"[barrier] worker connected: {len(clients) - 1}/{self.num_workers}"
                )

            if len(clients) < self.num_workers:
                return

            print("[barrier] all workers connected, start barrier loop")
            got = [False] * self.num_workers
            while not self._stop_evt.is_set():
                # 等待所有 worker 到达（每个 socket 收到 1 byte）
                for i, c in enumerate(clients):
                    if got[i]:
                        continue
                    try:
                        b = c.recv(1)
                        if not b:
                            # worker 正常退出/关闭连接：不视为错误，直接结束 barrier
                            print("[barrier] worker disconnected, stop barrier loop")
                            return
                        got[i] = True
                    except socket.timeout:
                        continue

                if not all(got):
                    continue

                # 广播继续信号（写 1 byte）
                for c in clients:
                    c.sendall(b"\x01")
                got = [False] * self.num_workers

        except BaseException as e:
            self._exc = e
        finally:
            try:
                for c in clients:
                    try:
                        c.close()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                listen_fd.close()
            except Exception:
                pass


def _remote_paths_for_test(test_name: str, rank: str, *, remote_log_dir: str) -> tuple[str, str]:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in test_name)
    rs = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in rank)
    log_path = f"{remote_log_dir}/{safe}-{rs}.log"
    pid_path = f"{remote_log_dir}/{safe}-{rs}.pid"
    return log_path, pid_path


def _replace_rank(cmd: str, rank: str) -> str:
    return cmd.replace("{rank}", rank)

def _build_remote_launcher(
    bench_cmd: str,
    test_name: str,
    rank: str,
    *,
    remote_workdir: str,
    remote_log_dir: str,
) -> tuple[str, str, str]:
    """
    返回 (launcher_cmd, log_path, pid_path)
    launcher_cmd 用于在远端 shell 执行（不包含 ssh）。
    """
    log_path, pid_path = _remote_paths_for_test(test_name, rank, remote_log_dir=remote_log_dir)
    bench_quoted = _sh_single_quote(_replace_rank(bench_cmd, rank))
    inner = (
        "if command -v setsid >/dev/null 2>&1; then "
        f"  setsid nohup bash -lc {bench_quoted} < /dev/null > {log_path} 2>&1 & "
        "else "
        f"  nohup bash -lc {bench_quoted} < /dev/null > {log_path} 2>&1 & "
        "fi; "
        f"pid=$!; echo $pid > {pid_path}"
    )
    launcher = (
        f"cd {remote_workdir} && "
        f"mkdir -p {remote_log_dir} && "
        f"bash -lc {_sh_single_quote(inner)}"
    )
    return launcher, log_path, pid_path


def _start_remote_bench(
    g: Group,
    bench_cmd: str,
    test_name: str,
    *,
    remote_workdir: str,
    remote_log_dir: str,
) -> None:
    # 按 CNS 顺序分配 rank：0..N-1（不读取远端 rank 文件）
    for rank, conn in enumerate(g):
        rk = str(rank)
        launcher, _, _ = _build_remote_launcher(
            bench_cmd,
            test_name,
            rk,
            remote_workdir=remote_workdir,
            remote_log_dir=remote_log_dir,
        )
        conn.run(launcher, hide=False, pty=False)


def _all_remote_done(
    g: Group,
    test_name: str,
    *,
    remote_workdir: str,
    remote_log_dir: str,
) -> bool:
    res_map = {}
    for rank, conn in enumerate(g):
        rk = str(rank)
        _, pid_path = _remote_paths_for_test(test_name, rk, remote_log_dir=remote_log_dir)
        check = (
            f"cd {remote_workdir} && "
            f"if [ -f {pid_path} ]; then "
            f"  pid=$(cat {pid_path}); "
            f"  if kill -0 $pid 2>/dev/null; then "
            f"    st=$(ps -o stat= -p $pid 2>/dev/null | tr -d ' \\n\\t'); "
            f"    if [ -z \"$st\" ]; then echo DONE; "
            f"    elif echo \"$st\" | grep -q Z; then echo DONE; "
            f"    else echo RUNNING; fi; "
            f"  else echo DONE; fi; "
            f"else echo NO_PID; fi"
        )
        res_map[conn] = conn.run(check, hide=True, warn=True, pty=False)
    states = []
    for conn, r in res_map.items():
        st = (
            (r.stdout or "").strip().splitlines()[-1]
            if (r.stdout or "").strip()
            else "UNKNOWN"
        )
        states.append((conn.host, st))
    print("[poll] " + ", ".join([f"{h}:{s}" for h, s in states]))
    return all(s in ("DONE", "NO_PID") for _, s in states)


def run_test(
    tc: TestCase,
    rep: int,
    *,
    barrier_port: int=DEFAULT_BARRIER_PORT,
    remote_workdir: str=DEFAULT_WORKDIR,
    remote_log_dir: str=DEFAULT_LOG_DIR,
    password: Optional[str] = None,
    print_only: bool = False,
) -> None:
    t0 = time.monotonic()
    tag = f"{tc.name}-r{rep}"
    print(f"\n=== [start] {tag} ===")

    # password 为空（None 或 ""）则读取环境变量 PASSWORD；若仍为空，则不传 password（支持 ssh key）
    pw = password or os.getenv("PASSWORD") or ""
    connect_kwargs = {"password": pw} if pw else {}
    g = Group(*tc.cns, connect_kwargs=connect_kwargs)
    num_workers = len(tc.cns)

    init_cmd = tc.init_cmd
    bench_cmd = tc.bench_cmd

    if print_only:
        print("[print-only] local init_cmd:\n" + (init_cmd or "<empty>"))
        print("[print-only] remote bench launchers:")
        for rank, cn in enumerate(tc.cns):
            rk = str(rank)
            launcher, log_path, pid_path = _build_remote_launcher(
                bench_cmd,
                tag,
                rk,
                remote_workdir=remote_workdir,
                remote_log_dir=remote_log_dir,
            )
            print(f"- host={cn} rank={rk}")
            print(f"  log: {remote_workdir}/{log_path}")
            print(f"  pid: {remote_workdir}/{pid_path}")
            print(f"  cmd: {launcher}")
        print(
            f"=== [done] {tag} (print-only, elapsed={_fmt_dur(time.monotonic() - t0)}) ==="
        )
        return

    if not init_cmd:
        raise ValueError(f"{tag}: init_cmd 为空，请在 TEST_PLAN 里填写")
    if not bench_cmd:
        raise ValueError(f"{tag}: bench_cmd 为空，请在 TEST_PLAN 里填写（可包含 '{{rank}}'）")

    print("[local] running init:", init_cmd)
    t_init0 = time.monotonic()
    init_cfg_raw = _run_local_init(init_cmd)
    t_init1 = time.monotonic()
    if init_cfg_raw:
        print("[local] init stdout last line (INIT_CFG):", init_cfg_raw)
    else:
        print("[local] init stdout last line (INIT_CFG): <empty>")
    print(f"[time] init: {_fmt_dur(t_init1 - t_init0)}")

    barrier = None
    try:
        t_bar0 = time.monotonic()
        barrier = BarrierServer(BARRIER_LISTEN_HOST, barrier_port, num_workers=num_workers)
        barrier.start()
        if not barrier.wait_ready(timeout_s=10.0):
            raise RuntimeError("barrier 未能按时进入监听状态")
        print(f"[time] barrier start: {_fmt_dur(time.monotonic() - t_bar0)}")

        # bench_cmd 中的 {rank} 会在 launcher 构造时按 CNS 顺序 0..N-1 替换
        # 注意：这里不使用 .format，避免命令里其他花括号（若存在）触发格式化
        print("[remote] starting bench via nohup")
        t_launch0 = time.monotonic()
        _start_remote_bench(
            g,
            bench_cmd=bench_cmd,
            test_name=tag,
            remote_workdir=remote_workdir,
            remote_log_dir=remote_log_dir,
        )
        print(f"[time] remote launch: {_fmt_dur(time.monotonic() - t_launch0)}")

        # 不持续轮询：先等待 socket/barrier 自然断开（通常意味着程序跑完或退出）
        print("[wait] waiting for barrier/socket disconnect...")
        t_wait0 = time.monotonic()
        while barrier.is_alive():
            barrier.raise_if_failed()
            barrier.join(timeout=1.0)
        print(f"[time] wait barrier: {_fmt_dur(time.monotonic() - t_wait0)}")

        # socket 断开后再检查是否全部结束；若仍有残留进程，再进入低频轮询收尾
        print("[check] barrier disconnected, checking remote processes...")
        t_tail0 = time.monotonic()
        while True:
            if _all_remote_done(
                g,
                test_name=tag,
                remote_workdir=remote_workdir,
                remote_log_dir=remote_log_dir,
            ):
                break
            time.sleep(POLL_INTERVAL_S)
        print(f"[time] tail check: {_fmt_dur(time.monotonic() - t_tail0)}")

    finally:
        if barrier is not None:
            barrier.stop()

    print(f"=== [done] {tag} (elapsed={_fmt_dur(time.monotonic() - t0)}) ===")


def run_all(
    test_plan: list[TestCase],
    *,
    barrier_port: int = DEFAULT_BARRIER_PORT,
    remote_workdir: str = DEFAULT_WORKDIR,
    remote_log_dir: str = DEFAULT_LOG_DIR,
    password: str = "",
    print_only: bool = False,
) -> None:
    t_all0 = time.monotonic()
    for tc in test_plan:
        for rep in range(tc.repeat):
            run_test(
                tc,
                rep,
                barrier_port=barrier_port,
                remote_workdir=remote_workdir,
                remote_log_dir=remote_log_dir,
                password=password,
                print_only=print_only,
            )

    print(f"\n=== [all done] elapsed={_fmt_dur(time.monotonic() - t_all0)} ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--print-only",
        action="store_true",
        help="仅输出每个 testcase 的 init_cmd/远端启动命令，不执行",
    )
    args = ap.parse_args()
    run_all(test_plan=[], print_only=args.print_only)
