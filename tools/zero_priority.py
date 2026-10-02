#!/usr/bin/env python3
"""Run a command on a shared GPU node at zero priority.

The job runs only on Ampere A100 GPUs (compute capability 8.0); Hopper and
newer are refused outright. It starts only when nobody else is using,
reserving, or waiting for any GPU on the node, and it yields immediately if
anyone else appears: when another user's process shows up on any GPU, when
anyone else holds a canhazgpu reservation, or when anyone is waiting in the
canhazgpu queue, the whole job process group is killed and its reservation
released, so the other user gets the GPU.

Usage (on the GPU node):
    zero_priority.py [--poll SECONDS] [--log FILE] -- COMMAND [ARGS...]

Exit codes:
    0   command finished
    2   refused to start (wrong GPU architecture or node not idle)
    3   yielded to another user (command was killed)
    other: the command's own exit code
"""

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time

REQUIRED_COMPUTE_CAP = "8.0"
REQUIRED_NAME = "A100"
EXIT_REFUSED = 2
EXIT_YIELDED = 3


def sh(cmd):
    """Run a shell command and return stdout ('' on failure)."""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


class Node:
    """Queries the node. Tests substitute a fake with the same methods."""

    def gpus(self):
        out = sh("nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader")
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                rows.append({"index": int(parts[0]), "name": parts[1], "compute_cap": parts[2]})
        return rows

    def gpu_process_users(self):
        """Owners of all compute processes on all GPUs."""
        out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
        users = []
        for line in out.split():
            pid = line.strip()
            if pid.isdigit():
                owner = sh(f"ps -o user= -p {pid}").strip()
                users.append((int(pid), owner or "?"))
        return users

    def chg_status(self):
        try:
            return json.loads(sh("chg status --json") or "[]")
        except json.JSONDecodeError:
            return None

    def chg_queue(self):
        try:
            return json.loads(sh("chg queue --json") or "{}")
        except json.JSONDecodeError:
            return None


def architecture_ok(gpus):
    if not gpus:
        return False, "no NVIDIA GPUs visible"
    for g in gpus:
        if REQUIRED_NAME not in g["name"] or g["compute_cap"] != REQUIRED_COMPUTE_CAP:
            return False, f"GPU {g['index']} is {g['name']} (sm {g['compute_cap']}); only A100 (sm 8.0) is allowed"
    return True, ""


def others_present(node, me):
    """Return a reason string if anyone other than `me` is using, holding or
    waiting for any GPU on the node; '' otherwise."""
    for pid, owner in node.gpu_process_users():
        if owner != me:
            return f"process {pid} of user {owner} on a GPU"
    status = node.chg_status()
    if status is None:
        return "chg status unavailable"
    for entry in status:
        # chg status --json: "status" is AVAILABLE, IN_USE, UNRESERVED or ERROR;
        # "type" is the reservation kind (run, manual)
        user = entry.get("user") or ""
        state = str(entry.get("status", "")).upper()
        if state == "IN_USE" and user != me:
            return f"GPU {entry.get('gpu_id')} reserved by {user or 'someone'}"
        if state == "UNRESERVED":
            return f"GPU {entry.get('gpu_id')} used without reservation by {entry.get('unreserved_users')}"
    queue = node.chg_queue()
    if queue is None:
        return "chg queue unavailable"
    waiting = [e for e in queue.get("entries") or [] if (e.get("actual_user") or e.get("user")) != me]
    if waiting or (queue.get("total_waiting", 0) and not queue.get("entries")):
        return f"{max(len(waiting), queue.get('total_waiting', 0))} request(s) waiting in the chg queue"
    return ""


def pick_gpu(node, me):
    status = node.chg_status() or []
    free = sorted(e["gpu_id"] for e in status if str(e.get("status", "")).upper() == "AVAILABLE")
    return free[0] if free else None


def kill_group(proc, log):
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    log(f"killing process group {pgid}")
    for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    # Orphans (e.g. CUDA helper processes) that escaped the group
    sh(f"pkill -KILL -g {pgid}")


def release(gpu):
    sh(f"chg release --gpu-ids {gpu}")


def run(argv, node=None, popen=subprocess.Popen, sleep=time.sleep, me=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("no command given")
    node = node or Node()
    me = me or getpass.getuser()

    def log(msg):
        line = f"[zero-priority {time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        if args.log:
            with open(args.log, "a") as f:
                f.write(line + "\n")

    ok, why = architecture_ok(node.gpus())
    if not ok:
        log(f"refusing: {why}")
        return EXIT_REFUSED
    reason = others_present(node, me)
    if reason:
        log(f"refusing: node not idle ({reason})")
        return EXIT_REFUSED
    gpu = pick_gpu(node, me)
    if gpu is None:
        log("refusing: no AVAILABLE GPU in chg status")
        return EXIT_REFUSED

    log(f"starting on GPU {gpu}: {' '.join(command)}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    # chg holds the reservation for exactly the lifetime of the child
    proc = popen(["chg", "run", "--gpu-ids", str(gpu), "--"] + command, env=env, start_new_session=True)
    while True:
        rc = proc.poll()
        if rc is not None:
            log(f"command exited with {rc}")
            return rc
        reason = others_present(node, me)
        if reason:
            log(f"YIELDING: {reason}")
            kill_group(proc, log)
            release(gpu)
            return EXIT_YIELDED
        sleep(args.poll)


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
