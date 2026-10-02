#!/usr/bin/env python3
"""Tests for zero_priority.py: architecture gate, idle check, and yielding."""
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import zero_priority as zp

A100 = {"name": "NVIDIA A100-SXM4-80GB", "compute_cap": "8.0"}


class FakeNode:
    def __init__(self, gpus=None):
        self._gpus = gpus or [dict(A100, index=i) for i in range(4)]
        self.processes = []
        self.status = [{"gpu_id": i, "status": "AVAILABLE"} for i in range(len(self._gpus))]
        self.queue = {"entries": [], "total_waiting": 0}

    def gpus(self):
        return self._gpus

    def gpu_process_users(self):
        return list(self.processes)

    def chg_status(self):
        return self.status

    def chg_queue(self):
        return self.queue


def local_popen(cmd, env=None, start_new_session=False):
    """Strip the `chg run ... --` prefix and run the real command."""
    real = cmd[cmd.index("--") + 1:]
    return subprocess.Popen(real, env=env, start_new_session=start_new_session)


class ZeroPriorityTest(unittest.TestCase):
    def run_zp(self, node, command, on_poll=None):
        polls = []

        def sleep(_):
            polls.append(1)
            if on_poll:
                on_poll(len(polls))
            time.sleep(0.05)

        with mock.patch.object(zp, "release") as release:
            rc = zp.run(["--poll", "0", "--"] + command, node=node, popen=local_popen, sleep=sleep, me="me")
        return rc, release

    def test_refuses_hopper_and_newer(self):
        for name, cap in [("NVIDIA H100 80GB HBM3", "9.0"), ("NVIDIA H200", "9.0"), ("NVIDIA B200", "10.0")]:
            node = FakeNode([{"index": 0, "name": name, "compute_cap": cap}])
            rc, _ = self.run_zp(node, ["true"])
            self.assertEqual(rc, zp.EXIT_REFUSED, name)

    def test_refuses_mixed_node(self):
        node = FakeNode([dict(A100, index=0), {"index": 1, "name": "NVIDIA H100", "compute_cap": "9.0"}])
        self.assertEqual(self.run_zp(node, ["true"])[0], zp.EXIT_REFUSED)

    def test_refuses_when_node_busy(self):
        busy = [
            lambda n: n.processes.append((123, "alice")),
            lambda n: n.status.__setitem__(1, {"gpu_id": 1, "status": "IN_USE", "type": "run", "user": "alice"}),
            lambda n: n.status.__setitem__(2, {"gpu_id": 2, "status": "IN_USE", "type": "manual", "user": "bob"}),
            lambda n: n.queue["entries"].append({"user": "carol", "actual_user": "carol"}),
            lambda n: n.status.__setitem__(3, {"gpu_id": 3, "status": "UNRESERVED", "unreserved_users": ["dave"]}),
        ]
        for make_busy in busy:
            node = FakeNode()
            make_busy(node)
            self.assertEqual(self.run_zp(node, ["true"])[0], zp.EXIT_REFUSED)

    def test_runs_to_completion_when_idle(self):
        node = FakeNode()
        rc, release = self.run_zp(node, ["sh", "-c", "exit 7"])
        self.assertEqual(rc, 7)
        release.assert_not_called()

    def test_ignores_own_processes_and_reservation(self):
        node = FakeNode()
        node.processes.append((99, "me"))
        node.status[0] = {"gpu_id": 0, "status": "IN_USE", "type": "run", "user": "me"}
        rc, _ = self.run_zp(node, ["true"])
        self.assertEqual(rc, 0)

    def yield_case(self, intrude):
        node = FakeNode()
        marker = f"/tmp/zp_test_{os.getpid()}_{time.time()}"

        def on_poll(n):
            if n == 3:
                intrude(node)

        # Child and grandchild must both die; the grandchild would create the marker
        cmd = ["sh", "-c", f"(sleep 2; touch {marker}) & sleep 30"]
        t0 = time.time()
        rc, release = self.run_zp(node, cmd, on_poll)
        self.assertEqual(rc, zp.EXIT_YIELDED)
        self.assertLess(time.time() - t0, 10)
        release.assert_called_once_with(0)
        time.sleep(2.5)
        self.assertFalse(os.path.exists(marker), "process group survived the yield")

    def test_yields_to_other_users_process(self):
        self.yield_case(lambda n: n.processes.append((4242, "alice")))

    def test_yields_to_other_users_reservation(self):
        self.yield_case(lambda n: n.status.__setitem__(3, {"gpu_id": 3, "status": "IN_USE", "type": "run", "user": "alice"}))

    def test_yields_to_queued_request(self):
        self.yield_case(lambda n: n.queue.update(entries=[{"user": "bob", "actual_user": "bob"}], total_waiting=1))


if __name__ == "__main__":
    unittest.main()
