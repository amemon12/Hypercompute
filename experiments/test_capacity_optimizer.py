"""Behavioral checks for the offline research model; no engine, GPU or network activity."""

import itertools
import unittest
from dataclasses import replace

from capacity_optimizer import GIB, Device, Model, optimize, segment, switch_decision


def device(name, capacity, speed=20, **kwargs):
    return Device(name, name, int(capacity * GIB), 0, speed, **kwargs)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.model = Model((GIB,) * 8, GIB // 4, GIB // 2, GIB // 16, 32768)
        self.cpu = device("cpu", 3)
        self.gpu = device("cuda", 3, 100)
        self.peer = device("mac", 3, 50, rtt_ms=5, link_mib_s=100)

    def test_added_peer_removes_paging_and_reduced_capacity_restores_it(self):
        before = optimize(self.model, self.cpu, self.gpu, [])
        after = optimize(self.model, self.cpu, self.gpu, [replace(self.peer, capacity_bytes=5 * GIB)])
        assert before and after
        self.assertGreater(sum(a.disk_read_bytes_high for a in before.assignments), 0)
        self.assertEqual(sum(a.disk_read_bytes_high for a in after.assignments), 0)
        self.assertLess(after.score_s, before.score_s)
        smaller = replace(self.peer, capacity_bytes=GIB)
        reduced = optimize(self.model, self.cpu, self.gpu, [smaller])
        assert reduced
        self.assertGreater(sum(a.disk_read_bytes_high for a in reduced.assignments), 0)

    def test_cpu_paging_is_explicitly_opt_in(self):
        self.assertIsNone(optimize(self.model, self.cpu, self.gpu, [], allow_disk=False))

    def test_kv_workspace_and_output_head_are_charged(self):
        too_small = device("gpu", 1)
        self.assertIsNone(segment(self.model, too_small, 7, 8, output=True))
        fits = segment(self.model, device("gpu", 2), 7, 8, output=True)
        assert fits
        self.assertEqual(fits.resident_bytes_est, GIB + GIB // 2 + GIB // 16)
        reserved = replace(device("gpu", 2), workspace_bytes=GIB)
        self.assertIsNone(segment(self.model, reserved, 7, 8, output=True))

    def test_uma_cannot_be_counted_twice(self):
        peer = replace(self.peer, pool=self.cpu.pool)
        with self.assertRaisesRegex(ValueError, "double-count UMA"):
            optimize(self.model, self.cpu, self.gpu, [peer])

    def test_upload_cache_never_changes_decode_cost(self):
        cold = segment(self.model, self.peer, 0, 2)
        warm = segment(self.model, replace(self.peer, cached_fraction=1), 0, 2)
        assert cold and warm
        self.assertEqual(cold.decode_s_high, warm.decode_s_high)
        self.assertGreater(cold.cold_upload_s, 0)
        self.assertEqual(warm.cold_upload_s, 0)

    def test_os_scan_miss_cost_is_not_only_overflow(self):
        paged = segment(self.model, self.cpu, 0, 5, cpu=True)
        managed = segment(self.model, self.cpu, 0, 5, cpu=True, cache_mode="fixed_resident")
        assert paged and managed
        self.assertGreater(paged.disk_read_bytes_high, paged.disk_read_bytes_low)
        self.assertEqual(paged.disk_read_bytes_high, 5 * GIB)
        self.assertEqual(managed.disk_read_bytes_high, managed.disk_read_bytes_low)

    def test_unhelpful_slow_peer_can_be_skipped(self):
        cpu = replace(self.cpu, capacity_bytes=20 * GIB)
        slow = replace(self.peer, rtt_ms=10000)
        alone = optimize(self.model, cpu, self.gpu, [])
        with_peer = optimize(self.model, cpu, self.gpu, [slow])
        assert alone and with_peer
        self.assertEqual(alone.score_s, with_peer.score_s)
        self.assertNotIn(slow.name, [a.device for a in with_peer.assignments])

    def test_cold_upload_and_horizon_change_the_decision(self):
        cpu = replace(self.cpu, capacity_bytes=20 * GIB)
        remote = replace(self.peer, link_mib_s=1, effective_read_gib_s=500)
        steady = optimize(self.model, cpu, self.gpu, [remote], expected_decode_tokens=128)
        cold = optimize(self.model, cpu, self.gpu, [remote], expected_decode_tokens=1, include_cold_upload=True)
        assert steady and cold
        self.assertIn(remote.name, [a.device for a in steady.assignments])
        self.assertNotIn(remote.name, [a.device for a in cold.assignments])

    def test_result_has_complete_nonoverlapping_coverage_and_admits_memory(self):
        result = optimize(self.model, self.cpu, self.gpu, [self.peer])
        assert result
        indices = [i for a in result.assignments for i in range(a.start_layer, a.end_layer)]
        self.assertEqual(indices, list(range(8)))
        budgets = {d.name: d.capacity_bytes for d in [self.cpu, self.gpu, self.peer]}
        self.assertTrue(all(a.resident_bytes_est <= budgets[a.device] for a in result.assignments))
        self.assertEqual(result.assignments[-1].device, "cuda")
        self.assertFalse(result.report()["can_apply_to_running_engine"])

    def test_dp_matches_independent_exhaustive_small_partition_search(self):
        model = Model(tuple(x * GIB // 5 for x in [2, 1, 3, 1, 2]), 0, GIB // 3, 17, 32768)
        peers = [device("m1", 1, 30, rtt_ms=2, link_mib_s=100),
                 device("m2", 1, 60, rtt_ms=5, link_mib_s=200)]
        cpu, gpu = device("c", 1, 10), device("g", 1, 100)
        actual = optimize(model, cpu, gpu, peers, expected_decode_tokens=1)
        assert actual
        best = float("inf")
        # Independent oracle enumerates all layer counts, including unused peers, rather than DP states.
        for order in itertools.permutations(peers):
            for counts in itertools.product(range(6), repeat=4):
                if sum(counts) != 5:
                    continue
                cursor, cost, valid = 0, 0.0, True
                for index, (d, count) in enumerate(zip([cpu, *order, gpu], counts, strict=True)):
                    if index in (1, 2) and count == 0:
                        continue
                    a = segment(model, d, cursor, cursor + count, cpu=index == 0, output=index == 3)
                    if a is None:
                        valid = False
                        break
                    cursor += count
                    cost += a.decode_s_high
                if valid:
                    best = min(best, cost)
        self.assertAlmostEqual(actual.score_s, best)

    def test_invalid_profiles_rejected(self):
        for speed in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                device("bad", 1, speed)
        with self.assertRaises(ValueError):
            optimize(self.model, self.cpu, self.gpu, [], disk_gib_s=0)


class ControllerTests(unittest.TestCase):
    def decision(self, **updates):
        inputs = dict(old_decode_s=8, new_decode_s=0.5, remaining_tokens=128,
                      reload_and_replay_s=100, idle=True, stable_samples=3)
        return switch_decision(**{**inputs, **updates})

    def test_waits_for_stable_resources_and_request_boundary(self):
        self.assertEqual(self.decision(stable_samples=1)["action"], "observe")
        self.assertEqual(self.decision(idle=False)["action"], "defer")
        self.assertEqual(self.decision()["action"], "prepare_replacement")

    def test_reload_cost_and_hysteresis_prevent_unhelpful_switches(self):
        self.assertEqual(self.decision(reload_and_replay_s=1000)["action"], "keep")
        self.assertEqual(self.decision(new_decode_s=7.5)["action"], "keep")
        self.assertEqual(self.decision(new_decode_s=9)["action"], "keep")


if __name__ == "__main__":
    unittest.main()
