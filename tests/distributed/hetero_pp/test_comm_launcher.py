# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard-library contracts for the two-node CPU/Gloo check launcher.

No Torch, vLLM package import, devices, SSH or real network interfaces are used.
The helper is loaded directly from its source path; socket tests use socketpair.
"""

import copy
import importlib.util
import json
import socket
import struct
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo.py"
PINS = {
    "verify_cpu_gloo.py": "a" * 64,
    "channel.py": "b" * 64,
    "cpu_staging.py": "c" * 64,
}


def load_helper() -> Any:
    if not SOURCE.is_file():
        raise AssertionError("the CPU/Gloo launcher helper is not implemented")
    spec = importlib.util.spec_from_file_location("comm_launcher_contract", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def peer(rank: int) -> dict[str, Any]:
    return {
        "node_rank": rank,
        "node_ip": "172.20.183.252" if rank == 0 else "172.20.183.225",
        "soc": "910B" if rank == 0 else "310P",
        "torch": "2.10.0+cpu",
        "byteorder": "little",
        "sources": dict(PINS),
    }


def receipt(rank: int) -> dict[str, Any]:
    receive_from = 3 if rank == 0 else rank - 1
    send_to = 0 if rank == 3 else rank + 1
    return {
        "rank": rank,
        "local_rank": rank % 2,
        "backend": "gloo",
        "destination_device": "npu",
        "passed": True,
        "checks": [
            {"M": size, "all_values_equal": True} for size in (1, 64, 256, 2048)
        ],
        "wire_calls": [
            {
                "device": "cpu",
                "backend": "gloo",
                "peer": destination,
                "operation": operation,
                "tag": 100,
                "bytes": 16,
                "dtype": "torch.uint8",
            }
            for operation, destination in (("irecv", receive_from), ("isend", send_to))
        ],
    }


def write_receipts(directory: Path, node_rank: int) -> None:
    # Write in reverse order so filesystem iteration cannot stand in for rank order.
    for rank in reversed(range(node_rank * 2, node_rank * 2 + 2)):
        (directory / f"cpu_gloo_rank{rank}.json").write_text(json.dumps(receipt(rank)))


@contextmanager
def socket_pair():
    left, right = socket.socketpair()
    try:
        yield left, right
    finally:
        left.close()
        right.close()


class LauncherCase(unittest.TestCase):
    helper: Any

    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper()


class NetworkTests(LauncherCase):
    def test_head_uses_own_address_interface_even_when_route_says_loopback(self):
        actual = self.helper.choose_network(
            "172.20.183.252",
            {"lo": ["127.0.0.1"], "eth0": ["172.20.183.252"]},
            "127.0.0.1",
        )
        self.assertEqual(
            actual,
            {
                "node_rank": 0,
                "node_ip": "172.20.183.252",
                "interface": "eth0",
                "head_ip": "172.20.183.252",
            },
        )

    def test_peer_uses_the_interface_owning_the_route_source(self):
        actual = self.helper.choose_network(
            "172.20.183.252",
            {"lo": ["127.0.0.1"], "eth0": ["172.20.183.225"], "eth1": ["10.2.0.2"]},
            "172.20.183.225",
        )
        self.assertEqual(
            actual,
            {
                "node_rank": 1,
                "node_ip": "172.20.183.225",
                "interface": "eth0",
                "head_ip": "172.20.183.252",
            },
        )

    def test_explicit_peer_interface_overrides_automatic_route_selection(self):
        actual = self.helper.choose_network(
            "172.20.183.252",
            {"eth0": ["172.20.183.225"], "eth1": ["10.2.0.2"]},
            "172.20.183.225",
            interface="eth1",
        )
        self.assertEqual(actual["node_rank"], 1)
        self.assertEqual((actual["node_ip"], actual["interface"]), ("10.2.0.2", "eth1"))

    def test_unusable_head_route_or_interface_is_rejected(self):
        interfaces = {"lo": ["127.0.0.1"], "eth0": ["172.20.183.225"]}
        for head, route, interface in [
            ("127.0.0.1", "172.20.183.225", None),
            ("172.20.183.252", "192.0.2.99", None),
            ("172.20.183.252", "172.20.183.225", "missing-interface"),
        ]:
            with (
                self.subTest(head=head, route=route, interface=interface),
                self.assertRaises(ValueError),
            ):
                self.helper.choose_network(head, interfaces, route, interface=interface)


class PeerTests(LauncherCase):
    def test_matching_base_torch_versions_allow_different_local_build_suffixes(self):
        local, remote = peer(0), peer(1)
        remote["torch"] = "2.10.0+ascend"
        self.assertIsNone(self.helper.validate_peer(local, remote))

    def test_homogeneous_soc_bringup_is_valid_without_claiming_mixed_hardware(self):
        local, remote = peer(0), peer(1)
        remote["soc"] = "910B"
        self.assertIsNone(self.helper.validate_peer(local, remote))

    def test_same_node_or_same_rank_is_rejected(self):
        for field, value in (("node_rank", 0), ("node_ip", "172.20.183.252")):
            with self.subTest(field=field):
                remote = peer(1)
                remote[field] = value
                with self.assertRaises(ValueError):
                    self.helper.validate_peer(peer(0), remote)

    def test_wire_and_source_incompatibilities_are_rejected(self):
        for field, value in [
            ("torch", "2.11.0+cpu"),
            ("byteorder", "big"),
            ("sources", {**PINS, "channel.py": "d" * 64}),
        ]:
            with self.subTest(field=field):
                remote = peer(1)
                remote[field] = value
                with self.assertRaises(ValueError):
                    self.helper.validate_peer(peer(0), remote)

    def test_unknown_or_missing_required_peer_metadata_is_rejected(self):
        for field, value in (("soc", "unknown"), ("torch", None), ("byteorder", None)):
            with self.subTest(field=field):
                remote = peer(1)
                remote[field] = value
                with self.assertRaises(ValueError):
                    self.helper.validate_peer(peer(0), remote)
        local, remote = peer(0), peer(1)
        local["sources"] = remote["sources"] = {}
        with self.assertRaises(ValueError):
            self.helper.validate_peer(local, remote)


class ReceiptTests(LauncherCase):
    def test_both_node_roles_accept_exactly_their_two_complete_rank_files(self):
        for node_rank in (0, 1):
            with (
                self.subTest(node_rank=node_rank),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp)
                write_receipts(path, node_rank)
                (path / "launcher.log").write_text("ordinary log is not another rank")
                rows = self.helper.validate_receipts(path, node_rank)
                self.assertEqual(
                    sorted(r["rank"] for r in rows),
                    list(range(node_rank * 2, node_rank * 2 + 2)),
                )

    def test_missing_or_extra_rank_receipts_are_rejected(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                write_receipts(path, 0)
                if missing:
                    (path / "cpu_gloo_rank1.json").unlink()
                else:
                    (path / "cpu_gloo_rank2.json").write_text(json.dumps(receipt(2)))
                with self.assertRaises(ValueError):
                    self.helper.validate_receipts(path, 0)

    def test_rank_content_and_local_rank_mapping_must_match_the_file(self):
        for field, value in (("rank", 3), ("local_rank", 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                write_receipts(path, 1)
                row = receipt(2)
                row[field] = value
                (path / "cpu_gloo_rank2.json").write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    self.helper.validate_receipts(path, 1)

    def test_all_four_payload_checks_are_required_and_must_be_true(self):
        invalid = [[], [{"M": 1, "all_values_equal": True}] * 4]
        bad = copy.deepcopy(receipt(0)["checks"])
        bad[2]["all_values_equal"] = "true"
        invalid.append(bad)
        bad = copy.deepcopy(receipt(0)["checks"])
        bad[3]["M"] = 9999
        invalid.append(bad)
        for checks in invalid:
            with self.subTest(checks=checks), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                write_receipts(path, 0)
                row = receipt(0)
                row["checks"] = checks
                (path / "cpu_gloo_rank0.json").write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    self.helper.validate_receipts(path, 0)

    def test_cpu_gloo_payload_boundaries_and_valid_edges_are_required(self):
        invalid: list[list[dict[str, Any]]] = [[]]
        for field, value in [
            ("device", "npu"),
            ("backend", "hccl"),
            ("peer", 99),
            ("operation", "broadcast"),
            ("tag", "100"),
            ("bytes", -1),
            ("dtype", None),
        ]:
            calls = copy.deepcopy(receipt(1)["wire_calls"])
            calls[0][field] = value
            invalid.append(calls)
        for calls in invalid:
            with self.subTest(calls=calls), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                write_receipts(path, 0)
                row = receipt(1)
                row["wire_calls"] = calls
                (path / "cpu_gloo_rank1.json").write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    self.helper.validate_receipts(path, 0)

    def test_truthy_success_or_wrong_destination_is_not_accepted(self):
        for field, value in [
            ("passed", False),
            ("passed", "true"),
            ("backend", "hccl"),
            ("destination_device", "cpu"),
        ]:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp)
                write_receipts(path, 0)
                row = receipt(0)
                row[field] = value
                (path / "cpu_gloo_rank0.json").write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    self.helper.validate_receipts(path, 0)


class FramingTests(LauncherCase):
    def test_multiple_json_messages_preserve_framing_and_unicode(self):
        with socket_pair() as (left, right):
            left.settimeout(1)
            right.settimeout(1)
            values = [
                {"node_rank": 0, "text": "通信✓"},
                {"passed": True, "value": None},
            ]
            for value in values:
                self.helper.send_message(left, value)
            self.assertEqual(
                [self.helper.receive_message(right) for _ in values], values
            )

    def test_zero_and_oversized_inbound_lengths_are_rejected_before_body_reads(self):
        for length in (0, 65537):
            with self.subTest(length=length), socket_pair() as (left, right):
                left.settimeout(1)
                right.settimeout(0.5)
                left.sendall(struct.pack("!I", length))
                with self.assertRaises(ValueError):
                    self.helper.receive_message(right)

    def test_oversized_outbound_json_is_rejected(self):
        with socket_pair() as (left, right):
            left.settimeout(1)
            right.settimeout(1)
            with self.assertRaises(ValueError):
                self.helper.send_message(left, {"oversized": "x" * 65536})


if __name__ == "__main__":
    unittest.main()
