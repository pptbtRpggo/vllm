# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

from vllm.v1.core.sched.tau_batch.types import MicroBatchList


class DispatchPolicy(Enum):
    """When Decode slots may be offered after Prefill fill.

    OVERLAP (default): a micro-batch may Decode as soon as its own Prefill
    has completed, even if later batches are still prefilling.
    DRAIN: no Decode until every micro-batch Prefill has completed.
    """

    OVERLAP = "overlap"
    DRAIN = "drain"


class DispatchPhase(Enum):
    """Work kind for one schedule() slot."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class DispatchSlot:
    """One micro-batch to expose to a single schedule() call.

    Attributes:
        microbatch_index: Index into MicroBatchList.tasks.
        phase: Prefill the whole prompt, or decode one step.
    """

    microbatch_index: int
    phase: DispatchPhase


class ListDispatcher:
    """Issues ordered Prefill then cyclic Decode for one MicroBatchList.

    Each task is issued first as n prefills, then as n decodes. A wave id
    is not stored here; the scheduler stamps that label at dispatch start.

    peek_slot does not advance. commit_slot advances only after the caller
    actually scheduled that slot. on_prefill_complete records that a
    committed Prefill has finished (update_from_output), which is the
    Decode gate.

    Args:
        policy: OVERLAP or DRAIN. Defaults to OVERLAP.
    """

    def __init__(
        self, policy: DispatchPolicy = DispatchPolicy.OVERLAP
    ) -> None:
        self.policy = policy
        self._list: MicroBatchList | None = None
        self._prefills_committed = 0
        self._prefill_done: set[int] = set()
        self._decode_cursor = 0

    def start(self, packed: MicroBatchList) -> None:
        """Begin dispatching ``packed``. Replaces any previous list.

        Args:
            packed: Non-empty micro-batch-task list.
        """
        if not packed.tasks:
            raise ValueError("MicroBatchList.tasks must be non-empty")
        self._list = packed
        self._prefills_committed = 0
        self._prefill_done = set()
        self._decode_cursor = 0

    def reset(self) -> None:
        """Drop the active list. peek_slot() is None until the next start()."""
        self._list = None
        self._prefills_committed = 0
        self._prefill_done = set()
        self._decode_cursor = 0

    @property
    def active_list(self) -> MicroBatchList | None:
        return self._list

    def peek_slot(self) -> DispatchSlot | None:
        """Return the next slot without advancing.

        Returns:
            The slot to hold back to, or None if the caller must wait for
            on_prefill_complete (not the end of the list). None if start()
            has not been called.
        """
        if self._list is None:
            return None
        p = len(self._list.tasks)
        if self._prefills_committed < p:
            return DispatchSlot(self._prefills_committed, DispatchPhase.PREFILL)
        if not self._decode_ready():
            return None
        return DispatchSlot(self._decode_cursor, DispatchPhase.DECODE)

    def commit_slot(self, slot: DispatchSlot) -> None:
        """Advance after schedule() actually ran ``slot``.

        Args:
            slot: Must equal the current peek_slot().

        Raises:
            ValueError: No active list, or slot is not the current peek.
        """
        expected = self.peek_slot()
        if expected is None:
            raise ValueError("no committable slot (waiting or not started)")
        if slot != expected:
            raise ValueError(f"commit {slot} does not match peek {expected}")
        if slot.phase is DispatchPhase.PREFILL:
            self._prefills_committed += 1
            return
        packed = self._list
        assert packed is not None
        self._decode_cursor = (slot.microbatch_index + 1) % len(packed.tasks)

    def on_prefill_complete(self, microbatch_index: int) -> None:
        """Record that Prefill for ``microbatch_index`` has finished.

        Safe to call more than once. Ignored for out-of-range indices or
        before start(). Completes during Prefill fill do not offer Decode
        until every Prefill slot has been committed.
        """
        if self._list is None:
            return
        p = len(self._list.tasks)
        if 0 <= microbatch_index < p:
            self._prefill_done.add(microbatch_index)

    def _decode_ready(self) -> bool:
        assert self._list is not None
        p = len(self._list.tasks)
        if self.policy is DispatchPolicy.DRAIN:
            if len(self._prefill_done) < p:
                return False
        return self._decode_cursor in self._prefill_done
