# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

from vllm.v1.core.sched.tau_batch.types import WavePlan


class WaveDispatchPolicy(Enum):
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
        microbatch_index: Index into WavePlan.microbatches.
        phase: Prefill the whole prompt, or decode one step.
    """

    microbatch_index: int
    phase: DispatchPhase


class WaveDispatcher:
    """Issues ordered Prefill then cyclic Decode for one WavePlan.

    peek_slot does not advance. commit_slot advances only after the caller
    actually scheduled that slot. on_prefill_complete records that a
    committed Prefill has finished (update_from_output), which is the
    Decode gate.

    Args:
        policy: OVERLAP or DRAIN. Defaults to OVERLAP.
    """

    def __init__(
        self, policy: WaveDispatchPolicy = WaveDispatchPolicy.OVERLAP
    ) -> None:
        self.policy = policy
        self._plan: WavePlan | None = None
        self._prefills_committed = 0
        self._prefill_done: set[int] = set()
        self._decode_cursor = 0

    def start(self, plan: WavePlan) -> None:
        """Begin dispatching ``plan``. Replaces any previous wave.

        Args:
            plan: Non-empty wave from TauBatchPlanner.plan_wave.
        """
        if not plan.microbatches:
            raise ValueError("WavePlan.microbatches must be non-empty")
        self._plan = plan
        self._prefills_committed = 0
        self._prefill_done = set()
        self._decode_cursor = 0

    @property
    def plan(self) -> WavePlan | None:
        return self._plan

    def peek_slot(self) -> DispatchSlot | None:
        """Return the next slot without advancing.

        Returns:
            The slot to hold back to, or None if the caller must wait for
            on_prefill_complete (not the end of the wave). None if start()
            has not been called.
        """
        if self._plan is None:
            return None
        p = len(self._plan.microbatches)
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
            ValueError: No active plan, or slot is not the current peek.
        """
        expected = self.peek_slot()
        if expected is None:
            raise ValueError("no committable slot (waiting or not started)")
        if slot != expected:
            raise ValueError(f"commit {slot} does not match peek {expected}")
        if slot.phase is DispatchPhase.PREFILL:
            self._prefills_committed += 1
            return
        plan = self._plan
        assert plan is not None
        self._decode_cursor = (slot.microbatch_index + 1) % len(plan.microbatches)

    def on_prefill_complete(self, microbatch_index: int) -> None:
        """Record that Prefill for ``microbatch_index`` has finished.

        Safe to call more than once. Ignored for out-of-range indices or
        before start(). Completes during Prefill fill do not offer Decode
        until every Prefill slot has been committed.
        """
        if self._plan is None:
            return
        p = len(self._plan.microbatches)
        if 0 <= microbatch_index < p:
            self._prefill_done.add(microbatch_index)

    def _decode_ready(self) -> bool:
        assert self._plan is not None
        p = len(self._plan.microbatches)
        if self.policy is WaveDispatchPolicy.DRAIN:
            if len(self._prefill_done) < p:
                return False
        return self._decode_cursor in self._prefill_done
