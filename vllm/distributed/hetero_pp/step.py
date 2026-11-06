# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous stage contract. No Ray or device imports here."""


def run_stage(rank, size, sequence, has_tokens, link, execute):
    inputs = link.recv(sequence) if has_tokens and rank > 0 else None
    output = execute(inputs)
    if has_tokens and rank < size - 1:
        for work in link.send(output, sequence):
            work.wait()
        return None
    return output


def submit_pipeline(size, submit, gather):
    # Never ray.get a receiver before the corresponding senders are submitted.
    # Gather ALL ranks, not just the final rank, to propagate middle-stage errors.
    refs = [submit(rank) for rank in range(size)]
    return gather(refs)[-1]
