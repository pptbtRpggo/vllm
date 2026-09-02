# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Render τ-Batch JSONL to a self-contained HTML occupancy Gantt.

Usage:
    python -m vllm.v1.core.sched.tau_batch.plot_trace TRACE.jsonl
    python -m vllm.v1.core.sched.tau_batch.plot_trace TRACE.jsonl -o out.html
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from vllm.v1.core.sched.tau_batch.trace import (
    load_events,
    pair_forwards,
    pipeline_cells,
)


def spans_to_html(spans, *, title: str = "τ-Batch PP occupancy") -> str:
    """Build an HTML Gantt. One row per forward; width is emit→done."""
    if not spans:
        return (
            "<!DOCTYPE html><html><body><p>No emit/done pairs in trace."
            "</p></body></html>"
        )
    t0 = min(s.emit_mono_ns for s in spans)
    t1 = max(s.done_mono_ns for s in spans)
    span_ns = max(t1 - t0, 1)
    row_h = 28
    top = 40
    left = 160
    width = 960
    height = top + row_h * len(spans) + 40
    bars: list[str] = []
    for i, s in enumerate(spans):
        x = left + (s.emit_mono_ns - t0) / span_ns * width
        w = max((s.done_mono_ns - s.emit_mono_ns) / span_ns * width, 2.0)
        y = top + i * row_h
        color = "#3b82f6" if s.phase == "prefill" else "#f59e0b"
        label = html.escape(s.job)
        reqs = html.escape(",".join(s.req_ids))
        dur = f"{s.duration_ms:.2f} ms"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{row_h - 6}" fill="{color}" rx="3">'
            f"<title>{label} {dur} {reqs}</title></rect>"
            f'<text x="8" y="{y + 16:.1f}" font-size="12" '
            f'font-family="ui-monospace,monospace">{label}</text>'
            f'<text x="{x + w + 6:.1f}" y="{y + 16:.1f}" font-size="11" '
            f'fill="#444" font-family="ui-monospace,monospace">{dur}</text>'
        )
    axis = (
        f'<line x1="{left}" y1="{top - 8}" x2="{left + width}" '
        f'y2="{top - 8}" stroke="#999"/>'
        f'<text x="{left}" y="18" font-size="12" fill="#666">'
        f"0 ms</text>"
        f'<text x="{left + width - 80}" y="18" font-size="12" fill="#666">'
        f"{span_ns / 1e6:.1f} ms</text>"
    )
    legend = (
        '<rect x="160" y="8" width="12" height="12" fill="#3b82f6"/>'
        '<text x="176" y="18" font-size="12">prefill</text>'
        '<rect x="240" y="8" width="12" height="12" fill="#f59e0b"/>'
        '<text x="256" y="18" font-size="12">decode</text>'
    )
    note = (
        "<p style='font:13px/1.4 system-ui;max-width:72rem'>"
        "Each bar is one micro-batch on the driver: "
        "<code>schedule()</code> emit until the PP Future completes. "
        "This is real wall time on the machine, not mock FLOPs. "
        "Bars overlap when the pipeline is filled. "
        "If the JSONL has <code>stage</code> events, use the pipeline plot "
        "(default when those events exist)."
        "</p>"
    )
    body = "\n".join(bars)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1 style='font:18px system-ui'>{html.escape(title)}</h1>"
        f"{note}"
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 120}" '
        f'height="{height}" style="background:#fff">'
        f"{legend}{axis}{body}</svg></body></html>"
    )


def pipeline_to_html(cells, *, title: str = "τ-Batch PP pipeline") -> str:
    """Build a PP Gantt: one row per stage, bars along wall-clock time."""
    if not cells:
        return (
            "<!DOCTYPE html><html><body><p>No stage events in trace. "
            "Re-run with the worker stage tracer, then plot again."
            "</p></body></html>"
        )
    t0 = min(c.start_ts_ns for c in cells)
    t1 = max(c.end_ts_ns for c in cells)
    span_ns = max(t1 - t0, 1)
    n_stages = max(c.pp_rank for c in cells) + 1
    row_h = 52
    top = 44
    left = 72
    width = 1100
    height = top + row_h * n_stages + 36
    bars: list[str] = []
    for rank in range(n_stages):
        y = top + rank * row_h
        bars.append(
            f'<text x="8" y="{y + 22:.1f}" font-size="13" '
            f'font-family="ui-monospace,monospace">PP{rank}</text>'
        )
        for c in cells:
            if c.pp_rank != rank:
                continue
            x = left + (c.start_ts_ns - t0) / span_ns * width
            w = max((c.end_ts_ns - c.start_ts_ns) / span_ns * width, 2.0)
            color = "#3b82f6" if c.phase == "prefill" else "#f59e0b"
            label = html.escape(c.job)
            dur = f"{c.duration_ms:.2f} ms"
            bars.append(
                f'<rect x="{x:.1f}" y="{y + 6:.1f}" width="{w:.1f}" '
                f'height="{row_h - 14}" fill="{color}" opacity="0.9" rx="2">'
                f"<title>{label} PP{rank} {dur}</title></rect>"
            )
    axis = (
        f'<line x1="{left}" y1="{top - 8}" x2="{left + width}" '
        f'y2="{top - 8}" stroke="#999"/>'
        f'<text x="{left}" y="18" font-size="12" fill="#666">0 ms</text>'
        f'<text x="{left + width - 80}" y="18" font-size="12" fill="#666">'
        f"{span_ns / 1e6:.1f} ms</text>"
    )
    legend = (
        '<rect x="72" y="8" width="12" height="12" fill="#3b82f6"/>'
        '<text x="88" y="18" font-size="12">prefill</text>'
        '<rect x="160" y="8" width="12" height="12" fill="#f59e0b"/>'
        '<text x="176" y="18" font-size="12">decode</text>'
    )
    note = (
        "<p style='font:13px/1.4 system-ui;max-width:72rem'>"
        "Each row is one PP rank. A bar is that rank's "
        "<code>execute_model</code> (rank 0) or the interval after the "
        "previous rank finished (recv unblocks, then this rank computes). "
        "Wall clock is <code>time.time_ns()</code> across worker processes."
        "</p>"
    )
    body = "\n".join(bars)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1 style='font:18px system-ui'>{html.escape(title)}</h1>"
        f"{note}"
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 40}" '
        f'height="{height}" style="background:#fff">'
        f"{legend}{axis}{body}</svg></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot τ-Batch JSONL Gantt to HTML."
    )
    parser.add_argument("trace", type=Path, help="JSONL from --tau-batch-trace")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="HTML path (default: TRACE with .html suffix)",
    )
    args = parser.parse_args(argv)
    events = load_events(args.trace)
    cells = pipeline_cells(events)
    spans = pair_forwards(events)
    out = args.output or args.trace.with_suffix(".html")
    if cells:
        out.write_text(
            pipeline_to_html(cells, title=str(args.trace)), encoding="utf-8"
        )
        print(
            f"wrote {out} ({len(cells)} stage cells, "
            f"{len(spans)} forwards, {len(events)} events)"
        )
    else:
        out.write_text(
            spans_to_html(spans, title=str(args.trace)), encoding="utf-8"
        )
        print(f"wrote {out} occupancy only ({len(spans)} forwards; no stage events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
