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

from vllm.v1.core.sched.tau_batch.trace import load_events, pair_forwards


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
        "Per-stage widths are not in this trace."
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot τ-Batch JSONL occupancy Gantt to HTML."
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
    spans = pair_forwards(events)
    out = args.output or args.trace.with_suffix(".html")
    out.write_text(spans_to_html(spans, title=str(args.trace)), encoding="utf-8")
    print(f"wrote {out} ({len(spans)} forwards, {len(events)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
