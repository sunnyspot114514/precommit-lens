"""Render the v4 confirmatory lead curve as a dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


COLORS = {
    "residual": "#c43c39",
    "tfidf": "#2878a8",
    "stats": "#2e8b57",
    "judge": "#8c6d31",
    "grid": "#d8dde3",
    "text": "#20252b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v4 lead-time results.")
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("results/trajectory_v4_confirmatory/Qwen__Qwen3-0.6B/v4_analysis.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/trajectory_v4_confirmatory/Qwen__Qwen3-0.6B/v4_lead_curve.svg"),
    )
    return parser.parse_args()


def line(points: list[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points
    )
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="3"/>{circles}'


def main() -> None:
    args = parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    primary_layer = int(payload["protocol"]["primary_layer"])
    experiment_label = str(payload["protocol"].get("experiment_label", "v4"))
    rows = [row for row in payload["checkpoint_results"] if row["checkpoint"] <= 12]
    checkpoints = [int(row["checkpoint"]) for row in rows]

    width, height = 1000, 720
    left, right = 88, 960
    top1, bottom1 = 82, 390
    top2, bottom2 = 482, 650

    def x_pos(checkpoint: int) -> float:
        return left + (right - left) * checkpoint / 12

    def y_auc(value: float) -> float:
        return bottom1 - (bottom1 - top1) * (value - 0.3) / 0.7

    def y_delta(value: float) -> float:
        return bottom2 - (bottom2 - top2) * (value + 0.12) / 0.26

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="{COLORS["text"]}">{html.escape(experiment_label)} pre-landing trajectory prediction</text>',
        f'<text x="{left}" y="58" font-family="Arial, sans-serif" font-size="14" fill="#58616b">Prompt-macro within-prompt AUC; frozen test prompts</text>',
    ]

    for value in (0.3, 0.5, 0.7, 0.9, 1.0):
        y = y_auc(value)
        dash = ' stroke-dasharray="6 5"' if value == 0.5 else ""
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{COLORS["grid"]}"{dash}/>'
        )
        svg.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#58616b">{value:.1f}</text>'
        )
    svg.append(
        f'<text x="24" y="{(top1 + bottom1) / 2:.1f}" transform="rotate(-90 24 {(top1 + bottom1) / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#39424c">Pairwise AUC</text>'
    )

    method_values: dict[str, list[float | None]] = {
        "residual": [row["metrics"][f"residual_layer_{primary_layer}"]["auc"] for row in rows],
        "tfidf": [row["metrics"]["visible_prefix_tfidf"]["auc"] for row in rows],
        "stats": [row["metrics"]["next_token_stats"]["auc"] for row in rows],
        "judge": [row["metrics"].get("prefix_model_judge", {}).get("auc") for row in rows],
    }
    for method, values in method_values.items():
        points = [
            (x_pos(checkpoint), y_auc(float(value)))
            for checkpoint, value in zip(checkpoints, values, strict=True)
            if value is not None
        ]
        svg.append(line(points, COLORS[method]))

    legend = [
        (f"Residual L{primary_layer}", "residual"),
        ("Visible-prefix TF-IDF", "tfidf"),
        ("Next-token stats", "stats"),
        ("Prefix model judge", "judge"),
    ]
    for idx, (label, method) in enumerate(legend):
        x = left + idx * 215
        svg.append(f'<line x1="{x}" y1="418" x2="{x + 24}" y2="418" stroke="{COLORS[method]}" stroke-width="3"/>')
        svg.append(f'<circle cx="{x + 12}" cy="418" r="4" fill="{COLORS[method]}"/>')
        svg.append(
            f'<text x="{x + 32}" y="423" font-family="Arial, sans-serif" font-size="13" fill="{COLORS["text"]}">{html.escape(label)}</text>'
        )

    for value in (-0.1, 0.0, 0.03, 0.1):
        y = y_delta(value)
        color = "#8fbf9f" if value == 0.03 else COLORS["grid"]
        dash = ' stroke-dasharray="6 5"' if value in (0.0, 0.03) else ""
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{color}"{dash}/>')
        svg.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#58616b">{value:+.2f}</text>'
        )
    svg.append(
        f'<text x="24" y="{(top2 + bottom2) / 2:.1f}" transform="rotate(-90 24 {(top2 + bottom2) / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#39424c">Residual - cheap envelope</text>'
    )
    for row in rows:
        checkpoint = int(row["checkpoint"])
        comparison: dict[str, Any] = row["primary_comparison"]
        value = comparison["delta"]
        low, high = comparison["ci95"]
        if value is None or low is None or high is None:
            continue
        x = x_pos(checkpoint)
        y = y_delta(float(value))
        y_low, y_high = y_delta(float(low)), y_delta(float(high))
        svg.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{COLORS["residual"]}" stroke-width="2"/>')
        svg.append(f'<line x1="{x - 5:.1f}" y1="{y_high:.1f}" x2="{x + 5:.1f}" y2="{y_high:.1f}" stroke="{COLORS["residual"]}" stroke-width="2"/>')
        svg.append(f'<line x1="{x - 5:.1f}" y1="{y_low:.1f}" x2="{x + 5:.1f}" y2="{y_low:.1f}" stroke="{COLORS["residual"]}" stroke-width="2"/>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS["residual"]}"/>')

    for checkpoint in checkpoints:
        x = x_pos(checkpoint)
        svg.append(
            f'<text x="{x:.1f}" y="680" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#58616b">{checkpoint}</text>'
        )
    svg.append(
        f'<text x="{(left + right) / 2:.1f}" y="706" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#39424c">Generated tokens before checkpoint</text>'
    )
    svg.append(
        f'<text x="{right}" y="466" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#5d6b64">frozen success margin +0.03</text>'
    )
    svg.append("</svg>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
