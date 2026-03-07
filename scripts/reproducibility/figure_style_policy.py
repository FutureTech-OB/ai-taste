#!/usr/bin/env python3
"""Shared panel-title and title-adjacent layout policy for figure scripts."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox


SUBTITLE_REWRITE_MAP = {
    "hardest pairs (adjacent tiers): where sft gains are largest": "Adjacent-tier pairwise difficulty",
    "rl checkpoint improves over frontier but trails sft": "RL checkpoint vs frontier and SFT",
    "rl collapse profile vs best frontier and sft ensemble": "Prediction concentration: RL, frontier reference, SFT",
    "distribution concentration (higher = less collapse)": "Distribution concentration index",
    "distribution balance ranking (higher = less collapse)": "Distribution balance ranking",
    "complementarity ceiling on expert-comparable articles": "Complementarity upper bound on expert-comparable articles",
}

PROHIBITED_SUBJECTIVE_TOKENS = (
    "where sft gains are largest",
    "improves over frontier but trails sft",
    "hardest pairs (adjacent tiers)",
)

_PANEL_TITLE_RE = re.compile(r"^\s*([a-zA-Z])(?:\s+([\s\S]+))?$")


def normalize_panel_subtitle(text: str | None) -> str:
    """Normalize panel subtitle wording to concise, objective labels."""
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    first = lines[0]
    panel_match = _PANEL_TITLE_RE.match(first)
    if panel_match and panel_match.group(2):
        letter = panel_match.group(1).lower()
        subtitle = panel_match.group(2).strip()
        mapped = SUBTITLE_REWRITE_MAP.get(subtitle.lower(), subtitle)
        lines[0] = f"{letter}  {mapped}"
    else:
        lines[0] = SUBTITLE_REWRITE_MAP.get(first.lower(), first)
    return "\n".join(lines)


def panel_title(
    ax: Axes,
    letter: str,
    subtitle: str | None = None,
    family: str = "main",
) -> Text:
    """Apply consistent panel title style: left anchor, regular weight."""
    letter_clean = str(letter).strip().lower()[:1] if str(letter).strip() else "a"
    subtitle_clean = normalize_panel_subtitle(subtitle)
    text = letter_clean if not subtitle_clean else f"{letter_clean}  {subtitle_clean}"
    if family == "edsi":
        fontsize = 7.2
        pad = 2.5
    else:
        fontsize = 7.5
        pad = 2.6
    return ax.set_title(text, loc="left", fontweight="normal", fontsize=fontsize, pad=pad)


def _artist_bbox(artist: Artist, renderer: Any) -> Bbox | None:
    if artist is None or (hasattr(artist, "get_visible") and not artist.get_visible()):
        return None
    if hasattr(artist, "get_window_extent"):
        try:
            bbox = artist.get_window_extent(renderer)
            if bbox is not None and bbox.width > 0 and bbox.height > 0:
                return bbox
        except Exception:
            return None
    return None


def _bbox_overlap_area(b1: Bbox, b2: Bbox) -> float:
    x0 = max(b1.x0, b2.x0)
    y0 = max(b1.y0, b2.y0)
    x1 = min(b1.x1, b2.x1)
    y1 = min(b1.y1, b2.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _title_artist(ax: Axes) -> Text | None:
    left = getattr(ax, "_left_title", None)
    if isinstance(left, Text) and left.get_text():
        return left
    center = ax.title
    if isinstance(center, Text) and center.get_text():
        return center
    right = getattr(ax, "_right_title", None)
    if isinstance(right, Text) and right.get_text():
        return right
    return None


def check_title_overlap(ax: Axes, artists: Sequence[Artist]) -> dict[str, Any]:
    """Return overlap metrics between panel title and provided artists."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    t_artist = _title_artist(ax)
    if t_artist is None:
        return {"overlap": False, "count": 0, "total_overlap_area": 0.0, "details": []}

    t_bbox = _artist_bbox(t_artist, renderer)
    if t_bbox is None:
        return {"overlap": False, "count": 0, "total_overlap_area": 0.0, "details": []}

    details: list[dict[str, Any]] = []
    total = 0.0
    for idx, artist in enumerate(artists):
        a_bbox = _artist_bbox(artist, renderer)
        if a_bbox is None:
            continue
        area = _bbox_overlap_area(t_bbox, a_bbox)
        if area > 0.0:
            total += area
            details.append({"index": idx, "artist": type(artist).__name__, "overlap_area": area})
    return {"overlap": bool(details), "count": len(details), "total_overlap_area": total, "details": details}


def place_legend_safely(
    ax: Axes,
    legend_kwargs: dict[str, Any],
    blockers: Iterable[Artist] | None = None,
) -> Legend:
    """Place legend with deterministic fallback locations to avoid title/callout overlap."""
    blockers_list = [b for b in (blockers or []) if b is not None]
    base_kwargs = dict(legend_kwargs)
    preferred_loc = base_kwargs.pop("loc", "upper right")
    fallback_order = []
    for loc in [preferred_loc, "upper right", "upper left", "lower right", "lower left", "best"]:
        if loc not in fallback_order:
            fallback_order.append(loc)

    fig = ax.figure
    best_loc = fallback_order[0]
    best_overlap = float("inf")

    for loc in fallback_order:
        leg = ax.legend(loc=loc, **base_kwargs)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overlap_total = check_title_overlap(ax, [leg])["total_overlap_area"]
        l_bbox = _artist_bbox(leg, renderer)
        if l_bbox is not None:
            for blocker in blockers_list:
                b_bbox = _artist_bbox(blocker, renderer)
                if b_bbox is not None:
                    overlap_total += _bbox_overlap_area(l_bbox, b_bbox)
        if overlap_total < best_overlap:
            best_overlap = overlap_total
            best_loc = loc
        leg.remove()
        if overlap_total == 0:
            break

    return ax.legend(loc=best_loc, **base_kwargs)


def place_reference_label_safely(
    ax: Axes,
    text: str,
    preferred_anchor: str = "upper left",
    blockers: Iterable[Artist] | None = None,
    **text_kwargs: Any,
) -> Text:
    """Place reference labels (for example Chance) with fallback anchors."""
    anchors = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "center left": (0.02, 0.50, "left", "center"),
        "center right": (0.98, 0.50, "right", "center"),
    }
    if preferred_anchor not in anchors:
        preferred_anchor = "upper left"
    order = [preferred_anchor] + [k for k in anchors.keys() if k != preferred_anchor]
    blockers_list = [b for b in (blockers or []) if b is not None]
    base = {"transform": ax.transAxes, "fontsize": 5.2, "color": "#666666"}
    base.update(text_kwargs)

    fig = ax.figure
    best_key = order[0]
    best_overlap = float("inf")
    best_text: Text | None = None

    for key in order:
        x, y, ha, va = anchors[key]
        artist = ax.text(x, y, text, ha=ha, va=va, **base)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overlap_total = check_title_overlap(ax, [artist])["total_overlap_area"]
        a_bbox = _artist_bbox(artist, renderer)
        if a_bbox is not None:
            for blocker in blockers_list:
                b_bbox = _artist_bbox(blocker, renderer)
                if b_bbox is not None:
                    overlap_total += _bbox_overlap_area(a_bbox, b_bbox)
        if overlap_total < best_overlap:
            best_overlap = overlap_total
            best_key = key
            if best_text is not None:
                try:
                    best_text.remove()
                except Exception:
                    pass
            best_text = artist
        else:
            artist.remove()
        if overlap_total == 0:
            return artist

    if best_text is not None and best_key in order:
        return best_text
    x, y, ha, va = anchors[best_key]
    return ax.text(x, y, text, ha=ha, va=va, **base)


def apply_title_policy(fig: plt.Figure, family: str = "main") -> None:
    """Normalize panel-title rendering and wording across all axes in a figure."""
    for ax in fig.axes:
        left_text = ax.get_title(loc="left")
        if left_text:
            left_clean = normalize_panel_subtitle(left_text)
            match = _PANEL_TITLE_RE.match(left_clean)
            if match:
                panel_title(ax, match.group(1), match.group(2), family=family)
            else:
                size = getattr(ax, "_left_title", ax.title).get_fontsize()
                ax.set_title(left_clean, loc="left", fontweight="normal", fontsize=size)

        center_text = ax.get_title(loc="center")
        if center_text:
            size = ax.title.get_fontsize()
            ax.set_title(normalize_panel_subtitle(center_text), loc="center", fontweight="normal", fontsize=size)

        right = getattr(ax, "_right_title", None)
        right_text = ax.get_title(loc="right")
        if right_text:
            size = right.get_fontsize() if isinstance(right, Text) else ax.title.get_fontsize()
            ax.set_title(normalize_panel_subtitle(right_text), loc="right", fontweight="normal", fontsize=size)
