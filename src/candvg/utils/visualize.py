"""Visualization utilities for plot bounding boxes and phrase labels on images."""

from pathlib import Path

import kaleido
import plotly.graph_objects as go
from PIL import Image

from .schema.predict import SentencePrediction

_KALEIDO_READY = False
_PHRASE_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)


def _phrase_color(index: int) -> str:
    """Return deterministic color by phrase index."""
    return _PHRASE_COLORS[index % len(_PHRASE_COLORS)]


def _intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """Return True if two rectangles intersect."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _find_label_position(
    anchor_x: float,
    anchor_y: float,
    text: str,
    image_width: int,
    image_height: int,
    placed_rects: list[tuple[float, float, float, float]],
) -> tuple[float, float, tuple[float, float, float, float]]:
    """Greedily place a label rectangle near anchor without overlapping existing labels."""
    char_width = 8.0
    label_height = 22.0
    label_width = min(max(90.0, len(text) * char_width), max(90.0, image_width * 0.9))

    x = min(max(anchor_x, 0.0), max(0.0, image_width - label_width))
    candidate_offsets = [0.0]
    step = label_height + 6.0
    max_steps = max(1, int(image_height / step))
    for i in range(1, max_steps + 1):
        candidate_offsets.append(-i * step)
        candidate_offsets.append(i * step)

    for offset in candidate_offsets:
        y = min(max(anchor_y + offset, label_height), float(image_height))
        rect = (x, y - label_height, x + label_width, y)
        if not any(_intersects(rect, existing) for existing in placed_rects):
            placed_rects.append(rect)
            return x, y, rect

    # Fallback to the last candidate if no free region is found.
    y = min(max(anchor_y, label_height), float(image_height))
    rect = (x, y - label_height, x + label_width, y)
    placed_rects.append(rect)
    return x, y, rect


def plot_results(
    image: Image.Image, prediction: SentencePrediction, output_file: Path
) -> None:
    """Plot predicted bounding boxes and phrase labels on an image."""
    global _KALEIDO_READY
    if not _KALEIDO_READY:
        kaleido.get_chrome_sync()
        _KALEIDO_READY = True

    w, h = image.size
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=image, x=0, y=0, sizex=w, sizey=h, xref="x", yref="y", layer="below"
        )
    )

    placed_label_rects: list[tuple[float, float, float, float]] = []
    for phrase_prediction in prediction.phrases:
        line_color = _phrase_color(phrase_prediction.index)
        label_bg = "rgba(255, 255, 255, 0.80)"
        for bb in phrase_prediction.bounding_boxes:
            r = bb.rect
            label_text = f"[{phrase_prediction.index}] {phrase_prediction.text}: {bb.confidence:.2f}"
            fig.add_shape(
                type="rect",
                x0=r.x1,
                y0=r.y1,
                x1=r.x1 + r.width,
                y1=r.y1 + r.height,
                line=dict(width=3, color=line_color),
            )
            label_x, label_y, _ = _find_label_position(
                anchor_x=r.x1,
                anchor_y=r.y1,
                text=label_text,
                image_width=w,
                image_height=h,
                placed_rects=placed_label_rects,
            )
            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=label_text,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(
                    size=15,
                    color="black",
                    family="IPAexGothic, TakaoGothic, sans-serif",
                ),
                bgcolor=label_bg,
                bordercolor=line_color,
                borderwidth=1,
            )

    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x")
    fig.update_layout(width=w, height=h, margin=dict(l=0, r=0, t=0, b=0))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_file)
