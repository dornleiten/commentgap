"""Small shared helpers for figures that are both saved and shown."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_display_figure(
    figure: Any,
    output_root: Path | None,
    stem: str,
    *,
    show: bool,
    dpi: int = 180,
):
    """Optionally save PNG/PDF copies and show a figure.

    ``output_root=None`` is the notebook/display-only path: it never touches
    report artifacts.  Figures are closed after ``show=False`` rendering so
    explicit notebook ``display(fig)`` calls do not leave duplicate live
    figures registered with the inline backend.
    """
    import matplotlib.pyplot as plt

    if output_root is not None:
        figures_root = Path(output_root) / "figures"
        figures_root.mkdir(parents=True, exist_ok=True)
        figure.savefig(figures_root / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(figures_root / f"{stem}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure
