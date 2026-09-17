"""Render each room's equation as a still for the grove's walls.

The palace's relief type is Cinzel: Latin letters, and no Greek or mathematical
operators at all. So an equation cannot be a wall line, and it should not be a
door sign either — a door says where you are going, not what is true there.
Each equation is therefore a picture, typeset here and hung as a `still`.

That division is better than a compromise would have been. Mathematics is the
same in every language, so one image serves every visitor; the sentence beside
it is prose, lives in the grove's label files, and gets translated. This script
writes the pictures and prints the sentences that belong with them.

Rendered with matplotlib's own mathtext, so no TeX installation is involved.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: The wing's rooms are indigo stone under cold white lamps, so a still with a
#: pale ground would read as a hole cut in the wall. Near-black with luminous
#: type reads as a plaque that is lit.
GROUND = "#0d0f1a"
TYPE = "#f2f0ea"
#: The one warm accent, matching the positive lobe of an orbital.
ACCENT = "#ffb84d"


@dataclass(frozen=True)
class Plaque:
    """One room's equation, and the sentence that goes on the wall beside it."""

    room: str
    slug: str
    #: mathtext source, without the surrounding dollar signs.
    equation: str
    #: Read in the room's own language; goes to `rooms.<room>.lines.<slug>`.
    wall_line: str
    #: What the exhibit cannot establish. Also a wall line, never buried.
    boundary: str
    #: A short caption under the equation, in ASCII, naming its parts.
    reading: str = ""


PLAQUES: tuple[Plaque, ...] = (
    Plaque(
        room="orangery",
        slug="orbitary",
        equation=r"-\frac{1}{2}\nabla^2\psi_i(\mathbf{r})"
        r" + v(\mathbf{r})\,\psi_i(\mathbf{r})"
        r" = \varepsilon_i\,\psi_i(\mathbf{r})",
        reading="one well, one wave, one energy",
        wall_line=(
            "Every shape in this room is a solution of the same equation, and each floats "
            "at its own energy. Nothing was modelled by hand. Walk the length of the room "
            "and you are pulling the two wells apart."
        ),
        boundary=(
            "These are non-interacting electrons in a well we chose, not a measured "
            "molecule. The height of each shape is its energy to a stated scale; the "
            "brightness on its surface is where its kinetic energy sits."
        ),
    ),
    Plaque(
        room="quantumflow",
        slug="shooting-gallery",
        equation=r"\psi_E(0) = 0 \;\;\mathrm{and}\;\; \psi_E(L) = 0"
        r" \;\;\Leftrightarrow\;\; E \in \{\varepsilon_0, \varepsilon_1, \varepsilon_2, \ldots\}",
        reading="the wave must land on both walls; almost no energy lets it",
        wall_line=(
            "Turn the dial and watch the wave miss the far wall. Out of every energy you "
            "can choose, only a handful let it land. Nobody put those values in — they are "
            "what is left when the wave is made to fit."
        ),
        boundary=(
            "The slider on this exhibit is trial energy, not time. Nothing here is moving; "
            "you are stepping through guesses, and the tape's frames are the guesses in order."
        ),
    ),
    Plaque(
        room="quantumflow/cloud",
        slug="cloud",
        equation=r"n(\mathbf{r}) = \sum_{i=1}^{N} \left| \psi_i(\mathbf{r}) \right|^2",
        reading="every wave, squared and added, once",
        wall_line=(
            "Six waves went in and one cloud came out. The claim this whole wing is built "
            "to test is that nothing was lost on the way — that the cloud alone still knows "
            "everything the waves knew."
        ),
        boundary=(
            "Each shell you can see holds a stated share of the electrons. The shells are "
            "measured, not chosen for looks, and the share is on the plaque."
        ),
    ),
    Plaque(
        room="quantumflow/bowl",
        slug="bowl",
        equation=r"T\!\left[\lambda n_1 + (1-\lambda) n_2\right] \;\leq\;"
        r" \lambda T[n_1] + (1-\lambda) T[n_2]",
        reading="the surface never rises above its own chords",
        wall_line=(
            "Two bowls, the same network, the same seed. One was built so that this "
            "inequality cannot fail; the other is the same network with that one constraint "
            "removed. The ribbons are the test. Go looking for red."
        ),
        boundary=(
            "The slice through density space is a straight one, which is what lets the "
            "picture stand as proof: a straight slice through a bowl is still a bowl. "
            "Neither network here has been trained on anything."
        ),
    ),
    Plaque(
        room="quantumflow/inversion",
        slug="inversion",
        equation=r"\mu = \frac{\delta T_s[n]}{\delta n(\mathbf{r})} + v(\mathbf{r})",
        reading="run it backwards: the cloud gives up the well",
        wall_line=(
            "This is the line the whole argument turns on. If it holds, you can hand the "
            "machine a cloud and it will hand back the well that shaped it — no waves, no "
            "orbitals, nothing but the density."
        ),
        boundary=(
            "The gap you can see between the two surfaces is the error, shown at full size "
            "and not flattered. The number on the plaque is the median over a thousand "
            "systems the machine was never trained on."
        ),
    ),
    Plaque(
        room="quantumflow/flow",
        slug="flow",
        equation=r"\frac{d\mathbf{x}}{dt} = \nabla_{\mathbf{x}} \Phi(t, \mathbf{x})"
        r" \qquad \nabla^2 \Phi = \lambda(\mathbf{x})\,\mathbf{I}",
        reading="mass follows a slope; the penalty asks that slope to curve evenly",
        wall_line=(
            "We had a good reason to believe the constraint on the right would help. We "
            "measured it. It cost accuracy at every strength we tried, and the cost grew "
            "the harder we leaned on it. That result is on the wall beside this one."
        ),
        boundary=(
            "The height of this flow is a potential fixed only up to an arbitrary offset, "
            "so two flows from different training runs cannot be compared by height. That "
            "is why only one hangs here, and why the comparison lives in the chart."
        ),
    ),
    Plaque(
        room="quantumflow/inside",
        slug="inside",
        equation=r"\psi_i(\mathbf{r}_0) = 0"
        r" \qquad \mathrm{and\ yet} \qquad"
        r" n(\mathbf{r}_0) = \sum_j \left| \psi_j(\mathbf{r}_0) \right|^2 > 0",
        reading="a hole in one wave is not a hole in the sum",
        wall_line=(
            "You are standing inside the cloud, and nowhere in here is it empty. Yet one of "
            "the waves that made it passes exactly through zero where you are standing. The "
            "surface you cannot cross belongs to that wave alone."
        ),
        boundary=(
            "The sum of waves that each vanish somewhere need not vanish anywhere. This is "
            "the price of the cloud: it keeps the total and forgets which wave paid it."
        ),
    ),
)


def render(plaque: Plaque, output_dir: Path, width_px: int = 2048) -> Path:
    """Typeset one equation onto a dark panel and save it as a PNG."""
    # A wall plaque, not a slide: wide and short, with the caption close enough
    # under the equation to read as part of it.
    height_px = round(width_px * 0.42)
    dpi = 200
    figure = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    figure.patch.set_facecolor(GROUND)

    figure.text(
        0.5,
        0.60 if plaque.reading else 0.5,
        f"${plaque.equation}$",
        color=TYPE,
        fontsize=26,
        ha="center",
        va="center",
    )
    if plaque.reading:
        figure.text(
            0.5,
            0.27,
            plaque.reading,
            color=ACCENT,
            fontsize=12,
            ha="center",
            va="center",
            alpha=0.9,
        )

    path = output_dir / f"equation-{plaque.slug}.png"
    figure.savefig(path, facecolor=GROUND, dpi=dpi)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stills"))
    parser.add_argument("--width", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines: dict[str, dict[str, str]] = {}
    for plaque in PLAQUES:
        path = render(plaque, args.output_dir, args.width)
        print(f"  {path.name:<32} {path.stat().st_size / 1e3:>7.1f} kB   {plaque.room}")
        lines[plaque.room] = {
            f"{plaque.slug}-line": plaque.wall_line,
            f"{plaque.slug}-boundary": plaque.boundary,
            f"{plaque.slug}-equation-caption": plaque.reading,
        }

    index = args.output_dir / "wall-lines.json"
    index.write_text(json.dumps(lines, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(PLAQUES)} equation stills -> {args.output_dir}")
    print(f"wall lines for translation: {index}")


if __name__ == "__main__":
    main()
