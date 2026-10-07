"""Step 6.2: Figure 1 of the paper, the loop schematic (no data; drawn with the common style).

    python maps/fig_loop.py   ->  paper/figures/fig_loop.pdf and .png

Head pulley on the right, tail on the left; the carry strand runs to the head on top, the
return strand to the tail below. The drive sits on the return strand at sigma_d (drawn
intermediate, as the general case; sigma_d = 0 is the head drive) and the gravity take-up
downstream of it at sigma_t: a pulley hung on n belt strands (n = 2 drawn) with its mass M.
sigma is measured from the head pulley along the return strand; the loop coordinate s starts at
the drive exit (s = 0) and runs with the belt to the drive entry (s = 2L). Strand A goes from
the drive exit to the take-up, strand B from the take-up through the tail, the carry strand and
the head back to the drive entry (Sections 3.3 and 5.1). Positions are illustrative.
"""
from pathlib import Path

import numpy as np

import style

OUT = Path(__file__).parents[1] / "paper" / "figures"

# geometry (drawing units, equal aspect)
XT, XH = 0.9, 15.1                 # tail and head pulley centres
YC, YR = 3.5, 1.9                  # carry and return strand heights (gap = pulley diameter)
R = (YC - YR) / 2                  # end pulley radius
YP = YR + R                        # end pulley centre height
SD, ST = 0.22, 0.52                # drive and take-up positions, units of L (illustrative)
RD = 0.32                          # drive pulley radius
WT, YT = 0.38, -0.15               # take-up loop half width (= pulley radius), pulley centre
YA = -2.35                         # sigma axis


def xs(sigma):
    """x of the point of the return strand at sigma (units of L) from the head pulley."""
    return XH - sigma * (XH - XT)


def arc(xc, yc, r, a0, a1, n=40):
    return [(xc + r * np.cos(a), yc + r * np.sin(a)) for a in np.linspace(a0, a1, n)]


def belt_path():
    """Belt centreline, closed: carry to the head, head wrap, return with the wrap under the
    drive pulley and the take-up loop, tail wrap."""
    xd, xt = xs(SD), xs(ST)
    pts = [(XT, YC), (XH, YC)]
    pts += arc(XH, YP, R, np.pi / 2, -np.pi / 2)
    pts += [(xd + RD, YR)] + arc(xd, YR, RD, 0.0, -np.pi)
    pts += [(xt + WT, YR), (xt + WT, YT)] + arc(xt, YT, WT, 0.0, -np.pi) + [(xt - WT, YR)]
    pts += [(XT, YR)] + arc(XT, YP, R, -np.pi / 2, -3 * np.pi / 2)
    return np.array(pts)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    style.apply()
    fig, ax = plt.subplots(figsize=(style.DOUBLE, 75 * style.MM))
    ax.set_aspect("equal")
    ax.axis("off")
    ink, grey, blue, red = "k", "0.45", style.OKABE_ITO[0], style.OKABE_ITO[1]
    fs = 7
    xd, xt = xs(SD), xs(ST)

    P = belt_path()
    ax.plot(P[:, 0], P[:, 1], color=ink, lw=1.6, solid_capstyle="round", zorder=3)

    def pulley(x, y, r, fill="w", lw=0.9):
        ax.add_patch(Circle((x, y), r, fc=fill, ec=ink, lw=lw, zorder=4))
        ax.add_patch(Circle((x, y), 0.06, fc=ink, ec=ink, zorder=5))

    def arrow(p0, p1, color=ink, lw=0.8, astyle="-|>", ms=6):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=astyle, mutation_scale=ms, color=color,
                                     lw=lw, zorder=6, shrinkA=0, shrinkB=0))

    # end pulleys
    pulley(XH, YP, 0.92 * R)
    pulley(XT, YP, 0.92 * R)
    ax.text(XH + R + 0.15, YP, "head\npulley", fontsize=fs, va="center", ha="left")
    ax.text(XT - R - 0.15, YP, "tail\npulley", fontsize=fs, va="center", ha="right")

    # strands and direction of travel
    ax.text((XT + XH) / 2, YC + 0.12, r"carry strand $c$:  $\mu_c$, $c_c$", fontsize=fs,
            ha="center", va="bottom")
    ax.text((XT + xt) / 2 - 0.4, YR + 0.12, r"return strand $r$:  $\mu_r$, $c_r$", fontsize=fs,
            ha="center", va="bottom")
    for x0, y0, dx in ((3.6, YC, 1), (11.8, YC, 1), (5.4, YR, -1), (14.0, YR, -1)):
        arrow((x0, y0 - 0.17), (x0 + 0.7 * dx, y0 - 0.17), color=grey, lw=0.7, ms=5)

    # drive pulley (shaded: prescribed velocity) and the two faces of the loop coordinate
    pulley(xd, YR, 0.86 * RD, fill="0.75", lw=1.0)
    ax.text(xd, YR - RD - 0.2, "drive\n(prescribed\nvelocity)", fontsize=fs, ha="center", va="top")
    ax.text(xd - RD - 0.08, YR + 0.1, r"$s=0$", fontsize=fs, ha="right", va="bottom", color=blue)
    ax.text(xd + RD + 0.08, YR + 0.1, r"$s=2L$", fontsize=fs, ha="left", va="bottom", color=blue)

    # gravity take-up: hanging pulley on n = 2 strands, mass M, travel y
    pulley(xt, YT, 0.86 * WT)
    yb = YT - 0.95
    ax.plot([xt, xt], [YT - 0.86 * WT, yb], color=ink, lw=0.7, zorder=2)
    ax.add_patch(Rectangle((xt - 0.45, yb - 0.5), 0.9, 0.5, fc="0.85", ec=ink, lw=0.8, zorder=3))
    ax.text(xt, yb - 0.25, r"$M$", fontsize=fs + 1, ha="center", va="center", zorder=4)
    arrow((xt + 0.75, yb - 0.05), (xt + 0.75, yb - 0.75), color=red, lw=0.9)
    ax.text(xt + 0.85, yb - 0.42, r"$Mg$", fontsize=fs, ha="left", va="center", color=red)
    arrow((xt - 0.8, YT + 0.1), (xt - 0.8, YT - 0.7), lw=0.7)
    ax.text(xt - 0.9, YT - 0.3, r"$y$", fontsize=fs, ha="right", va="center")
    ax.text(xt - WT - 0.3, YR - 0.4, "gravity take-up:\npulley hung on $n$ belt\nstrands ($n=2$ drawn)",
            fontsize=fs, ha="right", va="top")

    # strands A and B of the loop coordinate (blue)
    yA = YR + 0.62
    arrow((xt + WT, yA), (xd - RD, yA), color=blue, lw=0.8, astyle="|-|", ms=2.5)
    ax.text((xt + WT + xd - RD) / 2, yA + 0.06, r"strand A: $0\leq s\leq\xi L$", fontsize=fs,
            ha="center", va="bottom", color=blue)
    ax.text((XT + XH) / 2, YC + 0.62,
            r"strand B: $\xi L\leq s\leq 2L$, from the take-up by the tail, the carry strand "
            r"and the head to the drive entry", fontsize=fs, ha="center", va="bottom", color=blue)

    # sigma, from the head pulley along the return strand
    arrow((XH, YA), (XT, YA), lw=0.7)
    for x, lab in ((XH, r"$\sigma=0$"), (xd, r"$\sigma_d$"), (xt, r"$\sigma_t$"), (XT, r"$\sigma=L$")):
        ax.plot([x, x], [YA - 0.08, YA + 0.08], color=ink, lw=0.7)
        ax.text(x, YA - 0.15, lab, fontsize=fs, ha="center", va="top")
    ax.text((xd + XH) / 2, YA + 0.08, r"$\sigma$", fontsize=fs + 1, ha="center", va="bottom")

    ax.set_xlim(XT - 1.7, XH + 1.7)
    ax.set_ylim(YA - 0.6, YC + 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_loop.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_loop.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure:", OUT / "fig_loop.pdf")


if __name__ == "__main__":
    main()
