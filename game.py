"""
Relativistic Agar.io — two black holes compete to gravitationally capture
food stars in curved 2-D spacetime.

Rules
-----
* Each BH generates spacetime curvature; food stars fall along geodesics.
* A food star is swallowed when it enters 1.5 × the BH's Schwarzschild radius.
* Bigger BH → larger r_s → wider capture net → positive feedback loop.
* The Proper-Time HUD shows time dilation live (heavier / faster → smaller τ/t).
* Doppler colour on every body (blue = approaching field centre, red = receding).

Run (saves PNG frames to ./frames/):
    python game.py

Then stitch to video with ffmpeg:
    ffmpeg -framerate 30 -i frames/frame_%05d.png -c:v libx264 out.mp4
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from engine import Body, rk4_step, renormalize, bodies_overlap, elastic_bounce, doppler_color

# ── Tuneable constants ──────────────────────────────────────────────────────

DTAU             = 0.06     # proper-time step per physics tick
STEPS_PER_FRAME  = 4        # physics ticks per rendered frame
TRAIL_LENGTH     = 70       # history length per body
FPS              = 30
GAME_FRAMES      = 900      # ≈ 30 s wall time at 30 fps

FIELD_W, FIELD_H = 180.0, 120.0   # half-extents of playing field

BH_INIT_MASS     = 7.0            # r_s = 14
BH_VIS_SCALE     = 1.8            # visual_radius = r_s × scale
BH_ORBIT_R       = 65.0           # initial orbit radius from centre
BH_V             = 0.12           # orbital speed (c = 1)

FOOD_MASS        = 0.7
FOOD_RADIUS      = 3.5
FOOD_RING_RMIN   = 36.0           # inner radius of food spawn ring (per BH)
FOOD_RING_RMAX   = 58.0           # outer radius
N_FOOD           = 16             # 8 near each BH
ABSORB_SCALE     = 2.0            # absorb when sep < ABSORB_SCALE × r_s
RESPAWN_MARGIN   = 15.0           # extra clearance beyond absorption zone on respawn
MAX_BH_MASS      = 35.0           # cap mass to prevent infinite cascade
FOOD_V_INWARD    = 0.10           # inward radial speed toward nearest BH
FOOD_V_TANG      = 0.04           # small tangential component (orbit hint)

BG_COLOR         = "#06060f"
FIELD_EDGE       = "#182030"
OUT_DIR          = "frames"

rng = np.random.default_rng(7)


# ── Body factories ───────────────────────────────────────────────────────────

def _bh_vis_radius(bh: Body) -> float:
    return bh.r_s * BH_VIS_SCALE


def _food_vel(angle):
    """
    Velocity for a food star at given angle from its BH:
    FOOD_V_INWARD radially inward + FOOD_V_TANG tangential.
    """
    r_hat = np.array([ np.cos(angle),  np.sin(angle), 0.0])   # outward
    t_hat = np.array([-np.sin(angle),  np.cos(angle), 0.0])   # tangential
    vel3  = -FOOD_V_INWARD * r_hat + FOOD_V_TANG * t_hat * rng.choice([-1, 1])
    return vel3


def _food_around(bh_pos3, n, r_min, r_max):
    """Spawn n food stars in a ring, each falling inward toward the BH."""
    stars = []
    for i in range(n):
        angle = 2.0 * np.pi * i / n + rng.uniform(-0.4, 0.4)
        r     = rng.uniform(r_min, r_max)
        pos   = bh_pos3 + np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        pos   = np.clip(pos, [-FIELD_W * 0.9, -FIELD_H * 0.9, 0.0],
                              [ FIELD_W * 0.9,  FIELD_H * 0.9, 0.0])
        stars.append(Body("food", FOOD_MASS, pos, _food_vel(angle),
                          radius=FOOD_RADIUS, color="#88ffaa"))
    return stars


def make_food_respawn(bh1: Body, bh2: Body):
    """
    Spawn one replacement food star at a position that is safely outside
    both BHs' current absorption zones (ABSORB_SCALE × r_s + RESPAWN_MARGIN).
    """
    min1 = ABSORB_SCALE * bh1.r_s + RESPAWN_MARGIN
    min2 = ABSORB_SCALE * bh2.r_s + RESPAWN_MARGIN

    r_lo = max(min1, FOOD_RING_RMIN)
    r_hi = max(r_lo + 10.0, min(FOOD_RING_RMAX * 1.5, FIELD_W * 0.8))

    for _ in range(400):
        angle  = rng.uniform(0, 2 * np.pi)
        r      = rng.uniform(r_lo, r_hi)
        anchor = bh1.x[1:4] if rng.random() < 0.5 else bh2.x[1:4]
        pos    = anchor + np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        pos    = np.clip(pos, [-FIELD_W * 0.88, -FIELD_H * 0.88, 0.0],
                               [ FIELD_W * 0.88,  FIELD_H * 0.88, 0.0])

        d1 = float(np.linalg.norm(pos - bh1.x[1:4]))
        d2 = float(np.linalg.norm(pos - bh2.x[1:4]))
        if d1 > min1 and d2 > min2:
            return Body("food", FOOD_MASS, pos, _food_vel(angle), FOOD_RADIUS, "#88ffaa")

    # last-resort fallback: place near field edge away from both BHs
    for sign in (1, -1):
        pos = np.array([sign * FIELD_W * 0.6, 0.0, 0.0])
        if (np.linalg.norm(pos - bh1.x[1:4]) > min1 and
                np.linalg.norm(pos - bh2.x[1:4]) > min2):
            return Body("food", FOOD_MASS, pos, [0.0, 0.05, 0.0], FOOD_RADIUS, "#88ffaa")
    return Body("food", FOOD_MASS, [0.0, FIELD_H * 0.7, 0.0], [0.05, 0.0, 0.0],
                FOOD_RADIUS, "#88ffaa")


def make_scene():
    init_vis_r = BH_INIT_MASS * 2.0 * BH_VIS_SCALE
    bh1 = Body("BH-Cyan",    BH_INIT_MASS,
               [-BH_ORBIT_R, 0.0, 0.0], [0.0,  BH_V, 0.0],
               radius=init_vis_r, color="#00e5ff")
    bh2 = Body("BH-Magenta", BH_INIT_MASS,
               [ BH_ORBIT_R, 0.0, 0.0], [0.0, -BH_V, 0.0],
               radius=init_vis_r, color="#ff4dff")
    half = N_FOOD // 2
    food = (
        _food_around(bh1.x[1:4], half,          FOOD_RING_RMIN, FOOD_RING_RMAX) +
        _food_around(bh2.x[1:4], N_FOOD - half, FOOD_RING_RMIN, FOOD_RING_RMAX)
    )
    return bh1, bh2, food


# ── Figure construction ──────────────────────────────────────────────────────

def build_figure():
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-FIELD_W - 12, FIELD_W + 12)
    ax.set_ylim(-FIELD_H - 12, FIELD_H + 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # field border
    ax.add_patch(plt.Rectangle(
        (-FIELD_W, -FIELD_H), 2 * FIELD_W, 2 * FIELD_H,
        fill=False, edgecolor=FIELD_EDGE, lw=1.5, zorder=1,
    ))
    ax.axvline(0, color=FIELD_EDGE, lw=0.8, zorder=1)
    ax.add_patch(Circle((0, 0), 40, fill=False, edgecolor=FIELD_EDGE, lw=0.8, zorder=1))

    ax.text(-FIELD_W + 5, -FIELD_H + 5,
            "Doppler colour vs field centre  ·  blue = approach  ·  red = recede",
            color="#334455", fontsize=7, va="bottom", fontfamily="monospace", zorder=10)

    return fig, ax


# ── Main simulation + render loop ────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bh1, bh2, food = make_scene()

    score  = [0.0, 0.0]
    bhs    = [bh1, bh2]

    fig, ax = build_figure()

    # ── Persistent artists ────────────────────────────────────────────────
    bh1_trail, = ax.plot([], [], "-", color="#00e5ff", alpha=0.22, lw=1,   zorder=2)
    bh2_trail, = ax.plot([], [], "-", color="#ff4dff", alpha=0.22, lw=1,   zorder=2)

    bh1_horizon = Circle(bh1.x[1:3], bh1.r_s,    color="#000000",           zorder=4)
    bh2_horizon = Circle(bh2.x[1:3], bh2.r_s,    color="#000000",           zorder=4)
    bh1_patch   = Circle(bh1.x[1:3], bh1.radius, color=bh1.color, alpha=0.9, zorder=5)
    bh2_patch   = Circle(bh2.x[1:3], bh2.radius, color=bh2.color, alpha=0.9, zorder=5)
    for p in (bh1_horizon, bh2_horizon, bh1_patch, bh2_patch):
        ax.add_patch(p)

    food_patches: list[Circle] = []
    food_trails:  list         = []
    for f in food:
        fp = Circle(f.x[1:3], FOOD_RADIUS, color=f.color, zorder=3, alpha=0.85)
        ax.add_patch(fp)
        food_patches.append(fp)
        fl, = ax.plot([], [], "-", color="#88ffaa", alpha=0.12, lw=0.5, zorder=2)
        food_trails.append(fl)

    hud = ax.text(FIELD_W - 8, FIELD_H - 5, "",
                  color="#b0ccd0", fontsize=8, fontfamily="monospace",
                  va="top", ha="right", zorder=10)
    score_txt = ax.text(0, FIELD_H + 6, "Cyan  0.0 : 0.0  Magenta",
                        color="white", fontsize=14, fontweight="bold",
                        ha="center", va="top", zorder=10)
    msg_txt = ax.text(0, 20, "", color="#ffdd00", fontsize=30, fontweight="bold",
                      ha="center", va="center", zorder=11, alpha=0.0)

    # Timer bar
    ax.add_patch(plt.Rectangle((-FIELD_W, FIELD_H + 1), 2 * FIELD_W, 5,
                               color="#0e1520", zorder=9))
    timer_bar = ax.add_patch(plt.Rectangle((-FIELD_W, FIELD_H + 1), 0, 5,
                                           color="#334466", zorder=9))

    print(f"Rendering {GAME_FRAMES} frames → ./{OUT_DIR}/  ...")

    # ── Frame loop ─────────────────────────────────────────────────────────
    for frame in range(GAME_FRAMES + 1):

        # ── Physics ────────────────────────────────────────────────────────
        for _ in range(STEPS_PER_FRAME):
            rk4_step(bh1, [bh2], DTAU)
            rk4_step(bh2, [bh1], DTAU)
            renormalize(bh1, [bh2])
            renormalize(bh2, [bh1])

            if bodies_overlap(bh1, bh2):
                elastic_bounce(bh1, bh2)

            for f in food:
                rk4_step(f, bhs, DTAU)
                renormalize(f, bhs)

        # Soft boundary for BHs
        for bh in bhs:
            for i, lim in ((1, FIELD_W), (2, FIELD_H)):
                if abs(bh.x[i]) > lim:
                    bh.x[i]  = np.sign(bh.x[i]) * lim
                    bh.u[i] *= -0.5

        # Soft boundary for food
        for f in food:
            for i, lim in ((1, FIELD_W * 0.95), (2, FIELD_H * 0.95)):
                if abs(f.x[i]) > lim:
                    f.x[i]  = np.sign(f.x[i]) * lim
                    f.u[i] *= -0.7

        # ── Food absorption ────────────────────────────────────────────────
        eaten_idx = []
        for idx, f in enumerate(food):
            for bi, bh in enumerate(bhs):
                sep = float(np.linalg.norm(f.x[1:4] - bh.x[1:4]))
                if sep < ABSORB_SCALE * bh.r_s:
                    bh.mass    = min(bh.mass + f.mass, MAX_BH_MASS)
                    bh.r_s     = 2.0 * bh.mass
                    bh.radius  = bh.r_s * BH_VIS_SCALE
                    score[bi] += f.mass
                    eaten_idx.append(idx)
                    break

        # Replace eaten food stars
        for idx in sorted(set(eaten_idx), reverse=True):
            food_patches[idx].remove()
            food_trails[idx].remove()
            food.pop(idx)
            food_patches.pop(idx)
            food_trails.pop(idx)

            nf  = make_food_respawn(bh1, bh2)
            fp  = Circle(nf.x[1:3], FOOD_RADIUS, color=nf.color, zorder=3, alpha=0.85)
            ax.add_patch(fp)
            food.append(nf)
            food_patches.append(fp)
            fl, = ax.plot([], [], "-", color="#88ffaa", alpha=0.12, lw=0.5, zorder=2)
            food_trails.append(fl)

        # ── Update trails ──────────────────────────────────────────────────
        for bh, tl in ((bh1, bh1_trail), (bh2, bh2_trail)):
            bh.trail.append(bh.x[1:3].copy())
            if len(bh.trail) > TRAIL_LENGTH:
                bh.trail.pop(0)
            if len(bh.trail) > 1:
                pts = np.array(bh.trail)
                tl.set_data(pts[:, 0], pts[:, 1])

        for f, fl in zip(food, food_trails):
            f.trail.append(f.x[1:3].copy())
            if len(f.trail) > TRAIL_LENGTH // 2:
                f.trail.pop(0)
            if len(f.trail) > 1:
                pts = np.array(f.trail)
                fl.set_data(pts[:, 0], pts[:, 1])

        # ── Update patches ─────────────────────────────────────────────────
        for bh, patch, horiz in (
            (bh1, bh1_patch, bh1_horizon),
            (bh2, bh2_patch, bh2_horizon),
        ):
            pos = bh.x[1:3]
            patch.set_center(pos);   patch.set_radius(_bh_vis_radius(bh))
            patch.set_color(doppler_color(bh))
            horiz.set_center(pos);   horiz.set_radius(bh.r_s)

        for f, fp in zip(food, food_patches):
            fp.set_center(f.x[1:3])
            fp.set_color(doppler_color(f))

        # ── HUD ────────────────────────────────────────────────────────────
        t_coord   = max(bh1.x[0], bh2.x[0])
        eps       = max(t_coord, 1e-9)
        dil1      = bh1.tau / eps
        dil2      = bh2.tau / eps

        hud.set_text(
            f"coord time   t = {t_coord:8.1f}\n"
            f"\n"
            f"proper time  τ\n"
            f"  Cyan     {bh1.tau:7.2f}   τ/t={dil1:.4f}\n"
            f"  Magenta  {bh2.tau:7.2f}   τ/t={dil2:.4f}\n"
            f"\n"
            f"Lorentz γ\n"
            f"  Cyan     {bh1.lorentz_gamma():.4f}\n"
            f"  Magenta  {bh2.lorentz_gamma():.4f}\n"
            f"\n"
            f"speed\n"
            f"  Cyan     {bh1.speed():.4f} c\n"
            f"  Magenta  {bh2.speed():.4f} c\n"
            f"\n"
            f"mass eaten\n"
            f"  Cyan     {score[0]:.1f}\n"
            f"  Magenta  {score[1]:.1f}\n"
            f"\n"
            f"BH mass\n"
            f"  Cyan     {bh1.mass:.1f}   r_s={bh1.r_s:.1f}\n"
            f"  Magenta  {bh2.mass:.1f}   r_s={bh2.r_s:.1f}\n"
        )

        score_txt.set_text(f"Cyan  {score[0]:.1f} : {score[1]:.1f}  Magenta")

        frac = frame / GAME_FRAMES
        timer_bar.set_width(2 * FIELD_W * frac)

        # End-of-game message
        if frame == GAME_FRAMES:
            if score[0] > score[1]:
                win_msg = f"CYAN WINS\n{score[0]:.1f} vs {score[1]:.1f}"
            elif score[1] > score[0]:
                win_msg = f"MAGENTA WINS\n{score[0]:.1f} vs {score[1]:.1f}"
            else:
                win_msg = "DRAW"
            msg_txt.set_text(win_msg)
            msg_txt.set_alpha(1.0)

        # ── Save frame ─────────────────────────────────────────────────────
        fig.savefig(
            f"{OUT_DIR}/frame_{frame:05d}.png",
            dpi=80, bbox_inches="tight", facecolor=BG_COLOR,
        )

        if frame % 100 == 0:
            print(f"  frame {frame:4d}/{GAME_FRAMES}  "
                  f"Cyan {score[0]:.1f}  Magenta {score[1]:.1f}  "
                  f"t={t_coord:.1f}  τ_C={bh1.tau:.1f}  τ_M={bh2.tau:.1f}")

    plt.close(fig)
    print(f"\nDone.  Final score — Cyan: {score[0]:.1f}  Magenta: {score[1]:.1f}")
    print(f"Stitch frames:  ffmpeg -framerate {FPS} -i {OUT_DIR}/frame_%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p relativity.mp4")


if __name__ == "__main__":
    main()
