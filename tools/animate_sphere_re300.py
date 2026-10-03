#!/usr/bin/env python3
"""Animate the sphere wake at Re = 300: Q-criterion isosurfaces and the force history.

    animate_sphere_re300.py SOLUT_DIR FORCES.csv OUTPUT_BASE [--q 0.002] [--u 0.2]
        [--width 1920] [--fps 15] [--orbit 120] [--hold 2] [--gif-width 640]
        [--gif-every 2] [--t-start 600] [--prefix sphere]

SOLUT_DIR holds the volume VTU (or PVTU) series of examples/sphere_re300 with
U. Each snapshot is clipped to the near wake, Q = (|Omega|^2 - |S|^2) / 2 of
the velocity (interpolated to the vertices) is contoured at Q (in units of
(U / D)^2, U the free-stream speed) and colored by the streamwise velocity,
with the sphere, while the camera orbits the wake; the panels on the right
trace the drag and lift coefficients over the run, from t-start (in D / U)
on. Writes OUTPUT_BASE.mp4 (H.264, CRF 18), OUTPUT_BASE.gif (if --gif-width >
0) and a still of the last frame, OUTPUT_BASE_still.png. Needs pyvista besides
the packages in tools/README.md.
"""
import argparse
import glob
import os
import re
import tempfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from animate import write_gif, write_mp4
from plot_sphere_re300 import REFERENCES, load

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
BOUNDS = (-1.0, 12.0, -2.5, 2.5, -2.5, 2.5)


def snapshot_files(solut, prefix):
    text = open(os.path.join(solut, prefix + ".pvd")).read()
    entries = re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', text)
    return [(float(t), os.path.join(solut, f)) for t, f in entries if os.path.exists(os.path.join(solut, f))]


def q_surface(path, q_level):
    mesh = pv.read(path)
    mesh = mesh.clip_box(BOUNDS, invert=False, crinkle=True)
    mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    mesh = mesh.compute_derivative(scalars="U", gradient="G")
    g = np.asarray(mesh.point_data["G"]).reshape(-1, 3, 3)  # g[:, k, i] = d u_k / d x_i
    S = 0.5 * (g + g.transpose(0, 2, 1))
    W = 0.5 * (g - g.transpose(0, 2, 1))
    mesh.point_data["Q"] = 0.5 * ((W ** 2).sum((1, 2)) - (S ** 2).sum((1, 2)))
    mesh.point_data["UX"] = np.asarray(mesh.point_data["U"])[:, 0]
    return mesh.contour([q_level], scalars="Q")


def render_3d(plotter, surf, u, angle):
    plotter.clear_actors()
    if surf.n_points > 0:
        plotter.add_mesh(surf, scalars="UX", cmap="RdYlBu_r", clim=(-0.2 * u, 1.2 * u), smooth_shading=True,
                         specular=0.35, specular_power=20, show_scalar_bar=False)
    plotter.add_mesh(pv.Sphere(radius=0.5, theta_resolution=96, phi_resolution=48), color="#c9d1d9",
                     smooth_shading=True, specular=0.5)
    focus = np.array([4.0, 0.0, 0.0])
    a = np.radians(angle)
    r = 11.5
    eye = focus + r * np.array([-0.45, 0.89 * np.sin(a), 0.89 * np.cos(a)])
    up = np.array([0.0, np.cos(a), -np.sin(a)])
    plotter.camera_position = [tuple(eye), tuple(focus), tuple(up)]
    plotter.camera.view_angle = 32
    plotter.reset_camera_clipping_range()
    return plotter.screenshot(return_img=True)


def render_panels(t_now, tu, cd, cl, t_start, size, dpi, title_lines):
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG)
    fig.text(0.08, 0.94, title_lines[0], color=FG, fontsize=19, weight="bold")
    fig.text(0.08, 0.905, title_lines[1], color=MUTED, fontsize=12)
    fig.text(0.08, 0.875, f"t U / D = {t_now:6.1f}", color=FG, fontsize=15, family="monospace")
    m = (tu >= t_start) & (tu <= t_now)
    window = tu >= t_start
    for k, (y, name, ref, ylim) in enumerate([
            (cd, "$C_D$", REFERENCES[0][1], (0.62, 0.70)), (cl, "$C_L$", REFERENCES[0][2], (0.0, 0.14))]):
        ax = fig.add_axes([0.18, 0.49 - 0.41 * k, 0.76, 0.32], facecolor=BG)
        ax.axhline(ref, color=MUTED, lw=1.6, ls="--", label="Johnson & Patel 1999, mean")
        ax.plot(tu[m], y[m], color="#ff9e3d", lw=2.2, label="Mallard")
        if m.any():
            ax.plot(tu[m][-1], y[m][-1], "o", color="#ff9e3d", ms=8)
        ax.set_xlim(tu[window][0], tu[window][-1])
        ax.set_ylim(*ylim)
        ax.set_ylabel(name, color=FG, fontsize=16)
        ax.tick_params(colors=FG, labelsize=11)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, lw=0.8)
        if k == 0:
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc="upper right")
        else:
            ax.set_xlabel("$t\\,U/D$", color=FG, fontsize=15)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solut")
    ap.add_argument("forces")
    ap.add_argument("output")
    ap.add_argument("--prefix", default="sphere")
    ap.add_argument("--q", type=float, default=0.002, help="Isosurface level of Q, in (U / D)^2")
    ap.add_argument("--u", type=float, default=0.2, help="Free-stream speed")
    ap.add_argument("--t-start", type=float, default=600.0, help="Start of the force panels, in D / U")
    ap.add_argument("--subtitle", default="Q isosurfaces colored by streamwise velocity")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--orbit", type=float, default=120.0, help="Camera orbit about the stream axis, degrees")
    ap.add_argument("--hold", type=float, default=2.0, help="Seconds to hold the last frame")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-every", type=int, default=2, help="Use every n-th frame in the GIF")
    args = ap.parse_args()

    files = snapshot_files(args.solut, args.prefix)
    t, C = load(args.forces, args.u)
    tu = t * args.u
    direction = C[tu >= args.t_start, 1:].mean(0)
    direction /= np.linalg.norm(direction)
    cl = C[:, 1:] @ direction

    W, H = args.width, round(args.width * 9 / 16)
    w3d = round(W * 0.64)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(w3d, H))
    plotter.set_background(BG)
    plotter.enable_anti_aliasing("ssaa")
    title = ("Sphere at Re = 300, M = 0.2", args.subtitle)
    with tempfile.TemporaryDirectory() as tmp:
        for k, (t_file, path) in enumerate(files):
            surf = q_surface(path, args.q * (args.u / 1.0) ** 2)
            angle = args.orbit * k / max(len(files) - 1, 1)
            left = render_3d(plotter, surf, args.u, angle)
            right = render_panels(t_file * args.u, tu, C[:, 0], cl, args.t_start, (W - w3d, H), 100, title)
            frame = np.concatenate([left[:H, :w3d], right[:H]], axis=1)
            imageio.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
            if k % args.gif_every == 0:
                imageio.imwrite(os.path.join(tmp, f"g{k // args.gif_every:05d}.png"), frame)
            print(f"frame {k + 1}/{len(files)} t U / D = {t_file * args.u:.1f}", flush=True)
        n_gif = len(glob.glob(os.path.join(tmp, "g*.png")))
        for j in range(round(args.hold * args.fps)):
            imageio.imwrite(os.path.join(tmp, f"f{len(files) + j:05d}.png"), frame)
        for j in range(round(args.hold * args.fps / args.gif_every)):
            imageio.imwrite(os.path.join(tmp, f"g{n_gif + j:05d}.png"), frame)
        write_mp4(os.path.join(tmp, "f%05d.png"), args.output + ".mp4", args.fps)
        if args.gif_width > 0:
            write_gif(os.path.join(tmp, "g%05d.png"), args.output + ".gif", args.fps / args.gif_every,
                      args.gif_width, colors=128)
    imageio.imwrite(args.output + "_still.png", frame)
    print(f"wrote {args.output}.mp4")


if __name__ == "__main__":
    main()
