"""Turn results.json into the plots and the GIF used in the README."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from svd import load_image

R = json.loads(Path("results.json").read_text())
Path("assets").mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def two_by_two() -> None:
    s = R["svd_2x2"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    steps = [
        ("circle", "unit circle"),
        ("after_Vt", "1. rotate/reflect (Vᵀ)"),
        ("after_S", f"2. stretch (Σ = {s['S'][0]:.2f}, {s['S'][1]:.2f})"),
        ("after_U", "3. rotate/reflect (U) = M · circle"),
    ]
    for a, (k, title) in zip(axes, steps):
        pts = np.array(s[k])
        a.plot(pts[0], pts[1], color=BLUE, lw=2)
        a.plot(pts[0, :1], pts[1, :1], "o", color=RED)  # one tracked point
        a.axhline(0, color=GRAY, lw=0.6)
        a.axvline(0, color=GRAY, lw=0.6)
        a.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=title)
        a.grid(alpha=0.3)
    fig.suptitle(f"SVD: M = U Σ Vᵀ with M = {s['M']}. Every matrix is rotate · stretch · rotate", fontsize=11)
    save(fig, "svd_2x2.png")


def image() -> None:
    im = R["image"]
    S, rows = np.array(im["singular_values"]), im["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a1.semilogy(np.arange(1, len(S) + 1), S, color=BLUE, lw=2)
    a1.set(
        xlabel="index i",
        ylabel="singular value σᵢ (log)",
        title=f"Singular values of a {im['shape'][0]}×{im['shape'][1]} photo",
    )
    a1.grid(alpha=0.3, which="both")
    ks = [r["k"] for r in rows]
    a2.plot(ks, [r["energy"] for r in rows], "o-", color=GREEN, lw=2, label="energy kept Σσᵢ² (top k) / Σσᵢ²")
    a2.plot(ks, [r["share_of_pixels"] for r in rows], "s--", color=ORANGE, lw=1.5, label="numbers stored / pixels")
    a2.plot(ks, [r["relative_error"] for r in rows], "^-", color=RED, lw=1.5, label="relative error ‖X − Xₖ‖ / ‖X‖")
    a2.set(xscale="log", xlabel="rank k", ylim=(0, 1.02), title="Rank-k approximation")
    a2.set_xticks(ks, [str(k) for k in ks])
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    save(fig, "image_svd.png")


def lora() -> None:
    lo = R["lora"]
    s = lo["summary"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.6))
    ranks = [k for k in s if k.startswith("LoRA")]
    a1.errorbar(
        [s[k]["trainable"] for k in ranks],
        [s[k]["acc_B_mean"] for k in ranks],
        yerr=[s[k]["acc_B_std"] for k in ranks],
        fmt="o-",
        color=BLUE,
        lw=2,
        capsize=3,
        label="LoRA, rank r",
    )
    for k in ranks:
        a1.annotate(
            k.split()[1],
            (s[k]["trainable"], s[k]["acc_B_mean"]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
        )
    for k, col, mk in [("full fine-tuning", RED, "s"), ("head only", ORANGE, "D"), ("from scratch on B", GRAY, "^")]:
        a1.plot([s[k]["trainable"]], [s[k]["acc_B_mean"]], mk, color=col, ms=9, label=k)
    a1.set(
        xscale="log",
        xlabel="trainable parameters (log)",
        ylabel="test accuracy on rotated digits",
        title="Adapting to rotated digits: accuracy vs parameters",
        ylim=(0.75, 1.0),
    )
    a1.xaxis.set_minor_formatter(plt.NullFormatter())
    a1.legend(fontsize=8, loc="lower right")
    a1.grid(alpha=0.3, which="both")

    for r, col in zip(lo["runs"], [BLUE, GREEN, ORANGE]):
        S = np.array(r["full_update"]["singular_values_W1"])
        a2.plot(np.arange(1, len(S) + 1), S / S[0], "o-", ms=3, color=col, lw=1.2, label=f"seed {r['seed']}")
    a2.axvline(8, color=DARK, ls="--", lw=1)
    a2.set(
        xlabel="index i",
        ylabel="σᵢ / σ₁",
        title=f"Singular values of ΔW₁ from FULL fine-tuning\n(top 8 hold {lo['energy_top8_mean']:.0%} of the energy)",
    )
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)

    t = lo["trunc_mean"]
    ks = [int(k) for k in t]
    a3.plot(ks, list(t.values()), "o-", color=RED, lw=2, label="full FT with ΔW₁ cut to rank k")
    a3.axhline(s["full fine-tuning"]["acc_B_mean"], color=GRAY, ls="--", lw=1, label="full fine-tuning, all ranks")
    a3.set(
        xscale="log",
        xlabel="rank k kept from ΔW₁",
        ylabel="test accuracy on rotated digits",
        ylim=(0.3, 1.0),
        title="Keep only the top-k directions of the update",
    )
    a3.set_xticks(ks, [str(k) for k in ks])
    a3.legend(fontsize=8, loc="lower right")
    a3.grid(alpha=0.3)
    save(fig, "lora.png")


def gif() -> None:
    X = load_image()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    m, n = X.shape
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 200]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 5))
    img = a1.imshow(X, cmap="gray", vmin=0, vmax=1)
    a1.axis("off")
    a2.imshow(X, cmap="gray", vmin=0, vmax=1)
    a2.set_title("original", fontsize=10)
    a2.axis("off")

    def frame(i):
        k = ks[min(i, len(ks) - 1)]
        img.set_data(((U[:, :k] * S[:k]) @ Vt[:k]).clip(0, 1))
        energy = np.sum(S[:k] ** 2) / np.sum(S**2)
        a1.set_title(
            f"rank {k}: {k * (m + n + 1) / (m * n):.1%} of the numbers, {energy:.1%} of the energy", fontsize=10
        )
        return [img]

    fig.tight_layout()
    FuncAnimation(fig, frame, frames=len(ks) + 6).save("assets/svd_image.gif", writer=PillowWriter(fps=2), dpi=70)


if __name__ == "__main__":
    two_by_two()
    image()
    lora()
    gif()
