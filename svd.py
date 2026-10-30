"""Step 09 · Low-rank and SVD intuition: why a rank-8 LoRA update is enough.

Three experiments, NumPy only, fixed seeds (every number in the README comes from results.json):
1. SVD of a 2×2 matrix: every matrix = rotate · stretch · rotate (the unit circle becomes an ellipse).
2. SVD of a photo: keep the k largest singular values. How much of the image is in the first few?
   Eckart–Young: the truncated SVD is the best rank-k approximation, error = √(sum of dropped σ²).
3. LoRA on a small MLP: pretrain on handwritten digits, then adapt to digits rotated by 90°.
   Full fine-tuning vs a frozen base + low-rank update ΔW = A·B of rank r = 1 … 32.
   Then look at the singular values of the full fine-tuning update itself.

Run:  uv run python svd.py   → results.json
      uv run python plot.py  → assets/*.png, assets/svd_image.gif
"""

import json
from pathlib import Path

import numpy as np
from matplotlib import cbook
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEEDS = (0, 1, 2)
HIDDEN = 64
RANKS = (1, 2, 4, 8, 16, 32)


# --- 1. 2×2 --------------------------------------------------------------------------------
def svd_2x2() -> dict:
    M = np.array([[2.0, 1.0], [0.5, 1.5]])
    U, S, Vt = np.linalg.svd(M)
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)])
    return {
        "M": M.tolist(),
        "U": U.tolist(),
        "S": S.tolist(),
        "Vt": Vt.tolist(),
        "reconstruction_error": float(np.abs(U @ np.diag(S) @ Vt - M).max()),
        "det": float(np.linalg.det(M)),
        "product_of_singular_values": float(S.prod()),
        "circle": circle.tolist(),
        "after_Vt": (Vt @ circle).tolist(),
        "after_S": (np.diag(S) @ Vt @ circle).tolist(),
        "after_U": (M @ circle).tolist(),
    }


# --- 2. Image -----------------------------------------------------------------------------
def load_image() -> np.ndarray:
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        return np.asarray(Image.open(f).convert("L"), float) / 255.0


def image_svd(ks=(1, 2, 4, 8, 16, 32, 64, 128)) -> dict:
    X = load_image()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    m, n = X.shape
    energy = np.cumsum(S**2) / np.sum(S**2)
    rows = []
    for k in ks:
        Xk = (U[:, :k] * S[:k]) @ Vt[:k]
        err = np.linalg.norm(X - Xk)
        rows.append(
            {
                "k": k,
                "params": k * (m + n + 1),
                "share_of_pixels": k * (m + n + 1) / (m * n),
                "relative_error": float(err / np.linalg.norm(X)),
                "energy": float(energy[k - 1]),
                # Eckart–Young: the error of the best rank-k approximation is exactly the dropped singular values
                "eckart_young_gap": float(abs(err - np.sqrt(np.sum(S[k:] ** 2)))),
            }
        )
    # SVD vs a naive rank-k approximation (keep k random rows' worth of projection): SVD must win
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(n, 16)))
    random_err = float(np.linalg.norm(X - X @ Q @ Q.T) / np.linalg.norm(X))
    return {
        "shape": [m, n],
        "singular_values": S.tolist(),
        "energy": energy.tolist(),
        "rows": rows,
        "random_rank16_relative_error": random_err,
        "approximations": {str(k): ((U[:, :k] * S[:k]) @ Vt[:k]).clip(0, 1).round(3).tolist() for k in (1, 4, 16, 64)},
    }


# --- 3. LoRA ----------------------------------------------------------------------------------
def digits():
    d = load_digits()
    X = d.images / 16.0
    Xa = X.reshape(len(X), -1)
    Xb = np.rot90(X, 1, axes=(1, 2)).reshape(len(X), -1)  # task B: the same digits rotated by 90°
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.3, stratify=d.target, random_state=0)
    return Xa, Xb, d.target, tr, te


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class Adam:
    def __init__(self, params: dict, lr: float):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g**2
            mh, vh = self.m[k] / (1 - 0.9**self.t), self.v[k] / (1 - 0.999**self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def effective(base: dict, lora: dict | None, scale: float) -> dict:
    """W_eff = W + scale · A·B. The base W stays frozen; only A (in×r) and B (r×out) are trained."""
    w = dict(base)
    if lora:
        w["W1"] = base["W1"] + scale * lora["A1"] @ lora["B1"]
        w["W2"] = base["W2"] + scale * lora["A2"] @ lora["B2"]
    return w


def forward(w: dict, X):
    h_pre = X @ w["W1"] + w["b1"]
    h = np.maximum(h_pre, 0)
    return h_pre, h, softmax(h @ w["W2"] + w["b2"])


def grads_wrt_weights(w: dict, X, y) -> dict:
    """Manual backprop of mean cross-entropy (step 08: d loss / d logits = softmax − one-hot)."""
    h_pre, h, p = forward(w, X)
    d = p.copy()
    d[np.arange(len(y)), y] -= 1
    d /= len(y)
    dh = (d @ w["W2"].T) * (h_pre > 0)
    return {"W1": X.T @ dh, "b1": dh.sum(0), "W2": h.T @ d, "b2": d.sum(0)}


def accuracy(w: dict, X, y) -> float:
    return float((forward(w, X)[2].argmax(1) == y).mean())


def train(
    X,
    y,
    rng,
    base: dict | None = None,
    rank: int | None = None,
    epochs: int = 60,
    lr: float = 3e-3,
    batch: int = 64,
    alpha: float = 16.0,
):
    """No base → train from scratch. base + rank=None → full fine-tuning. base + rank=r → LoRA."""
    if base is None:
        params = {
            "W1": rng.normal(0, np.sqrt(2 / X.shape[1]), (X.shape[1], HIDDEN)),
            "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 10)),
            "b2": np.zeros(10),
        }
    elif rank is None:
        params = {k: v.copy() for k, v in base.items()}
    else:  # LoRA init as in the paper: A random, B zero → the update starts at exactly 0
        params = {
            "A1": rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], rank)),
            "B1": np.zeros((rank, HIDDEN)),
            "A2": rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, rank)),
            "B2": np.zeros((rank, 10)),
        }
    scale = alpha / rank if rank else 1.0
    opt = Adam(params, lr)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            b = order[i : i + batch]
            if rank is None:
                opt.step(params, grads_wrt_weights(params, X[b], y[b]))
                continue
            w = effective(base, params, scale)
            g = grads_wrt_weights(w, X[b], y[b])
            # chain rule through W_eff = W + s·A·B:  dA = s · dW · Bᵀ,  dB = s · Aᵀ · dW
            opt.step(
                params,
                {
                    "A1": scale * g["W1"] @ params["B1"].T,
                    "B1": scale * params["A1"].T @ g["W1"],
                    "A2": scale * g["W2"] @ params["B2"].T,
                    "B2": scale * params["A2"].T @ g["W2"],
                },
            )
    return params, (effective(base, params, scale) if rank else params)


def n_params(p: dict) -> int:
    return int(sum(v.size for v in p.values()))


def lora_experiment() -> dict:
    Xa, Xb, y, tr, te = digits()
    runs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        _, base = train(Xa[tr], y[tr], rng, epochs=80)
        r = {
            "seed": seed,
            "pretrained_acc_A": accuracy(base, Xa[te], y[te]),
            "pretrained_acc_B": accuracy(base, Xb[te], y[te]),
            "methods": {},
        }
        # baselines: train only the output layer (a linear probe) and train from scratch on B
        probe_params = {"W2": base["W2"].copy(), "b2": base["b2"].copy()}
        opt = Adam(probe_params, 3e-3)
        for _ in range(60):
            order = rng.permutation(len(tr))
            for i in range(0, len(tr), 64):
                b = tr[order[i : i + 64]]
                g = grads_wrt_weights(base | probe_params, Xb[b], y[b])
                opt.step(probe_params, {"W2": g["W2"], "b2": g["b2"]})
        probe = base | probe_params
        r["methods"]["head only"] = {
            "trainable": n_params(probe_params),
            "acc_B": accuracy(probe, Xb[te], y[te]),
            "acc_A": accuracy(probe, Xa[te], y[te]),
        }
        scratch_p, scratch = train(Xb[tr], y[tr], rng, epochs=60)
        r["methods"]["from scratch on B"] = {
            "trainable": n_params(scratch_p),
            "acc_B": accuracy(scratch, Xb[te], y[te]),
            "acc_A": accuracy(scratch, Xa[te], y[te]),
        }
        full_p, full = train(Xb[tr], y[tr], rng, base=base, epochs=60)
        r["methods"]["full fine-tuning"] = {
            "trainable": n_params(full_p),
            "acc_B": accuracy(full, Xb[te], y[te]),
            "acc_A": accuracy(full, Xa[te], y[te]),
        }
        for rank in RANKS:
            lp, w = train(Xb[tr], y[tr], rng, base=base, rank=rank, epochs=60)
            r["methods"][f"LoRA r={rank}"] = {
                "trainable": n_params(lp),
                "acc_B": accuracy(w, Xb[te], y[te]),
                "acc_A": accuracy(w, Xa[te], y[te]),
            }
        # how low-rank is the update that FULL fine-tuning found on its own?
        dW = full["W1"] - base["W1"]
        S = np.linalg.svd(dW, compute_uv=False)
        U, S_, Vt = np.linalg.svd(dW, full_matrices=False)
        trunc = {}
        for k in (1, 2, 4, 8, 16, 32, 64):
            w = dict(full)
            w["W1"] = base["W1"] + (U[:, :k] * S_[:k]) @ Vt[:k]  # keep only the top-k directions of ΔW1
            trunc[str(k)] = accuracy(w, Xb[te], y[te])
        r["full_update"] = {
            "singular_values_W1": S.tolist(),
            "energy_top8": float(np.sum(S[:8] ** 2) / np.sum(S**2)),
            "acc_B_with_dW1_truncated": trunc,
            "rank_W1_shape": list(dW.shape),
        }
        runs.append(r)
        print(
            f"seed {seed}: A {r['pretrained_acc_A']:.3f} B(before) {r['pretrained_acc_B']:.3f}  "
            + "  ".join(f"{k} {v['acc_B']:.3f}" for k, v in r["methods"].items())
        )
    summary = {}
    for k in runs[0]["methods"]:
        b = [r["methods"][k]["acc_B"] for r in runs]
        a = [r["methods"][k]["acc_A"] for r in runs]
        summary[k] = {
            "trainable": runs[0]["methods"][k]["trainable"],
            "acc_B_mean": float(np.mean(b)),
            "acc_B_std": float(np.std(b)),
            "acc_A_mean": float(np.mean(a)),
        }
    trunc_mean = {
        k: float(np.mean([r["full_update"]["acc_B_with_dW1_truncated"][k] for r in runs]))
        for k in runs[0]["full_update"]["acc_B_with_dW1_truncated"]
    }
    return {
        "hidden": HIDDEN,
        "seeds": list(SEEDS),
        "train": len(tr),
        "test": len(te),
        "runs": runs,
        "summary": summary,
        "trunc_mean": trunc_mean,
        "energy_top8_mean": float(np.mean([r["full_update"]["energy_top8"] for r in runs])),
        "pretrained_acc_A_mean": float(np.mean([r["pretrained_acc_A"] for r in runs])),
        "pretrained_acc_B_mean": float(np.mean([r["pretrained_acc_B"] for r in runs])),
    }


def main() -> None:
    d, r = 4096, 8  # one attention projection of a 7–8B model, LoRA rank 8
    llm = {"d": d, "rank": r, "full_params": d * d, "lora_params": 2 * d * r, "share": 2 * d * r / (d * d)}
    res = {"svd_2x2": svd_2x2(), "image": image_svd(), "lora": lora_experiment(), "llm_matrix_example": llm}
    s = res["svd_2x2"]
    print(f"2x2: S {np.round(s['S'], 3)}  |det| {abs(s['det']):.3f} = σ1σ2 {s['product_of_singular_values']:.3f}")
    for r in res["image"]["rows"]:
        print(
            f"image k={r['k']:4d}: params {r['share_of_pixels']:.1%} of pixels  energy {r['energy']:.4f}  "
            f"rel. error {r['relative_error']:.3f}  EY gap {r['eckart_young_gap']:.1e}"
        )
    print(f"random rank-16 projection rel. error {res['image']['random_rank16_relative_error']:.3f}")
    lo = res["lora"]
    for k, v in lo["summary"].items():
        print(
            f"{k:20s} trainable {v['trainable']:6d}  acc B {v['acc_B_mean']:.3f}±{v['acc_B_std']:.3f}  acc A {v['acc_A_mean']:.3f}"
        )
    print("full-FT dW1 truncated:", lo["trunc_mean"], " energy top 8:", round(lo["energy_top8_mean"], 3))
    Path("results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
