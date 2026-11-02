# Tutorial · 09 · Low-rank, SVD & why LoRA rank 8 is enough

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial walks through [`svd.py`](svd.py) in three parts: SVD on a 2×2 matrix, SVD on a photo, and LoRA on a small network written in NumPy. For each part it explains what is built, why, and how it works. The README has the tables.

**Goal:** answer the roadmap question *"why is a rank-8 LoRA update enough to fine-tune a big model?"* with your own measurements, before using LoRA for real in project 13.

**You need:** project 03 (matrices, rank, determinant), project 08 (softmax − one-hot gradient). NumPy, scikit-learn (digits dataset), Pillow.

---

## Part 1 · SVD on a 2×2 matrix (`svd_2x2`)

### Step 1 · Decompose and check

```python
M = np.array([[2.0, 1.0], [0.5, 1.5]])
U, S, Vt = np.linalg.svd(M)
reconstruction_error = abs(U @ np.diag(S) @ Vt - M).max()   # 7e-16
```

**What SVD says:** *every* matrix can be written as `M = U Σ Vᵀ`:
- `Vᵀ` is a rotation (or reflection),
- `Σ` is a diagonal stretch along the axes by the **singular values** σ₁ ≥ σ₂ ≥ … ≥ 0,
- `U` is another rotation.

So any linear map is "rotate, stretch along axes, rotate". Project 03 drew matrices as transformations; SVD says what every such transformation is made of.

### Step 2 · Watch it on the unit circle

```python
circle  = [cos t, sin t]
after_Vt = Vt @ circle            # still a circle (just turned)
after_S  = diag(S) @ Vt @ circle  # an ellipse with half-axes σ₁ = 2.56, σ₂ = 0.98
after_U  = M @ circle             # the ellipse turned into place
```

**Why the circle:** rotations don't change a circle, so the only visible change is the stretch. The singular values are literally the ellipse's half-axes. And since rotations keep area, the area scale is `σ₁·σ₂ = 2.50 = |det M|`, connecting back to project 03.

**What "rank" means now:** the number of non-zero singular values = the number of directions that survive. A singular matrix has σ₂ = 0 and squashes the plane onto a line.

---

## Part 2 · SVD on a photo (`image_svd`)

### Step 3 · Treat a photo as a matrix

```python
X = load_image()                         # 600×512 grayscale, values 0..1
U, S, Vt = np.linalg.svd(X, full_matrices=False)
```

`full_matrices=False` returns only the 512 directions that matter (the "thin" SVD), saving memory.

### Step 4 · Keep only the top k layers

```python
Xk = (U[:, :k] * S[:k]) @ Vt[:k]
```

**How it works:** SVD writes the image as a sum of rank-1 layers: `X = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + …`. Each layer is "a column pattern times a row pattern", weighted by σ. Keeping the first k gives the best possible approximation with k layers.

**Cost:** k·(600 + 512 + 1) numbers instead of 600·512.

**Energy:** `cumsum(S²) / sum(S²)` is the share of the image's "energy" (sum of squared pixel values) captured by the first k layers. Top 8 of 512: **91.7 % of the energy with 2.9 % of the numbers**, because the singular values fall off fast.

### Step 5 · Verify it is the *best* rank-k approximation

```python
err = np.linalg.norm(X - Xk)
eckart_young_gap = abs(err - np.sqrt(np.sum(S[k:] ** 2)))   # ≤ 1.4e-14
```

**Eckart–Young theorem:** no rank-k matrix is closer to X than `Xk`, and the error is exactly the root of the sum of the dropped σ². The code checks it for every k.

**Contrast with random directions:**
```python
Q, _ = np.linalg.qr(rng.normal(size=(n, 16)))   # 16 random orthonormal directions
random_err = norm(X - X @ Q @ Q.T) / norm(X)     # 0.978 vs 0.214 for top-16 SVD
```
Which directions you keep matters, not only how many. SVD finds the important ones.

---

## Part 3 · LoRA on a small network (`lora_experiment`)

### Step 6 · Build a pretrain task and a new task

```python
X = load_digits().images / 16.0                       # 8×8 handwritten digits
Xa = X.reshape(len(X), -1)                            # task A: normal digits
Xb = np.rot90(X, 1, axes=(1, 2)).reshape(len(X), -1)  # task B: same digits rotated by 90°
```

**Why this pair:** task B needs real adaptation (the pretrained model scores 10 %, i.e. chance, on rotated digits), but it is related to task A (same digits). That is exactly the fine-tuning situation: a model that knows something useful, and a new task.

### Step 7 · A small MLP with manual backprop in NumPy

```python
def forward(w, X):
    h_pre = X @ w["W1"] + w["b1"]
    h = np.maximum(h_pre, 0)                     # ReLU
    return h_pre, h, softmax(h @ w["W2"] + w["b2"])

def grads_wrt_weights(w, X, y):
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)   # softmax − one-hot (project 08)
    dh = (d @ w["W2"].T) * (h_pre > 0)                         # back through ReLU
    return {"W1": X.T @ dh, "b1": dh.sum(0), "W2": h.T @ d, "b2": d.sum(0)}
```

**Why NumPy, not PyTorch:** writing the gradients by hand makes the LoRA chain rule in Step 9 visible. The shapes follow project 03: `X (B,64) @ W1 (64,64) → (B,64)`.

**Optimizer:** a hand-written Adam (moving averages of the gradient and its square, with bias correction). Pretraining: 80 epochs on task A → 97.8 % test accuracy.

### Step 8 · LoRA: freeze W, learn a low-rank correction

```python
def effective(base, lora, scale):
    w["W1"] = base["W1"] + scale * lora["A1"] @ lora["B1"]    # (64×r)(r×64)
    w["W2"] = base["W2"] + scale * lora["A2"] @ lora["B2"]    # (64×r)(r×10)

A = rng.normal(0, 1/sqrt(d_in), (d_in, r))   # random
B = np.zeros((r, d_out))                     # zero
scale = alpha / r                            # alpha = 16
```

**How it works:** the pretrained `W` never changes. The update `ΔW = A·B` has rank at most `r`, because every column of it is a combination of A's `r` columns. Only A and B are trained.

**Why each detail:**
- **B = 0 at the start:** then ΔW = 0 exactly, so training begins from the pretrained model, not from a randomly disturbed one.
- **A random:** if both were zero, the gradient of each would be zero too (each gradient is multiplied by the other) and nothing would ever move.
- **`alpha / r` scaling:** keeps the size of the update roughly the same when you change `r`, so you don't need to re-tune the learning rate for every rank.

### Step 9 · The chain rule through `W + s·A·B`

```python
g = grads_wrt_weights(w_eff, X, y)              # dLoss/dW_eff, as if W_eff were a normal weight
dA = scale * g["W1"] @ B.T                      # (64×64)(64×r) → (64×r)
dB = scale * A.T @ g["W1"]                      # (r×64)(64×64) → (r×64)
```

**Why it works:** first compute the gradient for the effective weight as usual. Then `W_eff = W + s·A·B` is just a matrix product, and project 02's "multiplication swaps the inputs" rule gives the gradients for A and B. The shapes confirm it.

### Step 10 · Compare methods fairly

Every method trains 60 epochs with the same Adam settings, on 3 seeds:
- **head only** (train only the last layer, a "linear probe"): 650 params, 91.3 %,
- **LoRA r = 1, 2, 4, 8, 16, 32**: rank 8 → **95.9 % with 1616 params**,
- **full fine-tuning**: 4810 params, 96.9 %,
- **from scratch on B**: 97.6 % (the upper reference; this task has plenty of data).

**Why these baselines:** "head only" asks whether the old features are enough without touching them. "From scratch" asks whether pretraining helped at all. LoRA must sit between them to be interesting.

### Step 11 · Why rank 8 is enough: look at the full update itself

```python
dW = full["W1"] - base["W1"]                         # what full fine-tuning changed
U, S, Vt = np.linalg.svd(dW, full_matrices=False)
energy_top8 = sum(S[:8]**2) / sum(S**2)              # 86 %
w["W1"] = base["W1"] + (U[:, :k] * S[:k]) @ Vt[:k]   # keep only the top-k directions of ΔW
```

**This is the key experiment.** Full fine-tuning was free to change W₁ in all 64 directions. SVD of its update shows the spectrum drops just like the photo's: the top 8 of 64 directions hold 86 % of the change. Keeping only those 8 gives 94.7 % accuracy; 16 give 96.6 %; all 64 give 96.9 %.

**Conclusion:** the change a new task needs lives in a few directions (the "low intrinsic rank" from the LoRA paper). LoRA simply assumes it up front and learns only those directions.

### Step 12 · Scale up the arithmetic

```python
d, r = 4096, 8
lora_params = 2 * d * r     # 65 536
share = lora_params / d**2  # 0.39 %
```

On the toy network LoRA r = 8 trains 34 % of the parameters; on one 4096×4096 projection of a 7–8B LLM, only 0.39 %. **Savings grow with matrix size**, which is why LoRA matters for big models. At r = 32 on the toy, LoRA even has *more* parameters than full fine-tuning: LoRA only pays off when `r ≪` the matrix dimensions.

---

## Run it

```bash
uv run python svd.py     # ~40 s on CPU → results.json
uv run python plot.py    # assets/svd_2x2.png, image_svd.png, lora.png, svd_image.gif
```

**Why no Racing Lab chapter:** "a few directions hold most of the information" is clearest as a photo rebuilt layer by layer (the GIF), not as a car.

## Try it yourself

1. Change task B to a 180° rotation or a horizontal flip. Is the full update still low-rank?
2. Set `B = rng.normal(...)` instead of zeros. What is the accuracy on task B *before* any LoRA training?
3. Apply LoRA only to W₁ (leave W₂ frozen and without A·B). How much accuracy do you lose at r = 8?

## Recap

| Step | Idea |
|---|---|
| 1–2 | SVD: every matrix = rotate · stretch · rotate; σ = ellipse half-axes; rank = non-zero σ |
| 4–5 | Truncated SVD is the best rank-k approximation; error = dropped σ |
| 8 | LoRA: frozen W + s·A·B; B = 0 start; α/r scaling |
| 9 | Gradients for A and B by the chain rule through a matrix product |
| 11 | The update full fine-tuning finds is itself low-rank: that is why r = 8 works |
| 12 | Parameter savings grow with matrix size (0.39 % on a 4096² layer) |
