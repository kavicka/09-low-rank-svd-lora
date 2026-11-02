# Tutoriál · 09 · Nízky rank, SVD a prečo stačí LoRA s rankom 8

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál prejde [`svd.py`](svd.py) v troch častiach: SVD na matici 2×2, SVD na fotke a LoRA na malej sieti napísanej v NumPy. Pri každej časti vysvetlí, čo sa stavia, prečo a ako to funguje. README má tabuľky.

**Cieľ:** odpovedať na otázku z roadmapy *„prečo stačí LoRA update s rankom 8 na fine-tuning veľkého modelu?“* vlastnými meraniami, skôr než LoRA naozaj použiješ v projekte 13.

**Potrebuješ:** projekt 03 (matice, rank, determinant), projekt 08 (gradient softmax − one-hot). NumPy, scikit-learn (dataset číslic), Pillow.

---

## Časť 1 · SVD na matici 2×2 (`svd_2x2`)

### Krok 1 · Rozlož a over

```python
M = np.array([[2.0, 1.0], [0.5, 1.5]])
U, S, Vt = np.linalg.svd(M)
reconstruction_error = abs(U @ np.diag(S) @ Vt - M).max()   # 7e-16
```

**Čo hovorí SVD:** *každá* matica sa dá zapísať ako `M = U Σ Vᵀ`:
- `Vᵀ` je rotácia (alebo zrkadlenie),
- `Σ` je diagonálne natiahnutie pozdĺž osí o **singulárne hodnoty** σ₁ ≥ σ₂ ≥ … ≥ 0,
- `U` je ďalšia rotácia.

Každé lineárne zobrazenie je teda „otoč, natiahni pozdĺž osí, otoč“. Projekt 03 kreslil matice ako transformácie; SVD hovorí, z čoho sa každá taká transformácia skladá.

### Krok 2 · Pozri sa na to na jednotkovej kružnici

```python
circle  = [cos t, sin t]
after_Vt = Vt @ circle            # stále kružnica (len otočená)
after_S  = diag(S) @ Vt @ circle  # elipsa s poloosami σ₁ = 2,56, σ₂ = 0,98
after_U  = M @ circle             # elipsa otočená na miesto
```

**Prečo kružnica:** rotácie kružnicu nezmenia, takže jediná viditeľná zmena je natiahnutie. Singulárne hodnoty sú doslova poloosi elipsy. A keďže rotácie zachovávajú plochu, zmena plochy je `σ₁·σ₂ = 2,50 = |det M|`, čo nadväzuje na projekt 03.

**Čo teraz znamená „rank“:** počet nenulových singulárnych hodnôt = počet smerov, ktoré prežijú. Singulárna matica má σ₂ = 0 a zmačká rovinu na priamku.

---

## Časť 2 · SVD na fotke (`image_svd`)

### Krok 3 · Ber fotku ako maticu

```python
X = load_image()                         # 600×512 v odtieňoch sivej, hodnoty 0..1
U, S, Vt = np.linalg.svd(X, full_matrices=False)
```

`full_matrices=False` vráti len 512 smerov, na ktorých záleží („tenké“ SVD), a šetrí pamäť.

### Krok 4 · Nechaj len prvých k vrstiev

```python
Xk = (U[:, :k] * S[:k]) @ Vt[:k]
```

**Ako to funguje:** SVD zapíše obrázok ako súčet vrstiev ranku 1: `X = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + …`. Každá vrstva je „vzor stĺpca krát vzor riadku“, vážená σ. Prvých k dá najlepšiu možnú aproximáciu s k vrstvami.

**Cena:** k·(600 + 512 + 1) čísel namiesto 600·512.

**Energia:** `cumsum(S²) / sum(S²)` je podiel „energie“ obrázka (súčet štvorcov hodnôt pixelov), ktorý zachytí prvých k vrstiev. Top 8 z 512: **91,7 % energie s 2,9 % čísel**, lebo singulárne hodnoty rýchlo klesajú.

### Krok 5 · Over, že je to *najlepšia* aproximácia ranku k

```python
err = np.linalg.norm(X - Xk)
eckart_young_gap = abs(err - np.sqrt(np.sum(S[k:] ** 2)))   # ≤ 1,4e-14
```

**Veta Eckart–Young:** žiadna matica ranku k nie je bližšie k X než `Xk` a chyba je presne odmocnina súčtu zahodených σ². Kód to overí pre každé k.

**Kontrast s náhodnými smermi:**
```python
Q, _ = np.linalg.qr(rng.normal(size=(n, 16)))   # 16 náhodných ortonormálnych smerov
random_err = norm(X - X @ Q @ Q.T) / norm(X)     # 0,978 vs 0,214 pre top-16 SVD
```
Záleží na tom, ktoré smery si necháš, nie len koľko. SVD nájde tie dôležité.

---

## Časť 3 · LoRA na malej sieti (`lora_experiment`)

### Krok 6 · Postav úlohu na predtrénovanie a novú úlohu

```python
X = load_digits().images / 16.0                       # ručne písané číslice 8×8
Xa = X.reshape(len(X), -1)                            # úloha A: normálne číslice
Xb = np.rot90(X, 1, axes=(1, 2)).reshape(len(X), -1)  # úloha B: tie isté číslice otočené o 90°
```

**Prečo táto dvojica:** úloha B potrebuje skutočnú adaptáciu (predtrénovaný model má na otočených číslach 10 %, teda náhodu), ale súvisí s úlohou A (tie isté číslice). Presne to je situácia fine-tuningu: model, ktorý vie niečo užitočné, a nová úloha.

### Krok 7 · Malé MLP s ručným backpropom v NumPy

```python
def forward(w, X):
    h_pre = X @ w["W1"] + w["b1"]
    h = np.maximum(h_pre, 0)                     # ReLU
    return h_pre, h, softmax(h @ w["W2"] + w["b2"])

def grads_wrt_weights(w, X, y):
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)   # softmax − one-hot (projekt 08)
    dh = (d @ w["W2"].T) * (h_pre > 0)                         # späť cez ReLU
    return {"W1": X.T @ dh, "b1": dh.sum(0), "W2": h.T @ d, "b2": d.sum(0)}
```

**Prečo NumPy a nie PyTorch:** ručne písané gradienty zviditeľnia reťazové pravidlo pre LoRA v Kroku 9. Tvary podľa projektu 03: `X (B,64) @ W1 (64,64) → (B,64)`.

**Optimalizátor:** ručne napísaný Adam (kĺzavé priemery gradientu a jeho štvorca, s korekciou skreslenia). Predtrénovanie: 80 epoch na úlohe A → 97,8 % presnosť na teste.

### Krok 8 · LoRA: zmraz W, nauč sa korekciu nízkeho ranku

```python
def effective(base, lora, scale):
    w["W1"] = base["W1"] + scale * lora["A1"] @ lora["B1"]    # (64×r)(r×64)
    w["W2"] = base["W2"] + scale * lora["A2"] @ lora["B2"]    # (64×r)(r×10)

A = rng.normal(0, 1/sqrt(d_in), (d_in, r))   # náhodné
B = np.zeros((r, d_out))                     # nulové
scale = alpha / r                            # alpha = 16
```

**Ako to funguje:** predtrénované `W` sa nikdy nezmení. Update `ΔW = A·B` má rank najviac `r`, lebo každý jeho stĺpec je kombinácia `r` stĺpcov matice A. Trénujú sa len A a B.

**Prečo každý detail:**
- **B = 0 na začiatku:** potom ΔW = 0 presne, takže tréning začína z predtrénovaného modelu, nie z náhodne narušeného.
- **A náhodné:** keby boli obe nulové, gradient každej by bol tiež nulový (každý gradient sa násobí tou druhou) a nič by sa nepohlo.
- **Škálovanie `alpha / r`:** drží veľkosť updatu zhruba rovnakú, keď meníš `r`, takže nemusíš ladiť learning rate pre každý rank.

### Krok 9 · Reťazové pravidlo cez `W + s·A·B`

```python
g = grads_wrt_weights(w_eff, X, y)              # dLoss/dW_eff, ako keby W_eff bola obyčajná váha
dA = scale * g["W1"] @ B.T                      # (64×64)(64×r) → (64×r)
dB = scale * A.T @ g["W1"]                      # (r×64)(64×64) → (r×64)
```

**Prečo to funguje:** najprv spočítaj gradient pre efektívnu váhu ako obvykle. Potom `W_eff = W + s·A·B` je len súčin matíc a pravidlo z projektu 02 „násobenie vymení vstupy“ dá gradienty pre A a B. Tvary to potvrdzujú.

### Krok 10 · Porovnaj metódy férovo

Každá metóda trénuje 60 epoch s rovnakým nastavením Adamu, na 3 seedoch:
- **len hlava** (trénuje sa len posledná vrstva, „linear probe“): 650 parametrov, 91,3 %,
- **LoRA r = 1, 2, 4, 8, 16, 32**: rank 8 → **95,9 % so 1616 parametrami**,
- **plný fine-tuning**: 4810 parametrov, 96,9 %,
- **od nuly na B**: 97,6 % (horná referencia; táto úloha má dosť dát).

**Prečo tieto baseliny:** „len hlava“ sa pýta, či staré príznaky stačia bez toho, aby sa ich niekto dotkol. „Od nuly“ sa pýta, či predtrénovanie vôbec pomohlo. LoRA musí byť medzi nimi, aby bola zaujímavá.

### Krok 11 · Prečo stačí rank 8: pozri sa na samotný plný update

```python
dW = full["W1"] - base["W1"]                         # čo zmenil plný fine-tuning
U, S, Vt = np.linalg.svd(dW, full_matrices=False)
energy_top8 = sum(S[:8]**2) / sum(S**2)              # 86 %
w["W1"] = base["W1"] + (U[:, :k] * S[:k]) @ Vt[:k]   # nechaj len top-k smerov ΔW
```

**Toto je kľúčový experiment.** Plný fine-tuning mohol meniť W₁ vo všetkých 64 smeroch. SVD jeho updatu ukáže, že spektrum klesá rovnako ako pri fotke: top 8 zo 64 smerov drží 86 % zmeny. Keď necháš len týchto 8, presnosť je 94,7 %; 16 dá 96,6 %; všetkých 64 dá 96,9 %.

**Záver:** zmena, ktorú nová úloha potrebuje, žije v pár smeroch („nízky vnútorný rank“ z LoRA článku). LoRA to jednoducho predpokladá vopred a učí sa len tieto smery.

### Krok 12 · Prepočítaj to na veľký model

```python
d, r = 4096, 8
lora_params = 2 * d * r     # 65 536
share = lora_params / d**2  # 0,39 %
```

Na malej sieti LoRA r = 8 trénuje 34 % parametrov; na jednej projekcii 4096×4096 v LLM so 7–8 mld. parametrov len 0,39 %. **Úspora rastie s veľkosťou matice**, a preto je LoRA dôležitá pri veľkých modeloch. Pri r = 32 má LoRA na malej sieti dokonca *viac* parametrov než plný fine-tuning: LoRA sa oplatí, len keď `r ≪` rozmery matice.

---

## Spusti

```bash
uv run python svd.py     # ~40 s na CPU → results.json
uv run python plot.py    # assets/svd_2x2.png, image_svd.png, lora.png, svd_image.gif
```

**Prečo nie kapitola v Racing Lab:** „pár smerov drží väčšinu informácie“ je najjasnejšie na fotke skladanej vrstvu po vrstve (GIF), nie na autíčku.

## Vyskúšaj sám

1. Zmeň úlohu B na otočenie o 180° alebo vodorovné zrkadlenie. Je plný update stále nízkeho ranku?
2. Nastav `B = rng.normal(...)` namiesto núl. Aká je presnosť na úlohe B *pred* akýmkoľvek tréningom LoRA?
3. Použi LoRA len na W₁ (W₂ nechaj zmrazené bez A·B). Koľko presnosti stratíš pri r = 8?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1–2 | SVD: každá matica = otoč · natiahni · otoč; σ = poloosi elipsy; rank = nenulové σ |
| 4–5 | Orezané SVD je najlepšia aproximácia ranku k; chyba = zahodené σ |
| 8 | LoRA: zmrazené W + s·A·B; štart s B = 0; škálovanie α/r |
| 9 | Gradienty pre A a B reťazovým pravidlom cez súčin matíc |
| 11 | Update, ktorý nájde plný fine-tuning, má sám nízky rank: preto r = 8 funguje |
| 12 | Úspora parametrov rastie s veľkosťou matice (0,39 % na vrstve 4096²) |
