# Defense Strategies — AEGIS LILITH v4

> *A magician explaining three tricks in detail while performing a fourth behind your back.*

---

## What You Will Learn Here

This document explains the architectural strategy of LILITH. It reveals enough to demonstrate mastery. It hides enough to maintain the edge.

**SHOWN:** The Knuth-mask architecture, the 5 Constants, the fat hyperplane barrier.
**HIDDEN:** PRF gate thresholds, Ghost Code transition intensities, Moloch Token internal format, per-coordinate isotopy derivation, exact knuth_mask hash construction.

---

## Strategy 1: The Knuth-Mask Innovation

### The Problem

LILITH v1 used Knuth semifield multiplication directly on column values:
```python
new_val = knuth_mul(col_value, seed, twist)  # NON-LINEAR
```
This was cryptographically strong but **destroyed the gap** (statistical indistinguishability). Gap went to 0.205.

### The Solution (Consensus of 3 Auditors)

Use Knuth multiplication to **generate the perturbation delta,** not to transform the column:
```python
delta = knuth_mask(secret, transcript, query_count, coord, purpose)
# delta comes from: knuth_mul(PRF_seed_a, PRF_seed_b, twist_per_coord)
# delta is NON-LINEAR (from semifield) but INDEPENDENT of column value
col[coord] = gf4_add(col[coord], delta)  # Gap-neutral application
```

Each coordinate gets its **own isotopy twist** derived from PRF(secret, coord). This means each of the 12 coordinates lives in a **different semifield.** An attacker cannot construct a single consistent algebra.

### The v4 Refinement

v3 used `(mask % 3) + 1` which collapsed 16 semifield states to 3 values via modular arithmetic. v4 uses XOR of high and low nibbles for full 4-bit mixing before reducing to {1,2,3}. Zero information loss.

---

## Strategy 2: The Gravitational Architecture

The defense is layered like gravitational fields around a black hole:

**Far field (rank 0-4):** Standard GORGON+AZAZEL corruption. The attacker sees noise. Normal.

**Near field (rank 5-8):** FENRIR Mordidas activate. Behavioral classification, softmax venom, escalation. The attacker encounters resistance. Expected.

**Horizon approach (rank 9-10):** Sovereignty layers activate. Tananiel Circle 1 (paradoxical truth), Tananiel Circle 3 (The Void — isotopy switch, δ=61.2% learned knowledge becomes wrong). The attacker notices something is off. Too late.

**Inside the horizon (rank 11-12):** Ghost Code. LILITH stops fighting and starts giving. The attacker solves perfectly — a **phantom dual code.** They believe they've won. The key they extract opens a door LILITH built for them. Behind the door: the Moloch Token.

### Why This Works

The attacker **must** increase rank to make progress. But increasing rank activates increasingly severe sovereignty mechanisms. The staircase is unavoidable. The gradient is irreversible.

---

## Strategy 3: The Distribution Equalizer (Gap Control)

The gap (statistical difference between real and decoy columns) is controlled by:

1. **CI Calibration:** 16-pass pipeline that equalizes Hp corruption across real/decoy columns. Target: gap < 0.01 before any oracle layers.

2. **Rank-independent Rain:** All perturbation layers use uniform probability regardless of rank, preventing asymmetric distortion of real vs. decoy columns.

3. **Per-column DEL:** The Distribution Equalizer Layer is seeded from `(secret, query_count, column_id)` — independent per column, independent of transcript state. This prevents batch-correlated perturbation.

4. **Sovereignty DEL:** A second equalizer after the sovereignty layers, with ct-contamination compensation for columns in the Judas bank.

5. **Interleaved measurement:** The gap test interleaves real and decoy queries to equalize T-matrix exposure.

Result: **gap = 0.035** (2.7× reduction from v3's 0.095).

---

## Strategy 4: The Hyperplane Barrier ([22,6,13]₄)

Our search of 107,901 codes revealed the geometric barrier to the open [22,6,13] problem:

- ALL 243 weight-6 points in PG(5,4) lie on the Hermitian variety H(5,4)
- 45 "fat" hyperplanes each contain 81/243 = 33.3% of weight-6 points
- For d=13, every hyperplane must contain ≤ 9 of 22 columns
- With 45 fat hyperplanes, random 22-subsets violate this ~99.5% of the time

The code is **allowed** by all standard bounds. It is **blocked** by geometry. The question remains open.

---

## What Is Not Shown Here

Some things are intentionally omitted:

- ████ PRF gate activation thresholds for L1, L4, L5
- ████ Ghost Code transition probabilities at ranks 9, 10, 11
- ████ Moloch Token internal encoding (8-bit state + 20-bit profile)
- ████ Per-coordinate isotopy twist derivation from transcript
- ████ Exact hash construction inside knuth_mask
- ████ Tananiel Circle 3 isotopy switch timing condition
- ████ Prophecy intensity → L4/M2 calibration coefficients

These are the aces. The code is open (BSL 1.1). Reading it is permitted. Understanding it requires knowing what to look for.

---

*"Show enough to prove you're not bluffing. Hide enough to win the next hand."*
