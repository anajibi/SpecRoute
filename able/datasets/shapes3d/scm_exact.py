"""3DShapes Graph A -- the EXACT structural causal model.

    shape --> scale --> floor_hue <-- object_hue        unobserved: wall_hue, orientation

Deliberately NOT built on ``experiments/hdae/causal/scm.py``. That module fits mechanisms from
data and refuses parent-conditioned categorical nodes, because a fitted categorical child would
need Gumbel-max abduction. Here both children ARE categorical with parents, and nothing has to
be fitted: the mechanisms are mod arithmetic chosen for exact invertibility, so abduction is a
subtraction and a counterfactual is exact rather than approximate.

    C   = m_C(S) (+)_8  U_C ,  m_C = (0, 2, 5, 7)
    H_f = (H_o (+)_10 C) (+)_10 U_F
    U_C = C   (-)_8  m_C(S)
    U_F = H_f (-)_10 (H_o (+)_10 C)

P_K puts 0.25 on 0, 0.125 on +-1 and 0.5/(K-3) elsewhere, so support is full: every grid cell
has non-zero probability, every counterfactual is in support, and its true image is in the grid.
"""
import numpy as np

# h5 factor order and the grid strides it implies (orientation varies fastest)
FACTOR_ORDER = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
SIZES = (10, 10, 10, 8, 4, 15)
STRIDES = (48000, 4800, 480, 60, 15, 1)
N_GRID = 480000

MODELED = ("shape", "object_hue", "scale", "floor_hue")
UNOBSERVED = ("wall_hue", "orientation")
# train_layout.npy column order (protocol section 3.2)
LAYOUT_COLUMNS = ("shape", "object_hue", "scale", "floor_hue", "wall_hue", "orientation")
CARDINALITY = {"shape": 4, "object_hue": 10, "scale": 8, "floor_hue": 10,
               "wall_hue": 10, "orientation": 15}

M_C = np.array([0, 2, 5, 7], dtype=np.int64)


def noise_pmf(K):
    p = np.full(K, 0.5 / (K - 3), dtype=np.float64)
    p[0], p[1], p[K - 1] = 0.25, 0.125, 0.125
    return p


P8, P10 = noise_pmf(8), noise_pmf(10)


def grid_index(Hf, Hw, Ho, C, S, R):
    """idx = H_f*48000 + H_w*4800 + H_o*480 + C*60 + S*15 + R"""
    return (np.asarray(Hf) * 48000 + np.asarray(Hw) * 4800 + np.asarray(Ho) * 480
            + np.asarray(C) * 60 + np.asarray(S) * 15 + np.asarray(R))


def decode_index(idx):
    """Inverse of grid_index. Returns (Hf, Hw, Ho, C, S, R), each int64."""
    idx = np.asarray(idx, dtype=np.int64)
    out = []
    for stride in STRIDES:
        out.append(idx // stride)
        idx = idx % stride
    return tuple(out)


def abduce(S, Ho, C, Hf):
    """Exact noise recovery. Returns (U_C, U_F)."""
    S, Ho, C, Hf = (np.asarray(x, dtype=np.int64) for x in (S, Ho, C, Hf))
    return (C - M_C[S]) % 8, (Hf - Ho - C) % 10


def forward(S, Ho, U_C, U_F):
    """Run the mechanisms forward from roots + noise. Returns (C, Hf)."""
    S, Ho = np.asarray(S, dtype=np.int64), np.asarray(Ho, dtype=np.int64)
    C = (M_C[S] + np.asarray(U_C, dtype=np.int64)) % 8
    Hf = (Ho + C + np.asarray(U_F, dtype=np.int64)) % 10
    return C, Hf


def intervene(S, Ho, C, Hf, var, value):
    """do(var = value): abduct, set the value, recompute descendants, keep everything else.

    Returns (S, Ho, C, Hf). `value` may be a scalar or a per-row array.
    """
    if var not in MODELED:
        raise ValueError(f"{var!r} is not a modeled variable; {MODELED} only "
                         f"(wall_hue and orientation are exogenous and never intervened on)")
    U_C, U_F = abduce(S, Ho, C, Hf)
    S, Ho, C, Hf = (np.array(x, dtype=np.int64, copy=True) for x in (S, Ho, C, Hf))
    v = np.asarray(value, dtype=np.int64)
    if np.any((v < 0) | (v >= CARDINALITY[var])):
        raise ValueError(f"do({var}={value}) outside [0, {CARDINALITY[var]})")
    if var == "shape":
        S[...] = v
        C, Hf = forward(S, Ho, U_C, U_F)
    elif var == "object_hue":
        Ho[...] = v
        Hf = (Ho + C + U_F) % 10
    elif var == "scale":
        C[...] = v                      # mechanism cut: U_C no longer applies
        Hf = (Ho + C + U_F) % 10
    elif var == "floor_hue":
        Hf[...] = v                     # no descendants
    return S, Ho, C, Hf


def counterfactual_index(S, Ho, C, Hf, Hw, R, var, value):
    """Grid index of the counterfactual image. Exogenous Hw/R are preserved by construction."""
    S2, Ho2, C2, Hf2 = intervene(S, Ho, C, Hf, var, value)
    return grid_index(Hf2, Hw, Ho2, C2, S2, R)


def sample(n, rng):
    """n draws from the SCM (unrestricted -- the frozen training layout applies the hold-out).

    Returns an (n, 6) int64 array in LAYOUT_COLUMNS order.
    """
    S = rng.randint(0, 4, n).astype(np.int64)
    Ho = rng.randint(0, 10, n).astype(np.int64)
    Hw = rng.randint(0, 10, n).astype(np.int64)
    R = rng.randint(0, 15, n).astype(np.int64)
    U_C = np.searchsorted(np.cumsum(P8), rng.random_sample(n)).astype(np.int64)
    U_F = np.searchsorted(np.cumsum(P10), rng.random_sample(n)).astype(np.int64)
    C, Hf = forward(S, Ho, U_C, U_F)
    return np.stack([S, Ho, C, Hf, Hw, R], axis=1)
