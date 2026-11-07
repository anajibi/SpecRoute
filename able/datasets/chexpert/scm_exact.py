"""CheXpert protocol v0, section 2: the SCM, with exact abduction.

Four variables, one mechanism. Every variable -- the 3-way age bin and the
three binaries alike -- is a Gumbel-max:

    value = argmax_k ( logit_k(parents) + G_k ),   G_k iid Gumbel(0,1)

For K=2 with logits (0, t) this is the protocol's "logistic with Gumbel
noise": argmax is 1 exactly when G_1 - G_0 > -t, and G_1 - G_0 is standard
logistic, so P(1) = sigmoid(t). So one mechanism and one abduction routine
serve all four variables.

Abduction is EXACT, not a point estimate and not an approximation. Given the
observed category k and the logits, the posterior of the Gumbel vector is
available in closed form (Maddison et al. 2014's top-down construction):

  * the achieved maximum m = max_j (logit_j + G_j) is marginally
    Gumbel(logsumexp(logits)), regardless of which j achieved it;
  * conditioned on m, the winner's noise is forced: G_k = m - logit_k;
  * every loser's noise is an independent Gumbel truncated to lie below
    m - logit_j, sampled by inverse CDF.

Drawing from this posterior and re-running the mechanism returns the factual
value by construction -- which `check_consistency` asserts rather than trusts.

Parameters are fitted saturated (one logit per parent configuration), so the
fitted marginals and conditionals reproduce the training split exactly.
"""
import json
import numpy as np

ORDER = ["sex", "age_bin", "effusion", "cardiomegaly"]
PARENTS = {"sex": [], "age_bin": [], "effusion": ["age_bin"], "cardiomegaly": ["age_bin"]}
CARD = {"sex": 2, "age_bin": 3, "effusion": 2, "cardiomegaly": 2}


# --------------------------------------------------------------------------- fit

def fit(values, min_count=1):
    """Saturated log-probability logits from a dict of integer arrays.

    Counts are floored at `min_count` so that an unobserved cell becomes very
    improbable rather than impossible: a -inf logit would make that category
    unreachable under every intervention and would put a -inf in the abduction
    arithmetic.
    """
    p = {"cardinality": CARD, "parents": PARENTS, "order": ORDER, "logits": {}}
    n = len(values["sex"])
    for var in ORDER:
        K = CARD[var]
        pars = PARENTS[var]
        if not pars:
            c = np.bincount(values[var], minlength=K).astype(np.float64)
            c = np.maximum(c, min_count)
            p["logits"][var] = np.log(c / c.sum()).reshape(1, K).tolist()
        else:
            (par,) = pars  # the graph has no variable with two parents
            P = CARD[par]
            rows = np.zeros((P, K))
            for a in range(P):
                m = values[par] == a
                c = np.bincount(values[var][m], minlength=K).astype(np.float64)
                c = np.maximum(c, min_count)
                rows[a] = np.log(c / c.sum())
            p["logits"][var] = rows.tolist()
    p["n_fit"] = int(n)
    return p


def logits_for(var, params, values, n=None):
    """(N, K) logits for `var` given the parent values in `values`.

    A root variable has no parent to take its length from, and during a forward
    pass `values` is still being filled, so `n` has to be supplied there.
    """
    tab = np.asarray(params["logits"][var], dtype=np.float64)
    pars = params["parents"][var]
    if not pars:
        if n is None:
            n = len(next(iter(values.values())))
        return np.repeat(tab, n, axis=0)
    (par,) = pars
    return tab[np.asarray(values[par], dtype=np.int64)]


# ---------------------------------------------------------------------- abduction

def _truncated_gumbels(logits, k, rng):
    """Exact posterior draw of the Gumbel vector given argmax == k."""
    logits = np.asarray(logits, dtype=np.float64)
    n, K = logits.shape
    k = np.asarray(k, dtype=np.int64)
    idx = np.arange(n)

    # the achieved maximum: Gumbel located at logsumexp(logits)
    mx = logits.max(axis=1, keepdims=True)
    lse = (mx + np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))).ravel()
    m = lse - np.log(-np.log(rng.random(n)))

    # losers: Gumbel(0,1) truncated to g < t_j, by inverse CDF.
    # F(g) = exp(-exp(-g)); F(g)/F(t) = u  =>  g = -log(exp(-t) - log u).
    t = m[:, None] - logits
    u = rng.random((n, K))
    g = -np.log(np.exp(-t) - np.log(u))

    # winner: its noise is pinned by the maximum it achieved
    g[idx, k] = m - logits[idx, k]
    return g


def abduct(values, params, seed):
    """Per-unit exogenous noise consistent with the observed attributes."""
    rng = np.random.default_rng(seed)
    g = {}
    for var in params["order"]:
        g[var] = _truncated_gumbels(logits_for(var, params, values), values[var], rng)
    return g


# ------------------------------------------------------------------- forward pass

def forward(values, gumbels, params, fixed=None):
    """Re-run the SCM holding `fixed` and reusing `gumbels`.

    With fixed=None this must return `values` unchanged -- that is counterfactual
    consistency, and it is what makes the abduction above worth having. A
    variable moves only if one of its parents moved.
    """
    fixed = fixed or {}
    out = {}
    for var in params["order"]:
        if var in fixed:
            f = fixed[var]
            out[var] = (np.full(len(values[var]), int(f), dtype=np.int64)
                        if np.isscalar(f) else np.asarray(f, dtype=np.int64))
            continue
        lg = logits_for(var, params, out, n=len(gumbels[var]))
        out[var] = (lg + gumbels[var]).argmax(axis=1)
    return out


def counterfactual(values, params, var, new_value, seed):
    """do(var = new_value) on every unit. Returns (cf_values, gumbels)."""
    g = abduct(values, params, seed)
    base = forward(values, g, params)
    for v in params["order"]:
        if not np.array_equal(base[v], np.asarray(values[v], dtype=np.int64)):
            raise AssertionError(
                f"abduction is not consistent on {v}: re-running the mechanism with the "
                f"abducted noise and the factual parents did not reproduce the factual value")
    return forward(values, g, params, fixed={var: new_value}), g


def check_consistency(values, params, seed, reps=8):
    """Consistency must hold for EVERY noise draw, not one lucky one."""
    for r in range(reps):
        g = abduct(values, params, seed + r)
        out = forward(values, g, params)
        for v in params["order"]:
            if not np.array_equal(out[v], np.asarray(values[v], dtype=np.int64)):
                bad = int((out[v] != np.asarray(values[v])).sum())
                raise AssertionError(f"draw {r}: {v} differs on {bad} units")
    return True


def save(params, path):
    with open(path, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)


def load(path):
    with open(path) as f:
        return json.load(f)
