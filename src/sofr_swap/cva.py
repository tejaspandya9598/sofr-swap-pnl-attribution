"""
Counterparty exposure and CVA on the swap book.

A swap's mark-to-market is symmetric, its credit risk is not. If the counterparty
defaults while the book is worth +$4mm to us we recover a fraction of that; if it is
worth -$4mm we still owe it in full. CVA prices that asymmetry:

    CVA = (1 - R) * sum_k  DF(t_k) * EPE(t_k) * [S(t_{k-1}) - S(t_k)]

where EPE is the expected *positive* exposure, S is the counterparty's survival
probability, and R is recovery. DVA is the mirror image on our own credit and our own
negative exposure.

Exposure is simulated by resampling real daily curve moves - every path is a sequence
of days that actually happened, in blocks, so a fortnight of 2008 arrives intact
rather than as twenty independent draws from a fitted normal. Two consequences worth
stating up front:

- This is a *historical* CVA, the measure a risk department uses for EPE and PFE
  limits. A CVA desk marking a reserve wants risk-neutral exposure calibrated to
  swaption vols, which is a different number and needs data that is not free.
- The sampled changes are demeaned. Forty years of Treasury history has a large
  downward drift in it, and a bootstrap that inherits it walks the ten-year rate to
  somewhere absurd over a thirty-year horizon. Removing the drift is the minimum
  needed for the simulation to be about volatility rather than about 1986.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sofr_swap.conventions import TAU
from sofr_swap.curve import SofrCurve
from sofr_swap.instruments import Swap

RECOVERY = 0.40          # ISDA's standard assumption for senior unsecured


def hazard_from_spread(spread, recovery: float = RECOVERY):
    """The credit triangle: lambda = s / (1 - R).

    A flat hazard rate lambda gives survival exp(-lambda*t) and a par CDS spread of
    lambda*(1-R). Inverting it is the standard way to read a hazard rate off a single
    quoted spread, and it is exact for a flat term structure of credit.
    """
    return np.asarray(spread, dtype=float) / (1.0 - recovery)


def survival(hazard, times):
    """S(t) = exp(-lambda t), broadcasting hazard over paths and times over the grid."""
    return np.exp(-np.asarray(hazard) * np.asarray(times))


def bootstrap_dfs(par_rates: np.ndarray, n_years: int) -> np.ndarray:
    """Vectorised version of `SofrCurve.bootstrap` for many curves at once.

    `par_rates` is (n_paths, n_years) already interpolated onto the annual grid.
    Returns discount factors of the same shape. The recursion is the same forward
    substitution the single-curve bootstrap does; it just runs on whole columns.
    """
    n_paths = par_rates.shape[0]
    dfs = np.empty_like(par_rates)
    annuity = np.zeros(n_paths)
    for k in range(n_years):
        s = par_rates[:, k]
        dfs[:, k] = (1.0 - s * annuity) / (1.0 + s * TAU)
        annuity = annuity + TAU * dfs[:, k]
    return dfs


@dataclass
class ExposureSimulation:
    """Simulated curves and counterparty spreads on a common time grid."""
    times: np.ndarray            # valuation times in years from today
    par_rates: np.ndarray        # (n_paths, n_times, n_years) par curve at each node
    spreads: np.ndarray          # (n_paths, n_times) counterparty spread at each node
    tenor_grid: np.ndarray       # the annual maturities the par curves are quoted on
    variance_explained: np.ndarray = None   # of the retained curve factors
    clipped: float = 0.0         # share of pillar draws pinned to the historical envelope
    repaired: float = 0.0        # share of curves pulled back to a bootstrappable shape


def curve_factors(changes: np.ndarray, n_factors: int = 3):
    """PCA of daily curve moves: loadings, scores, and variance explained.

    Simulating each pillar as its own random walk is what breaks a bootstrap. Nothing
    stops the one-year from wandering to its floor while the thirty-year sits at nine
    percent, and that curve has no positive discount factor at the long end - the
    annuity multiplied by the par rate exceeds one and DF_k comes out negative. It is
    not a numerical accident, it is a curve that cannot exist.

    Three principal components of the real daily changes fix it, because every
    simulated move is then a combination of shapes the Treasury curve actually makes.
    On this sample they are the textbook three, and they carry 98% of the variance.
    """
    u, sv, vt = np.linalg.svd(changes - changes.mean(axis=0), full_matrices=False)
    explained = sv ** 2 / (sv ** 2).sum()
    loadings = vt[:n_factors]                       # (n_factors, n_pillars)
    scores = (changes - changes.mean(axis=0)) @ loadings.T
    return loadings, scores, explained[:n_factors]


def simulate(panel: pd.DataFrame, horizon: float, steps: int, n_paths: int,
             block: int = 10, seed: int = 0, decouple: bool = False,
             n_factors: int = 3) -> ExposureSimulation:
    """Block-bootstrap real daily moves of the curve and the spread forward.

    `block` is the length in days of each resampled run. Blocks preserve the
    autocorrelation that makes a crisis a crisis: spreads do not gap out for one day
    and mean-revert the next, and sampling days independently produces exposure paths
    that are far too tame in the tail.

    `decouple` resamples the spread from different days than the curve. Each marginal
    distribution is untouched, and their joint behaviour is destroyed, which is the
    counterfactual that isolates wrong-way risk.

    Two constraints keep the simulated curves inside shapes the market has printed,
    and both bounds are read off the sample rather than chosen. Pillars are clipped to
    the highest and lowest each tenor has reached since 1986, and the 2s30s slope to
    its own historical range. The second one is not cosmetic: every one of the 8,478
    real curves in the sample bootstraps to strictly positive discount factors, while
    an unconstrained thirty-year random walk produces 2s30s of seven hundred basis
    points against a historical maximum of four hundred, and a par swap paying 9.4% for
    thirty years against a 2% front end has no positive discount factor at the far end.
    The failure is not numerical - the curve does not exist.
    """
    rng = np.random.default_rng(seed)
    tenor_cols = [c for c in panel.columns if c != "spread"]
    tenors = np.array(tenor_cols, dtype=float)
    changes = panel.diff().dropna()
    changes = changes - changes.mean()                    # no drift; see the module docstring
    curve_moves = changes[tenor_cols].to_numpy()
    spread_moves = changes["spread"].to_numpy()
    loadings, scores, explained = curve_factors(curve_moves, n_factors)
    n_obs = len(changes)

    days_per_step = max(int(round(252 * horizon / steps)), 1)
    grid = np.arange(1, int(tenors.max()) + 1)
    start = panel.iloc[-1]
    par0 = np.interp(grid, tenors, start[tenor_cols].to_numpy())
    spread0 = float(start["spread"])
    lo = np.interp(grid, tenors, panel[tenor_cols].min().to_numpy())
    hi = np.interp(grid, tenors, panel[tenor_cols].max().to_numpy())
    # Two shape constraints, each read off the sample: the 2s30s slope, which PC2 is,
    # and the 1s5s front end, which PC3 is. Both are enforced by sliding the curve along
    # the corresponding factor, so a constrained curve is still a real curve shape.
    shape_rules = []
    for (short, long_), factor in (((2, 30), 1), ((1, 5), 2)):
        observed = panel[long_] - panel[short]
        direction = np.interp(grid, tenors, loadings[factor])
        i_s, i_l = int(short) - 1, int(long_) - 1
        gain = direction[i_l] - direction[i_s]
        shape_rules.append((i_s, i_l, float(observed.min()), float(observed.max()),
                            direction, gain))

    def block_index(generator, n_days):
        n_blocks = int(np.ceil(n_days / block))
        heads = generator.integers(0, n_obs - block, size=(n_paths, n_blocks))
        idx = (heads[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)
        return idx[:, :n_days]

    times = np.linspace(horizon / steps, horizon, steps)
    par_paths = np.empty((n_paths, steps, grid.size))
    spread_paths = np.empty((n_paths, steps))
    par_state = np.tile(par0, (n_paths, 1))
    spread_state = np.full(n_paths, spread0)
    alt = np.random.default_rng(seed + 10_000)
    clipped = repaired = 0

    for k in range(steps):
        idx = block_index(rng, days_per_step)
        factor_move = scores[idx].sum(axis=1)                    # (n_paths, n_factors)
        pillar_move = factor_move @ loadings                     # back to the 8 tenors
        par_state = par_state + np.array([np.interp(grid, tenors, m) for m in pillar_move])
        clipped += int(((par_state < lo) | (par_state > hi)).sum())
        par_state = np.clip(par_state, lo, hi)
        # Alternate the two shape projections; each one can nudge the other out of range,
        # and two passes is enough to leave both satisfied on this sample.
        for _pass in range(2):
            for i_s, i_l, s_lo, s_hi, direction, gain in shape_rules:
                spread_ = par_state[:, i_l] - par_state[:, i_s]
                adjust = (np.clip(spread_, s_lo, s_hi) - spread_) / gain
                par_state = np.clip(par_state + adjust[:, None] * direction[None, :], lo, hi)
        par_state, n_repaired = _make_bootstrappable(par_state, par0)
        repaired += n_repaired
        # The spread walks with the curve unless we are deliberately breaking the link:
        # same rows of history means the same days, which is what carries the real
        # correlation between rates and credit into the simulation.
        sp_idx = block_index(alt, days_per_step) if decouple else idx
        spread_state = np.maximum(spread_state + spread_moves[sp_idx].sum(axis=1), 1e-4)
        par_paths[:, k, :] = par_state
        spread_paths[:, k] = spread_state

    return ExposureSimulation(times, par_paths, spread_paths, grid, explained,
                              clipped / (n_paths * steps * grid.size),
                              repaired / (n_paths * steps))


def _make_bootstrappable(par_state: np.ndarray, anchor: np.ndarray,
                         max_passes: int = 8) -> tuple[np.ndarray, int]:
    """Pull any curve that has no positive discount factor back toward today's curve.

    The two shape rules above catch almost everything, but they are two constraints on
    a thirty-point object and a handful of draws still land on a curve the annual-grid
    bootstrap cannot price - a par swap whose fixed leg is worth more than the whole
    notional, which is to say a curve that does not exist. Rather than add a third
    hand-picked rule, those curves are shrunk toward the anchor until they do exist.
    The alternative is to drop the paths, which quietly deletes the most extreme
    scenarios from an exposure distribution whose whole purpose is the tail.

    On the samples here this touches well under a tenth of a percent of curves; the
    simulation reports the share so it cannot hide.
    """
    bad = (bootstrap_dfs(par_state, par_state.shape[1]) <= 0).any(axis=1)
    n_bad = int(bad.sum())
    if not n_bad:
        return par_state, 0
    weight = 1.0
    for _ in range(max_passes):
        weight *= 0.5
        blended = weight * par_state[bad] + (1.0 - weight) * anchor[None, :]
        par_state[bad] = blended
        still = (bootstrap_dfs(par_state[bad], par_state.shape[1]) <= 0).any(axis=1)
        if not still.any():
            break
        idx = np.where(bad)[0][still]
        bad = np.zeros_like(bad)
        bad[idx] = True
    return par_state, n_bad


def _interp_matrix(target_times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Weights W with (W @ log_df)[i] = log-linear interpolation of log df at target[i].

    Precomputing the weights turns per-path interpolation into one matrix product, which
    is what makes a 30-year, 5,000-path exposure run take seconds in numpy.
    """
    W = np.zeros((target_times.size, grid.size))
    for i, t in enumerate(target_times):
        if t <= grid[0]:                                  # log-linear from df(0) = 1
            W[i, 0] = t / grid[0]
            continue
        j = int(np.searchsorted(grid, t, side="right") - 1)
        if j >= grid.size - 1:                            # flat forward past the last pillar
            slope = 1.0 / (grid[-1] - grid[-2])
            W[i, -1] = 1.0 + (t - grid[-1]) * slope
            W[i, -2] = -(t - grid[-1]) * slope
            continue
        w = (t - grid[j]) / (grid[j + 1] - grid[j])
        W[i, j], W[i, j + 1] = 1.0 - w, w
    return W


def exposure(sim: ExposureSimulation, swaps: list[Swap]) -> np.ndarray:
    """Portfolio mark-to-market on every path at every grid time. Shape (paths, times)."""
    n_paths, steps, _ = sim.par_rates.shape
    values = np.zeros((n_paths, steps))
    for k, t_val in enumerate(sim.times):
        log_df = np.log(bootstrap_dfs(sim.par_rates[:, k, :], sim.tenor_grid.size))
        for swap in swaps:
            if swap.maturity <= t_val + 1e-9:
                continue                                   # matured; it is not exposure
            coupons = np.array([end - t_val for _s, end in swap.schedule() if end > t_val + 1e-9])
            start = max(swap.effective - t_val, 0.0)
            targets = np.concatenate([[start, swap.maturity - t_val], coupons])
            dfs = np.exp(log_df @ _interp_matrix(targets, sim.tenor_grid).T)
            float_pv = dfs[:, 0] - dfs[:, 1]
            fixed_pv = swap.fixed_rate * TAU * dfs[:, 2:].sum(axis=1)
            values[:, k] += swap.sign * swap.notional * (float_pv - fixed_pv)
    return values


def profile(values: np.ndarray, quantile: float = 0.975) -> pd.DataFrame:
    """The four numbers a credit-risk report is built from, at each grid time."""
    positive, negative = np.maximum(values, 0.0), np.minimum(values, 0.0)
    return pd.DataFrame({
        "ee": values.mean(axis=0),
        "epe": positive.mean(axis=0),
        "ene": negative.mean(axis=0),
        "pfe": np.quantile(positive, quantile, axis=0),
    })


def cva(sim: ExposureSimulation, values: np.ndarray, curve: SofrCurve,
        recovery: float = RECOVERY, own_spread: float | None = None) -> dict:
    """Path-wise CVA (and DVA, if we are given our own spread).

    Path-wise matters. Averaging exposure first and applying an average default
    probability afterwards throws away the covariance between the two, which is the
    entire content of wrong-way risk. Here each path carries its own simulated spread,
    so a path where the counterparty's credit blew out is weighted by the default
    probability that path implies.
    """
    times = sim.times
    edges = np.concatenate([[0.0], times])
    df = curve.df(times)

    hazard = hazard_from_spread(sim.spreads, recovery)          # (paths, times)
    surv = np.exp(-hazard * times[None, :])
    surv_prev = np.concatenate([np.ones((surv.shape[0], 1)), surv[:, :-1]], axis=1)
    pd_step = np.maximum(surv_prev - surv, 0.0)

    positive = np.maximum(values, 0.0)
    contributions = (1.0 - recovery) * df[None, :] * positive * pd_step
    cva_paths = contributions.sum(axis=1)
    out = {"cva": float(cva_paths.mean()), "cva_paths": cva_paths,
           "by_time": contributions.mean(axis=0), "grid_edges": edges}

    if own_spread is not None:
        own_hazard = hazard_from_spread(own_spread, recovery)
        own_surv = np.exp(-own_hazard * times)
        own_pd = np.diff(np.concatenate([[1.0], own_surv]))
        negative = np.minimum(values, 0.0)
        dva = (1.0 - recovery) * (df[None, :] * -negative * -own_pd[None, :]).sum(axis=1)
        out["dva"] = float(dva.mean())
        out["bcva"] = out["cva"] - out["dva"]
    return out
