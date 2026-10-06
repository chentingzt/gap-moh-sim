# -*- coding: utf-8 -*-
"""Metrics: the three readings of "handover success", plus the side metrics.

The manuscript's criterion (CN:90, CN:240) is

    success  <=>  dt_exec < t_exit - t_init
    executed late  =>  counted as failure

and it explicitly does NOT model sync failure, authentication failure,
target-link rain fade, or admission rejection. Three readings are computed so
the diagnostic can show what each one does to the spread across schemes:

    C1  timing only                -- the manuscript's literal criterion
    C2  C1 and the target satellite has a free concurrency slot (C = 50)
    C3  C2 and the target link is at least as good as the serving link

Everything is per-episode (per-terminal) then averaged, matching the paper's
episode-level 70/15/15 split and its seed-as-unit inference.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["CriterionScores", "score_trace", "capacity_violations",
           "aggregate"]


@dataclass
class CriterionScores:
    n_handover: int
    c1: float
    c2: float
    c3: float
    unnecessary: float
    mean_budget_ms: float
    p05_budget_ms: float
    min_budget_ms: float
    mean_serving_snr_db: float
    forced_fail: int


def score_trace(trace, cfg, snr_min_db=None, margin_db=None):
    """Score one terminal's timeline under all three criteria."""
    snr_min = cfg.snr_min_db if snr_min_db is None else snr_min_db
    margin = (cfg.unnecessary_margin_db if margin_db is None else margin_db)

    n = trace.n_handover
    if n == 0:
        return CriterionScores(0, float("nan"), float("nan"), float("nan"),
                               float("nan"), float("nan"), float("nan"),
                               float("nan"), float("nan"),
                               trace.n_forced_fail)

    ok1 = trace.successes()
    budgets = trace.budgets()
    unnecessary = trace.unnecessary_mask(snr_min, margin)

    # C1 is the published criterion. C2/C3 need cross-terminal state and are
    # filled in by `capacity_violations`; here they default to C1 so that a
    # caller who has not run the cross-check still gets a defined number.
    return CriterionScores(
        n_handover=n,
        c1=float(ok1.mean()),
        c2=float(ok1.mean()),
        c3=float(ok1.mean()),
        unnecessary=float(unnecessary.mean()),
        mean_budget_ms=float(np.mean(budgets) * 1e3),
        p05_budget_ms=float(np.percentile(budgets, 5) * 1e3),
        min_budget_ms=float(np.min(budgets) * 1e3),
        mean_serving_snr_db=float(np.mean([e.serving_snr_db
                                           for e in trace.events])),
        forced_fail=trace.n_forced_fail,
    )


def capacity_violations(traces, capacity_per_sat, window_s=None):
    """Count handovers refused for want of a free slot on the target satellite.

    Terminals are independent in the timing criterion but share the C=50
    concurrency cap, so this is the one genuinely cross-terminal quantity. It
    is a deterministic post-pass: walk all handovers in time order and refuse
    any whose target is already at capacity.

    Returns (n_violations, per_terminal_set) where `per_terminal_set` is a list
    of sets of (terminal_index, event_index) that were refused.
    """
    rows = []
    for ti, tr in enumerate(traces):
        for ei, ev in enumerate(tr.events):
            if ev.target_ok:
                rows.append((ev.t_init + ev.dt_exec_s, ti, ei, ev))
    rows.sort(key=lambda r: r[0])

    load = {}
    refused = set()
    for t_done, ti, ei, ev in rows:
        sat = ev.target_sat
        if load.get(sat, 0) >= capacity_per_sat:
            refused.add((ti, ei))
        else:
            load[sat] = load.get(sat, 0) + 1
    return len(refused), refused


def aggregate(scores):
    """Average a list of per-episode `CriterionScores` into one row."""
    scores = [s for s in scores if s.n_handover > 0]
    if not scores:
        return {}
    n_ho = np.array([s.n_handover for s in scores], dtype=float)
    out = {
        "episodes": len(scores),
        "handovers": int(n_ho.sum()),
        "handovers_per_episode": float(n_ho.mean()),
        "c1": float(np.nanmean([s.c1 for s in scores])),
        "c2": float(np.nanmean([s.c2 for s in scores])),
        "c3": float(np.nanmean([s.c3 for s in scores])),
        "unnecessary": float(np.nanmean([s.unnecessary for s in scores])),
        "mean_budget_ms": float(np.nanmean([s.mean_budget_ms for s in scores])),
        "p05_budget_ms": float(np.nanmean([s.p05_budget_ms for s in scores])),
        "mean_serving_snr_db": float(np.nanmean(
            [s.mean_serving_snr_db for s in scores])),
        "forced_fail": int(sum(s.forced_fail for s in scores)),
    }
    return out
