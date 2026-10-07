# -*- coding: utf-8 -*-
"""The four metrics that actually separate the six schemes.

WHY THIS MODULE EXISTS. The manuscript's headline criterion is timing-only:

    success  <=>  dt_exec < t_exit - t_init

Every scheme's execution duration (3-200 ms) is two to four orders of
magnitude below the time available (seconds to minutes), so under this
criterion all six schemes score 100% and the spread of the published row is
unreproducible from the stated definition. `scripts/01_diagnostic.py`,
`scripts/01b_probe_conjuncts.py` and the trained run in
`scripts/02_train_small.py` each establish that independently.

What the schemes DO differ in is (a) WHICH satellite they hand to and (b) WHEN
they hand over. Those are exactly what the four metrics below measure.

    M1  target quality      -- the link you get after the handover
    M2  service availability -- the link you have, over time, including rain
    M3  unnecessary handovers -- handovers that abandoned a still-good link
    M4  handover frequency   -- how often the serving satellite changes

All four are computed from quantities `sim.py` already produces (the event
list, plus the opt-in per-step serving trace), so none of them requires the
manuscript's numbers to be assumed or fitted.

INTEGRITY: no Table 7 value appears here -- enforced by
`tests/test_integrity_lint.py`.
"""

import numpy as np

from .channel import rain_loss_db, rain_margin_elev_deg

__all__ = ["target_quality", "availability", "unnecessary_v2", "frequency",
           "concurrency", "summarise"]

# Rain rates the availability metric is reported at. The first is the rate the
# manuscript's own 12 dB fade implies at the 15 deg mask (solved for, not
# assumed); the rest bracket it so a reader can substitute their own ITU-R
# P.837 value for the region without re-running anything.
DEFAULT_RAIN_RATES_MM_H = (2.0, 5.0, 10.0, 20.0)


def target_quality(events):
    """M1 -- the quality of the link the handover delivers.

    Reports the target link's SNR and its ELEVATION MARGIN above the mask
    (target elevation - min_elev), which is the geometric headroom that
    determines both the rain loss it will suffer and how long it will last.

    This is the metric the six schemes disagree on most: `longest_remaining`
    (Graph-Dijkstra, and the MADRL stand-in) systematically picks a
    just-risen, low-elevation satellite, while `highest_elev` (Grid-Only,
    GAP-MOH) picks the highest one available.
    """
    snr = np.array([e.target_snr_db for e in events], dtype=float)
    elev = np.array([e.target_elev_deg for e in events], dtype=float)
    # The peak the handed-to pass ever reaches. Not stored by older runs, so it
    # is read defensively and simply omitted from the statistics when absent --
    # `target_peak_elev_mean` then comes back NaN rather than raising, which
    # keeps every cached `new_metrics.json` from the 10-seed run readable.
    peak = np.array([getattr(e, "target_peak_elev_deg", float("nan"))
                     for e in events], dtype=float)
    snr = snr[np.isfinite(snr)]
    elev = elev[np.isfinite(elev)]
    peak = peak[np.isfinite(peak)]
    if snr.size == 0 or elev.size == 0:
        return {"n": 0, "target_snr_mean": float("nan"),
                "target_snr_p05": float("nan"),
                "target_snr_min": float("nan"),
                "target_elev_mean": float("nan"),
                "target_elev_p05": float("nan"),
                "target_elev_min": float("nan"),
                "target_peak_elev_n": 0,
                "target_peak_elev_mean": float("nan"),
                "target_peak_elev_p05": float("nan"),
                "target_peak_elev_min": float("nan")}
    out = {
        "n": int(snr.size),
        "target_snr_mean": float(snr.mean()),
        "target_snr_p05": float(np.percentile(snr, 5)),
        "target_snr_min": float(snr.min()),
        "target_elev_mean": float(elev.mean()),
        "target_elev_p05": float(np.percentile(elev, 5)),
        "target_elev_min": float(elev.min()),
        "target_peak_elev_n": int(peak.size),
        "target_peak_elev_mean": float(peak.mean()) if peak.size
        else float("nan"),
        "target_peak_elev_p05": float(np.percentile(peak, 5)) if peak.size
        else float("nan"),
        "target_peak_elev_min": float(peak.min()) if peak.size
        else float("nan"),
    }
    return out


def availability(serving_trace, cfg, rain_rate_mm_h=None, snr_min_db=None):
    """M2 -- the fraction of time the serving link actually meets the service
    requirement, clear-sky and under rain.

    The clear-sky figure is the degenerate one: with a 15 deg mask the
    clear-sky SNR never falls below `snr_min_db`, so A_clear is 100% for any
    scheme that keeps a link at all. It is reported precisely to make that
    point.

    The rain figure is the discriminating one. Rain loss grows as the slant
    path through the rain layer lengthens, i.e. as elevation falls, so there
    is a SERVICE-FLOOR ELEVATION below which the link cannot hold
    `snr_min_db` under the assumed rain rate. A scheme that rides its serving
    link all the way down to the 15 deg mask spends time below that floor; a
    scheme that hands over early does not. That is a real trade-off against
    M4, and it is the physical explanation for why the manuscript's
    "mean received SNR" row sits entirely below the clear-sky budget.
    """
    tr = np.asarray(serving_trace, dtype=float)
    if tr.size == 0:
        return {"a_clear": float("nan"), "a_rain": float("nan")}
    elev, snr_clear = tr[:, 1], tr[:, 2]
    floor = snr_min_db if snr_min_db is not None else cfg.snr_min_db
    out = {
        "a_clear": float(np.mean(snr_clear >= floor)),
        "a_rain": {},
        "serving_snr_mean": float(snr_clear.mean()),
        "serving_snr_p05": float(np.percentile(snr_clear, 5)),
        "serving_elev_mean_deg": float(np.rad2deg(elev.mean())),
        "min_elev_deg": float(np.rad2deg(elev.min())),
    }
    for rate in (DEFAULT_RAIN_RATES_MM_H if rain_rate_mm_h is None
                 else (rain_rate_mm_h,)):
        snr_eff = snr_clear - rain_loss_db(elev, cfg, rate)
        out["a_rain"][float(rate)] = float(np.mean(snr_eff >= floor))
    return out


def unnecessary_v2(events, service_floor_elev_deg=None,
                   t_need_s=(30.0, 60.0, 120.0)):
    """M3 -- share of handovers that abandoned a still-serviceable link.

    The manuscript defines an unnecessary handover as one initiated while the
    serving link had SNR > SNR_min + 3 dB. Under a clear-sky budget at a
    15 deg mask the serving SNR never drops below 20 dB, so that test is true
    for every handover and the metric cannot discriminate (measured in
    `scripts/01_diagnostic.py`).

    Two replacements are computed, and reported side by side because they
    answer slightly different questions:

      * ELEVATION-BASED (the headline). Unnecessary if, at t_init, the serving
        elevation was still above the SERVICE FLOOR -- the angle below which
        the link cannot hold SNR_min under rain. That is the point at which
        the serving link genuinely stops being able to promise the assumed
        availability, so switching before it is leaving a good link behind.
        This is the observable form of what the manuscript's SNR test meant.

      * TIME-BASED (robustness). Unnecessary if the serving link still had at
        least `t_need` seconds of visibility left. Reported at several
        thresholds because `t_need` is a policy choice, not a measurement.

    Both are computed from quantities `sim.py` already records
    (`serving_elev_deg`, `budget_s`); neither needs new physics.
    """
    if not events:
        return {"n": 0}
    b = np.array([e.budget_s for e in events], dtype=float)
    out = {"n": int(b.size), "budget_mean_s": float(b.mean()),
           "budget_median_s": float(np.median(b)), "share_by_time": {},
           "share_by_elevation": None}
    for t in t_need_s:
        out["share_by_time"][float(t)] = float(np.mean(b >= t))
    if service_floor_elev_deg is not None and np.isfinite(
            service_floor_elev_deg):
        se = np.array([e.serving_elev_deg for e in events], dtype=float)
        se = se[np.isfinite(se)]
        if se.size:
            out["share_by_elevation"] = float(
                np.mean(se > service_floor_elev_deg))
            out["serving_elev_at_init_median_deg"] = float(np.median(se))
    return out


def frequency(n_handover, horizon_s, n_terminals=1):
    """M4 -- serving-satellite changes per terminal-hour, counted directly.

    The manuscript derives this row instead: it takes a geometry-only minimum
    and divides by (1 - unnecessary share). That construction makes the row a
    restatement of the unnecessary row rather than an independent measurement,
    which is why the two cannot be checked against each other. Here it is
    simply counted from the events the simulation produced.
    """
    hours = (horizon_s / 3600.0) * max(n_terminals, 1)
    if hours <= 0:
        return float("nan")
    return float(n_handover / hours)


def concurrency(traces, capacity_per_sat, dt_s=None):
    """True CONCURRENT load per satellite, measured against the C cap.

    WHY NOT `metrics.capacity_violations`. That function walks handovers in time
    order and increments a per-satellite counter that is NEVER DECREMENTED:

        if load.get(sat, 0) >= capacity_per_sat: refused.add(...)
        else: load[sat] = load.get(sat, 0) + 1

    so it counts cumulative ARRIVALS, not occupancy. A satellite that serves one
    terminal after another for an hour reads as 60 arrivals and trips a cap of
    50, though at no instant was it serving more than one terminal. The C = 50
    limit in the manuscript is a CONCURRENCY cap (CN:90, "$C$=50 并发用户/星"),
    so occupancy is the quantity that has to be counted.

    This is the one genuinely cross-terminal metric: every other row in the
    comparison is computed per terminal and averaged.

    Requires `simulate_terminal(..., trace_sat=True)`. Traces without that
    trace are SKIPPED and counted in `n_traces_used`, never silently read as
    zero load -- which would make an unconfigured run look perfectly balanced.

    Returns None-valued fields if no trace carries the satellite trace.
    """
    dt = None
    used = []
    for tr in traces:
        s = getattr(tr, "serving_sat_trace", None)
        if s is None:
            continue
        used.append(np.asarray(s, dtype=np.int64))
        h = float(tr.horizon_s or 0.0)
        if h > 0 and len(s) > 1:
            dt = h / (len(s) - 1)
    if not used:
        return {"n_traces_used": 0, "peak_concurrent": None,
                "peak_over_capacity": None, "overflow_terminal_seconds": None,
                "top1_share": None, "n_active_sat": None,
                "mean_load_active_sat": None, "capacity_per_sat": None}

    n = min(len(u) for u in used)
    S = np.stack([u[:n] for u in used])
    if dt is None or dt <= 0:
        dt = float(dt_s) if dt_s else 1.0

    flat_sat = S.ravel()
    flat_step = np.tile(np.arange(n, dtype=np.int64), S.shape[0])
    ok = flat_sat >= 0
    pairs, counts = np.unique(flat_sat[ok].astype(np.int64) * n + flat_step[ok],
                              return_counts=True)
    sat_of = pairs // n
    excess = np.maximum(counts - int(capacity_per_sat), 0)

    # `bincount` sizes the array to max(sat_id)+1 and zero-fills the rest, so a
    # sparse satellite index set (the real constellation: a few hundred active
    # out of 5040) would report every unused low index as an active satellite
    # carrying zero load. Keep only satellites that actually carried load.
    per_sat = np.bincount(sat_of, weights=counts)
    per_sat = per_sat[per_sat > 0]
    total = float(per_sat.sum())

    return {
        "n_traces_used": len(used),
        "capacity_per_sat": int(capacity_per_sat),
        "peak_concurrent": int(counts.max()),
        "peak_over_capacity": int(max(0, int(counts.max())
                                      - int(capacity_per_sat))),
        "overflow_terminal_seconds": float(excess.sum() * dt),
        "top1_share": float(per_sat.max() / total) if total > 0 else None,
        "n_active_sat": int(per_sat.size),
        "mean_load_active_sat": float(total / per_sat.size / n)
        if per_sat.size else None,
    }


def summarise(traces, cfg, horizon_s, rain_rate_mm_h=None,
              service_floor_elev_deg=None):
    """All four metrics for one scheme's traces."""
    events = [e for tr in traces for e in tr.events]
    per_trace = [availability(tr.serving_trace, cfg, rain_rate_mm_h)
                 for tr in traces if tr.serving_trace is not None]
    avail = {}
    if per_trace:
        for key in ("a_clear", "serving_snr_mean", "serving_snr_p05",
                    "serving_elev_mean_deg", "min_elev_deg"):
            avail[key] = float(np.mean([a[key] for a in per_trace]))
        rates = list(per_trace[0]["a_rain"].keys())
        avail["a_rain"] = {
            r: float(np.mean([a["a_rain"][r] for a in per_trace]))
            for r in rates}
    return {
        "m1_target_quality": target_quality(events),
        "m2_availability": avail,
        "m3_unnecessary": unnecessary_v2(events, service_floor_elev_deg),
        "m4_frequency_per_h": frequency(len(events), horizon_s, len(traces)),
        "n_terminal": len(traces),
        "n_handover": len(events),
    }


def service_floor_report(cfg, rain_rate_mm_h):
    """The service-floor elevation and the manuscript's implied rain rate.

    Used by the driver to state, in the open, the one number the manuscript
    omits: the rain rate its 12 dB fade corresponds to.
    """
    from .channel import rain_rate_for_loss_db

    rate, achieved = rain_rate_for_loss_db(np.deg2rad(cfg.min_elev_deg), cfg,
                                           cfg.rain_margin_db)
    return {
        "implied_rain_rate_mm_h": float(rate),
        "achieved_loss_db": float(achieved),
        "target_loss_db": float(cfg.rain_margin_db),
        "service_floor_elev_deg_implied_rate": rain_margin_elev_deg(
            cfg, rain_rate_mm_h=rate),
        "service_floor_elev_deg_at_cfg_rate": rain_margin_elev_deg(
            cfg, rain_rate_mm_h=rain_rate_mm_h) if rain_rate_mm_h else None,
    }
