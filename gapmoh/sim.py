# -*- coding: utf-8 -*-
"""One terminal's handover timeline under one scheme's rules.

This is the object the timing-feasibility criterion is defined over. For every
handover the scheme initiates it records

    t_init            when the scheme decided to hand over
    t_exit_serving    when the serving satellite leaves the grid cell
    budget_s          t_exit_serving - t_init, i.e. the time available
    dt_exec_s         the scheme's execution-phase duration
    success           dt_exec_s < budget_s      (CN:90, CN:240)

EXACT TRIGGER INSTANTS. The decision interval is 100 ms, but the quantities
being compared to it are 3-200 ms. Snapping a trigger to the 100 ms grid
therefore does not measure the criterion, it measures the grid: a GBPT margin
of 5.2 ms falls entirely between two steps and the trigger is simply missed.
So the trigger instant is computed exactly -- analytically for the time-based
rules (GBPT fires at t_exit - dt_ho by definition) and by bisection for the
elevation-gate rules -- while the 100 ms grid carries the decision proper
(target selection, preparation).

This is the manuscript's own "double check" (CN:31): a 100 ms agent cycle for
the decision, plus a 1 kHz timer for the activation edge.

The budget is a pure geometric quantity, fixed by the trigger rule alone. It
does not depend on the learning, the reward, or the target choice. That fact
is the whole point of `scripts/01_diagnostic.py`.

Per-terminal and streamed: terminals do not interact in the timing criterion
(the only coupling in the paper is the C = 50 concurrency cap, checked
separately in `metrics.capacity_violations`), so an episode is M independent
one-terminal timelines.

No Table 7 number appears here; `dt_exec_s` is a caller-supplied argument.
"""

from dataclasses import dataclass, field

import numpy as np

from .channel import snr_from_elev_db
from .orbit import sat_elev_series, terminal_series
from .policies import select_target

__all__ = ["HandoverEvent", "TerminalTrace", "simulate_terminal", "elev_at"]

_MASK_EPS_S = 1e-9

# Lookahead used to test that a candidate target is still rising. A few
# seconds is enough to separate a rising pass from a setting one at 550 km,
# and short enough that no pass changes direction inside the window.
RISING_LOOK_S = 5.0


@dataclass
class HandoverEvent:
    t_init: float
    t_exit_serving: float
    budget_s: float
    dt_exec_s: float
    success: bool
    serving_elev_deg: float
    serving_snr_db: float
    target_remaining_s: float
    target_ok: bool
    target_sat: int = -1
    target_snr_db: float = float("nan")
    target_elev_deg: float = float("nan")


@dataclass
class TerminalTrace:
    events: list = field(default_factory=list)
    n_forced_fail: int = 0      # link lost with no handover under way
    n_pass_served: int = 0
    serving_time_s: float = 0.0
    horizon_s: float = 0.0
    # (n_step, 3) float32: [t, serving_elev_rad, serving_snr_db_clear] per
    # decision step. Opt-in via `trace_serving=True`; None otherwise, so the
    # default run stays exactly as it was and the existing tests are untouched.
    serving_trace: object = None
    # (n_step,) int64: SATELLITE index (not pass id) the terminal is served by
    # at each decision step, -1 when unserved. Opt-in via `trace_sat=True`.
    # This is what a CONCURRENCY metric needs: the C=50 cap is per satellite,
    # so counting handovers -- which is what `metrics.capacity_violations`
    # does -- measures arrivals, not occupancy.
    serving_sat_trace: object = None

    @property
    def n_handover(self):
        return len(self.events)

    def budgets(self):
        return np.array([e.budget_s for e in self.events], dtype=float)

    def successes(self):
        return np.array([e.success for e in self.events], dtype=bool)

    def unnecessary_mask(self, snr_min_db=10.0, margin_db=3.0):
        """CN:240 -- a handover is unnecessary if, at t_init, the SERVING link
        was still comfortably good: SNR_serving > SNR_min + margin."""
        if not self.events:
            return np.zeros(0, dtype=bool)
        thr = snr_min_db + margin_db
        return np.array([e.serving_snr_db > thr for e in self.events], dtype=bool)


def elev_at(table, terms_of_t, con, sat_id, t):
    """Elevation (rad) of one satellite at one instant."""
    return float(sat_elev_series(np.array([t]), np.array([terms_of_t(t)]),
                                 con, np.array([sat_id]))[0, 0])


def simulate_terminal(table, scheme, cfg, con, lat, lon0, speed_kn,
                      dt_exec_s, dt_ho_s=None, horizon_s=None, rain_fn=None,
                      trace_serving=False, require_rising=True,
                      policy_fn=None, trace_sat=False):
    """Run one terminal's handover timeline.

    `table` is a `PassTable`; `scheme` a `policies.Scheme`. `dt_exec_s` is the
    scheme's execution-phase duration (caller-supplied, never defaulted here).
    `rain_fn(t, elev_rad)` optionally returns extra loss in dB on the SERVING
    link -- it affects the reported SNR and the "unnecessary" classification,
    never the timing verdict.

    `trace_serving=True` additionally records the per-step serving-link
    trajectory into `TerminalTrace.serving_trace`, which is what the
    availability metric in `metrics_v2.py` is computed from. Off by default so
    the timing-criterion runs stay byte-identical to what they were.

    `policy_fn` optionally replaces the scheme's scripted TARGET rule with a
    learned one -- see `rl/policy_target.py`. It is called as
    `policy_fn(cand, ce, crem, e_srv, s_srv, ctx)` and returns an index into
    `cand`. The WHEN is still the scheme's own trigger; only the WHOM moves.
    None (the default) keeps `select_target`, so every existing caller and
    every existing number is unaffected.
    """
    horizon = float(table.horizon_s if horizon_s is None else horizon_s)
    dt = cfg.dt_dec_s
    n_step = int(horizon / dt) + 1
    ts = np.arange(n_step) * dt

    terms_arr = terminal_series(ts, lat, lon0, speed_kn)

    def terms_of(t):
        return terminal_series(np.array([t]), lat, lon0, speed_kn)[0]

    # Context for a learned target selector. Only built when one is supplied;
    # `elev_of_pass` exposes the same geometry the rest of this module uses, so
    # a policy can finite-difference an elevation rate exactly as the training
    # environment does.
    policy_ctx = None
    if policy_fn is not None:
        def elev_of_pass(pass_id, tt):
            return elev_at(table, terms_of, con, int(table.sat[pass_id]), tt)

        policy_ctx = {"lat_rad": float(lat), "lon_rad": float(lon0),
                      "speed_kn": float(speed_kn), "horizon_s": horizon,
                      "elev_of_pass": elev_of_pass}

    if (table._offsets is None or table._offsets.size != n_step + 1
            or getattr(table, "_idx_dt", None) != dt):
        table.build_active_index(n_step, dt)
        table._idx_dt = dt

    min_elev = np.deg2rad(cfg.min_elev_deg)
    gate = np.deg2rad(scheme.gate_deg)
    arm_thr = gate + np.deg2rad(scheme.hyst_db)

    trace = TerminalTrace(horizon_s=horizon)

    def elev_of(pass_ids, k):
        if len(pass_ids) == 0:
            return np.zeros(0)
        sats = table.sat[np.asarray(pass_ids, dtype=np.int64)]
        return sat_elev_series(ts[k:k + 1], terms_arr[k:k + 1], con, sats)[0]

    def acquire(k):
        """Choose a serving pass from what is visible at step k."""
        live = table.active_at(k)
        if live.size == 0:
            return -1
        e = elev_of(live, k)
        ok = e > min_elev
        if not ok.any():
            return -1
        cand, ce = live[ok], e[ok]
        j = select_target(scheme, cand, ce, table.t_exit[cand] - ts[k],
                          snr_from_elev_db(ce, cfg), -np.inf, -np.inf)
        return -1 if j is None else int(cand[j])

    def scheduled_fire(t_exit):
        """Exact instant this scheme's time-based trigger fires, or None."""
        if scheme.trigger == "never":
            return None
        if scheme.trigger == "gbpt":
            return t_exit - float(dt_ho_s)
        if scheme.trigger == "remaining_below":
            return t_exit - scheme.min_remaining_s
        if scheme.trigger == "predict_horizon":
            return t_exit - scheme.horizon_s
        return None   # elev_gate -> crossing bisection below

    serving = acquire(0)
    if serving < 0:
        return trace
    trace.n_pass_served = 1

    k = 0
    prev_e = None
    ttt_fire = None          # pending TTT expiry for gate-triggered schemes
    srv_rec = [] if trace_serving else None
    sat_rec = [] if trace_sat else None
    while k < n_step:
        t = float(ts[k])
        t_exit = table.t_exit[serving]
        e_srv = float(elev_of([serving], k)[0])
        if srv_rec is not None:
            srv_rec.append((t, e_srv, float(snr_from_elev_db(e_srv, cfg))))
        if sat_rec is not None:
            sat_rec.append(int(table.sat[serving]) if serving >= 0 else -1)

        # ---- 1. trigger, evaluated BEFORE the exit check -------------------
        # The GBPT margin (5.2 ms) is far shorter than the 100 ms decision
        # step, so t_fire and t_exit always land in the same step. Checking
        # the exit first would therefore swallow every GBPT handover.
        t_fire = scheduled_fire(t_exit)
        if t_fire is None and scheme.trigger == "elev_gate":
            # Elevation gate: fire at the downward crossing of the armed
            # threshold, after the crossing has PERSISTED for the scheme's
            # time-to-trigger. TTT is what stops a marginal crossing from
            # firing; without it the gate and the hysteresis alone leave the
            # handover rate dependent on the 100 ms decision grid.
            if prev_e is not None and prev_e >= arm_thr > e_srv:
                t_cross = _bisect_elev_crossing(
                    table, con, int(table.sat[serving]), t - dt, t, arm_thr,
                    terms_of)
                if scheme.ttt_s > 0.0:
                    ttt_fire = t_cross + scheme.ttt_s
                else:
                    t_fire = t_cross
            elif prev_e is None and e_srv < arm_thr:
                # already inside the band on acquisition: there is no crossing
                # to time, so the TTT window starts at acquisition
                if scheme.ttt_s > 0.0:
                    ttt_fire = t + scheme.ttt_s
                else:
                    t_fire = t

        # A pending TTT expires only if the link stayed degraded throughout:
        # a recovery above the armed threshold cancels the handover.
        if ttt_fire is not None and t >= ttt_fire - _MASK_EPS_S:
            if e_srv < arm_thr:
                t_fire = ttt_fire
            ttt_fire = None

        if t_fire is not None and t_fire <= t + _MASK_EPS_S:
            t_init = min(max(t_fire, 0.0), t_exit)
            e_at_init = e_srv if abs(t_init - t) < 1e-12 else \
                elev_at(table, terms_of, con, int(table.sat[serving]), t_init)
            nxt = _emit(trace, table, scheme, cfg, con, terms_of, serving,
                        t_init, t_exit, e_at_init, dt_exec_s, k, rain_fn,
                        require_rising, policy_fn, policy_ctx)
            if nxt >= 0:
                serving = nxt
                trace.n_pass_served += 1
            prev_e = None
            ttt_fire = None
            k += 1
            continue

        # ---- 2. no trigger: has the link already dropped? ------------------
        if t >= t_exit - _MASK_EPS_S:
            trace.n_forced_fail += 1
            nxt = acquire(k)
            if nxt < 0:
                k += 1
                prev_e = None
                ttt_fire = None
                continue
            serving = nxt
            trace.n_pass_served += 1
            prev_e = None
            ttt_fire = None
            k += 1
            continue

        prev_e = e_srv
        k += 1

    if srv_rec is not None:
        trace.serving_trace = np.asarray(srv_rec, dtype=np.float32)
    if sat_rec is not None:
        trace.serving_sat_trace = np.asarray(sat_rec, dtype=np.int64)
    trace.serving_time_s = horizon
    return trace


def _bisect_elev_crossing(table, con, sat_id, t_a, t_b, thr, terms_of,
                          iters=24):
    """Bisect the elevation crossing of `thr` inside [t_a, t_b]."""
    fa = elev_at(table, terms_of, con, sat_id, t_a) - thr
    for _ in range(iters):
        m = 0.5 * (t_a + t_b)
        fm = elev_at(table, terms_of, con, sat_id, m) - thr
        if (fm > 0) == (fa > 0):
            t_a = m
            fa = fm
        else:
            t_b = m
    return 0.5 * (t_a + t_b)


def _emit(trace, table, scheme, cfg, con, terms_of, serving, t_init, t_exit,
          e_at_init, dt_exec_s, k, rain_fn, require_rising=True,
          policy_fn=None, policy_ctx=None):
    """Record a handover, pick its target, return the next serving pass."""
    e_srv = float(e_at_init)
    s_srv = float(snr_from_elev_db(e_srv, cfg))
    if rain_fn is not None:
        s_srv -= float(rain_fn(t_init, e_srv))

    live = table.active_at(k)
    cand = live[live != serving]
    if cand.size:
        ce = np.array([elev_at(table, terms_of, con, int(table.sat[p]), t_init)
                       for p in cand])
        # admissibility floor: above the mask, and -- for gate-triggered
        # schemes -- above the ARMED threshold. Handing to a target already
        # inside the armed band would re-arm the trigger on the very next
        # step, which is a 10 Hz artefact, not scheme behaviour.
        floor = np.deg2rad(cfg.min_elev_deg)
        if scheme.trigger == "elev_gate":
            floor = max(floor, np.deg2rad(scheme.gate_deg + scheme.hyst_db))
        ok = ce >= floor
        cand, ce = cand[ok], ce[ok]
    if cand.size:
        crem = table.t_exit[cand] - t_init
        csnr = snr_from_elev_db(ce, cfg)
        if rain_fn is not None:
            csnr = csnr - np.array([rain_fn(t_init, float(x)) for x in ce])
        # [CHOICE] Admissibility: a target must still be RISING. No scheme in
        # the manuscript hands a terminal to a satellite that is already on its
        # way out, yet the literal target rules do not say so: "first visible
        # satellite meeting the gate" picks by list order, and `highest_elev`
        # picks whatever is near its peak. Without this filter RSS-Threshold
        # rides a train of departing satellites and switches every ~13 s
        # instead of every few minutes, which is a property of the rule as
        # written, not of the scheme. Applied uniformly to every scheme, and
        # only when at least one rising candidate exists, so it can never
        # strand a handover.
        if require_rising:
            ce_later = np.array([
                elev_at(table, terms_of, con, int(table.sat[p]),
                        t_init + RISING_LOOK_S) for p in cand])
            rising = ce_later > ce
            if rising.any():
                cand, ce = cand[rising], ce[rising]
                crem, csnr = crem[rising], csnr[rising]
        if policy_fn is not None:
            ctx = dict(policy_ctx or {})
            ctx.update({"t": t_init, "n_ho": len(trace.events),
                        "serving_pass": int(serving),
                        "serving_remaining_s": t_exit - t_init})
            j = policy_fn(cand, ce, crem, e_srv, s_srv, ctx)
        else:
            j = select_target(scheme, cand, ce, crem, csnr, e_srv, s_srv)
    else:
        j = None

    if j is None:
        # nothing to hand to: hold, and let the link drop (a forced failure)
        trace.events.append(HandoverEvent(
            t_init=t_init, t_exit_serving=t_exit, budget_s=t_exit - t_init,
            dt_exec_s=dt_exec_s, success=False,
            serving_elev_deg=float(np.rad2deg(e_srv)), serving_snr_db=s_srv,
            target_remaining_s=0.0, target_ok=False, target_sat=-1))
        return serving

    tgt = int(cand[j])
    remaining = t_exit - t_init
    target_ok = (t_init + dt_exec_s) < table.t_exit[tgt]
    success = (dt_exec_s < remaining) and target_ok
    trace.events.append(HandoverEvent(
        t_init=t_init, t_exit_serving=t_exit, budget_s=remaining,
        dt_exec_s=dt_exec_s, success=bool(success),
        serving_elev_deg=float(np.rad2deg(e_srv)), serving_snr_db=s_srv,
        target_remaining_s=float(table.t_exit[tgt] - t_init),
        target_ok=bool(target_ok), target_sat=int(table.sat[tgt]),
        target_snr_db=float(csnr[j]),
        target_elev_deg=float(np.rad2deg(ce[j]))))
    return tgt
