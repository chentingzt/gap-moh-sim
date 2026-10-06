# -*- coding: utf-8 -*-
"""The six schemes' trigger and target rules.

Transcribed verbatim from the Table 7 note (CN:240), which is the only place
in either manuscript that states them:

    RSS-Threshold    absolute elevation gate 20 deg + 3 dB hysteresis + TTT;
                     target = first visible satellite meeting the gate
    Graph-Dijkstra   remaining visible time >= 90 s;
                     target = longest remaining visible time
    CNN-LSTM-Predict prediction horizon 60 s; target = best predicted quality
    MADRL-Std        gate 20 deg + 1 deg hysteresis; target = policy output
    Grid-Only        GBPT fires at t_exit - dt_ho; target = highest elevation
    GAP-MOH          same as Grid-Only (the DRL refinement sits on top)

The note also says the schemes are deliberately NOT unified ("各方案按其原始
定义设定触发规则与目标星选择，不作统一化"), and that the spread in the
"unnecessary handover" row comes from exactly that.

INTEGRITY: no Table 7 number appears here. In particular `dt_exec_s` has no
default -- the published execution durations are an INPUT to the timing
criterion and are supplied by the caller (`scripts/01_diagnostic.py`), never
baked into this package. This keeps `tests/test_integrity_lint.py` able to
forbid every Table 7 value under `gapmoh/`.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Scheme", "SCHEMES", "trigger_fires", "select_target"]

_HYST_APPLIED = "hysteresis"


@dataclass(frozen=True)
class Scheme:
    key: str
    label: str
    trigger: str          # see `trigger_fires`
    target: str           # see `select_target`
    gate_deg: float = 20.0
    hyst_db: float = 3.0
    ttt_s: float = 0.0
    horizon_s: float = 60.0
    min_remaining_s: float = 90.0
    # Timeless per-scheme execution duration. Supplied by the caller; None
    # means "this scheme's value was not given, sweep it instead".
    dt_exec_s: float = None


SCHEMES = {
    # [CHOICE] The note says RSS-Threshold uses a TTT but never gives its
    # value, which is why the handover-frequency row could not be recomputed.
    # 3 s is the shortest value in the 3GPP TS 38.331 TTT set
    # for a fast-moving terminal, so it is the conservative reading: it
    # suppresses the least ping-pong while keeping the scheme's reactivity.
    # This is an assumption, not a manuscript figure, and it is reported as
    # such wherever the handover rate appears.
    "rss": Scheme("rss", "RSS-Threshold", "elev_gate", "first_above_gate",
                  gate_deg=20.0, hyst_db=3.0, ttt_s=3.0),
    "dijkstra": Scheme("dijkstra", "Graph-Dijkstra", "remaining_below",
                       "longest_remaining", min_remaining_s=90.0),
    "cnnlstm": Scheme("cnnlstm", "CNN-LSTM-Predict", "predict_horizon",
                      "best_quality", horizon_s=60.0),
    "madrl_std": Scheme("madrl_std", "MADRL-Std", "elev_gate", "policy",
                        gate_deg=20.0, hyst_db=1.0),
    "grid_only": Scheme("grid_only", "Grid-Only", "gbpt", "highest_elev"),
    "gapmoh": Scheme("gapmoh", "GAP-MOH", "gbpt", "highest_elev"),
}

# A no-op baseline: never hand over. Not one of the paper's six schemes --
# it exists to measure the floor of the criterion. Under `dt_exec < t_exit -
# t_init` the only way to score zero is to not initiate at all, so this
# establishes what the criterion actually rewards.
SCHEMES["hold"] = Scheme("hold", "Hold-Only", "never", "longest_remaining")

# The paper's Table 7 column order, for reporting only.
REPORT_ORDER = ["rss", "dijkstra", "cnnlstm", "madrl_std", "grid_only",
                "gapmoh"]


def trigger_fires(scheme, t, serving_elev_rad, serving_exit_t,
                  remaining_s, gate_rad, min_elev_rad, ttt_elapsed_s,
                  dt_ho_s=None):
    """Does this scheme initiate a handover at time `t`?

    `serving_elev_rad` is the serving satellite's elevation now; `gate_rad` is
    where hysteresis puts the entry threshold (gate + hysteresis, i.e. the
    link must have degraded INTO the band before the trigger is armed).
    `dt_ho_s` is the GBPT margin, = this scheme's execution duration + 2 ms
    (CN:199); it is required for the GBPT trigger and supplied by the caller.
    """
    if scheme.trigger == "gbpt":
        # Definition 3: fire dt_ho before the serving satellite's grid exit.
        if dt_ho_s is None:
            raise ValueError("GBPT trigger needs dt_ho_s")
        return (serving_exit_t - t) <= dt_ho_s
    if scheme.trigger == "elev_gate":
        # Absolute gate, hysteresis-armed, optionally time-hysteretic (TTT).
        if serving_elev_rad >= gate_rad:
            return False
        return ttt_elapsed_s >= scheme.ttt_s
    if scheme.trigger == "remaining_below":
        return remaining_s < scheme.min_remaining_s
    if scheme.trigger == "predict_horizon":
        # The predictor projects 60 s; hand over when the predicted exit lands
        # inside that horizon.
        return remaining_s < scheme.horizon_s
    raise ValueError(f"unknown trigger {scheme.trigger!r}")


def select_target(scheme, cand_pass, cand_elev_rad, cand_remaining_s,
                  cand_snr_db, serving_elev_rad, serving_snr_db, rng=None):
    """Pick a target pass index, or None.

    `cand_*` are aligned arrays over the currently visible candidate passes,
    excluding the serving pass. Returns an index INTO those arrays.
    """
    if cand_pass.size == 0:
        return None

    if scheme.target == "first_above_gate":
        # CN:240 -- "first visible satellite meeting the gate": the one that
        # entered coverage earliest among those clearing it.
        #
        # [CHOICE] The clearing threshold is gate + hysteresis, not the bare
        # gate. With a bare 20 deg gate a target sitting at 20.5 deg is already
        # inside the armed band, so the trigger condition holds again on the
        # very next 100 ms step and the scheme ping-pongs at 10 Hz -- which
        # the scheme's published handover rate rules out. Reading the
        # hysteresis as applying to the target as well is the only reading
        # consistent with both the scheme's name and that rate.
        gate = np.deg2rad(scheme.gate_deg + scheme.hyst_db)
        ok = cand_elev_rad >= gate
        if not ok.any():
            return None
        return int(np.flatnonzero(ok)[0])

    if scheme.target == "longest_remaining":
        # CN:240 -- "target = longest remaining visible time", subject to the
        # scheme's own >= 90 s admissibility floor. Without the floor the rule
        # re-fires at every 100 ms step whenever no candidate is long-lived,
        # which is a grid artefact rather than scheme behaviour.
        if scheme.min_remaining_s > 0:
            ok = cand_remaining_s >= scheme.min_remaining_s
            if not ok.any():
                return None
            return int(np.argmax(np.where(ok, cand_remaining_s, -np.inf)))
        return int(np.argmax(cand_remaining_s))

    if scheme.target == "highest_elev":
        return int(np.argmax(cand_elev_rad))

    if scheme.target == "best_quality":
        return int(np.argmax(cand_snr_db))

    if scheme.target == "policy":
        # [DEVIATION] The real MADRL-Std target comes from a policy network.
        # In this scripted diagnostic it is stood in by the same rule as
        # longest_remaining so that the trigger rule -- which is what the
        # timing criterion actually responds to -- is what gets measured.
        # Stage 1 replaces it with the trained net.
        return int(np.argmax(cand_remaining_s))

    raise ValueError(f"unknown target rule {scheme.target!r}")
