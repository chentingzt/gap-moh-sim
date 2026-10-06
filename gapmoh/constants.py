# -*- coding: utf-8 -*-
"""GAP-MOH simulation constants.

Every value carries an inline provenance tag:

    [SPEC  <file>:<line>]  taken verbatim from the manuscript specification
    [CHOICE]               not stated in the manuscript; we pick a value and say why
    [DEVIATION]            differs from the manuscript; the reason is recorded

The specification source is the ENGLISH manuscript
`paper/GAP-MOH_REVISED_v5_Clean.md` (referred to below as EN), per the decision
recorded in the authors' provenance record. Where the Chinese submission (`GBPT_中文版.md`, CN) differs
and the difference matters, it is noted.

HARD RULE (enforced by tests/test_integrity_lint.py): no published Table 7
value may appear anywhere under `gapmoh/`. Those numbers are OUTPUTS to be
compared against, never inputs.
"""

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Physical constants  [SPEC EN:308-317 region; standard values]
# --------------------------------------------------------------------------
R_E = 6371.0            # km, spherical Earth radius (same as the released geometry code)
MU = 398600.4418        # km^3/s^2, Earth gravitational parameter
OMEGA_E = 7.2921159e-5  # rad/s, Earth rotation rate
KNOT = 0.514444 / 1000.0  # km/s per knot
SPEED_OF_LIGHT = 299792.458  # km/s


@dataclass(frozen=True)
class Scenario:
    """Constellation, terminal population and link-budget parameters."""

    # ---- constellation -------------------------------------------------
    n_planes: int = 72              # [SPEC EN:1010]
    sats_per_plane: int = 70        # [SPEC EN:1010]  72*70 = 5040
    altitude_km: float = 550.0      # [SPEC EN:1010]
    inclination_deg: float = 53.0   # [SPEC EN:1010]
    phase_factor: int = 1           # [SPEC EN:1010]  Walker phasing F=1

    # ---- GEO control-plane relays --------------------------------------
    geo_lon_deg: tuple = (80.0, 110.0, 140.0)   # [SPEC EN:1011 / CN:187]

    # ---- terminal population -------------------------------------------
    n_terminals: int = 100          # [SPEC EN:1012]  M = 100 vessels
    lon_min_deg: float = 110.0      # [SPEC CN:190]   service theatre
    lon_max_deg: float = 150.0      # [SPEC CN:190]
    lat_center_deg: float = 20.0    # [SPEC CN:190]
    lat_spread_deg: float = 15.0    # [SPEC CN:190]   20 +/- 15 deg N
    speed_mean_kn: float = 15.0     # [SPEC CN:190]   N(15, 5) kn
    speed_sd_kn: float = 5.0        # [SPEC CN:190]

    # ---- capacity ------------------------------------------------------
    capacity_per_sat: int = 50      # [SPEC CN:189]   C = 20 Gbps / 400 Mbps

    # ---- link budget ---------------------------------------------------
    # CN Table 1 is the self-consistent link budget and is used here; EN's
    # parameters (P_T 40 dBm, G_T 38 dBi, G_R 35 dBi, T_s 300 K) give the same
    # G/T to within 0.2 dB, so the two manuscripts agree on the link.
    freq_hz: float = 30.0e9         # [SPEC CN:69 / EN:1015]  Ka band, lambda = 1 cm
    bandwidth_hz: float = 100.0e6   # [SPEC CN:77 / EN:1015]  B = 100 MHz
    eirp_dbm: float = 78.0          # [SPEC CN:74 / EN:1016]  downlink peak
    g_rx_dbi: float = 36.0          # [SPEC CN:75]  G_R ~ 36 dBi
    l_atm_db: float = 0.8           # [SPEC CN:73]  clear-sky gaseous loss
    noise_temp_k: float = 360.0     # [SPEC CN:76]  T_s = 360 K (EN says 300 K)
    g_over_t_db: float = 10.4       # [SPEC CN:75]  G/T = 10.4 dB/K (EN implies 10.2)
    # Reference SNRs the link budget yields; asserted in tests/test_channel.py.
    snr_ref_15deg_db: float = 20.6  # [SPEC CN:79]
    snr_ref_zenith_db: float = 30.0  # [SPEC CN:79]
    # Rain is a design margin, not a modelled outage: CN:80 budgets 12 dB at
    # 99.5% availability, EN:1387 says 15 dB, and CN:82 states it is absorbed
    # by adaptive coding and uplink power control. The §3.4 success criterion
    # explicitly does not model target-link rain fade.
    rain_margin_db: float = 12.0    # [SPEC CN:80]  (EN says 15 dB)
    rain_availability_pct: float = 99.5   # [SPEC CN:80]

    # ---- geometry / visibility -----------------------------------------
    min_elev_deg: float = 15.0      # [SPEC EN:1014]
    n_candidates: int = 5           # [SPEC EN:1025]  K = 5

    # ---- timing ---------------------------------------------------------
    dt_dec_s: float = 0.1           # [SPEC EN:1029]  decision interval 100 ms
    dt_ho_s: float = 10.0e-3        # [SPEC EN:685-692] GAP-MOH
                                    #   (search 1 + sync 5 + auth 2 + margin 2 ms)
                                    #   NOTE: CN:140 uses 5.2 ms; we follow EN.
    dt_margin_s: float = 2.0e-3     # [SPEC EN:685-692]

    # ---- reward thresholds ----------------------------------------------
    snr_min_db: float = 10.0        # [SPEC CN:90 / EN:381-389]
    snr_target_db: float = 20.0     # [SPEC EN:566]   r_quality = tanh(SNR/20)
    unnecessary_margin_db: float = 3.0   # [SPEC EN:381-389] SNR_min + 3 dB
    load_penalty_knee: float = 0.7  # [SPEC EN:575]   L/L_max - 0.7

    @property
    def total_sats(self) -> int:
        return self.n_planes * self.sats_per_plane


@dataclass(frozen=True)
class Reward:
    """Reward weights. [SPEC EN:590-595] (alpha1..alpha4) = (1.0, 0.5, 2.0, 1.0)."""
    alpha_quality: float = 1.0
    alpha_stability: float = 0.5
    alpha_interruption: float = 2.0
    alpha_load: float = 1.0
    lam_stability: float = 0.01     # [SPEC EN:567]  EN only; CN omits it
    terminal_penalty: float = -10.0  # [SPEC EN:571]  EN only; CN omits it


@dataclass(frozen=True)
class RLConfig:
    """Discrete-MADDPG hyperparameters, EN specification.

    The Chinese submission's Table 6 contradicts its own section 5.1 and EN on
    learning rate, network size and episode count; EN is used throughout
    (see the provenance record for the decision).
    """
    actor_hidden: tuple = (256, 256)        # [SPEC EN:625]
    critic_hidden: tuple = (512, 512, 256)  # [SPEC EN:626]
    actor_lr: float = 1.0e-4                # [SPEC EN:628]
    critic_lr: float = 3.0e-4               # [SPEC EN:629]
    beta1: float = 0.9                      # [SPEC EN:627]
    beta2: float = 0.999                    # [SPEC EN:627]
    gamma: float = 0.99                     # [SPEC EN:631]
    replay_size: int = 1_000_000            # [SPEC EN:630]
    batch_size: int = 256                   # [SPEC EN:632]
    tau: float = 0.01                       # [SPEC EN:633]
    eps_start: float = 1.0                  # [SPEC EN:634]
    eps_end: float = 0.05                   # [SPEC EN:634]
    eps_decay_episodes: int = 5000          # [SPEC EN:634]
    episodes: int = 10_000                  # [SPEC EN:635]
    n_seeds: int = 10                       # [SPEC EN:636]  seeds 0-9
    agent_share_weights: bool = True        # [DEVIATION] paper implies per-agent nets;
                                            #   homogeneous agents -> one shared net.
    critic_input: str = "summary"           # [DEVIATION] "summary" | "concat"
    update_every: int = 10                  # [DEVIATION] gradient cadence unspecified


# Observation / action dimensions -- derived, but stated in the manuscript.
# [SPEC EN:523-552] obs = 5(K+1) + 2 + 5 + (K+1) = 6K + 13
OBS_DIM_AT_K5 = 6 * 5 + 13  # = 43
ACTION_DIM_AT_K5 = 5 + 2    # = 7   [SPEC EN:554-561]


# Scheme names in the manuscript's Table 7 column order. Names only -- the
# published numbers themselves live in scripts/04_eval_table7.py, deliberately
# OUTSIDE this package, so that tests/test_integrity_lint.py can forbid every
# Table 7 value anywhere under `gapmoh/`.
SCHEME_ORDER = ["RSS-Threshold", "Graph-Dijkstra", "CNN-LSTM-Predict",
                "MADRL-Standard", "Grid-Only", "GAP-MOH"]
