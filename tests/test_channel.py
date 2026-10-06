# -*- coding: utf-8 -*-
"""Link-budget assertions against the Chinese manuscript's Table 1 (CN:65-80).

The point of these is that the channel can be checked against the manuscript
WITHOUT running the whole simulation: if the budget ever drifts, the failure
is localised to `channel.py` instead of showing up as a mysterious shift in
the SNR column.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapmoh.channel import (fspl_db, noise_power_dbm, slant_range_from_elev_km,
                            snr_from_elev_db, per_from_snr, rain_loss_db,
                            rain_rate_for_loss_db, rain_margin_elev_deg)
from gapmoh.constants import Scenario


@pytest.fixture(scope="module")
def cfg():
    return Scenario()


def test_noise_power(cfg):
    n = noise_power_dbm(cfg.bandwidth_hz, cfg.noise_temp_k)
    assert abs(n - (-93.0)) < 0.15, f"N = {n:.1f} dBm, paper says -93.0"


def test_slant_range_at_mask(cfg):
    d = slant_range_from_elev_km(np.deg2rad(cfg.min_elev_deg),
                                 cfg.altitude_km)
    assert abs(d - 1518.0) < 3.0, f"slant at 15 deg = {d:.0f} km, paper 1518"


def test_fspl_at_mask(cfg):
    d = slant_range_from_elev_km(np.deg2rad(cfg.min_elev_deg),
                                 cfg.altitude_km)
    lfs = fspl_db(d, cfg.freq_hz)
    assert abs(lfs - 185.6) < 0.3, f"L_fs = {lfs:.1f} dB, paper 185.6"


def test_snr_at_mask_and_zenith(cfg):
    snr15 = float(snr_from_elev_db(np.deg2rad(15.0), cfg))
    snr90 = float(snr_from_elev_db(np.deg2rad(90.0), cfg))
    assert abs(snr15 - cfg.snr_ref_15deg_db) < 0.2, \
        f"SNR@15 = {snr15:.1f} dB, paper {cfg.snr_ref_15deg_db}"
    # the paper rounds its zenith figure up to 30.0; the budget gives 29.4
    assert abs(snr90 - cfg.snr_ref_zenith_db) < 1.0, \
        f"SNR@zenith = {snr90:.1f} dB, paper {cfg.snr_ref_zenith_db}"


def test_snr_monotone_in_elevation(cfg):
    e = np.deg2rad(np.linspace(cfg.min_elev_deg, 90.0, 40))
    s = snr_from_elev_db(e, cfg)
    assert np.all(np.diff(s) > 0), "SNR must increase with elevation"


def test_per_monotone_decreasing():
    s = np.linspace(0.0, 30.0, 40)
    p = per_from_snr(s)
    assert np.all(np.diff(p) < 0), "PER must fall as SNR rises"


def test_gt_consistency(cfg):
    """The budget's G/T must be the paper's, i.e. G_R - 10 lg(T_s).

    Two routes to the same quantity: straight from the receiver figures
    (36 dBi - 10 lg 360 K) and recovered from the full SNR chain. They must
    agree with each other and with the manuscript's 10.4 dB/K.
    """
    direct = cfg.g_rx_dbi - 10.0 * np.log10(cfg.noise_temp_k)
    assert abs(direct - cfg.g_over_t_db) < 0.15, \
        f"G/T from G_R and T_s = {direct:.2f}, paper {cfg.g_over_t_db}"

    # Recover it from the full chain. Everything is kept in dBm here, so the
    # Boltzmann constant enters as -198.6 dBm/Hz/K:
    #   SNR = EIRP + G/T - L_fs - L_atm + 198.6 - 10 lg B
    # (the +10 lg T from writing G_R as G/T + 10 lg T cancels the -10 lg T
    # inside N). Mixing the dBW constant -228.6 in here instead, while EIRP
    # stays in dBm, is a silent 30 dB error -- hence the explicit check.
    d = slant_range_from_elev_km(np.deg2rad(15.0), cfg.altitude_km)
    lfs = float(fspl_db(d, cfg.freq_hz))
    snr = float(snr_from_elev_db(np.deg2rad(15.0), cfg))
    gt = (snr - cfg.eirp_dbm + lfs + cfg.l_atm_db
          - 198.6 + 10.0 * np.log10(cfg.bandwidth_hz))
    assert abs(gt - cfg.g_over_t_db) < 0.3, f"G/T = {gt:.2f} dB/K, paper 10.4"


# ---------------------------------------------------------------------------
# Rain. The slant path through the rain layer shortens as elevation rises, so
# attenuation must FALL with elevation. The earlier implementation held the
# path at a constant 5 km and then divided by cos(elevation), which made
# attenuation RISE and diverge at the zenith -- a sign error that would invert
# any availability ranking built on it. These lock the correct direction in.
# ---------------------------------------------------------------------------

def test_rain_falls_with_elevation(cfg):
    rate = 10.0
    e = np.deg2rad(np.linspace(cfg.min_elev_deg, 90.0, 50))
    a = rain_loss_db(e, cfg, rate)
    assert np.all(np.diff(a) < 0), (
        "rain attenuation must decrease with elevation; got "
        f"{a[0]:.2f} dB at {cfg.min_elev_deg:.0f} deg rising to {a[-1]:.2f} "
        "dB at zenith")


def test_rain_is_finite_at_zenith(cfg):
    a = float(rain_loss_db(np.deg2rad(90.0), cfg, 10.0))
    assert np.isfinite(a) and 0.0 < a < 30.0, f"zenith rain loss = {a}"


def test_rain_rises_with_rate(cfg):
    e = np.deg2rad(20.0)
    a = [float(rain_loss_db(e, cfg, r)) for r in (1.0, 5.0, 20.0, 50.0)]
    assert all(x < y for x, y in zip(a, a[1:])), f"rain loss not monotone: {a}"


def test_rain_margin_matches_the_manuscript_fade(cfg):
    """The manuscript budgets 12 dB of fade but never states the rain rate.

    Inverting for it must land on a plausible rate AND reproduce the margin;
    if the two disagreed, the manuscript's 12 dB and 99.5% figures could not
    both be right, and that would have to be reported rather than papered over.
    """
    rate, achieved = rain_rate_for_loss_db(
        np.deg2rad(cfg.min_elev_deg), cfg, cfg.rain_margin_db)
    assert abs(achieved - cfg.rain_margin_db) < 0.05, (
        f"inversion returned {achieved:.3f} dB for a "
        f"{cfg.rain_margin_db} dB target")
    assert 1.0 < rate < 60.0, (
        f"implied rain rate {rate:.2f} mm/h is not a plausible regional value")


def test_service_floor_is_above_the_mask(cfg):
    """At the implied rain rate the link cannot hold SNR_min at the mask.

    This is the reason the clear-sky availability metric is degenerate and the
    rain-aware one is not: the 15 deg mask sits BELOW the rain service floor.
    """
    rate, _ = rain_rate_for_loss_db(np.deg2rad(cfg.min_elev_deg), cfg,
                                    cfg.rain_margin_db)
    floor = rain_margin_elev_deg(cfg, rain_rate_mm_h=rate)
    assert cfg.min_elev_deg < floor < 45.0, (
        f"service floor {floor:.2f} deg should sit above the "
        f"{cfg.min_elev_deg:.0f} deg mask and below 45 deg")
    # just above the floor the link meets the requirement; just below it must not
    for e_deg, should_pass in ((floor + 1.0, True), (floor - 1.0, False)):
        net = (float(snr_from_elev_db(np.deg2rad(e_deg), cfg))
               - float(rain_loss_db(np.deg2rad(e_deg), cfg, rate)))
        assert (net >= cfg.snr_min_db) is should_pass, (
            f"at {e_deg:.2f} deg net SNR = {net:.2f} dB, expected "
            f"{'pass' if should_pass else 'fail'} against {cfg.snr_min_db}")
