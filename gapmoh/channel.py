# -*- coding: utf-8 -*-
"""Ka-band link budget, per the manuscript's own budget table.

Replaces the toy elevation->SNR curve used in the earlier scratch reproduction
(`snr_from_elev(e) = 30 - 12*((90-e)/75)^2`), which had no path loss, no
frequency dependence and no rain.

The budget is the Chinese manuscript's Table 1 (CN:65-80), which is the
self-consistent one; the English manuscript's parameters (P_T 40 dBm, G_T
38 dBi, G_R 35 dBi, T_s 300 K) give the same G/T to within 0.2 dB.

    L_fs   = 20 lg(4 pi d / lambda)          -> 185.6 dB at the 15 deg slant
    P_r    = EIRP + G_R - L_fs - L_atm
    N      = k T_s B                         -> -93.0 dBm at 360 K / 100 MHz
    SNR    = P_r - N                         -> 20.6 dB at 15 deg, 30.0 at zenith

`tests/test_channel.py` asserts those two reference SNRs and the 1518 km slant
range, so the link can be checked against the manuscript without running the
whole simulation.

Design note on rain: CN:80-82 budgets a 12 dB fade at 99.5% availability and
states it is absorbed by adaptive coding and uplink power control. Section 3.4's
success criterion explicitly does not model target-link rain fade. Rain is
therefore available as an optional impairment (`impairments=...`) but is OFF by
default, so that the handover simulation measures what the paper says it
measures.
"""

import numpy as np

from .constants import R_E

__all__ = ["slant_range_from_elev_km", "fspl_db", "noise_power_dbm",
           "snr_db_from_slant", "link_snr_db", "rain_attenuation_db",
           "rain_loss_db", "rain_rate_for_loss_db", "rain_margin_elev_deg",
           "per_from_snr", "snr_from_elev_db", "BOLTZMANN_DBM_PER_K_HZ"]


def snr_from_elev_db(elev_rad, cfg, alt_km=None, extra_loss_db=0.0):
    """Clear-sky downlink SNR (dB) as a function of elevation alone.

    Chain: elevation -> slant range -> free-space loss -> budget. This is the
    inverse of `slant_range_from_elev_km` composed with `snr_db_from_slant`,
    and it is how the per-step serving-link SNR is obtained without
    re-propagating a satellite that the pass table already located.
    """
    alt = cfg.altitude_km if alt_km is None else alt_km
    slant = slant_range_from_elev_km(elev_rad, alt)
    return snr_db_from_slant(slant, cfg.freq_hz, cfg.bandwidth_hz,
                             cfg.eirp_dbm, cfg.g_rx_dbi, cfg.l_atm_db,
                             cfg.noise_temp_k, extra_loss_db=extra_loss_db)

# 10*log10(k_B) in dBm/Hz/K
BOLTZMANN_DBM_PER_K_HZ = -198.6

# ITU-R P.838 specific-attenuation coefficients for linear horizontal
# polarisation at 30 GHz (the conservative maritime choice).
_K_H_30GHZ = 0.1874
_ALPHA_H_30GHZ = 0.7620


def slant_range_from_elev_km(elev_rad, alt_km=550.0):
    """Slant range (km) to a satellite at `alt_km` seen at elevation `elev_rad`.

    At 15 deg and 550 km this returns the manuscript's 1518 km.
    """
    r = R_E + alt_km
    e = np.asarray(elev_rad, dtype=float)
    inner = r ** 2 - (R_E * np.cos(e)) ** 2
    return -R_E * np.sin(e) + np.sqrt(np.maximum(inner, 0.0))


def fspl_db(slant_km, freq_hz):
    """Free-space path loss (dB): 20 lg(4 pi d / lambda)."""
    d = np.maximum(np.asarray(slant_km, dtype=float), 1e-3)
    return 92.45 + 20.0 * np.log10(freq_hz / 1e9) + 20.0 * np.log10(d)


def noise_power_dbm(bandwidth_hz, noise_temp_k):
    """System noise power N = k T_s B, in dBm."""
    return (BOLTZMANN_DBM_PER_K_HZ
            + 10.0 * np.log10(noise_temp_k)
            + 10.0 * np.log10(bandwidth_hz))


def rain_attenuation_db(elev_rad, freq_hz, rain_rate_mm_h,
                        rain_height_km=4.0):
    """Slant-path rain attenuation (dB), ITU-R P.618-13 (Sections 2.2.1.1).

    The three steps of the standard:

        L_S = h_R / sin(theta)                       slant path in the rain
        L_G = L_S cos(theta)                         its horizontal projection
        r   = 1 / (1 + 0.78 sqrt(L_G gamma_R / f)    horizontal reduction
                   - 0.38 (1 - exp(-2 L_G)))
        A   = gamma_R L_S r                          attenuation

    with gamma_R = k R^alpha from the P.838 30 GHz coefficients.

    WHY THE PREVIOUS VERSION WAS WRONG. It held the in-rain path at a constant
    5 km and then divided by cos(elevation), so attenuation GREW with elevation
    and diverged at the zenith -- the exact opposite of the physics, and enough
    to invert any availability ranking built on it. The corrected form makes
    L_S fall as h_R/sin(theta), which is what actually shortens the path a high
    satellite takes through the rain layer.

    The reduction factor r is capped at 1. As L_G -> 0 (near zenith) the raw
    expression tends to 1/(1-0.38) = 1.61, which would inflate the zenith path
    length and break the monotonicity the model is supposed to have. Capping r
    at unity is standard practice and keeps A strictly decreasing in elevation.

    Diagnostic only -- not part of the handover criterion. See
    `metrics_v2.py`, where it enters the availability metric.
    """
    e = np.asarray(elev_rad, dtype=float)
    gamma_r = _K_H_30GHZ * rain_rate_mm_h ** _ALPHA_H_30GHZ
    f_ghz = freq_hz / 1e9
    sin_e = np.sin(e)
    # theta >= 5 deg is the closed form in the standard; below it the Earth's
    # curvature shortens the path and the (2 h_R / R_E) term takes over.
    l_s = np.where(
        e >= np.deg2rad(5.0),
        rain_height_km / np.maximum(sin_e, 1e-9),
        2.0 * rain_height_km
        / (np.sqrt(sin_e ** 2 + 2.0 * rain_height_km / R_E) + sin_e),
    )
    l_g = l_s * np.cos(e)
    r = 1.0 / (1.0 + 0.78 * np.sqrt(l_g * gamma_r / f_ghz)
               - 0.38 * (1.0 - np.exp(-2.0 * l_g)))
    return gamma_r * l_s * np.minimum(r, 1.0)


def rain_loss_db(elev_rad, cfg, rain_rate_mm_h):
    """Rain loss (dB) at this elevation under the scenario's own link budget.

    Thin wrapper so callers never have to reach for `cfg.freq_hz` themselves.
    """
    return rain_attenuation_db(np.asarray(elev_rad, dtype=float), cfg.freq_hz,
                               rain_rate_mm_h)


def rain_rate_for_loss_db(elev_rad, cfg, target_loss_db, lo=0.1, hi=500.0,
                          iters=80):
    """Rain rate (mm/h) whose attenuation at `elev_rad` equals `target_loss_db`.

    The manuscript fixes a 12 dB fade but never states the rain rate that
    produces it, so this inversion is what makes the two comparable. Returns
    (rate, achieved_loss_db); when the target is not reachable inside
    [lo, hi] the bracket endpoint is returned with its true attenuation.
    """
    f = lambda r: float(rain_loss_db(np.asarray([elev_rad]), cfg, r)[0])  # noqa: E731
    if f(hi) < target_loss_db:
        return hi, f(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target_loss_db:
            lo = mid
        else:
            hi = mid
    rate = 0.5 * (lo + hi)
    return rate, f(rate)


def rain_margin_elev_deg(cfg, snr_min_db=None, rain_rate_mm_h=None):
    """Lowest elevation (deg) at which the link still meets `snr_min_db`
    AFTER rain at `rain_rate_mm_h`.

    Below this angle the link cannot claim the assumed rain availability even
    in the middle of a pass, so it is the elevation analogue of a service
    floor. Solved by bisection on the clear-sky budget minus the rain loss.
    """
    snr_min = cfg.snr_min_db if snr_min_db is None else snr_min_db
    e_lo, e_hi = np.deg2rad(5.0), np.deg2rad(90.0)

    def meets(e_deg):
        e = np.deg2rad(e_deg)
        snr = float(snr_from_elev_db(e, cfg)) - float(rain_loss_db(e, cfg, rain_rate_mm_h))
        return snr >= snr_min

    if not meets(90.0):
        return float("nan")
    if meets(5.0):
        return 5.0
    for _ in range(60):
        mid = 0.5 * (e_lo + e_hi)
        if meets(np.rad2deg(mid)):
            e_hi = mid
        else:
            e_lo = mid
    return float(np.rad2deg(0.5 * (e_lo + e_hi)))


def snr_db_from_slant(slant_km, freq_hz, bandwidth_hz, eirp_dbm, g_rx_dbi,
                      l_atm_db, noise_temp_k, extra_loss_db=0.0):
    """Downlink SNR (dB) from the manuscript's budget equation."""
    l_fs = fspl_db(slant_km, freq_hz)
    n = noise_power_dbm(bandwidth_hz, noise_temp_k)
    return eirp_dbm + g_rx_dbi - l_fs - l_atm_db - np.asarray(extra_loss_db) - n


def link_snr_db(sat_pos, sat_vel, terms, term_vel, cfg, extra_loss_db=0.0):
    """SNR (dB) for M terminals against N satellites.

    Returns (snr_db, slant_km, doppler_hz), each shaped (M, N).
    """
    from .orbit import (doppler_hz_multi, elevation_multi, slant_range_multi)

    elev = elevation_multi(sat_pos, terms)
    slant = slant_range_multi(sat_pos, terms)
    dopp = doppler_hz_multi(sat_pos, sat_vel, terms, term_vel, cfg.freq_hz)
    snr = snr_db_from_slant(
        slant, cfg.freq_hz, cfg.bandwidth_hz,
        cfg.eirp_dbm, cfg.g_rx_dbi, cfg.l_atm_db, cfg.noise_temp_k,
        extra_loss_db=extra_loss_db,
    )
    return snr, slant, dopp


def per_from_snr(snr_db, snr_min_db=10.0, per_floor=1e-6):
    """Packet error rate as a decreasing function of SNR.

    [CHOICE] The manuscript states no PER curve. This smooth logistic stand-in
    feeds only the observation vector's PER slot; it never defines handover
    success.
    """
    x = (np.asarray(snr_db, dtype=float) - snr_min_db) / 2.0
    return per_floor + (1.0 - per_floor) / (1.0 + np.exp(2.5 * x))
