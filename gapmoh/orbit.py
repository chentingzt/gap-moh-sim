# -*- coding: utf-8 -*-
"""Orbital geometry: Walker-delta constellation, ECEF propagation, elevation.

The propagator is an analytic circular Keplerian orbit in ECEF. This is the
same model as the already-released, already-validated geometry code
(`handover_frequency_sim/handover_frequency_sim.py`), lifted here so that
`scripts/00_verify_geometry.py` can assert it still reproduces the published
9.17 / 18.92 events-per-hour figures.

[DEVIATION] The manuscript (EN:997-1002) says the simulator uses SGP4 via the
`skyfield` library. It does not, here. Reason: `skyfield` propagates a
catalogue object and therefore needs a TLE; the paper's constellation is a
fictitious 72x70 Walker-delta for which no real TLE exists, and this repository
contains no TLE files at all. Manufacturing a TLE from our own analytic state
and feeding it back through SGP4 would be a no-op round trip that adds fitting
error and licenses no additional fidelity. The propagator seam is a single
function (`sat_ecef`, `sat_ecef_vel`), so an SGP4 backend can be dropped in if
real TLEs are ever supplied.

All positions in km, times in seconds, angles in radians unless a name ends in
`_deg`.
"""

from types import SimpleNamespace

import numpy as np

from .constants import MU, OMEGA_E, R_E

__all__ = [
    "build_constellation", "sat_ecef", "sat_ecef_vel", "sat_ecef_subset",
    "terminal_ecef", "terminal_ecef_multi", "terminal_series",
    "terminal_vel_multi", "elevation", "elevation_multi", "sat_elev_series",
    "slant_range_multi", "range_rate_multi", "doppler_hz_multi",
    "exit_times_for", "elevation_history_for",
]


# --------------------------------------------------------------------------
# Constellation
# --------------------------------------------------------------------------
def build_constellation(n_planes, sats_per_plane, altitude_km, inclination_rad,
                        phase_factor=1):
    """Build a Walker-delta constellation.

    Identical to the released geometry code: uniformly spaced RAANs, evenly
    distributed in-plane argument of latitude with the Walker phasing term.
    """
    total = n_planes * sats_per_plane
    p = np.repeat(np.arange(n_planes), sats_per_plane)
    s = np.tile(np.arange(sats_per_plane), n_planes)
    raan = 2.0 * np.pi * p / n_planes
    u0 = 2.0 * np.pi * (s / sats_per_plane + p * phase_factor / total)
    r = R_E + altitude_km
    n = np.sqrt(MU / r ** 3)          # mean motion, rad/s
    return SimpleNamespace(raan=raan, u0=u0, r=r, n=n, total=int(total),
                           inclination=inclination_rad)


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------
def _eci(t, con):
    """ECI position components of every satellite at time t."""
    u = con.u0 + con.n * t
    cu, su = np.cos(u), np.sin(u)
    cosi, sini = np.cos(con.inclination), np.sin(con.inclination)
    cosR, sinR = np.cos(con.raan), np.sin(con.raan)
    x = con.r * (cosR * cu - sinR * su * cosi)
    y = con.r * (sinR * cu + cosR * su * cosi)
    z = con.r * (su * sini)
    return x, y, z, cu, su, cosi, sini, cosR, sinR


def sat_ecef(t, con):
    """ECEF positions of all satellites at time t. Returns (N, 3) km."""
    x, y, z, *_ = _eci(t, con)
    return _to_ecef(t, x, y, z)


def _to_ecef(t, x, y, z):
    th = OMEGA_E * t
    ct, st = np.cos(th), np.sin(th)
    return np.stack([x * ct + y * st, -x * st + y * ct, z], axis=-1)


def sat_ecef_subset(ts, con, sat_idx):
    """ECEF positions of a satellite SUBSET over a time ARRAY.

    `ts` is (T,), `sat_idx` is (S,) satellite ids. Returns (T, S, 3) km.

    This exists purely for speed: `sat_ecef` propagates all 5040 satellites
    even when only a handful are wanted, which dominates the stepping loop
    (`exit_times_for` does exactly that once per look-ahead sample). Here the
    subset is propagated directly.
    """
    ts = np.asarray(ts, dtype=float)
    idx = np.asarray(sat_idx, dtype=np.int64).ravel()
    tt = ts[:, None]                                     # (T, 1)

    u = con.u0[idx][None, :] + con.n * tt                # (T, S)
    cu, su = np.cos(u), np.sin(u)
    cosi, sini = np.cos(con.inclination), np.sin(con.inclination)
    cosR = np.cos(con.raan[idx])[None, :]
    sinR = np.sin(con.raan[idx])[None, :]

    x = con.r * (cosR * cu - sinR * su * cosi)
    y = con.r * (sinR * cu + cosR * su * cosi)
    z = con.r * (su * sini)

    th = OMEGA_E * tt
    ct, st = np.cos(th), np.sin(th)
    return np.stack([x * ct + y * st, -x * st + y * ct, z + 0.0 * x], axis=-1)


def terminal_series(ts, lat, lon0, speed_kn, knots_to_kmps=0.514444 / 1000.0):
    """One terminal's ECEF track over a time array. Returns (T, 3) km."""
    ts = np.asarray(ts, dtype=float)
    lon = lon0 + (speed_kn * knots_to_kmps * ts) / (R_E * np.cos(lat))
    cl, sl = np.cos(lat), np.sin(lat)
    return np.stack([R_E * cl * np.cos(lon), R_E * cl * np.sin(lon),
                     R_E * sl + 0.0 * lon], axis=-1)


def sat_elev_series(ts, terms, con, sat_idx):
    """Elevation history of a satellite subset along a terminal track.

    `ts` (T,), `terms` (T, 3), `sat_idx` (S,). Returns (T, S) radians.
    """
    pos = sat_ecef_subset(ts, con, sat_idx)                       # (T, S, 3)
    up = terms / np.linalg.norm(terms, axis=1, keepdims=True)     # (T, 3)
    rho = pos - terms[:, None, :]
    cosz = np.einsum("tsk,tk->ts", rho, up) / np.linalg.norm(rho, axis=2)
    return np.arcsin(np.clip(cosz, -1.0, 1.0))


def sat_ecef_vel(t, con):
    """ECEF position and velocity of all satellites at time t.

    Returns (pos, vel), each (N, 3), in km and km/s. Velocity is the ECEF rate,
    i.e. the inertial velocity with the Earth-rotation term removed, so that
    differencing it against an Earth-fixed terminal gives the correct range
    rate for Doppler.
    """
    x, y, z, cu, su, cosi, sini, cosR, sinR = _eci(t, con)
    n = con.n
    # d/dt of the ECI components (du/dt = n)
    dx = con.r * (-cosR * su - sinR * cu * cosi) * n
    dy = con.r * (-sinR * su + cosR * cu * cosi) * n
    dz = con.r * (cu * sini) * n

    th = OMEGA_E * t
    ct, st = np.cos(th), np.sin(th)
    xe = x * ct + y * st
    ye = -x * st + y * ct

    vxe = (dx * ct + dy * st) + OMEGA_E * ye
    vye = (-dx * st + dy * ct) - OMEGA_E * xe
    vze = dz
    pos = np.stack([xe, ye, z], axis=-1)
    vel = np.stack([vxe, vye, vze], axis=-1)
    return pos, vel


# --------------------------------------------------------------------------
# Terminals
# --------------------------------------------------------------------------
def terminal_ecef(t, lat, lon0, speed_kn, knots_to_kmps=0.514444 / 1000.0):
    """Single terminal ECEF position, drifting eastward at `speed_kn` knots."""
    lon = lon0 + (speed_kn * knots_to_kmps * t) / (R_E * np.cos(lat))
    return np.array([R_E * np.cos(lat) * np.cos(lon),
                     R_E * np.cos(lat) * np.sin(lon),
                     R_E * np.sin(lat)])


def terminal_ecef_multi(t, lat, lon0, speed_kn, knots_to_kmps=0.514444 / 1000.0):
    """Vectorized terminal positions.

    `lat`, `lon0`, `speed_kn` are (M,) arrays (radians, radians, knots).
    Returns (M, 3) km. Matches the released multi-terminal code.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon0, dtype=float) + \
        (np.asarray(speed_kn, dtype=float) * knots_to_kmps * t) / (R_E * np.cos(lat))
    cl, sl = np.cos(lat), np.sin(lat)
    clon, slon = np.cos(lon), np.sin(lon)
    return np.stack([R_E * cl * clon, R_E * cl * slon, R_E * sl], axis=-1)


def terminal_vel_multi(lat, lon0, speed_kn, t, knots_to_kmps=0.514444 / 1000.0):
    """ECEF velocity of drifting terminals. Returns (M, 3) km/s.

    A point fixed to the rotating Earth has zero ECEF velocity, so only the
    ship's own eastward motion over the surface contributes.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon0, dtype=float) + \
        (np.asarray(speed_kn, dtype=float) * knots_to_kmps * t) / (R_E * np.cos(lat))
    v = np.asarray(speed_kn, dtype=float) * knots_to_kmps
    zeros = np.zeros_like(lon)
    return np.stack([-v * np.sin(lon), v * np.cos(lon), zeros], axis=-1)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def elevation(sat_xyz, term_xyz):
    """Elevation (rad) of each satellite above one terminal's local horizon."""
    rho = sat_xyz - term_xyz[None, :]
    up = term_xyz / np.linalg.norm(term_xyz)
    cos_zenith = (rho @ up) / np.linalg.norm(rho, axis=1)
    return np.arcsin(np.clip(cos_zenith, -1.0, 1.0))


def elevation_multi(sat_xyz, terms):
    """Elevation (rad) for M terminals against N satellites. Returns (M, N)."""
    rho = sat_xyz[None, :, :] - terms[:, None, :]          # (M, N, 3)
    up = terms / np.linalg.norm(terms, axis=1, keepdims=True)
    cos_zenith = np.einsum("mnk,mk->mn", rho, up) / np.linalg.norm(rho, axis=2)
    return np.arcsin(np.clip(cos_zenith, -1.0, 1.0))


def slant_range_multi(sat_xyz, terms):
    """Slant range (km), shape (M, N)."""
    rho = sat_xyz[None, :, :] - terms[:, None, :]
    return np.linalg.norm(rho, axis=2)


def range_rate_multi(sat_pos, sat_vel, terms, term_vel):
    """Range rate d|rho|/dt (km/s), shape (M, N). Positive = receding."""
    rho = sat_pos[None, :, :] - terms[:, None, :]
    d = np.linalg.norm(rho, axis=2)
    dv = sat_vel[None, :, :] - term_vel[:, None, :]
    return np.einsum("mnk,mnk->mn", rho, dv) / np.maximum(d, 1e-9)


def doppler_hz_multi(sat_pos, sat_vel, terms, term_vel, freq_hz,
                     c_kmps=299792.458):
    """Doppler shift (Hz), shape (M, N). Negative when approaching."""
    rr = range_rate_multi(sat_pos, sat_vel, terms, term_vel)
    return -freq_hz * rr / c_kmps


# --------------------------------------------------------------------------
# Exit times (the grid's pre-index quantity, computed by forward look-ahead)
# --------------------------------------------------------------------------
def elevation_history_for(t, term_xyz, con, sat_idx, dt_look, look_cap):
    """Elevation of a chosen satellite subset over a forward look-ahead.

    `sat_idx` is an index array of satellite ids. Returns (n_look, len(sat_idx)).
    Only the requested satellites are propagated, which is what makes this
    affordable inside the stepping loop.
    """
    idx = np.asarray(sat_idx).ravel()
    n_look = int(look_cap / dt_look)
    out = np.empty((n_look, idx.size))
    for k in range(n_look):
        tt = t + (k + 1) * dt_look
        out[k] = elevation(sat_ecef(tt, con)[idx], term_xyz)
    return out


def exit_times_for(t, term_xyz, con, sat_idx, min_elev, dt_look=0.1,
                   look_cap=60.0):
    """Remaining visibility (s) of each requested satellite.

    Returns an array aligned with `sat_idx`; a satellite that stays above the
    mask for the whole look-ahead window is reported as `look_cap`.
    """
    idx = np.asarray(sat_idx).ravel()
    rem = np.full(idx.size, float(look_cap))
    still_up = np.ones(idx.size, dtype=bool)
    for k in range(int(look_cap / dt_look)):
        tt = t + (k + 1) * dt_look
        e = elevation(sat_ecef(tt, con)[idx], term_xyz)
        newly = still_up & (e < min_elev)
        rem[newly] = tt - t
        still_up &= ~newly
        if not still_up.any():
            break
    return rem
