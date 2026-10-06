# -*- coding: utf-8 -*-
"""The scenario theatre is where the scenario says it is.

WHY THIS FILE EXISTS. `05_new_metrics.py` originally drew its terminals in
DEGREES and handed them straight to `build_pass_table`, which takes RADIANS.
Nothing raised. The terminals were simply re-placed somewhere else on the
sphere, and because `build_pass_table` returns an EMPTY table for a point the
constellation never reaches, those terminals contributed no events and no
serving trace -- they dropped out of every metric instead of failing.

Concretely: lat = 20 (read as radians) is 65.9 deg N, above the reach of a
53 deg constellation, so `n_pass` came back 0 and the terminal was a no-op.
The schemes that did have coverage still produced plausible-looking rows, so
the summary table looked fine. That is the failure mode these two tests close:
one on the geography the pass table actually sees, one on the units the
terminal generator actually returns.

Both are cheap (a handful of 600 s pass tables) and neither asserts any
performance figure.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gapmoh.constants import Scenario            # noqa: E402
from gapmoh.orbit import build_constellation     # noqa: E402
from gapmoh.passes import build_pass_table       # noqa: E402


def _load_new_metrics_script():
    """Import `scripts/05_new_metrics.py` by path (scripts/ is not a package)."""
    path = ROOT / "scripts" / "05_new_metrics.py"
    spec = importlib.util.spec_from_file_location("_new_metrics_05", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg():
    return Scenario()


@pytest.fixture(scope="module")
def con(cfg):
    return build_constellation(cfg.n_planes, cfg.sats_per_plane,
                               cfg.altitude_km,
                               np.deg2rad(cfg.inclination_deg),
                               cfg.phase_factor)


def test_theatre_terminals_actually_see_satellites(cfg, con):
    """Every corner of the service theatre must have a non-empty pass table.

    The scenario theatre is 20 +/- 15 deg N, 110-150 deg E (CN:190). A 53 deg
    constellation at 550 km covers it comfortably, so an empty table anywhere
    inside it means the terminal was placed somewhere else -- not that the
    geometry is hard.
    """
    for lat_deg in (cfg.lat_center_deg - cfg.lat_spread_deg,
                    cfg.lat_center_deg,
                    cfg.lat_center_deg + cfg.lat_spread_deg):
        for lon_deg in (cfg.lon_min_deg, cfg.lon_max_deg):
            table = build_pass_table(np.deg2rad(lat_deg), np.deg2rad(lon_deg),
                                     cfg.speed_mean_kn, con, cfg, 600.0,
                                     dt_samp=2.0)
            assert table.n_pass > 0, (
                f"no satellite above the {cfg.min_elev_deg:.0f} deg mask in "
                f"600 s for a terminal at {lat_deg:.1f} N, {lon_deg:.1f} E. "
                "The theatre is inside the constellation's coverage, so this "
                "means the terminal was placed at the wrong coordinates "
                "(most often: degrees passed where radians are expected).")


def test_make_terminals_returns_radians(cfg):
    """The generator's output must already be in the units consumers expect.

    Bound check rather than a value check: a value can be right for the wrong
    reason, a range cannot. Latitudes in radians live in [0, pi/2] and
    longitudes in [0, 2*pi); the same numbers in degrees do not.
    """
    mod = _load_new_metrics_script()
    lats, lons, spds = mod.make_terminals(50, cfg, seed=0)

    lat_floor = np.deg2rad(cfg.lat_center_deg - cfg.lat_spread_deg)
    lat_ceil = np.deg2rad(cfg.lat_center_deg + cfg.lat_spread_deg)
    assert np.all(lats >= lat_floor - 1e-12) and np.all(lats <= lat_ceil), (
        f"latitudes span [{lats.min():.3f}, {lats.max():.3f}] rad; the "
        f"theatre is [{lat_floor:.3f}, {lat_ceil:.3f}] rad "
        f"({cfg.lat_center_deg - cfg.lat_spread_deg:.0f}-"
        f"{cfg.lat_center_deg + cfg.lat_spread_deg:.0f} deg N). Values near "
        f"{cfg.lat_center_deg:.1f} mean degrees are being used as radians.")

    assert np.all(lons >= np.deg2rad(cfg.lon_min_deg) - 1e-12) and \
        np.all(lons <= np.deg2rad(cfg.lon_max_deg) + 1e-12), (
        f"longitudes span [{lons.min():.3f}, {lons.max():.3f}] rad; the "
        f"theatre is [{np.deg2rad(cfg.lon_min_deg):.3f}, "
        f"{np.deg2rad(cfg.lon_max_deg):.3f}] rad "
        f"({cfg.lon_min_deg:.0f}-{cfg.lon_max_deg:.0f} deg E).")

    # And the whole point of the conversion: these terminals must be locatable.
    assert np.all(np.abs(np.rad2deg(lats)) <= 90.0), \
        "latitude outside [-90, 90] deg: the value is not a latitude at all"
