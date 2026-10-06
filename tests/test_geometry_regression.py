# -*- coding: utf-8 -*-
"""Geometry regression as a pytest, wrapping scripts/00_verify_geometry.py.

The released geometry simulation (`handover_frequency_sim/`) publishes two
figures that any reimplementation must reproduce:

    Reactive (max-elev) : 18.9167 events/hour, window mean 3.1673 min, n = 227
    Necessary (GBPT)    :  9.1667 events/hour, window mean 6.5127 min, n = 110

If `orbit.py` drifts from that model, these fail. They are the anchor that
makes everything downstream (channel, passes, FSM) trustworthy.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "verify_geometry", ROOT / "scripts" / "00_verify_geometry.py")
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)


@pytest.fixture(scope="module")
def sims():
    react = vg.simulate(vg._select_max_elev)
    nec = vg.simulate(vg._select_max_remaining)
    return react, nec


def test_reactive_rate(sims):
    n, w = sims[0]
    rate = n / vg.DURATION_H
    assert abs(rate - vg.REF_REACTIVE_RATE) <= vg.TOL_RATE, \
        f"reactive {rate:.4f}/h vs ref {vg.REF_REACTIVE_RATE}"


def test_reactive_window(sims):
    n, w = sims[0]
    assert abs(w.mean() - vg.REF_REACTIVE_WINDOW_MIN) <= vg.TOL_WINDOW


def test_necessary_rate(sims):
    n, w = sims[1]
    rate = n / vg.DURATION_H
    assert abs(rate - vg.REF_NECESSARY_RATE) <= vg.TOL_RATE, \
        f"necessary {rate:.4f}/h vs ref {vg.REF_NECESSARY_RATE}"


def test_necessary_window(sims):
    n, w = sims[1]
    assert abs(w.mean() - vg.REF_NECESSARY_WINDOW_MIN) <= vg.TOL_WINDOW


def test_necessary_halves_the_handover_rate(sims):
    """GBPT should roughly halve the reactive rate -- the paper's framing."""
    react_rate = sims[0][0] / vg.DURATION_H
    nec_rate = sims[1][0] / vg.DURATION_H
    assert nec_rate < react_rate
    assert 1.8 < react_rate / nec_rate < 2.4
