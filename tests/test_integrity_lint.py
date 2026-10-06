# -*- coding: utf-8 -*-
"""Integrity lint: the anti-circularity rule, enforced mechanically.

The whole exercise is worthless if the simulation can read the numbers it is
supposed to predict. Table 7's seven rows are therefore forbidden anywhere
under `gapmoh/`. They may appear only in `scripts/04_eval_table7.py` (the
comparison table) and in `scripts/01_diagnostic.py` (which treats the
execution-duration row as an INPUT, exactly as CN:199 specifies).

Implementation note: a few Table 7 numbers are legitimately also SPEC
constants -- 10.4 dB/K is the link-budget G/T, 12.0 dB is the rain margin,
3.2 ms is GAP-MOH's own execution duration quoted in section 4.3. Those live
in `constants.py` on lines carrying a `[SPEC ...]` provenance tag. So the lint
is line-precise: a Table 7 value is a violation only on a line WITHOUT a
[SPEC] tag. That keeps the check strict where it matters (a bare literal in a
policy or the simulator) without false-positiving on the specification.
"""

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "gapmoh"

# The DISTINCTIVE numbers of Table 7 (CN:232-238): the success rates, the
# execution durations, the unnecessary shares, and the latency/SNR entries.
#
# CI half-widths and short round numbers (0.1, 0.5, 0.8, 1.2, 3.2, 10.0, ...)
# are deliberately NOT listed: they collide constantly with generic code
# (bisection midpoints, log coefficients, array indices), so including them
# buries the real signal in false positives. Every value below is one no
# plausible implementation would arrive at by accident.
TABLE7_VALUES = [
    # row 1, timing-feasibility success rate (%)
    "79.6", "87.0", "90.1", "92.4", "94.1", "96.8",
    # row 2, execution duration (ms)
    "187.3", "98.5", "45.2", "24.7", "12.5",
    # row 3, unnecessary handover (%)
    "23.4", "15.1", "11.6", "8.3", "6.7", "4.1",
    # row 4, handovers per hour
    "10.8", "9.8", "9.6",
    # row 5, mean received SNR (dB)
    "14.2", "16.8", "17.2", "17.9", "18.3", "19.1",
    # row 6, Doppler capture (ms)
    "45.3", "42.0",
    # row 7, per-decision latency (ms)
    "197.4", "48.2", "11.8", "10.3",
]

_LIT = re.compile(r"\d+\.\d+")


def _violations(text):
    """Table 7 literals on lines that are not tagged [SPEC]."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if "[SPEC" in line:
            continue
        hits = sorted(set(_LIT.findall(line)) & set(TABLE7_VALUES))
        if hits:
            out.append((i, hits, line.strip()))
    return out


def test_no_table7_values_in_package():
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        for ln, hits, src in _violations(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}:{ln} {hits} :: {src}")
    assert not offenders, (
        "Table 7 values found under gapmoh/ without a [SPEC] tag -- these are "
        "outputs to be compared against, never inputs:\n  "
        + "\n  ".join(offenders))


def test_dt_exec_has_no_default():
    """`simulate_terminal` must not default its execution duration."""
    src = (PKG / "sim.py").read_text(encoding="utf-8")
    m = re.search(r"def simulate_terminal\((.*?)\):", src, re.S)
    assert m, "simulate_terminal not found"
    args = [a.strip() for a in m.group(1).split(",")]
    dt = [a for a in args if a.startswith("dt_exec_s")]
    assert dt, "simulate_terminal must take dt_exec_s"
    assert "=" not in dt[0], (
        "dt_exec_s must have no default: the published execution durations "
        "are an input supplied by the caller, never a package constant")


def test_scheme_has_no_published_rate_field():
    """A Scheme must not carry any field named like a published result."""
    src = (PKG / "policies.py").read_text(encoding="utf-8")
    body = src.split("class Scheme:")[1].split("@dataclass")[0]
    for bad in ("success", "rate_per_hour", "unnecessary_pct", "mean_snr"):
        assert bad not in body, f"Scheme carries a result field {bad!r}"


def test_package_never_imports_the_eval_script():
    """The simulator must not reach into the comparison script.

    Only actual imports count -- a docstring is allowed to name the file.
    """
    for path in sorted(PKG.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "04_eval_table7" in line:
                assert not re.match(r"\s*(?:from|import)\b", line), \
                    f"{path.name}:{i} imports the eval script"
