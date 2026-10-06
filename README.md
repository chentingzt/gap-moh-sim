# gapmoh_sim — independent verification of the GAP-MOH handover criterion

A self-contained, NumPy-only simulator that re-implements the handover model of
the **GAP-MOH** paper *from its stated specification*, and then asks whether the
paper's own criterion reproduces the paper's reported numbers.

> **This is not a reconstruction of the paper's simulator.** It is an independent
> implementation of the criterion the paper defines, written from the manuscript
> text alone. Where the paper's numbers and this package's numbers disagree, the
> difference is reported — never tuned away. The detailed provenance record is
> available from the authors on request.

---

## Headline result

Under the paper's own timing-feasibility criterion

```
Δt_exec  <  t_exit − t_init          (manuscript §3.4)
```

all six schemes score **100.0%, spread 0.00 pp** — including a uniformly random
policy. The criterion is *binary*: never hand over → 0%, ever hand over → 100%.
There is no gradient for a reinforcement-learning agent to climb.

The reason is structural, not stochastic. The GBPT trigger is *defined* as
`t_init = t_exit − Δt_ho`, so the available budget is identically
`Δt_ho = Δt_exec + 2 ms > Δt_exec`; and for the duration-threshold schemes the
budget is identically their threshold. Measured budgets run 2–4 orders of
magnitude above each scheme's own `Δt_exec`.

Full derivation, the three probe tables, and the four candidate remedies that
were tested and rejected are in the provenance record kept by the
authors (available on request).

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Geometry regression against the published anchor values
python scripts/00_verify_geometry.py

# 2. Full test suite (42 assertions)
python -m pytest tests/ -q

# 3. Stage-0 gate — measures the criterion itself; does NOT train
python scripts/01_diagnostic.py --terminals 24 --horizon 3600 \
       --out diagnostic_stage0.json
```

Runtime: the test suite takes ~70 s on a single core. `01_diagnostic.py` at
24 terminals × 3600 s takes a few minutes; the 10-seed M=100 runs under
`results/multiseed/` took substantially longer and are committed as artefacts
rather than re-run.

---

## Repository layout

```
gapmoh/                 the simulator package
  constants.py          every specification constant, tagged [SPEC]/[CHOICE]/[DEVIATION]
  orbit.py              Walker-delta constellation, ECEF propagation, elevation, Doppler
  channel.py            Ka-band link budget (aligned to the paper's Table 1) + ITU-R P.618-13 rain
  passes.py             per-terminal pass tables, active-index on the decision grid
  policies.py           the six schemes' trigger / target-satellite rules
  sim.py                single-terminal handover timeline; trigger instants solved exactly
  metrics.py            the C1/C2/C3 criteria + capacity check
  metrics_v2.py         the rebuilt metric set (M1–M5) used by 05/08/09
  rl/                   NumPy DQN stack: env.py, nets.py, policy_target.py

scripts/                11 numbered entry points (see below)
tests/                  7 modules, 42 assertions
results/                committed run artefacts (JSON + logs + one PNG)
requirements.txt        numpy / matplotlib / pytest
```

### Scripts

| Script | Purpose |
|---|---|
| `00_verify_geometry.py` | geometry regression against the published anchor values |
| `01_diagnostic.py` | Stage-0 gate: measures the criterion itself (no training) |
| `01b_probe_conjuncts.py` | joint-criterion probe (timing ∧ resource ∧ quality) |
| `02_train_small.py` | MADRL-Standard training run → per-episode learning curves |
| `05_new_metrics.py` | the rebuilt metric set, single seed |
| `06_compare_table7.py` | renders the measured-vs-paper comparison table |
| `07_probe_success.py` | success-criterion probe |
| `08_multiseed.py` | 10-seed driver over `05`, with aggregation |
| `09_ablation_drl.py` | framework ablation (grid priors / GBPT / DRL arms) |
| `10_probe_headroom.py` | timing-headroom probe |
| `11_probe_load_regime.py` | per-satellite load-regime probe |

> Scripts `03_*` and `04_*` do not exist. They were planned and then dropped
> once the Stage-0 gate showed that generating a paper-format comparison table
> was premature. The numbering is left as-is rather than renumbered, so that the
> script numbers referenced elsewhere still resolve.

---

## Inputs and outputs

**There is no external input dataset.** The world is generated deterministically
from `gapmoh/constants.py`: a fictitious Walker-delta constellation
(72 planes × 70 satellites, 550 km, 53°), synthetic terminal tracks, and a
closed-form Ka-band link budget. No TLEs, no ephemeris files, no measured traces
are shipped or required — which is why `skyfield`/`sgp4` are not dependencies
(deviation D1, in the authors' provenance record).

**Outputs** live under `results/`. Each run writes a JSON result file plus a
human-readable `*_log.txt`. These are committed as run artefacts.

Two caveats about the committed artefacts, stated rather than silently cleaned:

1. **The logs contain the original machine's absolute paths.** Lines of the form
   `wrote D:\...\results\...` reflect where the run happened. They were *not*
   rewritten: they are evidence of a specific executed run, and editing them
   would falsify the record. Console encoding artefacts (mojibake on
   Chinese-locale Windows) are likewise left intact.
2. **Some non-ASCII paths inside `results/` doc-commentary refer to the
   manuscript sources**, which are not part of this repository. The numbers,
   however, are self-contained.

---

## Three inviolable rules

These are enforced in code and in review, not just documented:

1. **The `gapmoh/` package must not contain any of the paper's Table 7 numbers**,
   unless the line carries a `[SPEC]` provenance tag. Mechanically enforced by
   `tests/test_integrity_lint.py`.
2. **`simulate_terminal`'s `dt_exec_s` has no default value.** The Table 7
   execution durations are an *input* to the criterion, supplied by the caller;
   they must not flow back into the package as constants.
3. **Never tune a parameter to hit a paper number.** When results disagree with
   the paper, report the disagreement and change the data in neither direction.

---

## Verified anchors

Regression values the package reproduces exactly (tolerances in the tests):

| Quantity | Value | Cross-check |
|---|---|---|
| Max-elevation handover rate | 18.9167 /h | matches the published `handover_frequency_sim/` |
| Max-remaining handover rate | 9.1667 /h | matches the published `handover_frequency_sim/` |
| Mean window (max-elev / max-remaining) | 3.1673 / 6.5127 min (n = 227 / 110) | same source |
| Noise floor | −93.0 dBm | paper Table 1 |
| Free-space loss @ 15° | 185.6 dB | paper Table 1 |
| Clear-sky SNR @ 15° / zenith | 20.6 dB / 29.4 dB | paper Table 1 |
| Slant range @ 15° | 1518 km | paper Table 1 |
| G/T | 10.44 dB/K | paper Table 1 |

`handover_frequency_sim/` is a separate, sibling package and is **not** included
here; the anchor values above are the published ones it reports.

---

## Scope and limitations

- **The released environment implements two of the paper's four reward terms.**
  `r_interruption` and `r_load` are identically zero in the evaluated operating
  regime (handovers in this regime do not fail, and per-satellite load stays far
  below the `C = 50` cap), so their omission changes no reported number. The
  observation vector's load slots are likewise hard-coded to zero. This is
  disclosed in the manuscript.
- **The RL backend is NumPy, not PyTorch.** The manuscript specifies PyTorch 2.0
  + CUDA; this environment has neither. The actor network architecture, learning
  rate, and optimiser follow the specification; only the tensor backend differs.
- **The committed training artefacts are a reduced-scale run** (M=20, 1800 s,
  60 episodes, 3 seeds), not the manuscript's 10,000 episodes × 10 seeds at
  M=100. Hyperparameters match the specification; the run scale does not.
- **The ablation arms labelled `MADRL-Basic` / `Grid-Only` are scripted policy
  stand-ins**, not independently trained networks.

---

## Dependencies

```
numpy>=2.0
matplotlib>=3.8
pytest>=8.0
```

Python 3.10+ is assumed (uses `dataclasses`, `pathlib`, PEP 604 unions).

---

## Citation

Manuscript under revision:

> *GAP-MOH: Grid-Boundary Proactive Handover for Asia-Pacific Maritime LEO
> Satellite Networks* — author list and affiliations pending finalisation.

Please cite the paper, not this repository.

---

## License

**No license has been applied yet.** Absent a `LICENSE` file, all rights are
reserved by default. Choose and add a license before making this repository
public.
