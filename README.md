# CSAD: Cross-Satellite Authentication Diversity for Space-Based ADS-B

A Python simulator for **cross-satellite TESLA authentication** of space-based
ADS-B surveillance. It propagates a LEO constellation with SGP4, computes the
1090 MHz link budget and ALOHA garbling per beam, and compares four receiver
architectures on authentication delay, deadline probability and the
Authenticated Surveillance Age (ASA).

---

## 1. What the code computes

Space-based ADS-B collects 1090 MHz Extended Squitter (1090ES) position reports
through LEO satellites. The protocol carries no authentication, so reports can be
injected, modified or replayed. TESLA (Timed Efficient Stream Loss-tolerant
Authentication) is a bandwidth-efficient broadcast primitive that fits the 56-bit
ME field, but conventional designs require the report and the disclosing key to be
received by the *same* satellite.

In a LEO pass that assumption is costly: a satellite that hears a report high in
the sky may be below the horizon when the key is disclosed one or two seconds
later, while another satellite of the same constellation is rising over the same
aircraft.

CSAD removes the same-satellite restriction using the one-way property of the
TESLA hash chain

```
K_i = H^(j-i)(K_j)          for j >= i
```

so a report received by satellite *A* can be verified on the ground from a later
key disclosure received by satellite *B*, with no change to the aircraft, the
message format or the cryptography.

---

## 2. Repository contents

```
.
├── sim_csad.py        # simulator: geometry, link budget, ALOHA, four receivers, figures
├── requirements.txt   # Python dependencies
├── CITATION.cff       # software citation metadata
├── LICENSE            # MIT licence
└── README.md          # this file
```

`figs/` is created at runtime by the simulator and is not tracked by git.

---

## 3. Installation

Python 3.9 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Usage

### 4.1 Full experiment

```bash
python sim_csad.py
```

Roughly 10–20 minutes on a laptop. Prints the baseline and constellation-sweep
results to the console and writes all figures plus the result tables to `figs/`.

### 4.2 Quick run (smoke test)

```bash
python sim_csad.py --quick --seed 3
```

`--quick` reduces the aircraft sample from 60 to 12 per configuration. Results are
statistically noisier but the code path is identical.

### 4.3 Command-line options

| Option    | Default | Meaning                                             |
|-----------|---------|-----------------------------------------------------|
| `--quick` | off     | small aircraft sample (smoke test)                  |
| `--seed`  | `3`     | base random seed, reused across traffic densities   |
| `--dpi`   | `600`   | figure resolution override                          |

### 4.4 Real orbital elements instead of the synthetic Walker-star

```python
from sim_csad import Scenario

# any two-line-element file with "1 "/"2 " line pairs works
sc = Scenario(n_planes=6, n_per_plane=11, n_ac=40, tle="iridium.tle")
res = sc.run()
```

The constellation geometry, visibility statistics and all curves then follow the
published elements rather than the synthetic pattern; the conclusions depend on
constellation *overlap*, which both cases expose.

---

## 5. Model overview

### 5.1 Pipeline

```
constellation elements  ->  SGP4 (TEME)  ->  ECEF (GMST rotation)
        -> per-aircraft, per-satellite slant range and elevation
        -> link budget: PPM and PPM + 8PSK phase-overlay decoding probability,
           averaged over a 4 dB Gaussian gain variation (40-point Gauss-Hermite)
        -> pure-ALOHA garbling per satellite beam (N aircraft per beam)
        -> Bernoulli reception of reports, MACs and key disclosures per satellite
        -> four receiver architectures:
              IS-0   independent satellite, no hash-chain recovery
              IS-R   independent satellite, hash-chain recovery (standard TESLA)
              NF     naive ground fusion, exact key only
              CSAD   ground fusion + cross-satellite hash-chain recovery  (proposed)
```

### 5.2 Receiver architectures

| Architecture | Reports used  | Key used                  | Hash-chain recovery | Cross-satellite fusion |
|--------------|---------------|---------------------------|---------------------|------------------------|
| **IS-0**     | one satellite | exact `K_i`, same sat.    | no                  | no                     |
| **IS-R**     | one satellite | any later key, same sat.  | yes                 | no                     |
| **NF**       | all satellites| exact `K_i`, any satellite| no                  | yes                    |
| **CSAD**     | all satellites| any later key, any sat.   | yes                 | yes                    |

`NF` vs `IS-R` isolates the value of ground fusion; `CSAD` vs `NF` isolates the
value of cross-satellite hash-chain recovery.

### 5.3 Key parameters (`P` dictionary at the top of `sim_csad.py`)

| Parameter | Value | Unit | Meaning |
|-----------|-------|------|---------|
| `alt_sat` | 780 | km | satellite altitude |
| `inc` | 86.4 | deg | inclination (polar) |
| `h_ac`, `v_ac` | 11, 0.25 | km, km/s | aircraft altitude and speed |
| `el_mask` | 5 | deg | minimum elevation for visibility |
| `f_MHz` | 1090 | MHz | ADS-B carrier |
| `Pt_dBm`, `Gt_dBi` | 54, 0 | dBm, dBi | aircraft transmit power and antenna gain |
| `Gr_dBi`, `Tsys_K`, `B_Hz` | 12, 500, 2.6e6 | dBi, K, Hz | satellite receiver |
| `sigma_dB` | 4 | dB | std. dev. of gain variation |
| `tau` | 120 | µs | 1090ES message duration |
| `lam_ads` | 6.2 | s⁻¹ | ADS-B squitter rate per aircraft |
| `dt` | 0.5 | s | position report period |
| `T_int`, `d`, `r` | 1, 1, 1 | s, intervals, copies | TESLA interval, disclosure delay, copies |
| `Dmax` | 4 | s | authentication deadline |
| `N` | 200 | — | aircraft per satellite beam |
| `window`, `warmup` | 1800, 60 | s | track length and discarded warm-up |

### 5.4 Metrics

* **`P(D_auth ≤ Dmax)`** — deadline probability: fraction of reports authenticated
  within the deadline (default 4 s).
* **`eta`** — eventual verification ratio: fraction of reports ever verified (the
  ceiling set by report reception).
* **ASA (Authenticated Surveillance Age)** — age in seconds of the newest
  *verified* position report, i.e. age-of-information of the trusted picture.
* **`p95_rx`** — 95th-percentile age of the newest *received* (not necessarily
  verified) report, i.e. the freshness cost of authentication.

---

## 6. Outputs

Running `python sim_csad.py` writes to `figs/`:

| File | Content |
|------|---------|
| `fig1_link_and_visibility.{png,pdf}` | link budget model and visibility diversity |
| `fig2_handover_timeline.{png,pdf}` | behaviour during a satellite handover |
| `fig3_delay_cdf.{png,pdf}` | authentication-delay CDF |
| `fig4_deadline_vs_diversity.{png,pdf}` | gain vs. constellation overlap |
| `fig5_diversity_vs_repetition.{png,pdf}` | diversity vs. repetition |
| `fig6_latitude.{png,pdf}` | latitude dependence |
| `results.json` | all simulation results, nested |
| `source_data.csv` | the same numbers as one row per `(scope, metric, architecture, value)` |

Figures are exported at 600 dpi (PNG) and as PDF, with Times New Roman lettering.

### 6.1 Baseline (66 satellites, PO carriage, N = 200, r = 1, Dmax = 4 s)

| Metric | IS-0 | IS-R | NF | CSAD |
|--------|------|------|----|------|
| `P(D_auth ≤ 4 s)` | 0.492 | 0.713 | 0.586 | **0.751** |
| `eta` (ever verified) | 0.492 | 0.763 | 0.586 | **0.764** |
| ASA p95 (s) | 6.0 | 5.0 | 5.0 | **4.5** |
| ASA median (s) | 2.5 | 2.5 | 2.0 | **2.0** |

Mean number of visible satellites: **1.96**; two or more are visible **69 %** of
the time. The CSAD gain becomes significant above roughly **1.2** visible
satellites on average and grows with constellation overlap.

---

## 7. Reproducibility notes

* **Deterministic seeds.** All random draws use `numpy.random.default_rng` with
  explicit seeds. The base seed is reused across traffic densities, so differences
  between curves are not Monte-Carlo noise.
* **Self-test.** `run_all()` starts with `crypto_check()`, which builds a real
  80-bit SHA-256 key chain, computes a 32-bit truncated HMAC-SHA-256 with `K_12`,
  derives `K_12` from `K_19` by seven hash evaluations and verifies the MAC. The
  script aborts if the self-test fails, so a successful run also proves that the
  cross-satellite key-recovery operation used in the model is cryptographically
  sound.
* **Ordering guarantee.** CSAD never authenticates later than IS-0, IS-R or NF for
  the same reception events. All four architectures are evaluated on the *same*
  Bernoulli draws, so the ordering can be checked directly in `results.json`.
* **Source data.** `figs/source_data.csv` is sufficient to redraw every figure
  without re-running the simulation.

---

## 8. Assumptions and limitations

1. Constellations are synthetic Walker-star patterns; real two-line elements are
   accepted but shift the absolute visibility statistics, not the qualitative
   conclusion, which depends on overlap.
2. Reception events at different satellites are treated as **independent**.
   Correlated shadowing or a footprint-wide interferer would reduce the gain; the
   extreme case degenerates to IS-R.
3. The on-board receiver has no capture effect and traffic density is equal in all
   beams. Capture would raise `p_s,j` for all architectures and compress the curves
   towards each other.
4. The overlay code correcting `t_c = 8` symbol errors is an assumption, so the
   phase-overlay results are a **lower bound** on the benefit of the carriage
   method.
5. Key-disclosure starvation (denial of service) is not modelled quantitatively.

---

## 9. Citation and licence

If you use this code, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

Released under the MIT Licence (see `LICENSE`).

Rizky Pratama Hudhajanto — Politeknik Negeri Batam, Batam, Indonesia.
Issues and pull requests are welcome.
