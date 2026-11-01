#!/usr/bin/env python3
"""
Cross-Satellite Authentication Diversity (CSAD) for TESLA-authenticated
space-based ADS-B.

An ADS-B report (with its TESLA MAC) received by one LEO satellite can be
verified on the ground with a later TESLA key disclosure received by ANY
satellite, because K_i = H^(j-i)(K_j).

Pipeline
  constellation elements -> SGP4 (TEME) -> ECEF (GMST rotation)
  -> per-aircraft, per-satellite range and elevation
  -> link-budget decoding probability (PPM / PPM + 8PSK phase overlay)
  -> pure-ALOHA garbling per satellite beam (N aircraft per beam)
  -> Bernoulli reception of reports, MACs and key disclosures per satellite
  -> four receiver architectures:
        IS-0  independent satellite, no hash-chain recovery
        IS-R  independent satellite, hash-chain recovery (standard TESLA)
        NF    naive ground fusion, exact key only
        CSAD  ground fusion + cross-satellite hash-chain recovery (proposed)

By default an Iridium-like Walker-star constellation is generated and
propagated with SGP4. Real two-line elements can be used instead with
load_tle("file.tle").

Run:  python3 sim_csad.py        (writes all figures to ./figs)
"""
import hashlib
import hmac
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from sgp4.api import Satrec, SatrecArray, WGS72, jday

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# Figure typography: Times New Roman, 8-9 pt lettering on print-size canvases,
# so that exported figures stay readable at single-column width.
# (target: vector quality at column width, raster fallback at 600 dpi).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})
DPI = 600                       # raster resolution for the PNG exports
FIGSIZE = {                     # (width, height) in inches = printed size
    1: (6.20, 2.40),            # 15.8 cm wide, two panels
    2: (5.15, 3.65),            # 13.1 cm wide, three stacked panels
    3: (5.15, 3.35),            # 13.1 cm
    4: (5.15, 3.35),
    5: (6.20, 2.60),            # 15.8 cm, two panels
    6: (5.15, 3.35),
}

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
P = dict(
    alt_sat=780.0, inc=86.4, h_ac=11.0, v_ac=0.25,   # km, deg, km, km/s
    el_mask=5.0,
    f_MHz=1090.0, Pt_dBm=54.0, Gt_dBi=0.0, L_tx_dB=2.0, L_misc_dB=1.0,
    Gr_dBi=12.0, Tsys_K=500.0, B_Hz=2.6e6, sigma_dB=4.0,
    tau=120e-6, lam_ads=6.2, dt=0.5,                  # dt = position period
    n_bits=112, po_symbols=112, po_t=8, F_frag=2,
    T_int=1.0, d=1, r=1, Dmax=4.0, N=200,
    window=1800.0, warmup=60.0,
)
K_B = 1.380649e-23
MU = 398600.4418
EPOCH = (2026, 9, 30, 0, 0, 0.0)


# ----------------------------------------------------------------------------
# Constellation and SGP4 propagation
# ----------------------------------------------------------------------------
def walker_star(n_planes, n_per_plane, inc=None, alt=None):
    inc = P["inc"] if inc is None else inc
    alt = P["alt_sat"] if alt is None else alt
    a = 6378.135 + alt                       # WGS72 equatorial radius
    n_rad_min = np.sqrt(MU / a**3) * 60.0
    jd, fr = jday(*EPOCH)
    ep = jd + fr - 2433281.5
    sats = []
    for k in range(n_planes):
        raan = np.radians(180.0 * k / n_planes)          # star pattern
        for j in range(n_per_plane):
            M = np.radians((360.0 * j / n_per_plane
                            + (k % 2) * 180.0 / n_per_plane) % 360.0)
            s = Satrec()
            s.sgp4init(WGS72, "i", 1000 + k * 100 + j, ep, 0.0, 0.0, 0.0,
                       1e-4, 0.0, np.radians(inc), M, n_rad_min, raan)
            sats.append(s)
    return sats


def load_tle(path):
    lines = [l.rstrip() for l in open(path) if l.strip()]
    sats = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            sats.append(Satrec.twoline2rv(lines[i], lines[i + 1])); i += 2
        else:
            i += 1
    return sats


def gmst(jd_ut1):
    T = (jd_ut1 - 2451545.0) / 36525.0
    s = (67310.54841 + (876600 * 3600 + 8640184.812866) * T
         + 0.093104 * T**2 - 6.2e-6 * T**3)
    return np.radians((s % 86400) / 240.0)


def propagate_ecef(sats, duration, dt):
    jd0, fr0 = jday(*EPOCH)
    t = np.arange(0.0, duration, dt)
    jd = np.full(t.size, jd0)
    fr = fr0 + t / 86400.0
    e, r, _ = SatrecArray(sats).sgp4(jd, fr)
    assert np.all(e == 0), "SGP4 error"
    th = gmst(jd + fr)
    c, s = np.cos(th), np.sin(th)
    x = c * r[:, :, 0] + s * r[:, :, 1]
    y = -s * r[:, :, 0] + c * r[:, :, 1]
    return t, np.stack([x, y, r[:, :, 2]], axis=-1)      # (S, T, 3) km


def geodetic_to_ecef(lat, lon, h):
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    return np.stack([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - e2) + h) * np.sin(lat)], axis=-1)


def aircraft_track(lat0, lon0, hdg, n, dt):
    """Great-circle track at constant speed (spherical approx. for motion)."""
    R = 6371.0
    dist = P["v_ac"] * dt * np.arange(n) / R
    lat = np.arcsin(np.sin(lat0) * np.cos(dist)
                    + np.cos(lat0) * np.sin(dist) * np.cos(hdg))
    lon = lon0 + np.arctan2(np.sin(hdg) * np.sin(dist) * np.cos(lat0),
                            np.cos(dist) - np.sin(lat0) * np.sin(lat))
    return lat, lon, geodetic_to_ecef(lat, lon, P["h_ac"])


def look_angles(sat_ecef, lat, lon, ac_ecef):
    rho = sat_ecef - ac_ecef[None]
    rng = np.linalg.norm(rho, axis=-1)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                   np.sin(lat)], axis=-1)
    el = np.degrees(np.arcsin(np.einsum("stk,tk->st", rho, up) / rng))
    return el, rng


# ----------------------------------------------------------------------------
# Link model
# ----------------------------------------------------------------------------
def mean_snr_dB(rng_km):
    fspl = 20 * np.log10(rng_km) + 20 * np.log10(P["f_MHz"]) + 32.44
    pr = (P["Pt_dBm"] + P["Gt_dBi"] - P["L_tx_dB"] - P["L_misc_dB"]
          + P["Gr_dBi"] - fspl)
    return pr - (10 * np.log10(K_B * P["Tsys_K"] * P["B_Hz"]) + 30)


def ppm_success(snr_dB):
    ebn0 = 10**(snr_dB / 10) * P["B_Hz"] * 0.5e-6
    pb = 0.5 * np.exp(-ebn0 / 2)
    n = P["n_bits"]
    return (1 - pb)**n + n * pb * (1 - pb)**(n - 1)


def po_success(snr_dB):
    esn0 = 10**(snr_dB / 10) * P["B_Hz"] * 0.5e-6
    ps = np.clip(2 * norm.sf(np.sqrt(2 * esn0) * np.sin(np.pi / 8)), 0, 1)
    return binom.cdf(P["po_t"], P["po_symbols"], ps)


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()
RNG_GRID = np.linspace(700, 3600, 1200)
_s = mean_snr_dB(RNG_GRID)[:, None] + P["sigma_dB"] * GH_X[None]
PPM_G = (ppm_success(_s) * GH_W).sum(1)
PO_G = (ppm_success(_s) * po_success(_s) * GH_W).sum(1)


def p_ppm(rng):
    return np.interp(rng, RNG_GRID, PPM_G)


def p_po(rng):
    return np.interp(rng, RNG_GRID, PO_G)


def p_nc(lam):
    return np.exp(-2 * lam * P["tau"])


def added_rate(scheme, r, T):
    return (1 + P["F_frag"] * r) / T if scheme == "ES" else 0.0


# ----------------------------------------------------------------------------
# TESLA key chain check (real SHA-256 / HMAC)
# ----------------------------------------------------------------------------
def crypto_check():
    n = 40
    k = [None] * n
    k[-1] = hashlib.sha256(b"seed").digest()[:10]
    for i in range(n - 2, -1, -1):
        k[i] = hashlib.sha256(k[i + 1]).digest()[:10]
    msg = bytes.fromhex("8D4840D6202CC371C32CE0576098")
    mac = hmac.new(k[12], msg, hashlib.sha256).digest()[:4]   # "satellite A"
    kk = k[19]                                                  # "satellite B"
    for _ in range(19 - 12):
        kk = hashlib.sha256(kk).digest()[:10]
    ok_anchor = hashlib.sha256(k[1]).digest()[:10] == k[0]
    return ok_anchor and hmac.new(kk, msg, hashlib.sha256).digest()[:4] == mac


# ----------------------------------------------------------------------------
# Per-aircraft simulation
# ----------------------------------------------------------------------------
def next_true(mask):
    """For each index j, smallest j' >= j with mask[..., j'] True (else -1)."""
    n = mask.shape[-1]
    idx = np.where(mask, np.arange(n), n)
    nxt = np.minimum.accumulate(idx[..., ::-1], axis=-1)[..., ::-1]
    nxt = np.concatenate([nxt, np.full(nxt.shape[:-1] + (1,), n)], axis=-1)
    return np.where(nxt >= n, -1, nxt)


def simulate_aircraft(el, rng_km, rng, scheme="PO", N=None, T=None, d=None,
                      r=None, trace=False):
    N = P["N"] if N is None else N
    T = P["T_int"] if T is None else T
    d = P["d"] if d is None else d
    r = P["r"] if r is None else r
    dt = P["dt"]
    S, K = el.shape
    vis = el >= P["el_mask"]
    pnc = p_nc(N * (P["lam_ads"] + added_rate(scheme, r, T)))
    pe = np.where(vis, p_ppm(rng_km), 0.0) * pnc          # PPM, per squitter
    po = np.where(vis, p_po(rng_km), 0.0) * pnc           # PPM + overlay

    t = dt * np.arange(K)
    rx = rng.random((S, K)) < pe                           # position squitters
    spi = int(round(T / dt))                               # squitters/interval
    assert abs(spi * dt - T) < 1e-9, "T_int must be a multiple of dt"
    I = K // spi
    kr = spi * np.arange(I)                                # report index
    t_rep = t[kr]
    t_key = t_rep + T / 2
    per = pe[:, kr]; pov = po[:, kr]
    rep_rx = rx[:, kr]
    if scheme == "PO":
        cond = np.where(per > 0, pov / np.maximum(per, 1e-12), 0.0)
        rep_s = rep_rx & (rng.random((S, I)) < cond)        # report + MAC, same sat
        rep_any = rep_s.any(0)                               # fused
        key_s = rng.random((S, I)) < 1 - (1 - pov)**r
        key_any = key_s.any(0)
    else:
        mac_rx = rng.random((S, I)) < per
        rep_s = rep_rx & mac_rx
        rep_any = rep_rx.any(0) & mac_rx.any(0)             # report/MAC via any sat
        fr = rng.random((P["F_frag"], S, I)) < 1 - (1 - per[None])**r
        key_s = fr.all(0)
        key_any = fr.any(1).all(0)                          # fragments via any sat

    idx = np.arange(I)
    j0 = idx + d
    ok_j = j0 < I
    j0c = np.minimum(j0, I - 1)
    V = {}
    # IS-0: same satellite, exact key
    has = rep_s & key_s[:, j0c] & ok_j[None]
    V["IS-0"] = np.where(has.any(0), t_key[j0c], np.inf)
    # IS-R: same satellite, hash-chain recovery
    nx = next_true(key_s)                                  # (S, I+1)
    cand = np.where(rep_s & ok_j[None] & (nx[:, j0c] >= 0),
                    t_key[np.maximum(nx[:, j0c], 0)], np.inf)
    V["IS-R"] = cand.min(0)
    # NF: fused, exact key
    V["NF"] = np.where(rep_any & ok_j & key_any[j0c], t_key[j0c], np.inf)
    # CSAD: fused + cross-satellite recovery
    nf = next_true(key_any[None])[0]
    V["CSAD"] = np.where(rep_any & ok_j & (nf[j0c] >= 0),
                         t_key[np.maximum(nf[j0c], 0)], np.inf)

    # ages (evaluated on the squitter grid)
    rx_any = rx.any(0)
    last = np.where(rx_any, t, -np.inf)
    last = np.maximum.accumulate(last)
    A_rx = t - np.where(np.isfinite(last), last, 0.0)
    ages = {}
    for k_, v in V.items():
        ok = np.isfinite(v)
        if ok.any():
            o = np.argsort(v[ok])
            vs, ts = v[ok][o], np.maximum.accumulate(t_rep[ok][o])
            q = np.searchsorted(vs, t, side="right") - 1
            la = np.where(q >= 0, ts[np.maximum(q, 0)], 0.0)
        else:
            la = np.zeros_like(t)
        ages[k_] = t - la
    m = t >= P["warmup"]
    res = dict(D={k_: (v - t_rep)[ok_j] for k_, v in V.items()},
               A={k_: a[m] for k_, a in ages.items()}, A_rx=A_rx[m],
               nvis=vis.sum(0)[m])
    if trace:
        res.update(t=t, el=el, vis=vis, rep_s=rep_s, key_s=key_s, t_rep=t_rep,
                   t_key=t_key, V=V, ages=ages, A_rx_full=A_rx)
    return res


ARCH = ["IS-0", "IS-R", "NF", "CSAD"]
ALAB = {"IS-0": "Independent sat., no recovery",
        "IS-R": "Independent sat., recovery (TESLA)",
        "NF": "Naive ground fusion (exact key)",
        "CSAD": "CSAD (proposed)"}
ACOL = {"IS-0": "0.55", "IS-R": "C0", "NF": "C2", "CSAD": "C3"}
ALS = {"IS-0": ":", "IS-R": "--", "NF": "-.", "CSAD": "-"}


class Scenario:
    """Propagated constellation + random aircraft sample."""
    def __init__(self, n_planes, n_per_plane, n_ac=40, seed=3,
                 lat_range=(-60, 60), tle=None):
        rng = np.random.default_rng(seed)
        sats = load_tle(tle) if tle else walker_star(n_planes, n_per_plane)
        self.n_sats = len(sats)
        span = 3 * 3600.0
        self.t, self.sat = propagate_ecef(sats, span, P["dt"])
        K = int(P["window"] / P["dt"])
        self.geo = []
        lo, hi = np.sin(np.radians(lat_range[0])), np.sin(np.radians(lat_range[1]))
        for _ in range(n_ac):
            lat0 = np.arcsin(rng.uniform(lo, hi))
            lon0 = rng.uniform(-np.pi, np.pi)
            hdg = rng.uniform(0, 2 * np.pi)
            k0 = rng.integers(0, self.t.size - K)
            lat, lon, ac = aircraft_track(lat0, lon0, hdg, K, P["dt"])
            el, rk = look_angles(self.sat[:, k0:k0 + K], lat, lon, ac)
            keep = (el >= P["el_mask"]).any(1)
            self.geo.append((el[keep], rk[keep], np.degrees(lat0)))

    def run(self, seed=11, **kw):
        rng = np.random.default_rng(seed)
        out = [simulate_aircraft(e, rk, rng, **kw) for e, rk, _ in self.geo]
        D = {a: np.concatenate([o["D"][a] for o in out]) for a in ARCH}
        A = {a: np.concatenate([o["A"][a] for o in out]) for a in ARCH}
        Arx = np.concatenate([o["A_rx"] for o in out])
        nv = np.concatenate([o["nvis"] for o in out])
        return dict(
            D=D, A=A, nvis=float(nv.mean()),
            p_nvis=(lambda c: c / c.sum())(np.bincount(nv.astype(int))),
            dl={a: float(np.mean(D[a] <= P["Dmax"])) for a in ARCH},
            eta={a: float(np.mean(np.isfinite(D[a]))) for a in ARCH},
            p95={a: float(np.percentile(A[a], 95)) for a in ARCH},
            med={a: float(np.median(A[a])) for a in ARCH},
            p95_rx=float(np.percentile(Arx, 95)))


RES = {}


def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig1(res):
    """Link model + visibility diversity of the baseline constellation."""
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE[1],
                            gridspec_kw=dict(width_ratios=[1.35, 1]))
    el = np.linspace(5, 90, 200)
    a = 6378.135 + P["alt_sat"]; ra = 6371 + P["h_ac"]
    e = np.radians(el)
    rk = np.sqrt(a**2 - (ra * np.cos(e))**2) - ra * np.sin(e)
    ax = axs[0]
    ax.plot(el, p_ppm(rk), "k-", label="PPM")
    ax.plot(el, p_po(rk), "C4-", label="PPM + overlay")
    for N, ls in [(200, "--"), (400, ":")]:
        ax.plot(el, p_po(rk) * p_nc(N * P["lam_ads"]), color="C4", ls=ls,
                label=f"PPM + overlay, $N={N}$")
    ax.set_xlabel("Elevation (deg)"); ax.set_ylabel("Success probability")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right", fontsize=8.5)
    ax = axs[1]
    pv = np.asarray(res["p_nvis"])
    ax.bar(np.arange(pv.size), pv, color="C0")
    ax.set_xlabel("Visible satellites"); ax.set_ylabel("Fraction of time")
    ax.set_xticks(range(0, pv.size))
    fig.tight_layout(w_pad=0.8)
    savefig(fig, "fig1_link_and_visibility")
    RES["fig1"] = dict(snr_zen=float(mean_snr_dB(a - 6378.135 - P["h_ac"])),
                       pvis=np.asarray(res["p_nvis"]).tolist(), nvis=res["nvis"])


def fig2(sc, win=60.0):
    """Zoom on a handover: reports saved only by cross-satellite recovery."""
    rng = np.random.default_rng(5)
    best = None
    for k, (el, rk, _) in enumerate(sc.geo):
        out = simulate_aircraft(el, rk, rng, trace=True)
        V = out["V"]; tr = out["t_rep"]
        saved = (V["CSAD"] - tr <= P["Dmax"]) & ~(V["IS-R"] - tr <= P["Dmax"])
        n = int(win / P["T_int"])
        c = np.convolve(saved.astype(float), np.ones(n), "valid")
        j = int(np.argmax(c))
        if best is None or c[j] > best[0]:
            best = (c[j], k, out, tr[j], saved)
    _, k, out, t0, saved = best
    t = out["t"]; w = (t >= t0) & (t < t0 + win)
    tr, tk = out["t_rep"], out["t_key"]; wi = (tr >= t0) & (tr < t0 + win)
    shown = [s for s in range(out["el"].shape[0]) if (out["vis"][s][w]).any()]
    shown = sorted(shown, key=lambda s: -out["el"][s][w].mean())
    fig, axs = plt.subplots(3, 1, figsize=FIGSIZE[2], sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1], hspace=0.42))
    for c, s in enumerate(shown):
        axs[0].plot(t[w] - t0, out["el"][s][w], color=f"C{c}", label=f"Sat {chr(65 + c)}")
    axs[0].axhline(P["el_mask"], color="k", lw=0.6, ls=":")
    axs[0].set_ylabel("Elev. (deg)"); axs[0].legend(ncol=len(shown), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, frameon=False)
    for c, s in enumerate(shown):
        rr = out["rep_s"][s] & wi; kk = out["key_s"][s] & wi
        axs[1].plot(tr[rr] - t0, np.full(rr.sum(), c + 0.18), "|", color=f"C{c}", ms=5)
        axs[1].plot(tk[kk] - t0, np.full(kk.sum(), c - 0.18), "o", color=f"C{c}", ms=2)
    sv = saved & wi
    axs[1].plot(tr[sv] - t0, np.full(sv.sum(), len(shown) - 0.3), "v", color="C3", ms=3)
    axs[1].set_yticks(range(len(shown)))
    axs[1].set_yticklabels([f"Sat {chr(65 + c)}" for c in range(len(shown))])
    axs[1].set_ylim(-0.6, len(shown) + 0.1); axs[1].grid(False)
    axs[1].set_ylabel("Receptions")
    for a in ["IS-R", "CSAD"]:
        axs[2].step(t[w] - t0, out["ages"][a][w], where="post", color=ACOL[a],
                    ls=ALS[a], lw=1.0, label=ALAB[a])
    axs[2].set_ylabel("ASA (s)"); axs[2].set_xlabel("Time in window (s)")
    axs[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5, frameon=False)
    fig.align_ylabels(axs)
    savefig(fig, "fig2_handover_timeline")
    RES["fig2"] = dict(n_saved=int(sv.sum()), n_sats=len(shown),
                       max_isr=float(out["ages"]["IS-R"][w].max()),
                       max_csad=float(out["ages"]["CSAD"][w].max()))


def fig3(res):
    tg = np.linspace(0, 12, 481)
    fig, ax = plt.subplots(figsize=FIGSIZE[3])
    for a in ARCH:
        ax.plot(tg, [np.mean(res["D"][a] <= x) for x in tg], color=ACOL[a],
                ls=ALS[a], label=ALAB[a])
    ax.axvline(P["Dmax"], color="k", lw=0.6, ls=":")
    ax.set_xlabel("Authentication delay $D_{\\rm auth}$ (s)")
    ax.set_ylabel("$P(D_{\\rm auth}\\leq x)$"); ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    savefig(fig, "fig3_delay_cdf")
    RES["fig3"] = dict(eta=res["eta"], dl=res["dl"], p95=res["p95"],
                       med=res["med"], p95_rx=res["p95_rx"])


def fig4(consts):
    fig, ax = plt.subplots(figsize=FIGSIZE[4])
    xs = [c["nvis"] for c in consts]
    for a in ARCH:
        ax.plot(xs, [c["dl"][a] for c in consts], color=ACOL[a], ls=ALS[a],
                marker="o", ms=3, label=ALAB[a])
    for c in consts:
        ax.annotate(str(c["n_sats"]), (c["nvis"], c["dl"]["CSAD"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    ax.set_xlabel("Mean number of visible satellites")
    ax.set_ylabel(f"$P(D_{{\\rm auth}}\\leq {P['Dmax']:.0f}$ s$)$")
    ax.set_ylim(0, 1); ax.legend(loc="lower right")
    savefig(fig, "fig4_deadline_vs_diversity")
    RES["fig4"] = [dict(n_sats=c["n_sats"], nvis=c["nvis"], dl=c["dl"],
                        p95=c["p95"]) for c in consts]


def fig5(sc):
    """Cross-satellite diversity as a substitute for disclosure repetition."""
    Ns = [50, 100, 200, 300, 400]
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE[5], sharey=True)
    rec = {}
    for ax, scheme in zip(axs, ["PO", "ES"]):
        for a, r, mk in [("IS-R", 1, "o"), ("IS-R", 2, "s"), ("CSAD", 1, "o"), ("CSAD", 2, "s")]:
            ys = [sc.run(scheme=scheme, N=N, r=r)["dl"][a] for N in Ns]
            rec[f"{scheme}_{a}_r{r}"] = ys
            ax.plot(Ns, ys, color=ACOL[a], ls="-" if r == 1 else "--", marker=mk,
                    ms=2.8, mfc="none" if r == 2 else None,
                    label=f"{'CSAD' if a == 'CSAD' else 'Indep. sat.'}, $r={r}$")
        ax.set_title("Phase overlay" if scheme == "PO" else "Extra squitters", fontsize=10)
        ax.set_xlabel("Aircraft per beam $N$")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel(f"$P(D_{{\\rm auth}}\\leq {P['Dmax']:.0f}$ s$)$")
    axs[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(w_pad=0.5)
    savefig(fig, "fig5_diversity_vs_repetition")
    RES["fig5"] = dict(Ns=Ns, **rec)


def table_asa(sc):
    rows = {}
    for scheme in ["PO", "ES"]:
        for N in [100, 200, 400]:
            R = sc.run(scheme=scheme, N=N)
            rows[f"{scheme}_{N}"] = dict(p95=R["p95"], dl=R["dl"], eta=R["eta"],
                                         p95_rx=R["p95_rx"])
    RES["table2"] = rows


def fig6(sc):
    lats = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75)]
    rows = []
    for lo, hi in lats:
        s2 = Scenario(6, 11, n_ac=30, seed=21 + lo, lat_range=(lo, hi))
        r = s2.run()
        rows.append(dict(lat=(lo + hi) / 2, nvis=r["nvis"], dl=r["dl"], p95=r["p95"]))
    fig, ax = plt.subplots(figsize=FIGSIZE[6])
    x = [r_["lat"] for r_ in rows]
    for a in ["IS-R", "CSAD"]:
        ax.plot(x, [r_["dl"][a] for r_ in rows], color=ACOL[a], ls=ALS[a],
                marker="o", ms=3, label=ALAB[a])
    ax.set_xlabel("Aircraft latitude band centre (deg)")
    ax.set_ylabel(f"$P(D_{{\\rm auth}}\\leq {P['Dmax']:.0f}$ s$)$")
    ax2 = ax.twinx()
    ax2.bar(x, [r_["nvis"] for r_ in rows], width=6, color="C0", alpha=0.18)
    ax2.set_ylabel("Mean visible satellites", color="C0")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend(loc="upper left")
    savefig(fig, "fig6_latitude")
    RES["fig6"] = rows


def run_all(quick=False, seed=3, verbose=True):
    """Run every experiment and write figs/ + results."""
    assert crypto_check(), "hash-chain / HMAC self-test failed"
    n_ac = 12 if quick else 60
    base = Scenario(6, 11, n_ac=n_ac, seed=seed)      # Iridium-like 66 satellites
    r0 = base.run(seed=seed + 8)
    if verbose:
        print("baseline", {k: r0[k] for k in ["dl", "eta", "p95", "med", "p95_rx", "nvis"]})
    fig1(r0); fig2(base); fig3(r0)
    RES["baseline"] = {k: r0[k] for k in ["dl", "eta", "p95", "med", "p95_rx", "nvis"]}
    RES["baseline"]["p_nvis"] = r0["p_nvis"].tolist()

    consts = []
    for (pl, sp) in [(4, 6), (5, 8), (6, 11), (8, 11), (10, 12)]:
        sc = Scenario(pl, sp, n_ac=12 if quick else 40, seed=7)
        rr = sc.run(); rr["n_sats"] = pl * sp
        consts.append(rr)
        if verbose:
            print("const", pl * sp, round(rr["nvis"], 2), rr["dl"], rr["p95"])
    fig4(consts)
    fig5(base)
    table_asa(base)
    fig6(base)
    if verbose:
        print("fig5", RES["fig5"])
    json.dump(RES, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    export_source_data()
    return RES


def export_source_data():
    """Flatten the reported numbers into figs/source_data.csv (one row per metric)."""
    import csv
    rows = []

    def add(scope, metric, arch, value):
        rows.append(dict(scope=scope, metric=metric, architecture=arch, value=value))

    b = RES.get("baseline", {})
    for a in ARCH:
        add("baseline_PO_N200", "P(D_auth<=Dmax)", a, b["dl"][a])
        add("baseline_PO_N200", "P(verified_ever)", a, b["eta"][a])
        add("baseline_PO_N200", "ASA_p95_s", a, b["p95"][a])
        add("baseline_PO_N200", "ASA_median_s", a, b["med"][a])
    add("baseline_PO_N200", "received_age_p95_s", "all", b["p95_rx"])
    add("baseline_PO_N200", "mean_visible_satellites", "all", b["nvis"])
    for i, v in enumerate(b.get("p_nvis", [])):
        add("baseline_PO_N200", f"fraction_time_{i}_satellites", "all", v)
    for c in RES.get("fig4", []):
        for a in ARCH:
            add(f"constellation_{c['n_sats']}", "P(D_auth<=Dmax)", a, c["dl"][a])
            add(f"constellation_{c['n_sats']}", "ASA_p95_s", a, c["p95"][a])
        add(f"constellation_{c['n_sats']}", "mean_visible_satellites", "all", c["nvis"])
    f5 = RES.get("fig5", {})
    for N, vals in zip(f5.get("Ns", []), zip(*[f5[k] for k in f5 if k != "Ns"])):
        pass  # exported below in explicit form
    for scheme in ["PO", "ES"]:
        Ns = f5.get("Ns", [])
        for a in ["IS-R", "CSAD"]:
            for r in [1, 2]:
                ys = f5.get(f"{scheme}_{a}_r{r}", [])
                for N, y in zip(Ns, ys):
                    add(f"{scheme}_carriage", f"P(D_auth<=Dmax)_r{r}", a, y)
                    rows[-1]["metric"] = f"P(D_auth<=Dmax)_r{r}_N{N}"
    for key, v in RES.get("table2", {}).items():
        scheme, N = key.split("_")
        for a in ARCH:
            add(f"{scheme}_N{N}", "P(D_auth<=Dmax)", a, v["dl"][a])
            add(f"{scheme}_N{N}", "ASA_p95_s", a, v["p95"][a])
            add(f"{scheme}_N{N}", "P(verified_ever)", a, v["eta"][a])
        add(f"{scheme}_N{N}", "received_age_p95_s", "all", v["p95_rx"])
    for row in RES.get("fig6", []):
        for a in ["IS-R", "CSAD"]:
            add(f"latitude_{row['lat']:.1f}", "P(D_auth<=Dmax)", a, row["dl"][a])
            add(f"latitude_{row['lat']:.1f}", "ASA_p95_s", a, row["p95"][a])
        add(f"latitude_{row['lat']:.1f}", "mean_visible_satellites", "all", row["nvis"])
    for k in ("fig1", "fig2"):
        for kk, vv in RES.get(k, {}).items():
            if isinstance(vv, (int, float)):
                add(k, kk, "all", vv)
    with open(os.path.join(OUT, "source_data.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["scope", "metric", "architecture", "value"])
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", os.path.join(OUT, "source_data.csv"), len(rows), "rows")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    ap.add_argument("--quick", action="store_true", help="small sample (smoke test)")
    ap.add_argument("--seed", type=int, default=3, help="base random seed")
    ap.add_argument("--dpi", type=int, default=None, help="override figure dpi")
    args = ap.parse_args()
    if args.dpi:
        DPI = args.dpi
    run_all(quick=args.quick, seed=args.seed)
