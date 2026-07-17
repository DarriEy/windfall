#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Full RVK shielding-vs-energy analysis: the answer to (a) can a farm shield
the capital, and (b) what shield-siting costs — for Reykjavík.

SHIELDING  metro-box 10 m + hub wind-SPEED reduction downtown, per SW event
           (everyday / moderate / gale), weighted by SW-band frequency into
           an annual-mean reduction + the gale figure.
ENERGY     per-siting annual CF/AEP/LCOE. Spatial resource pattern from the
           WRF gale baseline hub-wind at the turbine cells; absolute annual
           level from the CARRA offshore Weibull (faxafloi, hub-extrapolated).
           Cost by foundation: nearshore (rvkS) cheaper than far-offshore
           (rvkE).
Deliverable: the energy/LCOE <-> shielding Pareto for RVK.

Robust to partial data; service+Pareto once all events present.
CARRA RVK is 2022-only -> annual figures are 2022-based (stated).
"""
import json
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')
from math import gamma

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyze_pilot import box_means, HUB
from designs import FUTURE20, CRF

PILOT = HERE / 'pilot'
DATA = REPO / 'data'
FIG = REPO / 'figures'
CITY = (64.13, -21.90)
R_KM = 2.5
SITINGS = ['rvkE', 'rvkD', 'rvkS']
GALE = 'rvk_swgale'
# foundation cost by siting: far-offshore deeper -> pricier; nearshore cheaper
CAPEX_KW = {'rvkE': 4000, 'rvkD': 3600, 'rvkS': 3200}
OPEX_KW = {'rvkE': 100, 'rvkD': 85, 'rvkS': 75}
SHEAR = 1.45            # CARRA 10 m -> 165 m hub (mean), per project note
WB_K = 2.0


def city_box(lat, lon):
    d = 111 * np.sqrt((lat - CITY[0]) ** 2
                      + ((lon - CITY[1]) * np.cos(np.radians(64.1))) ** 2)
    return d <= R_KM


def cf_from_weibull(A, k=WB_K):
    """Future-20 capacity factor for a Weibull(A,k) hub-speed distribution."""
    u = np.arange(0.5, 40, 0.5)
    pdf = (k / A) * (u / A) ** (k - 1) * np.exp(-(u / A) ** k)
    pdf /= pdf.sum()
    pw = FUTURE20.power(u)                       # MW per turbine
    return float((pw * pdf).sum() / FUTURE20.rated_power_mw * 100)


def hub_at_cells(ds, cells):
    """Mean hub speed over given (j,i) cells, back-half time mean."""
    nt = ds.sizes['Time']
    U, V, PH, PHB = (ds[k].values for k in ('U', 'V', 'PH', 'PHB'))
    vals = []
    for t in range(nt // 2, nt):
        for j, i in cells:
            u = 0.5 * (U[t, :, j, i] + U[t, :, j, i + 1])
            v = 0.5 * (V[t, :, j, i] + V[t, :, j + 1, i])
            z = (PH[t, :, j, i] + PHB[t, :, j, i]) / 9.81
            zc = 0.5 * (z[:-1] + z[1:]) - z[0]
            vals.append(np.interp(HUB, zc, np.sqrt(u ** 2 + v ** 2)))
    return float(np.mean(vals))


def cells_of(label, lat, lon):
    a = np.loadtxt(HERE / f'windturbines_{label}.txt')
    out = []
    for la, lo in a[:, :2]:
        j, i = np.unravel_index(
            np.argmin((lat - la) ** 2 + (lon - lo) ** 2), lat.shape)
        out.append((int(j), int(i)))
    return out


def energy():
    """Annual CF/AEP/LCOE per siting: WRF gale baseline gives the spatial
    resource ratio between sitings; CARRA offshore Weibull sets the level."""
    base = xr.open_dataset(PILOT / f'wrfout_d03_{GALE}_baseline.nc')
    lat, lon = base['XLAT'][0].values, base['XLONG'][0].values
    # annual offshore reference (faxafloi 10 m -> hub Weibull)
    fax = np.concatenate([np.load(f)['faxafloi_speed']
                          for f in sorted(DATA.glob('rvk_winds_2022_H*.npz'))])
    A_ref = fax.mean() * SHEAR / gamma(1 + 1 / WB_K)
    # WRF resource at faxafloi location (reference cell)
    jf, iff = np.unravel_index(
        np.argmin((lat - 64.25) ** 2 + (lon + 22.30) ** 2), lat.shape)
    u_ref = hub_at_cells(base, [(int(jf), int(iff))])
    out = {}
    for s in SITINGS:
        cells = cells_of(s, lat, lon)
        u_s = hub_at_cells(base, cells)
        A_s = A_ref * (u_s / u_ref)              # scale annual level by WRF ratio
        cf = cf_from_weibull(A_s)
        cap = 18 * FUTURE20.rated_power_mw
        aep = cf / 100 * cap * 8766 / 1000       # GWh
        annual = cap * 1000 * (CAPEX_KW[s] * CRF + OPEX_KW[s])
        lcoe = annual / (aep * 1000)
        out[s] = {'cf': cf, 'aep_gwh': aep, 'A_hub': A_s, 'lcoe': lcoe,
                  'capex_kw': CAPEX_KW[s]}
    return out


def shielding():
    """Per-event downtown 10 m + hub reduction for each siting."""
    ev = json.loads((HERE / 'pilot_events_rvk.json').read_text())['events']
    per = {}
    for e in ev:
        lbl = e['event_label']
        bp = PILOT / f'wrfout_d03_{lbl}_baseline.nc'
        if not bp.exists():
            continue
        base = xr.open_dataset(bp)
        lat, lon = base['XLAT'][0].values, base['XLONG'][0].values
        box = city_box(lat, lon)
        s10b, hubb = box_means(base, box)
        rec = {'band': e['band'], 'weight': e['weight'],
               'u10_base': s10b, 'hub_base': hubb, 'sitings': {}}
        for s in SITINGS:
            p = PILOT / f'wrfout_d03_{lbl}_{s}.nc'
            if not p.exists():
                continue
            ds = xr.open_dataset(p)
            s10, hub = box_means(ds, box)
            rec['sitings'][s] = {'red10': (1 - s10 / s10b) * 100,
                                 'redhub': (1 - hub / hubb) * 100,
                                 'factor10': s10 / s10b}
        per[e['band']] = rec
    return ev, per


def main():
    print('=== RVK ENERGY / COST (annual, 2022-based) ===')
    en = energy()
    print(f'  {"siting":<7}{"A_hub":>7}{"CF":>7}{"AEP":>9}{"capex":>8}{"LCOE":>8}')
    for s in SITINGS:
        e = en[s]
        print(f'  {s:<7}{e["A_hub"]:>6.1f} {e["cf"]:>5.1f}%{e["aep_gwh"]:>6.0f}GWh'
              f'{e["capex_kw"]:>6}/kW  ${e["lcoe"]:>4.0f}')

    print('\n=== RVK SHIELDING per SW event (downtown speed reduction) ===')
    ev, per = shielding()
    print(f'  {"band":<10}{"wt":>5}{"u10_b":>7}  ' +
          '  '.join(f'{s}(10m/hub)' for s in SITINGS))
    for e in ev:
        b = e['band']
        if b not in per:
            print(f'  {b:<10} (not run yet)')
            continue
        r = per[b]
        cells = '  '.join(
            f'{r["sitings"][s]["red10"]:+.1f}/{r["sitings"][s]["redhub"]:+.1f}%'
            if s in r['sitings'] else f'{s}:--' for s in SITINGS)
        print(f'  {b:<10}{r["weight"]:>5}{r["u10_base"]:>7.1f}  {cells}')

    have_all = all(e['band'] in per
                   and set(per[e['band']]['sitings']) >= set(SITINGS)
                   for e in ev)
    if not have_all:
        print('\n(annual service + Pareto deferred until all events x sitings)')
        return

    print('\n=== ANNUAL-WEIGHTED downtown shielding (SW regime) ===')
    ann = {}
    for s in SITINGS:
        r10 = sum(per[b]['weight'] * per[b]['sitings'][s]['red10'] for b in per)
        rhub = sum(per[b]['weight'] * per[b]['sitings'][s]['redhub'] for b in per)
        ann[s] = {'red10': r10, 'redhub': rhub,
                  'gale10': per['gale']['sitings'][s]['red10'],
                  'galehub': per['gale']['sitings'][s]['redhub']}
        print(f'  {s}: SW-mean {r10:+.1f}% (10m) / {rhub:+.1f}% (hub); '
              f'gale {ann[s]["gale10"]:+.1f}% / {ann[s]["galehub"]:+.1f}%')
    _pareto(en, ann)


def _pareto(en, ann):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    col = {'rvkE': '#1a9850', 'rvkD': '#fdae61', 'rvkS': '#d73027'}
    for s in SITINGS:
        x = ann[s]['galehub']
        y = en[s]['lcoe']
        ax.scatter(x, y, s=160, c=col[s], edgecolor='k', zorder=5)
        ax.annotate(f'{s}\nCF {en[s]["cf"]:.0f}%', (x, y),
                    xytext=(8, 6), textcoords='offset points')
    ax.set_xlabel('Gale shielding at city (hub speed reduction %)')
    ax.set_ylabel('LCOE ($/MWh)')
    ax.set_title('Reykjavík: energy cost vs SW-gale shielding\n'
                 '(18x Future-20, 1/cell, matched capacity)')
    plt.tight_layout()
    plt.savefig(FIG / 'pilot_pareto_rvk.png', dpi=120)
    print(f'\nwrote {FIG / "pilot_pareto_rvk.png"}')


if __name__ == '__main__':
    main()
