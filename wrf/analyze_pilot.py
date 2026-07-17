#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Analyse the Eyjafjörður shielding-vs-energy pilot.

For each siting (E energy / D dual / S shield), combine:
  SHIELDING  metro-averaged 10 m (and hub) wind-SPEED reduction at Akureyri
             (wall ON vs OFF), per event, then integrated over the 20-yr
             norðanátt climatology into an annual SERVICE (Δ hours/yr above
             15 and 20 m/s at the town).
  ENERGY     annual CF / AEP (model.aep + per-station CARRA Weibull, array
             wake included) and LCOE (foundation cost by siting).

Deliverable: the energy/LCOE <-> shielding-service Pareto frontier that
answers (a) can a farm do both, and (b) the cost of shielding-optimised
siting. Robust to partial data: reports whatever events are present;
service+Pareto only once all bands exist.
"""
import json
import sys
from pathlib import Path
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from make_rundeck import _AXIS
from model import TurbineRow, ChanneledWakeModel, WakeParams
import observed
from designs import FUTURE20, CRF

PILOT = HERE / 'pilot'
DATA = REPO / 'data'
FIG = REPO / 'figures'
AKUREYRI_LATLON = (65.68, -18.09)
METRO_R_KM = 2.5
HUB = 165.0
SITINGS = ['E', 'D', 'S']
# foundation cost basis by siting (consistent with designs.py SAMSETT):
# outer = deep floating, inner = shallow coastal.
CAPEX_KW = {'E': 4000, 'D': 3500, 'S': 2800}
OPEX_KW = {'E': 100, 'D': 85, 'S': 60}


# ── geometry ──────────────────────────────────────────────────────
def x_km_of_lat(lat):
    """Along-fjord km from latitude (invert the digitised centreline)."""
    return float(np.interp(lat, _AXIS[::-1, 1], _AXIS[::-1, 0]))


def metro_cells(lat, lon):
    d = 111 * np.sqrt((lat - AKUREYRI_LATLON[0]) ** 2
                      + ((lon - AKUREYRI_LATLON[1])
                         * np.cos(np.radians(65.7))) ** 2)
    return d <= METRO_R_KM


# ── wind at the metro box (quasi-steady back-half time mean) ──────
def box_means(ds, box):
    """Metro-box mean 10 m and hub-height speed, averaged over the
    back-half (quasi-steady) frames. Hub is computed only at box cells."""
    nt = ds.sizes['Time']
    js, iis = np.where(box)
    u10 = ds['U10'].values[nt // 2:]
    v10 = ds['V10'].values[nt // 2:]
    s10 = np.sqrt(u10 ** 2 + v10 ** 2)[:, js, iis].mean()
    U, V, PH, PHB = (ds[k].values for k in ('U', 'V', 'PH', 'PHB'))
    hub = []
    for t in range(nt // 2, nt):
        for j, i in zip(js, iis):
            u = 0.5 * (U[t, :, j, i] + U[t, :, j, i + 1])
            v = 0.5 * (V[t, :, j, i] + V[t, :, j + 1, i])
            z = (PH[t, :, j, i] + PHB[t, :, j, i]) / 9.81
            zc = 0.5 * (z[:-1] + z[1:]) - z[0]
            hub.append(np.interp(HUB, zc, np.sqrt(u ** 2 + v ** 2)))
    return float(s10), float(np.mean(hub))


# ── shielding per event ───────────────────────────────────────────
def event_shielding(event):
    bpath = PILOT / f'wrfout_d03_{event}_baseline.nc'
    if not bpath.exists():
        return None
    base = xr.open_dataset(bpath)
    lat, lon = base['XLAT'][0].values, base['XLONG'][0].values
    box = metro_cells(lat, lon)
    s10b, hubb = box_means(base, box)
    out = {'u10_base': float(s10b), 'hub_base': float(hubb), 'sitings': {}}
    for s in SITINGS:
        p = PILOT / f'wrfout_d03_{event}_{s}.nc'
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        s10, hub = box_means(ds, box)
        out['sitings'][s] = {
            'u10': float(s10),
            'hub': float(hub),
            'red10_pct': float((1 - s10 / s10b) * 100),
            'redhub_pct': float((1 - hub / hubb) * 100),
            'factor10': float(s10 / s10b),       # for service integration
        }
    return out


# ── energy + cost ─────────────────────────────────────────────────
def rows_from_layout(label):
    a = np.loadtxt(HERE / f'windturbines_{label}.txt')
    xs = np.array([x_km_of_lat(la) for la in a[:, 0]])
    rows = []
    for xb in np.unique(np.round(xs)):
        n = int((np.round(xs) == xb).sum())
        rows.append(TurbineRow(xb * 1000, n, FUTURE20))
    return rows


def energy_cost(label):
    rows = rows_from_layout(label)
    m = ChanneledWakeModel(__import__('designs').EYJAFJORDUR,
                           WakeParams(effective_height=200,
                                      recovery_length=30_000,
                                      channeling_fraction=0.7))
    aep = m.aep(rows, station_weibull=observed.weibull_callable())
    cap_mw = sum(r.capacity_mw for r in rows)
    annual = cap_mw * 1000 * (CAPEX_KW[label] * CRF + OPEX_KW[label])
    lcoe = annual / (aep['aep_gwh'] * 1000) if aep['aep_gwh'] > 0 else 9999
    return {'cap_mw': cap_mw, 'aep_gwh': aep['aep_gwh'],
            'cf_pct': aep['cf_pct'], 'capex_kw': CAPEX_KW[label],
            'lcoe': lcoe}


# ── annual shielding service (needs all bands) ────────────────────
def annual_service(per_event, events):
    """Apply each band's 10 m reduction factor to the 20-yr CARRA Akureyri
    norðanátt record (binned by mouth speed) -> Δ hours/yr > 15 & 20 m/s."""
    rec = []
    import glob
    import re
    import os
    for f in sorted(glob.glob(str(DATA / 'winds_*.npz'))):
        z = np.load(f)
        rec.append(np.column_stack([z['mouth_speed'], z['mouth_dir'],
                                    z['akureyri_speed']]))
    rec = np.vstack(rec)
    ms, md, aks = rec[:, 0], rec[:, 1], rec[:, 2]
    nor = ((md >= 330) | (md <= 30)) & (ms >= 10)
    msn, aksn = ms[nor], aks[nor]
    # band edges by mouth-speed quantile (match pilot_events BANDS order)
    edges = [0, 0.60, 0.80, 0.93, 0.985, 1.0]
    qv = [np.quantile(msn, e) for e in edges]
    bands = [e['band'] for e in events]
    factors = {s: np.ones(len(msn)) for s in SITINGS}
    for bi, ev in enumerate(events):
        lo, hi = qv[bi], qv[bi + 1]
        sel = (msn >= lo) & (msn <= hi) if bi == len(events) - 1 \
            else (msn >= lo) & (msn < hi)
        es = per_event.get(ev['event_label'], {}).get('sitings', {})
        for s in SITINGS:
            if s in es:
                factors[s][sel] = es[s]['factor10']
    hrs = 6.0 / 20                                   # per record -> h/yr
    out = {}
    for thr in (15, 20):
        base_h = float((aksn > thr).sum() * hrs)
        out[f'base_hrs_gt{thr}'] = base_h
        for s in SITINGS:
            farm_h = float(((aksn * factors[s]) > thr).sum() * hrs)
            out.setdefault(s, {})[f'dhrs_gt{thr}'] = base_h - farm_h
    return out


def main():
    ev_meta = json.loads((HERE / 'pilot_events.json').read_text())['events']

    print('=== ENERGY / COST (event-independent) ===')
    energy = {}
    print(f'  {"siting":<7}{"cap":>7}{"AEP":>9}{"CF":>7}{"capex":>8}{"LCOE":>8}')
    for s in SITINGS:
        e = energy_cost(s)
        energy[s] = e
        print(f'  {s:<7}{e["cap_mw"]:>4.0f}MW{e["aep_gwh"]:>6.0f}GWh'
              f'{e["cf_pct"]:>6.1f}%{e["capex_kw"]:>6}/kW  ${e["lcoe"]:>5.0f}')

    print('\n=== SHIELDING per event (metro 10 m speed reduction %) ===')
    per_event = {}
    for e in ev_meta:
        lbl = e['event_label']
        sh = event_shielding(lbl)
        if sh is None:
            print(f'  {e["band"]:<13} (not run yet: {lbl})')
            continue
        per_event[lbl] = sh
        cells = '  '.join(
            f'{s}:{sh["sitings"][s]["red10_pct"]:+.1f}%'
            for s in SITINGS if s in sh['sitings'])
        print(f'  {e["band"]:<13} u10_base={sh["u10_base"]:.1f}  {cells}')

    have_all = all(e['event_label'] in per_event
                   and set(per_event[e['event_label']]['sitings']) >= set(SITINGS)
                   for e in ev_meta)
    if not have_all:
        print('\n(service + Pareto deferred until all bands x sitings present)')
        return

    print('\n=== ANNUAL SHIELDING SERVICE (Δ hours/yr at Akureyri) ===')
    svc = annual_service(per_event, ev_meta)
    for thr in (15, 20):
        print(f'  baseline hours > {thr} m/s: {svc[f"base_hrs_gt{thr}"]:.0f} h/yr')
        for s in SITINGS:
            print(f'    {s}: avoided {svc[s][f"dhrs_gt{thr}"]:.0f} h/yr')

    _pareto(energy, svc)


def _pareto(energy, svc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5))
    col = {'E': '#1a9850', 'D': '#fdae61', 'S': '#d73027'}
    for s in SITINGS:
        x = svc[s]['dhrs_gt15']
        y = energy[s]['lcoe']
        ax.scatter(x, y, s=140, c=col[s], edgecolor='k', zorder=5)
        ax.annotate(f'{s}\n{energy[s]["cf_pct"]:.0f}% CF',
                    (x, y), xytext=(8, 6), textcoords='offset points')
    ax.set_xlabel('Annual shielding service  (Δ hours/yr > 15 m/s at Akureyri)')
    ax.set_ylabel('LCOE ($/MWh)')
    ax.set_title('Eyjafjörður: energy cost vs shielding service\n'
                 '(18× Future-20, matched capacity, 1/cell)')
    plt.tight_layout()
    plt.savefig(FIG / 'pilot_pareto.png', dpi=120)
    print(f'\nwrote {FIG / "pilot_pareto.png"}')


if __name__ == '__main__':
    main()
