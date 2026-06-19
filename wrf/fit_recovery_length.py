#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Fit the wake RECOVERY LENGTH L from the WRF 1 km runs, to replace the
1D model's assumed L = 55-80 km (which the mesoscale runs show is far
too long). We sample the hub-height speed deficit along the fjord as the
cross-fjord wake-core (max deficit among water cells in each along-fjord
km bin), then fit  delta(x) = delta0 * exp(-(x - x_peak) / L)  to the
decay downstream of the peak.

The outer-cluster run (turbines at km10-14) has ~40 km of open fjord
downstream and is the primary fit; the inner-cluster run is a check.
Writes data/wrf_recovery_calibration.json.
"""
import json
import importlib.util
from pathlib import Path
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
P = HERE / 'proof1km'
DATA = HERE.parent / 'data'
HUB = 165.0

_spec = importlib.util.spec_from_file_location('mr', HERE / 'make_rundeck.py')
mr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mr)
AX = mr._AXIS


def hub_field(ds):
    """Hub-height wind speed on the d03 grid, time-averaged."""
    U = 0.5 * (ds['U'][:, :, :, :-1] + ds['U'][:, :, :, 1:]).values
    V = 0.5 * (ds['V'][:, :, :-1, :] + ds['V'][:, :, 1:, :]).values
    ph = (ds['PH'] + ds['PHB']).values / 9.81
    z = 0.5 * (ph[:, :-1] + ph[:, 1:]) - ph[:, 0:1, :, :]
    s = np.sqrt(U ** 2 + V ** 2)
    nt, _, ny, nx = s.shape
    out = np.empty((nt, ny, nx))
    for t in range(nt):
        for j in range(ny):
            for i in range(nx):
                out[t, j, i] = np.interp(HUB, z[t, :, j, i], s[t, :, j, i])
    return out.mean(0)


def core_decay(base, turb):
    """Speed-deficit wake-core vs along-fjord distance (water cells)."""
    lm = base['LANDMASK'][0].values
    lat = base['XLAT'][0].values
    lon = base['XLONG'][0].values
    ub, ut = hub_field(base), hub_field(turb)
    deficit = 1.0 - ut / ub
    axkm = np.linspace(0, 55, 560)
    axla = np.interp(axkm, AX[:, 0], AX[:, 1])
    axlo = np.interp(axkm, AX[:, 0], AX[:, 2])
    bins = {}
    ny, nx = lm.shape
    for j in range(ny):
        for i in range(nx):
            if lm[j, i] != 0:
                continue
            d2 = (axla - lat[j, i]) ** 2 + (axlo - lon[j, i]) ** 2
            k = int(np.argmin(d2))
            if np.sqrt(d2[k]) * 111 > 2.5:
                continue
            kmbin = int(round(axkm[k]))
            bins.setdefault(kmbin, []).append(deficit[j, i])
    xs = np.array(sorted(bins))
    core = np.array([max(bins[k]) for k in xs])   # wake core = max deficit
    return xs, core


def fit_L(xs, core):
    ip = int(np.argmax(core))
    x0, d0 = xs[ip], core[ip]
    xd, dd = xs[ip:], core[ip:]
    m = dd > 0.002                                 # ignore noise floor
    if m.sum() < 3:
        return x0, d0, None, None
    # linear fit of ln(deficit) vs distance -> slope = -1/L
    A = np.vstack([xd[m] - x0, np.ones(m.sum())]).T
    slope, _ = np.linalg.lstsq(A, np.log(dd[m]), rcond=None)[0]
    L = -1.0 / slope if slope < 0 else None
    pred = d0 * np.exp((xd - x0) * slope)
    ss = 1 - np.sum((dd - pred) ** 2) / np.sum((dd - dd.mean()) ** 2)
    return x0, d0, L, ss


def main():
    runs = {'outer': 'wrfout_d03_turbines.nc', 'inner': 'wrfout_d03_shield.nc'}
    base = xr.open_dataset(P / 'wrfout_d03_baseline.nc')
    res = {}
    print(f'  {"run":<8}{"x_peak":>8}{"deficit0":>10}{"L (km)":>9}{"R2":>7}')
    for name, fn in runs.items():
        if not (P / fn).exists():
            continue
        xs, core = core_decay(base, xr.open_dataset(P / fn))
        x0, d0, L, ss = fit_L(xs, core)
        res[name] = {'x_peak_km': float(x0), 'deficit0': float(d0),
                     'L_km': None if L is None else float(L),
                     'r2': None if ss is None else float(ss),
                     'profile_km': xs.tolist(),
                     'core_deficit': [float(c) for c in core]}
        Ls = f'{L:.1f}' if L else '—'
        r2 = f'{ss:.2f}' if ss is not None else '—'
        print(f'  {name:<8}{x0:>8.0f}{d0*100:>9.1f}%{Ls:>9}{r2:>7}')
    DATA.mkdir(exist_ok=True)
    (DATA / 'wrf_recovery_calibration.json').write_text(json.dumps(res, indent=2))
    Lvals = [r['L_km'] for r in res.values() if r['L_km']]
    if Lvals:
        print(f'\n  WRF-derived recovery length L = {np.mean(Lvals):.1f} km '
              f'(was 55-80 km in the 1D presets).')
    print(f'  Saved {DATA}/wrf_recovery_calibration.json')


if __name__ == '__main__':
    main()
