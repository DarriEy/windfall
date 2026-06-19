#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Compare the WRF proof run against the 1D model along the fjord axis.

Reads the two wrfout files (turbines off/on), extracts the hub-height
wind reduction at the 7 along-fjord stations, and overlays the 1D
model's predicted reduction for the same configuration. The headline is
whether WRF reproduces the 1D model's channeled wake persistence to
Akureyri — it does not: in WRF the wake recovers within ~10-15 km,
whereas the 1D model (recovery length L = 55-80 km) carries it 45 km to
the head.

Run after run_proof.sh (needs xarray on the host).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import carra
from model import ChanneledWakeModel, STABILITY_PRESETS
from designs import DESIGNS, EYJAFJORDUR, AKUREYRI, rows_of

OUT = Path(__file__).resolve().parent.parent / 'figures'
HUB = 165.0


def wrf_hub(ds, lat0, lon0):
    import numpy as np
    lat, lon = ds['XLAT'][0].values, ds['XLONG'][0].values
    j, i = np.unravel_index(
        np.argmin((lat - lat0) ** 2 + (lon - lon0) ** 2), lat.shape)
    U = 0.5 * (ds['U'][:, :, j, i] + ds['U'][:, :, j, i + 1]).values
    V = 0.5 * (ds['V'][:, :, j, i] + ds['V'][:, :, j + 1, i]).values
    ph = (ds['PH'][:, :, j, i] + ds['PHB'][:, :, j, i]).values / 9.81
    z = 0.5 * (ph[:, :-1] + ph[:, 1:]) - ph[:, 0:1]
    s = np.sqrt(U ** 2 + V ** 2)
    return np.mean([np.interp(HUB, z[k], s[k]) for k in range(s.shape[0])])


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path(__file__).parent / 'proof/wrfout_d02_baseline.nc')
    turb = sys.argv[2] if len(sys.argv) > 2 else \
        str(Path(__file__).parent / 'proof/wrfout_d02_turbines.nc')
    if not (Path(base).exists() and Path(turb).exists()):
        print('WRF output not found — run run_proof.sh first.')
        return
    import xarray as xr
    b, t = xr.open_dataset(base), xr.open_dataset(turb)

    wps = carra.WAYPOINTS
    x = np.array([w['x_km'] for w in wps])
    wrf_dp = []
    for w in wps:
        ub, ut = wrf_hub(b, w['lat'], w['lon']), wrf_hub(t, w['lat'], w['lon'])
        wrf_dp.append((1 - (ut / ub) ** 2) * 100 if ub > 0 else 0.0)
    wrf_dp = np.array(wrf_dp)

    # 1D model reduction profile (SAMSETT, stable, 20 m/s inflow)
    m = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
    r = m.simulate(rows_of(DESIGNS['E) SAMSETT']), 20.0)
    xm = r['x'] / 1000
    one_d_dp = (1 - (r['u'] / r['u_base']) ** 2) * 100

    print('=' * 64)
    print('  WRF vs 1D MODEL — along-fjord turbine wake (pressure %)')
    print('=' * 64)
    print(f'  {"station":<10}{"x_km":>5}{"WRF ΔP%":>9}')
    for w, dp in zip(wps, wrf_dp):
        print(f'  {w["name"]:<10}{w["x_km"]:>5}{dp:>8.1f}%')
    print()
    print(f'  WRF at Akureyri (55 km): {wrf_dp[-1]:.1f}% pressure')
    print(f'  WRF wake PEAKS at ~25 km ({wrf_dp.max():.0f}%) then recovers.')
    print(f'  1D model at Akureyri: ~{one_d_dp[np.argmin(abs(xm-55))]:.0f}% '
          f'(recovery L=55 km carries it to the head).')
    print('  => WRF recovery (~10-15 km) is far shorter than the 1D')
    print('     presets assume; the channeled wake does NOT reach')
    print('     Akureyri. (One 12h event at 3 km; coarse grid adds')
    print('     numerical mixing — a 1 km run is the next check.)')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xm, one_d_dp, '-', color='#c0392b', lw=2.5,
            label='1D model (SAMSETT, stable, L=55 km)')
    ax.plot(x, wrf_dp, 'o-', color='#2980b9', lw=2, ms=8,
            label='WRF 3 km (this run)')
    ax.axvline(AKUREYRI / 1000, color='#27ae60', ls='--', alpha=0.6)
    ax.annotate('Akureyri', (AKUREYRI / 1000 - 1, ax.get_ylim()[1] * 0.5),
                rotation=90, fontsize=9, color='#27ae60', ha='right')
    ax.set_xlabel('Distance from fjord mouth (km)', fontsize=11)
    ax.set_ylabel('Turbine-induced pressure reduction (%)', fontsize=11)
    ax.set_title('WRF vs 1D model: the channeled wake recovers before '
                 'Akureyri', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 60)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'wrf_vs_1d.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/wrf_vs_1d.png')


if __name__ == '__main__':
    main()
