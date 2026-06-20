#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Can a turbine wall shield Akureyri? Summary of the WRF 1 km placement
experiments at the fjord head, with onshore valley-floor siting allowed.
Shows along-fjord pressure-reduction profiles for three head walls of
increasing density, and that only an industrial-scale DENSE wall right at
the town (a barrier, not a generator) delivers a meaningful shield.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import carra
from designs import AKUREYRI

HUB = 165.0
P = Path(__file__).resolve().parent / 'proof1km'
OUT = Path(__file__).resolve().parent.parent / 'figures'


def wrf_hub(ds, lat0, lon0):
    lat, lon = ds['XLAT'][0].values, ds['XLONG'][0].values
    j, i = np.unravel_index(
        np.argmin((lat - lat0) ** 2 + (lon - lon0) ** 2), lat.shape)
    U = 0.5 * (ds['U'][:, :, j, i] + ds['U'][:, :, j, i + 1]).values
    V = 0.5 * (ds['V'][:, :, j, i] + ds['V'][:, :, j + 1, i]).values
    ph = (ds['PH'][:, :, j, i] + ds['PHB'][:, :, j, i]).values / 9.81
    z = 0.5 * (ph[:, :-1] + ph[:, 1:]) - ph[:, 0:1]
    s = np.sqrt(U ** 2 + V ** 2)
    return np.mean([np.interp(HUB, z[k], s[k]) for k in range(s.shape[0])])


def profile(turb_ds, base_ds):
    dp = []
    for w in carra.WAYPOINTS:
        ub = wrf_hub(base_ds, w['lat'], w['lon'])
        ut = wrf_hub(turb_ds, w['lat'], w['lon'])
        dp.append((1 - (ut / ub) ** 2) * 100 if ub > 0 else 0.0)
    return np.array([w['x_km'] for w in carra.WAYPOINTS]), np.array(dp)


def main():
    import xarray as xr
    base = xr.open_dataset(P / 'wrfout_d03_baseline.nc')
    runs = [
        ('wrfout_d03_doorstep.nc',  '8 turbines, water only (160 MW)',  '#27ae60', 'o'),
        ('wrfout_d03_valleywall.nc', '22 turbines, water+onshore floor (440 MW)', '#e67e22', 's'),
        ('wrfout_d03_densewall.nc',  '66 turbines, dense onshore wall (1.3 GW)', '#c0392b', 'D'),
    ]
    ak = AKUREYRI / 1000
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fn, lbl, c, mk in runs:
        if not (P / fn).exists():
            continue
        x, dp = profile(xr.open_dataset(P / fn), base)
        ax.plot(x, dp, mk + '-', color=c, lw=2, ms=7,
                label=f'{lbl} — {dp[-1]:.1f}% at Akureyri')
    ax.axvspan(46, 53, color='grey', alpha=0.12, label='head wall (km46–53)')
    ax.axvline(ak, color='#2c3e50', ls='--', alpha=0.7)
    ax.annotate('Akureyri', (ak - 1, ax.get_ylim()[1] * 0.8),
                rotation=90, fontsize=9, color='#2c3e50', ha='right')
    ax.set_xlabel('Distance from fjord mouth (km)', fontsize=11)
    ax.set_ylabel('Turbine-induced pressure reduction (%)', fontsize=11)
    ax.set_title('Shielding Akureyri: only a dense onshore wall at the head '
                 'delivers it (WRF 1 km)', fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 60)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'wrf_akureyri_shield.png', dpi=200, bbox_inches='tight')
    print(f'Saved {OUT}/wrf_akureyri_shield.png')


if __name__ == '__main__':
    main()
