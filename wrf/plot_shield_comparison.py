#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Shield-placement experiment figure: the SAME 18-turbine array, sited at
the outer fjord (SAMSETT generation cluster, km10-14) vs relocated to
the inner fjord (km42-46), both from WRF 1 km. Shows that the wake is
real and placeable — even larger in the narrow inner fjord — but the
short recovery length means it still decays before the fjord head
unless the array is sited within a few km of Akureyri.
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
    wps = carra.WAYPOINTS
    dp = []
    for w in wps:
        ub = wrf_hub(base_ds, w['lat'], w['lon'])
        ut = wrf_hub(turb_ds, w['lat'], w['lon'])
        dp.append((1 - (ut / ub) ** 2) * 100 if ub > 0 else 0.0)
    return np.array([w['x_km'] for w in wps]), np.array(dp)


def main():
    import xarray as xr
    base = xr.open_dataset(P / 'wrfout_d03_baseline.nc')
    outer = xr.open_dataset(P / 'wrfout_d03_turbines.nc')   # SAMSETT km10-14
    inner = xr.open_dataset(P / 'wrfout_d03_shield.nc')     # relocated km42-46
    door = xr.open_dataset(P / 'wrfout_d03_doorstep.nc')    # max near-head km44-49

    x, dp_outer = profile(outer, base)
    _, dp_inner = profile(inner, base)
    _, dp_door = profile(door, base)
    ak = AKUREYRI / 1000

    print(f'  {"station":<10}{"x_km":>5}{"outer":>9}{"inner":>9}{"doorstep":>10}')
    for w, a, b, c in zip(carra.WAYPOINTS, dp_outer, dp_inner, dp_door):
        print(f'  {w["name"]:<10}{w["x_km"]:>5}{a:>8.1f}%{b:>8.1f}%{c:>9.1f}%')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, dp_outer, 'o-', color='#2980b9', lw=2, ms=7,
            label='18 turbines, outer fjord (km10–14, generation)')
    ax.plot(x, dp_inner, 's-', color='#c0392b', lw=2, ms=7,
            label='18 turbines, inner fjord (km42–46, shield)')
    ax.plot(x, dp_door, 'D-', color='#8e44ad', lw=2, ms=6,
            label='8 turbines, max near-head (km44–49, all the fjord fits)')
    ax.axvspan(10, 14, color='#2980b9', alpha=0.12)
    ax.axvspan(42, 46, color='#c0392b', alpha=0.12)
    ax.axvline(ak, color='#27ae60', ls='--', alpha=0.7)
    ax.annotate('Akureyri', (ak - 1, ax.get_ylim()[1] * 0.7),
                rotation=90, fontsize=9, color='#27ae60', ha='right')
    ax.set_xlabel('Distance from fjord mouth (km)', fontsize=11)
    ax.set_ylabel('Turbine-induced pressure reduction (%)', fontsize=11)
    ax.set_title('Shield-placement test (WRF 1 km): relocating the same '
                 '18-turbine array', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9.5, loc='upper center')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 60)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'wrf_shield_placement.png', dpi=200, bbox_inches='tight')
    print(f'\n  Saved {OUT}/wrf_shield_placement.png')


if __name__ == '__main__':
    main()
