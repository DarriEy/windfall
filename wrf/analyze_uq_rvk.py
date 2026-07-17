#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
UQ on the RVK gale shield: downtown rvkS reduction (10 m + hub) under
three model configs — MYNN (main), YSU PBL, and vertical refinement
(e_vert=50) — each vs its OWN baseline. Brackets the ~10% hub number.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_pilot import box_means

PILOT = HERE / 'pilot'
CITY = (64.13, -21.90)
CONFIGS = [('MYNN (main)', 'rvk_swgale'),
           ('YSU PBL', 'rvk_gale_ysu'),
           ('e_vert=50', 'rvk_gale_vert')]


def main():
    print('RVK gale — rvkS downtown shielding under model perturbations\n')
    print(f'  {"config":<14}{"base10":>8}{"red10":>8}{"basehub":>9}{"redhub":>8}')
    r10s, rhubs = [], []
    for name, ev in CONFIGS:
        bp = PILOT / f'wrfout_d03_{ev}_baseline.nc'
        sp = PILOT / f'wrfout_d03_{ev}_rvkS.nc'
        if not (bp.exists() and sp.exists()):
            print(f'  {name:<14} (not run yet: {ev})')
            continue
        base = xr.open_dataset(bp)
        lat, lon = base['XLAT'][0].values, base['XLONG'][0].values
        box = (111 * np.sqrt((lat - CITY[0]) ** 2
               + ((lon - CITY[1]) * np.cos(np.radians(64.1))) ** 2)) <= 2.5
        s10b, hubb = box_means(base, box)
        s10, hub = box_means(xr.open_dataset(sp), box)
        r10 = (1 - s10 / s10b) * 100
        rhub = (1 - hub / hubb) * 100
        r10s.append(r10); rhubs.append(rhub)
        print(f'  {name:<14}{s10b:>8.2f}{r10:>+7.1f}%{hubb:>9.2f}{rhub:>+7.1f}%')
    if len(rhubs) > 1:
        print(f'\n  hub shielding spread: {min(rhubs):.1f}–{max(rhubs):.1f}% '
              f'(mean {np.mean(rhubs):.1f}%, ±{(max(rhubs)-min(rhubs))/2:.1f})')
        print(f'  10m shielding spread: {min(r10s):.1f}–{max(r10s):.1f}% '
              f'(mean {np.mean(r10s):.1f}%)')


if __name__ == '__main__':
    main()
