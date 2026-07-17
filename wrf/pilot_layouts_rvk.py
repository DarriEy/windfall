#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Reykjavík (Faxaflói) pilot layouts — the SW-gale shielding test, mirroring
the Eyjafjörður pilot at honest 1 turbine/cell density. Matched 18x
Future-20 (360 MW); only the position changes:

  rvkE (energy)  windiest open Faxaflói, far SW of the city
  rvkS (shield)  wall in the water SW (upwind) of downtown, close in —
                 the same Faxaflói approach as the prior BREIDVEGGUR wall
                 (where 41% was claimed at 3/cell); does it hold at 1/cell?
  rvkD (dual)    between

SW gale arrives from ~225°, so upwind of the coastal city = SW over the
open sea. Reuses build() from pilot_layouts (nearest-N water cells).
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_layouts import build

GEO = HERE / 'rvk' / 'geo_em.d03.nc'
CITY = (64.13, -21.90)
# (label, (lat,lon) anchor, max_cols)
SITINGS = [
    ('rvkE', (63.95, -22.35), 6),   # far open Faxaflói (windiest), ~30 km SW
    ('rvkD', (64.04, -22.05), 5),   # mid approach
    ('rvkS', (64.08, -21.99), 5),   # close SW of city (~8 km upwind)
]


def main():
    ds = xr.open_dataset(GEO)
    lat = ds['XLAT_M'][0].values
    lon = ds['XLONG_M'][0].values
    lm = ds['LANDMASK'][0].values
    print(f'RVK d03 {lat.shape}, city (64.13,-21.90); SW gale from ~225°\n')
    for label, anchor, max_cols in SITINGS:
        clat, clon, cells, latlons = build(lat, lon, lm, label, anchor,
                                           max_cols)
        path = HERE / f'windturbines_{label}.txt'
        with open(path, 'w') as f:
            for la, lo in latlons:
                f.write(f'{la:.5f} {lo:.5f} 1\n')
        onwater = all(lm[j, i] == 0 for j, i in cells)
        tdist = 111 * np.sqrt(
            (np.array([la for la, _ in latlons]) - CITY[0]) ** 2
            + ((np.array([lo for _, lo in latlons]) - CITY[1])
               * np.cos(np.radians(64.1))) ** 2).min()
        print(f'{label}  center {clat:.3f}N {clon:.3f}W -> {len(latlons)} '
              f'turbines, all water: {onwater}, closest to city {tdist:.1f} km')
    print('\nwrote windturbines_rvk{E,D,S}.txt')


if __name__ == '__main__':
    main()
