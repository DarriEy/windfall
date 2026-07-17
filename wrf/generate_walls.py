#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Clever-design RVK shield variants, selected from the actual upwind water
cells (Faxaflói SW/W of the city), 1/cell (1 km ~ 3.6 D, the Fitch floor).
Vary GEOMETRY and SIZE, not spacing:

  rvkWide  18 turbines spread for MAX cross-wind width (a wall perpendicular
           to the SW gale) at ~mid upwind distance — vs rvkS's compact blob.
  rvkBig   36 turbines = the nearest 36 upwind water cells (double capacity,
           fills the available water) — does more thrust buy more shielding?

Cells projected onto along-wind (s, toward city) / cross-wind (t) axes
relative to downtown; SW gale from 225 deg.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
GEO = HERE / 'rvk' / 'geo_em.d03.nc'
CITY = (64.13, -21.90)
COS = np.cos(np.radians(64.1))
GALE_FROM = 225.0


def main():
    ds = xr.open_dataset(GEO)
    lat = ds['XLAT_M'][0].values
    lon = ds['XLONG_M'][0].values
    lm = ds['LANDMASK'][0].values
    dn = (lat - CITY[0]) * 111.0
    de = (lon - CITY[1]) * 111.0 * COS
    dist = np.sqrt(dn ** 2 + de ** 2)
    brg = (np.degrees(np.arctan2(de, dn)) + 360) % 360
    # upwind water cells (SW-W arc, 2-11 km from city)
    sel = (lm == 0) & (dist >= 2) & (dist <= 11) & (brg >= 175) & (brg <= 285)
    js, iis = np.where(sel)
    # along-wind s (positive toward city, i.e. along NE = 45 deg) & cross t
    a = np.radians(GALE_FROM - 180)              # 45 -> downwind toward city
    sn, se = np.cos(a), np.sin(a)
    cn, ce = np.cos(np.radians(GALE_FROM - 90)), np.sin(np.radians(GALE_FROM - 90))
    cells = []
    for j, i in zip(js, iis):
        # upwind distance (positive = SW of city, into the gale)
        u = -(dn[j, i] * sn + de[j, i] * se)
        t = dn[j, i] * cn + de[j, i] * ce        # cross-wind
        cells.append((j, i, u, t, dist[j, i]))

    def write(label, picks):
        with open(HERE / f'windturbines_{label}.txt', 'w') as f:
            for j, i in picks:
                f.write(f'{lat[j, i]:.5f} {lon[j, i]:.5f} 1\n')
        cd = min(dist[j, i] for j, i in picks)
        tt = [c[3] for c in cells if (c[0], c[1]) in picks]
        print(f'{label}: {len(picks)} turbines, closest {cd:.1f} km, '
              f'cross-wind span {max(tt)-min(tt):.1f} km')

    # rvkBig: nearest 36 to the city (fills the water, max thrust near town)
    big = [(c[0], c[1]) for c in sorted(cells, key=lambda c: c[4])[:36]]
    write('rvkBig', big)

    # rvkWide: a wall — cells in a mid upwind band (s in 3-7 km), widest t
    band = [c for c in cells if 3 <= c[2] <= 7]
    band.sort(key=lambda c: c[3])                # by cross-wind position
    # take up to 18 spread across the band's width
    if len(band) > 18:
        idx = np.linspace(0, len(band) - 1, 18).round().astype(int)
        band = [band[k] for k in idx]
    write('rvkWide', [(c[0], c[1]) for c in band])


if __name__ == '__main__':
    main()
