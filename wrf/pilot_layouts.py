#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate the three pilot turbine layouts for Eyjafjörður, snapped to the
d03 (1 km) grid at realistic 1 turbine / cell density (the audit fix —
no over-packed cells). All three are N=18 x Future-20 (360 MW), matched
capacity; only the along-fjord position differs:

  E (energy)  compact offshore array in the windy outer-mid fjord (x~13 km)
  S (shield)  channel-spanning wall in the narrow inner fjord (x~48 km,
              <=1 recovery length upwind of Akureyri at x=55)
  D (dual)    channel-spanning wall at mid-inner fjord (x~41 km)

Turbines occupy distinct WATER cells (LANDMASK==0) of the contiguous fjord
channel nearest the digitised centreline; rows are stacked along-fjord to
reach N. Writes wrf/windturbines_{E,S,D}.txt (lat lon type) + a summary.
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_rundeck import _AXIS                       # fjord centreline km->lat/lon

GEO = HERE / 'proof1km' / 'geo_em.d03.nc'
N_TURB = 18

# siting: (label, anchor, max_cols) where anchor is either a fjord x_km
# (axis-anchored; good for the narrow inner fjord) or an explicit
# (lat, lon) tuple (used for E, anchored to the modelled outer-fjord wind
# maximum since the digitised axis is offset where the fjord is widest).
SITINGS = [
    ('E', (65.99, -18.29), 5),   # outer fjord wind max (~28 m/s), ~36 km up
    ('D', 41, 5),                # mid-inner: ~7 km closest approach (~1 L)
    ('S', 48, 5),                # inner: ~2 km closest approach (<=1 L)
]


def axis_latlon(x_km):
    return (float(np.interp(x_km, _AXIS[:, 0], _AXIS[:, 1])),
            float(np.interp(x_km, _AXIS[:, 0], _AXIS[:, 2])))


def build(lat, lon, lm, label, anchor, max_cols):
    """Anchor either by fjord x_km (digitised centreline lat AND lon — the
    fjord, not the open ocean to its west) or by an explicit (lat, lon);
    take the N nearest WATER cells to that anchor (grid-index distance,
    max_cols wide). Guarantees 1 turbine / water cell, inside Eyjafjörður."""
    if isinstance(anchor, tuple):
        clat, clon = anchor
    else:
        clat, clon = axis_latlon(anchor)
    j0, i_c = np.unravel_index(
        np.argmin((lat - clat) ** 2 + (lon - clon) ** 2), lat.shape)
    # candidate water cells within a window, ranked by anisotropic grid
    # distance (cols weighted so the farm fills the channel width first)
    cand = []
    for j in range(max(0, j0 - 25), min(lat.shape[0], j0 + 25)):
        for i in range(lat.shape[1]):
            if lm[j, i] == 0:
                di, dj = i - i_c, j - j0
                if abs(di) <= max_cols // 2 + 1:
                    cand.append((dj * dj + (di * 1.3) ** 2, j, i))
    cand.sort()
    cells = [(j, i) for _, j, i in cand[:N_TURB]]
    latlons = [(float(lat[j, i]), float(lon[j, i])) for j, i in cells]
    clon = float(lon[j0, i_c])
    return clat, clon, cells, latlons


def main():
    ds = xr.open_dataset(GEO)
    lat = ds['XLAT_M'][0].values
    lon = ds['XLONG_M'][0].values
    lm = ds['LANDMASK'][0].values
    print(f'd03 grid {lat.shape}, town Akureyri x=55 km (65.68N,-18.09W)\n')
    for label, anchor, max_cols in SITINGS:
        clat, clon, cells, latlons = build(lat, lon, lm, label, anchor,
                                           max_cols)
        path = HERE / f'windturbines_{label}.txt'
        with open(path, 'w') as f:
            for la, lo in latlons:
                f.write(f'{la:.5f} {lo:.5f} 1\n')
        js = sorted({j for j, _ in cells})
        iis = sorted({i for _, i in cells})
        # all on water?
        onwater = all(lm[j, i] == 0 for j, i in cells)
        latspan = max(la for la, _ in latlons) - min(la for la, _ in latlons)
        tdist = 111 * np.sqrt((np.array([la for la, _ in latlons]) - 65.68)**2
                + ((np.array([lo for _, lo in latlons]) + 18.09)
                   * np.cos(np.radians(65.8)))**2).min()
        print(f'{label}  center {clat:.3f}N {clon:.3f}W  '
              f'-> {len(latlons)} turbines, {len(js)} rows x ~{len(iis)} cols, '
              f'all water: {onwater}, NS span {latspan*111:.1f} km, '
              f'closest to town {tdist:.1f} km')
    print('\nwrote windturbines_{E,S,D}.txt')


if __name__ == '__main__':
    main()
