#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Compare the WRF proof run against the 1D model at Akureyri.

Reads the two wrfout files (turbines off/on), extracts hub-height wind at
the Akureyri grid point, and reports the WRF marginal reduction — the
quantity the 1D model predicts via model.marginal_reduction. This closes
the validation loop once run_proof.sh has produced the output.

Requires netCDF4 or xarray (not in the base WRF image; run on the host).
"""
import sys
from pathlib import Path
import numpy as np

AK_LAT, AK_LON = 65.68, -18.09
HUB = 165.0


def hubwind(ncfile):
    import xarray as xr
    ds = xr.open_dataset(ncfile)
    lat, lon = ds['XLAT'][0].values, ds['XLONG'][0].values
    j, i = np.unravel_index(
        np.argmin((lat - AK_LAT) ** 2 + (lon - AK_LON) ** 2), lat.shape)
    # destagger U,V; approximate hub level by geopotential height
    U = 0.5 * (ds['U'][:, :, j, i] + ds['U'][:, :, j, i + 1]).values
    V = 0.5 * (ds['V'][:, :, j, i] + ds['V'][:, :, j + 1, i]).values
    ph = (ds['PH'][:, :, j, i] + ds['PHB'][:, :, j, i]).values / 9.81
    z = 0.5 * (ph[:, :-1] + ph[:, 1:]) - ph[:, 0:1]    # AGL mid-levels
    spd = np.sqrt(U ** 2 + V ** 2)
    out = [np.interp(HUB, z[t], spd[t]) for t in range(spd.shape[0])]
    ds.close()
    return float(np.mean(out))


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'wrfout_d03_baseline.nc'
    turb = sys.argv[2] if len(sys.argv) > 2 else 'wrfout_d03_turbines.nc'
    if not (Path(base).exists() and Path(turb).exists()):
        print('WRF output not found — run run_proof.sh first (needs the '
              'WPS_GEOG + met data and adequate compute; see README.md).')
        return
    ub, ut = hubwind(base), hubwind(turb)
    red = (1 - ut / ub) * 100
    pred = (1 - (ut / ub) ** 2) * 100
    print(f'  WRF Akureyri hub wind: baseline {ub:.2f} -> turbines {ut:.2f} '
          f'm/s')
    print(f'  WRF marginal reduction: {red:.1f}% speed, {pred:.1f}% pressure')
    print('  Compare with the 1D model.marginal_reduction() at the matching '
          'inflow and the §4.3 P10-P90 band (~3.5-16% pressure).')


if __name__ == '__main__':
    main()
