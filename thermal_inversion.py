#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Thermal inversion analysis from CARRA temperature data.

Computes the frequency of temperature inversions (T_2m > T_skin)
during norðanátt events, providing empirical confirmation that
stable stratification accompanies the storms we want to shield against.

Usage:
    python thermal_inversion.py                    # process available data
    python thermal_inversion.py --fetch --year 2022  # download + process
"""

import argparse
import numpy as np
import math
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import carra

OUT = Path('figures')
DATA_DIR = Path('data')

# We need co-located wind AND temperature at the same points.
# Temperature extraction uses the same waypoints as wind.


def extract_temperature(nc_file):
    """Extract 2m and skin temperature at fjord waypoints."""
    import xarray as xr

    try:
        ds = xr.open_dataset(nc_file)
    except OSError:
        ds = xr.open_dataset(nc_file, engine='cfgrib',
                             backend_kwargs={'indexpath': ''})

    # Find temperature variables
    t2m_var = skin_var = lat_var = lon_var = None
    for name in ds.data_vars:
        nl = name.lower()
        if '2m' in nl and 'temp' in nl:
            t2m_var = name
        elif ('skin' in nl or 'skt' in nl) and 'temp' in nl:
            skin_var = name
        elif nl in ('t2m',):
            t2m_var = name
        elif nl in ('skt', 'stl1'):
            skin_var = name

    if t2m_var is None:
        for name in ds.data_vars:
            if name.lower().startswith('t2'):
                t2m_var = name
                break

    for name in list(ds.coords) + list(ds.data_vars):
        nl = name.lower()
        if lat_var is None and 'lat' in nl:
            lat_var = name
        if lon_var is None and 'lon' in nl:
            lon_var = name

    if t2m_var is None:
        raise ValueError(f'Cannot find 2m temperature in {list(ds.data_vars)}')

    print(f'    Temp vars: t2m={t2m_var}, skin={skin_var}')

    lat_grid = ds[lat_var].values
    lon_grid = ds[lon_var].values % 360

    spatial_dims = [d for d in ds[t2m_var].dims
                    if d not in carra.TIME_DIMS]

    result = {}
    for wp in carra.WAYPOINTS:
        tlon = wp['lon'] % 360
        i, j = carra._find_nearest(lat_grid, lon_grid, wp['lat'], tlon)
        sel = {spatial_dims[0]: i, spatial_dims[1]: j}

        t2m = ds[t2m_var].isel(**sel).values.flatten()
        if skin_var:
            skt = ds[skin_var].isel(**sel).values.flatten()
        else:
            skt = np.full_like(t2m, np.nan)

        ok = np.isfinite(t2m)
        result[wp['name']] = {
            't2m': t2m[ok],
            'skt': skt[ok] if skin_var else None,
            'delta_t': (t2m[ok] - skt[ok]) if skin_var else None,
        }

    ds.close()
    return result


def analyze_inversions():
    """Combine wind and temperature data to analyze inversions."""
    # Load wind data
    wind_files = sorted(DATA_DIR.glob('winds_*.npz'))
    if not wind_files:
        print('  No wind data available')
        return

    # Check for temperature files
    temp_files = sorted(DATA_DIR.glob('carra_temp_*.nc')) + \
                 sorted(DATA_DIR.glob('_test_temp.nc'))
    if not temp_files:
        print('  No temperature data yet.')
        print('  Run: python thermal_inversion.py --fetch --year 2022')
        print('  Or wait for background CARRA temp download.')
        return

    print(f'  Found {len(temp_files)} temperature file(s)')

    all_delta_t = {wp['name']: [] for wp in carra.WAYPOINTS}

    for tf in temp_files:
        print(f'  Processing {tf.name}...')
        try:
            temps = extract_temperature(tf)
            for wp in carra.WAYPOINTS:
                if temps[wp['name']]['delta_t'] is not None:
                    all_delta_t[wp['name']].append(
                        temps[wp['name']]['delta_t'])
        except Exception as e:
            print(f'    Error: {e}')
            continue

    # Combine with wind data for stability analysis
    if not any(all_delta_t[wp['name']] for wp in carra.WAYPOINTS):
        print('  No valid temperature data extracted')
        return

    mouth_dt = np.concatenate(all_delta_t['mouth']) \
        if all_delta_t['mouth'] else None
    if mouth_dt is None:
        print('  No mouth temperature data')
        return

    print(f'\n  Temperature inversion analysis (mouth):')
    print(f'  Records: {len(mouth_dt)}')
    print(f'  ΔT (T_2m - T_skin) mean: {np.mean(mouth_dt):.2f} K')
    print(f'  Inversion (ΔT > 0): {(mouth_dt > 0).mean():.1%} of time')
    print(f'  Strong inversion (ΔT > 2K): {(mouth_dt > 2).mean():.1%}')

    # TODO: correlate with norðanátt events once we have
    # co-temporal wind + temperature data


def fetch_temp(year=2022, months=None):
    """Download CARRA temperature for stability analysis."""
    import cdsapi

    DATA_DIR.mkdir(exist_ok=True)
    if months is None:
        months = [1, 2, 3, 12]  # winter months

    client = cdsapi.Client()

    for m in months:
        outfile = DATA_DIR / f'carra_temp_{year}_{m:02d}.nc'
        if outfile.exists():
            print(f'  Already have: {outfile.name}')
            continue

        print(f'  Fetching temp {year}-{m:02d}...')
        try:
            client.retrieve(
                'reanalysis-carra-single-levels',
                {
                    'variable': ['2m_temperature', 'skin_temperature'],
                    'product_type': 'analysis',
                    'level_type': 'surface_or_atmosphere',
                    'leadtime_hour': '0',
                    'year': str(year),
                    'month': f'{m:02d}',
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'domain': 'west_domain',
                    'data_format': 'netcdf',
                },
                str(outfile),
            )
            sz = outfile.stat().st_size / 1e6
            print(f'  Downloaded: {sz:.0f} MB')
        except Exception as e:
            print(f'  Failed: {e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--year', type=int, default=2022)
    args = ap.parse_args()

    print('=' * 60)
    print('  THERMAL INVERSION ANALYSIS')
    print('=' * 60)

    if args.fetch:
        fetch_temp(args.year)

    analyze_inversions()
