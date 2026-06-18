#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
CARRA reanalysis wind data for Eyjafjordur.

Downloads 10m wind from the Copernicus Climate Data Store, extracts
wind climatology along the fjord axis, identifies nordanatt events,
and fits Weibull distributions. Supports multi-year ensemble averaging.

Usage:
    python carra.py                          # fetch 2022, process
    python carra.py --year 2021              # single year
    python carra.py --ensemble               # fetch 2010-2022, ensemble
    python carra.py --ensemble --start 2015  # custom range
    python carra.py --process-only --ensemble  # reprocess existing files
"""

import os
import sys
import json
import math
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
CLIMATOLOGY_FILE = DATA_DIR / 'wind_climatology.json'

HUB_HEIGHT = 150.0
REF_HEIGHT = 10.0
Z0_SEA = 0.0005
HEIGHT_FACTOR = math.log(HUB_HEIGHT / Z0_SEA) / math.log(REF_HEIGHT / Z0_SEA)

AREA = [66.2, -18.6, 65.6, -17.8]  # N, W, S, E

TIME_DIMS = {'time', 'valid_time', 'step', 'forecast_reference_time'}

WAYPOINTS = [
    {'name': 'mouth',    'lat': 66.15, 'lon': -18.10, 'x_km': 0},
    {'name': 'outer',    'lat': 66.08, 'lon': -18.20, 'x_km': 10},
    {'name': 'dalvik',   'lat': 66.00, 'lon': -18.30, 'x_km': 18},
    {'name': 'hrisey',   'lat': 65.97, 'lon': -18.40, 'x_km': 25},
    {'name': 'mid',      'lat': 65.85, 'lon': -18.25, 'x_km': 38},
    {'name': 'inner',    'lat': 65.75, 'lon': -18.15, 'x_km': 48},
    {'name': 'akureyri', 'lat': 65.68, 'lon': -18.09, 'x_km': 55},
]


# ── CDS download ──────────────────────────────────────────────────

def _has_credentials():
    if os.path.exists(os.path.expanduser('~/.cdsapirc')):
        return True
    return 'CDSAPI_URL' in os.environ and 'CDSAPI_KEY' in os.environ


def fetch(year=2022, months=None, chunk_months=6):
    """Download CARRA 10m wind, extract waypoints, discard bulk data.

    Downloads in multi-month chunks (default 6 = 2 requests/year) to
    minimize CDS queue overhead. Each chunk is ~3-4 GB for the full
    west_domain; we extract the 7 fjord waypoints (~10 KB/month) and
    delete the bulk file immediately.

    20 years = 40 requests instead of 240.
    """
    import cdsapi

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if months is None:
        months = list(range(1, 13))

    if not _has_credentials():
        print("CDS API credentials not found.")
        print()
        print("  1. Register at https://cds.climate.copernicus.eu")
        print("  2. Go to your profile and copy your API key")
        print("  3. Create ~/.cdsapirc:")
        print("       url: https://cds.climate.copernicus.eu/api")
        print("       key: <YOUR-API-KEY>")
        raise SystemExit(1)

    client = cdsapi.Client()
    extracted = []

    # Split months into chunks
    chunks = [months[i:i + chunk_months]
              for i in range(0, len(months), chunk_months)]

    for chunk in chunks:
        # Check if all months in chunk already extracted
        compacts = [DATA_DIR / f'winds_{year}_{m:02d}.npz' for m in chunk]
        if all(c.exists() for c in compacts):
            print(f"  Already extracted: {year} months {chunk[0]}-{chunk[-1]}")
            extracted.extend(compacts)
            continue

        label = f'{year}-{chunk[0]:02d}:{chunk[-1]:02d}'
        bulk = DATA_DIR / f'_bulk_{year}_{chunk[0]:02d}_{chunk[-1]:02d}.nc'
        print(f"  Fetching {label} ({len(chunk)} months)...",
              end='', flush=True)
        try:
            client.retrieve(
                'reanalysis-carra-single-levels',
                {
                    'variable': [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                    ],
                    'product_type': 'analysis',
                    'level_type': 'surface_or_atmosphere',
                    'leadtime_hour': '0',
                    'year': str(year),
                    'month': [f'{m:02d}' for m in chunk],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'domain': 'west_domain',
                    'data_format': 'netcdf',
                },
                str(bulk),
            )
            sz_gb = bulk.stat().st_size / 1e9
            print(f' {sz_gb:.1f} GB, extracting...', end='', flush=True)

            winds = _extract_winds([bulk])

            # Split by month would require time coordinate parsing;
            # for simplicity save the whole chunk as one file per chunk
            chunk_compact = DATA_DIR / f'winds_{year}_{chunk[0]:02d}_{chunk[-1]:02d}.npz'
            arrays = {}
            for wp_name, w in winds.items():
                arrays[f'{wp_name}_speed'] = w['speed_hub'].astype(np.float32)
                arrays[f'{wp_name}_dir'] = w['direction'].astype(np.float32)
            np.savez_compressed(chunk_compact, **arrays)

            bulk.unlink()
            sz_kb = chunk_compact.stat().st_size / 1024
            print(f' {sz_kb:.0f} KB saved, bulk deleted')
            extracted.append(chunk_compact)
        except Exception as e:
            if bulk.exists():
                bulk.unlink()
            print(f' failed: {e}')

    return extracted


def fetch_pressure_temp(year=2022, months=None, levels=(1000, 925, 850)):
    """Download CARRA pressure-level temperature and extract it at the
    fjord waypoints to data/ptemp_YYYY_HX.npz (keys <station>_t<level>),
    for the ridge-height stability analysis (ridge_stability.py).

    Mirrors fetch(): download a chunk, extract the 7 waypoints across the
    requested pressure levels, discard the bulk file. Requires CDS
    credentials (see fetch())."""
    import cdsapi
    import xarray as xr

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if months is None:
        months = list(range(1, 13))
    if not _has_credentials():
        raise SystemExit('CDS credentials not found (see fetch()).')

    client = cdsapi.Client()
    halves = {'H1': [m for m in months if m <= 6],
              'H2': [m for m in months if m > 6]}
    for half, mns in halves.items():
        if not mns:
            continue
        out = DATA_DIR / f'ptemp_{year}_{half}.npz'
        if out.exists():
            print(f"  Already have {out.name}")
            continue
        bulk = DATA_DIR / f'_pbulk_{year}_{half}.nc'
        print(f"  Fetching pressure temp {year} {half}...", flush=True)
        client.retrieve('reanalysis-carra-pressure-levels', {
            'variable': 'temperature',
            'pressure_level': [str(p) for p in levels],
            'product_type': 'analysis', 'leadtime_hour': '0',
            'year': str(year), 'month': [f'{m:02d}' for m in mns],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'domain': 'west_domain', 'data_format': 'netcdf'}, str(bulk))
        ds = _open_dataset(bulk)
        tname = next(n for n in ds.data_vars if 't' in n.lower())
        lat_c = [c for c in ds.coords if 'lat' in c.lower()][0]
        lon_c = [c for c in ds.coords if 'lon' in c.lower()][0]
        lev_c = [c for c in (list(ds.coords) + list(ds.dims))
                 if 'level' in c.lower() or 'pressure' in c.lower()][0]
        lat, lon, lev = ds[lat_c].values, ds[lon_c].values, ds[lev_c].values
        spatial = [d for d in ds[tname].dims
                   if d not in TIME_DIMS and d != lev_c]
        arrays = {}
        for wp in WAYPOINTS:
            i, j = _find_nearest(lat, lon, wp['lat'], wp['lon'])
            for p in levels:
                li = int(np.argmin(np.abs(lev - p)))
                t = ds[tname].isel(
                    {lev_c: li, spatial[0]: i, spatial[1]: j}).values.flatten()
                arrays[f"{wp['name']}_t{p}"] = t.astype(np.float32)
        ds.close()
        np.savez_compressed(out, **arrays)
        bulk.unlink()
        print(f"  Saved {out.name}")


def fetch_range(start=2010, end=2022):
    """Download CARRA data for a range of years."""
    print(f"  Fetching CARRA {start}-{end} ({end - start + 1} years)...")
    for year in range(start, end + 1):
        try:
            fetch(year)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  Warning: {year} failed: {e}")


# ── netCDF processing ─────────────────────────────────────────────

def _find_nearest(lat_grid, lon_grid, target_lat, target_lon):
    # Handle 0-360 vs -180/+180 longitude convention
    tlon = target_lon % 360
    lon_g = lon_grid % 360
    cos_lat = np.cos(np.radians(target_lat))
    if lat_grid.ndim == 1 and lon_g.ndim == 1:
        lat_d = (lat_grid[:, None] - target_lat) ** 2
        lon_d = ((lon_g[None, :] - tlon) * cos_lat) ** 2
        dist = lat_d + lon_d
    else:
        dist = ((lat_grid - target_lat) ** 2
                + ((lon_g - tlon) * cos_lat) ** 2)
    return np.unravel_index(np.argmin(dist), dist.shape)


def _detect_vars(ds):
    u_var = v_var = lat_var = lon_var = None
    for name in ds.data_vars:
        nl = name.lower()
        if u_var is None and ('u' in nl) and ('wind' in nl or '10' in nl):
            u_var = name
        elif v_var is None and ('v' in nl) and ('wind' in nl or '10' in nl):
            v_var = name
    if u_var is None:
        for try_u, try_v in [('u10', 'v10'), ('U10', 'V10')]:
            if try_u in ds.data_vars:
                u_var, v_var = try_u, try_v
                break
    for name in list(ds.coords) + list(ds.data_vars):
        nl = name.lower()
        if lat_var is None and 'lat' in nl:
            lat_var = name
        if lon_var is None and 'lon' in nl:
            lon_var = name
    if u_var is None or v_var is None:
        raise ValueError(f"Cannot find wind vars in {list(ds.data_vars)}")
    if lat_var is None or lon_var is None:
        raise ValueError(f"Cannot find lat/lon in {list(ds.coords)}")
    return u_var, v_var, lat_var, lon_var


def _fit_weibull(speeds):
    speeds = speeds[speeds > 0.1]
    if len(speeds) < 20:
        return {'k': 2.0, 'A': 1.0}
    mean = float(np.mean(speeds))
    std = float(np.std(speeds))
    if mean <= 0 or std <= 0:
        return {'k': 2.0, 'A': mean or 1.0}
    cv = std / mean
    k = cv ** (-1.086)
    k = max(1.2, min(k, 8.0))
    A = mean / math.gamma(1 + 1 / k)
    return {'k': round(k, 2), 'A': round(A, 2)}


def _open_dataset(path):
    import xarray as xr
    try:
        return xr.open_dataset(path)
    except OSError:
        return xr.open_dataset(path, engine='cfgrib',
                               backend_kwargs={'indexpath': ''})


def _extract_winds(nc_files):
    """Extract hub-height wind speed and direction at all waypoints.

    Accepts a single path or list of paths (monthly chunks).
    Returns {station_name: {'speed_hub': ndarray, 'direction': ndarray,
                             'grid_lat': float, 'grid_lon': float}}
    """
    if isinstance(nc_files, (str, Path)):
        nc_files = [nc_files]

    all_u = {wp['name']: [] for wp in WAYPOINTS}
    all_v = {wp['name']: [] for wp in WAYPOINTS}
    grid_info = {}

    for nc_file in nc_files:
        ds = _open_dataset(nc_file)
        u_var, v_var, lat_var, lon_var = _detect_vars(ds)
        lat_grid = ds[lat_var].values
        lon_grid = ds[lon_var].values
        spatial_dims = [d for d in ds[u_var].dims if d not in TIME_DIMS]

        for wp in WAYPOINTS:
            i, j = _find_nearest(lat_grid, lon_grid, wp['lat'], wp['lon'])
            sel = {spatial_dims[0]: i, spatial_dims[1]: j}
            u = ds[u_var].isel(**sel).values.flatten()
            v = ds[v_var].isel(**sel).values.flatten()
            ok = np.isfinite(u) & np.isfinite(v)
            all_u[wp['name']].append(u[ok])
            all_v[wp['name']].append(v[ok])
            if wp['name'] not in grid_info:
                grid_info[wp['name']] = {
                    'grid_lat': float(
                        lat_grid[i] if lat_grid.ndim == 1
                        else lat_grid[i, j]),
                    'grid_lon': float(
                        lon_grid[j] if lon_grid.ndim == 1
                        else lon_grid[i, j]),
                }
        ds.close()

    result = {}
    for wp in WAYPOINTS:
        u = np.concatenate(all_u[wp['name']])
        v = np.concatenate(all_v[wp['name']])
        speed_10 = np.sqrt(u ** 2 + v ** 2)
        result[wp['name']] = {
            'speed_hub': speed_10 * HEIGHT_FACTOR,
            'direction': (270 - np.degrees(np.arctan2(v, u))) % 360,
            **grid_info.get(wp['name'], {}),
        }
    return result


def _station_stats(speed_hub, direction, wp):
    """Compute per-station statistics from wind arrays."""
    wb = _fit_weibull(speed_hub)
    is_north = (direction >= 330) | (direction <= 30)
    is_nordanatt = is_north & (speed_hub > 10)
    nord_speeds = speed_hub[is_nordanatt]
    wb_nord = _fit_weibull(nord_speeds) if len(nord_speeds) > 30 else None

    st = {
        'name': wp['name'],
        'x_km': wp['x_km'],
        'n_records': int(len(speed_hub)),
        'mean_speed_hub': round(float(np.mean(speed_hub)), 2),
        'std_speed_hub': round(float(np.std(speed_hub)), 2),
        'weibull_k': wb['k'],
        'weibull_A': wb['A'],
        'nordanatt_fraction': round(float(np.mean(is_nordanatt)), 4),
        'nordanatt_hours_yr': round(float(np.mean(is_nordanatt)) * 8766),
        'nordanatt_mean_speed': (
            round(float(np.mean(nord_speeds)), 2)
            if len(nord_speeds) > 0 else 0.0),
    }
    if wb_nord:
        st['nordanatt_weibull_k'] = wb_nord['k']
        st['nordanatt_weibull_A'] = wb_nord['A']
    return st


# ── single-year processing ────────────────────────────────────────

def _year_files(year):
    """Find extracted .npz files (or legacy .nc) for a year."""
    npz = sorted(DATA_DIR.glob(f'winds_{year}_*.npz'))
    if npz:
        return npz
    nc = sorted(DATA_DIR.glob(f'carra_10m_{year}_*.nc'))
    if nc:
        return nc
    return []


def _load_compact(npz_files):
    """Load extracted waypoint data from .npz files."""
    result = {wp['name']: {'speed_hub': [], 'direction': []}
              for wp in WAYPOINTS}
    for f in npz_files:
        data = np.load(f)
        for wp in WAYPOINTS:
            name = wp['name']
            result[name]['speed_hub'].append(data[f'{name}_speed'])
            result[name]['direction'].append(data[f'{name}_dir'])
    for wp in WAYPOINTS:
        name = wp['name']
        result[name]['speed_hub'] = np.concatenate(
            result[name]['speed_hub'])
        result[name]['direction'] = np.concatenate(
            result[name]['direction'])
    return result


def process(year=2022):
    """Process CARRA data for a year into wind climatology JSON."""
    files = _year_files(year)
    if not files:
        print(f"  No data files found for {year}")
        return None
    print(f"  Processing {year} ({len(files)} files)...")
    if files[0].suffix == '.npz':
        winds = _load_compact(files)
    else:
        winds = _extract_winds(files)

    stations = {}
    for wp in WAYPOINTS:
        w = winds[wp['name']]
        st = _station_stats(w['speed_hub'], w['direction'], wp)
        st['grid_lat'] = w['grid_lat']
        st['grid_lon'] = w['grid_lon']
        stations[wp['name']] = st
        print(f"    {wp['name']:>10}: hub {st['mean_speed_hub']:.1f} m/s  "
              f"nordanatt {st['nordanatt_fraction']:.1%} "
              f"({st['nordanatt_hours_yr']:.0f} h/yr)")

    climatology = {
        'source': 'CARRA',
        'file': str(nc_file),
        'hub_height_m': HUB_HEIGHT,
        'height_factor': round(HEIGHT_FACTOR, 3),
        'stations': stations,
    }
    _save(climatology)
    return climatology


# ── multi-year ensemble ───────────────────────────────────────────

def process_ensemble(start=2010, end=2022):
    """Process all available CARRA years into ensemble climatology."""
    print(f"  Building ensemble {start}-{end}...")

    accum = {wp['name']: {'speed': [], 'direction': []}
             for wp in WAYPOINTS}
    yearly = []
    years_used = []

    for year in range(start, end + 1):
        files = _year_files(year)
        if not files:
            continue
        print(f"    {year} ({len(files)} files)...", end='', flush=True)
        try:
            if files[0].suffix == '.npz':
                winds = _load_compact(files)
            else:
                winds = _extract_winds(files)
        except Exception as e:
            print(f" error: {e}")
            continue

        yr = {'year': year, 'stations': {}}
        for wp in WAYPOINTS:
            w = winds[wp['name']]
            accum[wp['name']]['speed'].append(w['speed_hub'])
            accum[wp['name']]['direction'].append(w['direction'])
            yr['stations'][wp['name']] = {
                'mean': round(float(np.mean(w['speed_hub'])), 2),
            }
        yearly.append(yr)
        years_used.append(year)
        mouth_mean = yr['stations']['mouth']['mean']
        print(f" mouth={mouth_mean:.1f} m/s")

    if not years_used:
        print("  No CARRA files found! Run without --process-only first.")
        return None

    stations = {}
    for wp in WAYPOINTS:
        name = wp['name']
        speed = np.concatenate(accum[name]['speed'])
        direction = np.concatenate(accum[name]['direction'])
        st = _station_stats(speed, direction, wp)

        yr_means = [y['stations'][name]['mean'] for y in yearly]
        st['n_years'] = len(years_used)
        st['interannual_std'] = round(float(np.std(yr_means)), 2)
        st['yearly_means'] = yr_means
        st['year_range'] = [
            round(float(min(yr_means)), 2),
            round(float(max(yr_means)), 2),
        ]
        stations[name] = st

    climatology = {
        'source': 'CARRA_ensemble',
        'years': years_used,
        'n_years': len(years_used),
        'hub_height_m': HUB_HEIGHT,
        'height_factor': round(HEIGHT_FACTOR, 3),
        'stations': stations,
    }
    _save(climatology)

    print(f"\n  Ensemble: {len(years_used)} years")
    for wp in WAYPOINTS:
        st = stations[wp['name']]
        print(f"    {st['name']:>10}: {st['mean_speed_hub']:.1f} "
              f"+/- {st['interannual_std']:.1f} m/s  "
              f"nordanatt {st['nordanatt_hours_yr']:.0f} h/yr")

    return climatology


# ── helpers ───────────────────────────────────────────────────────

def _save(climatology):
    CLIMATOLOGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLIMATOLOGY_FILE, 'w') as fh:
        json.dump(climatology, fh, indent=2)
    print(f"  Saved: {CLIMATOLOGY_FILE}")


def load_climatology():
    if CLIMATOLOGY_FILE.exists():
        with open(CLIMATOLOGY_FILE) as fh:
            return json.load(fh)
    return None


def synthetic_climatology():
    return {
        'source': 'synthetic',
        'hub_height_m': HUB_HEIGHT,
        'height_factor': round(HEIGHT_FACTOR, 3),
        'note': 'Estimated. Run `python carra.py` for reanalysis.',
        'stations': {
            'mouth': {
                'name': 'mouth', 'x_km': 0,
                'mean_speed_hub': 8.5,
                'weibull_k': 2.0, 'weibull_A': 9.6,
                'nordanatt_fraction': 0.12,
                'nordanatt_hours_yr': 1050,
                'nordanatt_mean_speed': 18.0,
            },
            'outer': {
                'name': 'outer', 'x_km': 10,
                'mean_speed_hub': 8.3,
                'weibull_k': 2.0, 'weibull_A': 9.4,
                'nordanatt_fraction': 0.11,
                'nordanatt_hours_yr': 960,
                'nordanatt_mean_speed': 17.5,
            },
            'akureyri': {
                'name': 'akureyri', 'x_km': 55,
                'mean_speed_hub': 7.5,
                'weibull_k': 1.8, 'weibull_A': 8.5,
                'nordanatt_fraction': 0.10,
                'nordanatt_hours_yr': 876,
                'nordanatt_mean_speed': 17.0,
            },
        },
    }


# ── CLI ───────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='CARRA wind data for Eyjafjordur')
    ap.add_argument('--year', type=int, default=2022,
                    help='single-year mode (default 2022)')
    ap.add_argument('--ensemble', action='store_true',
                    help='multi-year ensemble mode')
    ap.add_argument('--start', type=int, default=2010,
                    help='ensemble start year (default 2010)')
    ap.add_argument('--end', type=int, default=2022,
                    help='ensemble end year (default 2022)')
    ap.add_argument('--months', type=int, nargs='+', default=None,
                    help='specific months (e.g. --months 1 2 3 12)')
    ap.add_argument('--process-only', action='store_true',
                    help='skip download, reprocess existing files')
    args = ap.parse_args()

    if args.ensemble:
        if not args.process_only:
            for yr in range(args.start, args.end + 1):
                fetch(yr)
        process_ensemble(args.start, args.end)
    else:
        if not args.process_only:
            fetch(args.year, months=args.months)
        process(args.year)

    print()
    print("  Ready. Run `python analyze.py` or `python sweep.py`.")


if __name__ == '__main__':
    main()
