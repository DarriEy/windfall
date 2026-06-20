#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Capital-region wind climatology for the "can we shield Reykjavik?" test.

Reuses the CARRA single-level pipeline (carra.py) but for waypoints in
and around Reykjavik / Faxafloi instead of Eyjafjordur, written to
data/rvk_winds_*.npz so the fjord data is untouched. The decisive
question is directional: a turbine wake only shields ~10 km downwind, so
shielding the capital is geometrically possible ONLY if the damaging
winds come from a sector where open water (for an offshore array) lies
upwind of the city. This fetch builds the storm-wind rose to settle it.

  python reykjavik.py            # fetch 2022, build the storm-wind rose
"""
import sys
from pathlib import Path
import numpy as np

import carra

DATA = Path('data')

# Reykjavik city + the approaches around it. Faxafloi (open sea) is to
# the W/NW; the interior highlands (Hellisheidi) are to the E/SE.
RVK_WAYPOINTS = [
    {'name': 'rvk_city', 'lat': 64.13, 'lon': -21.90, 'x_km': 0},   # target
    {'name': 'faxafloi', 'lat': 64.25, 'lon': -22.30, 'x_km': 0},   # NW sea
    {'name': 'rvk_wsea', 'lat': 64.10, 'lon': -22.30, 'x_km': 0},   # W sea
    {'name': 'rvk_ssea', 'lat': 63.95, 'lon': -22.05, 'x_km': 0},   # S sea
    {'name': 'hellis',   'lat': 64.05, 'lon': -21.40, 'x_km': 0},   # SE land
]


def fetch(year=2022):
    """Download CARRA 10 m wind for `year` and extract the capital-region
    waypoints to data/rvk_winds_<year>_HX.npz, reusing carra's retrieve +
    extraction by temporarily swapping the waypoint list."""
    import cdsapi
    if not carra._has_credentials():
        raise SystemExit('CDS credentials not found (see carra.fetch()).')
    DATA.mkdir(exist_ok=True)
    orig = carra.WAYPOINTS
    carra.WAYPOINTS = RVK_WAYPOINTS
    try:
        client = cdsapi.Client()
        for half, months in (('H1', range(1, 7)), ('H2', range(7, 13))):
            out = DATA / f'rvk_winds_{year}_{half}.npz'
            if out.exists():
                print(f'  Already have {out.name}')
                continue
            bulk = DATA / f'_rvkbulk_{year}_{half}.nc'
            print(f'  Fetching capital winds {year} {half}...', flush=True)
            client.retrieve('reanalysis-carra-single-levels', {
                'variable': ['10m_u_component_of_wind',
                             '10m_v_component_of_wind'],
                'product_type': 'analysis',
                'level_type': 'surface_or_atmosphere', 'leadtime_hour': '0',
                'year': str(year), 'month': [f'{m:02d}' for m in months],
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'domain': 'west_domain', 'data_format': 'netcdf'}, str(bulk))
            winds = carra._extract_winds([bulk])
            arrays = {}
            for nm, w in winds.items():
                arrays[f'{nm}_speed'] = w['speed_hub'].astype(np.float32)
                arrays[f'{nm}_dir'] = w['direction'].astype(np.float32)
            np.savez_compressed(out, **arrays)
            bulk.unlink()
            print(f'  saved {out.name} ({out.stat().st_size/1024:.0f} KB)')
    finally:
        carra.WAYPOINTS = orig


def fetch_ptemp(year=2022, levels=(1000, 925, 850)):
    """CARRA pressure-level temperature at the capital waypoints, for the
    ABL-stability gate. A farm wake only persists 10-55 km when the lower
    boundary layer is stable; under neutral/unstable gales the same farm
    recovers in 2-3 km (Platis 2018). Writes data/rvk_ptemp_<year>_HX.npz
    (keys <name>_t<level>), index-aligned with rvk_winds_<year>_HX.npz."""
    import cdsapi
    if not carra._has_credentials():
        raise SystemExit('CDS credentials not found (see carra.fetch()).')
    DATA.mkdir(exist_ok=True)
    client = cdsapi.Client()
    for half, months in (('H1', range(1, 7)), ('H2', range(7, 13))):
        out = DATA / f'rvk_ptemp_{year}_{half}.npz'
        if out.exists():
            print(f'  Already have {out.name}')
            continue
        bulk = DATA / f'_rvkpbulk_{year}_{half}.nc'
        print(f'  Fetching capital pressure-temp {year} {half}...', flush=True)
        client.retrieve('reanalysis-carra-pressure-levels', {
            'variable': 'temperature',
            'pressure_level': [str(p) for p in levels],
            'product_type': 'analysis', 'leadtime_hour': '0',
            'year': str(year), 'month': [f'{m:02d}' for m in months],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'domain': 'west_domain', 'data_format': 'netcdf'}, str(bulk))
        ds = carra._open_dataset(bulk)
        tname = next(n for n in ds.data_vars if 't' in n.lower())
        lat_c = [c for c in ds.coords if 'lat' in c.lower()][0]
        lon_c = [c for c in ds.coords if 'lon' in c.lower()][0]
        lev_c = [c for c in (list(ds.coords) + list(ds.dims))
                 if 'level' in c.lower() or 'pressure' in c.lower()][0]
        lat, lon, lev = ds[lat_c].values, ds[lon_c].values, ds[lev_c].values
        spatial = [d for d in ds[tname].dims
                   if d not in carra.TIME_DIMS and d != lev_c]
        arrays = {}
        for wp in RVK_WAYPOINTS:
            i, j = carra._find_nearest(lat, lon, wp['lat'], wp['lon'])
            for p in levels:
                li = int(np.argmin(np.abs(lev - p)))
                t = ds[tname].isel(
                    {lev_c: li, spatial[0]: i, spatial[1]: j}).values.flatten()
                arrays[f"{wp['name']}_t{p}"] = t.astype(np.float32)
        ds.close()
        np.savez_compressed(out, **arrays)
        bulk.unlink()
        print(f'  saved {out.name}')


def stratification_during_storms(year=2022, station='rvk_city', thresh=15.0):
    """Lower-ABL stability N (1000->925 hPa, ~110-760 m, the rotor/wake
    layer) during high-wind hours, paired index-for-index with the
    concurrent wind. Decides which wake regime the capital's gales are
    in: stable (long, shielding-capable wake) vs neutral/unstable (short,
    2-3 km wake that cannot reach the city)."""
    import ridge_stability as rs
    wfiles = sorted(DATA.glob(f'rvk_winds_{year}_*.npz'))
    pfiles = sorted(DATA.glob(f'rvk_ptemp_{year}_*.npz'))
    if not wfiles or not pfiles:
        print('Need both rvk_winds and rvk_ptemp — run the fetches first.')
        return
    sp = np.concatenate([np.load(f)[f'{station}_speed'] for f in wfiles])
    N = np.concatenate([rs.n_from_profile(
        np.load(f)[f'{station}_t1000'], 1000,
        np.load(f)[f'{station}_t925'], 925) for f in pfiles])
    n = min(sp.size, N.size)
    sp, N = sp[:n], N[:n]
    hi = sp > thresh
    if hi.sum() == 0:
        print('  no high-wind samples'); return
    Nh = N[hi]
    bins = [('unstable (N2<0)', Nh < 0),
            ('near-neutral 0-0.005', (Nh >= 0) & (Nh < 0.005)),
            ('stable 0.005-0.010', (Nh >= 0.005) & (Nh < 0.010)),
            ('very stable >0.010', Nh >= 0.010)]
    print(f'  Lower-ABL stability during gales at {station} '
          f'(>{thresh} m/s, {hi.sum()} hrs, {year})')
    for lbl, m in bins:
        share = 100 * m.sum() / hi.sum()
        print(f'  {lbl:<24}{share:>6.1f}%  {"#"*int(share/2)}')
    long_wake = 100 * (Nh >= 0.005).mean()
    print(f'  --> long-wake (stable) regime: {long_wake:.0f}% of gale hours')


def storm_rose(year=2022, station='rvk_city', thresh=15.0):
    """Directional distribution of HIGH-wind hours at the city: which
    sectors bring the damaging winds (10 m speed > thresh m/s)."""
    files = sorted(DATA.glob(f'rvk_winds_{year}_*.npz'))
    if not files:
        print('No rvk_winds data yet — run reykjavik.py fetch first.')
        return
    sp = np.concatenate([np.load(f)[f'{station}_speed'] for f in files])
    di = np.concatenate([np.load(f)[f'{station}_dir'] for f in files])
    sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    edges = np.arange(-22.5, 360, 45)
    hi = sp > thresh
    idx = (np.floor(((di + 22.5) % 360) / 45)).astype(int)
    print(f'  Storm-wind rose at {station} ({year}), speed > {thresh} m/s')
    print(f'  {hi.sum()} high-wind samples of {sp.size} '
          f'({100*hi.mean():.1f}%)')
    print(f'  {"sector":>7}{"% of storm hrs":>16}{"mean spd":>10}')
    for s in range(8):
        m = hi & (idx == s)
        share = 100 * m.sum() / max(hi.sum(), 1)
        ms = sp[m].mean() if m.any() else 0
        bar = '#' * int(share / 2)
        print(f'  {sectors[s]:>7}{share:>13.1f}%  {ms:>7.1f}  {bar}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'rose':
        storm_rose()
    elif cmd == 'ptemp':
        fetch_ptemp()
        stratification_during_storms()
    elif cmd == 'strat':
        stratification_during_storms()
    else:
        fetch()
        storm_rose()
