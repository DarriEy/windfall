#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Generalize the capital-region wind pipeline (reykjavik.py) to additional
Icelandic regional centres, to test where community wind-sheltering /
reduction is realizable: Ísafjörður (extreme deep-fjord, Westfjords) and
Egilsstaðir (inland valley, East Iceland). Reuses carra.py extraction;
writes data/<site>_winds_*.npz and <site>_ptemp_*.npz so the existing
fjord/capital data is untouched.

  python multisite.py fetch isafjordur
  python multisite.py rose  isafjordur
"""
import sys
from pathlib import Path
import numpy as np
import carra

DATA = Path('data')

# town (target) + surrounding approach/terrain points for each site
SITES = {
    'isafjordur': [                        # Skutulsfjörður / Ísafjarðardjúp, deep narrow fjord
        {'name': 'isf_town',  'lat': 66.075, 'lon': -23.135, 'x_km': 0},
        {'name': 'isf_fjord', 'lat': 66.110, 'lon': -23.200, 'x_km': 0},   # out the fjord (NW)
        {'name': 'isf_djup',  'lat': 66.160, 'lon': -23.000, 'x_km': 0},   # Ísafjarðardjúp (NE)
        {'name': 'isf_sw',    'lat': 66.030, 'lon': -23.250, 'x_km': 0},   # up-fjord (SW)
    ],
    'egilsstadir': [                       # Fljótsdalshérað inland valley (Lagarfljót)
        {'name': 'egs_town', 'lat': 65.270, 'lon': -14.395, 'x_km': 0},
        {'name': 'egs_n',    'lat': 65.420, 'lon': -14.420, 'x_km': 0},    # valley N -> Héraðsflói coast
        {'name': 'egs_s',    'lat': 65.130, 'lon': -14.420, 'x_km': 0},    # valley S -> Fljótsdalur
        {'name': 'egs_e',    'lat': 65.270, 'lon': -14.050, 'x_km': 0},    # highland E
    ],
}
# Egilsstaðir is far east — may fall outside CARRA west_domain; try west then east.
DOMAINS = {'isafjordur': ['west_domain'], 'egilsstadir': ['west_domain', 'east_domain']}


def _retrieve(client, dataset, req, dest):
    client.retrieve(dataset, req, str(dest))


def fetch(site, year=2022):
    import cdsapi
    if not carra._has_credentials():
        raise SystemExit('CDS credentials not found.')
    DATA.mkdir(exist_ok=True)
    orig = carra.WAYPOINTS
    carra.WAYPOINTS = SITES[site]
    client = cdsapi.Client()
    try:
        for half, months in (('H1', range(1, 7)), ('H2', range(7, 13))):
            out = DATA / f'{site}_winds_{year}_{half}.npz'
            if out.exists():
                print(f'  have {out.name}'); continue
            bulk = DATA / f'_{site}bulk_{year}_{half}.nc'
            ok = False
            for dom in DOMAINS[site]:
                print(f'  fetch {site} winds {year} {half} ({dom})...', flush=True)
                try:
                    _retrieve(client, 'reanalysis-carra-single-levels', {
                        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                        'product_type': 'analysis', 'level_type': 'surface_or_atmosphere',
                        'leadtime_hour': '0', 'year': str(year),
                        'month': [f'{m:02d}' for m in months],
                        'day': [f'{d:02d}' for d in range(1, 32)],
                        'time': ['00:00', '06:00', '12:00', '18:00'],
                        'domain': dom, 'data_format': 'netcdf'}, bulk)
                    w = carra._extract_winds([bulk])
                    if np.isfinite(w[SITES[site][0]['name']]['speed_hub']).mean() > 0.5:
                        ok = True; break
                    print(f'    {dom}: town point is NaN (outside domain), trying next')
                except Exception as e:
                    print(f'    {dom} failed: {e}')
                finally:
                    if bulk.exists() and not ok:
                        bulk.unlink()
            if not ok:
                print(f'  !! {site} {half}: no domain covered the town'); continue
            arrays = {}
            for nm, d in w.items():
                arrays[f'{nm}_speed'] = d['speed_hub'].astype(np.float32)
                arrays[f'{nm}_dir'] = d['direction'].astype(np.float32)
            np.savez_compressed(out, **arrays); bulk.unlink()
            print(f'  saved {out.name}')
    finally:
        carra.WAYPOINTS = orig


def rose(site, year=2022, thresh=15.0):
    town = SITES[site][0]['name']
    files = sorted(DATA.glob(f'{site}_winds_{year}_*.npz'))
    if not files:
        print(f'no {site} winds yet'); return
    sp = np.concatenate([np.load(f)[f'{town}_speed'] for f in files])
    di = np.concatenate([np.load(f)[f'{town}_dir'] for f in files])
    names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = (np.floor(((di + 22.5) % 360) / 45)).astype(int)
    hi = sp > thresh
    print(f'  {site} ({town}) 2022: mean {sp.mean():.1f} m/s, gale-hours(>{thresh}) {100*hi.mean():.1f}%')
    for s in range(8):
        m = hi & (idx == s); share = 100 * m.sum() / max(hi.sum(), 1)
        print(f'    {names[s]:>3} {share:>5.1f}%  mean {sp[m].mean() if m.any() else 0:4.1f}  {"#"*int(share/2)}')


if __name__ == '__main__':
    cmd, site = sys.argv[1], sys.argv[2]
    if cmd == 'fetch':
        fetch(site); rose(site)
    elif cmd == 'rose':
        rose(site)
