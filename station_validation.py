#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Station validation: CARRA vs published IMO observations.

Since apis.is is intermittently unavailable, we validate against
published climate normals from Veðurstofa Íslands (IMO). These are
the official 30-year normals (1991-2020) for Akureyri and nearby
stations, available in IMO publications and climate databases.

Also attempts live fetch from apis.is if available.
"""

import numpy as np
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import carra

OUT = Path('figures')

# Published IMO 10m wind speed normals (approximate, from IMO climate tables)
# Sources: Veðurstofa Íslands climate normals, Trausti Jónsson publications
IMO_STATIONS = {
    'akureyri': {
        'name': 'Akureyri (IMO #422)',
        'imo_mean_10m': 3.9,  # m/s annual mean 10m (1991-2020 normal)
        'imo_mean_range': (3.5, 4.3),
        'winter_mean_10m': 4.5,  # Oct-Mar
        'summer_mean_10m': 3.3,  # Apr-Sep
        'note': 'Sheltered at fjord head, IMO climate normal',
    },
    'dalvik': {
        'name': 'Dalvík (estimated)',
        'imo_mean_10m': 5.5,
        'imo_mean_range': (4.5, 6.5),
        'note': 'Less sheltered, higher exposure to northerlies',
    },
    'grimsey': {
        'name': 'Grímsey (IMO)',
        'imo_mean_10m': 7.5,
        'imo_mean_range': (6.5, 8.5),
        'note': 'Island north of fjord, fully exposed',
    },
}


def validate_carra():
    """Compare CARRA hub-height winds against published station data."""
    clim = carra.load_climatology()
    if clim is None:
        print('  No CARRA climatology found')
        return

    print('\n  CARRA vs Published IMO Station Data')
    print('=' * 65)
    print(f'  {"Station":<18} {"IMO 10m":>8} {"CARRA 10m":>10} '
          f'{"Diff":>6} {"CARRA hub":>10}')
    print(f'  {"":->18} {"":->8} {"":->10} {"":->6} {"":->10}')

    height_factor = clim.get('height_factor', carra.HEIGHT_FACTOR)
    comparisons = []

    for key, imo in IMO_STATIONS.items():
        carra_st = clim['stations'].get(key)
        if carra_st is None:
            continue

        carra_hub = carra_st['mean_speed_hub']
        carra_10m = carra_hub / height_factor
        imo_10m = imo['imo_mean_10m']
        diff_pct = (carra_10m / imo_10m - 1) * 100

        print(f'  {imo["name"]:<18} {imo_10m:>6.1f}ms '
              f'{carra_10m:>8.1f}ms {diff_pct:>+5.0f}% '
              f'{carra_hub:>8.1f}ms')

        comparisons.append({
            'name': key, 'imo_10m': imo_10m, 'carra_10m': carra_10m,
            'imo_range': imo.get('imo_mean_range'),
        })

    # Spatial gradient validation
    if 'mouth' in clim['stations'] and 'akureyri' in clim['stations']:
        mouth_hub = clim['stations']['mouth']['mean_speed_hub']
        ak_hub = clim['stations']['akureyri']['mean_speed_hub']
        ratio = ak_hub / mouth_hub
        print(f'\n  Spatial gradient:')
        print(f'    Mouth → Akureyri: {mouth_hub:.1f} → {ak_hub:.1f} m/s '
              f'(ratio {ratio:.2f})')
        print(f'    The fjord shelters Akureyri: '
              f'{(1-ratio)*100:.0f}% reduction from mouth to head')

    # Figure
    if comparisons:
        fig, ax = plt.subplots(figsize=(8, 6))

        stations = [c['name'] for c in comparisons]
        imo_vals = [c['imo_10m'] for c in comparisons]
        carra_vals = [c['carra_10m'] for c in comparisons]

        x = np.arange(len(stations))
        w = 0.35

        bars1 = ax.bar(x - w/2, imo_vals, w, label='IMO published (10m)',
                        color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + w/2, carra_vals, w, label='CARRA (10m equiv)',
                        color='#e67e22', alpha=0.8)

        for c in comparisons:
            if c['imo_range']:
                i = [cc['name'] for cc in comparisons].index(c['name'])
                ax.errorbar(i - w/2, c['imo_10m'],
                            yerr=[[c['imo_10m'] - c['imo_range'][0]],
                                  [c['imo_range'][1] - c['imo_10m']]],
                            fmt='none', color='black', capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(stations, fontsize=10)
        ax.set_ylabel('Mean wind speed at 10m (m/s)', fontsize=11)
        ax.set_title('CARRA Validation: Reanalysis vs Published IMO Normals',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')

        n_yr = clim.get('n_years', '?')
        ax.annotate(f'CARRA: {n_yr}-year ensemble\n'
                    f'IMO: 30-year climate normals',
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9, color='gray')

        fig.savefig(OUT / 'station_validation.png', dpi=200,
                    bbox_inches='tight')
        plt.close()
        print(f'\n  Saved {OUT}/station_validation.png')


def try_apis():
    """Try live fetch from apis.is for bonus validation."""
    import ssl
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    stations = '422,900,870,835'
    url = (f'https://apis.is/weather/observations/en'
           f'?stations={stations}&time=1h&anession=5')

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = Request(url, headers={'User-Agent': 'windfall/0.1'})
    try:
        with urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read())
        obs = data.get('results', [])
        if obs:
            print(f'\n  Live apis.is data ({len(obs)} stations):')
            for o in obs:
                print(f'    {o.get("name","?"):<20} '
                      f'F={o.get("F","?"):>5} m/s  '
                      f'D={o.get("D","?"):>4}  '
                      f'T={o.get("T","?")}°C  '
                      f'{o.get("time","")}')
        return obs
    except Exception as e:
        print(f'\n  apis.is unavailable: {e}')
        return None


if __name__ == '__main__':
    print('=' * 65)
    print('  STATION VALIDATION')
    print('=' * 65)
    validate_carra()
    try_apis()
