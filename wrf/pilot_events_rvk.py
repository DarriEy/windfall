#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
RVK SW-wind event set for the full Reykjavík shielding-vs-energy campaign.

Picks representative SW-sector events at the city across the intensity
distribution (everyday / moderate / gale), weighted by SW-band frequency,
so per-event WRF results integrate to an annual service. The gale
(2022-03-17, already staged as rvk_swgale, 24 h run) anchors the top band;
everyday + moderate are staged fresh (12 h runs) from ERA5.

Also emits, for each fresh event, wrf/<stagedir>/namelist.wps + .input
templated from the swgale namelists with the event's dates/run length.

CARRA RVK winds are 2022-only (data/rvk_winds_2022_H*.npz); the annual
service is therefore a 2022-based estimate (noted as a limitation).
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RVK = HERE / 'rvk'

SW = (190, 260)                       # SW sector (deg)
BASE = datetime(2022, 1, 1, 0)        # rvk_winds index 0 (verified vs gale)

# (label, target SW-speed quantile, band weight within SW, stagedir, run_h)
BANDS = [
    ('everyday', 0.50, 0.60, 'rvk_sw_everyday', 12),
    ('moderate', 0.78, 0.28, 'rvk_sw_moderate', 12),
    ('gale',     0.95, 0.12, 'rvk_swgale',      24),   # already staged
]


def load():
    h1 = np.load(DATA / 'rvk_winds_2022_H1.npz')
    h2 = np.load(DATA / 'rvk_winds_2022_H2.npz')
    sp = np.concatenate([h1['rvk_city_speed'], h2['rvk_city_speed']])
    di = np.concatenate([h1['rvk_city_dir'], h2['rvk_city_dir']])
    return sp, di


def write_namelists(stagedir, d, run_h):
    """Template namelist.wps + namelist.input for a fresh event from the
    swgale versions, substituting dates and run length."""
    out = HERE / stagedir
    out.mkdir(exist_ok=True)
    sh = 6 if d.hour >= 14 else 0                 # 12 h window within the day
    start = d.replace(hour=sh)
    end = start + timedelta(hours=run_h)
    wps = (RVK / 'namelist_swgale.wps').read_text()
    wps = wps.replace("3*'2022-03-17_06:00:00'",
                      f"3*'{start:%Y-%m-%d_%H:%M:%S}'")
    wps = wps.replace("end_date=3*'2022-03-18_06:00:00'",
                      f"end_date=3*'{end:%Y-%m-%d_%H:%M:%S}'")
    (out / 'namelist.wps').write_text(wps)
    inp = (RVK / 'namelist_swgale.input').read_text()
    inp = inp.replace('run_hours=24', f'run_hours={run_h}')
    inp = inp.replace('start_month=3*03, start_day=3*17, start_hour=3*06',
                      f'start_month=3*{start.month:02d}, '
                      f'start_day=3*{start.day:02d}, start_hour=3*{start.hour:02d}')
    inp = inp.replace('end_month=3*03, end_day=3*18, end_hour=3*06',
                      f'end_month=3*{end.month:02d}, '
                      f'end_day=3*{end.day:02d}, end_hour=3*{end.hour:02d}')
    (out / 'namelist.input').write_text(inp)
    return start, end


def main():
    sp, di = load()
    sw = (di >= SW[0]) & (di <= SW[1])
    sw_sp = sp[sw]
    hours_per_yr = sw.sum() / 1 * 6.0            # 1 yr (2022), 6-hourly
    print(f'SW-sector: {sw.sum()}/{len(sp)} recs ({100*sw.sum()/len(sp):.0f}%)'
          f' ~ {hours_per_yr:.0f} h/yr (2022)')

    idx = np.where(sw)[0]
    order = idx[np.argsort(-sp[idx])]
    absday = lambda i: i // 4                    # noqa: E731  6-hourly -> day
    distinct = []
    for i in order:
        if all(abs(absday(i) - absday(p)) > 2 for p in distinct):
            distinct.append(i)
    distinct = np.array(distinct)
    peaks = sp[distinct]

    events = []
    for label, q, w, stagedir, run_h in BANDS:
        target = np.quantile(sw_sp, q)
        if label == 'gale':
            d = datetime(2022, 3, 17, 12)        # the staged swgale (Mar 17-18)
            j = int(round((d - BASE).total_seconds() / 21600))
        else:
            j = int(distinct[np.argmin(np.abs(peaks - target))])
            d = BASE + timedelta(hours=6 * j)
        ev = {'band': label, 'event_label': stagedir, 'stagedir': stagedir,
              'date': f'{d:%Y-%m-%d}', 'peak_hour_utc': d.hour,
              'city_peak_ms': round(float(sp[j]), 1),
              'city_dir': round(float(di[j])),
              'run_hours': run_h, 'weight': round(w, 3),
              'hours_per_yr': round(w * hours_per_yr, 1),
              'target_speed': round(float(target), 1),
              'staged': label == 'gale'}
        if label != 'gale':
            start, end = write_namelists(stagedir, d, run_h)
            ev['wrf_start'] = f'{start:%Y-%m-%d_%H}'
            ev['fetch_days'] = sorted({f'{start:%d}', f'{end:%d}'})
            ev['fetch_month'] = f'{start:%m}'
        events.append(ev)

    out = {'regime': 'SW-sector at Reykjavík (190-260 deg)',
           'sw_frac': round(float(sw.mean()), 4),
           'sw_hours_per_yr': round(hours_per_yr, 1),
           'note': 'CARRA RVK is 2022-only; annual service is 2022-based.',
           'events': events}
    (HERE / 'pilot_events_rvk.json').write_text(json.dumps(out, indent=2))
    print(f'\n{"band":<10}{"date":<12}{"pk_ms":>6}{"dir":>5}{"runh":>5}'
          f'{"wt":>6}{"h/yr":>7}{"staged":>8}')
    for e in events:
        print(f'  {e["band"]:<8}{e["date"]:<12}{e["city_peak_ms"]:>6}'
              f'{e["city_dir"]:>5}{e["run_hours"]:>5}{e["weight"]:>6}'
              f'{e["hours_per_yr"]:>7}{str(e["staged"]):>8}')
    print(f'\nwrote pilot_events_rvk.json + namelists for fresh events')


if __name__ == '__main__':
    main()
