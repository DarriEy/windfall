#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Lock the 5 norðanátt events for the Eyjafjörður shielding-vs-energy pilot.

Selects one representative event per intensity band of the 20-yr CARRA
norðanátt distribution (mouth wind from the north sector, >=10 m/s),
weighted by the band's share of norðanátt hours, so the per-event WRF
results can be integrated into an ANNUAL shielding service (not n=1).

Writes wrf/pilot_events.json: for each event a date, the WRF run window
(12 h, peak in the back third for quasi-steady analysis), the mouth peak
speed, and the climatological weight (fraction of norðanátt hours +
absolute hours/yr).
"""
import json
import glob
import os
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'

NORTH = lambda d: (d >= 330) | (d <= 30)     # noqa: E731  N sector
MIN_SPEED = 10.0                              # norðanátt threshold (m/s)

# Intensity bands by cumulative probability within the norðanátt regime.
# (lo, hi quantile, representative target quantile, label)
BANDS = [
    (0.00, 0.60, 0.50, 'moderate'),
    (0.60, 0.80, 0.75, 'strong'),
    (0.80, 0.93, 0.90, 'severe'),
    (0.93, 0.985, 0.95, 'extreme'),
    (0.985, 1.00, 0.997, 'catastrophic'),
]


def load_records():
    """Every 6-hourly CARRA record as (year,month,day,hour, mouth_speed,
    mouth_dir, akureyri_speed). Dates reconstructed from filename+index
    (124 recs/Jan = 31 days x 4)."""
    rows = []
    for f in sorted(glob.glob(str(DATA / 'winds_*.npz'))):
        m = re.search(r'winds_(\d{4})_(\d{2})', os.path.basename(f))
        yr, mo = int(m[1]), int(m[2])
        z = np.load(f)
        ms, md, aks = z['mouth_speed'], z['mouth_dir'], z['akureyri_speed']
        for i in range(len(ms)):
            rows.append((yr, mo, i // 4 + 1, (i % 4) * 6,
                         float(ms[i]), float(md[i]), float(aks[i])))
    return np.array(rows)


def main():
    r = load_records()
    yr, mo, day, hr, ms, md, aks = (r[:, k] for k in range(7))
    nmask = NORTH(md) & (ms >= MIN_SPEED)
    n_tot = len(ms)
    n_nor = int(nmask.sum())
    hours_per_yr = n_nor / 20 * 6.0           # 20 yr, 6-hourly records
    print(f'norðanátt: {n_nor}/{n_tot} recs ({100*n_nor/n_tot:.1f}%) '
          f'≈ {hours_per_yr:.0f} h/yr')

    nor_speeds = ms[nmask]

    # distinct events: greedy, sorted by speed desc, >=2 calendar days apart
    idx = np.where(nmask)[0]
    order = idx[np.argsort(-ms[idx])]
    absday = lambda i: yr[i] * 366 + mo[i] * 31 + day[i]   # noqa: E731
    picked = []
    for i in order:
        if all(abs(absday(i) - absday(p)) > 2 for p in picked):
            picked.append(i)
    picked = np.array(picked)
    peak_speeds = ms[picked]

    events = []
    for lo, hi, q, label in BANDS:
        target = np.quantile(nor_speeds, q)
        # representative distinct event whose peak is closest to target
        j = picked[np.argmin(np.abs(peak_speeds - target))]
        weight = hi - lo
        # WRF 12-h window kept within one day (00-12Z or 06-18Z) so met
        # staging needs only that day; peak falls in the analysed back half.
        ph = int(hr[j])
        start_hour = 6 if ph >= 14 else 0
        date = f'{int(yr[j])}-{int(mo[j]):02d}-{int(day[j]):02d}'
        # The severe band reuses the already-staged 2022-01-01 norðanátt
        # (mouth ~20 m/s, P93 — equivalent representative, met already in
        # proof1km/, no fetch needed). Others stage fresh under ev_<label>.
        if label == 'severe':
            date = '2022-01-01'
            ph, start_hour = 18, 6
            event_label, stagedir = 'jan2022_p90', 'proof1km'
        else:
            event_label = f'ev_{label}'
            stagedir = event_label
        events.append({
            'band': label,
            'q': q,
            'target_speed': round(float(target), 1),
            'date': date,
            'event_label': event_label,
            'stagedir': stagedir,
            'peak_hour_utc': ph,
            'mouth_peak_ms': round(float(ms[j]), 1),
            'mouth_dir': round(float(md[j])),
            'akureyri_ms': round(float(aks[j]), 1),
            'wrf_start_hour': start_hour,
            'wrf_run_hours': 12,
            'weight': round(weight, 3),
            'hours_per_yr': round(weight * hours_per_yr, 1),
        })

    out = {
        'regime': 'norðanátt (mouth N-sector >=10 m/s)',
        'nordanatt_frac': round(n_nor / n_tot, 4),
        'nordanatt_hours_per_yr': round(hours_per_yr, 1),
        'percentiles': {p: round(float(np.percentile(nor_speeds, int(p[1:]))), 1)
                        for p in ['p50', 'p75', 'p90', 'p95', 'p99']},
        'events': events,
    }
    (HERE / 'pilot_events.json').write_text(json.dumps(out, indent=2,
                                                       ensure_ascii=False))
    print(f'\n{"band":<13}{"date":<12}{"peakUTC":>8}{"mouth":>7}'
          f'{"dir":>5}{"akur":>6}{"start":>6}{"wt":>6}{"h/yr":>7}')
    for e in events:
        print(f'  {e["band"]:<11}{e["date"]:<12}{e["peak_hour_utc"]:>6}Z'
              f'{e["mouth_peak_ms"]:>7}{e["mouth_dir"]:>5}{e["akureyri_ms"]:>6}'
              f'{e["wrf_start_hour"]:>5}Z{e["weight"]:>6}{e["hours_per_yr"]:>7}')
    print(f'\nwrote {HERE / "pilot_events.json"}')


if __name__ == '__main__':
    main()
