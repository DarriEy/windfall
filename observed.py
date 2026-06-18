#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Observed along-fjord wind profiles from raw CARRA records.

This module works directly from the per-month ``winds_YYYY_MM.npz`` files
(7 stations along the Eyjafjordur axis, hub-height speed + direction,
6-hourly, 2003-2022) rather than the pre-aggregated climatology JSON, so
that we can:

  * build the along-fjord speed profile for arbitrary event subsets
    (all conditions vs nordanatt), with a *consistent* event mask defined
    at one anchor station and applied to all stations at the same
    timestamps -- i.e. "during the same northerly events, how does the
    wind decay from mouth to head?";
  * attach honest interannual error bars (between-year std of the mean);
  * expose the raw arrays to the calibration code.

The station order down the fjord is mouth -> outer -> dalvik -> hrisey
-> mid -> inner -> akureyri, at 0, 10, 18, 25, 38, 48, 55 km.
"""

import glob
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

# Station axis (name, x_km) -- matches carra.WAYPOINTS ordering.
STATIONS = [
    ('mouth', 0), ('outer', 10), ('dalvik', 18), ('hrisey', 25),
    ('mid', 38), ('inner', 48), ('akureyri', 55),
]
STATION_NAMES = [s for s, _ in STATIONS]
STATION_X_KM = np.array([x for _, x in STATIONS], dtype=float)

# Nordanatt definition (matches carra._station_stats): northerly sector
# and a hub-height speed strong enough to matter.
NORTH_SECTOR = (330, 30)      # degrees: dir >= 330 OR dir <= 30
NORDANATT_MIN_SPEED = 10.0    # m/s at hub height


def _files():
    fs = sorted(glob.glob(str(DATA_DIR / 'winds_*.npz')))
    if not fs:
        raise FileNotFoundError(
            f'No winds_*.npz files in {DATA_DIR}. Run carra.py first.')
    return fs


def load_raw():
    """Load and concatenate all raw records, preserving cross-station
    time alignment and tagging each record with its calendar year.

    Returns a dict with, per station, ``speed`` and ``dir`` arrays plus a
    shared ``year`` array (same length, aligned across stations).
    """
    fs = _files()
    speed = {s: [] for s in STATION_NAMES}
    direction = {s: [] for s in STATION_NAMES}
    years = []
    for f in fs:
        yr = int(Path(f).stem.split('_')[1])
        d = np.load(f)
        n = d[f'{STATION_NAMES[0]}_speed'].shape[0]
        for s in STATION_NAMES:
            speed[s].append(d[f'{s}_speed'])
            direction[s].append(d[f'{s}_dir'])
        years.append(np.full(n, yr, dtype=int))
    out = {'year': np.concatenate(years)}
    for s in STATION_NAMES:
        out[s] = {
            'speed': np.concatenate(speed[s]).astype(float),
            'dir': np.concatenate(direction[s]).astype(float),
        }
    return out


def is_northerly(direction):
    lo, hi = NORTH_SECTOR
    return (direction >= lo) | (direction <= hi)


def nordanatt_mask(raw, anchor='mouth', min_speed=NORDANATT_MIN_SPEED):
    """Boolean mask of nordanatt events, defined at a single anchor
    station so the *same* timestamps are sampled at every station."""
    a = raw[anchor]
    return is_northerly(a['dir']) & (a['speed'] > min_speed)


def along_fjord_profile(raw, mask=None):
    """Mean hub-height speed at each station (down-fjord order) for the
    selected records, with an interannual error bar (between-year std of
    the annual-mean speed) and the per-station sample size.

    Returns dict with arrays: x_km, mean, interannual_std, n.
    """
    years = raw['year']
    if mask is None:
        mask = np.ones_like(years, dtype=bool)
    yr_vals = np.unique(years)
    means, inter_std, ns = [], [], []
    for s in STATION_NAMES:
        sp = raw[s]['speed']
        sel = sp[mask]
        means.append(float(np.mean(sel)) if sel.size else np.nan)
        ns.append(int(sel.size))
        # Between-year scatter of the annual mean -> honest error bar.
        ym = []
        for y in yr_vals:
            m = mask & (years == y)
            if m.sum() > 0:
                ym.append(float(np.mean(sp[m])))
        inter_std.append(float(np.std(ym)) if len(ym) > 1 else 0.0)
    return {
        'x_km': STATION_X_KM.copy(),
        'mean': np.array(means),
        'interannual_std': np.array(inter_std),
        'n': np.array(ns),
    }


if __name__ == '__main__':
    raw = load_raw()
    n_tot = raw['year'].size
    nmask = nordanatt_mask(raw)
    allp = along_fjord_profile(raw)
    norp = along_fjord_profile(raw, nmask)

    print('=' * 64)
    print('  OBSERVED ALONG-FJORD WIND PROFILE (raw CARRA records)')
    print('=' * 64)
    print(f'  Records: {n_tot:,} (6-hourly, '
          f'{raw["year"].min()}-{raw["year"].max()})')
    print(f'  Nordanatt events (anchored at mouth, '
          f'dir in N sector & >{NORDANATT_MIN_SPEED:.0f} m/s): '
          f'{nmask.sum():,} ({nmask.mean():.1%})')
    print()
    print(f'  {"station":<9}{"x_km":>5}{"all mean":>11}'
          f'{"nordanatt mean":>16}')
    print(f'  {"":-<9}{"":->5}{"":->11}{"":->16}')
    for i, s in enumerate(STATION_NAMES):
        print(f'  {s:<9}{allp["x_km"][i]:>5.0f}'
              f'{allp["mean"][i]:>8.2f} ±{allp["interannual_std"][i]:>3.2f}'
              f'{norp["mean"][i]:>11.2f} ±{norp["interannual_std"][i]:>3.2f}')
    print()
    a0, a1 = allp['mean'][0], allp['mean'][-1]
    n0, n1 = norp['mean'][0], norp['mean'][-1]
    print(f'  Mouth -> Akureyri (all):       {a0:.2f} -> {a1:.2f} m/s'
          f'  ({(1-a1/a0)*100:.0f}% drop)')
    print(f'  Mouth -> Akureyri (nordanatt): {n0:.2f} -> {n1:.2f} m/s'
          f'  ({(1-n1/n0)*100:.0f}% drop)')
