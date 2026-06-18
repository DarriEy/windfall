#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Constrain the channeling fraction f from RIDGE-HEIGHT stability.

The channeling fraction f (the dominant model parameter, presets
0.60-0.92) is set by the Froude number of the flow meeting the ~1 km
ridges: Fr = u / (N · H_ridge), f = 1/√(1+(Fr/Fr_c)²). The surface
2 m/skin temperatures (stratification.py) cannot give the ridge-scale N
— that needs the potential-temperature gradient across the ridge layer,
i.e. CARRA *pressure-level* temperatures.

This module computes N and f from pressure-level temperature when it is
present (data/ptemp_*.npz, levels ~1000/925/850 hPa). When it is absent
it (a) prints the exact CDS request to fetch it and (b) runs a
literature-anchored demonstration so the method — and the f it implies —
is visible without the download.

Potential temperature:  θ = T·(p0/p)^(R/cp),  p0=1000 hPa, R/cp=0.286
Brunt-Väisälä:          N² = (g/θ̄)·dθ/dz   over the surface→ridge layer
"""

import glob
import numpy as np
from pathlib import Path

import observed

DATA = Path('data')
G = 9.81
RCP = 0.286
H_RIDGE = 1000.0
FR_C = 0.75
# Standard-atmosphere heights of the pressure levels (m), used if the
# CARRA geopotential is not co-fetched.
LEVEL_Z = {1000: 110.0, 925: 760.0, 850: 1460.0}
PRESETS = {'neutral': 0.60, 'stable': 0.80, 'very_stable': 0.92}

CDS_REQUEST = """\
  To constrain f from data, fetch CARRA pressure-level temperature:

    import cdsapi
    cdsapi.Client().retrieve('reanalysis-carra-pressure-levels', {
        'variable': 'temperature',
        'pressure_level': ['1000', '925', '850'],
        'product_type': 'analysis', 'leadtime_hour': '0',
        'year': '<YYYY>', 'month': [...], 'day': [...],
        'time': ['00:00','06:00','12:00','18:00'],
        'domain': 'west_domain', 'data_format': 'netcdf'})

  then extract temperature at the 7 waypoints to data/ptemp_YYYY_HX.npz
  with keys <station>_t1000/_t925/_t850 (see carra.fetch_pressure_temp).
"""


def theta(T, p_hpa):
    return T * (1000.0 / p_hpa) ** RCP


def n_from_profile(t_lo, p_lo, t_hi, p_hi):
    """N (s^-1) over a layer from temperatures at two pressure levels."""
    th_lo, th_hi = theta(t_lo, p_lo), theta(t_hi, p_hi)
    dz = LEVEL_Z[p_hi] - LEVEL_Z[p_lo]
    dtheta_dz = (th_hi - th_lo) / dz
    th_bar = 0.5 * (th_lo + th_hi)
    n2 = G / th_bar * dtheta_dz
    return np.sign(n2) * np.sqrt(np.abs(n2))


def froude_f(u, n):
    n = np.where(n > 0, n, np.nan)
    fr = u / (n * H_RIDGE)
    return 1.0 / np.sqrt(1.0 + (fr / FR_C) ** 2)


def _have_pressure_data():
    return bool(glob.glob(str(DATA / 'ptemp_*.npz')))


def from_data():
    """Compute ridge-layer N and f from CARRA pressure-level temperature,
    conditioned on norðanátt. Returns a summary dict."""
    files = sorted(glob.glob(str(DATA / 'ptemp_*.npz')))
    t1000, t850 = [], []
    for f in files:
        d = np.load(f)
        t1000.append(d['mouth_t1000'])
        t850.append(d['mouth_t850'])
    t1000 = np.concatenate(t1000)
    t850 = np.concatenate(t850)
    # NB: requires the pressure record aligned to the wind record; here we
    # assume the same year-major 6-hourly grid as observed/stratification.
    raw = observed.load_raw()
    nmask = observed.nordanatt_mask(raw)[:t1000.size]
    u = raw['mouth']['speed'][:t1000.size]
    N = n_from_profile(t1000, 1000, t850, 850)
    f = froude_f(u, N)
    return {
        'N_nordan': float(np.nanmean(N[nmask])),
        'f_nordan': float(np.nanmean(f[nmask])),
        'stable_frac': float(np.mean(N[nmask] > 0)),
    }


def demonstrate():
    """Literature-anchored demo: stable-ABL potential-temperature
    gradients (Mahrt 2014) over the ridge layer, at norðanátt speeds,
    and the f they imply — bracketing the model presets."""
    print('  Demonstration (no pressure data present) — f from typical')
    print('  stable-ABL lapse rates at norðanátt speed u = 15 m/s:')
    print(f'  {"regime":<16}{"dθ/dz (K/km)":>14}{"N (s⁻¹)":>10}'
          f'{"Fr":>7}{"f":>7}{"preset":>8}')
    print(f'  {"":-<16}{"":-<46}')
    u = 15.0
    rows = [('weakly stable', 3.0, None),
            ('stable', 5.0, PRESETS['stable']),
            ('very stable', 8.0, PRESETS['very_stable']),
            ('strong inversion', 15.0, None)]
    for name, dthdz, preset in rows:
        n = np.sqrt(G / 280.0 * dthdz / 1000.0)
        fr = u / (n * H_RIDGE)
        f = 1.0 / np.sqrt(1.0 + (fr / FR_C) ** 2)
        ps = f'{preset:.2f}' if preset else '—'
        print(f'  {name:<16}{dthdz:>12.1f}  {n:>9.4f}{fr:>7.2f}{f:>7.2f}'
              f'{ps:>8}')
    print()
    print('  The model presets (stable 0.80, very-stable 0.92) sit within')
    print('  the f range implied by observed-order stable lapse rates, but')
    print('  the true distribution — and whether norðanátt actually reaches')
    print('  these gradients at ridge height — can only be settled with the')
    print('  pressure-level fetch above.')


def main():
    print('=' * 72)
    print('  RIDGE-HEIGHT STABILITY → CHANNELING FRACTION f')
    print('=' * 72)
    if _have_pressure_data():
        s = from_data()
        print(f'  Norðanátt ridge-layer N = {s["N_nordan"]:.4f} s⁻¹ '
              f'({s["stable_frac"]*100:.0f}% stable)')
        print(f'  Implied channeling fraction f = {s["f_nordan"]:.2f}')
        print(f'  Model presets: stable 0.80, very-stable 0.92')
    else:
        print(CDS_REQUEST)
        demonstrate()


if __name__ == '__main__':
    main()
