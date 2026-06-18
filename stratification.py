#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Near-surface static stability from CARRA temperatures, and what it
implies for the Froude-number channeling fraction.

The wake model classifies stability by a wind-speed proxy. Here we
replace that proxy with an *observed* thermal stability derived from the
CARRA 2 m air temperature and skin (surface) temperature, time-aligned
with the wind record, and conditioned on norðanátt events.

    Brunt-Vaisala:  N^2 = (g / theta_bar) * d(theta)/dz
    theta(z) = T(z) + Gamma_d * z      (Gamma_d = 9.8 K/km)
    using z = 2 m (air) and z = 0 (skin); theta_bar in Kelvin.

IMPORTANT LIMITATION: CARRA single levels give only a 2 m / surface
pair, so this N is a *surface-layer* stability. The channeling fraction
f in the Froude analysis is governed by the stratification the flow
meets at ridge height (~1 km), which this cannot measure. We therefore
report the surface N as a diagnostic and an honest check on the
stability assumption — NOT as a replacement for the ridge-scale N (which
needs CARRA pressure-level data, identified as future work). The result
is itself informative: over the warm fjord, norðanátt is a cold-air-
advection regime with an UNSTABLE surface layer beneath the stable
airmass aloft, exactly the cold-outbreak structure the wake model
assumes.
"""

import glob
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import observed

OUT = Path('figures')
DATA = Path('data')
G = 9.81
GAMMA_D = 0.0098          # K/m dry adiabatic lapse rate
Z_AIR = 2.0               # m (2 m temperature)
H_RIDGE = 1000.0          # m (Eyjafjordur ridge height)
FR_C = 0.75               # critical Froude for continuous ridges (paper Eq. 4)
STATIONS = observed.STATION_NAMES


def _temp_by_year():
    """Half-year temp files grouped by year, concatenated H1 then H2."""
    by_year = {}
    for f in sorted(glob.glob(str(DATA / 'temp_*_H*.npz'))):
        yr = int(Path(f).stem.split('_')[1])
        by_year.setdefault(yr, []).append(np.load(f))
    return by_year


def load_aligned():
    """Wind (speed/dir) and temperature (t2m/skt) per station, time-
    aligned. Years are kept only when the wind and temperature record
    lengths match (same 6-hourly grid)."""
    raw = observed.load_raw()
    years = raw['year']
    temp_by_year = _temp_by_year()

    out = {s: {'speed': [], 'dir': [], 't2m': [], 'skt': []}
           for s in STATIONS}
    used = []
    for yr in np.unique(years):
        if yr not in temp_by_year:
            continue
        sel = years == yr
        n_wind = int(sel.sum())
        parts = temp_by_year[yr]
        n_temp = sum(p['mouth_t2m'].shape[0] for p in parts)
        if n_wind != n_temp:
            continue
        for s in STATIONS:
            out[s]['speed'].append(raw[s]['speed'][sel])
            out[s]['dir'].append(raw[s]['dir'][sel])
            out[s]['t2m'].append(np.concatenate(
                [p[f'{s}_t2m'] for p in parts]))
            out[s]['skt'].append(np.concatenate(
                [p[f'{s}_skt'] for p in parts]))
        used.append(int(yr))
    for s in STATIONS:
        for k in out[s]:
            out[s][k] = np.concatenate(out[s][k]).astype(float)
    return out, used


def brunt_vaisala(t2m, skt):
    """Surface-layer N (s^-1); negative where the layer is unstable
    (returned as -sqrt(-N^2) so the sign is visible)."""
    theta_air = t2m + GAMMA_D * Z_AIR
    theta_sfc = skt
    theta_bar = 0.5 * (t2m + skt)            # Kelvin
    dtheta_dz = (theta_air - theta_sfc) / Z_AIR
    n2 = G / theta_bar * dtheta_dz
    return np.sign(n2) * np.sqrt(np.abs(n2))


def froude_f(speed, n):
    """Channeling fraction from Froude number (paper Eq. 4). Unstable or
    neutral layers (n<=0) give Fr->inf, f->0."""
    f = np.zeros_like(speed)
    stable = n > 0
    fr = np.where(stable, speed / (np.maximum(n, 1e-9) * H_RIDGE), np.inf)
    f[stable] = 1.0 / np.sqrt(1.0 + (fr[stable] / FR_C) ** 2)
    return f


def main():
    data, years = load_aligned()
    mouth, head = data['mouth'], data['akureyri']

    # nordanatt defined at the mouth, applied to the aligned record
    nmask = (observed.is_northerly(mouth['dir'])
             & (mouth['speed'] > observed.NORDANATT_MIN_SPEED))

    print('=' * 74)
    print('  NEAR-SURFACE STABILITY FROM CARRA TEMPERATURES')
    print('=' * 74)
    print(f'  Aligned years: {years[0]}-{years[-1]} '
          f'({len(years)} yr, {mouth["speed"].size:,} records)')
    print(f'  Nordanatt (mouth): {nmask.sum():,} ({nmask.mean():.1%})')
    print()
    print(f'  {"station":<10}{"subset":<11}{"mean dT":>9}{"N (s^-1)":>11}'
          f'{"% stable":>10}{"Froude f":>10}')
    print(f'  {"":-<10}{"":-<11}{"":->9}{"":->11}{"":->10}{"":->10}')
    rows_for_fig = {}
    for label, st in [('mouth (sea)', mouth), ('akureyri (head)', head)]:
        name = label.split()[0]
        dT = st['t2m'] - st['skt']
        N = brunt_vaisala(st['t2m'], st['skt'])
        f = froude_f(st['speed'], N)
        for sub, m in [('all', np.ones_like(nmask)), ('nordanatt', nmask)]:
            stable_frac = float(np.mean(N[m] > 0))
            # mean f over events where the surface layer is stable
            fm = f[m & (N > 0)]
            f_mean = float(np.mean(fm)) if fm.size else 0.0
            print(f'  {name:<10}{sub:<11}{np.mean(dT[m]):>+8.2f}'
                  f'{np.mean(N[m]):>11.4f}{stable_frac*100:>9.0f}%'
                  f'{f_mean:>10.2f}')
            rows_for_fig[(name, sub)] = (dT[m], N[m])
        print()

    print('  Interpretation:')
    print('  - Over the fjord (mouth), norðanátt surface layers are')
    print('    predominantly UNSTABLE (cold air over warm sea) — the')
    print('    cold-air-advection signature the model assumes, but the')
    print('    OPPOSITE sign of stability at the surface vs aloft.')
    print('  - The Froude f implied by the *surface* N is therefore not a')
    print('    valid estimate of ridge-scale channeling; the stable layer')
    print('    that channels the flow sits aloft and needs CARRA pressure')
    print('    levels (future work). The model\'s f presets (0.60-0.92)')
    print('    remain assumptions, now bracketed by this thermal evidence.')

    _figure(mouth, head, nmask)


def _figure(mouth, head, nmask):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, st, title in [(axes[0], mouth, 'Mouth (over sea)'),
                          (axes[1], head, 'Akureyri (fjord head)')]:
        N = brunt_vaisala(st['t2m'], st['skt'])
        ax.hist(N[~nmask], bins=60, density=True, alpha=0.5,
                color='#95a5a6', label='all conditions')
        ax.hist(N[nmask], bins=60, density=True, alpha=0.6,
                color='#c0392b', label='norðanátt')
        ax.axvline(0, color='black', lw=1)
        ax.annotate('unstable', xy=(ax.get_xlim()[0], 0), xytext=(0.05, 0.92),
                    textcoords='axes fraction', fontsize=8, color='#2980b9')
        ax.annotate('stable', xy=(0, 0), xytext=(0.78, 0.92),
                    textcoords='axes fraction', fontsize=8, color='#e67e22')
        ax.set_xlabel('Surface-layer Brunt–Väisälä N (s⁻¹)\n'
                      '(<0 = unstable)', fontsize=10)
        ax.set_ylabel('density', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    fig.suptitle('CARRA near-surface stability: norðanátt is surface-'
                 'unstable over the warm fjord (cold-air advection)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'stratification.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/stratification.png')


if __name__ == '__main__':
    main()
