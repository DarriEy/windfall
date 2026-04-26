#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Stability-corrected hub-height extrapolation and its impact on results.

The current analysis uses a neutral log profile (factor 1.27) for ALL
conditions. During stable norðanátt, the power law exponent α increases
to 0.20–0.35, giving hub-height factors of 1.7–2.5. This module:

1. Recomputes the 20-year climatology with stability-dependent shear
2. Shows the impact on AEP, LCOE, and wind reduction estimates
3. Produces comparison figures

Without CARRA temperature data, we use a proxy: norðanátt events
(strong northerly, identified by direction/speed) are classified as
stable with α=0.25. All other conditions use the neutral log profile.
This is conservative — not all norðanátt events are strongly stable,
and some non-norðanátt hours may also be stable.
"""

import numpy as np
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from model import WakeParams, ChanneledWakeModel, STABILITY_PRESETS
from designs import DESIGNS, EYJAFJORDUR, AKUREYRI
import carra

OUT = Path('figures')
DATA_DIR = Path('data')

Z_HUB = 150.0
Z_REF = 10.0
Z0 = 0.0005

FACTOR_LOG = math.log(Z_HUB / Z0) / math.log(Z_REF / Z0)  # 1.274
FACTOR_STABLE = (Z_HUB / Z_REF) ** 0.25  # 1.968
FACTOR_VSTABLE = (Z_HUB / Z_REF) ** 0.30  # 2.153


def recompute_climatology():
    """Recompute the ensemble climatology with stability-dependent shear.

    For each time step:
    - If direction 330-030° AND 10m speed > 8 m/s: use stable α=0.25
    - Otherwise: use neutral log profile
    """
    files = sorted(DATA_DIR.glob('winds_*.npz'))
    if not files:
        print('  No wind data found')
        return None

    accum = {wp['name']: {'speed_neutral': [], 'speed_corrected': [],
                           'direction': []}
             for wp in carra.WAYPOINTS}

    for f in files:
        data = np.load(f)
        for wp in carra.WAYPOINTS:
            name = wp['name']
            sk = f'{name}_speed'
            dk = f'{name}_dir'
            if sk not in data:
                continue

            # Current hub speed (neutral log profile)
            speed_hub_neutral = data[sk]
            direction = data[dk]

            # Back-compute 10m speed
            speed_10m = speed_hub_neutral / FACTOR_LOG

            # Classify: norðanátt proxy for stable
            is_north = (direction >= 330) | (direction <= 30)
            is_strong = speed_10m > 8  # lower threshold for stability proxy
            is_stable = is_north & is_strong

            # Corrected hub speed
            speed_hub_corrected = np.where(
                is_stable,
                speed_10m * FACTOR_STABLE,
                speed_hub_neutral  # keep neutral for non-stable
            )

            accum[name]['speed_neutral'].append(speed_hub_neutral)
            accum[name]['speed_corrected'].append(speed_hub_corrected)
            accum[name]['direction'].append(direction)

    # Compute statistics
    stations_neutral = {}
    stations_corrected = {}
    for wp in carra.WAYPOINTS:
        name = wp['name']
        s_n = np.concatenate(accum[name]['speed_neutral'])
        s_c = np.concatenate(accum[name]['speed_corrected'])
        d = np.concatenate(accum[name]['direction'])

        stations_neutral[name] = carra._station_stats(s_n, d, wp)
        stations_corrected[name] = carra._station_stats(s_c, d, wp)

    return stations_neutral, stations_corrected


def compare_climatologies(neutral, corrected):
    """Print comparison of neutral vs stability-corrected."""
    print('\n  Hub-height climatology: neutral vs stability-corrected')
    print(f'  {"Station":<12} {"Neutral":>10} {"Corrected":>10} '
          f'{"Change":>8} {"Wb k_n":>6} {"Wb k_c":>6} '
          f'{"Wb A_n":>6} {"Wb A_c":>6}')
    print(f'  {"":->12} {"":->10} {"":->10} {"":->8} '
          f'{"":->6} {"":->6} {"":->6} {"":->6}')

    for wp in carra.WAYPOINTS:
        name = wp['name']
        n = neutral[name]
        c = corrected[name]
        pct = (c['mean_speed_hub'] / n['mean_speed_hub'] - 1) * 100
        print(f'  {name:<12} {n["mean_speed_hub"]:>8.2f}ms '
              f'{c["mean_speed_hub"]:>8.2f}ms '
              f'{pct:>+6.1f}% '
              f'{n["weibull_k"]:>6.2f} {c["weibull_k"]:>6.2f} '
              f'{n["weibull_A"]:>6.2f} {c["weibull_A"]:>6.2f}')


def compare_aep(neutral, corrected):
    """Compare AEP and LCOE with both climatologies."""
    print('\n  AEP and LCOE comparison:')
    print(f'  {"Design":<12} {"AEP_n":>7} {"AEP_c":>7} {"Chg":>5} '
          f'{"LCOE_n":>7} {"LCOE_c":>7} {"Chg":>5}')
    print(f'  {"":->12} {"":->7} {"":->7} {"":->5} '
          f'{"":->7} {"":->7} {"":->5}')

    crf = 0.04 * 1.04 ** 30 / (1.04 ** 30 - 1)

    for dname, info in DESIGNS.items():
        short = dname.split(')')[1].strip()
        rows = info['rows']
        cap = sum(r.capacity_mw for r in rows)
        annual_cost = cap * 1000 * info['capex_kw'] * crf + cap * 1000 * 100

        m = ChanneledWakeModel(EYJAFJORDUR, WakeParams(200, 30_000, 0.7))

        mn = neutral['mouth']
        mc = corrected['mouth']

        aep_n = m.aep(rows, weibull_k=mn['weibull_k'],
                      weibull_A=mn['weibull_A'])
        aep_c = m.aep(rows, weibull_k=mc['weibull_k'],
                      weibull_A=mc['weibull_A'])

        lcoe_n = annual_cost / (aep_n['aep_gwh'] * 1000) \
            if aep_n['aep_gwh'] > 0 else 999
        lcoe_c = annual_cost / (aep_c['aep_gwh'] * 1000) \
            if aep_c['aep_gwh'] > 0 else 999

        aep_chg = (aep_c['aep_gwh'] / aep_n['aep_gwh'] - 1) * 100 \
            if aep_n['aep_gwh'] > 0 else 0
        lcoe_chg = (lcoe_c / lcoe_n - 1) * 100 if lcoe_n > 0 else 0

        print(f'  {short:<12} {aep_n["aep_gwh"]:>5.0f}G '
              f'{aep_c["aep_gwh"]:>5.0f}G {aep_chg:>+4.0f}% '
              f'${lcoe_n:>5.0f} ${lcoe_c:>5.0f} {lcoe_chg:>+4.0f}%')


def plot_comparison(neutral, corrected):
    """Plot Weibull distributions: neutral vs corrected."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    u = np.arange(0.5, 35, 0.5)

    for ax, (label, stats) in zip(axes, [('Neutral log profile', neutral),
                                          ('Stability-corrected', corrected)]):
        for wp_name, color in [('mouth', '#c0392b'),
                                ('hrisey', '#e67e22'),
                                ('akureyri', '#2980b9')]:
            s = stats[wp_name]
            k, A = s['weibull_k'], s['weibull_A']
            pdf = (k / A) * (u / A) ** (k - 1) * np.exp(-(u / A) ** k)
            ax.plot(u, pdf, linewidth=2, color=color,
                    label=f'{wp_name} (k={k:.2f}, A={A:.1f})')

        ax.set_xlabel('Hub-height wind speed (m/s)', fontsize=11)
        ax.set_ylabel('Probability density', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 30)

    plt.tight_layout()
    fig.savefig(OUT / 'stability_correction.png', dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f'  Saved {OUT}/stability_correction.png')


def main():
    print('=' * 60)
    print('  STABILITY-CORRECTED HUB-HEIGHT ANALYSIS')
    print('=' * 60)

    print('\n  Extrapolation factors:')
    print(f'    Neutral (log):      {FACTOR_LOG:.3f}  (10m → {10*FACTOR_LOG:.1f} m/s)')
    print(f'    Stable (α=0.25):    {FACTOR_STABLE:.3f}  (10m → {10*FACTOR_STABLE:.1f} m/s)')
    print(f'    Very stable (α=0.30): {FACTOR_VSTABLE:.3f}  (10m → {10*FACTOR_VSTABLE:.1f} m/s)')
    print(f'    Correction applied when: dir 330-030°, 10m speed > 8 m/s')

    neutral, corrected = recompute_climatology()
    compare_climatologies(neutral, corrected)
    compare_aep(neutral, corrected)
    plot_comparison(neutral, corrected)

    # Save corrected climatology
    clim_corrected = {
        'source': 'CARRA_ensemble_stability_corrected',
        'note': 'Stable events (northerly >8m/s at 10m) use α=0.25 power law; '
                'all others use neutral log profile',
        'n_years': 20,
        'hub_height_m': Z_HUB,
        'height_factor_neutral': round(FACTOR_LOG, 3),
        'height_factor_stable': round(FACTOR_STABLE, 3),
        'stations': corrected,
    }
    outfile = DATA_DIR / 'wind_climatology_corrected.json'
    with open(outfile, 'w') as f:
        json.dump(clim_corrected, f, indent=2)
    print(f'\n  Saved corrected climatology: {outfile}')


if __name__ == '__main__':
    main()
