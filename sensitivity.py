#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
One-at-a-time sensitivity of the headline shielding to the structural
assumptions the Monte-Carlo (uncertainty.py) does not vary: effective
channel height H_eff, channeling fraction f, wake-recovery length L,
baseline decay length Λ, the wake-superposition rule, and the fjord
width profile. Produces a tornado diagram around a central case
(JAFNVÆGI, stable, 20 m/s inflow), so the reader can see which
assumptions the result actually hinges on.
"""

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import replace

from model import ChanneledWakeModel, WakeParams, marginal_reduction
from designs import DESIGNS, EYJAFJORDUR, AKUREYRI, rows_of

OUT = Path('figures')
ROWS = rows_of(DESIGNS['C) JAFNVAEGI'])
U_IN = 20.0

# Central case = the 'stable' preset.
CENTRAL = dict(effective_height=200.0, channeling_fraction=0.80,
               recovery_length=55_000.0, baseline_length=84_300.0,
               baseline_anchor=10_000.0, superposition='product')


def _scaled_fjord(scale):
    f = copy.deepcopy(EYJAFJORDUR)
    f.control_points = [(x, w * scale) for x, w in f.control_points]
    return f


def dp20(fjord=EYJAFJORDUR, **overrides):
    params = WakeParams(**{**CENTRAL, **overrides})
    m = ChanneledWakeModel(fjord, params)
    r = m.simulate(ROWS, U_IN, target_x=AKUREYRI)
    return marginal_reduction(r)[1]    # pressure reduction %


def main():
    base = dp20()
    print('=' * 70)
    print('  SENSITIVITY OF MARGINAL ΔP20 (JAFNVÆGI, stable, 20 m/s)')
    print('=' * 70)
    print(f'  Central case ΔP20 = {base:.1f}%')
    print()

    # (label, low_kwargs, high_kwargs, low_desc, high_desc)
    cases = [
        ('H_eff (channel height)',
         dict(effective_height=150), dict(effective_height=300),
         '150 m', '300 m'),
        ('Channeling fraction f',
         dict(channeling_fraction=0.60), dict(channeling_fraction=0.92),
         '0.60', '0.92'),
        ('Wake-recovery length L',
         dict(recovery_length=30_000), dict(recovery_length=84_000),
         '30 km', '84 km'),
        ('Baseline decay Λ',
         dict(baseline_length=66_000), dict(baseline_length=120_000),
         '66 km', '120 km'),
        ('Superposition',
         dict(superposition='sos'), dict(superposition='product'),
         'SoS', 'product'),
    ]
    results = []
    print(f'  {"parameter":<26}{"low":>16}{"high":>16}{"swing":>9}')
    print(f'  {"":-<26}{"":-<41}')
    for label, lo, hi, lod, hid in cases:
        v_lo, v_hi = dp20(**lo), dp20(**hi)
        swing = abs(v_hi - v_lo)
        results.append((label, v_lo, v_hi, swing))
        print(f'  {label:<26}{f"{lod}: {v_lo:.1f}%":>16}'
              f'{f"{hid}: {v_hi:.1f}%":>16}{swing:>8.1f}')

    # fjord width (separate: needs a scaled geometry)
    v_lo = dp20(fjord=_scaled_fjord(0.8))
    v_hi = dp20(fjord=_scaled_fjord(1.2))
    results.append(('Fjord width ±20%', v_lo, v_hi, abs(v_hi - v_lo)))
    print(f'  {"Fjord width ±20%":<26}{f"-20%: {v_lo:.1f}%":>16}'
          f'{f"+20%: {v_hi:.1f}%":>16}{abs(v_hi-v_lo):>8.1f}')
    print()

    results.sort(key=lambda r: r[3])
    print(f'  Ranked by impact: '
          + ', '.join(f'{r[0].split("(")[0].strip()} ({r[3]:.0f}pp)'
                      for r in reversed(results)))
    print('  KEY POINT: the effective channel height H_eff — an ASSUMED')
    print('  parameter (200 m) that the §4.3 Monte-Carlo never varied — is')
    print('  the single largest sensitivity (it scales blockage β directly),')
    print('  alongside the superposition rule and fjord width. The baseline')
    print('  decay Λ barely matters (it cancels in the marginal ratio).')
    print('  Implication: H_eff should be added to the uncertainty budget')
    print('  and justified physically (boundary-layer / inversion depth).')

    _figure(base, results)


def _figure(base, results):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [r[0] for r in results]
    y = np.arange(len(labels))
    for i, (label, lo, hi, sw) in enumerate(results):
        ax.plot([lo, hi], [i, i], '-', color='#bdc3c7', lw=8,
                solid_capstyle='round', zorder=2)
        ax.plot(lo, i, 'o', color='#2980b9', ms=9, zorder=3)
        ax.plot(hi, i, 'o', color='#c0392b', ms=9, zorder=3)
    ax.axvline(base, color='k', ls='--', lw=1.5, label=f'central {base:.1f}%')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Marginal ΔP20 at Akureyri (%) — JAFNVÆGI, stable, 20 m/s',
                  fontsize=11)
    ax.set_title('Sensitivity to structural assumptions (one-at-a-time)',
                 fontsize=12.5, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/sensitivity.png')


if __name__ == '__main__':
    main()
