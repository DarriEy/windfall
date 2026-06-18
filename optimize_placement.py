#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Placement optimisation: where should the shield cluster go?

The named configurations were hand-designed. Here we search shield-cluster
placement systematically against the right objective — avoided norðanátt
storm damage per unit cost (shielding_value.py) — holding a fixed
outer-fjord generation cluster. Physics intuition: shielding per turbine
grows as the cluster moves toward Akureyri (less wake recovery before the
target), but the channel narrows (fewer turbines fit) and the wind weakens
(worse generation). The search finds where that trade-off lands.

Outputs the Pareto frontier (avoided-damage fraction vs annual premium),
the placement that maximises avoided damage at zero net premium, and a
comparison against the hand-designed SAMSETT / JAFNVÆGI.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import observed
from designs import (
    EYJAFJORDUR, _rows, FUTURE20, SG14_170, V236_175, DESIGNS, rows_of,
    eval_design,
)
from shielding_value import avoided_damage_fraction, annual_premium_usd

OUT = Path('figures')
ISK_USD = 138

# Fixed outer-fjord generation cluster (the SAMSETT generator, ~$144/MWh
# standalone). The search varies only the shield cluster.
GEN = {'turbine': FUTURE20, 'capex_kw': 4000, 'opex_kw': 100,
       'rows': _rows([10, 12, 14], 6, FUTURE20)}


def _max_per_row(center_km, rotor_d, shipping_m=1500, spacing_d=5.0):
    w = EYJAFJORDUR.width(center_km * 1000)
    avail = w - shipping_m
    return max(0, int(avail / (spacing_d * rotor_d)) + 1) if avail > rotor_d else 0


def shield_candidates():
    """Enumerate shield-cluster placements: inner-fjord position sets,
    turbine type, and density (turbines per row capped by channel width)."""
    cands = []
    turbines = [('F20', FUTURE20, 2800), ('SG14', SG14_170, 2800)]
    position_sets = [
        [37, 39], [40, 42], [43, 45], [46, 48], [48, 50],
        [36, 38, 40], [40, 43, 46], [44, 47, 50],
        [42, 45, 48], [38, 41, 44, 47], [44, 46, 48, 50],
    ]
    for tlabel, turb, capex in turbines:
        for frac in (1.0, 0.6):
            for pos in position_sets:
                rows = []
                for km in pos:
                    nmax = _max_per_row(km, turb.rotor_diameter)
                    n = max(1, int(nmax * frac))
                    if nmax == 0:
                        continue
                    rows.append(_rows([km], n, turb)[0])
                if not rows:
                    continue
                cands.append({
                    'label': f'{tlabel} {pos[0]}-{pos[-1]}km '
                             f'x{"dense" if frac==1 else "sparse"}',
                    'turbine': turb, 'capex_kw': capex, 'opex_kw': 60,
                    'rows': rows,
                })
    return cands


def evaluate(shield, raw, nmask):
    info = {'turbine': shield['turbine'], 'capex_kw': 3700,
            'clusters': [GEN, shield]}
    rows = rows_of(info)
    fr = avoided_damage_fraction(rows, raw, nmask, beta=4.0)
    prem_usd, d = annual_premium_usd('adhoc', info)
    prem_misk = prem_usd * ISK_USD / 1e6
    n_shield = sum(r.n_turbines for r in shield['rows'])
    return {
        'label': shield['label'],
        'avoided': fr['avoided_fraction'],
        'premium_misk': prem_misk,
        'lcoe': d['lcoe'], 'cap': d['cap'], 'n_shield': n_shield,
        'dhrs15': fr['hrs15_base'] - fr['hrs15_turb'],
    }


def main():
    raw = observed.load_raw()
    nmask = observed.nordanatt_mask(raw)
    cands = shield_candidates()
    print('=' * 80)
    print('  PLACEMENT OPTIMISATION  (fixed outer generator + variable '
          'inner shield)')
    print('=' * 80)
    print(f'  Searching {len(cands)} shield placements; objective = avoided '
          'storm damage / cost')
    print()
    res = [evaluate(c, raw, nmask) for c in cands]

    # reference hand-designed configs
    refs = {}
    for name in ['C) JAFNVAEGI', 'E) SAMSETT']:
        fr = avoided_damage_fraction(rows_of(DESIGNS[name]), raw, nmask, beta=4)
        prem, d = annual_premium_usd(name, DESIGNS[name])
        refs[name] = {'avoided': fr['avoided_fraction'],
                      'premium_misk': prem * ISK_USD / 1e6, 'lcoe': d['lcoe']}

    # Pareto frontier: max avoided at each premium level (lower premium better)
    res_sorted = sorted(res, key=lambda r: r['premium_misk'])
    frontier, best_av = [], -1
    for r in res_sorted:
        if r['avoided'] > best_av:
            best_av = r['avoided']
            frontier.append(r)

    print('  PARETO FRONTIER (avoided damage vs annual premium):')
    print(f'  {"placement":<26}{"shield n":>9}{"avoided":>9}'
          f'{"premium":>11}{"LCOE":>7}{"Δhrs>15":>9}')
    print(f'  {"":-<26}{"":-<45}')
    for r in frontier:
        print(f'  {r["label"]:<26}{r["n_shield"]:>9}{r["avoided"]*100:>8.0f}%'
              f'{r["premium_misk"]:>9.0f}M{r["lcoe"]:>7.0f}{r["dhrs15"]:>9.0f}')
    print()
    print('  Hand-designed references:')
    for name, r in refs.items():
        print(f'  {name.split(")")[1].strip():<26}{"":>9}{r["avoided"]*100:>8.0f}%'
              f'{r["premium_misk"]:>9.0f}M{r["lcoe"]:>7.0f}')
    print()

    # Best "free" config: highest avoided damage at <= ~0 premium
    free = [r for r in res if r['premium_misk'] <= 5]
    if free:
        best_free = max(free, key=lambda r: r['avoided'])
        print(f'  Best near-zero-premium placement: {best_free["label"]} '
              f'— {best_free["avoided"]*100:.0f}% avoided at '
              f'{best_free["premium_misk"]:.0f} MISK/yr premium '
              f'(LCOE ${best_free["lcoe"]:.0f}).')
    print('  Interpretation: shielding per turbine rises as the cluster moves')
    print('  toward Akureyri, but the narrowing channel caps the turbine')
    print('  count, so the frontier flattens in the inner fjord.')

    _figure(res, refs, frontier)


def _figure(res, refs, frontier):
    fig, ax = plt.subplots(figsize=(9, 6))
    av = np.array([r['avoided'] * 100 for r in res])
    pm = np.array([r['premium_misk'] for r in res])
    ax.scatter(pm, av, s=30, c='#bdc3c7', label='candidate placements',
               zorder=3)
    fx = [r['premium_misk'] for r in frontier]
    fy = [r['avoided'] * 100 for r in frontier]
    ax.plot(fx, fy, '-o', color='#16a085', lw=2, ms=6, zorder=5,
            label='Pareto frontier')
    for name, r, col in [('JAFNVÆGI', refs['C) JAFNVAEGI'], '#27ae60'),
                         ('SAMSETT', refs['E) SAMSETT'], '#2980b9')]:
        ax.plot(r['premium_misk'], r['avoided'] * 100, '*', color=col,
                ms=18, zorder=6, markeredgecolor='k', label=name)
    ax.set_xlabel('Annual shielding premium vs pure generation (MISK/yr)',
                  fontsize=11)
    ax.set_ylabel('Avoided norðanátt storm damage at Akureyri (%, β=4)',
                  fontsize=11)
    ax.set_title('Placement optimisation: shielding value vs cost',
                 fontsize=12.5, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=-20)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'placement_optimisation.png', dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/placement_optimisation.png')


if __name__ == '__main__':
    main()
