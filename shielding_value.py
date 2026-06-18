#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
The VALUE of the shielding — the missing half of the cost-benefit.

The economics modules compute the *cost* of placing turbines where they
shield Akureyri (a small LCOE premium). They never ask what the shielding
is *worth*. This module supplies a transparent, parameterised benefit:

  * Storm-damage proxy. Wind damage rises steeply with speed; insurance
    catastrophe models use a power law D(u) ~ (u - u0)^beta above a
    threshold u0, with beta ~ 3-6 (we sweep it). We apply the turbine
    speed reduction (a function of the synoptic inflow, from the wake
    model) to the OBSERVED Akureyri norðanátt wind record and compute the
    fraction of expected storm damage avoided.
  * Disruptive hours. Hours/year with Akureyri wind above 15 and 20 m/s,
    with and without the turbines.

Because the absolute ISK value of a "storm-damage unit" is uncertain, we
report the dimensionless avoided fraction and the BREAK-EVEN annual
damage cost: the shielding premium is justified once Akureyri's annual
norðanátt damage-plus-disruption cost exceeds this figure. That turns
"shielding is cheap" into a falsifiable cost-benefit statement.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import observed
from model import ChanneledWakeModel, STABILITY_PRESETS
from designs import (DESIGNS, EYJAFJORDUR, AKUREYRI, rows_of, eval_design)

OUT = Path('figures')
ISK_USD = 138
HOURS_PER_RECORD = 6          # CARRA 6-hourly
N_YEARS = 20


def reduction_curve(rows, stability='stable',
                    inflows=(8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30)):
    """Turbine speed-retention factor at Akureyri vs mouth inflow:
    f_red(u_in) = u_with_turbines / u_baseline (both at Akureyri)."""
    m = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS[stability])
    sw = m.sweep_speeds(rows, AKUREYRI, list(inflows))
    u = np.array([r['u_in'] for r in sw])
    fred = np.array([1 - r['turbine_reduction_pct'] / 100 for r in sw])
    return u, fred


def avoided_damage_fraction(rows, raw=None, nmask=None, u0=15.0, beta=4.0,
                            stability='stable'):
    """Fraction of expected norðanátt storm damage avoided at Akureyri,
    using the OBSERVED Akureyri wind record reduced by the modelled
    turbine factor. Also returns disruptive-hour reductions."""
    if raw is None:
        raw = observed.load_raw()
    if nmask is None:
        nmask = observed.nordanatt_mask(raw)
    mouth = raw['mouth']['speed'][nmask]
    ak = raw['akureyri']['speed'][nmask]

    u_in, fred = reduction_curve(rows, stability)
    fac = np.interp(mouth, u_in, fred)        # per-event retention factor
    ak_turb = ak * fac

    def damage(u):
        return np.maximum(u - u0, 0.0) ** beta

    d_base = damage(ak).sum()
    d_turb = damage(ak_turb).sum()
    avoided = 1 - d_turb / d_base if d_base > 0 else 0.0

    def hours_above(arr, thr):
        return float((arr > thr).sum()) * HOURS_PER_RECORD / N_YEARS

    return {
        'avoided_fraction': float(avoided),
        'mean_speed_base': float(ak.mean()),
        'mean_speed_turb': float(ak_turb.mean()),
        'hrs15_base': hours_above(ak, 15), 'hrs15_turb': hours_above(ak_turb, 15),
        'hrs20_base': hours_above(ak, 20), 'hrs20_turb': hours_above(ak_turb, 20),
    }


def annual_premium_usd(name, info, ref='B) ORKA'):
    """Annual $ cost of the shielding configuration above a pure-
    generation reference of equal energy: (LCOE_cfg - LCOE_ref) x AEP."""
    wb_k, wb_A = 1.63, 9.17
    d = eval_design(name, info, EYJAFJORDUR, wb_k, wb_A)
    dref = eval_design(ref, DESIGNS[ref], EYJAFJORDUR, wb_k, wb_A)
    premium_per_mwh = max(0.0, d['lcoe'] - dref['lcoe'])
    return premium_per_mwh * d['aep']['aep_gwh'] * 1000, d  # $/yr, design


def main():
    raw = observed.load_raw()
    nmask = observed.nordanatt_mask(raw)
    print('=' * 76)
    print('  VALUE OF SHIELDING  (norðanátt storm-damage + disruption)')
    print('=' * 76)
    print(f'  Akureyri norðanátt baseline: mean {raw["akureyri"]["speed"][nmask].mean():.1f} m/s, '
          f'{(raw["akureyri"]["speed"][nmask] > 15).sum() * HOURS_PER_RECORD / N_YEARS:.0f} h/yr > 15 m/s')
    print()

    configs = ['C) JAFNVAEGI', 'E) SAMSETT', 'A) LOGN']
    betas = [3.0, 4.0, 5.0]
    rows = {c: rows_of(DESIGNS[c]) for c in configs}

    print('  Avoided storm-damage fraction at Akureyri (threshold 15 m/s):')
    print(f'  {"config":<12}' + ''.join(f'{"β="+str(b):>9}' for b in betas)
          + f'{"Δhrs>15/yr":>12}{"Δhrs>20/yr":>12}')
    print(f'  {"":-<12}{"":-<27}{"":-<24}')
    bc = {}
    for c in configs:
        fr = {b: avoided_damage_fraction(rows[c], raw, nmask, beta=b)
              for b in betas}
        base = fr[4.0]
        print(f'  {c.split(")")[1].strip():<12}'
              + ''.join(f'{fr[b]["avoided_fraction"]*100:>8.0f}%' for b in betas)
              + f'{base["hrs15_base"]-base["hrs15_turb"]:>11.0f} '
              + f'{base["hrs20_base"]-base["hrs20_turb"]:>11.0f}')
        bc[c] = fr

    print()
    print('  Break-even: annual norðanátt damage cost above which the')
    print('  shielding premium pays for itself (β=4, vs ORKA):')
    print(f'  {"config":<12}{"premium MISK/yr":>17}{"avoided %":>11}'
          f'{"break-even MISK/yr":>20}')
    print(f'  {"":-<12}{"":-<48}')
    for c in configs:
        prem_usd, d = annual_premium_usd(c, DESIGNS[c])
        prem_misk = prem_usd * ISK_USD / 1e6
        af = bc[c][4.0]['avoided_fraction']
        be = prem_misk / af if af > 0 else float('inf')
        print(f'  {c.split(")")[1].strip():<12}{prem_misk:>16.0f} '
              f'{af*100:>9.0f}% {be:>18.0f}')
    print()
    print('  Reading: if Akureyri\'s annual norðanátt damage + disruption')
    print('  exceeds the break-even, the shield is net-positive. For a town')
    print('  of 19,000 these are modest figures — the shield clears the bar')
    print('  under plausible storm-cost assumptions, but the estimate is')
    print('  order-of-magnitude (damage exponent and baseline cost are the')
    print('  main unknowns).')

    _figure(bc, raw, nmask)


def _figure(bc, raw, nmask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ak = raw['akureyri']['speed'][nmask]
    # left: damage-weighted wind distribution base vs turbine (JAFNVAEGI)
    rows = rows_of(DESIGNS['C) JAFNVAEGI'])
    u_in, fred = reduction_curve(rows)
    fac = np.interp(raw['mouth']['speed'][nmask], u_in, fred)
    ax1.hist(ak, bins=40, alpha=0.5, color='#c0392b', label='no turbines')
    ax1.hist(ak * fac, bins=40, alpha=0.5, color='#27ae60',
             label='with JAFNVÆGI')
    ax1.axvline(15, color='k', ls='--', lw=1)
    ax1.annotate('damage\nthreshold', xy=(15, 0), xytext=(16, ax1.get_ylim()[1]*0.7),
                 fontsize=8)
    ax1.set_xlabel('Akureyri wind speed during norðanátt (m/s)', fontsize=10)
    ax1.set_ylabel('records (20 yr)', fontsize=10)
    ax1.set_title('Storm-wind distribution at Akureyri', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # right: avoided-damage fraction vs damage exponent
    betas = np.arange(2, 7, 0.5)
    for c, col in [('C) JAFNVAEGI', '#27ae60'), ('E) SAMSETT', '#16a085'),
                   ('A) LOGN', '#c0392b')]:
        rws = rows_of(DESIGNS[c])
        af = [avoided_damage_fraction(rws, raw, nmask, beta=b)['avoided_fraction']
              for b in betas]
        ax2.plot(betas, np.array(af) * 100, '-o', color=col, ms=4,
                 label=c.split(')')[1].strip())
    ax2.set_xlabel('Damage exponent β  (D ∝ (u−15)^β)', fontsize=10)
    ax2.set_ylabel('Avoided storm-damage fraction (%)', fontsize=10)
    ax2.set_title('Shielding benefit vs damage steepness', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle('Value of shielding: avoided norðanátt storm damage at '
                 'Akureyri', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'shielding_value.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/shielding_value.png')


if __name__ == '__main__':
    main()
