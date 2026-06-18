#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Monte-Carlo uncertainty quantification for the Windfall results.

The headline shielding and LCOE numbers are point estimates that hide
large parametric and structural uncertainty. This module propagates that
uncertainty to P10/P50/P90 bands by sampling, for each draw:

  * superposition  -- 'product' (multiplicative, optimistic) vs 'sos'
                      (Katic sum-of-squares, industry standard). This is
                      a STRUCTURAL uncertainty and turns out to dominate.
  * channeling f   -- U(0.45, 0.92): Froude-derived lower bound (§2.2)
                      to the valley-confinement upper bound. Because the
                      deficit depends on the product Ct·β·f, this band
                      also absorbs thrust-coefficient uncertainty.
  * recovery L     -- U(30, 84) km: neutral preset up to the empirical
                      friction-length ceiling Λ (calibrate.py).
  * baseline Λ     -- N(84.3, 18) km from the calibration fit, clipped.

Reported for the stable (norðanátt-relevant) regime: marginal turbine
wind-speed and pressure reduction at Akureyri, plus AEP and LCOE.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import carra
from model import ChanneledWakeModel, WakeParams
from designs import (
    DESIGNS, EYJAFJORDUR, AKUREYRI, CRF, OPEX_KW, ZONE_OPEX,
)

OUT = Path('figures')
N_DRAWS = 800
SEED = 20260618
U_REF = 20.0   # reference inflow for the reduction bands (m/s)


def _rows_and_cost(name, info):
    """Return (rows, capex_m, annual_cost) for a monolithic or split
    design, mirroring designs.eval_design economics."""
    clusters = info.get('clusters')
    if clusters:
        rows = [r for c in clusters for r in c['rows']]
        capex_m = sum(sum(r.capacity_mw for r in c['rows']) * c['capex_kw']
                      for c in clusters) / 1000
        annual = sum(
            sum(r.capacity_mw for r in c['rows']) * 1000
            * (c['capex_kw'] * CRF + c.get('opex_kw', OPEX_KW))
            for c in clusters)
    else:
        rows = info['rows']
        cap = sum(r.capacity_mw for r in rows)
        opex_kw = info.get('_opex_kw', ZONE_OPEX.get(name, OPEX_KW))
        capex_m = cap * info['capex_kw'] / 1000
        annual = cap * 1000 * (info['capex_kw'] * CRF + opex_kw)
    return rows, capex_m, annual


def sample_params(rng):
    return {
        'superposition': rng.choice(['product', 'sos']),
        'f': float(rng.uniform(0.45, 0.92)),
        'L': float(rng.uniform(30_000, 84_000)),
        'Lambda': float(np.clip(rng.normal(84_300, 18_000),
                                55_000, 120_000)),
        # Effective channeled-layer depth: from the rotor scale (~250 m)
        # to the inversion-capped cold-airmass depth. 200 m is the shallow
        # (optimistic-for-shielding) end. This is the largest single
        # sensitivity (sensitivity.py), so it must be in the budget.
        'H_eff': float(rng.uniform(200.0, 500.0)),
    }


def run_design(name, info, wb_k, wb_A, rng):
    rows, capex_m, annual = _rows_and_cost(name, info)
    is_onshore = info.get('_is_onshore', False)

    # LCOE is a single best estimate: AEP comes from the per-station
    # Weibull resource and is independent of the sampled wake parameters
    # (those move the shielding, not the energy), so it is not in the MC.
    if is_onshore:
        lcoe = annual / (info.get('_aep_gwh', 1175) * 1e6)
    else:
        from designs import station_weibull
        m0 = ChanneledWakeModel(EYJAFJORDUR, WakeParams(200, 30_000, 0.7))
        aep = m0.aep(rows, station_weibull=station_weibull())
        lcoe = annual / (aep['aep_gwh'] * 1000) if aep['aep_gwh'] > 0 else np.nan

    du, dp = [], []
    for _ in range(N_DRAWS):
        if is_onshore:
            du.append(0.0)
            dp.append(0.0)
            continue
        s = sample_params(rng)
        wp = WakeParams(
            effective_height=s['H_eff'], recovery_length=s['L'],
            channeling_fraction=s['f'], baseline_length=s['Lambda'],
            superposition=s['superposition'])
        r = ChanneledWakeModel(EYJAFJORDUR, wp).simulate(
            rows, U_REF, target_x=AKUREYRI)
        ub = r['baseline_u']
        du.append(r['turbine_reduction_pct'])
        dp.append((1 - (r['target_u'] / ub) ** 2) * 100 if ub > 0 else 0)
    return {'du': np.array(du), 'dp': np.array(dp),
            'lcoe': lcoe, 'capex_m': capex_m}


def pcts(a):
    a = a[np.isfinite(a)]
    return np.percentile(a, [10, 50, 90])


def main():
    rng = np.random.default_rng(SEED)
    clim = carra.load_climatology() or carra.synthetic_climatology()
    mouth = clim['stations'].get('mouth', {})
    wb_k = mouth.get('weibull_k', 2.0)
    wb_A = mouth.get('weibull_A', 9.6)

    targets = ['A) LOGN', 'C) JAFNVAEGI', 'E) SAMSETT', 'B) ORKA']
    results = {}
    print('=' * 78)
    print(f'  MONTE-CARLO UNCERTAINTY  ({N_DRAWS} draws, stable regime, '
          f'{U_REF:.0f} m/s inflow)')
    print('=' * 78)
    print('  Sampling: superposition {product|sos}, f~U(0.45,0.92), '
          'L~U(30,84)km, Λ~N(84,18)km, H_eff~U(200,500)m')
    print()
    print(f'  {"design":<12}{"marg Δu20 %":>22}{"marg ΔP20 %":>22}'
          f'{"LCOE":>10}')
    print(f'  {"":<12}{"P10  P50  P90":>22}{"P10  P50  P90":>22}'
          f'{"$/MWh":>10}')
    print(f'  {"":-<12}{"":->22}{"":->22}{"":->10}')
    for name in targets:
        if name not in DESIGNS:
            continue
        r = run_design(name, DESIGNS[name], wb_k, wb_A, rng)
        results[name] = r
        du, dp = pcts(r['du']), pcts(r['dp'])
        short = name.split(')')[1].strip()
        print(f'  {short:<12}'
              f'{du[0]:>6.1f}{du[1]:>7.1f}{du[2]:>7.1f}  '
              f'{dp[0]:>6.1f}{dp[1]:>7.1f}{dp[2]:>7.1f}  '
              f'{r["lcoe"]:>9.0f}')
    print()
    print('  NB: the wake-superposition rule and the channel depth H_eff')
    print('  are the dominant drivers of the wide ΔP band; the model\'s')
    print('  default (product, shallow H_eff) sits near the P90 edge.')

    _figure(results)


def _figure(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {'LOGN': '#c0392b', 'JAFNVAEGI': '#27ae60',
              'SAMSETT': '#16a085', 'ORKA': '#2980b9'}
    names = [n for n in results]
    shorts = [n.split(')')[1].strip() for n in names]

    # Left: marginal pressure-reduction bands (box-ish from percentiles)
    for i, n in enumerate(names):
        short = n.split(')')[1].strip()
        p = np.percentile(results[n]['dp'], [10, 50, 90])
        c = colors.get(short, '#555')
        ax1.plot([i, i], [p[0], p[2]], color=c, lw=3, solid_capstyle='round')
        ax1.plot(i, p[1], 'o', color=c, ms=10, zorder=5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(shorts, fontsize=10)
    ax1.set_ylabel('Marginal pressure reduction at Akureyri (%)\n'
                   'stable, 20 m/s — P10–P90', fontsize=10)
    ax1.set_title('Shielding uncertainty', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y')

    # Right: best-estimate LCOE (per-station resource; deterministic)
    for i, n in enumerate(names):
        short = n.split(')')[1].strip()
        c = colors.get(short, '#555')
        ax2.bar(i, results[n]['lcoe'], color=c, alpha=0.8, width=0.6)
        ax2.annotate(f"${results[n]['lcoe']:.0f}", (i, results[n]['lcoe']),
                     textcoords='offset points', xytext=(0, 3),
                     ha='center', fontsize=9)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(shorts, fontsize=10)
    ax2.set_ylabel('LCOE ($/MWh), per-station resource', fontsize=10)
    ax2.set_title('Best-estimate cost', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Monte-Carlo shielding uncertainty + best-estimate LCOE '
                 f'({N_DRAWS} draws)', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'uncertainty_bands.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/uncertainty_bands.png')


if __name__ == '__main__':
    main()
