#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Comprehensive re-evaluation of turbine placement, sizing, and hub height.

Uses the full 20-year CARRA spatial wind data and stability-corrected
height extrapolation to optimize all three scenarios.

Key improvements over initial analysis:
1. Local wind speed at each turbine position (not just mouth Weibull)
2. Hub-height sensitivity with stability-dependent shear
3. Turbine catalog with different hub heights
4. Zone-specific capacity factors
"""

import numpy as np
import math
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List

from model import (
    TurbineSpec, FjordGeometry, TurbineRow, WakeParams,
    ChanneledWakeModel, Constriction, STABILITY_PRESETS,
    marginal_reduction,
)
from designs import EYJAFJORDUR, AKUREYRI, rows_of, design_cost
import carra

OUT = Path('figures')

# ── CARRA spatial wind data ──────────────────────────────────────

def load_spatial_wind():
    """Load CARRA 20-year statistics at each waypoint."""
    clim = carra.load_climatology()
    if clim is None:
        return None
    return {wp['name']: clim['stations'][wp['name']]
            for wp in carra.WAYPOINTS
            if wp['name'] in clim['stations']}


def local_weibull(x_km, spatial):
    """Interpolate Weibull A and k at a given fjord position."""
    positions = [(wp['x_km'], wp['name']) for wp in carra.WAYPOINTS
                 if wp['name'] in spatial]
    xs = [p[0] for p in positions]
    As = [spatial[p[1]]['weibull_A'] for p in positions]
    ks = [spatial[p[1]]['weibull_k'] for p in positions]
    A = float(np.interp(x_km, xs, As))
    k = float(np.interp(x_km, xs, ks))
    return k, A


# ── Hub height analysis ──────────────────────────────────────────

Z0 = 0.0005
ALPHA_NEUTRAL = 0.11
ALPHA_STABLE = 0.25

def height_factor(z_hub, stability='neutral'):
    """Height extrapolation factor from 10m."""
    if stability == 'log':
        return math.log(z_hub / Z0) / math.log(10 / Z0)
    alpha = ALPHA_NEUTRAL if stability == 'neutral' else ALPHA_STABLE
    return (z_hub / 10) ** alpha


def adjust_weibull_for_height(k, A, z_hub_new, z_hub_old=150):
    """Scale Weibull A parameter for a different hub height.

    The existing A was computed at z_hub_old using a log profile.
    Re-derive: A_new = A_old × (factor_new / factor_old).
    """
    f_old = math.log(z_hub_old / Z0) / math.log(10 / Z0)
    f_new = math.log(z_hub_new / Z0) / math.log(10 / Z0)
    return k, A * f_new / f_old


# ── Turbine catalog with height variants ─────────────────────────

TURBINES = {
    'V236_120m': TurbineSpec('V236 @ 120m', 15, 236, 120, 3, 12.5, 25, 0.78),
    'V236_150m': TurbineSpec('V236 @ 150m', 15, 236, 150, 3, 12.5, 25, 0.78),
    'V236_175m': TurbineSpec('V236 @ 175m', 15, 236, 175, 3, 12.5, 25, 0.78),
    'SG14_170m': TurbineSpec('SG 14-236 @ 170m', 14, 236, 170, 3, 12, 25, 0.79),
    'F20_150m':  TurbineSpec('Future20 @ 150m', 20, 275, 150, 3, 13, 30, 0.77),
    'F20_175m':  TurbineSpec('Future20 @ 175m', 20, 275, 175, 3, 13, 30, 0.77),
    'F20_200m':  TurbineSpec('Future20 @ 200m', 20, 275, 200, 3, 13, 30, 0.77),
    'HaliX_140m': TurbineSpec('Haliade-X @ 140m', 12, 220, 140, 3, 12.5, 25, 0.79),
}

CAPEX_KW = {
    'V236_120m': 4200, 'V236_150m': 4500, 'V236_175m': 4800,
    'SG14_170m': 4600,
    'F20_150m': 3800, 'F20_175m': 4000, 'F20_200m': 4200,
    'HaliX_140m': 5000,
}

CRF = 0.04 * 1.04 ** 30 / (1.04 ** 30 - 1)
OPEX = 100


# ── Zone definitions ─────────────────────────────────────────────

ZONES = {
    'A': {'rows_km': [10, 12, 14, 16], 'shipping_m': 2000, 'label': 'Outer'},
    'B': {'rows_km': [20, 22, 24],     'shipping_m': 2000, 'label': 'Hrísey'},
    'C': {'rows_km': [36, 39],         'shipping_m': 1500, 'label': 'Inner'},
}


def max_per_row(zone_km, rotor_d, shipping_m, spacing_d=5.0):
    w = EYJAFJORDUR.width(zone_km * 1000)
    avail = w - shipping_m
    if avail <= rotor_d:
        return 0
    return max(1, int(avail / (spacing_d * rotor_d)) + 1)


# ── Site-specific AEP ────────────────────────────────────────────

def site_aep(rows, spatial, hours=8766):
    """AEP using local Weibull at each row position."""
    m = ChanneledWakeModel(EYJAFJORDUR, WakeParams(200, 30_000, 0.7))
    u_bins = np.arange(0.5, 36.0, 1.0)
    total_mwh = 0.0

    for ub in u_bins:
        res = m.simulate(rows, float(ub))
        for rd in res['rows']:
            x_km = rd['x_km']
            k, A = local_weibull(x_km, spatial)
            # Adjust Weibull for this turbine's hub height
            turb = None
            for row in rows:
                if abs(row.x_position / 1000 - x_km) < 0.5:
                    turb = row.turbine
                    break
            if turb:
                k, A = adjust_weibull_for_height(k, A, turb.hub_height)

            pdf = (k / A) * (ub / A) ** (k - 1) * np.exp(-(ub / A) ** k)
            total_mwh += rd['mw_total'] * hours * float(pdf)

    cap = sum(r.capacity_mw for r in rows)
    cf = total_mwh / (cap * hours) if cap > 0 else 0
    return total_mwh / 1000, cf  # GWh, fraction


# ── Configuration builder ────────────────────────────────────────

def build_config(turb_key, zone_combo, density='std'):
    turb = TURBINES[turb_key]
    rows = []
    for z in zone_combo:
        zi = ZONES[z]
        n = max_per_row(zi['rows_km'][0], turb.rotor_diameter, zi['shipping_m'])
        if density == 'sparse':
            n = max(1, n // 2)
        n_rows = len(zi['rows_km'])
        if density == 'sparse':
            n_rows = max(1, n_rows // 2)
        for pos_km in zi['rows_km'][:n_rows]:
            rows.append(TurbineRow(pos_km * 1000, n, turb))
    return rows


# ── Main analysis ────────────────────────────────────────────────

def main():
    spatial = load_spatial_wind()
    if spatial is None:
        print('  No CARRA data — run carra.py first')
        return

    print('=' * 72)
    print('  TURBINE RE-EVALUATION WITH SPATIAL CARRA DATA')
    print('=' * 72)

    # 1. Show spatial wind gradient
    print('\n  CARRA 20-year wind at each zone:')
    print(f'  {"Position":<12} {"Hub mean":>8} {"Weibull k":>9} '
          f'{"Weibull A":>9} {"Nord h/yr":>9}')
    print(f'  {"":->12} {"":->8} {"":->9} {"":->9} {"":->9}')
    for wp in carra.WAYPOINTS:
        if wp['name'] not in spatial:
            continue
        s = spatial[wp['name']]
        print(f'  {wp["name"]:<12} {s["mean_speed_hub"]:>6.1f}ms '
              f'{s["weibull_k"]:>8.2f} {s["weibull_A"]:>8.2f} '
              f'{s.get("nordanatt_hours_yr", 0):>8.0f}')

    # 2. Hub height impact
    print('\n  Hub height effect on mean wind speed (mouth):')
    base_A = spatial['mouth']['weibull_A']
    base_k = spatial['mouth']['weibull_k']
    print(f'  {"Height":>6} {"A (neutral)":>11} {"Mean":>6} '
          f'{"A (stable)":>10} {"Mean":>6} {"CF gain":>7}')
    print(f'  {"":->6} {"":->11} {"":->6} {"":->10} {"":->6} {"":->7}')

    base_mean = base_A * math.gamma(1 + 1/base_k)
    for z in [120, 150, 175, 200]:
        k_n, A_n = adjust_weibull_for_height(base_k, base_A, z)
        mean_n = A_n * math.gamma(1 + 1/k_n)
        # Stable: scale by ratio of power-law factors
        f_stable = (z / 10) ** ALPHA_STABLE
        f_log = math.log(z / Z0) / math.log(10 / Z0)
        A_s = base_A / (math.log(150/Z0)/math.log(10/Z0)) * f_stable
        mean_s = A_s * math.gamma(1 + 1/base_k)
        gain = (mean_n / base_mean - 1) * 100
        print(f'  {z:>4}m  {A_n:>9.2f}   {mean_n:>5.1f}  '
              f'{A_s:>9.2f}  {mean_s:>5.1f}  {gain:>+5.1f}%')

    # 3. Evaluate configurations
    configs = []

    # Original three designs
    from designs import DESIGNS
    for dname, info in DESIGNS.items():
        short = dname.split(')')[1].strip()
        rows = rows_of(info)
        aep_gwh, cf = site_aep(rows, spatial)
        cap, _, _, _, annual = design_cost(dname, info)
        lcoe = annual / (aep_gwh * 1000) if aep_gwh > 0 else 999

        m_s = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
        r20 = m_s.simulate(rows, 20.0, target_x=AKUREYRI)
        du20, dp20 = marginal_reduction(r20)   # turbine-only, vs baseline
        turb = info.get('turbine') or rows[0].turbine

        configs.append({
            'name': f'{short} (original)',
            'turb': turb.name,
            'hub': turb.hub_height,
            'n': sum(r.n_turbines for r in rows),
            'mw': cap, 'aep': aep_gwh, 'cf': cf * 100,
            'lcoe': lcoe, 'du20s': du20, 'dp20s': dp20,
        })

    # New optimized variants
    new_designs = [
        # Taller V236 variants for JAFNVÆGI
        ('JAFNVÆGI @175m', 'V236_175m', ('B', 'C'), 'std'),
        # Future20 at different heights
        ('LOGN @175m', 'F20_175m', ('A', 'B', 'C'), 'std'),
        ('LOGN @200m', 'F20_200m', ('A', 'B', 'C'), 'std'),
        # Smaller turbines, more of them (higher blockage)
        ('JAFNVÆGI HaliX', 'HaliX_140m', ('B', 'C'), 'std'),
        # ORKA with taller turbines
        ('ORKA @175m', 'V236_175m', ('A',), 'sparse'),
        # Mixed: V236 in wide zones, smaller in narrow
        ('B+C dense V236', 'V236_150m', ('B', 'C'), 'std'),
        ('B+C dense @175m', 'V236_175m', ('B', 'C'), 'std'),
        # More rows in Zone B (wider, better wind)
        ('B heavy + C', 'V236_150m', ('B', 'C'), 'std'),
        # Fewer but taller in Zone C only
        ('C only @175m', 'V236_175m', ('C',), 'std'),
        ('C only F20@175', 'F20_175m', ('C',), 'std'),
        # All zones sparse — minimum cost, some shielding
        ('A+B+C sparse', 'V236_150m', ('A', 'B', 'C'), 'sparse'),
        # SG14 — middle ground turbine
        ('JAFNVÆGI SG14', 'SG14_170m', ('B', 'C'), 'std'),
    ]

    for name, turb_key, zones, density in new_designs:
        turb = TURBINES[turb_key]
        rows = build_config(turb_key, zones, density)
        if not rows:
            continue

        aep_gwh, cf = site_aep(rows, spatial)
        cap = sum(r.capacity_mw for r in rows)
        capex_kw = CAPEX_KW[turb_key]
        annual = cap * 1000 * capex_kw * CRF + cap * 1000 * OPEX
        lcoe = annual / (aep_gwh * 1000) if aep_gwh > 0 else 999

        m_s = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
        r20 = m_s.simulate(rows, 20.0, target_x=AKUREYRI)
        du20, dp20 = marginal_reduction(r20)   # turbine-only, vs baseline

        configs.append({
            'name': name,
            'turb': turb.name[:15],
            'hub': turb.hub_height,
            'n': sum(r.n_turbines for r in rows),
            'mw': cap, 'aep': aep_gwh, 'cf': cf * 100,
            'lcoe': lcoe, 'du20s': du20, 'dp20s': dp20,
        })

    # Sort by LCOE
    print('\n  ALL CONFIGURATIONS (sorted by LCOE):')
    print(f'  {"#":>2} {"Config":<22} {"Turbine":<16} {"Hub":>4} '
          f'{"MW":>4} {"N":>3} {"AEP":>5} {"CF":>5} '
          f'{"LCOE":>5} {"du20s":>5} {"dP20s":>5}')
    print(f'  {"":->2} {"":->22} {"":->16} {"":->4} '
          f'{"":->4} {"":->3} {"":->5} {"":->5} '
          f'{"":->5} {"":->5} {"":->5}')

    for i, c in enumerate(sorted(configs, key=lambda x: x['lcoe']), 1):
        print(f'  {i:>2} {c["name"]:<22} {c["turb"]:<16} {c["hub"]:>3}m '
              f'{c["mw"]:>4.0f} {c["n"]:>3} {c["aep"]:>4.0f}G '
              f'{c["cf"]:>4.1f}% ${c["lcoe"]:>3.0f} '
              f'{c["du20s"]:>4.1f}% {c["dp20s"]:>4.1f}%')

    # Sort by wind reduction
    print('\n  TOP BY WIND SHIELDING (stable norðanátt):')
    for i, c in enumerate(sorted(configs, key=lambda x: -x['dp20s'])[:10], 1):
        print(f'  {i:>2} {c["name"]:<22} {c["turb"]:<16} '
              f'dP20={c["dp20s"]:>4.1f}%  LCOE=${c["lcoe"]:.0f}  '
              f'AEP={c["aep"]:.0f}G  CF={c["cf"]:.1f}%')

    # Sort by shielding efficiency (dP20 / LCOE)
    print('\n  TOP BY SHIELDING EFFICIENCY (dP20% per $/MWh):')
    for i, c in enumerate(
            sorted(configs,
                   key=lambda x: -x['dp20s'] / max(x['lcoe'], 1))[:10], 1):
        eff = c['dp20s'] / c['lcoe'] * 100
        print(f'  {i:>2} {c["name"]:<22} '
              f'dP20={c["dp20s"]:>4.1f}%  LCOE=${c["lcoe"]:.0f}  '
              f'efficiency={eff:.2f}')

    # Recommended updated designs
    print('\n' + '=' * 72)
    print('  RECOMMENDED UPDATED DESIGNS')
    print('=' * 72)

    best_energy = min(configs, key=lambda x: x['lcoe'])
    best_shield = max(configs, key=lambda x: x['dp20s'])
    best_balanced = max(configs,
                        key=lambda x: x['dp20s'] / max(x['lcoe'], 1))

    for label, c in [('ORKA (best LCOE)', best_energy),
                      ('LOGN (max shielding)', best_shield),
                      ('JAFNVÆGI (best efficiency)', best_balanced)]:
        print(f'\n  {label}:')
        print(f'    {c["name"]}')
        print(f'    {c["turb"]}, hub {c["hub"]}m')
        print(f'    {c["n"]} turbines, {c["mw"]:.0f} MW')
        print(f'    AEP: {c["aep"]:.0f} GWh, CF: {c["cf"]:.1f}%')
        print(f'    LCOE: ${c["lcoe"]:.0f}/MWh')
        print(f'    Shielding (stable): du20={c["du20s"]:.1f}%, '
              f'dP20={c["dp20s"]:.1f}%')


if __name__ == '__main__':
    main()
