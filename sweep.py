#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Windfall scenario sweep: turbine sizes, placements, and counts.

Evaluates a grid of configurations across the design space and
ranks by wind reduction, cost-effectiveness, and energy production.

Usage:
    python sweep.py              # full sweep, top 15 per metric
    python sweep.py --top 25     # show more results
"""

import argparse
import numpy as np

from model import (
    TurbineSpec, FjordGeometry, TurbineRow, WakeParams, ChanneledWakeModel,
    Constriction, STABILITY_PRESETS,
)
import carra


# ── Turbine catalog ───────────────────────────────────────────────

CATALOG = {
    'SG8': TurbineSpec(
        'SG 8.0-167 DD', 8.0, 167, 119, 3.0, 12.0, 25.0, 0.80),
    'HaliadeX': TurbineSpec(
        'Haliade-X 12 MW', 12.0, 220, 140, 3.0, 12.5, 25.0, 0.79),
    'V236': TurbineSpec(
        'V236-15.0 MW', 15.0, 236, 150, 3.0, 12.5, 25.0, 0.78),
    'Future20': TurbineSpec(
        'Future 20 MW', 20.0, 275, 165, 3.0, 13.0, 30.0, 0.77),
}

CAPEX_PER_KW = {
    'SG8':      5500,   # older platform, less scale benefit
    'HaliadeX': 5000,   # current gen, moderate scale
    'V236':     4500,   # latest gen, good scale
    'Future20': 4000,   # next-gen, maximum scale benefit
}

OPEX_PER_KW_YR = 100   # 2030 projection with learning curve
LIFETIME = 30           # design life, extendable
DISCOUNT = 0.04         # Landsvirkjun-style (state-owned, green bonds)
CRF = DISCOUNT * (1 + DISCOUNT) ** LIFETIME / (
    (1 + DISCOUNT) ** LIFETIME - 1)


# ── Fjord / zones ────────────────────────────────────────────────

EYJAFJORDUR = FjordGeometry(
    name='Eyjafjordur',
    control_points=[
        (0, 12_000), (5_000, 11_000), (10_000, 10_000),
        (15_000, 9_500), (20_000, 9_000), (25_000, 8_500),
        (30_000, 6_500), (35_000, 5_000), (40_000, 4_000),
        (45_000, 3_500), (50_000, 3_000), (55_000, 2_500),
        (60_000, 2_000),
    ],
    ridge_height=1000, length=60_000,
    constrictions=[Constriction(25_000, 2_500, 3_000)],
)

AKUREYRI_X = 55_000

ZONES = {
    'A': {'positions_km': [10, 12, 14, 16], 'shipping_m': 2000},
    'B': {'positions_km': [20, 22, 24, 26], 'shipping_m': 2000},
    'C': {'positions_km': [35, 37, 39],     'shipping_m': 1500},
}

ZONE_COMBOS = [
    ('A',), ('B',), ('C',),
    ('A', 'B'), ('A', 'C'), ('B', 'C'),
    ('A', 'B', 'C'),
]


# ── configuration generator ──────────────────────────────────────

def _max_per_row(center_km, rotor_d, shipping_m, spacing_d=5.0):
    w = EYJAFJORDUR.width(center_km * 1000)
    avail = w - shipping_m
    if avail <= rotor_d:
        return 0
    return max(1, int(avail / (spacing_d * rotor_d)) + 1)


def generate_configs():
    """Generate all sensible turbine-placement configurations."""
    configs = []

    for turb_key, turb in CATALOG.items():
        for combo in ZONE_COMBOS:
            zone_info = [(z, ZONES[z]) for z in combo]

            max_rows_per_zone = [len(zi['positions_km']) for _, zi in zone_info]
            n_per_row_per_zone = [
                _max_per_row(
                    zi['positions_km'][0], turb.rotor_diameter,
                    zi['shipping_m'])
                for _, zi in zone_info
            ]

            if any(n == 0 for n in n_per_row_per_zone):
                continue

            for density_label, frac in [('sparse', 0.5),
                                         ('std', 0.75),
                                         ('dense', 1.0)]:
                rows = []
                for (zname, zi), max_r, full_n in zip(
                        zone_info, max_rows_per_zone, n_per_row_per_zone):
                    n_rows = max(1, int(max_r * frac))
                    n_turb = max(1, int(full_n * frac)) if frac < 1 else full_n
                    for pos_km in zi['positions_km'][:n_rows]:
                        rows.append(TurbineRow(
                            pos_km * 1000, n_turb, turb))

                label = '+'.join(combo)
                configs.append({
                    'label': f'{label} {density_label}',
                    'turbine_key': turb_key,
                    'zones': combo,
                    'density': density_label,
                    'rows': rows,
                })

    return configs


# ── evaluation ────────────────────────────────────────────────────

ISK_PER_USD = 138
WHOLESALE_USD_MWH = 36
HH_COUNT = 7500
HH_KWH = 25_000
RETAIL_ISK = 17  # ISK/kWh


def evaluate(config, model, wb_k, wb_A):
    rows = config['rows']
    n_total = sum(r.n_turbines for r in rows)
    cap = sum(r.capacity_mw for r in rows)
    turb_key = config['turbine_key']

    aep = model.aep(rows, weibull_k=wb_k, weibull_A=wb_A)
    capex_m = cap * CAPEX_PER_KW[turb_key] / 1000
    annual_cost = (cap * 1000 * CAPEX_PER_KW[turb_key] * CRF
                   + cap * 1000 * OPEX_PER_KW_YR)
    lcoe = annual_cost / (aep['aep_gwh'] * 1000) if aep['aep_gwh'] > 0 else 9999

    revenue = aep['aep_gwh'] * 1000 * WHOLESALE_USD_MWH
    premium = max(0, annual_cost - revenue)
    surcharge_isk = premium / (HH_COUNT * HH_KWH) * ISK_PER_USD
    hh_kisk = premium / HH_COUNT * ISK_PER_USD / 1000
    bill_pct = surcharge_isk / RETAIL_ISK * 100

    r12 = model.simulate(rows, 12.0, target_x=AKUREYRI_X)
    r20 = model.simulate(rows, 20.0, target_x=AKUREYRI_X)

    # Marginal turbine-induced reduction, on top of the natural fjord
    # baseline (which the calibrated model reproduces). Pressure scales
    # as u^2, both measured against the no-turbine baseline at Akureyri.
    def _du(r):
        return r.get('turbine_reduction_pct', r['reduction_pct'])

    def _dp(r):
        ub = r.get('baseline_u', r['u_in'])
        return round((1 - (r['target_u'] / ub) ** 2) * 100, 1) if ub > 0 \
            else 0.0

    du20 = _du(r20)
    return {
        'label': config['label'],
        'turb': turb_key,
        'turb_name': CATALOG[turb_key].name,
        'zones': config['zones'],
        'n': n_total,
        'n_rows': len(rows),
        'mw': cap,
        'aep': aep['aep_gwh'],
        'cf': aep['cf_pct'],
        'capex_m': round(capex_m),
        'lcoe': round(lcoe),
        'natural_du20': r20.get('natural_reduction_pct', 0.0),
        'du12': _du(r12),
        'dp12': _dp(r12),
        'du20': du20,
        'dp20': _dp(r20),
        'du20_per_bn': round(du20 / (capex_m / 1000), 2)
        if capex_m > 0 else 0,
        'isk_kwh': round(surcharge_isk, 1),
        'hh_kisk': round(hh_kisk),
        'bill_pct': round(bill_pct),
    }


# ── output ────────────────────────────────────────────────────────

W = 100
HDR = (f'  {"#":>2}  {"Config":<16} {"Turb":<8} '
       f'{"MW":>4} {"N":>3} {"du20":>5} {"dP20":>5} '
       f'{"AEP":>5} {"LCOE":>5} '
       f'{"ISK/kWh":>7} {"kISK/hh":>7} {"Bill%":>5}')
SEP = (f'  {"":->2}  {"":->16} {"":->8} '
       f'{"":->4} {"":->3} {"":->5} {"":->5} '
       f'{"":->5} {"":->5} '
       f'{"":->7} {"":->7} {"":->5}')


def _row(i, r):
    return (f'  {i:>2}  {r["label"]:<16} {r["turb"]:<8} '
            f'{r["mw"]:>4.0f} {r["n"]:>3} '
            f'{r["du20"]:>4.1f}% {r["dp20"]:>4.1f}% '
            f'{r["aep"]:>4.0f}G ${r["lcoe"]:>3.0f} '
            f'{r["isk_kwh"]:>6.1f} {r["hh_kisk"]:>6.0f}k '
            f'{r["bill_pct"]:>4.0f}%')


def print_ranked(results, key, title, top, reverse=True, filter_fn=None):
    print()
    print('=' * W)
    print(f'  {title}')
    print('=' * W)
    print(HDR)
    print(SEP)
    pool = [r for r in results if filter_fn(r)] if filter_fn else results
    ranked = sorted(pool, key=lambda r: r[key], reverse=reverse)
    for i, r in enumerate(ranked[:top], 1):
        print(_row(i, r))
    print()


def print_pareto(results):
    """Show Pareto-optimal configs: maximum du20 at each cost level."""
    print()
    print('=' * W)
    print('  PARETO FRONTIER: max wind reduction vs CAPEX')
    print('=' * W)
    print()

    by_cost = sorted(results, key=lambda r: r['capex_m'])
    frontier = []
    best_du = -1
    for r in by_cost:
        if r['du20'] > best_du:
            best_du = r['du20']
            frontier.append(r)

    print(HDR)
    print(SEP)
    for i, r in enumerate(frontier, 1):
        print(_row(i, r))
    print()
    print(f'  {len(frontier)} Pareto-optimal configurations from '
          f'{len(results)} evaluated')
    print()


def print_header(n_configs, clim):
    print()
    print('=' * W)
    print('  WINDFALL: Scenario Sweep')
    print('=' * W)
    print()
    print('  Turbine catalog:')
    for key, t in CATALOG.items():
        print(f'    {key:<12} {t.rated_power_mw:>2.0f} MW  '
              f'rotor {t.rotor_diameter:.0f}m  hub {t.hub_height:.0f}m  '
              f'cut-out {t.cut_out:.0f} m/s  '
              f'CAPEX ${CAPEX_PER_KW[key]:,}/kW')
    print()
    print(f'  Zones: A (10-16 km)  B (20-26 km)  C (35-39 km)')
    print(f'  Target: Akureyri ({AKUREYRI_X / 1000:.0f} km from mouth)')
    print(f'  Configurations: {n_configs}')
    src = clim.get('source', '?')
    n_yr = clim.get('n_years', '')
    print(f'  Wind data: {src}'
          + (f' ({n_yr} years)' if n_yr else ''))
    print()
    print(f'  du12/du20 = TURBINE-induced speed reduction at Akureyri for')
    print(f'              12/20 m/s, marginal on top of natural sheltering')
    print(f'  dP20 = turbine pressure reduction at 20 m/s (force ~ u^2),')
    print(f'         both measured against the no-turbine fjord baseline')


# ── main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--stability', type=str, default='neutral',
                    choices=list(STABILITY_PRESETS.keys()),
                    help='atmospheric stability preset')
    args = ap.parse_args()

    clim = carra.load_climatology() or carra.synthetic_climatology()
    mouth = clim['stations'].get('mouth', {})
    wb_k = mouth.get('weibull_k', 2.0)
    wb_A = mouth.get('weibull_A', 9.6)

    params = STABILITY_PRESETS[args.stability]
    model = ChanneledWakeModel(EYJAFJORDUR, params)

    configs = generate_configs()
    print_header(len(configs), clim)
    print(f'  Stability: {args.stability} '
          f'(L={params.recovery_length/1000:.0f}km, '
          f'f={params.channeling_fraction:.0%})')

    print()
    print(f'  Evaluating {len(configs)} configs...', flush=True)
    results = [evaluate(c, model, wb_k, wb_A) for c in configs]
    print(f'  Done.')

    print_ranked(results, 'du20',
                 'TOP BY WIND REDUCTION AT 20 m/s',
                 args.top)
    print_ranked(results, 'du20_per_bn',
                 'TOP BY WIND REDUCTION PER $B CAPEX (bang for buck)',
                 args.top)
    print_ranked(results, 'bill_pct',
                 'LOWEST HOUSEHOLD BILL IMPACT (with >1% du20)',
                 args.top, reverse=False,
                 filter_fn=lambda r: r['du20'] > 1.0)

    print_pareto(results)


if __name__ == '__main__':
    main()
