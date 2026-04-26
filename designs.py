#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Three optimized wind farm designs for Eyjafjordur.

  A) LOGN    — maximize nordanatt wind reduction at Akureyri
  B) ORKA    — maximize electricity production, minimize LCOE
  C) JAFNVAEGI — balanced: best wind reduction per cost

Usage:
    python designs.py
    python designs.py --synthetic
"""

import argparse
import numpy as np

from model import (
    TurbineSpec, FjordGeometry, TurbineRow, WakeParams,
    ChanneledWakeModel, Constriction, STABILITY_PRESETS,
)
import carra


# ── Shared constants ──────────────────────────────────────────────

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

AKUREYRI = 55_000

# Turbines — updated with optimized hub heights
FUTURE20 = TurbineSpec('Future 20 MW', 20, 275, 165, 3, 13, 30, 0.77)
V236 = TurbineSpec('V236-15.0 MW', 15, 236, 150, 3, 12.5, 25, 0.78)
V236_175 = TurbineSpec('V236 @ 175m', 15, 236, 175, 3, 12.5, 25, 0.78)
SG14_170 = TurbineSpec('SG 14-236 DD', 14, 236, 170, 3, 12.0, 25, 0.79)
SG8_119 = TurbineSpec('SG 8.0-167 DD', 8, 167, 119, 3, 12.0, 25, 0.80)

# Financing (Landsvirkjun-style)
DISCOUNT = 0.04
LIFETIME = 30
CRF = DISCOUNT * (1 + DISCOUNT) ** LIFETIME / ((1 + DISCOUNT) ** LIFETIME - 1)
OPEX_KW = 100

# Akureyri
ISK_USD = 138
HH = 7500
HH_KWH = 25_000
RETAIL_ISK = 17
BILL_ISK = RETAIL_ISK * HH_KWH

REVENUE_SCENARIOS = [
    ('Wholesale',    36),
    ('PPA',          60),
    ('Data center',  80),
    ('EU export',   120),
    ('Green H2',    150),
]


# ── Three designs ────────────────────────────────────────────────

def _rows(positions_km, n_per_row, turbine):
    if isinstance(n_per_row, int):
        n_per_row = [n_per_row] * len(positions_km)
    return [TurbineRow(x * 1000, n, turbine)
            for x, n in zip(positions_km, n_per_row)]


DESIGNS = {
    'A) LOGN': {
        'subtitle': 'Maximum calm weather at Akureyri',
        'rationale': [
            'All three zones for compound wake across 29 km of fjord.',
            'Future20 at 165m hub: 30 m/s cut-out keeps working in',
            'extreme events where standard turbines shut down.',
            'Dense packing maximizes blockage ratio at each row.',
            'Zone-specific foundations: floating outer, nearshore mid,',
            'coastal inner. Blended CAPEX $3,867/kW.',
        ],
        'turbine': FUTURE20,
        'capex_kw': 3867,
        'rows': (
            _rows([10, 12, 14, 16], 6, FUTURE20) +     # Zone A: wide
            _rows([20, 22, 24, 26], [5, 4, 4, 3], FUTURE20) +  # Zone B
            _rows([35, 37, 39], 3, FUTURE20)            # Zone C
        ),
    },
    'B) ORKA': {
        'subtitle': 'Maximum energy production',
        'rationale': [
            'Zone A only: strongest wind (8.7 m/s hub at outer fjord),',
            'widest section, minimal wake interaction.',
            'V236 at 175m hub: taller tower captures stable-shear bonus.',
            'Sparse layout (2 rows, 4 km apart) minimizes wake losses.',
            'Deep water requires floating foundations ($4,500/kW).',
        ],
        'turbine': V236_175,
        'capex_kw': 4500,
        'rows': _rows([10, 14], 7, V236_175),
    },
    'C) JAFNVAEGI': {
        'subtitle': 'Balanced: cost-effective wind shield',
        'rationale': [
            'Zones B+C: Hrisey as O&M hub, Zone C close to Akureyri.',
            'SG 14-236 DD at 170m hub: same 236m rotor as V236 but at',
            '14 MW rating — stays at rated power (max Ct) longer before',
            'pitching, giving more shielding per turbine-hour.',
            'Zone-specific foundations: floating at Hrisey (B), fixed-',
            'bottom nearshore and coastal at inner fjord (C).',
            'Blended CAPEX $3,500/kW — cheaper than pure-offshore ORKA',
            'because shallower inner zones use cheaper foundations.',
        ],
        'turbine': SG14_170,
        'capex_kw': 3500,
        'rows': (
            _rows([20, 22, 24], [5, 5, 4], SG14_170) +   # Zone B: 14 turb
            _rows([36, 38, 40], [4, 3, 3], SG14_170)      # Zone C: 10 turb
        ),
    },
    'D) SULUR': {
        'subtitle': 'Onshore mountain: power only, no shielding',
        'rationale': [
            'Equivalent generation on Sulur ridgeline (1213m), the',
            'iconic mountain directly above Akureyri, visible from',
            'every window in town.',
            'Onshore CAPEX is 2-3x cheaper. LCOE is unbeatable.',
            'But: zero wind shielding, major visual impact on the',
            'landscape that defines Akureyri, alpine road construction,',
            'sub-arctic ecosystem disruption.',
            'The question is not "which is cheaper per kWh" but',
            '"what do you get for what you pay?"',
        ],
        'turbine': SG8_119,
        'capex_kw': 2200,
        'rows': _rows(
            # 30 turbines along the Sulur ridgeline, ~300m spacing
            # This is a conceptual placement, not a real site plan
            list(range(30)), 1, SG8_119),
        '_is_onshore': True,
        '_mountain': 'Súlur',
        '_opex_kw': 50,
    },
}


# ── Evaluation ───────────────────────────────────────────────────

ZONE_OPEX = {
    'A) LOGN': 85,       # blended: floating + nearshore + coastal
    'B) ORKA': 100,      # all floating
    'C) JAFNVAEGI': 75,  # blended: floating + nearshore + coastal
}


def eval_design(name, info, fjord, wb_k, wb_A):
    rows = info['rows']
    turb = info['turbine']
    capex_kw = info['capex_kw']
    cap = sum(r.capacity_mw for r in rows)
    n = sum(r.n_turbines for r in rows)
    n_rows = len(rows)

    opex_kw = info.get('_opex_kw', ZONE_OPEX.get(name, OPEX_KW))
    annual_cost = cap * 1000 * capex_kw * CRF + cap * 1000 * opex_kw

    is_onshore = info.get('_is_onshore', False)

    if is_onshore:
        # Mountain wind resource: stronger and steadier than fjord
        # Sulur ridgeline at ~900m elevation: mean ~11 m/s, k~2.3
        mountain_k, mountain_A = 2.3, 12.5
        aep_gwh = 0.0
        u_bins = np.arange(0.5, 36.0, 1.0)
        pdf = ((mountain_k / mountain_A)
               * (u_bins / mountain_A) ** (mountain_k - 1)
               * np.exp(-(u_bins / mountain_A) ** mountain_k))
        for ub, p in zip(u_bins, pdf):
            if p < 1e-8 or ub < 2:
                continue
            pw = float(turb.power(np.array([ub]))[0])
            aep_gwh += pw * n * 8766 * float(p) / 1000
        aep = {'aep_gwh': round(aep_gwh, 1),
               'capacity_mw': cap,
               'cf_pct': round(aep_gwh * 1000 / (cap * 8766) * 100, 1)}
    else:
        m_aep = ChanneledWakeModel(fjord, WakeParams(
            effective_height=200, recovery_length=30_000,
            channeling_fraction=0.7))
        aep = m_aep.aep(rows, weibull_k=wb_k, weibull_A=wb_A)
    lcoe = annual_cost / (aep['aep_gwh'] * 1000) if aep['aep_gwh'] > 0 else 9999

    # Wind reduction under each stability regime
    speeds = [10, 12, 16, 20, 25]
    if turb.cut_out >= 30:
        speeds.append(30)

    if is_onshore:
        # Mountain turbines provide ZERO shielding to Akureyri
        zero = [{'u_in': u, 'u_target': u, 'reduction_pct': 0.0,
                 'power_mw': 0} for u in speeds]
        stab_results = {s: zero for s in STABILITY_PRESETS}
        ht_results = zero
    else:
        stab_results = {}
        for slab, sparams in STABILITY_PRESETS.items():
            m = ChanneledWakeModel(fjord, sparams)
            stab_results[slab] = m.sweep_speeds(rows, AKUREYRI, speeds)

        ht_params = WakeParams(
            effective_height=200, recovery_length=80_000,
            channeling_fraction=0.92, high_thrust=True)
        m_ht = ChanneledWakeModel(fjord, ht_params)
        ht_results = m_ht.sweep_speeds(rows, AKUREYRI, speeds)

    # Revenue scenarios
    rev_data = []
    for rev_name, price in REVENUE_SCENARIOS:
        rev = aep['aep_gwh'] * 1000 * price
        prem = max(0, annual_cost - rev)
        isk_kwh = prem / (HH * HH_KWH) * ISK_USD
        hh_k = prem / HH * ISK_USD / 1000
        bill = isk_kwh / RETAIL_ISK * 100
        rev_data.append({
            'name': rev_name, 'price': price,
            'revenue_m': rev / 1e6, 'premium_m': prem / 1e6,
            'isk_kwh': isk_kwh, 'hh_kisk': hh_k, 'bill_pct': bill,
        })

    return {
        'name': name, 'info': info,
        'n': n, 'n_rows': n_rows, 'cap': cap,
        'capex_m': cap * capex_kw / 1000,
        'annual_m': annual_cost / 1e6,
        'aep': aep, 'lcoe': lcoe,
        'stab': stab_results, 'ht': ht_results,
        'rev': rev_data,
    }


# ── Output ───────────────────────────────────────────────────────

W = 72


def hr(c='='):
    print(c * W)


def print_design(d):
    info = d['info']
    print()
    hr()
    print(f'  {d["name"]}: {info["subtitle"]}')
    hr()
    print()
    for line in info['rationale']:
        print(f'    {line}')
    print()
    t = info['turbine']
    print(f'  Turbine:  {t.name}')
    print(f'            {t.rotor_diameter:.0f}m rotor | '
          f'{t.hub_height:.0f}m hub | '
          f'cut-out {t.cut_out:.0f} m/s')
    print(f'  Layout:   {d["n"]} turbines | '
          f'{d["n_rows"]} rows | '
          f'{d["cap"]:.0f} MW')
    print(f'  CAPEX:    ${d["capex_m"]:,.0f}M  '
          f'(${info["capex_kw"]:,}/kW)')
    print(f'  AEP:      {d["aep"]["aep_gwh"]:,.0f} GWh  |  '
          f'CF {d["aep"]["cf_pct"]:.1f}%  |  '
          f'LCOE ${d["lcoe"]:.0f}/MWh')
    print()

    # Wind reduction table
    print(f'  Wind reduction at Akureyri by atmospheric stability:')
    print()
    print(f'  {"Incoming":>8}  {"Neutral":>9}  {"Stable":>9}  '
          f'{"V.Stable":>9}  {"VS+HT":>9}')
    print(f'  {"":->8}  {"":->9}  {"":->9}  {"":->9}  {"":->9}')

    for i, spd_result in enumerate(d['stab']['neutral']):
        u = spd_result['u_in']
        vals = []
        for slab in ['neutral', 'stable', 'very_stable']:
            r = d['stab'][slab][i]
            dp = (1 - (r['u_target'] / r['u_in']) ** 2) * 100 \
                if r['u_in'] > 0 else 0
            vals.append(f'{r["reduction_pct"]:>4.1f}/{dp:>4.1f}%')
        # HT
        rh = d['ht'][i]
        dph = (1 - (rh['u_target'] / rh['u_in']) ** 2) * 100 \
            if rh['u_in'] > 0 else 0
        vals.append(f'{rh["reduction_pct"]:>4.1f}/{dph:>4.1f}%')
        print(f'  {u:>5.0f} m/s  {vals[0]:>9}  {vals[1]:>9}  '
              f'{vals[2]:>9}  {vals[3]:>9}')
    print()
    print(f'  Format: speed_reduction / pressure_reduction')
    print()

    # Revenue
    print(f'  Household economics by electricity revenue model:')
    print()
    print(f'  {"Revenue":<14} {"$/MWh":>5} {"Rev":>5} '
          f'{"Prem":>5} {"ISK/kWh":>7} {"kISK/hh":>7} {"Bill":>5}')
    print(f'  {"":->14} {"":->5} {"":->5} '
          f'{"":->5} {"":->7} {"":->7} {"":->5}')
    for rv in d['rev']:
        print(f'  {rv["name"]:<14} ${rv["price"]:>3} '
              f'${rv["revenue_m"]:>3.0f}M '
              f'${rv["premium_m"]:>3.0f}M '
              f'{rv["isk_kwh"]:>6.1f} '
              f'{rv["hh_kisk"]:>6.0f}k '
              f'{rv["bill_pct"]:>4.0f}%')
    print()


def print_comparison(designs):
    print()
    hr()
    print(f'  HEAD-TO-HEAD COMPARISON')
    hr()
    print()

    labels = [d['name'].split(')')[0] + ')' for d in designs]
    header = f'  {"Metric":<28}'
    for lb in labels:
        header += f' {lb:>12}'
    print(header)
    print(f'  {"":->28}' + f' {"":->12}' * len(labels))

    def row(metric, vals, fmt='{:>12}'):
        s = f'  {metric:<28}'
        for v in vals:
            s += ' ' + fmt.format(v)
        print(s)

    row('Turbines',
        [f'{d["n"]}x {d["info"]["turbine"].rated_power_mw:.0f}MW'
         for d in designs])
    row('Total capacity',
        [f'{d["cap"]:.0f} MW' for d in designs])
    row('CAPEX',
        [f'${d["capex_m"]:,.0f}M' for d in designs])
    row('AEP',
        [f'{d["aep"]["aep_gwh"]:,.0f} GWh' for d in designs])
    row('Capacity factor',
        [f'{d["aep"]["cf_pct"]:.1f}%' for d in designs])
    row('LCOE',
        [f'${d["lcoe"]:.0f}/MWh' for d in designs])
    print()

    # Wind reduction at 20 m/s under stable conditions
    for slab, label in [('neutral', 'du20 neutral'),
                        ('stable', 'du20 stable'),
                        ('very_stable', 'du20 v.stable')]:
        vals = []
        for d in designs:
            r20 = [r for r in d['stab'][slab] if r['u_in'] == 20][0]
            vals.append(f'{r20["reduction_pct"]:.1f}%')
        row(label, vals)

    # Pressure reduction
    vals_dp = []
    for d in designs:
        r20 = [r for r in d['stab']['stable'] if r['u_in'] == 20][0]
        dp = (1 - (r20['u_target'] / 20.0) ** 2) * 100
        vals_dp.append(f'{dp:.1f}%')
    row('dP20 stable', vals_dp)

    # High-thrust at very stable
    vals_ht = []
    for d in designs:
        r20 = [r for r in d['ht'] if r['u_in'] == 20][0]
        vals_ht.append(f'{r20["reduction_pct"]:.1f}%')
    row('du20 v.stable+HT', vals_ht)

    print()

    # Economics at different price points
    for rev_name in ['Wholesale', 'Data center', 'EU export', 'Green H2']:
        vals = []
        for d in designs:
            rv = [r for r in d['rev'] if r['name'] == rev_name][0]
            if rv['premium_m'] < 1:
                vals.append('FREE')
            else:
                vals.append(f'{rv["hh_kisk"]:.0f}k ISK')
        row(f'Cost/hh @ {rev_name}', vals)

    print()
    print(f'  Cost/hh = annual premium per household in ISK thousands')
    print(f'  Current annual electricity bill: ISK {BILL_ISK:,}')
    print()


def print_verdict():
    print()
    hr()
    print('  VERDICT')
    hr()
    print()
    lines = [
        'SULUR is the cheapest electricity. Nobody disputes that.',
        'Onshore mountain wind beats everything on LCOE.',
        'But Sulur gives Akureyri nothing except a changed skyline.',
        '',
        'The surprise: JAFNVAEGI is CHEAPER than ORKA.',
        'The shielding configuration uses shallower inner-fjord',
        'sites with fixed-bottom and coastal foundations ($2,800-',
        '$3,500/kW) while ORKA uses deep outer water ($4,500/kW).',
        'Blended CAPEX: $3,500/kW vs $4,500/kW.',
        '',
        'The dual-purpose framing does not just add value — it',
        'accesses cheaper foundation sites that happen to also',
        'be the right sites for wake shielding.',
        '',
        'JAFNVAEGI delivers:',
        '  - 14% pressure reduction during stable nordanatt',
        '  - LCOE competitive with European offshore wind',
        '  - Sulur undisturbed',
        '  - Hrisey gets an O&M economy',
        '  - Cheaper per MWh than placing turbines offshore',
        '    where they would provide no shielding',
        '',
        'The question is not "what is the cheapest kWh?"',
        'The question is "what does Akureyri get for what it pays?"',
        '',
        'SULUR: cheapest electricity + ruined skyline + nordanatt',
        'JAFNVAEGI: competitive electricity + logn + Sulur intact',
    ]
    for line in lines:
        print(f'  {line}')
    print()


# ── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    args = ap.parse_args()

    clim = None
    if not args.synthetic:
        clim = carra.load_climatology()
    if clim is None:
        clim = carra.synthetic_climatology()

    mouth = clim['stations'].get('mouth', {})
    wb_k = mouth.get('weibull_k', 2.0)
    wb_A = mouth.get('weibull_A', 9.6)

    print()
    hr()
    print('  WINDFALL: Three Designs for Eyjafjordur')
    hr()
    print()
    print(f'  Wind data: {clim["source"]}')
    print(f'  Financing: {DISCOUNT:.0%} / {LIFETIME}yr '
          f'(Landsvirkjun-style)')
    print(f'  Target: Akureyri ({AKUREYRI / 1000:.0f} km from mouth)')

    results = []
    for name, info in DESIGNS.items():
        d = eval_design(name, info, EYJAFJORDUR, wb_k, wb_A)
        print_design(d)
        results.append(d)

    print_comparison(results)
    print_verdict()


if __name__ == '__main__':
    main()
