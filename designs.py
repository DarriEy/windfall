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
    load_calibrated_baseline,
)
import carra
import observed

# Per-station Weibull resource (fitted once from raw CARRA records), used
# for AEP so each turbine sees its true local distribution shape rather
# than the mouth shape scaled by the baseline decay.
_STATION_WEIBULL = None


def station_weibull():
    global _STATION_WEIBULL
    if _STATION_WEIBULL is None:
        _STATION_WEIBULL = observed.weibull_callable()
    return _STATION_WEIBULL


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
    'E) SAMSETT': {
        'subtitle': 'Split: outer generation + dense inner shield',
        'rationale': [
            'Two separately-costed clusters instead of one blended farm,',
            'so the shield is not judged as a (poor) generator.',
            'Generation cluster: 18x Future20 at the windy outer fjord',
            '(km 10-14, floating $4,000/kW) — a standalone $144/MWh',
            'project, competitive and independently financeable.',
            'Shield cluster: 6x Future20 packed densely in the inner',
            'fjord (km 37-39, coastal $2,800/kW), close to Akureyri where',
            'the wake has not recovered. As a generator it is poor',
            '($196/MWh, CF 13%), but it buys pressure reduction at',
            '~46 %/$B — 2.2x the shielding-per-ISK of the blended',
            'JAFNVAEGI farm. It is costed as a storm-protection asset.',
        ],
        'turbine': FUTURE20,        # representative (for fallback display)
        'capex_kw': 3700,           # nominal; true blend computed per-cluster
        'clusters': [
            {'turbine': FUTURE20, 'capex_kw': 4000, 'opex_kw': 100,
             'rows': _rows([10, 12, 14], 6, FUTURE20)},
            {'turbine': FUTURE20, 'capex_kw': 2800, 'opex_kw': 60,
             'rows': _rows([37, 39], 3, FUTURE20)},
        ],
    },
}


# ── Evaluation ───────────────────────────────────────────────────

ZONE_OPEX = {
    'A) LOGN': 85,       # blended: floating + nearshore + coastal
    'B) ORKA': 100,      # all floating
    'C) JAFNVAEGI': 75,  # blended: floating + nearshore + coastal
}


# ── Canonical design accessors (use these everywhere; do NOT index
#    info['rows'] directly — a design may instead carry 'clusters') ──

def rows_of(info):
    """All turbine rows for a design, whether monolithic ('rows') or
    split into separately-costed clusters ('clusters')."""
    return info.get('rows') or [r for c in info.get('clusters', [])
                                for r in c['rows']]


def design_cost(name, info):
    """Economics for any design. Returns
    (cap_mw, n_turbines, capex_m_usd, blended_capex_kw, annual_cost_usd).
    Clusters are summed cluster-by-cluster with their own CAPEX/OPEX."""
    clusters = info.get('clusters')
    if clusters:
        rows = rows_of(info)
        cap = sum(r.capacity_mw for r in rows)
        n = sum(r.n_turbines for r in rows)
        capex_m = sum(sum(r.capacity_mw for r in c['rows']) * c['capex_kw']
                      for c in clusters) / 1000
        annual = sum(
            sum(r.capacity_mw for r in c['rows']) * 1000
            * (c['capex_kw'] * CRF + c.get('opex_kw', OPEX_KW))
            for c in clusters)
        capex_kw = round(capex_m * 1000 / cap) if cap else 0
    else:
        rows = info['rows']
        cap = sum(r.capacity_mw for r in rows)
        n = sum(r.n_turbines for r in rows)
        capex_kw = info['capex_kw']
        opex_kw = info.get('_opex_kw', ZONE_OPEX.get(name, OPEX_KW))
        capex_m = cap * capex_kw / 1000
        annual = cap * 1000 * (capex_kw * CRF + opex_kw)
    return cap, n, capex_m, capex_kw, annual


def eval_design(name, info, fjord, wb_k, wb_A):
    # A design is either monolithic (single turbine + single CAPEX) or a
    # set of separately-costed clusters (e.g. outer-fjord generation on
    # floating foundations + a dense inner-fjord shield on cheap coastal
    # foundations). Clusters are evaluated as one combined wake for
    # shielding, but their economics are summed cluster-by-cluster so a
    # weak-resource shield cluster is not credited with outer-fjord wind.
    rows = rows_of(info)
    cap, n, capex_m, capex_kw, annual_cost = design_cost(name, info)
    turb = info.get('turbine') or rows[0].turbine
    n_rows = len(rows)

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
        aep = m_aep.aep(rows, station_weibull=station_weibull())
    lcoe = annual_cost / (aep['aep_gwh'] * 1000) if aep['aep_gwh'] > 0 else 9999

    # Wind reduction under each stability regime
    speeds = [10, 12, 16, 20, 25]
    max_cut_out = max((r.turbine.cut_out for r in rows), default=turb.cut_out)
    if max_cut_out >= 30:
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
            channeling_fraction=0.92, high_thrust=True,
            baseline_length=load_calibrated_baseline())
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
        'capex_m': capex_m, 'capex_kw': capex_kw,
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
    clusters = info.get('clusters')
    if clusters:
        for ci, c in enumerate(clusters, 1):
            ct = c['turbine']
            cn = sum(r.n_turbines for r in c['rows'])
            ccap = sum(r.capacity_mw for r in c['rows'])
            role = 'generation' if ci == 1 else 'shield'
            print(f'  Cluster {ci} ({role}): {cn}x {ct.name} | '
                  f'{ccap:.0f} MW | ${c["capex_kw"]:,}/kW')
    else:
        t = info['turbine']
        print(f'  Turbine:  {t.name}')
        print(f'            {t.rotor_diameter:.0f}m rotor | '
              f'{t.hub_height:.0f}m hub | '
              f'cut-out {t.cut_out:.0f} m/s')
    print(f'  Layout:   {d["n"]} turbines | '
          f'{d["n_rows"]} rows | '
          f'{d["cap"]:.0f} MW')
    print(f'  CAPEX:    ${d["capex_m"]:,.0f}M  '
          f'(${d["capex_kw"]:,}/kW blended)')
    print(f'  AEP:      {d["aep"]["aep_gwh"]:,.0f} GWh  |  '
          f'CF {d["aep"]["cf_pct"]:.1f}%  |  '
          f'LCOE ${d["lcoe"]:.0f}/MWh')
    print()

    # Wind reduction table -- MARGINAL turbine effect, on top of the
    # natural fjord sheltering (which the CARRA-calibrated baseline
    # already reproduces). Showing the total vs mouth-inflow would
    # conflate the turbines with the fjord's own ~45% nordanatt decay.
    print(f'  TURBINE-INDUCED wind reduction at Akureyri (marginal, on')
    print(f'  top of natural fjord sheltering) by atmospheric stability:')
    print()
    if not d['info'].get('_is_onshore', False):
        nb = d['stab']['neutral'][0]
        nat = nb.get('natural_reduction_pct', 0.0)
        print(f'  Natural baseline sheltering (no turbines): '
              f'{nat:.0f}% slower at Akureyri than at the mouth')
        print()
    print(f'  {"Incoming":>8}  {"Neutral":>9}  {"Stable":>9}  '
          f'{"V.Stable":>9}  {"VS+HT":>9}')
    print(f'  {"":->8}  {"":->9}  {"":->9}  {"":->9}  {"":->9}')

    def _marginal(r):
        ub = r.get('baseline_u', r['u_in'])
        sr = r.get('turbine_reduction_pct', r['reduction_pct'])
        dp = (1 - (r['u_target'] / ub) ** 2) * 100 if ub > 0 else 0.0
        return f'{sr:>4.1f}/{dp:>4.1f}%'

    for i, spd_result in enumerate(d['stab']['neutral']):
        u = spd_result['u_in']
        vals = [_marginal(d['stab'][slab][i])
                for slab in ['neutral', 'stable', 'very_stable']]
        vals.append(_marginal(d['ht'][i]))
        print(f'  {u:>5.0f} m/s  {vals[0]:>9}  {vals[1]:>9}  '
              f'{vals[2]:>9}  {vals[3]:>9}')
    print()
    print(f'  Format: turbine speed_reduction / turbine pressure_reduction')
    print(f'  (both measured against the no-turbine fjord baseline)')
    print(f'  NB: product (multiplicative) superposition — the OPTIMISTIC '
          f'end.')
    print(f'  Sum-of-squares gives ~40-60% of these; see uncertainty.py '
          f'for P10/P50/P90.')
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

    # Marginal turbine-induced wind reduction at 20 m/s inflow, by
    # stability (separate from the natural fjord baseline).
    def _turb_red20(d, slab):
        r = [r for r in d['stab'][slab] if r['u_in'] == 20][0]
        return r.get('turbine_reduction_pct', r['reduction_pct'])

    for slab, label in [('neutral', 'du20 neutral'),
                        ('stable', 'du20 stable'),
                        ('very_stable', 'du20 v.stable')]:
        row(label, [f'{_turb_red20(d, slab):.1f}%' for d in designs])

    # Turbine-induced pressure reduction (vs no-turbine baseline)
    vals_dp = []
    for d in designs:
        r20 = [r for r in d['stab']['stable'] if r['u_in'] == 20][0]
        ub = r20.get('baseline_u', 20.0)
        dp = (1 - (r20['u_target'] / ub) ** 2) * 100 if ub > 0 else 0.0
        vals_dp.append(f'{dp:.1f}%')
    row('dP20 stable (turbine)', vals_dp)

    # High-thrust at very stable
    vals_ht = []
    for d in designs:
        r20 = [r for r in d['ht'] if r['u_in'] == 20][0]
        vals_ht.append(f'{r20.get("turbine_reduction_pct", 0):.1f}%')
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
        'Onshore mountain wind beats everything on LCOE ($36/MWh).',
        'But Sulur gives Akureyri nothing except a changed skyline.',
        '',
        'The honest result, once each turbine is costed at its true',
        'along-fjord wind: shielding is NOT free. The inner-fjord sites',
        'that shelter Akureyri best sit in the weakest wind, so the',
        'monolithic shield (JAFNVAEGI, $161/MWh) is dearer than pure',
        'outer-fjord generation (ORKA, $140/MWh) — not cheaper.',
        '',
        'The fix is not to blend, but to split (SAMSETT):',
        '  - an outer generation cluster that stands alone at $144/MWh,',
        '  - plus a small dense inner shield costed as a storm-protection',
        '    asset, buying pressure reduction at ~46 %/$B (2.2x the',
        '    shielding-per-ISK of the blended JAFNVAEGI farm).',
        'SAMSETT delivers ~23% stable-nordanatt pressure reduction at',
        '$155/MWh blended (lower LCOE than JAFNVAEGI). Being a larger',
        'build its absolute household premium is higher at low power',
        'prices, but lower once prices cover cost (green H2: 79k vs',
        '121k ISK/household).',
        '',
        'Shielding is a co-benefit bought at a modest premium, not a',
        'free by-product. The question is not "what is the cheapest',
        'kWh?" but "what does Akureyri get for what it pays?"',
        '',
        'SULUR:   cheapest electricity + ruined skyline + nordanatt',
        'SAMSETT: competitive generation + a priced logn shield + Sulur',
        '         intact',
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
    print('  WINDFALL: Five Designs for Eyjafjordur')
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
