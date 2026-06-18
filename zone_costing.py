#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""Zone-specific placement classification and differential CAPEX."""

import numpy as np
from designs import DESIGNS, EYJAFJORDUR, rows_of
from model import (WakeParams, ChanneledWakeModel, STABILITY_PRESETS,
                   load_calibrated_baseline, marginal_reduction)
import carra

PLACEMENT = {
    10: ('floating',  4500, 100),
    12: ('floating',  4500, 100),
    14: ('floating',  4500, 100),
    16: ('nearshore', 3500, 80),
    20: ('floating',  4200, 100),
    22: ('floating',  4200, 100),
    24: ('nearshore', 3500, 80),
    26: ('nearshore', 3500, 80),
    30: ('nearshore', 3200, 75),
    35: ('coastal',   2800, 60),
    36: ('coastal',   2800, 60),
    38: ('coastal',   2800, 60),
    39: ('coastal',   2800, 60),
    40: ('coastal',   2800, 60),
}

CRF = 0.04 * 1.04**30 / (1.04**30 - 1)


def main():
    clim = carra.load_climatology() or carra.synthetic_climatology()
    mouth = clim['stations'].get('mouth', {})
    wb_k = mouth.get('weibull_k', 2.0)
    wb_A = mouth.get('weibull_A', 9.6)

    print('=' * 72)
    print('  ZONE-SPECIFIC PLACEMENT AND COSTING')
    print('=' * 72)
    print()
    print('  Placement types:')
    print('    Floating (>80m depth):    $4,200-4,500/kW, OPEX $100/kW/yr')
    print('    Nearshore (<50m, fixed):  $3,200-3,500/kW, OPEX $75-80/kW/yr')
    print('    Coastal (lowland/shore):  $2,800/kW,       OPEX $60/kW/yr')
    print()

    results = []
    for dname, info in DESIGNS.items():
        rows = rows_of(info)
        is_onshore = info.get('_is_onshore', False)

        if is_onshore:
            total_cap = sum(r.capacity_mw for r in rows)
            capex_kw = info['capex_kw']
            opex_kw = info.get('_opex_kw', 50)
            total_capex = total_cap * 1000 * capex_kw
            total_opex = total_cap * 1000 * opex_kw
            aep_gwh = 1175
            lcoe = (total_capex * CRF + total_opex) / (aep_gwh * 1000)
            print(f'  {dname}: onshore mountain')
            print(f'    {sum(r.n_turbines for r in rows)} turbines, '
                  f'{total_cap:.0f} MW')
            print(f'    CAPEX: ${capex_kw:,}/kW (onshore)')
            print(f'    LCOE: ${lcoe:.0f}/MWh')
            print(f'    dP20: 0.0%')
            print()
            results.append({
                'name': dname, 'cap': total_cap,
                'blend': capex_kw, 'orig': capex_kw,
                'capex_m': total_capex / 1e6,
                'aep': aep_gwh, 'lcoe': lcoe, 'dp20': 0.0,
            })
            continue

        total_capex = 0
        total_opex = 0
        print(f'  {dname}:')

        for row in rows:
            km = row.x_position / 1000
            nearest = min(PLACEMENT.keys(), key=lambda k: abs(k - km))
            typ, capex_kw, opex_kw = PLACEMENT[nearest]
            cap = row.capacity_mw
            total_capex += cap * 1000 * capex_kw
            total_opex += cap * 1000 * opex_kw
            print(f'    km{km:>3.0f}: {row.n_turbines}x = {cap:.0f}MW  '
                  f'[{typ}] ${capex_kw:,}/kW')

        total_cap = sum(r.capacity_mw for r in rows)
        blended = total_capex / (total_cap * 1000)
        annual = total_capex * CRF + total_opex

        m = ChanneledWakeModel(EYJAFJORDUR, WakeParams(
            200, 30_000, 0.7, baseline_length=load_calibrated_baseline()))
        aep = m.aep(rows, weibull_k=wb_k, weibull_A=wb_A)
        lcoe = annual / (aep['aep_gwh'] * 1000) if aep['aep_gwh'] > 0 else 999

        orig_kw = info['capex_kw']
        orig_annual = (total_cap * 1000 * orig_kw * CRF
                       + total_cap * 1000 * 100)
        orig_lcoe = (orig_annual / (aep['aep_gwh'] * 1000)
                     if aep['aep_gwh'] > 0 else 999)

        m_s = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
        r20 = m_s.simulate(rows, 20.0, target_x=55000)
        _, dp20 = marginal_reduction(r20)   # turbine-only, vs baseline

        saving = orig_lcoe - lcoe
        aep_val = aep['aep_gwh']
        cf_val = aep['cf_pct']

        print(f'    Blended: ${blended:,.0f}/kW (was ${orig_kw:,})')
        print(f'    CAPEX: ${total_capex/1e6:,.0f}M '
              f'(was ${total_cap * orig_kw / 1000:,.0f}M)')
        print(f'    AEP {aep_val:.0f}G | CF {cf_val:.1f}%')
        print(f'    LCOE: ${lcoe:.0f}/MWh (was ${orig_lcoe:.0f})  '
              f'saving ${saving:.0f}/MWh')
        print(f'    dP20 stable: {dp20:.1f}%')
        print()

        results.append({
            'name': dname, 'cap': total_cap,
            'blend': blended, 'orig': orig_kw,
            'capex_m': total_capex / 1e6,
            'aep': aep_val, 'lcoe': lcoe, 'dp20': dp20,
        })

    print()
    print('  UPDATED HEAD-TO-HEAD:')
    print(f'  {"Design":<20} {"MW":>5} {"Blend":>7} '
          f'{"CAPEX":>7} {"AEP":>5} {"LCOE":>6} {"dP20s":>5}')
    print(f'  {"-"*20} {"-"*5} {"-"*7} '
          f'{"-"*7} {"-"*5} {"-"*6} {"-"*5}')
    for r in results:
        short = r['name'].split(')')[1].strip() if ')' in r['name'] else r['name']
        print(f'  {short:<20} {r["cap"]:>5.0f} '
              f'${r["blend"]:>5,.0f} ${r["capex_m"]:>5,.0f}M '
              f'{r["aep"]:>4.0f}G ${r["lcoe"]:>4.0f} '
              f'{r["dp20"]:>4.1f}%')

    # Shielding premium
    orka = next(r for r in results if 'ORKA' in r['name'])
    jafn = next(r for r in results if 'JAFN' in r['name'])
    premium = jafn['lcoe'] - orka['lcoe']
    print(f'\n  Shielding premium (JAFNVÆGI - ORKA): '
          f'${premium:+.0f}/MWh')


if __name__ == '__main__':
    main()
