#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Regression tests locking in the model contract after the 2026-06 rework.
Runnable with pytest, or directly: `python tests/test_invariants.py`.

These guard the bugs surfaced in review: the marginal-vs-total reduction
distinction, the clusters schema, calibrated-baseline wiring, and the
product/sos superposition behaviour.
"""
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import carra
from model import (
    ChanneledWakeModel, WakeParams, STABILITY_PRESETS,
    load_calibration, marginal_reduction,
)
from designs import (
    DESIGNS, EYJAFJORDUR, AKUREYRI, rows_of, design_cost, eval_design,
)

_clim = carra.load_climatology()
_WK = _clim['stations']['mouth']['weibull_k']
_WA = _clim['stations']['mouth']['weibull_A']


def test_rows_of_handles_clusters():
    # SAMSETT is cluster-based and has no 'rows' key.
    rows = rows_of(DESIGNS['E) SAMSETT'])
    assert len(rows) == 5            # 3 outer + 2 inner rows
    assert sum(r.capacity_mw for r in rows) == 480


def test_design_cost_blends_clusters():
    cap, n, capex_m, capex_kw, annual = design_cost(
        'E) SAMSETT', DESIGNS['E) SAMSETT'])
    assert cap == 480 and n == 24
    # blended CAPEX between the coastal ($2800) and floating ($4000) ends
    assert 2800 < capex_kw < 4000


def test_marginal_is_not_total():
    # The fjord baseline shelters Akureyri, so turbine-only reduction
    # must be well below the total-vs-inflow reduction.
    m = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
    r = m.simulate(DESIGNS['C) JAFNVAEGI']['rows'], 20.0, target_x=AKUREYRI)
    total = r['reduction_pct']
    _, marg_p = marginal_reduction(r)
    assert total > r['turbine_reduction_pct']        # natural sheltering on top
    assert r['natural_reduction_pct'] > 30           # fjord does the bulk


def test_orka_pressure_reduction_is_small():
    # ORKA is pure outer-fjord generation: its turbine-only stable
    # pressure reduction is a few %, NOT the ~67% total-fjord figure
    # that the pre-fix scripts reported.
    m = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
    r = m.simulate(DESIGNS['B) ORKA']['rows'], 20.0, target_x=AKUREYRI)
    _, dp = marginal_reduction(r)
    assert dp < 8.0


def test_sos_below_product():
    # Sum-of-squares superposition is the conservative end.
    rows = DESIGNS['C) JAFNVAEGI']['rows']
    base = STABILITY_PRESETS['stable']
    rp = ChanneledWakeModel(EYJAFJORDUR, replace(base, superposition='product')
                            ).simulate(rows, 20.0, target_x=AKUREYRI)
    rs = ChanneledWakeModel(EYJAFJORDUR, replace(base, superposition='sos')
                            ).simulate(rows, 20.0, target_x=AKUREYRI)
    _, p_prod = marginal_reduction(rp)
    _, p_sos = marginal_reduction(rs)
    assert p_sos < p_prod


def test_baseline_matches_calibration():
    # No-turbine baseline at Akureyri should follow the calibrated decay.
    cal = load_calibration('nordanatt')
    m = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
    r = m.simulate([], 20.0, target_x=AKUREYRI)
    import math
    expected = 20.0 * math.exp(-(AKUREYRI - cal['anchor']) / cal['length'])
    assert abs(r['baseline_u'] - expected) < 0.05


def test_all_designs_eval():
    # Every named design (monolithic and clustered) evaluates cleanly.
    for name, info in DESIGNS.items():
        d = eval_design(name, info, EYJAFJORDUR, _WK, _WA)
        assert d['cap'] > 0 and d['lcoe'] > 0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fn.__name__}: {e}')
    print(f'\n  {len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
