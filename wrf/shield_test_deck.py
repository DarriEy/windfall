#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Shield-placement WRF test: relocate the SAME 18-turbine array that
produced the ~8% wake at Hrisey (SAMSETT outer cluster, km10/12/14) to
the INNER fjord (km42/44/46), ~9-13 km upstream of Akureyri, so its
wake should peak at the fjord head if the effect is placement-attainable.

Controlled experiment: identical array (18x Future-20), identical
domain/met forcing/baseline — only the along-fjord position changes.
Writes windturbines_shield.txt (lat lon type) for the Fitch scheme.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from designs import FUTURE20
from model import TurbineRow
from make_rundeck import turbine_latlons, write_windturbines

# Same 3x6 geometry as the SAMSETT outer generation cluster, relocated
# up-fjord so the wake (which peaks ~2-6 km downstream) lands near
# Akureyri (km55). Default = inner-fjord shield siting (km42-46); pass
# positions + output label on the command line for other placements,
# e.g. the "doorstep" case:  shield_test_deck.py doorstep 49 51 53
SHIELD_KM = [42, 44, 46]
LABEL = 'shield'
if len(sys.argv) > 1:
    LABEL = sys.argv[1]
    SHIELD_KM = [float(a) for a in sys.argv[2:]] or SHIELD_KM


def main():
    rows = [TurbineRow(x * 1000, 6, FUTURE20) for x in SHIELD_KM]
    latlons = turbine_latlons(rows)
    out = HERE / f'windturbines_{LABEL}.txt'
    write_windturbines(latlons, out)
    print(f'{LABEL} array: {len(latlons)} x {FUTURE20.name} at km '
          f'{SHIELD_KM} (upstream of Akureyri at km55)')
    print(f'  lat {min(l for l,_ in latlons):.3f}-{max(l for l,_ in latlons):.3f}, '
          f'lon {min(o for _,o in latlons):.3f}-{max(o for _,o in latlons):.3f}')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
