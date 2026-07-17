#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Compare RVK shield DESIGNS on the gale: compact (rvkS, main) vs wide wall
(rvkWide) vs double-size (rvkBig). Downtown shielding (10 m + hub) and
LCOE each, to see if clever geometry / more turbines boost shielding at
acceptable cost. 1/cell throughout (the Fitch floor).
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')
from math import gamma

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyze_pilot import box_means, HUB
from analyze_pilot_rvk import cf_from_weibull, hub_at_cells, cells_of, SHEAR, WB_K
from designs import FUTURE20, CRF

PILOT = HERE / 'pilot'
DATA = REPO / 'data'
CITY = (64.13, -21.90)
GALE = 'rvk_swgale'
CAPEX_KW = 3200          # nearshore Faxaflói (all designs same foundation class)
OPEX_KW = 75
DESIGNS = ['rvkS', 'rvkWide', 'rvkBig']


def main():
    base = xr.open_dataset(PILOT / f'wrfout_d03_{GALE}_baseline.nc')
    lat, lon = base['XLAT'][0].values, base['XLONG'][0].values
    box = (111 * np.sqrt((lat - CITY[0]) ** 2
           + ((lon - CITY[1]) * np.cos(np.radians(64.1))) ** 2)) <= 2.5
    s10b, hubb = box_means(base, box)
    # annual resource reference (faxafloi hub Weibull) + WRF ref cell
    fax = np.concatenate([np.load(f)['faxafloi_speed']
                          for f in sorted(DATA.glob('rvk_winds_2022_H*.npz'))])
    A_ref = fax.mean() * SHEAR / gamma(1 + 1 / WB_K)
    jf, iff = np.unravel_index(
        np.argmin((lat - 64.25) ** 2 + (lon + 22.30) ** 2), lat.shape)
    u_ref = hub_at_cells(base, [(int(jf), int(iff))])

    print(f'RVK gale, downtown baseline 10m {s10b:.2f} / hub {hubb:.2f} m/s\n')
    print(f'  {"design":<9}{"N":>4}{"MW":>6}{"red10":>8}{"redhub":>8}'
          f'{"CF":>7}{"LCOE":>7}{"hub%/100MW":>12}')
    for s in DESIGNS:
        p = PILOT / f'wrfout_d03_{GALE}_{s}.nc'
        if not p.exists():
            print(f'  {s:<9} (not run yet)')
            continue
        ds = xr.open_dataset(p)
        s10, hub = box_means(ds, box)
        r10 = (1 - s10 / s10b) * 100
        rhub = (1 - hub / hubb) * 100
        cells = cells_of(s, lat, lon)
        n = len(cells)
        cap = n * FUTURE20.rated_power_mw
        A_s = A_ref * (hub_at_cells(base, cells) / u_ref)
        cf = cf_from_weibull(A_s)
        aep = cf / 100 * cap * 8766 / 1000
        lcoe = cap * 1000 * (CAPEX_KW * CRF + OPEX_KW) / (aep * 1000)
        per100 = rhub / cap * 100
        print(f'  {s:<9}{n:>4}{cap:>6.0f}{r10:>+7.1f}%{rhub:>+7.1f}%'
              f'{cf:>6.1f}%${lcoe:>5.0f}{per100:>11.2f}')
    print('\nhub%/100MW = shielding efficiency (downtown hub reduction per '
          '100 MW installed)')


if __name__ == '__main__':
    main()
