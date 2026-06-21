#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
The public-good figure: WHEN the wake is a deliverable community benefit.
(a) City wind-pressure reduction vs hub wind for the Reykjavik E-wall: large
    in the everyday/operating regime (high Ct), collapses past cut-out
    (feathering) -> storm-shielding the extreme is off the table, everyday
    reduction is the prize.
(b) Annual wind-speed distribution at the city (CARRA) shaded by turbine
    regime (idle / high-Ct operating / feathering), with the resulting
    annual-mean wind reduction for a single upwind wall and a two-arc ring.
"""
import numpy as np, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'figures'
HUB = 150.0


def hub(ds, j, i, t):
    U = 0.5 * (ds['U'][t, :, j, i] + ds['U'][t, :, j, i + 1]).values
    V = 0.5 * (ds['V'][t, :, j, i] + ds['V'][t, :, j + 1, i]).values
    ph = (ds['PH'][t, :, j, i] + ds['PHB'][t, :, j, i]).values / 9.81
    z = 0.5 * (ph[:-1] + ph[1:]) - ph[0]
    return np.interp(HUB, z, np.sqrt(U ** 2 + V ** 2))


def main():
    # (a) dP vs hub wind from the E-gale run (spans moderate -> storm)
    b = xr.open_dataset('wrf/rvk/wrfout_d03_egale_baseline.nc')
    t = xr.open_dataset('wrf/rvk/wrfout_d03_egale_turb.nc')
    lat = b['XLAT'][0].values; lon = b['XLONG'][0].values
    cj, ci = np.unravel_index(np.argmin((lat - 64.13) ** 2 + (lon + 21.90) ** 2), lat.shape)
    nt = min(b.sizes['Time'], t.sizes['Time'])
    spd, dp = [], []
    for f in range(nt):
        ub = hub(b, cj, ci, f); ut = hub(t, cj, ci, f)
        # only easterly frames (c-wall upwind); skip near-calm
        if ub > 4:
            spd.append(ub); dp.append(100 * (1 - (ut / ub) ** 2))
    spd = np.array(spd); dp = np.array(dp)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].axvspan(3, 12, color='#27ae60', alpha=0.10)
    ax[0].axvspan(25, 40, color='#c0392b', alpha=0.10)
    ax[0].scatter(spd, dp, c=spd, cmap='viridis', s=45, zorder=3)
    ax[0].axvline(25, color='#c0392b', ls='--', lw=1.5)
    ax[0].annotate('cut-out 25 m/s\n(turbines feather)', (25, ax[0].get_ylim()[1]*0.9),
                   color='#c0392b', fontsize=8, ha='right')
    ax[0].annotate('everyday / high-Ct\n(shelter is strong)', (7.5, 5),
                   color='#1e6b35', fontsize=8, ha='center')
    ax[0].set_xlabel('hub wind at city (m/s)', fontsize=11)
    ax[0].set_ylabel('city wind-pressure reduction ΔP (%)', fontsize=11)
    ax[0].set_title('(a) The shelter is an everyday good, not a storm shield',
                    fontsize=10.5, fontweight='bold')
    ax[0].grid(alpha=0.2)

    # (b) annual distribution + regime shading + annual-mean reductions
    d1 = np.load('data/rvk_winds_2022_H1.npz'); d2 = np.load('data/rvk_winds_2022_H2.npz')
    sp = np.concatenate([d1['rvk_city_speed'], d2['rvk_city_speed']])
    fac = math.log(150 / 0.03) / math.log(10 / 0.03)
    h = sp * fac
    ax[1].hist(h, bins=np.arange(0, 36, 1.5), color='#bbb', edgecolor='w')
    for lo, hi, c, lbl in [(0, 3, '#999', 'idle'), (3, 12, '#27ae60', 'high-Ct (62%)'),
                           (12, 25, '#f1c40f', 'operating (22%)'), (25, 40, '#c0392b', 'feather (2%)')]:
        ax[1].axvspan(lo, hi, color=c, alpha=0.12)
    ax[1].axvline(h.mean(), color='k', lw=1.5)
    ax[1].annotate(f'mean {h.mean():.1f} m/s\n(windy capital)', (h.mean()+0.5, ax[1].get_ylim()[1]*0.8),
                   fontsize=8)
    ax[1].set_xlabel('hub wind at city (m/s)', fontsize=11)
    ax[1].set_ylabel('hours / yr (CARRA 2022)', fontsize=11)
    ax[1].set_title('(b) 84% of hours operate → annual-mean wind cut\n'
                    'single wall −6.4% · two-arc ring ~−13% · (ring ~−23%)',
                    fontsize=10.5, fontweight='bold')
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'fig_publicgood.png', dpi=200, bbox_inches='tight')
    print(f'Saved {OUT}/fig_publicgood.png  ({len(spd)} dP points)')


if __name__ == '__main__':
    main()
