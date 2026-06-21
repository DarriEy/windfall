#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Signature figure for the public-good paper: the confined fjord LEAKS the
shelter over its own ridges. Maps the turbine-induced hub-wind change
(turbine run - baseline) for the Akureyri dense valley wall, over terrain
contours: a deficit on the valley floor (the intended shelter) but strong
SPEED-UP on the bounding ridges (the diverted flow that refills the wake)
-> only ~16% survives at the town. Contrast panel: Reykjavik open-coast
wake stays coherent (no ridge leak).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'figures'


def hubfield(ds, t, hub=150.0):
    U = 0.5 * (ds['U'][t, :, :, :-1] + ds['U'][t, :, :, 1:]).values
    V = 0.5 * (ds['V'][t, :, :-1, :] + ds['V'][t, :, 1:, :]).values
    ph = (ds['PH'][t] + ds['PHB'][t]).values / 9.81
    z = 0.5 * (ph[:-1] + ph[1:]) - ph[0:1]
    s = np.sqrt(U ** 2 + V ** 2)
    ny, nx = s.shape[1:]
    out = np.empty((ny, nx))
    for j in range(ny):
        for i in range(nx):
            out[j, i] = np.interp(hub, z[:, j, i], s[:, j, i])
    return out


def panel(ax, bpath, tpath, frac, hub, title, hgtcontours=True):
    b = xr.open_dataset(bpath); t = xr.open_dataset(tpath)
    tt = int(min(b.sizes['Time'], t.sizes['Time']) * frac)
    lat = b['XLAT'][0].values; lon = b['XLONG'][0].values
    hgt = b['HGT'][0].values
    dlt = (hubfield(t, tt, hub) - hubfield(b, tt, hub))  # m/s change
    im = ax.pcolormesh(lon, lat, dlt, cmap='RdBu_r', vmin=-6, vmax=6, shading='auto')
    if hgtcontours:
        ax.contour(lon, lat, hgt, levels=[300, 600, 900], colors='k',
                   linewidths=0.4, alpha=0.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    im = panel(axes[0],
               'wrf/proof1km/wrfout_d03_baseline.nc',
               'wrf/proof1km/wrfout_d03_densewall.nc', 0.35, 165.0,
               'Eyjafjörður fjord: wall blocks the floor,\nflow SPEEDS UP over the ridges (leak)')
    axes[0].plot(-18.09, 65.68, 'k*', ms=13)
    axes[0].annotate('Akureyri', (-18.09, 65.68), fontsize=8, ha='left', va='top')
    panel(axes[1],
          'wrf/rvk/wrfout_d03_swgale_baseline.nc',
          'wrf/rvk/wrfout_d03_swgale_turb.nc', 0.7, 150.0,
          'Reykjavík open coast: clean wake,\nno ridge leak — deficit reaches the city',
          hgtcontours=True)
    axes[1].plot(-21.90, 64.13, 'k*', ms=13)
    axes[1].annotate('Reykjavík', (-21.90, 64.13), fontsize=8, ha='left', va='bottom')
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label('turbine-induced hub-wind change (m/s)\n(blue = slower / shelter, red = faster / leak)', fontsize=9)
    fig.suptitle('Why confined terrain leaks the shelter and open coast holds it',
                 fontsize=12, fontweight='bold')
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'fig_diversion.png', dpi=200, bbox_inches='tight')
    print(f'Saved {OUT}/fig_diversion.png')


if __name__ == '__main__':
    main()
