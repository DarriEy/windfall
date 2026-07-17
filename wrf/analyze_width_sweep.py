#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Idealized fjord-WIDTH sweep analysis.

Same northerly forcing, same turbine wall, only the valley width changes
(W3 / W6 / W12 [/ W25]). Question: does valley width change how much the
turbine wall shelters the downwind town, or is the shielding really just
wake recovery length (terrain-independent)?

Metric = project standard: hub-height (165 m) wind POWER-DENSITY reduction
    reduction = (1 - (U_wall / U_base)**2) * 100   [%]
evaluated (a) at the fjord waypoints and (b) as a clean downwind profile
along the wall's mean longitude (norðanátt ~ wind from the north, wake
runs south).
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import carra

HUB = 165.0
IDEAL = Path(__file__).resolve().parent / 'ideal'
WIDTHS = [3, 6, 12, 25]          # 25 only if its wrfout exists


def hub_speed(ds, j, i, tidx):
    """Hub-height wind speed at grid (j,i), time index tidx."""
    U = 0.5 * (ds['U'][tidx, :, j, i] + ds['U'][tidx, :, j, i + 1]).values
    V = 0.5 * (ds['V'][tidx, :, j, i] + ds['V'][tidx, :, j + 1, i]).values
    ph = (ds['PH'][tidx, :, j, i] + ds['PHB'][tidx, :, j, i]).values / 9.81
    z = 0.5 * (ph[:-1] + ph[1:]) - ph[0]
    s = np.sqrt(U ** 2 + V ** 2)
    return float(np.interp(HUB, z, s))


def nearest(lat, lon, lat0, lon0):
    j, i = np.unravel_index(
        np.argmin((lat - lat0) ** 2 + (lon - lon0) ** 2), lat.shape)
    return int(j), int(i)


def avg_over_time(ds, j, i):
    """Time-mean hub speed over the back half of the run (quasi-steady)."""
    nt = ds.sizes['Time']
    t0 = nt // 2
    return np.mean([hub_speed(ds, j, i, t) for t in range(t0, nt)])


def wall_center(width):
    txt = np.loadtxt(IDEAL / f'wall_W{width}.txt')
    return float(txt[:, 0].mean()), float(txt[:, 1].mean()), len(txt)


def main():
    avail = [w for w in WIDTHS
             if (IDEAL / f'wrfout_d03_W{w}_baseline.nc').exists()
             and (IDEAL / f'wrfout_d03_W{w}_wall.nc').exists()]
    print(f'Widths with complete (baseline+wall) pairs: {avail}\n')

    summary = []
    for w in avail:
        base = xr.open_dataset(IDEAL / f'wrfout_d03_W{w}_baseline.nc')
        wall = xr.open_dataset(IDEAL / f'wrfout_d03_W{w}_wall.nc')
        lat = base['XLAT'][0].values
        lon = base['XLONG'][0].values
        latw, lonw, nturb = wall_center(w)

        # (a) waypoint table -------------------------------------------------
        print(f'=== W{w}  ({nturb} turbines, wall center '
              f'{latw:.3f}N {lonw:.3f}E) ===')
        print(f'  {"station":<10}{"x_km":>5}{"U_base":>8}{"U_wall":>8}{"redux%":>8}')
        wp_red = {}
        for wp in carra.WAYPOINTS:
            j, i = nearest(lat, lon, wp['lat'], wp['lon'])
            ub = avg_over_time(base, j, i)
            ut = avg_over_time(wall, j, i)
            red = (1 - (ut / ub) ** 2) * 100 if ub > 0 else 0.0
            wp_red[wp['name']] = red
            print(f'  {wp["name"]:<10}{wp["x_km"]:>5}{ub:>8.2f}{ut:>8.2f}{red:>7.1f}%')

        # (b) downwind profile = CROSS-VALLEY AVERAGE over the flat floor -----
        # straight valley runs due south; at each downwind distance average the
        # power-density reduction over every valley-floor cell (hgt < 50 m).
        hgt = base['HGT'][0].values
        dn_km = np.arange(0, 26, 2)
        deg_per_km = 1.0 / 111.0
        prof = []
        for d in dn_km:
            lat_d = latw - d * deg_per_km
            jrow = np.argmin(np.abs(lat[:, lat.shape[1] // 2] - lat_d))
            floor = np.where(hgt[jrow] < 50.0)[0]      # valley-floor columns
            if len(floor) == 0:
                prof.append(0.0); continue
            reds = []
            for i in floor:
                ub = avg_over_time(base, jrow, i)
                ut = avg_over_time(wall, jrow, i)
                if ub > 0:
                    reds.append((1 - (ut / ub) ** 2) * 100)
            prof.append(np.mean(reds) if reds else 0.0)
        prof = np.array(prof)
        peak = prof.max()
        peak_d = dn_km[prof.argmax()]
        # recovery length: distance for wake to decay to half of peak
        half = peak / 2
        rec = next((dn_km[k] for k in range(prof.argmax(), len(prof))
                    if prof[k] <= half), np.nan)

        town = float(np.interp(13, dn_km, prof))   # town ~13 km downwind of wall
        print(f'  cross-valley wake: peak {peak:.1f}% @ {peak_d} km south; '
              f'half by ~{rec} km; @13 km (town): {town:.1f}%\n')
        summary.append(dict(W=w, nturb=nturb, peak=peak, peak_d=peak_d,
                            rec=rec, town=town))
        base.close(); wall.close()

    # ---- the actual question -------------------------------------------------
    print('=' * 60)
    print('WIDTH TREND  (does valley width move the needle?)')
    print('  cross-valley-averaged hub power-density reduction, wall ON vs OFF')
    print(f'  {"W":>4}{"#turb":>6}{"peak%":>7}{"peak@km":>8}'
          f'{"half@km":>8}{"town13km%":>10}')
    for s in summary:
        print(f'  {s["W"]:>4}{s["nturb"]:>6}{s["peak"]:>6.1f}%{s["peak_d"]:>7} '
              f'{str(s["rec"]):>7}{s["town"]:>9.1f}%')


if __name__ == '__main__':
    main()
