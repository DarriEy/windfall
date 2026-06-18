#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Generate a WRF + WPS run deck for the Eyjafjörður validation, straight
from the study's turbine designs (designs.py). Produces, in this folder:

  namelist.wps        WPS geogrid/ungrib/metgrid (9/3/1 km nest on the fjord)
  namelist.input      WRF run with the Fitch wind-farm scheme (windfarm_opt=1)
  wind-turbine-1.tbl   turbine spec (hub, rotor, Ct & power curves)
  windturbines.txt     turbine lat/lon + type, one per line (windfarm_ij=0)
  README.md            run sequence + data prerequisites

The WRF wake is driven by the thrust-coefficient curve, which comes
directly from TurbineSpec, so the deck reproduces the same turbines used
in the 1D model. Default config = SAMSETT (designs.py); change CONFIG.
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from designs import DESIGNS, EYJAFJORDUR, rows_of

CONFIG = 'E) SAMSETT'
HERE = Path(__file__).resolve().parent

# Fjord centreline (km -> lat/lon), digitised from the DEM (matches
# make_figures.py _AXIS), used to place turbines at real coordinates.
_AXIS = np.array([
    [0, 66.150, -18.100], [5, 66.105, -18.130], [10, 66.060, -18.175],
    [12, 66.040, -18.200], [14, 66.025, -18.220], [16, 66.010, -18.240],
    [20, 65.980, -18.310], [22, 65.965, -18.340], [24, 65.945, -18.360],
    [26, 65.920, -18.370], [30, 65.880, -18.330], [35, 65.835, -18.260],
    [36, 65.825, -18.245], [38, 65.808, -18.218], [39, 65.798, -18.205],
    [40, 65.785, -18.190], [45, 65.745, -18.140], [50, 65.715, -18.105],
    [55, 65.680, -18.090],
])
REF_LAT, REF_LON = 65.85, -18.20


def turbine_latlons(rows):
    """Lat/lon of every turbine: rows placed along the axis, turbines in
    a row spread cross-fjord (≈E-W) across a fraction of the width."""
    out = []
    for r in rows:
        x_km = r.x_position / 1000.0
        lat = float(np.interp(x_km, _AXIS[:, 0], _AXIS[:, 1]))
        lon = float(np.interp(x_km, _AXIS[:, 0], _AXIS[:, 2]))
        w_km = EYJAFJORDUR.width(r.x_position) / 1000.0
        dlon = w_km / (111.0 * np.cos(np.radians(lat))) * 0.35
        offs = np.linspace(-dlon, dlon, r.n_turbines) if r.n_turbines > 1 \
            else [0.0]
        for dl in offs:
            out.append((lat, lon + dl))
    return out


def write_turbine_tbl(turb, path):
    """Fitch wind-turbine-1.tbl: line1 = hub diameter standing_Ct npower;
    then npower rows of (windspeed, thrust_coeff, power_kW)."""
    speeds = np.arange(float(turb.cut_in), float(turb.cut_out) + 0.5, 1.0)
    ct = turb.ct(speeds)                     # thrust coeff curve
    pw_kw = turb.power(speeds) * 1000.0      # MW -> kW
    standing_ct = 0.158                      # idling Ct (WRF default order)
    with open(path, 'w') as f:
        f.write(f'{turb.hub_height:.1f} {turb.rotor_diameter:.1f} '
                f'{standing_ct:.3f} {len(speeds)}\n')
        for s, c, p in zip(speeds, ct, pw_kw):
            f.write(f'{s:.1f} {c:.4f} {p:.2f}\n')


def write_windturbines(latlons, path, type_id=1):
    with open(path, 'w') as f:
        for lat, lon in latlons:
            f.write(f'{lat:.5f} {lon:.5f} {type_id}\n')


NAMELIST_WPS = """\
&share
 wrf_core = 'ARW', max_dom = 3, start_date = 3*'{start}',
 end_date = 3*'{end}', interval_seconds = 21600,
/
&geogrid
 parent_id         = 1, 1, 2,
 parent_grid_ratio = 1, 3, 3,
 i_parent_start    = 1, 33, 30,
 j_parent_start    = 1, 33, 30,
 e_we              = 100, 121, 121,
 e_sn              = 100, 121, 121,
 geog_data_res     = 'default','default','default',
 dx = 9000, dy = 9000,
 map_proj = 'lambert',
 ref_lat = {ref_lat}, ref_lon = {ref_lon},
 truelat1 = 64.0, truelat2 = 67.0, stand_lon = {ref_lon},
 geog_data_path = '/BUILD_WRF/WPS_GEOG',
/
&ungrib
 out_format = 'WPS', prefix = 'FILE',
/
&metgrid
 fg_name = 'FILE',
/
"""

NAMELIST_INPUT = """\
&time_control
 run_days = 0, run_hours = {run_hours}, run_minutes = 0,
 start_year  = 3*{sy}, start_month = 3*{sm}, start_day = 3*{sd}, start_hour = 3*{sh},
 end_year    = 3*{ey}, end_month   = 3*{em}, end_day   = 3*{ed}, end_hour   = 3*{eh},
 interval_seconds = 21600, input_from_file = 3*.true.,
 history_interval = 60, 60, 30, frames_per_outfile = 1000, 1000, 1000,
 restart = .false., io_form_history = 2, io_form_restart = 2,
 io_form_input = 2, io_form_boundary = 2,
/
&domains
 time_step = 45, max_dom = 3,
 e_we = 100, 121, 121, e_sn = 100, 121, 121, e_vert = 45, 45, 45,
 dx = 9000, 3000, 1000, dy = 9000, 3000, 1000,
 grid_id = 1, 2, 3, parent_id = 1, 1, 2,
 i_parent_start = 1, 33, 30, j_parent_start = 1, 33, 30,
 parent_grid_ratio = 1, 3, 3, parent_time_step_ratio = 1, 3, 3,
 p_top_requested = 5000, num_metgrid_levels = 30,
/
&physics
 mp_physics = 6, 6, 6, ra_lw_physics = 4, 4, 4, ra_sw_physics = 4, 4, 4,
 sf_sfclay_physics = 5, 5, 5, sf_surface_physics = 2, 2, 2,
 bl_pbl_physics = 5, 5, 5, cu_physics = 1, 0, 0,
 ! --- Fitch wind-farm parameterization (the turbines) ---
 windfarm_opt = 0, 0, 1,
 windfarm_ij  = 0,
/
&dynamics
 hybrid_opt = 2, diff_opt = 2, km_opt = 4, diff_6th_opt = 0,
 damp_opt = 3, zdamp = 5000, dampcoef = 0.2,
/
&bdy_control
 spec_bdy_width = 5, specified = .true.,
/
&namelist_quilt
 nio_tasks_per_group = 0, nio_groups = 1,
/
"""


def main():
    rows = rows_of(DESIGNS[CONFIG])
    turb = DESIGNS[CONFIG].get('turbine') or rows[0].turbine
    latlons = turbine_latlons(rows)

    write_turbine_tbl(turb, HERE / 'wind-turbine-1.tbl')
    write_windturbines(latlons, HERE / 'windturbines.txt')

    start, end = '2022-01-15_00:00:00', '2022-01-16_12:00:00'
    (HERE / 'namelist.wps').write_text(NAMELIST_WPS.format(
        start=start, end=end, ref_lat=REF_LAT, ref_lon=REF_LON))
    (HERE / 'namelist.input').write_text(NAMELIST_INPUT.format(
        run_hours=36, sy=2022, sm='01', sd=15, sh='00',
        ey=2022, em='01', ed=16, eh=12))

    print(f'Run deck for {CONFIG} ({turb.name}): '
          f'{len(latlons)} turbines, hub {turb.hub_height:.0f} m, '
          f'rotor {turb.rotor_diameter:.0f} m')
    print('  wrote: namelist.wps, namelist.input, wind-turbine-1.tbl, '
          'windturbines.txt')
    print(f'  turbines span lat {min(l for l,_ in latlons):.3f}'
          f'–{max(l for l,_ in latlons):.3f}, '
          f'lon {min(o for _,o in latlons):.3f}–{max(o for _,o in latlons):.3f}')


if __name__ == '__main__':
    main()
