#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parameterised ERA5 fetch for RVK pilot events.
   python wrf/fetch_rvk_event.py <out_label> <month> <day> [<day> ...]
Outputs wrf/met/<label>_pl.grib + _sfc.grib (same vars/levels/area as the
validated _fetch_era5.py, so num_metgrid_levels=33)."""
import sys
from pathlib import Path
import cdsapi

label, month, days = sys.argv[1], sys.argv[2], sys.argv[3:]
area = [72, -30, 60, -6]
times = ['00:00', '06:00', '12:00', '18:00']
out = Path('wrf/met')
c = cdsapi.Client()

print(f'=== {label}: ERA5 pressure levels {month}/{days} ===')
c.retrieve('reanalysis-era5-pressure-levels', {
    'product_type': 'reanalysis', 'data_format': 'grib',
    'variable': ['geopotential', 'temperature', 'u_component_of_wind',
                 'v_component_of_wind', 'relative_humidity'],
    'pressure_level': ['1000', '975', '950', '925', '900', '875', '850',
        '825', '800', '775', '750', '700', '650', '600', '550', '500',
        '450', '400', '350', '300', '250', '225', '200', '175', '150',
        '125', '100', '70', '50', '30', '20', '10'],
    'year': '2022', 'month': month, 'day': days, 'time': times, 'area': area,
}, str(out / f'{label}_pl.grib'))
print('  pl done', (out / f'{label}_pl.grib').stat().st_size / 1e6, 'MB')

print(f'=== {label}: ERA5 single levels ===')
c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis', 'data_format': 'grib',
    'variable': ['surface_pressure', 'mean_sea_level_pressure',
        '2m_temperature', '2m_dewpoint_temperature',
        '10m_u_component_of_wind', '10m_v_component_of_wind',
        'sea_surface_temperature', 'skin_temperature', 'land_sea_mask',
        'sea_ice_cover', 'snow_depth', 'geopotential',
        'soil_temperature_level_1', 'soil_temperature_level_2',
        'soil_temperature_level_3', 'soil_temperature_level_4',
        'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4'],
    'year': '2022', 'month': month, 'day': days, 'time': times, 'area': area,
}, str(out / f'{label}_sfc.grib'))
print('  sfc done', (out / f'{label}_sfc.grib').stat().st_size / 1e6, 'MB')
print(f'{label} ERA5 DOWNLOAD COMPLETE')
