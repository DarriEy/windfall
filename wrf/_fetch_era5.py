import cdsapi
from pathlib import Path
c = cdsapi.Client()
area = [72, -30, 60, -6]   # N, W, S, E  (covers the 9km parent domain)
days = ['01', '02']; times = ['00:00','06:00','12:00','18:00']
out = Path('wrf/met')

print('=== ERA5 pressure levels ===')
c.retrieve('reanalysis-era5-pressure-levels', {
    'product_type': 'reanalysis', 'data_format': 'grib',
    'variable': ['geopotential','temperature','u_component_of_wind',
                 'v_component_of_wind','relative_humidity'],
    'pressure_level': ['1000','975','950','925','900','875','850','825','800',
        '775','750','700','650','600','550','500','450','400','350','300',
        '250','225','200','175','150','125','100','70','50','30','20','10'],
    'year':'2022','month':'01','day':days,'time':times,'area':area,
}, str(out/'era5_pl.grib'))
print('  pl done', (out/'era5_pl.grib').stat().st_size/1e6, 'MB')

print('=== ERA5 single levels ===')
c.retrieve('reanalysis-era5-single-levels', {
    'product_type':'reanalysis','data_format':'grib',
    'variable':['surface_pressure','mean_sea_level_pressure',
        '2m_temperature','2m_dewpoint_temperature',
        '10m_u_component_of_wind','10m_v_component_of_wind',
        'sea_surface_temperature','skin_temperature','land_sea_mask',
        'sea_ice_cover','snow_depth','geopotential',
        'soil_temperature_level_1','soil_temperature_level_2',
        'soil_temperature_level_3','soil_temperature_level_4',
        'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4'],
    'year':'2022','month':'01','day':days,'time':times,'area':area,
}, str(out/'era5_sfc.grib'))
print('  sfc done', (out/'era5_sfc.grib').stat().st_size/1e6, 'MB')
print('ERA5 DOWNLOAD COMPLETE')
