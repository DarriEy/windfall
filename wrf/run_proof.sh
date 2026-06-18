#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Drive one norðanátt proof case end-to-end inside the WRF container,
# twice (turbines OFF then ON), so the Akureyri wake can be compared to
# the 1D model (compare_wrf_1d.py).
#
# Run INSIDE the container with this repo's wrf/ mounted, e.g.:
#   docker run --rm --platform linux/amd64 \
#     -v "$PWD/wrf":/work -v "$WPS_GEOG":/BUILD_WRF/WPS_GEOG \
#     -v "$MET_GRIB":/work/grib windfall/wrf:4.3.3 bash /work/run_proof.sh
#
# Prerequisites (see README.md):
#   - WPS_GEOG static data (geog.sh, ~29 GB) mounted at /BUILD_WRF/WPS_GEOG
#   - ERA5/CARRA met GRIB for the case dates mounted at /work/grib
set -e
WPS=/BUILD_WRF/WPS
WRFRUN=/BUILD_WRF/WRF/run
WORK=/work
VTABLE=${VTABLE:-Vtable.ECMWF}    # ERA5 on pressure levels

cd "$WPS"
cp "$WORK/namelist.wps" .
echo "== geogrid (terrain) =="; ./geogrid.exe
echo "== ungrib (met) =="
ln -sf ungrib/Variable_Tables/$VTABLE Vtable
./link_grib.csh "$WORK"/grib/*
./ungrib.exe
echo "== metgrid =="; ./metgrid.exe

cd "$WRFRUN"
cp "$WORK/namelist.input" .
cp "$WORK"/wind-turbine-1.tbl "$WORK"/windturbines.txt .
ln -sf "$WPS"/met_em.d0* .
echo "== real.exe =="; ./real.exe

for mode in baseline turbines; do
  if [ "$mode" = baseline ]; then
    sed -i 's/^ windfarm_opt = .*/ windfarm_opt = 0, 0, 0,/' namelist.input
  else
    sed -i 's/^ windfarm_opt = .*/ windfarm_opt = 0, 0, 1,/' namelist.input
  fi
  echo "== wrf.exe ($mode) =="; ./wrf.exe
  cp wrfout_d03_* "$WORK/wrfout_d03_${mode}.nc"
done
echo "Done. Outputs: wrfout_d03_baseline.nc, wrfout_d03_turbines.nc"
