#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Channeled-flow wind farm wake model for fjord environments.

Velocity deficit behind a turbine row in a partially-blocked channel:
  d = 1 - sqrt(1 - Ct * beta * f)
where Ct = thrust coefficient, beta = swept_area / channel_cross_section,
f = channeling fraction (terrain confinement).

Downstream recovery from vertical turbulent mixing:
  d(x) = d0 * exp(-x / L)
where L ~ H_eff^2 / (K_z/u), typically 20-60 km in fjord geometry.

Multiple rows compound -- each sees the reduced inflow from upstream.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class TurbineSpec:
    name: str
    rated_power_mw: float
    rotor_diameter: float
    hub_height: float
    cut_in: float
    rated_speed: float
    cut_out: float
    ct_rated: float

    @property
    def swept_area(self):
        return np.pi * (self.rotor_diameter / 2) ** 2

    def power(self, u):
        u = np.atleast_1d(np.float64(u))
        p = np.zeros_like(u)
        below = (u >= self.cut_in) & (u < self.rated_speed)
        full = (u >= self.rated_speed) & (u <= self.cut_out)
        p[below] = self.rated_power_mw * (
            (u[below] - self.cut_in) / (self.rated_speed - self.cut_in)
        ) ** 3
        p[full] = self.rated_power_mw
        return p

    def ct(self, u, high_thrust=False):
        u = np.atleast_1d(np.float64(u))
        c = np.zeros_like(u)
        below = (u >= self.cut_in) & (u < self.rated_speed)
        above = (u >= self.rated_speed) & (u <= self.cut_out)
        c[below] = self.ct_rated
        if high_thrust:
            # Wind-shield mode: maintain rated Ct above rated speed
            # at cost of higher structural loads and some power reduction
            c[above] = self.ct_rated
        else:
            c[above] = self.ct_rated * (self.rated_speed / u[above]) ** 2
        return c


@dataclass
class Constriction:
    """Island or headland that narrows the effective channel."""
    x_position: float   # m from mouth
    width_reduction: float  # m, how much narrower
    influence_radius: float = 3000.0  # m, Gaussian spread

    def reduction_at(self, x):
        dist = abs(x - self.x_position)
        if dist > 3 * self.influence_radius:
            return 0.0
        return self.width_reduction * np.exp(
            -dist ** 2 / (2 * self.influence_radius ** 2))


@dataclass
class FjordGeometry:
    name: str
    control_points: list
    ridge_height: float
    length: float
    constrictions: List[Constriction] = field(default_factory=list)

    def width(self, x):
        xs, ws = zip(*self.control_points)
        w = float(np.interp(x, xs, ws))
        for c in self.constrictions:
            w -= c.reduction_at(x)
        return max(w, 500.0)

    def width_array(self, x):
        xs, ws = zip(*self.control_points)
        w = np.interp(x, xs, ws)
        for c in self.constrictions:
            w = w - c.width_reduction * np.exp(
                -(x - c.x_position) ** 2
                / (2 * c.influence_radius ** 2))
        return np.maximum(w, 500.0)


@dataclass
class TurbineRow:
    x_position: float
    n_turbines: int
    turbine: TurbineSpec

    @property
    def total_swept(self):
        return self.n_turbines * self.turbine.swept_area

    @property
    def capacity_mw(self):
        return self.n_turbines * self.turbine.rated_power_mw


@dataclass
class WakeParams:
    effective_height: float = 200.0
    recovery_length: float = 30_000.0
    channeling_fraction: float = 0.7
    high_thrust: bool = False
    # Calibrated frictional-decay baseline (see calibrate.py). When
    # baseline_length is set, the no-turbine flow decays down-fjord as
    # u_base(x) = u_in * exp(-(x - anchor)+ / baseline_length), matching
    # the observed CARRA along-fjord profile. Turbine deficits are then
    # applied as a perturbation on top of this baseline, so the reported
    # "turbine reduction" is the marginal effect, separate from the large
    # natural sheltering the fjord already provides.
    baseline_length: Optional[float] = None     # m; None -> flat baseline
    baseline_anchor: float = 10_000.0            # m; outer-fjord wind peak
    # Multi-row wake-deficit combination. 'product' compounds deficits
    # multiplicatively (each row sees the reduced inflow from upstream);
    # 'sos' combines them in quadrature (Katic/Jensen sum-of-squares,
    # each row referenced to the local baseline). The two bracket a real
    # structural uncertainty and are compared in uncertainty.py.
    superposition: str = 'product'


# Natural momentum-decay length of Eyjafjordur from calibrate.py
# (CARRA 2003-2022): ~84 km under nordanatt, ~88 km all-conditions.
# This is the physical ceiling on the turbine wake-recovery length.
CALIBRATED_BASELINE_LENGTH = 84_300.0  # m (nordanatt fit)
CALIBRATED_BASELINE_ANCHOR = 10_000.0  # m (outer-fjord wind peak)
CALIBRATION_FILE = Path(__file__).parent / 'data' / 'baseline_calibration.json'


def load_calibration(profile='nordanatt'):
    """Calibrated baseline parameters as a dict with keys 'length' (m),
    'anchor' (m) and 'source' (provenance string).

    Fails *loudly*: if the calibration file is missing, it warns and
    falls back to the hard-coded defaults (so a stale checkout still
    runs, but the user is told). If the file is present but malformed or
    missing the requested profile, the underlying exception propagates —
    a bad file is never silently treated as a valid calibrated run.
    """
    import json
    import warnings
    if not CALIBRATION_FILE.exists():
        warnings.warn(
            f'Calibration file {CALIBRATION_FILE.name} not found; using '
            f'hard-coded Lambda={CALIBRATED_BASELINE_LENGTH/1000:.0f} km, '
            f'anchor={CALIBRATED_BASELINE_ANCHOR/1000:.0f} km. '
            f'Run `python calibrate.py` to regenerate.', stacklevel=2)
        return {'length': CALIBRATED_BASELINE_LENGTH,
                'anchor': CALIBRATED_BASELINE_ANCHOR,
                'source': 'hard-coded default (no calibration file)'}
    with open(CALIBRATION_FILE) as fh:
        cal = json.load(fh)                 # malformed JSON -> raises
    prof = cal['profiles'][profile]         # missing profile -> raises
    return {
        'length': prof['lambda_km'] * 1000.0,
        'anchor': prof.get('anchor_x_km', 10.0) * 1000.0,
        'source': f"{cal.get('source', '?')} [{profile}: "
                  f"Lambda={prof['lambda_km']}km, R2={prof.get('r2')}]",
    }


def load_calibrated_baseline(profile='nordanatt'):
    """Calibrated baseline decay length in metres (back-compat shim)."""
    return load_calibration(profile)['length']


def marginal_reduction(result):
    """(speed_pct, pressure_pct) reduction attributable to the turbines,
    measured against the no-turbine baseline at the target — the honest
    'turbine benefit'. Use this instead of recomputing against u_in,
    which would also count the fjord's natural sheltering."""
    ub = result.get('baseline_u', result['u_in'])
    ut = result['target_u']
    sr = result.get('turbine_reduction_pct',
                    (1 - ut / result['u_in']) * 100 if result['u_in'] else 0)
    pr = (1 - (ut / ub) ** 2) * 100 if ub > 0 else 0.0
    return sr, pr


# Atmospheric stability determines both vertical mixing rate and
# terrain channeling strength. Stable stratification (common during
# nordanatt) suppresses mixing and traps flow in the fjord, so wakes
# persist much longer and wind reduction at Akureyri is amplified.
_CAL = load_calibration()
_BASE = _CAL['length']
_ANCHOR = _CAL['anchor']
STABILITY_PRESETS: Dict[str, WakeParams] = {
    'neutral': WakeParams(
        effective_height=200, recovery_length=30_000,
        channeling_fraction=0.60,
        baseline_length=_BASE, baseline_anchor=_ANCHOR),
    'stable': WakeParams(
        effective_height=200, recovery_length=55_000,
        channeling_fraction=0.80,
        baseline_length=_BASE, baseline_anchor=_ANCHOR),
    'very_stable': WakeParams(
        effective_height=200, recovery_length=80_000,
        channeling_fraction=0.92,
        baseline_length=_BASE, baseline_anchor=_ANCHOR),
}


# ── WRF-anchored recalibration ───────────────────────────────────
# The presets above assume L = 30-80 km. The WRF 1 km runs
# (wrf/fit_recovery_length.py) show the turbine wake recovers far
# faster: an open-fjord fit gives L = 10 km (R2 = 0.83 over ~40 km of
# downstream fetch), a near-head fit ~2 km. We adopt the open-fjord
# value as the WRF-anchored recovery length. Combined with the CARRA
# pressure-level channeling fraction (f = 0.31, ridge_stability.py),
# this re-anchors the reduced-order model to the mesoscale evidence,
# making it a fast, WRF-consistent tool for placement search. The long-L
# presets are kept above only for comparison with the original study.
WRF_RECOVERY_LENGTH = 10_000.0     # m (open-fjord WRF fit, fallback)
WRF_CHANNELING_FRACTION = 0.31     # CARRA pressure-level ridge N
WRF_RECOVERY_FILE = Path(__file__).parent / 'data' / 'wrf_recovery_calibration.json'


def load_wrf_recovery():
    """WRF-derived open-fjord recovery length L (m), from
    wrf/fit_recovery_length.py. Warns and falls back to the hard-coded
    WRF_RECOVERY_LENGTH if the calibration file is absent; a present but
    malformed file raises (a bad file is never silently trusted)."""
    import json
    import warnings
    if not WRF_RECOVERY_FILE.exists():
        warnings.warn(
            f'WRF recovery calibration {WRF_RECOVERY_FILE.name} not found; '
            f'using hard-coded L = {WRF_RECOVERY_LENGTH/1000:.0f} km. Run '
            f'`python wrf/fit_recovery_length.py` to regenerate.', stacklevel=2)
        return WRF_RECOVERY_LENGTH
    res = json.load(open(WRF_RECOVERY_FILE))      # malformed -> raises
    outer = res.get('outer') or {}                # open-fjord fit preferred
    L_km = outer.get('L_km')
    return L_km * 1000.0 if L_km else WRF_RECOVERY_LENGTH


def wrf_calibrated_presets():
    """STABILITY_PRESETS re-anchored to the WRF recovery length and the
    data-derived channeling fraction f — the mesoscale-consistent
    configuration for placement search. Baseline (natural sheltering) is
    unchanged from the observed CARRA calibration."""
    from dataclasses import replace
    L = load_wrf_recovery()
    return {name: replace(p, recovery_length=L,
                          channeling_fraction=WRF_CHANNELING_FRACTION)
            for name, p in STABILITY_PRESETS.items()}


class ChanneledWakeModel:

    def __init__(self, fjord: FjordGeometry, params: WakeParams = None):
        self.fjord = fjord
        self.params = params or WakeParams()

    def simulate(self, rows: List[TurbineRow], u_in: float,
                 dx: float = 100.0, target_x: float = None) -> dict:
        n = int(self.fjord.length / dx) + 1
        x = np.linspace(0, self.fjord.length, n)
        # Natural (no-turbine) baseline: frictional decay down-fjord,
        # calibrated to CARRA (calibrate.py). Falls back to a flat
        # profile when baseline_length is unset.
        # NB: the mouth->anchor segment is held flat at u_in, whereas the
        # observed profile peaks ~5% above the mouth at the outer fjord.
        # This slightly understates outer-fjord (Zone A) generation, i.e.
        # it is conservative for the shielding-vs-generation LCOE gap.
        if self.params.baseline_length:
            u_base = float(u_in) * np.exp(
                -np.maximum(x - self.params.baseline_anchor, 0.0)
                / self.params.baseline_length)
        else:
            u_base = np.full(n, float(u_in))
        u = u_base.copy()
        L = self.params.recovery_length
        H = self.params.effective_height
        f = self.params.channeling_fraction

        sos = getattr(self.params, 'superposition', 'product') == 'sos'
        sq_deficit = np.zeros(n)   # accumulated squared deficit (SoS mode)
        clip_hits = 0              # rows where Ct·β·f saturated the cap

        row_data = []
        for row in sorted(rows, key=lambda r: r.x_position):
            idx = min(int(np.searchsorted(x, row.x_position)), n - 1)
            # Product superposition lets a row see the wake-reduced inflow
            # of upstream rows; SoS references each row to the baseline.
            u_loc = u_base[idx] if sos else u[idx]
            W = self.fjord.width(row.x_position)
            beta = row.total_swept / (W * H)
            ct_val = float(row.turbine.ct(
                np.array([u_loc]), high_thrust=self.params.high_thrust)[0])

            eff_raw = ct_val * beta * f
            if eff_raw > 0.95:
                clip_hits += 1
            eff = min(eff_raw, 0.95)
            d0 = 1.0 - np.sqrt(1.0 - eff)

            mask = x > row.x_position
            decay = d0 * np.exp(-(x[mask] - row.x_position) / L)
            if sos:
                sq_deficit[mask] += decay ** 2
            else:
                u[mask] *= 1.0 - decay

            pw = float(row.turbine.power(np.array([u_loc]))[0])
            row_data.append({
                'x_km': row.x_position / 1000,
                'n': row.n_turbines,
                'u_ms': round(u_loc, 2),
                'mw_each': round(pw, 2),
                'mw_total': round(pw * row.n_turbines, 1),
                'ct': round(ct_val, 3),
                'blockage': round(beta, 4),
                'deficit_pct': round(d0 * 100, 2),
            })

        if sos:
            u = u_base * (1.0 - np.sqrt(np.minimum(sq_deficit, 1.0)))

        result = {'x': x, 'u': u, 'u_base': u_base, 'u_in': u_in,
                  'rows': row_data, 'clip_hits': clip_hits}
        if target_x is not None:
            ti = min(int(np.searchsorted(x, target_x)), n - 1)
            result['target_u'] = round(float(u[ti]), 2)
            result['baseline_u'] = round(float(u_base[ti]), 2)
            # Total reduction vs the synoptic inflow at the mouth, kept for
            # backward compatibility, plus its decomposition into the
            # natural fjord sheltering and the marginal turbine effect.
            result['reduction_pct'] = round((1 - u[ti] / u_in) * 100, 2)
            result['natural_reduction_pct'] = round(
                (1 - u_base[ti] / u_in) * 100, 2)
            result['turbine_reduction_pct'] = round(
                (1 - u[ti] / u_base[ti]) * 100, 2) if u_base[ti] > 0 else 0.0
        return result

    @staticmethod
    def _weibull_pdf(u, k, A):
        return (k / A) * (u / A) ** (k - 1) * np.exp(-(u / A) ** k)

    def aep(self, rows, weibull_k=2.0, weibull_A=9.0, hours=8766,
            station_weibull=None):
        """Annual energy production.

        Default: drive the farm with the mouth Weibull (weibull_k/A) and
        let the calibrated baseline decay the inflow down-fjord — this
        keeps the mouth *shape* everywhere.

        station_weibull (a callable x_position_m -> (k, A)): use each
        turbine's own local resource distribution instead. The natural
        down-fjord decay is then carried by the per-station A, so the
        baseline decay is switched OFF for the energy integral (using it
        too would double-count the decay) and only the turbine-to-turbine
        array wake is applied.
        """
        u_bins = np.arange(0.5, 36.0, 1.0)
        cap = sum(r.capacity_mw for r in rows)

        if station_weibull is None:
            pdf = self._weibull_pdf(u_bins, weibull_k, weibull_A)
            total = 0.0
            for ub, p in zip(u_bins, pdf):
                if p < 1e-8 or ub < 2:
                    continue
                res = self.simulate(rows, float(ub))
                total += sum(r['mw_total']
                             for r in res['rows']) * hours * float(p)
        else:
            # Array wake only (baseline off): row i's waked speed at an
            # ambient ub is ub * array_factor_i. Each row's ambient is
            # drawn from its own local Weibull.
            from dataclasses import replace
            flat = ChanneledWakeModel(
                self.fjord, replace(self.params, baseline_length=None))
            srows = sorted(rows, key=lambda r: r.x_position)
            total = 0.0
            for ub in u_bins:
                if ub < 2:
                    continue
                res = flat.simulate(srows, float(ub))
                for row, rd in zip(srows, res['rows']):
                    k_r, A_r = station_weibull(row.x_position)
                    p = self._weibull_pdf(ub, k_r, A_r)
                    if p < 1e-8:
                        continue
                    waked = rd['u_ms']          # ub * array attenuation
                    pw = float(row.turbine.power(np.array([waked]))[0])
                    total += pw * row.n_turbines * hours * float(p)

        cf = total / (cap * hours) if cap > 0 else 0
        return {
            'aep_gwh': round(total / 1000, 1),
            'capacity_mw': cap,
            'cf_pct': round(cf * 100, 1),
        }

    def sweep_speeds(self, rows, target_x, speeds):
        results = []
        for u in speeds:
            res = self.simulate(rows, float(u), target_x=target_x)
            results.append({
                'u_in': u,
                'u_target': res['target_u'],
                'baseline_u': res.get('baseline_u', u),
                'reduction_pct': res['reduction_pct'],
                'natural_reduction_pct': res.get('natural_reduction_pct', 0.0),
                'turbine_reduction_pct': res.get(
                    'turbine_reduction_pct', res['reduction_pct']),
                'power_mw': sum(r['mw_total'] for r in res['rows']),
            })
        return results
