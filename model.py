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


# Atmospheric stability determines both vertical mixing rate and
# terrain channeling strength. Stable stratification (common during
# nordanatt) suppresses mixing and traps flow in the fjord, so wakes
# persist much longer and wind reduction at Akureyri is amplified.
STABILITY_PRESETS: Dict[str, WakeParams] = {
    'neutral': WakeParams(
        effective_height=200, recovery_length=30_000,
        channeling_fraction=0.60),
    'stable': WakeParams(
        effective_height=200, recovery_length=55_000,
        channeling_fraction=0.80),
    'very_stable': WakeParams(
        effective_height=200, recovery_length=80_000,
        channeling_fraction=0.92),
}


class ChanneledWakeModel:

    def __init__(self, fjord: FjordGeometry, params: WakeParams = None):
        self.fjord = fjord
        self.params = params or WakeParams()

    def simulate(self, rows: List[TurbineRow], u_in: float,
                 dx: float = 100.0, target_x: float = None) -> dict:
        n = int(self.fjord.length / dx) + 1
        x = np.linspace(0, self.fjord.length, n)
        u = np.full(n, float(u_in))
        L = self.params.recovery_length
        H = self.params.effective_height
        f = self.params.channeling_fraction

        row_data = []
        for row in sorted(rows, key=lambda r: r.x_position):
            idx = min(int(np.searchsorted(x, row.x_position)), n - 1)
            u_loc = u[idx]
            W = self.fjord.width(row.x_position)
            beta = row.total_swept / (W * H)
            ct_val = float(row.turbine.ct(
                np.array([u_loc]), high_thrust=self.params.high_thrust)[0])

            eff = min(ct_val * beta * f, 0.95)
            d0 = 1.0 - np.sqrt(1.0 - eff)

            mask = x > row.x_position
            u[mask] *= 1.0 - d0 * np.exp(-(x[mask] - row.x_position) / L)

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

        result = {'x': x, 'u': u, 'u_in': u_in, 'rows': row_data}
        if target_x is not None:
            ti = min(int(np.searchsorted(x, target_x)), n - 1)
            result['target_u'] = round(float(u[ti]), 2)
            result['reduction_pct'] = round((1 - u[ti] / u_in) * 100, 2)
        return result

    def aep(self, rows, weibull_k=2.0, weibull_A=9.0, hours=8766):
        u_bins = np.arange(0.5, 36.0, 1.0)
        pdf = ((weibull_k / weibull_A)
               * (u_bins / weibull_A) ** (weibull_k - 1)
               * np.exp(-(u_bins / weibull_A) ** weibull_k))
        total = 0.0
        for ub, p in zip(u_bins, pdf):
            if p < 1e-8 or ub < 2:
                continue
            res = self.simulate(rows, float(ub))
            total += sum(r['mw_total'] for r in res['rows']) * hours * float(p)
        cap = sum(r.capacity_mw for r in rows)
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
                'reduction_pct': res['reduction_pct'],
                'power_mw': sum(r['mw_total'] for r in res['rows']),
            })
        return results
