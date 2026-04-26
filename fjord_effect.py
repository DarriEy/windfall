#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
The Fjord Effect: proving that channeled geometry fundamentally changes
wake persistence, with first-principles derivation of recovery length
and Jensen model comparison.

This is the core scientific contribution of the paper.
"""

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from model import (
    TurbineSpec, FjordGeometry, TurbineRow, WakeParams,
    ChanneledWakeModel, STABILITY_PRESETS,
)
from designs import DESIGNS, EYJAFJORDUR, AKUREYRI, V236

OUT = Path('figures')
RHO = 1.225
KAPPA = 0.4  # von Karman


# ═══════════════════════════════════════════════════════════════════
# 1. JENSEN vs CHANNELED MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════

def jensen_wake_deficit(x, r0, ct, k=0.04):
    """Standard Jensen (1983) free-stream wake deficit at distance x.

    Single turbine, free lateral expansion. No terrain confinement.
    δu/u = (1 - √(1-Ct)) / (1 + k·x/r₀)²
    """
    a = 1 - np.sqrt(1 - ct)
    return a / (1 + k * x / r0) ** 2


def jensen_row_deficit(x, n_turbines, r0, ct, spacing, k=0.04):
    """Jensen deficit for a row of N turbines (sum-of-squares superposition)."""
    total_sq = np.zeros_like(x, dtype=float)
    for i in range(n_turbines):
        d = jensen_wake_deficit(x, r0, ct, k)
        total_sq += d ** 2
    return np.sqrt(total_sq)


def fig_jensen_vs_channeled():
    """The key figure: Jensen (open ocean) vs channeled (fjord) wake."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # JAFNVAEGI configuration
    rows = DESIGNS['C) JAFNVAEGI']['rows']
    x_km = np.linspace(0, 60, 600)

    # --- Jensen model (standard offshore, no channeling) ---
    # For a row of 5 V236 at zone B (x=20 km), compute wake at Akureyri
    r0 = V236.rotor_diameter / 2  # 118 m
    ct_rated = V236.ct_rated  # 0.78

    # Single row of 5 turbines — Jensen at 20 m/s (above rated, Ct drops)
    ct_20 = float(V236.ct(np.array([20.0]))[0])

    # Jensen for each row, combined
    deficit_jensen = np.zeros_like(x_km)
    for row in rows:
        x_from_row = (x_km - row.x_position / 1000) * 1000  # meters
        valid = x_from_row > 0
        d = np.zeros_like(x_km)
        # Sum of squares for N turbines in row
        for _ in range(row.n_turbines):
            d_single = np.where(
                valid,
                jensen_wake_deficit(x_from_row, r0, ct_20, k=0.04),
                0.0)
            d = np.sqrt(d ** 2 + d_single ** 2)
        deficit_jensen = np.sqrt(deficit_jensen ** 2 + d ** 2)

    ax.plot(x_km, deficit_jensen * 100, color='#3498db', linewidth=2.5,
            label='Jensen model (open ocean, k=0.04)')

    # --- Channeled model: three stability regimes ---
    colors = {'neutral': '#95a5a6', 'stable': '#e67e22',
              'very_stable': '#c0392b'}
    labels = {'neutral': 'Channeled: neutral (L=30 km)',
              'stable': 'Channeled: stable (L=55 km)',
              'very_stable': 'Channeled: very stable (L=80 km)'}

    for slab in ['neutral', 'stable', 'very_stable']:
        m = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS[slab])
        res = m.simulate(rows, 20.0)
        deficit_ch = (1 - res['u'] / res['u_in'])
        ax.plot(res['x'] / 1000, deficit_ch * 100, color=colors[slab],
                linewidth=2.5, label=labels[slab])

    # Mark Akureyri
    ax.axvline(x=55, color='black', linewidth=1, linestyle=':')
    ax.annotate('Akureyri', xy=(55.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 8),
                fontsize=11, fontweight='bold')

    # Mark turbine zones
    ax.axvspan(20, 24, alpha=0.05, color='green')
    ax.axvspan(36, 39, alpha=0.05, color='orange')

    # The fjord effect annotation
    # Get values at Akureyri (55 km)
    i_ak = np.argmin(np.abs(x_km - 55))
    jensen_at_ak = deficit_jensen[i_ak] * 100

    m_stable = ChanneledWakeModel(EYJAFJORDUR, STABILITY_PRESETS['stable'])
    res_s = m_stable.simulate(rows, 20.0, target_x=AKUREYRI)
    ch_stable_at_ak = res_s['reduction_pct']

    ratio = ch_stable_at_ak / jensen_at_ak if jensen_at_ak > 0.01 else float('inf')

    ax.annotate(
        f'FJORD EFFECT\n'
        f'Jensen at Akureyri: {jensen_at_ak:.2f}%\n'
        f'Channeled (stable): {ch_stable_at_ak:.1f}%\n'
        f'Amplification: {ratio:.0f}×',
        xy=(42, 5), fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdebd0',
                  edgecolor='#e67e22', alpha=0.9))

    ax.set_xlabel('Distance from fjord mouth (km)', fontsize=12)
    ax.set_ylabel('Wind speed deficit at Akureyri (%)', fontsize=12)
    ax.set_title('The Fjord Effect: Jensen Open-Ocean vs '
                 'Channeled Wake Model\n'
                 '(JAFNVÆGI, 21 turbines, 20 m/s incoming)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 62)
    ax.set_ylim(0, max(10, ch_stable_at_ak * 1.3))

    fig.savefig(OUT / 'fjord_effect.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved {OUT}/fjord_effect.png')
    print(f'  Jensen at Akureyri: {jensen_at_ak:.3f}%')
    print(f'  Channeled (stable): {ch_stable_at_ak:.1f}%')
    print(f'  Fjord amplification: {ratio:.0f}×')


# ═══════════════════════════════════════════════════════════════════
# 2. FIRST-PRINCIPLES DERIVATION OF L
# ═══════════════════════════════════════════════════════════════════

def derive_recovery_length():
    """Derive L from Monin-Obukhov similarity theory."""
    print('\n  Recovery length from first principles:')
    print('  L = u × H²_eff / K_z')
    print('  K_z = κ × u* × z × (1 - z/h_BL) / φ_m(z/L_MO)')
    print('  φ_m = 1 + 5z/L_MO  (stable)')
    print()

    u = 15.0  # m/s at hub
    z = 200.0  # effective height
    z0 = 0.0005
    h_BL = 1000.0
    H_eff = 200.0

    u_star = KAPPA * u / math.log(z / z0)
    Kz_neutral = KAPPA * u_star * z * (1 - z / h_BL)

    print(f'  u* = {u_star:.3f} m/s')
    print(f'  K_z (neutral) = {Kz_neutral:.1f} m²/s')
    print()

    results = []
    print(f'  {"L_MO (m)":>10} {"Regime":<16} {"φ_m":>6} '
          f'{"K_z":>8} {"L (km)":>8} {"Preset":>8}')
    print(f'  {"":->10} {"":->16} {"":->6} {"":->8} {"":->8} {"":->8}')

    for L_MO, regime, preset in [
            (1e6, 'Neutral', 30),
            (500, 'Slightly stable', None),
            (200, 'Stable', 55),
            (100, 'Very stable', 80),
            (50, 'Extremely stable', None)]:
        phi_m = 1 + 5 * z / L_MO
        Kz = Kz_neutral / phi_m
        L_recovery = u * H_eff ** 2 / Kz / 1000  # km

        preset_str = f'{preset} km' if preset else '—'
        print(f'  {L_MO:>10.0f} {regime:<16} {phi_m:>5.1f}  '
              f'{Kz:>6.1f}ms  {L_recovery:>6.0f} km  {preset_str:>8}')

        results.append((L_MO, regime, L_recovery, preset))

    print()
    print('  Key finding: L=80 km (very stable preset) corresponds to')
    print('  L_MO ≈ 100 m, which is CONSERVATIVE — the first-principles')
    print('  derivation gives L ≈ 200+ km for very stable conditions.')
    print('  The presets underestimate wake persistence in stable regimes.')

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))

    L_MO_range = np.logspace(1.5, 4, 100)
    L_recovery = []
    for lmo in L_MO_range:
        phi = 1 + 5 * z / lmo
        kz = Kz_neutral / phi
        L_recovery.append(u * H_eff ** 2 / kz / 1000)
    L_recovery = np.array(L_recovery)

    ax.semilogx(L_MO_range, L_recovery, 'k-', linewidth=2.5)

    # Mark presets
    for L_MO, regime, L_km, preset in results:
        if preset:
            phi = 1 + 5 * z / L_MO
            kz = Kz_neutral / phi
            L_derived = u * H_eff ** 2 / kz / 1000
            color = {'Neutral': '#3498db', 'Stable': '#e67e22',
                     'Very stable': '#c0392b'}[regime]
            ax.plot(L_MO, L_derived, 'o', color=color, markersize=12,
                    zorder=5, label=f'{regime} (derived: {L_derived:.0f} km)')
            ax.plot(L_MO, preset, 's', color=color, markersize=10,
                    markeredgecolor='black', zorder=5)
            ax.annotate(f'Preset: {preset} km', (L_MO, preset),
                        textcoords='offset points', xytext=(15, -5),
                        fontsize=9, color=color)

    ax.set_xlabel('Monin-Obukhov length L_MO (m)', fontsize=11)
    ax.set_ylabel('Recovery length L (km)', fontsize=11)
    ax.set_title('Wake Recovery Length from First Principles\n'
                 '(Monin-Obukhov stability theory, H_eff=200m, u=15 m/s)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which='both')
    ax.set_ylim(0, 300)
    ax.set_xlim(30, 10000)
    ax.invert_xaxis()

    ax.annotate('← more stable          less stable →',
                xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=10, color='gray')

    fig.savefig(OUT / 'recovery_length_derivation.png', dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f'\n  Saved {OUT}/recovery_length_derivation.png')


# ═══════════════════════════════════════════════════════════════════
# 3. STABILITY FREQUENCY FROM WIND DATA PROXY
# ═══════════════════════════════════════════════════════════════════

def stability_frequency():
    """Estimate stability regime frequency using norðanátt speed as proxy.

    Without temperature data, we use 10m wind speed during northerly
    events as a stability proxy:
    - Strong norðanátt (10m > 12 m/s): likely very stable (cold air drainage)
    - Moderate norðanátt (10m 8-12 m/s): likely stable
    - Weak northerly (10m < 8 m/s): neutral or weakly stable
    """
    import carra

    files = sorted(Path('data').glob('winds_*.npz'))
    all_speed = []
    all_dir = []

    for f in files:
        data = np.load(f)
        if 'mouth_speed' in data:
            all_speed.append(data['mouth_speed'])
            all_dir.append(data['mouth_dir'])

    speed = np.concatenate(all_speed)
    direction = np.concatenate(all_dir)
    speed_10m = speed / carra.HEIGHT_FACTOR

    is_north = (direction >= 330) | (direction <= 30)
    total_hours = len(speed) * 6  # 6-hourly data

    # Classify
    weak_north = is_north & (speed_10m < 8)
    moderate_nord = is_north & (speed_10m >= 8) & (speed_10m < 12)
    strong_nord = is_north & (speed_10m >= 12)

    categories = [
        ('All conditions', np.ones(len(speed), dtype=bool)),
        ('Northerly (any speed)', is_north),
        ('Weak northerly (<8 m/s 10m)', weak_north),
        ('Moderate norðanátt (8-12 m/s)', moderate_nord),
        ('Strong norðanátt (>12 m/s)', strong_nord),
    ]

    stability_map = [
        ('Weak northerly', 'Neutral–slight stable', weak_north),
        ('Moderate norðanátt', 'Stable (L≈55 km)', moderate_nord),
        ('Strong norðanátt', 'Very stable (L≈80 km)', strong_nord),
    ]

    print('\n  Stability regime frequency (20-year ensemble):')
    print(f'  {"Category":<35} {"Records":>8} {"Hours/yr":>9} '
          f'{"% time":>7}')
    print(f'  {"":->35} {"":->8} {"":->9} {"":->7}')

    for name, mask in categories:
        n = mask.sum()
        hrs_yr = n * 6 * 365.25 * 24 / total_hours  # normalize to 1 year
        pct = n / len(speed) * 100
        print(f'  {name:<35} {n:>8,} {hrs_yr:>8.0f} {pct:>6.1f}%')

    print()
    print('  Stability regime mapping:')
    print(f'  {"Wind regime":<25} {"Stability":<25} '
          f'{"Hours/yr":>9} {"% time":>7}')
    print(f'  {"":->25} {"":->25} {"":->9} {"":->7}')

    for wind_regime, stab_regime, mask in stability_map:
        n = mask.sum()
        hrs_yr = n * 6 * 365.25 * 24 / total_hours
        pct = n / len(speed) * 100
        print(f'  {wind_regime:<25} {stab_regime:<25} '
              f'{hrs_yr:>8.0f} {pct:>6.1f}%')

    # Effective shielding hours
    print()
    print('  Effective shielding hours:')
    stable_hrs = (moderate_nord.sum() + strong_nord.sum()) * 6 * 365.25 * 24 / total_hours
    vstable_hrs = strong_nord.sum() * 6 * 365.25 * 24 / total_hours
    print(f'  Stable + very stable (L≥55 km): {stable_hrs:.0f} h/yr '
          f'({stable_hrs/8766*100:.1f}%)')
    print(f'  Very stable only (L≥80 km):     {vstable_hrs:.0f} h/yr '
          f'({vstable_hrs/8766*100:.1f}%)')
    print(f'  → Wind shielding is meaningful for {stable_hrs:.0f} hours/year')


# ═══════════════════════════════════════════════════════════════════
# 4. STORM LOAD CALCULATION
# ═══════════════════════════════════════════════════════════════════

def storm_loads():
    """Detailed thrust force analysis for the paper."""
    A = V236.swept_area
    print('\n  Storm Load Analysis (V236-15MW):')
    print(f'  Swept area: {A:,.0f} m²')
    print(f'  T = ½ρAu²Ct')
    print()
    print(f'  {"Mode":<12} {"Speed":>6} {"Ct":>6} {"Thrust":>8} '
          f'{"vs Rated":>9} {"Moment@sea":>11}')
    print(f'  {"":->12} {"":->6} {"":->6} {"":->8} {"":->9} {"":->11}')

    T_rated = 0.5 * RHO * A * 12.5 ** 2 * 0.78

    for mode, speeds in [('Standard', [12.5, 15, 20, 25]),
                          ('High-thrust', [12.5, 15, 20, 25, 30])]:
        for u in speeds:
            if mode == 'Standard':
                ct = float(V236.ct(np.array([u]), high_thrust=False)[0])
            else:
                ct = float(V236.ct(np.array([u]), high_thrust=True)[0])
                if u > V236.cut_out and mode == 'Standard':
                    continue

            T = 0.5 * RHO * A * u ** 2 * ct
            ratio = T / T_rated
            # Overturning moment at sea level (hub height = 150m)
            M = T * V236.hub_height / 1e6  # MN·m

            print(f'  {mode:<12} {u:>4.0f}ms {ct:>5.3f} '
                  f'{T/1e6:>6.2f} MN {ratio:>7.1f}× '
                  f'{M:>9.1f} MNm')

    print()
    print('  Foundation design implications:')
    print(f'  Standard design thrust:     {T_rated/1e6:.2f} MN (at rated)')
    T_ht_25 = 0.5 * RHO * A * 25**2 * 0.78
    T_ht_30 = 0.5 * RHO * A * 30**2 * 0.77
    print(f'  HT mode at 25 m/s:          {T_ht_25/1e6:.2f} MN ({T_ht_25/T_rated:.1f}× rated)')
    print(f'  HT mode at 30 m/s (F20):    {T_ht_30/1e6:.2f} MN ({T_ht_30/T_rated:.1f}× rated)')
    print(f'  → "Shield-class" foundations need {T_ht_25/T_rated:.0f}× the')
    print(f'    lateral capacity of standard floating platforms')
    print(f'  → Comparable to survival loads already designed for')
    print(f'    in 50-year storm conditions (IEC Class I)')


if __name__ == '__main__':
    print('=' * 60)
    print('  THE FJORD EFFECT')
    print('=' * 60)

    print('\n── 1. Jensen vs Channeled Model ──')
    fig_jensen_vs_channeled()

    print('\n── 2. Recovery Length Derivation ──')
    derive_recovery_length()

    print('\n── 3. Stability Frequency ──')
    stability_frequency()

    print('\n── 4. Storm Loads ──')
    storm_loads()

    print('\n  Done.')
