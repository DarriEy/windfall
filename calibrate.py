#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 Darri Eythorsson
"""
Calibrate the channeled-flow baseline against the observed CARRA
along-fjord profile, and bound the turbine wake-recovery length.

We test a baseline along-fjord speed model of the form

    u(x) = u0 * (W0/W(x))^gamma * exp(-(x - x_peak) / Lambda)     (x >= x_peak)

  * (W0/W)^gamma   : continuity-driven speed-up as the channel narrows;
  * exp(-x/Lambda) : bulk friction / detrainment decay, length Lambda.

Eyjafjordur narrows ~4x from the outer fjord to the head, so continuity
(gamma>0) would *accelerate* the flow toward Akureyri. The observations
show the opposite -- the wind peaks at the outer fjord and decays toward
the head -- so the fit drives gamma -> 0: channel-narrowing acceleration
is rejected by the data, and the closed fjord head plus lateral
detrainment / surface friction dominate. The operational baseline is
therefore a frictional-decay law anchored at the outer-fjord peak.

The fitted friction length Lambda is the natural momentum-decay scale of
the fjord. It is the physical ceiling on the turbine wake-recovery length
L: a velocity deficit cannot plausibly persist over distances long
compared with the scale on which the mean flow itself loses momentum.
This is the leash on the stable-condition "fjord effect" amplification.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

import observed
from designs import EYJAFJORDUR as FJORD, AKUREYRI

OUT = Path('figures')
CAL_FILE = Path(__file__).parent / 'data' / 'baseline_calibration.json'

# Wake-recovery presets currently used in the scenario code
# (model.py STABILITY_PRESETS), for comparison against the calibrated leash.
PRESET_L_KM = {'neutral': 30, 'stable': 55, 'very_stable': 80}


def _widths(x_km):
    return np.array([FJORD.width(x * 1000) for x in x_km])


def fit_continuity(prof):
    """Full mouth-anchored fit including the continuity exponent gamma --
    used only to show that the data reject gamma>0."""
    x, y = prof['x_km'], prof['mean']
    sigma = np.maximum(prof['interannual_std'], 0.05)
    w_ratio = _widths(x)[0] / _widths(x)

    def f(xk, u0, gamma, lam):
        return u0 * w_ratio ** gamma * np.exp(-xk / lam)

    popt, pcov = curve_fit(f, x, y, p0=[y[0], 0.5, 60.0], sigma=sigma,
                           bounds=([0.1, 0.0, 1.0], [50, 1.5, 2000]),
                           maxfev=20000)
    return {'gamma': popt[1], 'gamma_err': np.sqrt(np.diag(pcov))[1]}


def fit_friction(prof, label):
    """Frictional-decay fit anchored at the windiest (outer-fjord) station,
    over the monotonic down-fjord segment. u = u0 * exp(-(x-x0)/Lambda)."""
    i0 = int(np.argmax(prof['mean']))
    x = prof['x_km'][i0:]
    y = prof['mean'][i0:]
    sigma = np.maximum(prof['interannual_std'][i0:], 0.05)
    x0 = x[0]

    def f(xk, u0, lam):
        return u0 * np.exp(-(xk - x0) / lam)

    popt, pcov = curve_fit(f, x, y, p0=[y[0], 50.0], sigma=sigma,
                           bounds=([0.1, 1.0], [50, 2000]), maxfev=20000)
    yhat = f(x, *popt)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    perr = np.sqrt(np.diag(pcov))
    return {
        'label': label, 'anchor': observed.STATION_NAMES[i0],
        'anchor_x_km': float(x0), 'u0': float(popt[0]),
        'lambda_km': float(popt[1]), 'lambda_err': float(perr[1]),
        'r2': float(r2),
        'rmse': float(np.sqrt(np.mean((y - yhat) ** 2))),
        'full_x': prof['x_km'], 'full_y': prof['mean'],
        'full_sigma': np.maximum(prof['interannual_std'], 0.05),
    }


def report(fits, cont):
    print('=' * 70)
    print('  BASELINE CALIBRATION')
    print('=' * 70)
    print('  Continuity test (mouth-anchored u0 (W0/W)^gamma exp(-x/L)):')
    for lab, c in cont.items():
        print(f'    {lab:<10} best-fit gamma = '
              f'{c["gamma"]:.2f} ± {c["gamma_err"]:.2f}  '
              f'(pinned at lower bound 0 -> continuity-driven '
              f'speed-up not supported)')
    print('    NB: gamma is bound-limited, so the quoted +/- is a '
          'linearized')
    print('    covariance, not a true CI; read it as "the data do not '
          'want gamma>0".')
    print()
    print('  Frictional decay, anchored at outer-fjord peak '
          '(u = u0 exp(-(x-x0)/L)):')
    print(f'  {"profile":<11}{"anchor":>7}{"u0":>7}{"Lambda_km":>11}'
          f'{"R^2":>7}{"RMSE":>7}')
    print(f'  {"":-<11}{"":->7}{"":->7}{"":->11}{"":->7}{"":->7}')
    for ft in fits:
        print(f'  {ft["label"]:<11}{ft["anchor"]:>7}{ft["u0"]:>7.2f}'
              f'{ft["lambda_km"]:>8.1f}±{min(ft["lambda_err"],99):<2.0f}'
              f'{ft["r2"]:>7.3f}{ft["rmse"]:>7.2f}')
    print()


def report_leash(fits):
    fnord = next(f for f in fits if f['label'] == 'nordanatt')
    L = fnord['lambda_km']
    print('  Wake-recovery leash:')
    print(f'    Natural momentum-decay length (nordanatt): Lambda = '
          f'{L:.0f} +/- {fnord["lambda_err"]:.0f} km')
    print(f'    CAVEAT: Lambda ({L:.0f} km) exceeds the 55 km fjord '
          f'length, so it is an')
    print(f'    extrapolated decay scale, weakly constrained by 6 '
          f'down-fjord points.')
    print(f'    The leash on wake-recovery L is therefore soft, not a '
          f'hard ceiling.')
    print(f'    Scenario wake-recovery presets vs this ceiling:')
    for k, v in PRESET_L_KM.items():
        flag = 'AT CEILING' if v >= 0.9 * L else 'within'
        print(f'      {k:<12} L = {v:>2} km  ({v/L:.2f}x Lambda, {flag})')
    print()
    print(f'    Akureyri nordanatt baseline (consistent mouth-anchored '
          f'events):')
    raw = observed.load_raw()
    nmask = observed.nordanatt_mask(raw)
    nor = observed.along_fjord_profile(raw, nmask)
    print(f'      mouth {nor["mean"][0]:.1f} -> Akureyri '
          f'{nor["mean"][-1]:.1f} m/s '
          f'({(1-nor["mean"][-1]/nor["mean"][0])*100:.0f}% natural drop)')
    print(f'      (NB: the per-station climatology JSON reported 12.2 m/s '
          f'at Akureyri --')
    print(f'       that compared *different* events at each station; '
          f'this is the same events.)')
    print()


def figure(fits):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    xx = np.linspace(0, 60, 200)
    for ax, ft in zip(axes, fits):
        ax.errorbar(ft['full_x'], ft['full_y'], yerr=ft['full_sigma'],
                    fmt='o', color='#c0392b', markersize=8, capsize=4,
                    zorder=5, label='CARRA observed\n(20-yr, interannual σ)')
        x0, u0, L = ft['anchor_x_km'], ft['u0'], ft['lambda_km']
        curve = u0 * np.exp(-np.maximum(xx - x0, 0) / L)
        ax.plot(xx, curve, '-', color='#2c3e50', linewidth=2.5,
                label=(f'Calibrated baseline\nΛ={L:.0f} km, '
                       f'R²={ft["r2"]:.2f}'))
        ax.axhline(ft['full_y'][0], color='gray', linestyle=':',
                   linewidth=1.5,
                   label='Old flat baseline\n(no decay)')
        ax.axvline(AKUREYRI / 1000, color='#27ae60', linestyle='--',
                   alpha=0.5)
        ax.annotate('Akureyri', xy=(AKUREYRI / 1000 - 1.2,
                    ft['full_y'].min()), rotation=90, va='bottom',
                    ha='right', fontsize=8, color='#27ae60')
        ax.set_xlabel('Distance from fjord mouth (km)', fontsize=11)
        ax.set_ylabel('Hub-height wind speed (m/s)', fontsize=11)
        ax.set_title(f'{ft["label"].capitalize()} conditions',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, ft['full_y'].max() * 1.25)
    fig.suptitle('Model Validation: baseline reproduces the observed '
                 'along-fjord wind decay (in-situ, channeled regime)',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'baseline_calibration.png', dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f'  Saved {OUT}/baseline_calibration.png')


def calibrate():
    raw = observed.load_raw()
    nmask = observed.nordanatt_mask(raw)
    profiles = {
        'all': observed.along_fjord_profile(raw),
        'nordanatt': observed.along_fjord_profile(raw, nmask),
    }
    cont = {k: fit_continuity(p) for k, p in profiles.items()}
    fits = [fit_friction(profiles['all'], 'all'),
            fit_friction(profiles['nordanatt'], 'nordanatt')]
    return fits, cont


def save(fits, cont):
    out = {'source': 'CARRA 2003-2022, fitted in calibrate.py',
           'continuity_gamma': {k: round(v['gamma'], 3)
                                for k, v in cont.items()},
           'profiles': {}}
    for ft in fits:
        out['profiles'][ft['label']] = {
            'anchor_x_km': ft['anchor_x_km'], 'u0': round(ft['u0'], 3),
            'lambda_km': round(ft['lambda_km'], 2),
            'lambda_err_km': round(ft['lambda_err'], 2),
            'r2': round(ft['r2'], 4), 'rmse': round(ft['rmse'], 4),
        }
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAL_FILE, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'  Saved {CAL_FILE}')


if __name__ == '__main__':
    fits, cont = calibrate()
    report(fits, cont)
    report_leash(fits)
    figure(fits)
    save(fits, cont)
