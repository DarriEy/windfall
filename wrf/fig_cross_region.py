#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Cross-region key-result figure: a wind farm shields an OPEN-COAST city
(Reykjavík) ~7x better than a FJORD town (Akureyri), and at far lower
energy cost. Two panels, from the two-region pilot (1/cell, 18x Future-20,
strong-event = severe norðanátt / SW gale). Values per
docs/two_region_shielding_study.md. Writes figures/cross_region_summary.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / 'figures'
AKU, RVK = '#2c7fb8', '#d7301f'

# Shielding by the shield-optimal farm, strong event (surface, hub).
# Akureyri = severe norðanátt (single event); RVK = SW gale.
shield = {
    'Akureyri\n(fjord)':   {'surf': 0.7, 'hub': 1.4, 'hub_lo': 1.4, 'hub_hi': 1.4,
                            'surf_lo': 0.7, 'surf_hi': 0.7},
    'Reykjavík\n(open coast)': {'surf': 5.1, 'hub': 10.6,
                                'hub_lo': 7.3, 'hub_hi': 10.7,    # MYNN/YSU/vert UQ
                                'surf_lo': 1.0, 'surf_hi': 5.1},
}
# Capacity factor: energy-optimal (E) vs shield-optimal (S) siting.
cf = {
    'Akureyri\n(fjord)':   {'E': 24.5, 'S': 14.7},
    'Reykjavík\n(open coast)': {'E': 60.4, 'S': 56.8},
}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5.2))
regions = list(shield)
x = np.arange(len(regions))
w = 0.36

# Panel A — shelter delivered to the town
def errbars(d, key):
    lo = [d[r][f'{key}_lo'] for r in regions]
    hi = [d[r][f'{key}_hi'] for r in regions]
    val = [d[r][key] for r in regions]
    return [np.array(val) - np.array(lo), np.array(hi) - np.array(val)]

surf = [shield[r]['surf'] for r in regions]
hub = [shield[r]['hub'] for r in regions]
axa.bar(x - w/2, surf, w, yerr=errbars(shield, 'surf'), capsize=4,
        color='#74a9cf', label='surface (10 m)', edgecolor='k', lw=0.5)
axa.bar(x + w/2, hub, w, yerr=errbars(shield, 'hub'), capsize=4,
        color='#045a8d', label='hub (165 m)', edgecolor='k', lw=0.5)
for xi, s, h in zip(x, surf, hub):
    axa.text(xi - w/2, s + 0.3, f'{s:.1f}%', ha='center', fontsize=9)
    axa.text(xi + w/2, h + 0.3, f'{h:.1f}%', ha='center', fontsize=9)
axa.set_xticks(x); axa.set_xticklabels(regions)
axa.set_ylabel('downtown wind-speed reduction (%)')
axa.set_title('(a) Shelter delivered by the shield farm\n(strong event, 18×360 MW, 1/cell)')
axa.legend(frameon=False, loc='upper left'); axa.set_ylim(0, 13)
axa.annotate('~7× more\n(wake reaches\nthe city)', xy=(1 + w/2, 10.6),
             xytext=(0.55, 8.5), fontsize=9, color=RVK,
             arrowprops=dict(arrowstyle='->', color=RVK))

# Panel B — energy cost of shield-siting
cfE = [cf[r]['E'] for r in regions]
cfS = [cf[r]['S'] for r in regions]
axb.bar(x - w/2, cfE, w, color='#8c96c6', label='energy-optimal siting',
        edgecolor='k', lw=0.5)
axb.bar(x + w/2, cfS, w, color='#88419d', label='shield-optimal siting',
        edgecolor='k', lw=0.5)
for xi, e, s in zip(x, cfE, cfS):
    axb.text(xi - w/2, e + 0.8, f'{e:.1f}%', ha='center', fontsize=9)
    axb.text(xi + w/2, s + 0.8, f'{s:.1f}%', ha='center', fontsize=9)
    pen = (1 - s / e) * 100
    axb.text(xi, max(e, s) + 4, f'−{pen:.0f}% CF', ha='center',
             fontsize=10, fontweight='bold',
             color='#b30000' if pen > 20 else '#238b45')
axb.set_xticks(x); axb.set_xticklabels(regions)
axb.set_ylabel('capacity factor (%)')
axb.set_title('(b) Energy cost of siting for shelter\n(fjord shield = weak wind; coast = near-uniform)')
axb.legend(frameon=False, loc='center left'); axb.set_ylim(0, 72)

fig.suptitle('Wind-farm-as-shield: geometry is decisive — strong in open coast '
             'at low energy cost, weak in fjords at high cost',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG / 'cross_region_summary.png', dpi=130, bbox_inches='tight')
print(f'wrote {FIG / "cross_region_summary.png"}')
