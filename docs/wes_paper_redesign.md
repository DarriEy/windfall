# WES paper — rethought from scratch (2026-06)

The original paper asked "does the Eyjafjörður fjord channel turbine wakes to
shelter Akureyri?" and answered yes (the "Fjord Effect"). Every load-bearing
claim of that paper is now overturned by the WRF work. This is not a revision —
it is a different, and stronger, paper. Below is the from-scratch design.

## Working title
**"When do wind-farm wakes shelter a community? A mesoscale comparison of
confined-fjord and open-coast terrain in Iceland."**
(Alt: "Wind farms as community wind-reduction infrastructure: why confined
terrain leaks the shelter and open coast holds it.")

## One-sentence thesis
Turbine wakes *can* shelter a downstream community, but whether they do is set
by terrain and turbine cut-out — not by the channeling the original hypothesis
relied on: confined fjords *leak* the shelter over their ridges while suiting
generation, open coasts *hold* a coherent wake, and the robust prize everywhere
is reduction of the everyday wind, not the catastrophic storm.

## Abstract (draft)
Wind-turbine wakes lower the wind downstream, raising the question of whether a
wind farm could be sited deliberately to shelter a community — from storms or
from chronic wind. We test this with WRF-ARW simulations (nested to 1 km, Fitch
wind-farm parameterization, ERA5-forced) at two contrasting north-Iceland sites:
the narrow ridge-flanked Eyjafjörður fjord (town Akureyri at its head) and the
open coast of the capital, Reykjavík. A reduced-order channeled-flow model,
calibrated to 20 yr of CARRA reanalysis, had predicted the fjord would prolong
the wake and shelter the fjord head. The mesoscale model overturns this on two
counts. (1) The wake-recovery length is ~10 km, not the 55-80 km assumed. (2)
Decisively, the fjord's confinement is a *liability*: a wall blocking the
low-level jet forces the air up and over the bounding ridges (turbine-induced
speed-up to 71 % on the 600-1200 m flanks), which descends and refills the
wake — the fjord leaks the shelter over its own walls, capping the reduction at
the town near 16 % even for a 1.3 GW array. In open coastal terrain the wake
stays coherent (<=7 % diversion) and the deficit survives: a wide offshore array
cuts city wind pressure ~20-40 % during a sea-sector gale, uniformly across the
metro. Turbine cut-out (25 m/s) further switches the shelter off in the
strongest storms — the capital's downslope easterly gales reach 32 m/s at hub
and feather the rotors — so deliberate storm-shielding is bounded to gales that
arrive below cut-out. The robust, achievable benefit is reduction of the
*everyday* wind: turbines operate ~84 % of hours at high thrust, an upwind
array cuts the prevailing-direction city wind ~29 %, and the annual-mean wind
falls ~6 % per gigawatt. Shelter and generation are therefore antagonistic and
terrain-selective: confined fjords suit generation (the channeled floor jet)
but leak shelter; open coasts hold a coherent wake and can deliver uniform
city-wide everyday-wind reduction.

## Section structure
1. **Introduction** — momentum extraction -> wakes -> measured large-farm wakes
   (Christiansen & Hasager 2005; Platis 2018); the new question of *deliberate*
   community sheltering; Iceland's wind exposure; the fjord hypothesis as the
   thing being tested; contributions (two-terrain mesoscale test; the diversion
   mechanism; the cut-out limit; everyday-reduction as the real prize).
2. **Sites, data, methods**
   2.1 Two sites + CARRA climatology (Eyjafjörður norðanátt; Reykjavík rose
       SW31/E29; 84 % of gales stable — stability is not the limiter).
   2.2 Reduced-order channeled-flow model + CARRA baseline calibration (the
       hypothesis under test), now WRF-recalibrated (L~10 km).
   2.3 WRF-ARW config: 9/3/1 km nests on each site, Fitch scheme, ERA5 events;
       turbine curves incl. the cut-out/feather table (note the segfault-on-
       overrun lesson as a methods caveat).
3. **The reduced-order prediction** — calibrated natural sheltering (R^2 0.85-
   0.92); the persistence assumption and the ~20 % shielding it predicts.
4. **Mesoscale results**
   4.1 Wake recovery is short (~10 km), both sites; recalibration of L.
   4.2 The fjord leaks the shelter: diversion mechanism (ridge speed-up map);
       density curve to the 16 % ceiling; matched-thrust fjord-vs-coast gap.
   4.3 Open coast holds the wake: Reykjavík SW shield (41 %); wide-wall uniform
       city-wide coverage (BREIDVEGGUR) and its scaling.
   4.4 The cut-out limit: ΔP vs wind speed; E-gale 32 m/s feathers -> 2-5 %,
       SW-gale 18 m/s operates -> 41 %.
   4.5 Everyday-wind reduction: high-Ct operating regime; -29 % in-sector;
       annual-mean ~6 %/GW (single wall) to ~12-14 % (two-arc ring).
5. **Design implications**
   5.1 Shelter vs generation are antagonistic and terrain-selective.
   5.2 Optimized configs from the multi-agent + WRF design study: fjord
       generation = floor jet (22 m/s), open-coast wide shield, everyday ring;
       what was ruled out (shoulder harvest, trigger-wall hybrid, thin tiers,
       feather-recovery) and why.
   5.3 Economics sketch (capacity vs benefit; both extremes unshieldable).
6. **Discussion & limits** — single design event per case (multi-event campaign
   is the decisive next step); Fitch idealization; generalization to other
   fjords/coasts; the corrected status of the original "Fjord Effect."
7. **Conclusions.**

## Figures (new set)
1. Two-site map + CARRA wind roses.
2. Reduced-order baseline calibration (along-fjord profile, R^2).
3. 1D vs WRF along-fjord ΔP — the persistence failure + L recalibration.
4. **The diversion mechanism** — fjord ridge speed-up map (the shelter leaking
   over the walls). The signature figure.
5. Akureyri shield vs capacity (density curve to the 16 % ceiling).
6. Reykjavík: SW clean-wake shield field + wide-wall uniform-coverage map.
7. Cut-out/feathering: ΔP vs hub wind (storm-shield collapse > 25 m/s).
8. Everyday-wind reduction: ΔP vs hub wind (high-Ct) + annual-mean per-GW.
9. Design-implications schematic: generator-in-fjord vs shield-on-coast.

## What carries over vs what is cut
- KEEP: CARRA climatology, baseline calibration, the reduced-order model (now as
  the *hypothesis under test*, not the result), the uncertainty mindset.
- CUT/REFRAME: the "Fjord Effect shields Akureyri" headline, the 4-6x
  amplification, the LCOE-of-shielding economics (the shielding it priced is
  largely illusory at the fjord head), SAMSETT-as-shield framing.
- ADD: the two-site WRF study, the diversion mechanism, the cut-out limit, the
  everyday-reduction result, the design-optimization synthesis.

## Honest framing
This is a *negative-result-that-became-a-better-question* paper. The original
hypothesis was clean and testable; the mesoscale test refuted it and, in doing
so, revealed a more general and more useful set of rules for when wind farms can
shelter communities. That arc is the paper's strength — lead with it.
