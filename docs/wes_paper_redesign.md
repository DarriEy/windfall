# WES paper — rethought from scratch (2026-06)

The central idea is unchanged and is the heart of the paper: **wind energy as
multipurpose infrastructure — the turbine wake, normally treated as a pure loss,
reframed as a community wind-reduction PUBLIC GOOD delivered alongside
electricity.** The original Eyjafjörður draft (make_wes_paper.py, kept as a prior
draft — NOT a separate paper) located that public good in storm-shielding the
fjord head ("the Fjord Effect"). The WRF work does not retire the public-good
thesis; it *refines where and how the public good is realizable*. This document
is the from-scratch design of the single paper we are converging on.

## Working title
**"The wake as a public good: wind farms as dual-purpose generation-and-shelter
infrastructure, tested across fjord and open-coast terrain."**
(Alt: "Multipurpose wind energy: when a turbine wake is a community
wind-reduction public good — a mesoscale study in Iceland.")

## One-sentence thesis
A wind farm's wake can be a public good — measurable community wind reduction
delivered jointly with power — but the *form* that public good takes is set by
terrain and turbine cut-out: confined fjords leak the shelter over their ridges
(so there the public value is generation), open coasts hold a coherent wake (so
there the wake delivers real, uniform, city-wide wind reduction), and the most
broadly realizable public good is a persistent cut to the *everyday* wind of
exposed communities, not protection from the rare catastrophic storm.

## Why this is still a multipurpose / public-good paper (not a refutation)
The dual-purpose framing is the contribution. The wake is reframed from a
generation loss (array-efficiency penalty) into a sited, deliverable amenity.
The mesoscale study is what turns that from an assertion into an engineering
result: it says *which terrains and which wind regimes* convert the wake into a
public good, and at what scale. The Eyjafjörður storm-shield was one candidate
form of the public good; the analysis replaces it with better-supported forms
(open-coast everyday-wind reduction; terrain-selective generation+shelter), so
the public-good thesis comes out sharper, quantified, and generalizable.

## Abstract (draft)
A wind farm's wake lowers the wind downstream — usually counted only as a loss
of array efficiency. We ask whether that wake can instead be treated as a
deliberate public good: community wind reduction delivered jointly with
electricity, making the farm dual-purpose infrastructure. We test where and how
this is realizable with WRF-ARW simulations (nested to 1 km, Fitch
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
Lead with the multipurpose / public-good thesis, not with a refutation. The
narrative: wind farms can be dual-purpose (power + community wind reduction); the
mesoscale study establishes the *engineering rules* for when the wake becomes a
public good (terrain that holds the wake, winds below cut-out, the high-Ct
everyday regime) and quantifies the deliverable (uniform city-wide everyday-wind
reduction, ~6 %/GW). The fjord storm-shield is presented as the initial,
intuitive candidate form that the physics replaces with better ones — a
refinement of the thesis, not its collapse. The original draft is cited as
prior work-in-progress, superseded.

## Process note
make_wes_paper.py is gitignored (generator), so preserve the prior draft as a
local copy (e.g. make_wes_paper_v1_fjordeffect.py) before rewriting toward this
target; git history of the tracked code already records the evolution.
