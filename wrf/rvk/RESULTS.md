# Reykjavík capital wind-shield / wind-reduction — WRF 1 km results

Domain: 9/3/1 km nest on the capital (ref 64.10°N, 21.90°W), d03 72×72 km.
Turbine: V236 15 MW (Fitch table extended to 50 m/s with feathering >25 m/s).
Met forcing: ERA5. Climatology: CARRA 2022 at the city (reykjavik.py).

## Climatology (CARRA 2022, rvk_city)
- Mean 10 m wind 6.0 m/s (≈8.7 m/s at 150 m hub) — a genuinely windy capital.
- Operating regime (hub 3–25 m/s): 84% of hours; feathering (>25): only 2%.
- Storm rose (>15 m/s): SW 31%, E 29%, SE 14%, S 12% → ~55% land-sector,
  ~14% clean sea. Two dominant gale sectors = SW and E (the events run).
- Lower-ABL stability in gales: 84% stable (long-wake regime). Stability is
  NOT the limiter — geometry and turbine cut-out are.

## Storm shielding (ΔP at the city)
| event | array | wind at city | turbines | ΔP |
|-------|-------|--------------|----------|----|
| SW-gale (Mar 18) | 1.35 GW offshore, 7–14 km SW | 18 m/s | operating | **41%** (downtown 60%) |
| E-gale (Feb 25)  | 1.08 GW onshore, 4–10 km E  | 32 m/s | FEATHERING | 2–5% |

Decider = does the gale exceed cut-out (25 m/s) at the city. SW gales are
pre-sheltered by the Reykjanes peninsula (18 m/s → operate → big shield);
E gales are downslope-accelerated over Hellisheiði (32 m/s → feather → none).

## Everyday wind reduction (the strongest result)
Moderate-E day (~11 m/s hub, high Ct), 1.08 GW onshore E wall:
- Reykjavík city −29% speed (ΔP 49%), downtown −40%, E suburb −35%.

Annual-mean wind reduction (WRF curve applied to full-year CARRA):
- single E-wall (1.08 GW): **−6.4%**
- city ringed by arrays (~4–6 GW, idealized): **−23%**

## Takeaway
Turbines as wind-shields work in the OPERATING regime (everyday winds +
moderate storms) and fail in the EXTREME regime (feathering). The
achievable prize is a persistent ~6–20% cut to a windy city's everyday
wind, plus real protection from storm directions that arrive below cut-out
(SW for Reykjavík) — NOT stopping the rare catastrophic downslope gale.
