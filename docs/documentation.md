# Method notes

## Class legend

Defined in `config.yaml` (`classes`). MapBiomas Collection 10 codes are mapped to 5 model classes; unmapped codes (including 0 and 27, not observed) become 0 = no data and are ignored everywhere.

## Demand (Markov)

`P` is estimated on 1985 -> 2015 (30 years). For a period of `t` years the matrix used is `P^(t/30)` (fractional matrix power; negative entries from numerical error are clipped and rows renormalized). Expected cells per class = `counts_t0 @ P_t`. Only the urban gain is used, floored at 0.

## Suitability

For factor `f` (distance to urban at t0, distance to main roads), among candidate cells (not urban at t0, not water, valid at both dates):

    lift_f(bin) = P(became urban | f in bin) / P(became urban)
    score = P(became urban) * prod_f lift_f(bin_f(cell))

Bins are quantiles of the factor over candidate cells. The distance to urban area is recomputed every simulated year.

## Allocation (CA)

Each year `k = demand / years` cells are converted, choosing the cells with highest `score * neighborhood * noise`, restricted to cells with at least one urban cell in the 5x5 window. Unallocated demand is carried to the next year and reported as a warning.

## Validation

Simulate from the end of the training period to the last observed year. Reported in `outputs/metrics.json`:
hits, misses, false alarms, figure of merit, producer/user accuracy of the new urban cells, kappa (urban vs. non-urban). A figure of merit of 0 is what a "no change" model scores, so any positive value is skill beyond persistence.
