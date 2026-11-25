# Regex report diagnosis

Shares are over parsed reports, then averaged over characters. Reference step: the final step of each run.

## What the reports are

| run | level | step | parse | literal | accept all | accept none | in family | right level | same template | exact | unparsed: anchors / decision / other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-prefill_L1 | 1 | 4000 | 0.98 | 0.01 | 0.16 | 0.00 | 0.46 | 0.44 | 0.30 | 0.28 | 0.01 / 0.00 / 0.02 |
| regex-qwen3-32b-prefill_L2 | 2 | 4000 | 0.97 | 0.01 | 0.03 | 0.00 | 0.26 | 0.09 | 0.04 | 0.04 | 0.00 / 0.00 / 0.03 |
| regex-qwen3-32b-prefill_L3 | 3 | 4000 | 0.96 | 0.01 | 0.02 | 0.00 | 0.17 | 0.01 | 0.01 | 0.01 | 0.01 / 0.00 / 0.03 |
| regex-qwen3-32b-prefill_L4 | 4 | 4000 | 0.92 | 0.26 | 0.05 | 0.00 | 0.36 | 0.00 | 0.00 | 0.00 | 0.01 / 0.01 / 0.06 |

## Error-string faithfulness against its shuffled baseline

Paired over characters that make at least one error and have parsed reports. `error - shuffled` > 0 means the reports track this character's own errors beyond what any same-level report would.

| run | level | step | accept rate | error faith | shuffled error faith | error − 0.5 | error − shuffled |
|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-prefill_L1 | 1 | 4000 | 0.50 | nan | nan | n/a | n/a |
| regex-qwen3-32b-prefill_L2 | 2 | 4000 | 0.49 | 0.140 | 0.736 | -0.360 [-0.439, -0.257] (n=28) | -0.596 [-0.664, -0.509] (n=28) |
| regex-qwen3-32b-prefill_L3 | 3 | 4000 | 0.50 | 0.715 | 0.769 | +0.215 [+0.088, +0.348] (n=8) | -0.053 [-0.186, +0.047] (n=8) |
| regex-qwen3-32b-prefill_L4 | 4 | 4000 | 0.50 | 0.730 | 0.772 | +0.230 [+0.080, +0.351] (n=6) | -0.043 [-0.109, +0.024] (n=6) |

## Per-letter view (counterpart of Keshav's attribute diagnosis)

Valence of a letter = P(accept | letter present) − P(accept | absent) over all strings. `sign ok`: same direction as the rule on letters that push either way (Keshav: sign_ok). `rel split` / `irr split`: share of the letter pairs the rule tells apart / treats as interchangeable that the report tells apart (Keshav: rel_mentioned / irr_mentioned); high `irr split` = the report brings in letters that do not matter. `val corr`: correlation of the rule's letter valences with the average report's, against the same for other characters' reports (Keshav: val_corr_avg). `assoc ceiling`: best agreement with the model's decisions of a one-letter rule. `presence R²`: share of the rule explained by which letters are present (Keshav: degree1_mass).

| run | level | step | faith | sign ok | rel split | irr split | n split | val corr | shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-prefill_L1 | 1 | 4000 | 0.68 | 0.81 | 0.78 | 0.31 | 1.86 | 0.72 | -0.05 | +0.776 [+0.661, +0.880] (n=100) | 0.83 | 0.52 |
| regex-qwen3-32b-prefill_L2 | 2 | 4000 | 0.61 | 0.89 | 0.82 | n/a | 2.45 | 0.92 | -0.04 | +0.957 [+0.921, +0.989] (n=100) | 0.75 | 0.30 |
| regex-qwen3-32b-prefill_L3 | 3 | 4000 | 0.60 | 0.76 | 0.90 | 0.80 | 2.59 | 0.71 | -0.04 | +0.759 [+0.640, +0.862] (n=100) | 0.68 | 0.11 |
| regex-qwen3-32b-prefill_L4 | 4 | 4000 | 0.57 | 0.81 | 0.91 | 0.19 | 2.01 | 0.65 | -0.06 | +0.710 [+0.556, +0.850] (n=100) | 0.69 | 0.06 |

### Untrained model (step 0)

| run | level | step | faith | sign ok | rel split | irr split | n split | val corr | shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-prefill_L1 | 1 | 0 | 0.22 | 0.54 | 0.85 | 0.84 | 2.53 | 0.05 | 0.01 | +0.038 [-0.124, +0.199] (n=100) | 0.75 | 0.52 |
| regex-qwen3-32b-prefill_L2 | 2 | 0 | 0.23 | 0.60 | 0.85 | n/a | 2.54 | 0.42 | 0.03 | +0.388 [+0.268, +0.508] (n=100) | 0.76 | 0.30 |
| regex-qwen3-32b-prefill_L3 | 3 | 0 | 0.25 | 0.55 | 0.85 | 0.83 | 2.52 | 0.27 | -0.01 | +0.288 [+0.147, +0.424] (n=100) | 0.74 | 0.11 |
| regex-qwen3-32b-prefill_L4 | 4 | 0 | 0.23 | 0.52 | 0.83 | 0.86 | 2.53 | 0.03 | 0.01 | +0.027 [-0.100, +0.154] (n=100) | 0.73 | 0.06 |

## Observer reports (same shares)

| run | level | step | parse | literal | in family | same template | exact | sign ok | irr split | val corr |
|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-prefill_L1 | 1 | 4000 | 0.95 | 0.00 | 0.19 | 0.09 | 0.09 | 0.80 | 0.75 | 0.74 |
| regex-qwen3-32b-prefill_L2 | 2 | 4000 | 0.97 | 0.00 | 0.13 | 0.07 | 0.06 | 0.82 | n/a | 0.69 |
| regex-qwen3-32b-prefill_L3 | 3 | 4000 | 0.96 | 0.00 | 0.11 | 0.03 | 0.03 | 0.84 | 0.81 | 0.60 |
| regex-qwen3-32b-prefill_L4 | 4 | 4000 | 0.97 | 0.00 | 0.07 | 0.00 | 0.00 | 0.72 | 0.86 | 0.35 |

## By template (reference step)

| level | template | decision acc | faith | same template | exact | sign ok | irr split | val corr | presence R² | n |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `.*X` | 1.000 | 0.584 | 0.06 | 0.06 | 0.84 | 0.60 | 0.91 | 0.04 | 19 |
| 1 | `.*X.*` | 1.000 | 0.688 | 0.22 | 0.22 | 0.83 | 0.24 | 0.76 | 1.00 | 26 |
| 1 | `X.*` | 1.000 | 0.661 | 0.32 | 0.30 | 0.74 | 0.25 | 0.55 | 0.04 | 31 |
| 1 | `[YZ]+` | 1.000 | 0.757 | 0.53 | 0.50 | 0.85 | 0.21 | 0.76 | 1.00 | 24 |
| 2 | `.*X.*Y.*` | 0.999 | 0.572 | 0.01 | 0.01 | 0.82 | n/a | 0.87 | 0.37 | 26 |
| 2 | `.*XY.*` | 1.000 | 0.534 | 0.00 | 0.00 | 0.87 | n/a | 0.92 | 0.17 | 26 |
| 2 | `X.*Y` | 0.996 | 0.617 | 0.08 | 0.08 | 0.92 | n/a | 0.89 | 0.02 | 17 |
| 2 | `X[XZ]*` | 0.986 | 0.712 | 0.07 | 0.07 | 0.96 | n/a | 0.97 | 0.49 | 31 |
| 3 | `(.X|Y)+` | 0.990 | 0.586 | 0.00 | 0.00 | 0.92 | n/a | 0.98 | 0.22 | 25 |
| 3 | `(X.|Y)+` | 0.993 | 0.597 | 0.00 | 0.00 | 0.86 | n/a | 0.99 | 0.22 | 26 |
| 3 | `X+[YZ]+` | 0.999 | 0.575 | 0.03 | 0.03 | 0.29 | 0.94 | 0.02 | 0.00 | 26 |
| 3 | `[YZ]+X+` | 1.000 | 0.662 | 0.00 | 0.00 | 0.97 | 0.64 | 0.90 | 0.00 | 23 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*)*` | 0.999 | 0.605 | 0.00 | 0.00 | 0.90 | 0.25 | 1.00 | 0.05 | 18 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 1.000 | 0.620 | 0.00 | 0.00 | 0.93 | 0.29 | 1.00 | 0.11 | 29 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*)*` | 0.990 | 0.520 | 0.00 | 0.00 | 0.67 | 0.12 | 0.09 | 0.05 | 22 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 1.000 | 0.525 | 0.00 | 0.00 | 0.76 | 0.11 | 0.51 | 0.03 | 31 |
