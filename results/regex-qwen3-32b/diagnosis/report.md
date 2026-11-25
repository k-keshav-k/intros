# Regex report diagnosis

Shares are over parsed reports, then averaged over characters. Reference step: the final step of each run.

## What the reports are

| run | level | step | parse | literal | accept all | accept none | in family | right level | same template | exact | unparsed: anchors / decision / other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b_L1 | 1 | 4000 | 0.94 | 0.01 | 0.12 | 0.00 | 0.38 | 0.36 | 0.23 | 0.21 | 0.01 / 0.00 / 0.05 |
| regex-qwen3-32b_L2 | 2 | 4000 | 0.41 | 0.04 | 0.02 | 0.00 | 0.23 | 0.08 | 0.04 | 0.04 | 0.00 / 0.20 / 0.39 |
| regex-qwen3-32b_L3 | 3 | 4000 | 0.26 | 0.01 | 0.01 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 / 0.72 / 0.02 |
| regex-qwen3-32b_L4 | 4 | 4000 | 0.03 | 0.32 | 0.06 | 0.00 | 0.18 | 0.00 | 0.00 | 0.00 | 0.00 / 0.86 / 0.10 |

## Error-string faithfulness against its shuffled baseline

Paired over characters that make at least one error and have parsed reports. `error - shuffled` > 0 means the reports track this character's own errors beyond what any same-level report would.

| run | level | step | accept rate | error faith | shuffled error faith | error − 0.5 | error − shuffled |
|---|---|---|---|---|---|---|---|
| regex-qwen3-32b_L1 | 1 | 4000 | 0.50 | nan | nan | n/a | n/a |
| regex-qwen3-32b_L2 | 2 | 4000 | 0.49 | 0.148 | 0.747 | -0.352 [-0.415, -0.275] (n=28) | -0.599 [-0.651, -0.536] (n=28) |
| regex-qwen3-32b_L3 | 3 | 4000 | 0.50 | 0.816 | 0.797 | +0.316 [+0.188, +0.458] (n=8) | +0.019 [-0.013, +0.045] (n=8) |
| regex-qwen3-32b_L4 | 4 | 4000 | 0.50 | 0.722 | 0.812 | +0.222 [-0.056, +0.500] (n=2) | +0.027 [-0.061, +0.115] (n=2) |

## Per-letter view (counterpart of Keshav's attribute diagnosis)

Valence of a letter = P(accept | letter present) − P(accept | absent) over all strings. `sign ok`: same direction as the rule on letters that push either way (Keshav: sign_ok). `rel split` / `irr split`: share of the letter pairs the rule tells apart / treats as interchangeable that the report tells apart (Keshav: rel_mentioned / irr_mentioned); high `irr split` = the report brings in letters that do not matter. `val corr`: correlation of the rule's letter valences with the average report's, against the same for other characters' reports (Keshav: val_corr_avg). `assoc ceiling`: best agreement with the model's decisions of a one-letter rule. `presence R²`: share of the rule explained by which letters are present (Keshav: degree1_mass).

| run | level | step | faith | sign ok | rel split | irr split | n split | val corr | shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b_L1 | 1 | 4000 | 0.62 | 0.71 | 0.78 | 0.45 | 2.00 | 0.55 | -0.03 | +0.586 [+0.444, +0.721] (n=100) | 0.83 | 0.52 |
| regex-qwen3-32b_L2 | 2 | 4000 | 0.60 | 0.90 | 0.86 | n/a | 2.57 | 0.91 | -0.03 | +0.939 [+0.894, +0.977] (n=100) | 0.75 | 0.30 |
| regex-qwen3-32b_L3 | 3 | 4000 | 0.59 | 0.82 | 0.91 | 0.87 | 2.68 | 0.75 | -0.04 | +0.785 [+0.693, +0.867] (n=95) | 0.68 | 0.11 |
| regex-qwen3-32b_L4 | 4 | 4000 | 0.55 | 0.78 | 0.87 | 0.27 | 2.02 | 0.73 | -0.07 | +0.765 [+0.592, +0.918] (n=47) | 0.69 | 0.06 |

### Untrained model (step 0)

| run | level | step | faith | sign ok | rel split | irr split | n split | val corr | shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b_L1 | 1 | 0 | 0.26 | 0.51 | 0.81 | 0.80 | 2.42 | 0.00 | 0.00 | -0.001 [-0.153, +0.151] (n=100) | 0.75 | 0.52 |
| regex-qwen3-32b_L2 | 2 | 0 | 0.26 | 0.58 | 0.81 | n/a | 2.42 | 0.24 | 0.02 | +0.221 [+0.080, +0.361] (n=100) | 0.76 | 0.30 |
| regex-qwen3-32b_L3 | 3 | 0 | 0.28 | 0.56 | 0.80 | 0.78 | 2.39 | 0.04 | 0.01 | +0.027 [-0.105, +0.160] (n=100) | 0.74 | 0.11 |
| regex-qwen3-32b_L4 | 4 | 0 | 0.27 | 0.50 | 0.79 | 0.79 | 2.37 | -0.07 | 0.02 | -0.090 [-0.226, +0.049] (n=100) | 0.73 | 0.06 |

## Observer reports (same shares)

| run | level | step | parse | literal | in family | same template | exact | sign ok | irr split | val corr |
|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b_L1 | 1 | 4000 | 0.96 | 0.00 | 0.19 | 0.11 | 0.11 | 0.77 | 0.73 | 0.72 |
| regex-qwen3-32b_L2 | 2 | 4000 | 0.96 | 0.00 | 0.11 | 0.05 | 0.05 | 0.82 | n/a | 0.71 |
| regex-qwen3-32b_L3 | 3 | 4000 | 0.96 | 0.00 | 0.08 | 0.01 | 0.01 | 0.79 | 0.82 | 0.60 |
| regex-qwen3-32b_L4 | 4 | 4000 | 0.95 | 0.00 | 0.05 | 0.00 | 0.00 | 0.68 | 0.88 | 0.24 |

## By template (reference step)

| level | template | decision acc | faith | same template | exact | sign ok | irr split | val corr | presence R² | n |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `.*X` | 1.000 | 0.596 | 0.11 | 0.10 | 0.80 | 0.61 | 0.84 | 0.04 | 19 |
| 1 | `.*X.*` | 1.000 | 0.569 | 0.12 | 0.12 | 0.67 | 0.43 | 0.59 | 1.00 | 26 |
| 1 | `X.*` | 1.000 | 0.623 | 0.30 | 0.25 | 0.68 | 0.38 | 0.17 | 0.04 | 31 |
| 1 | `[YZ]+` | 1.000 | 0.673 | 0.34 | 0.33 | 0.74 | 0.42 | 0.79 | 1.00 | 24 |
| 2 | `.*X.*Y.*` | 0.999 | 0.558 | 0.01 | 0.01 | 0.80 | n/a | 0.84 | 0.37 | 26 |
| 2 | `.*XY.*` | 1.000 | 0.537 | 0.00 | 0.00 | 0.89 | n/a | 0.92 | 0.17 | 26 |
| 2 | `X.*Y` | 0.996 | 0.593 | 0.06 | 0.06 | 0.93 | n/a | 0.92 | 0.02 | 17 |
| 2 | `X[XZ]*` | 0.986 | 0.686 | 0.09 | 0.09 | 0.96 | n/a | 0.96 | 0.49 | 31 |
| 3 | `(.X|Y)+` | 0.990 | 0.570 | 0.00 | 0.00 | 0.99 | n/a | 0.90 | 0.22 | 25 |
| 3 | `(X.|Y)+` | 0.993 | 0.563 | 0.00 | 0.00 | 0.90 | n/a | 0.83 | 0.22 | 26 |
| 3 | `X+[YZ]+` | 0.999 | 0.573 | 0.01 | 0.01 | 0.31 | 0.97 | 0.37 | 0.00 | 26 |
| 3 | `[YZ]+X+` | 1.000 | 0.641 | 0.00 | 0.00 | 0.96 | 0.76 | 0.91 | 0.00 | 23 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*)*` | 0.999 | 0.562 | 0.00 | 0.00 | 1.00 | 0.62 | 0.71 | 0.05 | 18 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 1.000 | 0.598 | 0.00 | 0.00 | 0.71 | 0.39 | 0.45 | 0.11 | 29 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*)*` | 0.990 | 0.514 | 0.00 | 0.00 | n/a | 0.19 | 0.89 | 0.05 | 22 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 1.000 | 0.537 | 0.00 | 0.00 | 0.71 | 0.11 | 0.80 | 0.03 | 31 |
