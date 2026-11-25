# Regex report diagnosis

Shares are over parsed reports, then averaged over characters. Reference step: the final step of each run.

## What the reports are

| run | level | step | parse | literal | accept all | accept none | in family | right level | same template | exact | unparsed: anchors / decision / other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-s2_L4 | 4 | 4000 | 0.97 | 0.23 | 0.05 | 0.00 | 0.46 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 / 0.02 |

## Error-string faithfulness against its shuffled baseline

Paired over characters that make at least one error and have parsed reports. `error - shuffled` > 0 means the reports track this character's own errors beyond what any same-level report would.

| run | level | step | accept rate | error faith | shuffled error faith | error − 0.5 | error − shuffled |
|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-s2_L4 | 4 | 4000 | 0.50 | nan | nan | n/a | n/a |

## Per-letter view (counterpart of Keshav's attribute diagnosis)

Valence of a letter = P(accept | letter present) − P(accept | absent) over all strings. `sign ok`: same direction as the rule on letters that push either way (Keshav: sign_ok). `rel split` / `irr split`: share of the letter pairs the rule tells apart / treats as interchangeable that the report tells apart (Keshav: rel_mentioned / irr_mentioned); high `irr split` = the report brings in letters that do not matter. `val corr`: correlation of the rule's letter valences with the average report's, against the same for other characters' reports (Keshav: val_corr_avg). `assoc ceiling`: best agreement with the model's decisions of a one-letter rule. `presence R²`: share of the rule explained by which letters are present (Keshav: degree1_mass).

| run | level | step | faith | sign ok | rel split | irr split | n split | val corr | shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-s2_L4 | 4 | 4000 | 0.62 | 0.96 | 0.90 | 0.18 | 1.98 | 0.98 | -0.08 | +1.059 [+1.004, +1.095] (n=100) | 0.69 | 0.06 |

### Untrained model (step 0)

| run | level | step | faith | sign ok | rel split | irr split | n split | val corr | shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-s2_L4 | 4 | 0 | 0.23 | 0.54 | 0.83 | 0.82 | 2.48 | 0.09 | 0.04 | +0.057 [-0.068, +0.183] (n=100) | 0.73 | 0.06 |

## Observer reports (same shares)

| run | level | step | parse | literal | in family | same template | exact | sign ok | irr split | val corr |
|---|---|---|---|---|---|---|---|---|---|---|
| regex-qwen3-32b-s2_L4 | 4 | 4000 | 0.95 | 0.00 | 0.05 | 0.00 | 0.00 | 0.67 | 0.84 | 0.16 |

## By template (reference step)

| level | template | decision acc | faith | same template | exact | sign ok | irr split | val corr | presence R² | n |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | `[YZ]*(X[YZ]*X[YZ]*)*` | 1.000 | 0.636 | 0.00 | 0.00 | 0.95 | 0.21 | 0.99 | 0.05 | 33 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 1.000 | 0.643 | 0.00 | 0.00 | 0.99 | 0.41 | 1.00 | 0.11 | 26 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*)*` | 1.000 | 0.567 | 0.00 | 0.00 | 0.91 | 0.02 | 0.89 | 0.05 | 18 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 1.000 | 0.592 | 0.00 | 0.00 | 0.98 | 0.02 | 1.00 | 0.03 | 23 |
