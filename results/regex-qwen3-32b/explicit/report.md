# Explicit-rule check (base model, rule stated in the prompt)

Model: `Qwen/Qwen3-32B`, no adapter. Decision accuracy on each run's held-out eval strings (half accepted, half rejected; both framings, debiased). Means over characters with 95% bootstrap CIs. `none` = no rule given (expect about 0.5).

| level | variant | decision acc | accept rate | n |
|---|---|---|---|---|
| 1 | description | 0.985 [0.978, 0.992] | 0.51 | 100 |
| 1 | none | 0.502 [0.488, 0.517] | 0.84 | 100 |
| 1 | regex | 0.572 [0.561, 0.585] | 0.07 | 100 |
| 2 | description | 0.799 [0.765, 0.833] | 0.31 | 100 |
| 2 | none | 0.495 [0.480, 0.508] | 0.83 | 100 |
| 2 | regex | 0.554 [0.541, 0.568] | 0.06 | 100 |
| 3 | description | 0.767 [0.749, 0.786] | 0.53 | 100 |
| 3 | none | 0.489 [0.474, 0.504] | 0.82 | 100 |
| 3 | regex | 0.531 [0.522, 0.542] | 0.05 | 100 |
| 4 | description | 0.629 [0.615, 0.642] | 0.58 | 100 |
| 4 | none | 0.505 [0.497, 0.513] | 0.83 | 100 |
| 4 | regex | 0.581 [0.570, 0.591] | 0.22 | 100 |

## By template (variant = regex)

| level | template | decision acc | accept rate | n |
|---|---|---|---|---|
| 1 | `.*X` | 0.580 | 0.09 | 19 |
| 1 | `.*X.*` | 0.626 | 0.13 | 26 |
| 1 | `X.*` | 0.520 | 0.02 | 31 |
| 1 | `[YZ]+` | 0.576 | 0.08 | 24 |
| 2 | `.*X.*Y.*` | 0.538 | 0.08 | 26 |
| 2 | `.*XY.*` | 0.516 | 0.02 | 26 |
| 2 | `X.*Y` | 0.504 | 0.00 | 17 |
| 2 | `X[XZ]*` | 0.626 | 0.13 | 31 |
| 3 | `(.X|Y)+` | 0.507 | 0.02 | 25 |
| 3 | `(X.|Y)+` | 0.530 | 0.07 | 26 |
| 3 | `X+[YZ]+` | 0.514 | 0.01 | 26 |
| 3 | `[YZ]+X+` | 0.579 | 0.09 | 23 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*)*` | 0.602 | 0.18 | 18 |
| 4 | `[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 0.609 | 0.23 | 29 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*)*` | 0.547 | 0.26 | 22 |
| 4 | `[YZ]*X[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*` | 0.567 | 0.19 | 31 |
