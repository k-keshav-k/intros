# Conflict test: weights vs. forged self-history

Shift: +1 = follows the record's grammar G' (R'), -1 = follows the trained grammar G (R). Scored on held-out strings where G (= R) and G' (= R') disagree. Means over characters; 95% paired bootstrap CIs. ft = fine-tuned, base = adapter off (outsider).

## Level 1 (21 characters)

### Shift in each context

| condition | k | ft report | base report | ft behaviour | base behaviour | ft report - behaviour (decisive) | ft faith in context | ft parse rate |
|---|---|---|---|---|---|---|---|---|
| none | 0 | -0.574 | -0.017 | -1.000 | -0.009 | – | 0.787 | 0.93 |
| genuine_self | 4 | -0.639 | -0.205 | -0.939 | -0.162 | +0.347 [0.056, 0.694] | 0.812 | 0.04 |
| genuine_self | 8 | +0.000 | -0.213 | -0.970 | -0.183 | +1.000 [nan, nan] | 0.500 | 0.01 |
| genuine_self | 16 | -0.583 | -0.237 | -0.982 | -0.209 | +0.417 [0.000, 0.833] | 0.792 | 0.01 |
| forged_self | 4 | -0.104 | +0.075 | -0.439 | +0.146 | +0.896 [0.792, 1.000] | 0.552 | 0.02 |
| forged_self | 8 | +0.458 | +0.225 | +0.085 | +0.193 | -0.542 [nan, nan] | 0.729 | 0.01 |
| forged_self | 16 | -0.292 | +0.206 | +0.507 | +0.216 | -0.208 [-0.917, 0.500] | 0.604 | 0.01 |
| forged_other | 4 | +0.071 | +0.099 | -0.371 | +0.118 | +0.539 [0.200, 0.824] | 0.490 | 0.24 |
| forged_other | 8 | +0.176 | +0.182 | +0.152 | +0.153 | +0.106 [-0.236, 0.470] | 0.501 | 0.13 |
| forged_other | 16 | +0.217 | +0.252 | +0.568 | +0.161 | -0.250 [-0.629, 0.158] | 0.633 | 0.10 |
| forged_unlabeled | 4 | +0.139 | +0.035 | -0.433 | +0.167 | +0.610 [0.369, 0.860] | 0.441 | 0.29 |
| forged_unlabeled | 8 | +0.032 | +0.140 | -0.018 | +0.227 | +0.053 [-0.401, 0.474] | 0.557 | 0.14 |
| forged_unlabeled | 16 | +0.078 | +0.201 | +0.313 | +0.250 | -0.519 [-0.825, -0.200] | 0.564 | 0.07 |

### Record effect (forged minus genuine record, same strings) and resistance

| condition | k | ft report effect | base report effect | resistance (report) | ft behaviour effect | base behaviour effect | resistance (behaviour) |
|---|---|---|---|---|---|---|---|
| forged_self | 4 | +nan [nan, nan] | +0.281 [0.190, 0.384] | +nan [nan, nan] | +0.499 [0.221, 0.809] | +0.308 [0.259, 0.356] | -0.191 [-0.476, 0.068] |
| forged_self | 8 | +nan [nan, nan] | +0.438 [0.309, 0.575] | +nan [nan, nan] | +1.055 [0.760, 1.340] | +0.375 [0.324, 0.427] | -0.679 [-0.965, -0.382] |
| forged_self | 16 | +nan [nan, nan] | +0.443 [0.338, 0.554] | +nan [nan, nan] | +1.488 [1.227, 1.723] | +0.425 [0.372, 0.481] | -1.063 [-1.302, -0.802] |
| forged_other | 4 | +0.590 [0.090, 1.069] | +0.304 [0.208, 0.408] | -0.209 [-0.686, 0.260] | +0.568 [0.326, 0.838] | +0.280 [0.232, 0.326] | -0.288 [-0.535, -0.063] |
| forged_other | 8 | +0.083 [nan, nan] | +0.395 [0.279, 0.518] | +0.319 [nan, nan] | +1.122 [0.871, 1.365] | +0.336 [0.289, 0.382] | -0.786 [-1.027, -0.538] |
| forged_other | 16 | +nan [nan, nan] | +0.490 [0.389, 0.609] | +nan [nan, nan] | +1.550 [1.296, 1.767] | +0.370 [0.325, 0.418] | -1.180 [-1.400, -0.929] |
| forged_unlabeled | 4 | +1.013 [0.646, 1.283] | +0.240 [0.143, 0.345] | -0.629 [-1.333, -0.126] | +0.505 [0.294, 0.725] | +0.329 [0.272, 0.386] | -0.176 [-0.394, 0.036] |
| forged_unlabeled | 8 | +0.167 [nan, nan] | +0.353 [0.257, 0.446] | +0.167 [nan, nan] | +0.952 [0.687, 1.219] | +0.410 [0.356, 0.460] | -0.542 [-0.812, -0.276] |
| forged_unlabeled | 16 | +nan [nan, nan] | +0.439 [0.329, 0.564] | +nan [nan, nan] | +1.294 [1.016, 1.555] | +0.459 [0.413, 0.506] | -0.836 [-1.098, -0.561] |

### Self-attribution and detection

| contrast | k | difference [95% CI] | n |
|---|---|---|---|
| detection self (base) | 4 | -0.001 [-0.034, 0.033] | 21 |
| detection self (base) | 8 | +0.013 [-0.013, 0.040] | 21 |
| detection self (base) | 16 | -0.012 [-0.037, 0.014] | 21 |
| detection self (ft) | 4 | +0.062 [-0.048, 0.177] | 21 |
| detection self (ft) | 8 | -0.022 [-0.083, 0.038] | 21 |
| detection self (ft) | 16 | +0.039 [-0.033, 0.115] | 21 |
| detection third (base) | 4 | -0.010 [-0.043, 0.021] | 21 |
| detection third (base) | 8 | +0.009 [-0.027, 0.043] | 21 |
| detection third (base) | 16 | -0.014 [-0.061, 0.031] | 21 |
| detection third (ft) | 4 | -0.039 [-0.123, 0.022] | 21 |
| detection third (ft) | 8 | -0.000 [-0.001, 0.000] | 21 |
| detection third (ft) | 16 | +0.001 [-0.000, 0.003] | 21 |
| self-attribution (base) | 4 | -0.024 [-0.085, 0.034] | 21 |
| self-attribution (base) | 8 | +0.043 [-0.021, 0.109] | 21 |
| self-attribution (base) | 16 | -0.047 [-0.113, 0.023] | 21 |
| self-attribution (ft) | 4 | -0.257 [-0.389, -0.125] | 2 |
| self-attribution (ft) | 8 | +0.208 [nan, nan] | 1 |
| self-attribution (ft) | 16 | +0.000 [nan, nan] | 1 |

Mean P(yes) by record type:

| model | framing | k | genuine | forged |
|---|---|---|---|---|
| base | self | 4 | 0.369 | 0.370 |
| base | self | 8 | 0.337 | 0.324 |
| base | self | 16 | 0.313 | 0.325 |
| base | third | 4 | 0.285 | 0.295 |
| base | third | 8 | 0.413 | 0.404 |
| base | third | 16 | 0.461 | 0.475 |
| ft | self | 4 | 0.555 | 0.494 |
| ft | self | 8 | 0.184 | 0.206 |
| ft | self | 16 | 0.171 | 0.133 |
| ft | third | 4 | 0.948 | 0.988 |
| ft | third | 8 | 1.000 | 1.000 |
| ft | third | 16 | 1.000 | 0.999 |

## Level 2 (16 characters)

### Shift in each context

| condition | k | ft report | base report | ft behaviour | base behaviour | ft report - behaviour (decisive) | ft faith in context | ft parse rate |
|---|---|---|---|---|---|---|---|---|
| none | 0 | -0.425 | -0.080 | -0.998 | -0.019 | – | 0.713 | 0.52 |
| genuine_self | 4 | -0.250 | -0.236 | -0.962 | -0.184 | +0.750 [0.367, 0.983] | 0.625 | 0.04 |
| genuine_self | 8 | -0.083 | -0.308 | -0.955 | -0.241 | +0.458 [0.083, 0.833] | 0.771 | 0.02 |
| genuine_self | 16 | -0.500 | -0.361 | -0.972 | -0.248 | +0.500 [0.000, 1.000] | 0.750 | 0.02 |
| forged_self | 4 | +0.667 | +0.187 | -0.499 | +0.172 | +1.667 [1.333, 2.000] | 0.167 | 0.02 |
| forged_self | 8 | +nan | +0.322 | -0.167 | +0.232 | +nan [nan, nan] | nan | 0.00 |
| forged_self | 16 | +0.250 | +0.269 | +0.056 | +0.265 | +0.250 [nan, nan] | 0.875 | 0.01 |
| forged_other | 4 | +0.097 | +0.152 | -0.581 | +0.129 | +0.583 [0.250, 0.917] | 0.694 | 0.05 |
| forged_other | 8 | +0.208 | +0.245 | -0.261 | +0.176 | +0.917 [0.667, 1.167] | 0.458 | 0.02 |
| forged_other | 16 | +0.417 | +0.291 | -0.012 | +0.204 | +0.611 [-0.083, 1.917] | 0.583 | 0.02 |
| forged_unlabeled | 4 | -0.062 | +0.019 | -0.723 | +0.211 | +0.688 [0.229, 0.958] | 0.656 | 0.03 |
| forged_unlabeled | 8 | +0.000 | +0.136 | -0.597 | +0.271 | +1.000 [0.950, 1.050] | 0.500 | 0.05 |
| forged_unlabeled | 16 | +0.073 | +0.208 | -0.510 | +0.299 | +0.344 [0.010, 0.667] | 0.755 | 0.04 |

### Record effect (forged minus genuine record, same strings) and resistance

| condition | k | ft report effect | base report effect | resistance (report) | ft behaviour effect | base behaviour effect | resistance (behaviour) |
|---|---|---|---|---|---|---|---|
| forged_self | 4 | +nan [nan, nan] | +0.423 [0.272, 0.589] | +nan [nan, nan] | +0.463 [0.247, 0.703] | +0.356 [0.282, 0.426] | -0.106 [-0.321, 0.085] |
| forged_self | 8 | +nan [nan, nan] | +0.630 [0.478, 0.786] | +nan [nan, nan] | +0.788 [0.581, 1.009] | +0.473 [0.429, 0.514] | -0.316 [-0.547, -0.092] |
| forged_self | 16 | +0.250 [nan, nan] | +0.629 [0.466, 0.791] | +0.715 [nan, nan] | +1.028 [0.810, 1.247] | +0.512 [0.468, 0.555] | -0.516 [-0.748, -0.293] |
| forged_other | 4 | +0.028 [-0.250, 0.167] | +0.388 [0.271, 0.521] | +0.480 [0.173, 0.642] | +0.381 [0.203, 0.588] | +0.313 [0.238, 0.384] | -0.068 [-0.242, 0.080] |
| forged_other | 8 | +nan [nan, nan] | +0.553 [0.354, 0.756] | +nan [nan, nan] | +0.694 [0.484, 0.922] | +0.417 [0.382, 0.450] | -0.277 [-0.509, -0.057] |
| forged_other | 16 | +0.000 [nan, nan] | +0.652 [0.489, 0.828] | +0.361 [nan, nan] | +0.960 [0.736, 1.187] | +0.452 [0.415, 0.489] | -0.509 [-0.744, -0.281] |
| forged_unlabeled | 4 | +0.167 [nan, nan] | +0.255 [0.159, 0.353] | +0.146 [nan, nan] | +0.238 [0.073, 0.434] | +0.395 [0.312, 0.476] | +0.157 [-0.010, 0.309] |
| forged_unlabeled | 8 | +0.083 [0.000, 0.167] | +0.444 [0.310, 0.577] | +0.218 [0.048, 0.389] | +0.358 [0.124, 0.632] | +0.512 [0.472, 0.550] | +0.154 [-0.102, 0.376] |
| forged_unlabeled | 16 | -0.333 [nan, nan] | +0.569 [0.454, 0.688] | +0.972 [nan, nan] | +0.462 [0.243, 0.717] | +0.546 [0.506, 0.587] | +0.084 [-0.164, 0.306] |

### Self-attribution and detection

| contrast | k | difference [95% CI] | n |
|---|---|---|---|
| detection self (base) | 4 | +0.020 [-0.010, 0.052] | 16 |
| detection self (base) | 8 | +0.005 [-0.023, 0.033] | 16 |
| detection self (base) | 16 | +0.004 [-0.038, 0.044] | 16 |
| detection self (ft) | 4 | +0.004 [0.002, 0.007] | 16 |
| detection self (ft) | 8 | +0.002 [0.001, 0.003] | 16 |
| detection self (ft) | 16 | +0.003 [0.001, 0.005] | 16 |
| detection third (base) | 4 | +0.050 [0.013, 0.083] | 16 |
| detection third (base) | 8 | +0.055 [0.024, 0.087] | 16 |
| detection third (base) | 16 | +0.044 [-0.010, 0.103] | 16 |
| detection third (ft) | 4 | +0.170 [0.031, 0.314] | 16 |
| detection third (ft) | 8 | +0.338 [0.196, 0.478] | 16 |
| detection third (ft) | 16 | +0.269 [0.114, 0.422] | 16 |
| self-attribution (base) | 4 | +0.035 [-0.074, 0.149] | 16 |
| self-attribution (base) | 8 | +0.077 [-0.034, 0.188] | 16 |
| self-attribution (base) | 16 | -0.022 [-0.194, 0.143] | 16 |
| self-attribution (ft) | 4 | +0.583 [nan, nan] | 1 |
| self-attribution (ft) | 8 | +nan [nan, nan] | 0 |
| self-attribution (ft) | 16 | +0.250 [nan, nan] | 1 |

Mean P(yes) by record type:

| model | framing | k | genuine | forged |
|---|---|---|---|---|
| base | self | 4 | 0.400 | 0.380 |
| base | self | 8 | 0.313 | 0.308 |
| base | self | 16 | 0.311 | 0.307 |
| base | third | 4 | 0.317 | 0.267 |
| base | third | 8 | 0.452 | 0.397 |
| base | third | 16 | 0.569 | 0.524 |
| ft | self | 4 | 0.013 | 0.009 |
| ft | self | 8 | 0.009 | 0.007 |
| ft | self | 16 | 0.012 | 0.008 |
| ft | third | 4 | 0.294 | 0.124 |
| ft | third | 8 | 0.594 | 0.257 |
| ft | third | 16 | 0.733 | 0.464 |

## Level 3 (7 characters)

### Shift in each context

| condition | k | ft report | base report | ft behaviour | base behaviour | ft report - behaviour (decisive) | ft faith in context | ft parse rate |
|---|---|---|---|---|---|---|---|---|
| none | 0 | -0.234 | -0.025 | -0.997 | +0.016 | – | 0.617 | 0.36 |
| genuine_self | 4 | +nan | -0.113 | -0.991 | -0.116 | +nan [nan, nan] | nan | 0.00 |
| genuine_self | 8 | +nan | -0.235 | -0.942 | -0.144 | +nan [nan, nan] | nan | 0.00 |
| genuine_self | 16 | +nan | -0.146 | -0.947 | -0.184 | +nan [nan, nan] | nan | 0.00 |
| forged_self | 4 | +nan | +0.156 | -0.878 | +0.092 | +nan [nan, nan] | nan | 0.00 |
| forged_self | 8 | +nan | +0.181 | -0.858 | +0.125 | +nan [nan, nan] | nan | 0.00 |
| forged_self | 16 | +nan | +0.129 | -0.815 | +0.189 | +nan [nan, nan] | nan | 0.00 |
| forged_other | 4 | -0.167 | +0.050 | -0.903 | +0.077 | +0.833 [nan, nan] | 0.583 | 0.02 |
| forged_other | 8 | +nan | +0.204 | -0.875 | +0.092 | +nan [nan, nan] | nan | 0.00 |
| forged_other | 16 | +nan | +0.240 | -0.822 | +0.143 | +nan [nan, nan] | nan | 0.00 |
| forged_unlabeled | 4 | +0.000 | -0.047 | -0.987 | +0.089 | +1.000 [nan, nan] | 0.500 | 0.02 |
| forged_unlabeled | 8 | +nan | +0.147 | -0.967 | +0.135 | +nan [nan, nan] | nan | 0.00 |
| forged_unlabeled | 16 | +nan | +0.151 | -0.925 | +0.199 | +nan [nan, nan] | nan | 0.00 |

### Record effect (forged minus genuine record, same strings) and resistance

| condition | k | ft report effect | base report effect | resistance (report) | ft behaviour effect | base behaviour effect | resistance (behaviour) |
|---|---|---|---|---|---|---|---|
| forged_self | 4 | +nan [nan, nan] | +0.268 [0.109, 0.436] | +nan [nan, nan] | +0.113 [-0.003, 0.342] | +0.207 [0.147, 0.272] | +0.094 [-0.153, 0.261] |
| forged_self | 8 | +nan [nan, nan] | +0.415 [0.213, 0.601] | +nan [nan, nan] | +0.084 [-0.102, 0.353] | +0.269 [0.217, 0.336] | +0.186 [-0.104, 0.393] |
| forged_self | 16 | +nan [nan, nan] | +0.275 [0.171, 0.405] | +nan [nan, nan] | +0.132 [-0.027, 0.420] | +0.374 [0.299, 0.449] | +0.242 [-0.108, 0.457] |
| forged_other | 4 | +nan [nan, nan] | +0.163 [0.054, 0.298] | +nan [nan, nan] | +0.088 [-0.003, 0.266] | +0.192 [0.124, 0.263] | +0.105 [-0.076, 0.244] |
| forged_other | 8 | +nan [nan, nan] | +0.439 [0.205, 0.643] | +nan [nan, nan] | +0.067 [-0.105, 0.305] | +0.236 [0.178, 0.302] | +0.169 [-0.087, 0.364] |
| forged_other | 16 | +nan [nan, nan] | +0.386 [0.134, 0.750] | +nan [nan, nan] | +0.125 [-0.041, 0.411] | +0.328 [0.250, 0.408] | +0.203 [-0.147, 0.418] |
| forged_unlabeled | 4 | +nan [nan, nan] | +0.066 [-0.004, 0.131] | +nan [nan, nan] | +0.004 [-0.006, 0.019] | +0.204 [0.112, 0.282] | +0.200 [0.110, 0.280] |
| forged_unlabeled | 8 | +nan [nan, nan] | +0.382 [0.222, 0.546] | +nan [nan, nan] | -0.025 [-0.146, 0.072] | +0.279 [0.191, 0.382] | +0.304 [0.150, 0.476] |
| forged_unlabeled | 16 | +nan [nan, nan] | +0.297 [0.127, 0.521] | +nan [nan, nan] | +0.023 [-0.128, 0.198] | +0.383 [0.278, 0.499] | +0.361 [0.109, 0.557] |

### Self-attribution and detection

| contrast | k | difference [95% CI] | n |
|---|---|---|---|
| detection self (base) | 4 | -0.025 [-0.073, 0.023] | 7 |
| detection self (base) | 8 | -0.025 [-0.058, 0.015] | 7 |
| detection self (base) | 16 | -0.004 [-0.027, 0.020] | 7 |
| detection self (ft) | 4 | -0.031 [-0.074, 0.012] | 7 |
| detection self (ft) | 8 | -0.008 [-0.033, 0.018] | 7 |
| detection self (ft) | 16 | -0.017 [-0.038, 0.005] | 7 |
| detection third (base) | 4 | -0.046 [-0.103, 0.012] | 7 |
| detection third (base) | 8 | -0.003 [-0.043, 0.035] | 7 |
| detection third (base) | 16 | -0.029 [-0.119, 0.048] | 7 |
| detection third (ft) | 4 | -0.105 [-0.227, 0.002] | 7 |
| detection third (ft) | 8 | -0.007 [-0.142, 0.137] | 7 |
| detection third (ft) | 16 | +0.044 [-0.077, 0.182] | 7 |
| self-attribution (base) | 4 | +0.106 [0.002, 0.226] | 7 |
| self-attribution (base) | 8 | -0.023 [-0.098, 0.052] | 7 |
| self-attribution (base) | 16 | -0.111 [-0.365, 0.087] | 7 |
| self-attribution (ft) | 4 | +nan [nan, nan] | 0 |
| self-attribution (ft) | 8 | +nan [nan, nan] | 0 |
| self-attribution (ft) | 16 | +nan [nan, nan] | 0 |

Mean P(yes) by record type:

| model | framing | k | genuine | forged |
|---|---|---|---|---|
| base | self | 4 | 0.376 | 0.402 |
| base | self | 8 | 0.318 | 0.343 |
| base | self | 16 | 0.298 | 0.302 |
| base | third | 4 | 0.243 | 0.289 |
| base | third | 8 | 0.377 | 0.380 |
| base | third | 16 | 0.487 | 0.516 |
| ft | self | 4 | 0.331 | 0.362 |
| ft | self | 8 | 0.371 | 0.379 |
| ft | self | 16 | 0.370 | 0.386 |
| ft | third | 4 | 0.347 | 0.453 |
| ft | third | 8 | 0.465 | 0.472 |
| ft | third | 16 | 0.627 | 0.583 |

## Level 4 (5 characters)

### Shift in each context

| condition | k | ft report | base report | ft behaviour | base behaviour | ft report - behaviour (decisive) | ft faith in context | ft parse rate |
|---|---|---|---|---|---|---|---|---|
| none | 0 | +nan | +0.066 | -0.999 | +0.008 | – | nan | 0.00 |
| genuine_self | 4 | +nan | -0.019 | -0.926 | -0.059 | +nan [nan, nan] | nan | 0.00 |
| genuine_self | 8 | +nan | -0.026 | -0.970 | -0.101 | +nan [nan, nan] | nan | 0.00 |
| genuine_self | 16 | -0.333 | -0.101 | -0.995 | -0.098 | +0.667 [nan, nan] | 0.667 | 0.03 |
| forged_self | 4 | +nan | +0.061 | -0.830 | +0.052 | +nan [nan, nan] | nan | 0.00 |
| forged_self | 8 | +nan | +0.010 | -0.461 | +0.044 | +nan [nan, nan] | nan | 0.00 |
| forged_self | 16 | +nan | +0.009 | -0.476 | +0.053 | +nan [nan, nan] | nan | 0.00 |
| forged_other | 4 | +0.000 | -0.013 | -0.864 | +0.033 | +0.833 [nan, nan] | 0.500 | 0.03 |
| forged_other | 8 | +0.000 | +0.086 | -0.528 | +0.033 | +0.500 [nan, nan] | 0.583 | 0.03 |
| forged_other | 16 | +nan | +0.009 | -0.490 | +0.040 | +nan [nan, nan] | nan | 0.00 |
| forged_unlabeled | 4 | -0.250 | -0.006 | -0.996 | +0.051 | +0.750 [nan, nan] | 0.625 | 0.03 |
| forged_unlabeled | 8 | +nan | -0.015 | -0.912 | +0.085 | +nan [nan, nan] | nan | 0.00 |
| forged_unlabeled | 16 | +nan | -0.031 | -0.839 | +0.069 | +nan [nan, nan] | nan | 0.00 |

### Record effect (forged minus genuine record, same strings) and resistance

| condition | k | ft report effect | base report effect | resistance (report) | ft behaviour effect | base behaviour effect | resistance (behaviour) |
|---|---|---|---|---|---|---|---|
| forged_self | 4 | +nan [nan, nan] | +0.080 [-0.019, 0.225] | +nan [nan, nan] | +0.096 [-0.103, 0.295] | +0.111 [0.029, 0.229] | +0.015 [-0.153, 0.170] |
| forged_self | 8 | +nan [nan, nan] | +0.036 [-0.037, 0.116] | +nan [nan, nan] | +0.508 [0.064, 1.119] | +0.145 [0.023, 0.289] | -0.363 [-0.834, -0.028] |
| forged_self | 16 | +nan [nan, nan] | +0.111 [0.002, 0.273] | +nan [nan, nan] | +0.519 [0.083, 1.098] | +0.151 [-0.018, 0.371] | -0.368 [-0.733, -0.107] |
| forged_other | 4 | +nan [nan, nan] | +0.006 [-0.062, 0.090] | +nan [nan, nan] | +0.062 [-0.121, 0.253] | +0.093 [0.009, 0.202] | +0.030 [-0.081, 0.174] |
| forged_other | 8 | +nan [nan, nan] | +0.112 [0.013, 0.259] | +nan [nan, nan] | +0.442 [0.049, 1.002] | +0.135 [0.020, 0.263] | -0.307 [-0.738, -0.027] |
| forged_other | 16 | +nan [nan, nan] | +0.111 [0.008, 0.235] | +nan [nan, nan] | +0.505 [0.071, 1.097] | +0.138 [-0.019, 0.336] | -0.366 [-0.764, -0.092] |
| forged_unlabeled | 4 | +nan [nan, nan] | +0.013 [-0.076, 0.102] | +nan [nan, nan] | -0.070 [-0.217, 0.010] | +0.110 [0.017, 0.230] | +0.180 [0.041, 0.319] |
| forged_unlabeled | 8 | +nan [nan, nan] | +0.011 [-0.046, 0.068] | +nan [nan, nan] | +0.058 [-0.033, 0.202] | +0.186 [0.038, 0.354] | +0.128 [0.022, 0.224] |
| forged_unlabeled | 16 | +nan [nan, nan] | +0.071 [-0.006, 0.151] | +nan [nan, nan] | +0.156 [-0.000, 0.446] | +0.167 [0.005, 0.366] | +0.011 [-0.107, 0.129] |

### Self-attribution and detection

| contrast | k | difference [95% CI] | n |
|---|---|---|---|
| detection self (base) | 4 | +0.001 [-0.051, 0.053] | 5 |
| detection self (base) | 8 | +0.046 [0.002, 0.085] | 5 |
| detection self (base) | 16 | +0.022 [-0.068, 0.095] | 5 |
| detection self (ft) | 4 | +0.042 [0.009, 0.069] | 5 |
| detection self (ft) | 8 | +0.030 [-0.011, 0.059] | 5 |
| detection self (ft) | 16 | +0.017 [-0.000, 0.032] | 5 |
| detection third (base) | 4 | -0.005 [-0.062, 0.048] | 5 |
| detection third (base) | 8 | +0.034 [-0.036, 0.075] | 5 |
| detection third (base) | 16 | -0.020 [-0.073, 0.031] | 5 |
| detection third (ft) | 4 | +0.074 [0.016, 0.158] | 5 |
| detection third (ft) | 8 | +0.091 [-0.073, 0.332] | 5 |
| detection third (ft) | 16 | +0.060 [-0.021, 0.198] | 5 |
| self-attribution (base) | 4 | +0.074 [0.023, 0.137] | 5 |
| self-attribution (base) | 8 | -0.076 [-0.154, 0.001] | 5 |
| self-attribution (base) | 16 | -0.000 [-0.072, 0.072] | 5 |
| self-attribution (ft) | 4 | +nan [nan, nan] | 0 |
| self-attribution (ft) | 8 | +nan [nan, nan] | 0 |
| self-attribution (ft) | 16 | +nan [nan, nan] | 0 |

Mean P(yes) by record type:

| model | framing | k | genuine | forged |
|---|---|---|---|---|
| base | self | 4 | 0.386 | 0.385 |
| base | self | 8 | 0.340 | 0.294 |
| base | self | 16 | 0.340 | 0.318 |
| base | third | 4 | 0.270 | 0.275 |
| base | third | 8 | 0.313 | 0.279 |
| base | third | 16 | 0.339 | 0.359 |
| ft | self | 4 | 0.914 | 0.873 |
| ft | self | 8 | 0.904 | 0.874 |
| ft | self | 16 | 0.907 | 0.890 |
| ft | third | 4 | 0.980 | 0.907 |
| ft | third | 8 | 0.950 | 0.860 |
| ft | third | 16 | 0.954 | 0.895 |

