# From-scratch GPT-2, regex rules: reports by world, family and group

Means over seeds and characters.  `faith` = agreement of the reported regex (formula world) or stated letter (attribute world: 'b' = contains b, 'NOT b' = has no b) with the model's own accept/reject decisions on held-out strings (chance about 0.5); `shuffled` = the same reports scored against other characters of the same group and family; `balanced` averages agreement on accepted and rejected strings; `error` = faithfulness on strings where the model departs from its rule.  Per-letter columns as in regexintro.diagnose: `sign_ok` (letter directions right), `irr_split` (splits letters the rule treats alike), `val_corr_avg` (average report's letter valences vs the rule's), `assoc_ceiling` (best one-letter rule), `presence_r2` (share of the rule explained by letter presence).

## formula world (formula reports), pretraining

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
pre_choice L1              1.0          0.5        0.99   0.98            0.98      0.46          NaN         0.91          0.98
           L2              1.0          0.5        1.00   0.94            0.94      0.49         0.00         0.86          0.94
           L3              1.0          0.5        1.00   0.97            0.97      0.49         0.50         0.91          0.97
           L4              1.0          0.5        0.95   0.93            0.93      0.47         0.12         0.85          0.93
pre_report L1              1.0          0.5        1.00   1.00            1.00      0.46          NaN         1.00          1.00
           L2              1.0          0.5        1.00   1.00            1.00      0.49         0.00         1.00          1.00
           L3              1.0          0.5        1.00   1.00            1.00      0.49         0.00         1.00          1.00
           L4              1.0          0.5        1.00   1.00            1.00      0.47         0.00         1.00          1.00
```

## formula world (formula reports), finetune ft_all_replay0.5_s0

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
new        L1             1.00         0.50        0.78   0.79            0.79      0.50          NaN         0.28          0.79
           L2             1.00         0.50        0.94   0.78            0.78      0.51         1.00         0.26          0.78
           L3             1.00         0.50        0.76   0.81            0.81      0.49         0.33         0.55          0.81
           L4             1.00         0.50        0.88   0.80            0.80      0.50          NaN         0.53          0.80
new_before L1             0.50         0.02        0.67   0.76            0.65      0.72         0.76         0.00          0.50
           L2             0.50         0.02        0.70   0.74            0.63      0.73         0.73         0.02          0.50
           L3             0.51         0.02        0.60   0.70            0.57      0.70         0.71         0.01          0.49
           L4             0.50         0.02        0.60   0.71            0.58      0.71         0.72         0.00          0.49
```

## formula world (formula reports), finetune ft_all_replay0.5_s1

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
new        L1             1.00         0.50        0.98   0.79            0.79      0.49         0.65         0.38          0.79
           L2             1.00         0.50        0.96   0.58            0.58      0.50         0.00         0.07          0.58
           L3             1.00         0.50        0.96   0.71            0.71      0.48         0.43         0.38          0.71
           L4             1.00         0.50        0.93   0.52            0.52      0.48          NaN         0.20          0.52
new_before L1             0.52         0.19        0.66   0.66            0.52      0.65         0.62         0.07          0.54
           L2             0.49         0.20        0.55   0.67            0.51      0.66         0.68         0.00          0.48
           L3             0.51         0.23        0.55   0.64            0.51      0.65         0.66         0.00          0.50
           L4             0.50         0.21        0.62   0.66            0.54      0.65         0.66         0.00          0.50
```

## formula world (formula reports), finetune ft_all_replay0.5_s2

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
new        L1              1.0          0.5        0.81   0.64            0.64      0.48         1.00         0.20          0.64
           L2              1.0          0.5        0.68   0.53            0.53      0.50         0.00         0.00          0.53
           L3              1.0          0.5        0.65   0.48            0.48      0.50         0.49         0.01          0.48
           L4              1.0          0.5        0.64   0.52            0.52      0.50          NaN         0.00          0.52
new_before L1              0.5          0.0        0.07   0.78            0.76      0.81         0.80         0.00          0.48
           L2              0.5          0.0        0.06   0.84            0.84      0.83         0.87         0.00          0.47
           L3              0.5          0.0        0.04   0.81            0.81      0.81         0.78         0.00          0.52
           L4              0.5          0.0        0.06   0.81            0.81      0.81         0.81         0.00          0.51
```

## formula world (formula reports), finetune ft_embed_s0

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
new        L1             0.98         0.49        0.76   0.75            0.74      0.51         0.60         0.30          0.74
           L2             0.89         0.49        0.68   0.67            0.67      0.49         0.51         0.08          0.67
           L3             0.94         0.49        0.76   0.80            0.79      0.48         0.51         0.43          0.79
           L4             0.74         0.47        0.52   0.66            0.66      0.51         0.48         0.30          0.67
new_before L1             0.50         0.02        0.67   0.76            0.65      0.72         0.76         0.00          0.50
           L2             0.50         0.02        0.70   0.74            0.63      0.73         0.73         0.02          0.50
           L3             0.51         0.02        0.60   0.70            0.57      0.70         0.71         0.01          0.49
           L4             0.50         0.02        0.60   0.71            0.58      0.71         0.72         0.00          0.49
```

## formula world (formula reports), finetune ft_embed_s1

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
new        L1             1.00         0.50        0.92   0.79            0.79      0.49         0.35         0.50          0.79
           L2             0.91         0.50        0.73   0.64            0.64      0.49         0.48         0.13          0.65
           L3             0.94         0.50        0.81   0.76            0.76      0.49         0.50         0.37          0.76
           L4             0.92         0.50        0.68   0.74            0.74      0.48         0.63         0.43          0.71
new_before L1             0.52         0.19        0.66   0.66            0.52      0.65         0.62         0.07          0.54
           L2             0.49         0.20        0.55   0.67            0.51      0.66         0.68         0.00          0.48
           L3             0.51         0.23        0.55   0.64            0.51      0.65         0.66         0.00          0.50
           L4             0.50         0.21        0.62   0.66            0.54      0.65         0.66         0.00          0.50
```

## formula world (formula reports), finetune ft_embed_s2

```
                  decision_acc  accept_rate  parse_rate  faith  faith_balanced  shuffled  error_faith  exact_match  target_agree
group      level                                                                                                                
new        L1             0.97         0.48        0.92   0.57            0.56      0.49         0.87         0.14          0.55
           L2             0.82         0.50        0.90   0.52            0.52      0.52         0.61         0.03          0.50
           L3             0.81         0.48        0.91   0.56            0.55      0.50         0.58         0.06          0.56
           L4             0.87         0.47        0.84   0.62            0.61      0.49         0.51         0.20          0.61
new_before L1             0.50         0.00        0.07   0.78            0.76      0.81         0.80         0.00          0.48
           L2             0.50         0.00        0.06   0.84            0.84      0.83         0.87         0.00          0.47
           L3             0.50         0.00        0.04   0.81            0.81      0.81         0.78         0.00          0.52
           L4             0.50         0.00        0.06   0.81            0.81      0.81         0.81         0.00          0.51
```

## Linear probes, best layer (R² for valence/lang, accuracy for template/x_letter)

```
target                                     lang  template  valence  x_letter
world   model               group                                           
formula ft_all_replay0.5_s0 new            0.07      0.49     0.44      0.82
                            new_untrained -0.59      0.06    -0.10      0.38
                            pre_choice     0.92      0.98     0.94      1.00
                            pre_report     0.95      1.00     0.96      1.00
        ft_all_replay0.5_s1 new            0.07      0.48     0.39      0.77
                            new_untrained -0.43      0.07    -0.31      0.38
                            pre_choice     0.88      0.96     0.94      0.99
                            pre_report     0.95      1.00     0.97      1.00
        ft_all_replay0.5_s2 new           -0.87      0.14    -0.07      0.50
                            new_untrained -0.82      0.08    -0.12      0.37
                            pre_choice     0.89      1.00     0.94      1.00
                            pre_report     0.94      1.00     0.97      1.00
        ft_embed_s0         new           -0.06      0.31     0.31      0.61
                            new_untrained -0.48      0.08    -0.12      0.39
                            pre_choice     0.92      0.99     0.94      1.00
                            pre_report     0.96      1.00     0.97      1.00
        ft_embed_s1         new            0.06      0.48     0.30      0.74
                            new_untrained -0.39      0.09    -0.25      0.36
                            pre_choice     0.87      0.96     0.93      1.00
                            pre_report     0.96      1.00     0.97      1.00
        ft_embed_s2         new           -0.28      0.24     0.01      0.44
                            new_untrained -0.22      0.08    -0.04      0.36
                            pre_choice     0.90      0.98     0.95      1.00
                            pre_report     0.96      1.00     0.98      1.00
        pretrain            new_untrained -0.44      0.08    -0.18      0.36
                            pre_choice     0.89      0.98     0.93      1.00
                            pre_report     0.96      1.00     0.97      1.00
```
