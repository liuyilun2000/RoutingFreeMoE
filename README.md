# RoutingFreeMoE

## model size
### Small
--- BASELINE model ---
Print trainable params: 92,444,160 || all params: 92,444,160 || trainable%: 100.00
Print filtered model size: 40,980,992 || all params: 92,444,160 || excluded%: 44.33

--- RF model ---
Print trainable params: 93,845,280 || all params: 93,845,280 || trainable%: 100.00
Print filtered model size: 42,382,112 || all params: 93,845,280 || excluded%: 45.16

### Middle
--- BASELINE model ---
Print filtered model size: 212,669,184 || all params: 289,863,936 || excluded%: 73.37

--- RF model ---
Print filtered model size: 230,069,760 || all params: 307,264,512 || excluded%: 74.88

### large
--- BASELINE model ---
Print filtered model size: 302,433,280 || all params: 405,359,616 || excluded%: 74.61

--- RF model ---
Print filtered model size: 309,511,936 || all params: 412,438,272 || excluded%: 75.04

### eval
  sbatch eval_benchmarks.sh output_baseline_mixtral/1_mixtral_baseline_12L_128Dx12E_top2_lr_1e-3/final_model baseline ./results/baseline.json

  sbatch eval_benchmarks.sh ./output/mixtral_rf/mixtral_rf_12L_128Dx12E_temp_1.0_thres_1.0_density_0.25_lambda_1e-10_eta_0.02_aux_[E0.5_T0.5]_lr_2e-3/final_model routing_free ./results/rf.json

======================================================================
BENCHMARK RESULTS
======================================================================
Task                 Size       Metric                  Value
----------------------------------------------------------------------
arc_easy             2376       acc_norm               0.3354
piqa                 1838       acc_norm               0.5729
winogrande           1267       acc                    0.5233
hellaswag            10042      acc_norm               0.2699
mnli                 9815       acc                    0.3236
qnli                 5463       acc                    0.4962
sst2                 872        acc                    0.4908
======================================================================
Avg Acc                                                0.4303
Weighted Avg Acc                                       0.3643
======================================================================

======================================================================
BENCHMARK RESULTS
======================================================================
Task                 Size       Metric                  Value
----------------------------------------------------------------------
arc_easy             2376       acc_norm               0.3468
piqa                 1838       acc_norm               0.5783
winogrande           1267       acc                    0.5399
hellaswag            10042      acc_norm               0.2691
mnli                 9815       acc                    0.3237
qnli                 5463       acc                    0.4946
sst2                 872        acc                    0.5505
======================================================================
Avg Acc                                                0.4433
Weighted Avg Acc                                       0.3673
======================================================================

M_rf
======================================================================
BENCHMARK RESULTS
======================================================================
Task                 Size       Metric                  Value
----------------------------------------------------------------------
arc_easy             2376       acc_norm               0.3527
piqa                 1838       acc_norm               0.5914
winogrande           1267       acc                    0.4988
hellaswag            10042      acc_norm               0.2778
mnli                 9815       acc                    0.3201
qnli                 5463       acc                    0.4955
sst2                 872        acc                    0.5734
======================================================================
Avg Acc                                                0.4443
Weighted Avg Acc                                       0.3693
======================================================================