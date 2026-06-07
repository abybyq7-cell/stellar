# Submission Order

The primary submissions now use the full 15-model level-1 pool. This follows the
"add models, do not delete unless extremely redundant and weak" rule.

1. `001_lr_logits_full15_all.csv`
   - Source: `outputs/stacking/stack_full_15_lr_all/lr_logits_submission.csv`
   - CV: LR logits over all 15 level-1 models, accuracy 0.969663, balanced accuracy 0.959017, logloss 0.085524

2. `002_autogluon_full15_all.csv`
   - Source: `outputs/stacking/stack_full_15_ag_all_10m/autogluon_submission.csv`
   - CV: AutoGluon stacker over all 15 level-1 models, holdout neg logloss -0.075800

3. `003_best_single_lgbm_autofe_full.csv`
   - Source: `outputs/layer1_oof/layer1_full_selected_gpu/single_model_submissions/lgbm_gbdt_s777_autofe.csv`
   - CV: best full level-1 single model, accuracy 0.969303, balanced accuracy 0.958285, logloss 0.086292

Two-stage diagnostic submissions from the earlier stack exploration:

4. `004_two_stage_accuracy_conservative.csv`
   - Source: earlier two-stage stack exploration
   - Note: conservative accuracy-oriented variant

5. `005_two_stage_balanced_aggressive.csv`
   - Source: earlier two-stage stack exploration
   - Note: aggressive balanced-accuracy-oriented variant

6. `006_two_stage_stargalaxy_reduced_swaps.csv`
   - Source: earlier two-stage stack exploration
   - Note: reduced swap variant around STAR/GALAXY corrections

Backups from the earlier 7-model selected pool:

7. `007_lr_logits_7model_backup.csv`
8. `008_autogluon_7model_backup.csv`
9. `009_layer1_7model_backup.csv`

New full-pool submissions after adding AutoGluon weighted and learned two-stage OOF
models:

10. `010_lr_logits_full19_corr9995.csv`
   - Source: `outputs/stacking/stack_full19_corr9995_lr/lr_logits_submission.csv`
   - CV: LR logits over 17 selected models from a 19-model pool, accuracy 0.969588, balanced accuracy 0.958730, logloss 0.085361
   - Selection: dropped `xgb_s920_targeted_wide` and `lgbm_xt_s314_groupagg` for probability correlation >= 0.9995 with stronger kept models

11. `011_lr_logits_full19_all.csv`
   - Source: `outputs/stacking/stack_full19_all_lr/lr_logits_submission.csv`
   - CV: LR logits over all 19 models, accuracy 0.969613, balanced accuracy 0.958757, logloss 0.085372
   - Note: no correlation pruning; this is the add-only control submission

12. `012_full19_threshold_accuracy_conservative.csv`
   - Source: `outputs/two_stage_threshold/full19_corr9995_lr_aggressive_grid_v2/best_accuracy_submission.csv`
   - CV: threshold search on `010` probabilities, accuracy 0.969659, balanced accuracy 0.958694
   - Thresholds: qso 0.36, star 0.52, replace 0.45

13. `013_full19_threshold_balanced_aggressive.csv`
   - Source: `outputs/two_stage_threshold/full19_corr9995_lr_aggressive_grid_v2/best_balanced_submission.csv`
   - CV: aggressive threshold search on `010` probabilities, accuracy 0.963836, balanced accuracy 0.966521
   - Thresholds: qso 0.30, star 0.20, replace 0.00

Follow-up blends of the two public-score leaders:

14. `014_blend005_013_w013080_qso305_star185.csv`
   - Source: probability blend of `005` and `013` LR-stack probability files
   - OOF signal: `0.2 * 005 + 0.8 * 013`, then two-stage thresholds qso 0.305, star 0.185
   - CV: accuracy 0.963237, balanced accuracy 0.966613 on the shared OOF rows
   - Test diffs: 686 vs `005`, 355 vs `013`

15. `015_disagreement_oof_balanced_arbitration.csv`
   - Source: start from `013`, then use OOF balanced-accuracy utility to arbitrate only 005/013 disagreement flows
   - Rule changes from `013`: use `005` when `005=QSO,013=GALAXY` or `005=STAR,013=QSO`
   - CV: accuracy 0.963672, balanced accuracy 0.966666 on the shared OOF rows
   - Test diffs: 582 vs `005`, 120 vs `013`

Full21 weighted-metric round using class weights GALAXY=1.0, QSO=3.2, STAR=4.6:

16. `016_full21_weighted_lr_cat_reduced.csv`
   - Source: `outputs/stacking/stack_full21_weighted_lr_cat_reduced/lr_logits_submission.csv`
   - Added OOF: `lgbm_conservative_targeted_wide_s2756` with `star_lowz_hard`, and `xgboost_star_recall_targeted_s2703` with no weighting
   - Selection: 21 base models, 15 selected. Dropped/zeroed weak CatBoost variants `cat_s42_baseline`, `cat_s920_targeted`, `ag_cat_s1303_autofe_hardsgq`; low-weighted `cat_s2026_autofe` at 0.35 and `cat_s921_targeted_wide` at 0.25; replaced correlated `xgb_s921_targeted` with the new XGB candidate
   - CV: accuracy 0.961403, balanced accuracy 0.967153, weighted accuracy 0.967143

17. `017_full21_weighted_threshold.csv`
   - Source: `outputs/two_stage_threshold/full21_weighted_lr_threshold_grid/best_weighted_submission.csv`
   - CV: best weighted threshold on `016` probabilities, accuracy 0.961429, balanced accuracy 0.967163, weighted accuracy 0.967154
   - Thresholds: qso 0.46, star 0.38, replace 0.57

18. `018_full21_weighted_vs_015_arbitration.csv`
   - Source: `outputs/disagreement_arbitration/full21_weighted_vs_015_arbitration/arbitrated_submission.csv`
   - CV: start from `017`, then use OOF weighted-accuracy utility against reconstructed `015` on disagreement flows
   - Selected flows: `GALAXY -> QSO` and `GALAXY -> STAR` where `015` had positive weighted utility
   - CV: accuracy 0.961370, balanced accuracy 0.967174, weighted accuracy 0.967165
   - Test diffs: 13 vs `017`, 1313 vs `015`

Single-model check from the strongest new conservative LGBM:

19. `019_lgbm_conservative_targeted_wide_s2756.csv`
   - Source: `outputs/layer1_oof/layer1_full_lgbm_conservative_targeted_wide_star_lowz_hard/single_model_submissions/lgbm_conservative_targeted_wide_s2756.csv`
   - CV: accuracy 0.966736, balanced accuracy 0.960704, logloss 0.091663
   - Training: full train, `targeted_wide` features, `star_lowz_hard` sample weighting
   - Test distribution: GALAXY=160529, QSO=49946, STAR=36960
   - Test diffs: 4482 vs `016`, 4488 vs `018`, 3770 vs `013`

20. `020_most_aggressive_single_ag_cat_s1303_autofe_hardsgq.csv`
   - Source: `outputs/layer1_oof/layer1_full_ag_weighted_twostage/single_model_submissions/ag_cat_s1303_autofe_hardsgq.csv`
   - Definition: highest STAR test prediction count among full-train single-model submissions
   - CV: accuracy 0.960242, balanced accuracy 0.955586, logloss 0.109000
   - Training: AutoGluon CAT extension, `autofe` features, `hard_sgq_to_galaxy` weighting
   - Test distribution: GALAXY=159186, QSO=50790, STAR=37459
   - Test diffs: 4314 vs `019`, 5095 vs `018`, 5027 vs `013`

Full25 weighted-metric round after adding fresh AutoGluon seeds, HGB local-guard
diversity, and a local 015/018 GALAXY guard diagnostic:

21. `021_full25_fresh_hgb_weighted_lr.csv`
   - Source: `outputs/stacking/stack_full25_fresh_hgb_weighted_lr/lr_logits_submission.csv`
   - Added OOF: `ag_gbm_s2305_targeted_wide_starw`, `ag2_xgbgbm_s2308_targeted_wide_hardsgq`, `hgb_local_guard_starhard_s3101`, and `hgb_local_guard_bothwrong_s3103`
   - Selection: 25 base models, 18 selected. Kept fresh AG models, low-weighted `hgb_local_guard_starhard_s3101` at 0.35, zeroed `hgb_local_guard_bothwrong_s3103`; kept prior weak CatBoost reductions/zeros
   - CV: accuracy 0.961476, balanced accuracy 0.967184, weighted accuracy 0.967176, logloss 0.107224
   - Test distribution: GALAXY=156311, QSO=51353, STAR=39771

22. `022_full25_fresh_hgb_weighted_threshold.csv`
   - Source: `outputs/two_stage_threshold/full25_fresh_hgb_weighted_lr_threshold_grid/best_weighted_submission.csv`
   - CV: best weighted threshold on `021` probabilities, accuracy 0.961491, balanced accuracy 0.967199, weighted accuracy 0.967191
   - Thresholds: qso 0.45, star 0.33, replace 0.63
   - Test distribution: GALAXY=156314, QSO=51342, STAR=39779
   - Test diffs: 11 vs `021`

23. `023_full25_weighted_vs_015_arbitration.csv`
   - Source: `outputs/disagreement_arbitration/full25_weighted_vs_015_arbitration/arbitrated_submission.csv`
   - CV: start from `022`, then use OOF weighted-accuracy utility against reconstructed `015` on disagreement flows
   - Selected flows: `GALAXY -> STAR` and `GALAXY -> QSO` where `015` had positive weighted utility
   - CV: accuracy 0.961389, balanced accuracy 0.967210, weighted accuracy 0.967202
   - Test distribution: GALAXY=156298, QSO=51344, STAR=39793
   - Test diffs: 16 vs `022`, 179 vs `018`, 1266 vs `015`

Local guard diagnostic not submitted:

- `outputs/local_guard_threshold/local_guard_018_hgb_v1/guard_submission.csv`
  - Binary local GALAXY-vs-STAR model: OOF AUC 0.996183, logloss 0.076702
  - Best guard grid chose no flips; nonzero recovery settings recovered many 015-only true GALAXY rows but injured more correct STAR/QSO rows, reducing weighted accuracy

Full26 neural-diversity round after adding the full 5-fold sklearn MLP OOF to the
level-1 pool with low feature weight:

24. `024_full26_nn_balanced_lr.csv`
   - Source: `outputs/stacking/stack_full26_nn_balanced_lr/lr_logits_submission.csv`
   - Added OOF: `skmlp_targeted_starqso_s4101` from `layer1_full_nn_skmlp_targeted_starqso`
   - Selection: 26 base models, 19 selected. MLP was kept for diversity but low-weighted at 0.20; weak CatBoost/HGB variants kept zeroed or reduced as in the weighted full25 stack
   - CV: accuracy 0.961669, balanced accuracy 0.967192, weighted accuracy 0.967183, logloss 0.107302
   - Test distribution: GALAXY=156373, QSO=51350, STAR=39712

25. `025_full26_nn_balanced_threshold.csv`
   - Source: `outputs/two_stage_threshold/full26_nn_balanced_lr_threshold_grid/best_weighted_submission.csv`
   - CV: best weighted threshold on `024` probabilities, accuracy 0.961692, balanced accuracy 0.967216, weighted accuracy 0.967208
   - Thresholds: qso 0.47, star 0.33, replace 0.51
   - Test distribution: GALAXY=156380, QSO=51316, STAR=39739
   - Test diffs: 35 vs `024`

26. `026_full26_nn_weighted_vs_015_arbitration.csv`
   - Source: `outputs/disagreement_arbitration/full26_nn_weighted_vs_015_arbitration/arbitrated_submission.csv`
   - CV: start from `025`, then use OOF weighted-accuracy utility against reconstructed `015` on disagreement flows
   - Selected flow: `GALAXY -> QSO` where `015` had positive weighted utility
   - CV: accuracy 0.961671, balanced accuracy 0.967221, weighted accuracy 0.967213
   - Test distribution: GALAXY=156378, QSO=51318, STAR=39739
   - Test diffs: 2 vs `025`, 1210 vs `015`

AutoGluon level-2 diagnostic not submitted:

- `outputs/stacking/stack_full26_nn_autogluon_oof_fast/autogluon_submission.csv`
  - 5-fold AutoGluon level-2 OOF over the same full26 stack features
  - CV: accuracy 0.969505, balanced accuracy 0.958931, weighted accuracy 0.958845, logloss 0.088045
  - Diagnosis: strong overall accuracy but too conservative for the current weighted/balanced target; test STAR count was only 35382
- `outputs/two_stage_threshold/full26_nn_autogluon_oof_fast_threshold_grid/best_weighted_submission.csv`
  - CV after aggressive threshold: accuracy 0.966926, balanced accuracy 0.965089, weighted accuracy 0.965041
  - Still below `025`/`026`, so kept as diagnostic rather than numbered submission
- `outputs/disagreement_arbitration/full26_015_arbitration_vs_ag_threshold_arbitration/arbitrated_submission.csv`
  - OOF flow arbitration from `026` to AG-threshold selected a tiny `GALAXY -> QSO` flow
  - CV: accuracy 0.961669, balanced accuracy 0.967226, weighted accuracy 0.967218
  - Not numbered because the selected flow had zero test hits; the produced test file is identical to `026`

Full31 hard-slice pairwise round:

27. `027_full31_pairwise_lr.csv`
   - Source: `outputs/stacking/stack_full31_pairwise_balanced_lr/lr_logits_submission.csv`
   - Added OOF: five local pairwise HGB-derived blocks from `layer1_full_pairwise_local_hard`
   - Pairwise binary signal: GALAXY-vs-STAR OOF AUC 0.995672, GALAXY-vs-QSO OOF AUC 0.997961
   - Selection: 31 base models, 23 selected. Best new block `pair_joint_hard_blend_a025_s3303` had single-block balanced accuracy 0.966710 and was kept at feature weight 0.70; more aggressive pairwise blocks were low-weighted
   - CV: accuracy 0.961446, balanced accuracy 0.967240, weighted accuracy 0.967231, logloss 0.107212
   - Test distribution: GALAXY=156353, QSO=51336, STAR=39746
   - Test diffs: 275 vs `026`

28. `028_full31_pairwise_threshold.csv`
   - Source: `outputs/two_stage_threshold/full31_pairwise_balanced_lr_threshold_grid/best_weighted_submission.csv`
   - CV: best weighted threshold on `027` probabilities, accuracy 0.961469, balanced accuracy 0.967269, weighted accuracy 0.967261
   - Thresholds: qso 0.48, star 0.29, replace 0.51
   - Test distribution: GALAXY=156360, QSO=51308, STAR=39767
   - Test diffs: 265 vs `026`, 31 vs `027`

29. `029_full31_pairwise_vs_026_arbitration.csv`
   - Source: `outputs/disagreement_arbitration/full31_pairwise_threshold_vs_026_arbitration/arbitrated_submission.csv`
   - CV: start from `028`, then use OOF weighted-accuracy utility against `026`
   - Selected flows: `QSO -> STAR` and `STAR -> QSO` where `026` had positive weighted utility
   - CV: accuracy 0.961476, balanced accuracy 0.967281, weighted accuracy 0.967273
   - Test distribution: GALAXY=156360, QSO=51303, STAR=39772
   - Public-safe check: weighted OOF improved over `026`; test diffs 246 vs `026`; in the narrow lowz+Blue_Cloud+compact_color+low_mag_std slice it changes only 3 test rows vs `026`, avoiding a large hard-slice rewrite
