# Exploratory (not registered)

**text signals (15)**, leave one round out: 0.420 [0.379, 0.463] on 1431 pairs. Per round: coh_long 0.57, coh_pair 0.55, fill_pair 0.45, gq_pair 0.16, llm_rating 0.88, llm_rating2 0.29, ng_pair 0.51, p1t_pair 0.64, planner_pair 0.62, q1t_pair 0.65, q3_long 0.93, q3_pair 0.55, tw_pair 0.45.

Weights fitted on all rounds (standardised differences): burst_cw -0.82, oov +0.79, surp_cf +0.53, surp_mean -0.41, commit_text +0.33, junk -0.26, sl_cv_h -0.24, ent_reuse +0.23, rep3 +0.18, commit_prompt -0.16, surp_ac1 +0.10, surp_sd -0.08, fw_js -0.07, fw_share_h +0.05, sl_mean_h +0.05.

**text signals + SCRIBE (17)**, leave one round out: 0.467 [0.426, 0.507] on 1424 pairs. Per round: coh_long 0.60, coh_pair 0.54, fill_pair 0.52, gq_pair 0.18, llm_rating 0.90, llm_rating2 0.41, ng_pair 0.49, p1t_pair 0.73, planner_pair 0.52, q1t_pair 0.70, q3_long 0.79, q3_pair 0.64, tw_pair 0.53.

Weights fitted on all rounds (standardised differences): burst_cw -0.88, oov +0.79, scribe_v2 +0.72, surp_cf +0.60, scribe_v31 -0.45, surp_mean -0.31, sl_cv_h -0.27, ent_reuse +0.27, commit_text +0.25, junk -0.24, rep3 +0.18, commit_prompt -0.15, surp_sd -0.13, surp_ac1 +0.12, sl_mean_h +0.09, fw_js -0.09, fw_share_h +0.00.

surp_mean without gq_pair: 0.671 [0.629, 0.710] (1209 pairs).

scribe_v2 without gq_pair: 0.576 [0.535, 0.615] (1209 pairs).

plain K5 variants (llm_rating, llm_rating2, tw_pair, ng_pair, coh_pair, coh_long): surp_mean 0.738 [0.700, 0.774] (991 pairs).

plain K5 variants (llm_rating, llm_rating2, tw_pair, ng_pair, coh_pair, coh_long): surp_sd 0.296 [0.249, 0.342] (991 pairs).

plain K5 variants (llm_rating, llm_rating2, tw_pair, ng_pair, coh_pair, coh_long): commit_text 0.450 [0.405, 0.498] (991 pairs).

plain K5 variants (llm_rating, llm_rating2, tw_pair, ng_pair, coh_pair, coh_long): scribe_v2 0.547 [0.502, 0.589] (991 pairs).

planner and fill rounds (p1t_pair, planner_pair, q1t_pair, q3_pair, q3_long, fill_pair): surp_mean 0.367 [0.272, 0.465] (218 pairs).

planner and fill rounds (p1t_pair, planner_pair, q1t_pair, q3_pair, q3_long, fill_pair): surp_sd 0.725 [0.644, 0.807] (218 pairs).

planner and fill rounds (p1t_pair, planner_pair, q1t_pair, q3_pair, q3_long, fill_pair): commit_text 0.702 [0.618, 0.783] (218 pairs).

planner and fill rounds (p1t_pair, planner_pair, q1t_pair, q3_pair, q3_long, fill_pair): scribe_v2 0.706 [0.617, 0.784] (218 pairs).

today's writer, K5 from 337 MB (tw_pair, ng_pair, coh_pair, coh_long): surp_mean 0.643 [0.596, 0.689], minus SCRIBE v2 +0.079 [+0.007, +0.150] (594 pairs).

today's writer, K5 from 337 MB (tw_pair, ng_pair, coh_pair, coh_long): scribe_v2 0.566 [0.514, 0.618], minus SCRIBE v2 +0.000 [+0.000, +0.000] (594 pairs).

today's writer, K5 from 337 MB (tw_pair, ng_pair, coh_pair, coh_long): scribe_v31 0.468 [0.412, 0.524], minus SCRIBE v2 -0.098 [-0.148, -0.049] (594 pairs).
