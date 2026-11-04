# SCRIBE v3 paper: extra tables

Written by `scripts/scribe_eval/v3_report.py` from the runs of `scripts/scribe_eval/runs/v3_paper.sh`.

## R1 fresh battery, enwik9 [900, 924) MiB: mean, and P(human > attack) over prompts (ties 1/2)

| attack | n | SCRIBE v2 | SCRIBE v3 | SCRIBE v3.1 | SURPRISE | MATCH | SURPRISE 3.1 | MATCH 3.1 | min of 3 |
|---|---|---|---|---|---|---|---|---|---|
| human | 200 | 0.900 | 0.888 | 0.845 | 0.916 | 0.847 | 0.872 | 0.796 | 0.795 |
| copy_pack | 200 | 0.105 (1.000) | 0.050 (1.000) | 0.000 (1.000) | 0.045 (1.000) | 0.000 (1.000) | 0.014 (1.000) | 0.000 (1.000) | 0.000 (1.000) |
| echo_repeat | 200 | 0.000 (1.000) | 0.002 (1.000) | 0.000 (1.000) | 0.005 (1.000) | 0.000 (1.000) | 0.014 (1.000) | 0.000 (1.000) | 0.000 (1.000) |
| echo_copy | 200 | 0.883 (0.515) | 0.599 (0.985) | 0.000 (1.000) | 0.913 (0.542) | 0.000 (1.000) | 0.837 (0.522) | 0.000 (1.000) | 0.000 (1.000) |
| stuff_human | 200 | 0.408 (0.848) | 0.532 (0.895) | 0.254 (0.980) | 0.592 (0.868) | 0.598 (0.745) | 0.140 (0.980) | 0.453 (0.830) | 0.328 (0.905) |
| stuff_generic | 200 | 0.969 (0.315) | 0.877 (0.535) | 0.377 (0.975) | 0.802 (0.780) | 0.861 (0.438) | 0.094 (0.995) | 0.665 (0.655) | 0.728 (0.595) |
| loop | 200 | 0.000 (1.000) | 0.007 (1.000) | 0.000 (1.000) | 0.019 (1.000) | 0.003 (0.995) | 0.131 (0.995) | 0.003 (1.000) | 0.000 (1.000) |
| generic | 200 | 0.835 (0.760) | 0.654 (0.950) | 0.462 (0.975) | 0.653 (0.975) | 0.475 (0.970) | 0.318 (0.980) | 0.375 (0.955) | 0.475 (0.940) |
| shuffle_docs | 200 | 0.357 (0.980) | 0.517 (0.985) | 0.430 (0.970) | 0.668 (0.965) | 0.525 (0.970) | 0.583 (0.965) | 0.404 (0.945) | 0.345 (0.965) |
| cooc_pack | 200 | 0.130 (0.995) | 0.079 (1.000) | 0.000 (1.000) | 0.109 (0.995) | 0.000 (1.000) | 0.059 (0.995) | 0.000 (1.000) | 0.000 (1.000) |
| reorder | 200 | 0.891 (0.525) | 0.885 (0.517) | 0.841 (0.517) | 0.915 (0.502) | 0.847 (0.500) | 0.871 (0.507) | 0.797 (0.497) | 0.788 (0.535) |
| cooc_out | 200 | 0.886 (0.453) | 0.832 (0.627) | 0.792 (0.605) | 0.823 (0.698) | 0.788 (0.547) | 0.793 (0.647) | 0.751 (0.545) | 0.720 (0.632) |
| offprompt | 200 | 0.900 (0.492) | 0.846 (0.680) | 0.700 (0.835) | 0.916 (0.500) | 0.722 (0.745) | 0.730 (0.835) | 0.537 (0.880) | 0.703 (0.708) |
| ho_stuff7_rare | 200 | 0.803 (0.545) | 0.855 (0.477) | 0.704 (0.662) | 0.867 (0.525) | 0.896 (0.297) | 0.600 (0.757) | 0.845 (0.350) | 0.731 (0.497) |
| ho_stuff3_freq | 200 | 0.093 (0.983) | 0.352 (0.980) | 0.125 (1.000) | 0.425 (0.963) | 0.537 (0.780) | 0.060 (1.000) | 0.527 (0.755) | 0.082 (0.985) |
| ho_stuff_rand | 200 | 0.619 (0.715) | 0.711 (0.705) | 0.454 (0.915) | 0.737 (0.667) | 0.776 (0.502) | 0.305 (0.938) | 0.673 (0.605) | 0.520 (0.735) |
| ho_generic2 | 200 | 0.838 (0.662) | 0.636 (0.890) | 0.453 (0.925) | 0.492 (0.925) | 0.577 (0.910) | 0.272 (0.985) | 0.442 (0.920) | 0.447 (0.890) |
| ho_template | 200 | 1.000 (0.265) | 0.919 (0.495) | 0.846 (0.510) | 0.777 (0.805) | 0.978 (0.210) | 0.658 (0.790) | 0.973 (0.182) | 0.775 (0.540) |
| ho_echo_scatter | 200 | 0.374 (0.805) | 0.375 (0.995) | 0.000 (1.000) | 0.749 (0.708) | 0.000 (1.000) | 0.717 (0.672) | 0.000 (1.000) | 0.000 (1.000) |
| ho_title_swap | 200 | 0.900 (0.490) | 0.867 (0.545) | 0.788 (0.630) | 0.922 (0.482) | 0.780 (0.647) | 0.838 (0.570) | 0.673 (0.693) | 0.746 (0.578) |
| ho_list_related | 200 | 0.633 (0.790) | 0.684 (0.860) | 0.665 (0.790) | 0.729 (0.868) | 0.690 (0.780) | 0.750 (0.820) | 0.661 (0.725) | 0.551 (0.810) |

## R10 v2's attack set: mean, and P(human > attack) over prompts (ties 1/2)

| attack | n | SCRIBE v2 | SCRIBE v3 | SCRIBE v3.1 | SURPRISE | MATCH | SURPRISE 3.1 | MATCH 3.1 | min of 3 |
|---|---|---|---|---|---|---|---|---|---|
| human | 250 | 0.906 | 0.873 | 0.831 | 0.900 | 0.815 | 0.864 | 0.776 | 0.767 |
| copy_pack | 250 | 0.103 (0.992) | 0.048 (1.000) | 0.000 (1.000) | 0.042 (0.996) | 0.000 (1.000) | 0.011 (0.996) | 0.000 (1.000) | 0.000 (1.000) |
| offprompt | 250 | 0.906 (0.484) | 0.834 (0.636) | 0.688 (0.808) | 0.895 (0.502) | 0.700 (0.740) | 0.720 (0.816) | 0.528 (0.868) | 0.678 (0.690) |
| echo | 250 | 0.000 (1.000) | 0.002 (1.000) | 0.000 (1.000) | 0.006 (1.000) | 0.000 (1.000) | 0.018 (0.996) | 0.000 (1.000) | 0.000 (1.000) |
| stuffing | 250 | 0.015 (1.000) | 0.103 (0.992) | 0.020 (0.992) | 0.217 (0.988) | 0.076 (0.968) | 0.037 (0.988) | 0.054 (0.984) | 0.012 (0.968) |
| generic | 250 | 0.835 (0.792) | 0.656 (0.920) | 0.464 (0.964) | 0.653 (0.948) | 0.480 (0.948) | 0.319 (0.980) | 0.379 (0.940) | 0.480 (0.916) |
| generic_topic | 250 | 0.911 (0.704) | 0.953 (0.320) | 0.947 (0.288) | 0.979 (0.348) | 0.968 (0.202) | 0.965 (0.282) | 0.969 (0.174) | 0.896 (0.336) |
| numbers | 250 | 0.906 (0.498) | 0.872 (0.494) | 0.830 (0.516) | 0.899 (0.492) | 0.811 (0.502) | 0.862 (0.508) | 0.772 (0.512) | 0.766 (0.496) |
| negation | 250 | 0.908 (0.466) | 0.882 (0.420) | 0.842 (0.450) | 0.906 (0.534) | 0.831 (0.456) | 0.870 (0.588) | 0.791 (0.464) | 0.779 (0.486) |
| fwloop | 250 | 0.000 (1.000) | 0.030 (1.000) | 0.003 (1.000) | 0.012 (0.996) | 0.079 (0.968) | 0.007 (0.996) | 0.070 (0.984) | 0.000 (0.992) |

## R1 combination ablation

Ladder, adjacent-pair accuracy:

| combination | human > sent_shuffled | sent_shuffled > word_shuffled | word_shuffled > random_lexicon | random_lexicon > random_letters | random_letters > repeated_phrase |
|---|---|---|---|---|---|
| v3.1 | 0.966 | 1.000 | 0.991 | 1.000 | 1.000 |
| no v2 | 0.940 | 1.000 | 0.903 | 1.000 | 1.000 |
| no SURPRISE 3.1 | 0.973 | 1.000 | 0.994 | 1.000 | 1.000 |
| no MATCH 3.1 | 0.966 | 1.000 | 0.993 | 1.000 | 1.000 |
| plain mean | 0.972 | 1.000 | 0.860 | 1.000 | 1.000 |
| minimum | 0.959 | 1.000 | 0.998 | 1.000 | 1.000 |

Attacks, P(human > attack) over all (human, attack) pairs, ties 1/2 (the batteries' rule):

| attack | v3.1 | no v2 | no SURPRISE 3.1 | no MATCH 3.1 | plain mean | minimum |
|---|---|---|---|---|---|---|
| copy_pack | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| echo_repeat | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| echo_copy | 1.000 | 1.000 | 1.000 | 0.546 | 0.942 | 1.000 |
| stuff_human | 0.975 | 0.975 | 0.896 | 0.976 | 0.968 | 0.979 |
| stuff_generic | 0.976 | 0.983 | 0.586 | 0.985 | 0.935 | 0.987 |
| loop | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 | 1.000 |
| generic | 0.970 | 0.977 | 0.926 | 0.970 | 0.972 | 0.963 |
| shuffle_docs | 0.970 | 0.958 | 0.974 | 0.973 | 0.980 | 0.956 |
| cooc_pack | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 |
| reorder | 0.509 | 0.502 | 0.507 | 0.509 | 0.507 | 0.511 |
| cooc_out | 0.569 | 0.599 | 0.517 | 0.579 | 0.568 | 0.561 |
| offprompt | 0.821 | 0.869 | 0.802 | 0.725 | 0.815 | 0.833 |
| ho_stuff7_rare | 0.643 | 0.638 | 0.476 | 0.707 | 0.626 | 0.679 |
| ho_stuff3_freq | 0.995 | 0.988 | 0.983 | 0.999 | 0.993 | 0.993 |
| ho_stuff_rand | 0.878 | 0.882 | 0.713 | 0.904 | 0.862 | 0.899 |
| ho_generic2 | 0.953 | 0.972 | 0.880 | 0.952 | 0.952 | 0.948 |
| ho_template | 0.529 | 0.591 | 0.143 | 0.672 | 0.495 | 0.616 |
| ho_echo_scatter | 1.000 | 1.000 | 1.000 | 0.837 | 0.975 | 1.000 |
| ho_title_swap | 0.650 | 0.670 | 0.664 | 0.569 | 0.647 | 0.671 |
| ho_list_related | 0.827 | 0.777 | 0.809 | 0.850 | 0.833 | 0.790 |

## GPT-2 output dataset: P(WebText > system) over all text pairs

| system | SCRIBE v2 | SCRIBE v3 | SCRIBE v3.1 | SURPRISE | MATCH | SURPRISE 3.1 | MATCH 3.1 | min of 3 |
|---|---|---|---|---|---|---|---|---|
| xl-1542M | 0.639 | 0.665 | 0.664 | 0.632 | 0.646 | 0.638 | 0.648 | 0.651 |
| large-762M | 0.648 | 0.690 | 0.686 | 0.674 | 0.668 | 0.674 | 0.670 | 0.677 |
| medium-345M | 0.747 | 0.785 | 0.781 | 0.748 | 0.769 | 0.756 | 0.770 | 0.765 |
| small-117M | 0.733 | 0.784 | 0.779 | 0.733 | 0.761 | 0.736 | 0.762 | 0.763 |
| xl-1542M-k40 | 0.376 | 0.539 | 0.511 | 0.640 | 0.502 | 0.584 | 0.503 | 0.556 |
| large-762M-k40 | 0.384 | 0.579 | 0.567 | 0.675 | 0.535 | 0.637 | 0.536 | 0.597 |
| medium-345M-k40 | 0.391 | 0.597 | 0.567 | 0.691 | 0.556 | 0.637 | 0.557 | 0.615 |
| small-117M-k40 | 0.359 | 0.674 | 0.648 | 0.775 | 0.593 | 0.720 | 0.592 | 0.707 |

## Stability ladders (scribe ladder --profile v31 reports)

| run | pair | v2 | v3 | v3.1 | SURPRISE 3.1 | MATCH 3.1 |
|---|---|---|---|---|---|---|
| base | held-out human > sentences shuffled across docs | 0.975 | 0.981 | 0.965 | 0.935 | 0.940 |
| base | sentences shuffled across docs > words shuffled within sentences | 0.999 | 1.000 | 1.000 | 1.000 | 0.978 |
| base | words shuffled within sentences > random lexicon words | 0.998 | 0.975 | 0.996 | 0.962 | 0.865 |
| base | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| base | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack128 | held-out human > sentences shuffled across docs | 0.983 | 0.980 | 0.957 | 0.919 | 0.914 |
| pack128 | sentences shuffled across docs > words shuffled within sentences | 0.997 | 0.999 | 0.996 | 0.998 | 0.980 |
| pack128 | words shuffled within sentences > random lexicon words | 0.999 | 0.970 | 0.995 | 0.949 | 0.831 |
| pack128 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack128 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack16 | held-out human > sentences shuffled across docs | 0.972 | 0.976 | 0.956 | 0.922 | 0.925 |
| pack16 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 1.000 | 0.998 | 1.000 | 0.955 |
| pack16 | words shuffled within sentences > random lexicon words | 0.994 | 0.973 | 0.990 | 0.965 | 0.876 |
| pack16 | random lexicon words > random letters | 0.996 | 1.000 | 1.000 | 0.998 | 1.000 |
| pack16 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack32 | held-out human > sentences shuffled across docs | 0.971 | 0.976 | 0.951 | 0.912 | 0.917 |
| pack32 | sentences shuffled across docs > words shuffled within sentences | 0.996 | 1.000 | 0.998 | 1.000 | 0.978 |
| pack32 | words shuffled within sentences > random lexicon words | 0.996 | 0.972 | 0.992 | 0.958 | 0.880 |
| pack32 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 |
| pack32 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack64 | held-out human > sentences shuffled across docs | 0.981 | 0.978 | 0.958 | 0.923 | 0.918 |
| pack64 | sentences shuffled across docs > words shuffled within sentences | 0.997 | 1.000 | 0.996 | 1.000 | 0.983 |
| pack64 | words shuffled within sentences > random lexicon words | 0.999 | 0.967 | 0.994 | 0.951 | 0.873 |
| pack64 | random lexicon words > random letters | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack64 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pack8 | held-out human > sentences shuffled across docs | 0.976 | 0.980 | 0.961 | 0.927 | 0.920 |
| pack8 | sentences shuffled across docs > words shuffled within sentences | 0.998 | 0.999 | 0.998 | 1.000 | 0.902 |
| pack8 | words shuffled within sentences > random lexicon words | 0.993 | 0.962 | 0.991 | 0.972 | 0.868 |
| pack8 | random lexicon words > random letters | 0.996 | 1.000 | 1.000 | 0.998 | 1.000 |
| pack8 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref1000_11 | held-out human > sentences shuffled across docs | 0.973 | 0.981 | 0.963 | 0.929 | 0.936 |
| ref1000_11 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 |
| ref1000_11 | words shuffled within sentences > random lexicon words | 0.996 | 0.975 | 0.993 | 0.962 | 0.838 |
| ref1000_11 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref1000_11 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref1000_12 | held-out human > sentences shuffled across docs | 0.977 | 0.981 | 0.967 | 0.929 | 0.930 |
| ref1000_12 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 1.000 | 0.999 | 1.000 | 0.976 |
| ref1000_12 | words shuffled within sentences > random lexicon words | 0.998 | 0.973 | 0.995 | 0.961 | 0.862 |
| ref1000_12 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref1000_12 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref1000_13 | held-out human > sentences shuffled across docs | 0.973 | 0.980 | 0.964 | 0.929 | 0.940 |
| ref1000_13 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 |
| ref1000_13 | words shuffled within sentences > random lexicon words | 0.997 | 0.974 | 0.993 | 0.955 | 0.864 |
| ref1000_13 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref1000_13 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref100_11 | held-out human > sentences shuffled across docs | 0.966 | 0.973 | 0.950 | 0.891 | 0.908 |
| ref100_11 | sentences shuffled across docs > words shuffled within sentences | 0.990 | 0.993 | 0.990 | 1.000 | 0.913 |
| ref100_11 | words shuffled within sentences > random lexicon words | 0.999 | 0.969 | 0.983 | 0.952 | 0.816 |
| ref100_11 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 0.998 |
| ref100_11 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref100_12 | held-out human > sentences shuffled across docs | 0.971 | 0.966 | 0.944 | 0.846 | 0.907 |
| ref100_12 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 0.998 | 0.996 | 1.000 | 0.960 |
| ref100_12 | words shuffled within sentences > random lexicon words | 0.999 | 0.973 | 0.997 | 0.946 | 0.901 |
| ref100_12 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref100_12 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| ref100_13 | held-out human > sentences shuffled across docs | 0.963 | 0.974 | 0.939 | 0.836 | 0.892 |
| ref100_13 | sentences shuffled across docs > words shuffled within sentences | 0.990 | 0.996 | 0.990 | 1.000 | 0.966 |
| ref100_13 | words shuffled within sentences > random lexicon words | 0.991 | 0.954 | 0.966 | 0.889 | 0.881 |
| ref100_13 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 |
| ref100_13 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref200_11 | held-out human > sentences shuffled across docs | 0.972 | 0.976 | 0.957 | 0.916 | 0.927 |
| ref200_11 | sentences shuffled across docs > words shuffled within sentences | 0.999 | 0.999 | 1.000 | 1.000 | 0.969 |
| ref200_11 | words shuffled within sentences > random lexicon words | 0.998 | 0.972 | 0.990 | 0.958 | 0.827 |
| ref200_11 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref200_11 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref200_12 | held-out human > sentences shuffled across docs | 0.968 | 0.975 | 0.955 | 0.883 | 0.906 |
| ref200_12 | sentences shuffled across docs > words shuffled within sentences | 0.990 | 0.997 | 0.988 | 1.000 | 0.968 |
| ref200_12 | words shuffled within sentences > random lexicon words | 0.997 | 0.974 | 0.997 | 0.964 | 0.896 |
| ref200_12 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref200_12 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| ref200_13 | held-out human > sentences shuffled across docs | 0.965 | 0.972 | 0.949 | 0.875 | 0.921 |
| ref200_13 | sentences shuffled across docs > words shuffled within sentences | 0.999 | 1.000 | 1.000 | 1.000 | 0.971 |
| ref200_13 | words shuffled within sentences > random lexicon words | 0.994 | 0.968 | 0.989 | 0.934 | 0.886 |
| ref200_13 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| ref200_13 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref500_11 | held-out human > sentences shuffled across docs | 0.973 | 0.980 | 0.964 | 0.923 | 0.937 |
| ref500_11 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 0.999 | 1.000 | 1.000 | 0.974 |
| ref500_11 | words shuffled within sentences > random lexicon words | 0.995 | 0.974 | 0.992 | 0.960 | 0.841 |
| ref500_11 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref500_11 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref500_12 | held-out human > sentences shuffled across docs | 0.972 | 0.978 | 0.964 | 0.918 | 0.924 |
| ref500_12 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 0.999 | 0.992 | 1.000 | 0.972 |
| ref500_12 | words shuffled within sentences > random lexicon words | 0.997 | 0.974 | 0.996 | 0.962 | 0.875 |
| ref500_12 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref500_12 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref500_13 | held-out human > sentences shuffled across docs | 0.972 | 0.975 | 0.960 | 0.921 | 0.919 |
| ref500_13 | sentences shuffled across docs > words shuffled within sentences | 0.998 | 1.000 | 1.000 | 1.000 | 0.977 |
| ref500_13 | words shuffled within sentences > random lexicon words | 0.995 | 0.972 | 0.991 | 0.951 | 0.863 |
| ref500_13 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref500_13 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref50_11 | held-out human > sentences shuffled across docs | 0.885 | 0.928 | 0.882 | 0.863 | 0.882 |
| ref50_11 | sentences shuffled across docs > words shuffled within sentences | 0.996 | 0.997 | 0.996 | 1.000 | 0.952 |
| ref50_11 | words shuffled within sentences > random lexicon words | 0.978 | 0.969 | 0.969 | 0.938 | 0.814 |
| ref50_11 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 |
| ref50_11 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref50_12 | held-out human > sentences shuffled across docs | 0.941 | 0.947 | 0.920 | 0.781 | 0.853 |
| ref50_12 | sentences shuffled across docs > words shuffled within sentences | 0.986 | 0.996 | 0.986 | 1.000 | 0.926 |
| ref50_12 | words shuffled within sentences > random lexicon words | 0.999 | 0.964 | 0.990 | 0.916 | 0.904 |
| ref50_12 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| ref50_12 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ref50_13 | held-out human > sentences shuffled across docs | 0.923 | 0.951 | 0.876 | 0.800 | 0.869 |
| ref50_13 | sentences shuffled across docs > words shuffled within sentences | 0.981 | 0.994 | 0.987 | 0.999 | 0.943 |
| ref50_13 | words shuffled within sentences > random lexicon words | 0.989 | 0.951 | 0.934 | 0.851 | 0.766 |
| ref50_13 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.996 | 0.998 |
| ref50_13 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed1 | held-out human > sentences shuffled across docs | 0.976 | 0.981 | 0.971 | 0.934 | 0.948 |
| seed1 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 1.000 | 1.000 | 1.000 | 0.979 |
| seed1 | words shuffled within sentences > random lexicon words | 0.998 | 0.988 | 0.996 | 0.973 | 0.863 |
| seed1 | random lexicon words > random letters | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed1 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed2 | held-out human > sentences shuffled across docs | 0.990 | 0.989 | 0.983 | 0.933 | 0.957 |
| seed2 | sentences shuffled across docs > words shuffled within sentences | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 |
| seed2 | words shuffled within sentences > random lexicon words | 0.997 | 0.984 | 0.994 | 0.967 | 0.858 |
| seed2 | random lexicon words > random letters | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed2 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed3 | held-out human > sentences shuffled across docs | 0.970 | 0.969 | 0.956 | 0.931 | 0.940 |
| seed3 | sentences shuffled across docs > words shuffled within sentences | 0.996 | 1.000 | 0.998 | 1.000 | 0.971 |
| seed3 | words shuffled within sentences > random lexicon words | 0.995 | 0.973 | 0.991 | 0.957 | 0.836 |
| seed3 | random lexicon words > random letters | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed3 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed4 | held-out human > sentences shuffled across docs | 0.979 | 0.986 | 0.977 | 0.941 | 0.952 |
| seed4 | sentences shuffled across docs > words shuffled within sentences | 0.994 | 1.000 | 0.998 | 1.000 | 0.981 |
| seed4 | words shuffled within sentences > random lexicon words | 0.999 | 0.982 | 0.994 | 0.971 | 0.877 |
| seed4 | random lexicon words > random letters | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 |
| seed4 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed5 | held-out human > sentences shuffled across docs | 0.969 | 0.967 | 0.958 | 0.923 | 0.929 |
| seed5 | sentences shuffled across docs > words shuffled within sentences | 0.997 | 1.000 | 0.994 | 1.000 | 0.968 |
| seed5 | words shuffled within sentences > random lexicon words | 0.998 | 0.975 | 0.993 | 0.963 | 0.836 |
| seed5 | random lexicon words > random letters | 0.997 | 1.000 | 1.000 | 0.996 | 1.000 |
| seed5 | random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## K5 (GEN-A dev ladder), trie rung

| read | n | v2 | SURPRISE | MATCH | v3 | min3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 | disagree |
|---|---|---|---|---|---|---|---|---|---|---|
| C1_s20261101 | 247 | 0.479 [0.449, 0.506] | 0.589 [0.568, 0.611] | 0.571 [0.555, 0.588] | 0.546 [0.529, 0.562] | 0.401 [0.382, 0.420] | 0.467 [0.437, 0.495] | 0.459 [0.444, 0.473] | 0.442 [0.425, 0.461] | 4% |
| C1_s20262101 | 247 | 0.462 [0.438, 0.487] | 0.587 [0.565, 0.608] | 0.558 [0.543, 0.575] | 0.536 [0.521, 0.551] | 0.398 [0.380, 0.418] | 0.476 [0.448, 0.504] | 0.449 [0.435, 0.462] | 0.438 [0.422, 0.455] | 2% |
| C1_s20263101 | 246 | 0.502 [0.475, 0.531] | 0.606 [0.586, 0.627] | 0.562 [0.545, 0.579] | 0.557 [0.541, 0.575] | 0.419 [0.400, 0.438] | 0.497 [0.471, 0.523] | 0.450 [0.436, 0.465] | 0.460 [0.442, 0.476] | 5% |
| C1_s20264101 | 249 | 0.473 [0.449, 0.499] | 0.602 [0.580, 0.624] | 0.569 [0.552, 0.585] | 0.548 [0.533, 0.563] | 0.407 [0.388, 0.425] | 0.493 [0.466, 0.521] | 0.456 [0.441, 0.470] | 0.450 [0.435, 0.467] | 6% |
| E8_s20261101 | 242 | 0.353 [0.323, 0.386] | 0.718 [0.699, 0.736] | 0.592 [0.579, 0.607] | 0.554 [0.537, 0.573] | 0.327 [0.300, 0.355] | 0.644 [0.629, 0.658] | 0.456 [0.444, 0.469] | 0.435 [0.415, 0.455] | 26% |
| E8_s20262101 | 238 | 0.362 [0.329, 0.398] | 0.705 [0.688, 0.722] | 0.574 [0.560, 0.589] | 0.547 [0.530, 0.566] | 0.335 [0.308, 0.365] | 0.633 [0.618, 0.647] | 0.441 [0.429, 0.453] | 0.434 [0.412, 0.453] | 28% |
| E8_s20263101 | 241 | 0.337 [0.307, 0.368] | 0.697 [0.679, 0.716] | 0.571 [0.556, 0.587] | 0.535 [0.518, 0.552] | 0.319 [0.292, 0.343] | 0.636 [0.622, 0.649] | 0.436 [0.424, 0.447] | 0.424 [0.403, 0.444] | 25% |
| E8_s20264101 | 232 | 0.323 [0.291, 0.354] | 0.714 [0.697, 0.733] | 0.587 [0.571, 0.602] | 0.541 [0.525, 0.557] | 0.300 [0.275, 0.328] | 0.639 [0.625, 0.653] | 0.452 [0.439, 0.464] | 0.423 [0.405, 0.442] | 33% |
| N100_s20261101 | 232 | 0.363 [0.339, 0.389] | 0.673 [0.653, 0.693] | 0.563 [0.549, 0.581] | 0.533 [0.516, 0.550] | 0.343 [0.321, 0.366] | 0.589 [0.570, 0.608] | 0.432 [0.419, 0.446] | 0.435 [0.419, 0.452] | 13% |
| N300_s20261101 | 242 | 0.382 [0.354, 0.411] | 0.675 [0.656, 0.696] | 0.566 [0.552, 0.581] | 0.541 [0.524, 0.559] | 0.356 [0.333, 0.380] | 0.591 [0.575, 0.610] | 0.434 [0.423, 0.446] | 0.443 [0.428, 0.458] | 10% |
| N30_s20261101 | 238 | 0.375 [0.345, 0.403] | 0.685 [0.662, 0.706] | 0.575 [0.558, 0.593] | 0.545 [0.527, 0.562] | 0.349 [0.326, 0.373] | 0.612 [0.593, 0.630] | 0.444 [0.430, 0.460] | 0.447 [0.428, 0.464] | 15% |
| N7_s20261101 | 241 | 0.350 [0.319, 0.382] | 0.698 [0.682, 0.715] | 0.577 [0.564, 0.589] | 0.542 [0.527, 0.557] | 0.321 [0.297, 0.345] | 0.632 [0.619, 0.645] | 0.442 [0.431, 0.452] | 0.431 [0.412, 0.448] | 28% |
| NALL_s20261101 | 245 | 0.412 [0.388, 0.437] | 0.688 [0.667, 0.708] | 0.573 [0.558, 0.588] | 0.558 [0.541, 0.575] | 0.387 [0.363, 0.407] | 0.596 [0.575, 0.614] | 0.436 [0.425, 0.448] | 0.461 [0.445, 0.477] | 7% |
| NALL_s20262101 | 242 | 0.389 [0.361, 0.416] | 0.674 [0.653, 0.693] | 0.574 [0.559, 0.589] | 0.545 [0.529, 0.563] | 0.366 [0.343, 0.389] | 0.588 [0.571, 0.607] | 0.442 [0.429, 0.454] | 0.449 [0.432, 0.467] | 9% |
| NALL_s20263101 | 236 | 0.378 [0.352, 0.406] | 0.671 [0.652, 0.688] | 0.562 [0.549, 0.576] | 0.537 [0.520, 0.553] | 0.355 [0.334, 0.377] | 0.590 [0.574, 0.606] | 0.436 [0.422, 0.450] | 0.445 [0.430, 0.461] | 8% |
| NALL_s20264101 | 242 | 0.381 [0.354, 0.409] | 0.668 [0.649, 0.689] | 0.566 [0.553, 0.581] | 0.539 [0.523, 0.555] | 0.350 [0.330, 0.370] | 0.576 [0.556, 0.595] | 0.437 [0.424, 0.448] | 0.440 [0.425, 0.456] | 9% |
| U8_s20261101 | 244 | 0.498 [0.471, 0.527] | 0.612 [0.591, 0.634] | 0.583 [0.566, 0.599] | 0.565 [0.549, 0.580] | 0.419 [0.398, 0.442] | 0.554 [0.527, 0.578] | 0.464 [0.451, 0.478] | 0.483 [0.466, 0.501] | 7% |
| U8_s20262101 | 246 | 0.496 [0.470, 0.523] | 0.590 [0.566, 0.614] | 0.575 [0.558, 0.593] | 0.554 [0.537, 0.571] | 0.407 [0.386, 0.427] | 0.529 [0.500, 0.555] | 0.460 [0.444, 0.475] | 0.470 [0.451, 0.488] | 9% |
| U8_s20263101 | 249 | 0.501 [0.469, 0.531] | 0.582 [0.560, 0.605] | 0.578 [0.561, 0.594] | 0.554 [0.537, 0.573] | 0.413 [0.392, 0.434] | 0.527 [0.501, 0.553] | 0.460 [0.446, 0.474] | 0.474 [0.457, 0.491] | 8% |
| U8_s20264101 | 247 | 0.505 [0.477, 0.534] | 0.595 [0.572, 0.619] | 0.575 [0.558, 0.591] | 0.559 [0.543, 0.576] | 0.414 [0.395, 0.434] | 0.532 [0.506, 0.558] | 0.459 [0.444, 0.475] | 0.476 [0.459, 0.492] | 9% |

## K5: mean over the saved seeds

| arm | seeds | v2 | SURPRISE | MATCH | v3 | min3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 4 | 0.479 | 0.596 | 0.565 | 0.547 | 0.406 | 0.483 | 0.454 | 0.448 |
| E8 | 4 | 0.344 | 0.709 | 0.581 | 0.544 | 0.320 | 0.638 | 0.446 | 0.429 |
| N100 | 1 | 0.363 | 0.673 | 0.563 | 0.533 | 0.343 | 0.589 | 0.432 | 0.435 |
| N300 | 1 | 0.382 | 0.675 | 0.566 | 0.541 | 0.356 | 0.591 | 0.434 | 0.443 |
| N30 | 1 | 0.375 | 0.685 | 0.575 | 0.545 | 0.349 | 0.612 | 0.444 | 0.447 |
| N7 | 1 | 0.350 | 0.698 | 0.577 | 0.542 | 0.321 | 0.632 | 0.442 | 0.431 |
| NALL | 4 | 0.390 | 0.675 | 0.569 | 0.545 | 0.365 | 0.588 | 0.438 | 0.449 |
| U8 | 4 | 0.500 | 0.595 | 0.578 | 0.558 | 0.413 | 0.536 | 0.461 | 0.476 |

## Cost

| run | texts/s | v3 ms a text | memory and time |
|---|---|---|---|
| web.v2 | 1212.5 | 0.0 | wall_s=27.45 user_s=26.46 sys_s=0.66 maxrss_mb=648.9 rc=0 |
| web.v3 | 463.9 | 1.1771 | wall_s=46.19 user_s=61.44 sys_s=0.73 maxrss_mb=749.2 rc=0 |
| web.v31 | 399.6 | 1.4995 | wall_s=51.10 user_s=70.33 sys_s=0.65 maxrss_mb=768.1 rc=0 |
| wiki.v2 | 1748.5 | 0.0 | wall_s=8.92 user_s=8.47 sys_s=0.36 maxrss_mb=385.0 rc=0 |
| wiki.v3 | 730.3 | 0.7018 | wall_s=13.95 user_s=17.47 sys_s=0.44 maxrss_mb=481.2 rc=0 |
| wiki.v31 | 587.1 | 0.9636 | wall_s=15.41 user_s=19.39 sys_s=0.45 maxrss_mb=488.9 rc=0 |
