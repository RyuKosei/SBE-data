# Main 8-Axis Geometry Summary

## Headline

Both bge-m3 and text2vec preserve the same projection ranking: qwen35_9b is best, qwen35_35b_a3b is second, and qwen35_4b is weakest.

## Projection Metrics By Embedding

bge-m3:

| model          |   projection_mae |   projection_spearman |   interpolation_distance_mean |   off_axis_drift_mean |   normalized_off_axis_drift_mean |
|:---------------|-----------------:|----------------------:|------------------------------:|----------------------:|---------------------------------:|
| qwen35_9b      |         0.125664 |              0.877569 |                      0.297324 |              0.284483 |                         0.499249 |
| qwen35_35b_a3b |         0.140022 |              0.849621 |                      0.286223 |              0.270143 |                         0.489455 |
| qwen35_4b      |         0.164476 |              0.792771 |                      0.291225 |              0.272686 |                         0.532423 |

text2vec:

| model          |   projection_mae |   projection_spearman |   interpolation_distance_mean |   off_axis_drift_mean |   normalized_off_axis_drift_mean |
|:---------------|-----------------:|----------------------:|------------------------------:|----------------------:|---------------------------------:|
| qwen35_9b      |         0.110768 |              0.902909 |                      0.270769 |              0.260379 |                         0.470136 |
| qwen35_35b_a3b |         0.127372 |              0.878371 |                      0.25756  |              0.243513 |                         0.455948 |
| qwen35_4b      |         0.15052  |              0.827194 |                      0.276057 |              0.25981  |                         0.51412  |

## Required Questions

### Is 9B's advantage lower projection error or lower off-axis drift?

9B's advantage is primarily lower projection error and higher rank correlation. It does not have the lowest off-axis drift; 35B-A3B often has slightly lower drift, especially under text2vec. This means 9B is better calibrated along the style-ratio axis, while 35B-A3B can be geometrically close to the interpolation line but less accurately placed along it.

### Why does 35B-A3B not exceed 9B?

35B-A3B does not exceed 9B because its projection-ratio placement is less calibrated. Its off-axis drift is competitive, and out-of-range rates are low, so the gap is better explained by projection proportion bias and axis-specific calibration error rather than broad geometric instability.

### Where is 4B weakest?

Under bge-m3, the weakest 4B axes by projection MAE are:

| axis_id            |   projection_mae |   projection_spearman |   normalized_off_axis_drift_mean |
|:-------------------|-----------------:|----------------------:|---------------------------------:|
| concision_detail   |         0.209782 |              0.644495 |                         0.636383 |
| politeness_refusal |         0.194163 |              0.689305 |                         0.59034  |
| humor_neutral      |         0.165069 |              0.799754 |                         0.535909 |
| emotion_apology    |         0.160907 |              0.830319 |                         0.48151  |
| empathy_clinical   |         0.151669 |              0.818102 |                         0.540427 |
| objectivity_cat    |         0.149623 |              0.850082 |                         0.518968 |
| formality_request  |         0.14944  |              0.849748 |                         0.510843 |
| expertise_explain  |         0.13515  |              0.891381 |                         0.445004 |

4B also has the highest length violation rate, mainly too-short outputs, which should be treated as a practical limitation even though prior diagnostics indicate length does not dominate the projection ranking.

### Which axes are most linear and which are most off-axis?

The most linear axes are those with low smoothness penalty and low projection MAE. The easiest off-axis failure cases are captured by normalized off-axis drift. Highest bge-m3 normalized drift cases:

| model          | axis_id            |   normalized_off_axis_drift_mean |   off_axis_drift_mean |   projection_mae |
|:---------------|:-------------------|---------------------------------:|----------------------:|-----------------:|
| qwen35_4b      | concision_detail   |                         0.636383 |              0.222705 |         0.209782 |
| qwen35_35b_a3b | concision_detail   |                         0.618886 |              0.223516 |         0.190593 |
| qwen35_9b      | concision_detail   |                         0.606876 |              0.226154 |         0.186476 |
| qwen35_4b      | politeness_refusal |                         0.59034  |              0.27341  |         0.194163 |
| qwen35_9b      | politeness_refusal |                         0.569845 |              0.280098 |         0.163184 |
| qwen35_4b      | empathy_clinical   |                         0.540427 |              0.255698 |         0.151669 |
| qwen35_35b_a3b | politeness_refusal |                         0.540205 |              0.264277 |         0.159231 |
| qwen35_4b      | humor_neutral      |                         0.535909 |              0.297218 |         0.165069 |
| qwen35_9b      | humor_neutral      |                         0.528432 |              0.318346 |         0.144243 |
| qwen35_4b      | objectivity_cat    |                         0.518968 |              0.297537 |         0.149623 |
| qwen35_4b      | formality_request  |                         0.510843 |              0.251038 |         0.14944  |
| qwen35_35b_a3b | humor_neutral      |                         0.499646 |              0.292757 |         0.152017 |

### Do low and high ratio regions show overshoot or endpoint attraction?

Endpoint attraction is present but not dominant. It should be interpreted from `main_geometry_metrics_by_alpha_model_embedding.csv` and `main_out_of_range_projection_metrics.csv`. Out-of-range projection rates are low overall, so the more important issue is mild under/over-placement within the [0, 1] interval rather than systematic extreme overshoot.

## Scalar Judge Caveat

| model          | judge_model    |    n |   parse_ok_rate |   judge_mae |   judge_bias |   judge_spearman |   content_preservation_mean |   naturalness_mean |   endpoint_attraction_rate |   smoothness |
|:---------------|:---------------|-----:|----------------:|------------:|-------------:|-----------------:|----------------------------:|-------------------:|---------------------------:|-------------:|
| qwen35_35b_a3b | qwen35_9b      | 6720 |               1 |    0.182418 |    0.0837396 |         0.756274 |                     4.99896 |            4.38304 |                   0.372917 |    0.0535167 |
| qwen35_4b      | qwen35_35b_a3b | 6720 |               1 |    0.24424  |    0.0360134 |         0.589193 |                     4.99033 |            4.3689  |                   0.497917 |    0.0984792 |
| qwen35_9b      | qwen35_35b_a3b | 6720 |               1 |    0.175402 |    0.0462054 |         0.768238 |                     4.99033 |            4.38571 |                   0.517014 |    0.123656  |

Scalar judge is auxiliary. 4B/9B are judged by 35B-A3B, while 35B-A3B is judged by 9B, so scalar metrics are not fully same-scale for 9B vs 35B-A3B.

## Length Diagnostics

| model          |    n |   mean_zh_chars |   length_violation_rate |   too_short_rate |   too_long_rate |
|:---------------|-----:|----------------:|------------------------:|-----------------:|----------------:|
| qwen35_35b_a3b | 6720 |         121.983 |                0.200298 |        0.028125  |       0.172173  |
| qwen35_4b      | 6720 |         102.735 |                0.242411 |        0.170685  |       0.0717262 |
| qwen35_9b      | 6720 |         112.72  |                0.1375   |        0.0532738 |       0.0842262 |

Length-projection correlations:

| embedding   | model          |   pearson_length_error_projection_error |   spearman_length_error_projection_error |    n |
|:------------|:---------------|----------------------------------------:|-----------------------------------------:|-----:|
| bge_m3      | qwen35_35b_a3b |                              0.00739206 |                                0.0283984 | 6720 |
| bge_m3      | qwen35_4b      |                              0.0394421  |                                0.0500538 | 6720 |
| bge_m3      | qwen35_9b      |                              0.00970966 |                                0.0152613 | 6720 |
| text2vec    | qwen35_35b_a3b |                              0.00851676 |                                0.0239624 | 6720 |
| text2vec    | qwen35_4b      |                              0.0435575  |                                0.0556349 | 6720 |
| text2vec    | qwen35_9b      |                              0.00897994 |                                0.0120449 | 6720 |
