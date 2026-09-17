# Metric Definitions

Version: 1 (frozen before the new Qwen-family rankings)

For a generated trajectory with embedding vectors `z_alpha`, its self-anchor
axis is `delta = z_B - z_A`, where `z_A = z_0` and `z_B = z_1`. A shared-anchor
analysis replaces these two endpoints by the same independently reviewed pair
for every tested model. `epsilon = 1e-12` is used only to protect normalized
denominators. A trajectory with endpoint separation at or below epsilon is
flagged unstable and is not silently assigned coordinate 0.5.

## Primary five-interior-point metrics

All primary trajectory means use only `alpha` in `{1/6, 2/6, 3/6, 4/6,
5/6}`. Endpoints are retained to define the axis but never enter these means.

- Raw projection coordinate: `c_alpha = ((z_alpha-z_A) dot delta) /
  ||delta||^2`.
- Interior Calibration Error (ICE): mean `|c_alpha-alpha|`, without clipping.
- Clipped projection error: the same calculation after clipping `c_alpha` to
  `[0,1]`; this is an ablation, not the primary ICE.
- Spearman monotonicity: Spearman correlation between the five target ratios
  and their raw projection coordinates.
- Normalized off-axis drift: mean
  `||(z_alpha-z_A)-c_alpha*delta|| / (||delta||+epsilon)`.
- Normalized interpolation distance: distance to
  `z_A + alpha*delta`, divided by endpoint separation. Both its arithmetic mean
  and root-mean-square are reported.
- Out-of-range rate: fraction for which raw `c_alpha < 0` or `c_alpha > 1`,
  evaluated before any clipping.
- Endpoint separation: `||delta||`.

The projection decomposition identity is checked for every row. The maximum
relative numerical residual must be below `1e-6` before full evaluation.

## Nearest-neighbor rank

For each ideal linear target position, the candidate set is the seven generated
outputs from the same scenario, model, seed, prompt, anchor mode, and encoder.
Candidates are ranked by Euclidean distance to that target. Exact ties receive
their average (mid-)rank. Rank is 1-based; normalized rank is `(rank-1)/(K-1)`
with `K=7`. Primary mean rank again excludes endpoint targets.

## Trajectory diagnostics

- PCA PC1 explained variance is calculated after centering all seven points.
- Endpoint-to-PC1 alignment is the absolute cosine (and corresponding acute
  angle) between `delta` and PC1.
- Ordered Path Length Ratio is the sum of the six consecutive segment lengths
  divided by endpoint separation.
- Discrete curvature is the angle between each pair of adjacent path segments;
  mean and maximum angles are reported.
- Local projection slopes are consecutive changes in raw coordinate divided by
  consecutive changes in target ratio. Negative slopes and sign-changing
  turning points are counted.

Leave-one-seed-out diagnostics form an empirical center at every target ratio
from two seeds and test the third, rotating over all three seeds. They compare
the training-endpoint line, exact per-ratio empirical centers, and the piecewise
linear empirical path. For each reference, three separate quantities are
reported: (i) along-trajectory ratio error, (ii) nearest-ratio identification
accuracy, and (iii) off-trajectory distance normalized by the training-center
endpoint separation. The endpoint-line and true-centroid normalized Euclidean
target distances are retained as additional diagnostics, but are not compared
as though they had the same units as ratio error.

## Baselines

- Constant 0.5 predicts 0.5 for every interior point. Its theoretical MAE is
  0.260 on the legacy irregular grid `{0.1,0.25,0.5,0.75,0.9}` and exactly
  0.200 on the revision sixth grid. The two values must not be interchanged.
- Endpoint-distance ratio is `d(z,A)/(d(z,A)+d(z,B))`, with 0.5 used only when
  both distances collapse numerically.
- Length-only uses the endpoint-normalized character-length coordinate as its
  only feature and fits an intercept plus linear term using deterministic,
  axis-stratified, scenario-grouped five-fold cross-validation.
- Isotonic calibration maps raw projection to target ratio using increasing
  PAVA regression and the same scenario-grouped five-fold splits.

## Statistical unit

Seeds are first aggregated within `scenario × target ratio × model` cells.
Inference and 10,000-replicate bootstrap intervals use scenario as the
independent unit while preserving style axis as a stratum. Paired comparisons
use the same scenarios and apply Holm correction to families of tests.

The prescribed three seeds are retained for main generation, leave-one-seed-out
trajectory construction, and the prespecified prompt/temperature experiment.
No additional seed sweep is applied to encoders, deterministic baselines,
shared-anchor projection, content checking, or downstream statistical models.

## Content preservation

The fixed automatic evaluator judges every atomic content checkpoint, factual
contradiction, newly added key information, whether an addition changes the task
result, naturalness, and scalar style-B intensity. Content coverage is the
fraction of atomic checkpoints marked covered. `content-preservation pass` is
true exactly when coverage is at least 0.90, no factual contradiction is found,
no addition changes the task result, and the item is not marked unjudgeable.
The evaluator is fixed by `config/evaluator.yaml` and its validation against
human labels remains external pending until genuine annotations are returned.
