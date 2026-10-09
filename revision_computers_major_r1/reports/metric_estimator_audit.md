# Metric and estimator audit

## Seven-point self-anchor identity

For one complete self-anchored trajectory with the seven equally weighted targets

\[
\alpha\in\{0,1/6,2/6,3/6,4/6,5/6,1\},
\]

the generated endpoint used as anchor A has coordinate 0 and the generated endpoint used as anchor B has coordinate 1 (unless the endpoint embeddings collapse). Their two absolute coordinate errors are therefore mechanically zero. If the five internal absolute errors are \(e_1,\ldots,e_5\),

\[
\operatorname{MAE}_7=\frac{e_1+\cdots+e_5}{7}
=\frac{5}{7}\operatorname{MAE}_{internal}.
\]

Thus the relative reduction from including the two mechanical zero-error endpoints is

\[
1-5/7=2/7=0.285714\ldots.
\]

This is an algebraic property of that estimator design, not an empirical result and not a contribution. If the internal MAE is zero, the ratio of the two MAEs is not a meaningful relative-error statistic even though both values are zero.

The identity does **not** automatically apply to external-reference anchors, pooled anchors, unequal weighting, missing points, or endpoints that are not the same outputs used to construct the coordinate. Linear means of endpoint-zero quantities can show the same weighting mechanism; RMS and other nonlinear aggregations do not generally have a 5/7 factor.

Executable tests are in `revision_computers_major_r1/tests/test_metric_audit.py`. They check the exact 5/7 mean identity, the undefined zero-error relative ratio case, and that RMS does not obey the same factor. The full revision test suite also checks collapsed fixed anchors are recorded as NaN rather than 0.5.

## Metric definitions used in this revision

For fixed anchors \(z_A,z_B\), let \(\Delta=z_B-z_A\) and \(L=\lVert\Delta\rVert\):

- coordinate: \(c=(z-z_A)^T\Delta/(\Delta^T\Delta)\);
- ICE: mean internal \(|c-\alpha|\);
- drift: \(\lVert z-z_A-c\Delta\rVert/L\);
- interpolation distance: \(\lVert z-[(1-\alpha)z_A+\alpha z_B]\rVert/L\);
- out of range: \(c<0\) or \(c>1\).

Raw coordinate is primary. Clipping to [0,1] is saved only as a sensitivity field. The implementation retains the legacy numerical convention of declaring an anchor pair unstable at \(L\le10^{-12}\) and dividing normalized distances by \(L+10^{-12}\). It additionally saves the exact-\(L\) values. On the current six-model main data, the maximum absolute legacy-vs-exact difference is below `4e-10` for all three encoders (see `manifests/geometry_*.json`).

Nearest-target rank retains the old meaning: for each ideal target point, rank the output requested at that ratio among all seven generated outputs in the same trajectory, using 1-based midranks, then normalize as `(rank-1)/6`. Primary summaries average only the five internal targets. It is not redefined as ranking several ideal points from one output.
