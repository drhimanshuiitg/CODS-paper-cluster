"""Small shared plotting helpers for the results-aggregation scripts.

Not part of the sleep_quadnet experiment package -- these are pure
presentation utilities with no effect on any computed metric.
"""

from __future__ import annotations


def annotate_scatter(axis, xs, ys, labels, fontsize=8, x_cluster_frac=0.06, y_gap_frac=0.06):
    """Label scatter points without overlapping, for a handful of points.

    matplotlib's plain axis.annotate(xytext=(4, 3), textcoords="offset points")
    stacks every label at the same fixed offset, so points that share a
    similar x value (e.g. several PCA dimensions, or several representations
    with near-identical latency/feature-dimension) get illegible overlapping
    labels -- an observed defect in this project's own
    performance_vs_feature_dimension.pdf and performance_vs_latency.pdf.

    This buckets points by x-proximity (within x_cluster_frac of the overall
    x-range), and within each bucket stacks labels top-to-bottom in DATA
    coordinates (matching the axis's own ylim, so the enforced gap is exact
    regardless of figure size/DPI), starting from each point's own y and
    only pushing a label further down when it would otherwise land within
    y_gap_frac of the label above it. A thin leader line is drawn whenever a
    label ends up displaced from its point.
    """
    xs = list(xs)
    ys = list(ys)
    labels = list(labels)
    axis.scatter(xs, ys)
    if not xs:
        return
    x_range = (max(xs) - min(xs)) or 1.0
    y0, y1 = axis.get_ylim()
    min_gap = y_gap_frac * (y1 - y0)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    buckets = [[order[0]]]
    for i in order[1:]:
        if xs[i] - xs[buckets[-1][-1]] > x_cluster_frac * x_range:
            buckets.append([i])
        else:
            buckets[-1].append(i)
    for bucket in buckets:
        prev_label_y = None
        for i in sorted(bucket, key=lambda idx: -ys[idx]):  # highest point first
            label_y = ys[i] + 0.012 * (y1 - y0)
            if prev_label_y is not None and prev_label_y - label_y < min_gap:
                label_y = prev_label_y - min_gap
            prev_label_y = label_y
            displaced = abs(label_y - ys[i]) > 0.008 * (y1 - y0)
            axis.annotate(
                labels[i], xy=(xs[i], ys[i]), xytext=(xs[i] + 0.012 * x_range, label_y),
                textcoords="data", fontsize=fontsize, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.6, "shrinkA": 0, "shrinkB": 2} if displaced else None,
            )
