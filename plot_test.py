import textwrap

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _pretty_label(label, width=18):
    label = str(label).replace("_", " ")
    return "\n".join(textwrap.wrap(label, width=width)) or label


def plot_filter_model_variables(
    corr_mat_list: list,
    norm_contribution_df: pd.DataFrame,
    title=None,
    figsize=None,
    cmap="vlag",
    show_subject_points=True,
    label_width=18,
):
    """Plot model-variable correlations and normalized contributions.

    This is a prettier drop-in replacement for
    lecilab_behavior_analysis.plots.plot_filter_model_variables.
    It accepts the same two core inputs but returns the matplotlib figure.
    """
    if not corr_mat_list:
        raise ValueError("corr_mat_list is empty")
    if norm_contribution_df.empty:
        raise ValueError("norm_contribution_df is empty")

    variables = list(corr_mat_list[0].index)
    corr_stack = np.stack(
        [
            corr_mat.reindex(index=variables, columns=variables).to_numpy()
            for corr_mat in corr_mat_list
        ]
    )
    corr_mean = np.nanmean(corr_stack, axis=0)
    corr_mean_df = pd.DataFrame(corr_mean, index=variables, columns=variables)

    contrib_mean = norm_contribution_df.mean(axis=1).sort_values()
    contrib_sem = norm_contribution_df.sem(axis=1).reindex(contrib_mean.index)

    n_vars = len(variables)
    if figsize is None:
        figsize = (
            max(8.0, 0.75 * n_vars + 4.5),
            max(8.5, 0.52 * n_vars + 6.0),
        )

    fig = plt.figure(figsize=figsize, dpi=160)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.15, 0.95],
        hspace=0.42,
    )
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_contrib = fig.add_subplot(gs[1, 0])

    mask = np.triu(np.ones_like(corr_mean_df, dtype=bool), k=1)
    tick_labels = [_pretty_label(var, width=label_width) for var in variables]
    sns.heatmap(
        corr_mean_df,
        mask=mask,
        ax=ax_corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Mean r", "shrink": 0.75},
    )
    ax_corr.set_title("Mean Predictor Correlation", fontsize=12, pad=10)
    ax_corr.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax_corr.set_yticklabels(tick_labels, rotation=0)
    ax_corr.tick_params(axis="both", length=0, labelsize=8)

    y_pos = np.arange(len(contrib_mean))
    bar_colors = sns.color_palette("crest", len(contrib_mean))
    ax_contrib.barh(
        y_pos,
        contrib_mean.values,
        xerr=contrib_sem.values if norm_contribution_df.shape[1] > 1 else None,
        color=bar_colors,
        edgecolor="0.25",
        linewidth=0.4,
        alpha=0.9,
        error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "0.25"},
    )

    if show_subject_points:
        rng = np.random.default_rng(0)
        for y_idx, var in enumerate(contrib_mean.index):
            values = pd.to_numeric(
                norm_contribution_df.loc[var],
                errors="coerce",
            ).dropna()
            jitter = rng.normal(0, 0.055, size=len(values))
            ax_contrib.scatter(
                values,
                y_idx + jitter,
                s=18,
                color="black",
                alpha=0.45,
                linewidth=0,
                zorder=3,
            )

    ax_contrib.set_yticks(y_pos)
    ax_contrib.set_yticklabels(
        [_pretty_label(var, width=label_width) for var in contrib_mean.index],
        fontsize=9,
    )
    ax_contrib.set_xlabel("Normalized contribution", fontsize=10)
    ax_contrib.set_title("Drop-One Variable Contribution", fontsize=12, pad=10)
    ax_contrib.grid(axis="x", color="0.88", linewidth=0.8)
    ax_contrib.set_axisbelow(True)
    ax_contrib.spines[["top", "right", "left"]].set_visible(False)
    ax_contrib.tick_params(axis="y", length=0)

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    fig.tight_layout()
    return fig


def plot_traj_speed(df_bp, cmap, ax, norm):
    """
    df_bp: DataFrame with columns ['x', 'y', 'mean_speed'] (no NaNs).
    Plot trajectory as colored line segments based on speed.
    """

    x = df_bp['x'].to_numpy()
    y = df_bp['y'].to_numpy()
    sp = df_bp['mean_speed'].to_numpy()

    # Line segments with shape (N-1, 2, 2)
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    # Speed per segment (using the average of start and end speeds)
    sp_seg = (sp[:-1] + sp[1:]) / 2.0

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(sp_seg)
    lc.set_linewidth(7)
    lc.set_capstyle('round')
    lc.set_joinstyle('round')
    ax.add_collection(lc)

    # # Mark start and end points
    # ax.scatter(x[0], y[0], color='k', s=200, marker='o',
    #            edgecolors='k', zorder=3)
    # ax.scatter(x[-1], y[-1], color='w', s=200, marker='o',
    #            edgecolors='k', zorder=3)
