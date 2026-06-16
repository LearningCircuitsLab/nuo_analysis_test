import textwrap
from pathlib import Path
import os
import re

import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
from scipy.special import expit
from statsmodels.tsa.stattools import acf

import behavior_utils


DEFAULT_SVG_SAVE_DIR = r"E:\data\LeciLab\behavioral_data\tmp"


def _resolve_save_dir(save_dir=DEFAULT_SVG_SAVE_DIR):
    save_dir = str(save_dir)
    windows_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", save_dir)
    if os.name != "nt" and windows_drive:
        drive = windows_drive.group(1).lower()
        rest = windows_drive.group(2).replace("\\", "/")
        return Path("/mnt") / drive / rest
    return Path(save_dir)


def _slug(text):
    text = str(text)
    text = re.sub(r"[^\w.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "figure"


def _figure_title_parts(fig):
    title_parts = []

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        title_parts.append(suptitle.get_text())

    for ax in fig.axes:
        for text in [ax.get_title(), ax.get_ylabel()]:
            if text and not text.startswith("State t"):
                title_parts.append(text)

    clean_parts = []
    seen = set()
    for title in title_parts:
        title = str(title).strip()
        if title and title not in seen:
            clean_parts.append(title)
            seen.add(title)

    return clean_parts


def figure_title_filename(fig, fallback="figure", max_len=180):
    title_parts = _figure_title_parts(fig)
    filename = "_".join(title_parts) if title_parts else str(fallback)
    filename = _slug(filename)
    if len(filename) > max_len:
        filename = filename[:max_len].rstrip("_.-")
    return filename or _slug(fallback)


def _unique_filename(filename, used_names):
    filename = _slug(filename)
    stem = filename[:-4] if filename.lower().endswith(".svg") else filename
    candidate = stem
    counter = 2
    while candidate in used_names:
        candidate = f"{stem}_{counter:02d}"
        counter += 1
    used_names.add(candidate)
    return f"{candidate}.svg"


def save_figure_svg(
    fig,
    filename,
    save_dir=DEFAULT_SVG_SAVE_DIR,
    enabled=True,
    subfolder=None,
):
    if not enabled:
        return None

    save_path = _resolve_save_dir(save_dir)
    if subfolder is not None:
        save_path = save_path / _slug(subfolder)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = _slug(filename)
    if not filename.lower().endswith(".svg"):
        filename = f"{filename}.svg"
    out_path = save_path / filename
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi = 150)
    return out_path


def save_figures_svg(
    figures,
    prefix,
    save_dir=DEFAULT_SVG_SAVE_DIR,
    enabled=True,
    use_titles=True,
):
    if not enabled:
        return []

    saved_paths = []
    batch_dir = _resolve_save_dir(save_dir) / _slug(prefix)
    used_names = set()

    if isinstance(figures, dict):
        iterable = figures.items()
    else:
        iterable = enumerate(figures)

    for key, fig in iterable:
        if use_titles:
            filename = figure_title_filename(fig, fallback=f"{prefix}_{key}")
        else:
            filename = f"{prefix}_{key}"
        filename = _unique_filename(filename, used_names)
        saved_path = save_figure_svg(
            fig,
            filename,
            save_dir=batch_dir,
            enabled=enabled,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)

    return saved_paths


def p_to_star(p_value):
    if p_value is None or not np.isfinite(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_paired_significance_label(
    ax,
    values_left,
    values_right,
    label,
    x0=0,
    x1=1,
):
    values = pd.to_numeric(
        pd.concat(
            [
                pd.Series(values_left),
                pd.Series(values_right),
            ],
            ignore_index=True,
        ),
        errors="coerce",
    ).dropna()
    if values.empty:
        y = 1
        h = 0.05
    else:
        y_min = values.min()
        y_max = values.max()
        y_range = y_max - y_min
        if not np.isfinite(y_range) or y_range == 0:
            y_range = max(abs(y_max), 1) * 0.1
        y = y_max + y_range * 0.08
        h = y_range * 0.04

    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="black", linewidth=1)
    ax.text(
        (x0 + x1) / 2,
        y + h,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
    )
    ax.set_ylim(bottom=0, top=max(y + h * 3, 1e-6))


def _state_colors(n_states, colors=None):
    if colors is not None and len(colors) >= n_states:
        return list(colors)

    default_colors = [plt.get_cmap("tab10")(i % 10) for i in range(n_states)]
    if colors is None:
        return default_colors

    colors = list(colors)
    return colors + default_colors[len(colors):]


def _condition_from_observation(observation):
    observation = str(observation).lower()
    if "dcz" in observation:
        return "DCZ"
    if "saline" in observation:
        return "saline"
    return None


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


def _fit_logistic_probability(x, y, x_grid):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or len(np.unique(x)) < 2:
        return None

    x_mean = np.mean(x)
    x_std = np.std(x)
    if not np.isfinite(x_std) or x_std == 0:
        return None

    x_scaled = (x - x_mean) / x_std

    def neg_log_likelihood(params):
        logits = params[0] + params[1] * x_scaled
        p = np.clip(expit(logits), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )
    if not result.success and not np.isfinite(result.fun):
        return None

    x_grid_scaled = (np.asarray(x_grid, dtype=float) - x_mean) / x_std
    return expit(result.x[0] + result.x[1] * x_grid_scaled)


def _psychometric_group_columns(df, preferred=None):
    if preferred is not None:
        return [column for column in preferred if column in df.columns]

    group_cols = []
    for column in ["subject", "year_month_day", "session"]:
        if column in df.columns:
            group_cols.append(column)
    return group_cols


def _filter_paired_psychometric_trials(
    df,
    paired_dates,
    subject_col="subject",
    date_col="year_month_day",
):
    if paired_dates is None or paired_dates.empty:
        return df.copy()

    if subject_col not in df.columns or date_col not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["_psych_date_key"] = pd.to_datetime(
        df_copy[date_col],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    df_copy[subject_col] = df_copy[subject_col].astype(str)

    pair_rows = []
    for _, pair_row in paired_dates.iterrows():
        subject = str(pair_row[subject_col])
        pair_id = pair_row["pair_id"]
        for condition, date_column in [
            ("saline", "saline_date"),
            ("DCZ", "DCZ_date"),
        ]:
            date_key = pd.to_datetime(
                pair_row[date_column],
                errors="coerce",
            ).strftime("%Y-%m-%d")
            pair_df = df_copy[
                (df_copy[subject_col] == subject)
                & (df_copy["_psych_date_key"] == date_key)
            ].copy()
            if pair_df.empty:
                continue

            pair_df["_psych_pair_id"] = pair_id
            pair_df["observation_group"] = condition
            pair_rows.append(pair_df)

    if not pair_rows:
        return pd.DataFrame()

    paired_df = pd.concat(pair_rows, ignore_index=True)
    complete_pair_ids = (
        paired_df.groupby("_psych_pair_id")["observation_group"]
        .nunique()
        .loc[lambda counts: counts >= 2]
        .index
    )
    return paired_df[
        paired_df["_psych_pair_id"].isin(complete_pair_ids)
    ].copy()


def plot_condition_psychometric_curves(
    df,
    group_name,
    x_col="total_evidence_strength",
    y_col="first_choice_numeric",
    observation_col="observations",
    paired_dates=None,
    group_cols=None,
    min_trials=20,
    colors=None,
    figsize=(6, 4),
    ax=None,
    valueType="continue",
    bins=6,
    log=False,
):
    import utils_test

    colors = colors or {"saline": "blue", "DCZ": "red"}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    required_cols = {x_col, y_col, observation_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        ax.text(
            0.5,
            0.5,
            f"Missing columns: {sorted(missing_cols)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, pd.DataFrame()

    if paired_dates is not None:
        plot_df = _filter_paired_psychometric_trials(
            df,
            paired_dates,
        )
    else:
        plot_df = df.copy()
        plot_df["observation_group"] = plot_df[observation_col].apply(
            _condition_from_observation
        )

    plot_df = plot_df.dropna(subset=[x_col, y_col, "observation_group"])
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No saline/DCZ psychometric data for {group_name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, pd.DataFrame()

    if paired_dates is not None:
        group_cols = ["_psych_pair_id"]
    else:
        group_cols = _psychometric_group_columns(plot_df, preferred=group_cols)
        if not group_cols:
            group_cols = ["observation_group"]

    summary_rows = []
    label_added = {"saline": False, "DCZ": False}
    for group_keys, df_one in plot_df.groupby(group_cols + ["observation_group"], sort=True):
        if len(df_one) < min_trials:
            continue

        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        condition = group_keys[-1]

        try:
            utils_test.psychometric_plot_easy_logistic(
                df_one,
                x=x_col,
                y=y_col,
                ax=ax,
                point_kwargs={
                    "color": colors.get(condition, "gray"),
                    "alpha": 0.12,
                    "label": "_nolegend_",
                    "markersize": 3,
                },
                line_kwargs={
                    "color": colors.get(condition, "gray"),
                    "alpha": 0.55,
                    "linewidth": 1.4,
                    "label": condition
                    if not label_added.get(condition, False)
                    else "_nolegend_",
                },
                valueType=valueType,
                bins=bins,
                log=log,
            )
        except Exception:
            continue

        label_added[condition] = True

        row = {
            "group": group_name,
            "condition": condition,
            "n_trials": len(df_one),
        }
        for column, value in zip(group_cols, group_keys[:-1]):
            row[column] = value
        summary_rows.append(row)

    ax.set_title(f"{group_name} saline/DCZ psychometric curves")
    ax.set_xlabel(x_col)
    ax.set_ylabel("P(left choice)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    if any(label_added.values()):
        handles, labels = ax.get_legend_handles_labels()
        legend_items = []
        seen_labels = set()
        for handle, label in zip(handles, labels):
            if label in {"_nolegend_", ""} or label in seen_labels:
                continue
            legend_items.append((handle, label))
            seen_labels.add(label)
        if legend_items:
            ax.legend(
                [item[0] for item in legend_items],
                [item[1] for item in legend_items],
                frameon=False,
            )
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        ax.text(
            0.5,
            0.5,
            "No paired sessions with enough trials.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    return fig, pd.DataFrame(summary_rows)


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


def _normalize_from_values(values, quantiles=(0.01, 0.99), default=(0, 1)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return mpl.colors.Normalize(vmin=default[0], vmax=default[1])

    vmin = np.nanquantile(values, quantiles[0])
    vmax = np.nanquantile(values, quantiles[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = default
    if vmin == vmax:
        vmax = vmin + 1
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def prep_traj_speed_df(dlc_df, bodypart="Center"):
    return (
        dlc_df[bodypart][["x", "y", "mean_speed"]]
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(limit_direction="both")
        .dropna()
        .copy()
    )


def occupancy_for_trials(
    behav_df,
    trial_df,
    bodypart="Center",
    timestamp_col=("timestamp", ""),
    xbins=15,
    ybins=15,
    x_range=(0, 640),
    y_range=(0, 480),
):
    """Accumulate seconds spent in each spatial bin within trial bounds."""
    empty_occupancy = np.zeros((xbins, ybins))
    if behav_df.empty or trial_df.empty:
        return empty_occupancy

    try:
        timestamp = behavior_utils.get_behavior_column(
            behav_df,
            timestamp_col,
        )
        x = behavior_utils.get_behavior_column(
            behav_df,
            (bodypart, "x"),
        )
        y = behavior_utils.get_behavior_column(
            behav_df,
            (bodypart, "y"),
        )
    except KeyError:
        return empty_occupancy

    frame_df = (
        pd.DataFrame(
            {
                "timestamp": pd.to_numeric(
                    timestamp,
                    errors="coerce",
                ).to_numpy(),
                "x": pd.to_numeric(
                    x,
                    errors="coerce",
                ).to_numpy(),
                "y": pd.to_numeric(
                    y,
                    errors="coerce",
                ).to_numpy(),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
    )
    if frame_df.empty:
        return empty_occupancy

    t = frame_df["timestamp"].to_numpy(dtype=float)
    x_values = frame_df["x"].to_numpy(dtype=float)
    y_values = frame_df["y"].to_numpy(dtype=float)
    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0
    frame_end = np.empty_like(t)
    if len(t) > 1:
        frame_end[:-1] = t[1:]
    frame_end[-1] = t[-1] + median_dt

    occupancy = np.zeros((xbins, ybins), dtype=float)
    valid_position = np.isfinite(x_values) & np.isfinite(y_values)
    for _, trial_row in trial_df.iterrows():
        try:
            trial_start = float(trial_row["TRIAL_START"])
            trial_end = float(trial_row["TRIAL_END"])
        except (TypeError, ValueError):
            continue

        if (
            not np.isfinite(trial_start)
            or not np.isfinite(trial_end)
            or trial_end <= trial_start
        ):
            continue

        overlap_start = np.maximum(t, trial_start)
        overlap_end = np.minimum(frame_end, trial_end)
        overlap = np.clip(overlap_end - overlap_start, 0, None)
        frame_mask = valid_position & (overlap > 0)
        if not frame_mask.any():
            continue

        trial_occupancy, _, _ = np.histogram2d(
            x_values[frame_mask],
            y_values[frame_mask],
            bins=[xbins, ybins],
            range=[list(x_range), list(y_range)],
            weights=overlap[frame_mask],
        )
        occupancy += trial_occupancy

    return occupancy


def plot_four_condition_occupancy(
    pair_id,
    condition_dfs,
    split_column,
    roi_left,
    roi_right,
    roi_bottom,
    roi_top,
    bodypart="Center",
    timestamp_col=("timestamp", ""),
    xbins=15,
    ybins=15,
    x_range=(0, 640),
    y_range=(0, 480),
    cmap="viridis",
    interpolation="gaussian",
    figsize=(18, 4.5),
):
    """Plot four trial-clipped occupancy maps in one row."""
    occ_maps = [
        occupancy_for_trials(
            behav_df,
            trial_df,
            bodypart=bodypart,
            timestamp_col=timestamp_col,
            xbins=xbins,
            ybins=ybins,
            x_range=x_range,
            y_range=y_range,
        )
        for _, behav_df, trial_df in condition_dfs
    ]
    occ_values = np.concatenate([occ.ravel() for occ in occ_maps])
    occ_norm = _normalize_from_values(occ_values)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    last_im = None
    for ax, (condition_label, _, trial_df), occ in zip(
        axes,
        condition_dfs,
        occ_maps,
    ):
        last_im = ax.imshow(
            occ.T,
            origin="lower",
            extent=(*x_range, *y_range),
            cmap=cmap,
            norm=occ_norm,
            interpolation=interpolation,
        )
        if trial_df.empty or not np.any(occ):
            ax.text(
                0.5,
                0.5,
                "No trial frames",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="white",
            )
        ax.add_patch(
            patches.Rectangle(
                (roi_left, roi_bottom),
                roi_right - roi_left,
                roi_top - roi_bottom,
                linewidth=1.5,
                edgecolor="white",
                facecolor="none",
                linestyle="--",
            )
        )
        ax.set_title(condition_label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].invert_yaxis()
    fig.suptitle(f"{pair_id} occupancy by {split_column}", fontsize=12)
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes,
            label="occupancy (s/pixels)",
            shrink=0.7,
            fraction=0.025,
            pad=0.02,
            extend="both",
        )
    return fig


def trajectory_segments_in_trials(
    behav_df,
    trial_df,
    bodypart="Center",
    timestamp_col=("timestamp", ""),
):
    """Prepare one speed-colored trajectory DataFrame per valid trial."""
    if behav_df.empty or trial_df.empty:
        return []

    try:
        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                timestamp_col,
            ),
            errors="coerce",
        )
    except KeyError:
        return []

    segments = []
    for _, trial_row in trial_df.iterrows():
        try:
            trial_start = float(trial_row["TRIAL_START"])
            trial_end = float(trial_row["TRIAL_END"])
        except (TypeError, ValueError):
            continue

        if (
            not np.isfinite(trial_start)
            or not np.isfinite(trial_end)
            or trial_end <= trial_start
        ):
            continue

        trial_mask = timestamp.between(
            trial_start,
            trial_end,
            inclusive="both",
        ).fillna(False)
        trial_behav_df = behav_df.loc[trial_mask]
        if len(trial_behav_df) < 2:
            continue

        try:
            trajectory_df = prep_traj_speed_df(
                trial_behav_df,
                bodypart=bodypart,
            )
        except KeyError:
            continue
        if len(trajectory_df) >= 2:
            segments.append(trajectory_df)
    return segments


def plot_four_condition_traj_speed(
    pair_id,
    condition_dfs,
    split_column,
    bodypart="Center",
    timestamp_col=("timestamp", ""),
    cmap="inferno",
    x_range=(0, 640),
    y_range=(0, 480),
    figsize=(18, 4.5),
):
    """Plot four trial-separated speed-colored trajectories in one row."""
    trajectory_segments = [
        trajectory_segments_in_trials(
            behav_df,
            trial_df,
            bodypart=bodypart,
            timestamp_col=timestamp_col,
        )
        for _, behav_df, trial_df in condition_dfs
    ]
    speed_series = [
        trajectory_df["mean_speed"]
        for segments in trajectory_segments
        for trajectory_df in segments
    ]
    speed_values = (
        pd.concat(speed_series, ignore_index=True).dropna()
        if speed_series
        else pd.Series(dtype=float)
    )
    speed_norm = _normalize_from_values(speed_values)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    for ax, (condition_label, _, _), segments in zip(
        axes,
        condition_dfs,
        trajectory_segments,
    ):
        if segments:
            for trajectory_df in segments:
                plot_traj_speed(
                    trajectory_df,
                    cmap=cmap,
                    ax=ax,
                    norm=speed_norm,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No trajectory",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(condition_label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].invert_yaxis()
    fig.suptitle(f"{pair_id} trajectory speed by {split_column}", fontsize=12)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=speed_norm, cmap=cmap),
        ax=axes,
        label="mean speed (pixels/s)",
        shrink=0.7,
        fraction=0.025,
        pad=0.02,
    )
    return fig


def analyze_paired_behavior_by_trial_column(
    split_column,
    df_dic_saline,
    df_dic_dcz,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    behav_pair_map,
    roi_left,
    roi_right,
    roi_bottom,
    roi_top,
    hM4Di_mice,
    hM3Dq_mice,
    behavior_svg_dir,
    save_behavior_svg,
    stimulus=None,
    bodypart="Center",
    true_label=None,
    false_label=None,
    include_stationary=False,
    include_occupancy=True,
    include_trajectory_speed=True,
    include_roi_time=True,
    include_stationary_speed=True,
    include_stationary_time=True,
    stationary_speed_threshold=10,
    speed_col="mean_speed",
    mo=None,
):
    """
    Split paired saline/DCZ behavior by a trial-level boolean column.

    Each pair is organized into four conditions:
    true + saline, true + DCZ, false + saline, and false + DCZ.
    Depending on the include_* switches, the function can create occupancy
    maps, speed-colored trajectories, ROI-time summaries, stationary-speed
    traces, and stationary-time summaries.

    All time-based calculations are clipped to TRIAL_START and TRIAL_END.
    Frames and time gaps between trials are not included.
    """

    def _markdown(text):
        return mo.md(text) if mo is not None else text

    def _vstack(items):
        return mo.vstack(items) if mo is not None else items

    # 1. Convert common split columns into readable True/False labels.
    default_labels = {
        "engaged": ("engaged", "disengaged"),
        "correct": ("correct", "incorrect"),
        "previous_correct": ("previous correct", "previous incorrect"),
    }
    default_true_label, default_false_label = default_labels.get(
        split_column,
        (f"{split_column}_true", f"{split_column}_false"),
    )
    true_label = true_label or default_true_label
    false_label = false_label or default_false_label

    # 2. behavior_utils performs stimulus filtering and all trial splitting.
    split_result = behavior_utils.split_paired_behavior_by_trial_column(
        split_column=split_column,
        df_dic_saline=df_dic_saline,
        df_dic_dcz=df_dic_dcz,
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        behav_pair_map=behav_pair_map,
        stimulus=stimulus,
        bodypart=bodypart,
    )
    stimulus = split_result["stimulus"]
    behav_pair_map = split_result["behav_pair_map"]
    pair_ids = split_result["pair_ids"]
    true_pair_ids = split_result["true_pair_ids"]
    false_pair_ids = split_result["false_pair_ids"]
    four_condition_pair_ids = split_result["four_condition_pair_ids"]

    behav_df_dic_saline_true = split_result[
        "behav_df_dic_saline_true"
    ]
    behav_df_dic_saline_false = split_result[
        "behav_df_dic_saline_false"
    ]
    behav_df_dic_dcz_true = split_result["behav_df_dic_dcz_true"]
    behav_df_dic_dcz_false = split_result["behav_df_dic_dcz_false"]
    trial_df_dic_saline_true = split_result[
        "trial_df_dic_saline_true"
    ]
    trial_df_dic_saline_false = split_result[
        "trial_df_dic_saline_false"
    ]
    trial_df_dic_dcz_true = split_result["trial_df_dic_dcz_true"]
    trial_df_dic_dcz_false = split_result["trial_df_dic_dcz_false"]

    # Standard four-condition input consumed by plot_test functions.
    def trial_conditions_for_pair(pair_id):
        return [
            (
                f"{true_label} saline",
                behav_df_dic_saline[pair_id],
                trial_df_dic_saline_true[pair_id],
            ),
            (
                f"{true_label} DCZ",
                behav_df_dic_dcz[pair_id],
                trial_df_dic_dcz_true[pair_id],
            ),
            (
                f"{false_label} saline",
                behav_df_dic_saline[pair_id],
                trial_df_dic_saline_false[pair_id],
            ),
            (
                f"{false_label} DCZ",
                behav_df_dic_dcz[pair_id],
                trial_df_dic_dcz_false[pair_id],
            ),
        ]

    # 3. Generate the per-pair occupancy and/or trajectory figures requested
    # by include_occupancy and include_trajectory_speed.
    four_condition_pair_figures = []
    if include_occupancy or include_trajectory_speed:
        for pair_id in four_condition_pair_ids:
            if include_occupancy:
                four_condition_pair_figures.append(
                    plot_four_condition_occupancy(
                        pair_id=pair_id,
                        condition_dfs=trial_conditions_for_pair(pair_id),
                        split_column=split_column,
                        roi_left=roi_left,
                        roi_right=roi_right,
                        roi_bottom=roi_bottom,
                        roi_top=roi_top,
                        bodypart=bodypart,
                    )
                )
            if include_trajectory_speed:
                four_condition_pair_figures.append(
                    plot_four_condition_traj_speed(
                        pair_id=pair_id,
                        condition_dfs=trial_conditions_for_pair(pair_id),
                        split_column=split_column,
                        bodypart=bodypart,
                    )
                )

        save_figures_svg(
            four_condition_pair_figures,
            f"paired_behavior_{true_label}_vs_{false_label}",
            save_dir=behavior_svg_dir,
            enabled=save_behavior_svg,
        )

    true_roi_time_ratio_summary = pd.DataFrame()
    false_roi_time_ratio_summary = pd.DataFrame()
    roi_split_summary = pd.DataFrame()
    roi_time_ratio_four_condition_fig = None

    # 4. Helpers shared by ROI and stationary four-condition statistics.
    def roi_summary_for_merge(summary_df, split_label):
        """Rename saline/DCZ columns before merging True and False results."""
        if summary_df.empty:
            return pd.DataFrame(
                columns=[
                    "pair_id",
                    "subject",
                    f"{split_label}_saline",
                    f"{split_label}_dcz",
                ]
            )

        roi_df = summary_df.copy()
        if "subject" not in roi_df.columns and "pair_id" in roi_df.columns:
            roi_df["subject"] = roi_df["pair_id"].str[:6]
        for column in ["pair_id", "subject", "saline", "DCZ"]:
            if column not in roi_df.columns:
                roi_df[column] = pd.Series(dtype=float)
        return roi_df[
            ["pair_id", "subject", "saline", "DCZ"]
        ].rename(
            columns={
                "saline": f"{split_label}_saline",
                "DCZ": f"{split_label}_dcz",
            }
        )

    def comparison_pvalue(group_df, left_col, right_col):
        """Run a paired Wilcoxon test after removing incomplete pairs."""
        paired_df = group_df[[left_col, right_col]].dropna()
        if paired_df.empty:
            return float("nan")
        try:
            return stats.wilcoxon(
                paired_df[left_col],
                paired_df[right_col],
            ).pvalue
        except ValueError:
            return float("nan")

    def add_sig_bar(ax, x0, x1, y, h, label):
        """Draw one significance bracket and its star label."""
        ax.plot(
            [x0, x0, x1, x1],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1,
        )
        ax.text(
            (x0 + x1) / 2,
            y + h,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    def plot_roi_ratio_four_conditions(summary_df):
        """Compare four ROI ratios separately for hM4Di and hM3Dq mice."""
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharey=True,
        )

        condition_cols = [
            "true_saline",
            "true_dcz",
            "false_saline",
            "false_dcz",
        ]
        condition_labels = [
            f"{true_label}\nsaline",
            f"{true_label}\nDCZ",
            f"{false_label}\nsaline",
            f"{false_label}\nDCZ",
        ]
        condition_colors = ["#2563eb", "#dc2626", "#93c5fd", "#fca5a5"]
        # Four paired comparisons:
        # True saline vs DCZ, False saline vs DCZ,
        # saline True vs False, and DCZ True vs False.
        comparisons = [
            (0, 1, "true_saline", "true_dcz"),
            (2, 3, "false_saline", "false_dcz"),
            (0, 2, "true_saline", "false_saline"),
            (1, 3, "true_dcz", "false_dcz"),
        ]

        for ax, mice, group_name in zip(
            axes,
            [hM4Di_mice, hM3Dq_mice],
            ["hM4Di", "hM3Dq"],
        ):
            group_df = summary_df[
                summary_df["subject"].isin(mice)
            ].copy()
            for column in condition_cols:
                if column not in group_df.columns:
                    group_df[column] = pd.Series(dtype=float)

            for x0, x1, left_col, right_col in comparisons:
                paired_df = group_df[[left_col, right_col]].dropna()
                for _, row in paired_df.iterrows():
                    ax.plot(
                        [x0, x1],
                        [row[left_col], row[right_col]],
                        color="gray",
                        alpha=0.25,
                        linewidth=1,
                        zorder=1,
                    )

            for x, (column, color) in enumerate(zip(condition_cols, condition_colors)):
                values = pd.to_numeric(group_df[column], errors="coerce").dropna()
                ax.scatter(
                    [x] * len(values),
                    values,
                    color=color,
                    edgecolor="black",
                    zorder=3,
                )

            all_values = pd.to_numeric(
                group_df[condition_cols].stack(),
                errors="coerce",
            ).dropna()
            if all_values.empty:
                y_max = 1.0
                y_range = 0.1
            else:
                y_max = float(all_values.max())
                y_min = float(all_values.min())
                y_range = y_max - y_min
                if not np.isfinite(y_range) or y_range == 0:
                    y_range = max(abs(y_max), 1.0) * 0.1

            h = y_range * 0.04
            for idx, (x0, x1, left_col, right_col) in enumerate(comparisons):
                p_value = comparison_pvalue(group_df, left_col, right_col)
                add_sig_bar(
                    ax,
                    x0,
                    x1,
                    y_max + y_range * (0.10 + 0.13 * idx),
                    h,
                    p_to_star(p_value),
                )

            ax.set_xticks(range(4))
            ax.set_xticklabels(condition_labels)
            ax.set_ylabel("fraction of time in ROI")
            ax.set_title(f"{group_name}, n={group_df['pair_id'].nunique()}")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(
                bottom=0,
                top=y_max + y_range * 0.75,
            )

        fig.tight_layout()
        return fig

    # 5. ROI-time ratio.
    # Formula for one condition:
    # sum(time inside ROI and selected trials) / sum(selected trial durations).
    def roi_time_ratio_in_trials(behav_df, trial_df):
        """Calculate a trial-clipped ROI ratio and its numerator/denominator."""
        if behav_df.empty or trial_df.empty:
            return np.nan, np.nan, 0.0

        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                ("timestamp", ""),
            ),
            errors="coerce",
        )
        x = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, "x"),
            ),
            errors="coerce",
        )
        y = pd.to_numeric(
            behavior_utils.get_behavior_column(
                behav_df,
                (bodypart, "y"),
            ),
            errors="coerce",
        )
        frame_df = (
            pd.DataFrame(
                {
                    "timestamp": timestamp.to_numpy(),
                    "x": x.to_numpy(),
                    "y": y.to_numpy(),
                }
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
        if frame_df.empty:
            return np.nan, np.nan, 0.0

        t = frame_df["timestamp"].to_numpy(dtype=float)
        x_values = frame_df["x"].to_numpy(dtype=float)
        y_values = frame_df["y"].to_numpy(dtype=float)
        dt = np.diff(t)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0
        frame_end = np.empty_like(t)
        if len(t) > 1:
            frame_end[:-1] = t[1:]
        frame_end[-1] = t[-1] + median_dt

        # Position validity and ROI membership are properties of each frame.
        valid_position = np.isfinite(x_values) & np.isfinite(y_values)
        in_roi = (
            valid_position
            & (x_values >= roi_left)
            & (x_values <= roi_right)
            & (y_values >= roi_bottom)
            & (y_values <= roi_top)
        )

        roi_time = 0.0
        trial_time = 0.0
        for _, trial_row in trial_df.iterrows():
            try:
                trial_start = float(trial_row["TRIAL_START"])
                trial_end = float(trial_row["TRIAL_END"])
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            # The denominator is the sum of this condition's trial lengths.
            trial_time += trial_end - trial_start
            # The numerator only receives frame time inside this trial.
            overlap_start = np.maximum(t, trial_start)
            overlap_end = np.minimum(frame_end, trial_end)
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            roi_time += float(overlap[in_roi].sum())

        if trial_time <= 0:
            return np.nan, roi_time, trial_time
        return roi_time / trial_time, roi_time, trial_time

    def paired_trial_roi_time_ratio_comparison(target_value):
        """Calculate saline/DCZ ROI ratios for True or False trials."""
        saline_trial_df_dic = (
            trial_df_dic_saline_true
            if target_value
            else trial_df_dic_saline_false
        )
        dcz_trial_df_dic = (
            trial_df_dic_dcz_true
            if target_value
            else trial_df_dic_dcz_false
        )
        rows = []
        for pair_id in pair_ids:
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
                or pair_id not in saline_trial_df_dic
                or pair_id not in dcz_trial_df_dic
            ):
                continue

            saline_ratio, saline_roi_time, saline_trial_time = (
                roi_time_ratio_in_trials(
                    behav_df_dic_saline[pair_id],
                    saline_trial_df_dic[pair_id],
                )
            )
            dcz_ratio, dcz_roi_time, dcz_trial_time = (
                roi_time_ratio_in_trials(
                    behav_df_dic_dcz[pair_id],
                    dcz_trial_df_dic[pair_id],
                )
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "saline": saline_ratio,
                    "DCZ": dcz_ratio,
                    "DCZ_minus_saline": dcz_ratio - saline_ratio,
                    "saline_roi_time": saline_roi_time,
                    "saline_trial_time": saline_trial_time,
                    "DCZ_roi_time": dcz_roi_time,
                    "DCZ_trial_time": dcz_trial_time,
                    "split_column": split_column,
                    "split_value": target_value,
                }
            )

        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            summary_df = summary_df.dropna(subset=["saline", "DCZ"])
        if (
            behav_pair_map is not None
            and not behav_pair_map.empty
            and not summary_df.empty
        ):
            summary_df = behav_pair_map.merge(
                summary_df,
                on="pair_id",
                how="inner",
            )
        return summary_df

    # True and False summaries are calculated separately, then merged into
    # true_saline, true_dcz, false_saline, and false_dcz columns.
    if include_roi_time:
        true_roi_time_ratio_summary = (
            paired_trial_roi_time_ratio_comparison(True)
        )
        false_roi_time_ratio_summary = (
            paired_trial_roi_time_ratio_comparison(False)
        )
        roi_split_summary = roi_summary_for_merge(
            true_roi_time_ratio_summary,
            "true",
        ).merge(
            roi_summary_for_merge(false_roi_time_ratio_summary, "false"),
            on=["pair_id", "subject"],
            how="outer",
        )
        roi_time_ratio_four_condition_fig = plot_roi_ratio_four_conditions(
            roi_split_summary,
        )

        save_figure_svg(
            roi_time_ratio_four_condition_fig,
            f"roi_time_ratio_groups_{true_label}_vs_{false_label}",
            save_dir=behavior_svg_dir,
            enabled=save_behavior_svg,
        )

    stationary_speed_figures = []
    true_stationary_time_ratio_summary = pd.DataFrame()
    false_stationary_time_ratio_summary = pd.DataFrame()
    stationary_split_summary = pd.DataFrame()
    stationary_time_ratio_four_condition_fig = None

    # 6. Stationary-speed analysis.
    # A frame is stationary when speed <= stationary_speed_threshold.
    def has_speed_data(behav_df):
        """Check whether the requested speed column contains usable data."""
        if behav_df.empty:
            return False
        if bodypart not in behav_df.columns.get_level_values(0):
            return False
        if speed_col not in behav_df[bodypart].columns:
            return False
        speed = pd.to_numeric(behav_df[(bodypart, speed_col)], errors="coerce")
        return speed.notna().any()

    def stationary_conditions_for_pair(pair_id):
        """Return behavior, trial rows, and color for four conditions."""
        return [
            (
                f"{true_label} saline",
                behav_df_dic_saline_true[pair_id],
                trial_df_dic_saline_true[pair_id],
                "#2563eb",
            ),
            (
                f"{true_label} DCZ",
                behav_df_dic_dcz_true[pair_id],
                trial_df_dic_dcz_true[pair_id],
                "#dc2626",
            ),
            (
                f"{false_label} saline",
                behav_df_dic_saline_false[pair_id],
                trial_df_dic_saline_false[pair_id],
                "#93c5fd",
            ),
            (
                f"{false_label} DCZ",
                behav_df_dic_dcz_false[pair_id],
                trial_df_dic_dcz_false[pair_id],
                "#fca5a5",
            ),
        ]

    def frame_time_speed_df(behav_df):
        """Create a clean, time-sorted table containing timestamp and speed."""
        timestamp = pd.to_numeric(
            behavior_utils.get_behavior_column(behav_df, ("timestamp", "")),
            errors="coerce",
        )
        speed = pd.to_numeric(
            behavior_utils.get_behavior_column(behav_df, (bodypart, speed_col)),
            errors="coerce",
        )
        frame_df = pd.DataFrame(
            {
                "timestamp": timestamp.to_numpy(),
                "speed": speed.to_numpy(),
            }
        ).replace([np.inf, -np.inf], np.nan)
        return frame_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    def frame_intervals(frame_df):
        """Convert frame timestamps into [start, end) intervals."""
        t = frame_df["timestamp"].to_numpy(dtype=float)
        speed_values = frame_df["speed"].to_numpy(dtype=float)
        dt = np.diff(t)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0
        frame_end = np.empty_like(t)
        if len(t) > 1:
            frame_end[:-1] = t[1:]
        frame_end[-1] = t[-1] + median_dt
        return t, frame_end, speed_values

    def shade_stationary_segments_in_trials(ax, behav_df, trial_df):
        """Shade stationary intervals after clipping them to trial bounds."""
        if behav_df.empty or trial_df.empty:
            return

        frame_df = frame_time_speed_df(behav_df)
        if frame_df.empty:
            return

        t, frame_end, speed_values = frame_intervals(frame_df)
        stationary = np.isfinite(speed_values) & (
            speed_values <= stationary_speed_threshold
        )

        for _, trial_row in trial_df.iterrows():
            try:
                trial_start = float(trial_row["TRIAL_START"])
                trial_end = float(trial_row["TRIAL_END"])
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            overlap_start = np.maximum(t, trial_start)
            overlap_end = np.minimum(frame_end, trial_end)
            draw_mask = stationary & (overlap_end > overlap_start)
            starts = overlap_start[draw_mask]
            ends = overlap_end[draw_mask]
            if len(starts) == 0:
                continue

            # Merge adjacent stationary frame intervals into one shaded span.
            run_start = starts[0]
            run_end = ends[0]
            for start, end in zip(starts[1:], ends[1:]):
                if start <= run_end + 1e-9:
                    run_end = max(run_end, end)
                else:
                    ax.axvspan(
                        run_start,
                        run_end,
                        color="gray",
                        alpha=0.18,
                        linewidth=0,
                    )
                    run_start = start
                    run_end = end

            ax.axvspan(
                run_start,
                run_end,
                color="gray",
                alpha=0.18,
                linewidth=0,
            )

    def speed_trace_segments_in_trials(behav_df, trial_df):
        """Return one speed trace per trial to prevent cross-trial lines."""
        if behav_df.empty or trial_df.empty:
            return []

        frame_df = frame_time_speed_df(behav_df)
        if frame_df.empty:
            return []

        segments = []
        for _, trial_row in trial_df.iterrows():
            try:
                trial_start = float(trial_row["TRIAL_START"])
                trial_end = float(trial_row["TRIAL_END"])
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            trial_segment = frame_df.loc[
                frame_df["timestamp"].between(
                    trial_start,
                    trial_end,
                    inclusive="both",
                )
            ].dropna(subset=["speed"])
            if not trial_segment.empty:
                segments.append(trial_segment)
        return segments

    def plot_four_condition_stationary_speed_trace(pair_id):
        """Plot four speed traces with trial-clipped stationary shading."""
        condition_dfs = stationary_conditions_for_pair(pair_id)
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(18, 4),
            sharey=True,
        )

        for ax, (condition_label, behav_df, trial_df, color) in zip(axes, condition_dfs):
            if has_speed_data(behav_df):
                trace_segments = speed_trace_segments_in_trials(
                    behav_df,
                    trial_df,
                )
                if not trace_segments:
                    ax.text(
                        0.5,
                        0.5,
                        "No speed",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                else:
                    shade_stationary_segments_in_trials(ax, behav_df, trial_df)
                    for trace_segment in trace_segments:
                        ax.plot(
                            trace_segment["timestamp"],
                            trace_segment["speed"],
                            color=color,
                            linewidth=1,
                            alpha=0.85,
                        )
                    ax.set_xlabel("time (s)")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No speed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.axhline(
                stationary_speed_threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
            ax.set_title(condition_label, fontsize=10)
            ax.grid(axis="y", alpha=0.25)

        axes[0].set_ylabel(f"{bodypart} {speed_col} (pixels/s)")
        fig.suptitle(f"{pair_id} stationary speed trace by {split_column}", fontsize=12)
        fig.tight_layout()
        return fig

    def stationary_time_ratio_in_trials(behav_df, trial_df):
        """Return stationary_time / selected_trial_time for one condition."""
        if behav_df.empty or trial_df.empty:
            return np.nan, np.nan, 0.0

        frame_df = frame_time_speed_df(behav_df)
        if frame_df.empty:
            return np.nan, np.nan, 0.0

        t, frame_end, speed_values = frame_intervals(frame_df)
        stationary = np.isfinite(speed_values) & (
            speed_values <= stationary_speed_threshold
        )
        stationary_time = 0.0
        trial_time = 0.0
        for _, trial_row in trial_df.iterrows():
            try:
                trial_start = float(trial_row["TRIAL_START"])
                trial_end = float(trial_row["TRIAL_END"])
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(trial_start)
                or not np.isfinite(trial_end)
                or trial_end <= trial_start
            ):
                continue

            # As for ROI, each condition uses only its own trial durations.
            trial_time += trial_end - trial_start
            overlap_start = np.maximum(t, trial_start)
            overlap_end = np.minimum(frame_end, trial_end)
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            stationary_time += float(overlap[stationary].sum())

        if trial_time <= 0:
            return np.nan, stationary_time, trial_time
        return stationary_time / trial_time, stationary_time, trial_time

    def paired_trial_stationary_time_ratio_comparison(target_value):
        """Calculate paired saline/DCZ stationary ratios for one split value."""
        saline_trial_df_dic = (
            trial_df_dic_saline_true
            if target_value
            else trial_df_dic_saline_false
        )
        dcz_trial_df_dic = (
            trial_df_dic_dcz_true
            if target_value
            else trial_df_dic_dcz_false
        )
        rows = []
        for pair_id in pair_ids:
            if (
                pair_id not in behav_df_dic_saline
                or pair_id not in behav_df_dic_dcz
                or pair_id not in saline_trial_df_dic
                or pair_id not in dcz_trial_df_dic
            ):
                continue

            saline_ratio, saline_stationary_time, saline_trial_time = (
                stationary_time_ratio_in_trials(
                    behav_df_dic_saline[pair_id],
                    saline_trial_df_dic[pair_id],
                )
            )
            dcz_ratio, dcz_stationary_time, dcz_trial_time = (
                stationary_time_ratio_in_trials(
                    behav_df_dic_dcz[pair_id],
                    dcz_trial_df_dic[pair_id],
                )
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "saline": saline_ratio,
                    "DCZ": dcz_ratio,
                    "DCZ_minus_saline": dcz_ratio - saline_ratio,
                    "saline_stationary_time": saline_stationary_time,
                    "saline_trial_time": saline_trial_time,
                    "DCZ_stationary_time": dcz_stationary_time,
                    "DCZ_trial_time": dcz_trial_time,
                    "speed_threshold": stationary_speed_threshold,
                    "split_column": split_column,
                    "split_value": target_value,
                }
            )

        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            summary_df = summary_df.dropna(subset=["saline", "DCZ"])
        if (
            behav_pair_map is not None
            and not behav_pair_map.empty
            and not summary_df.empty
        ):
            summary_df = behav_pair_map.merge(
                summary_df,
                on="pair_id",
                how="inner",
            )
        return summary_df

    def stationary_summary_for_merge(summary_df, split_label):
        """Rename stationary columns before combining True and False groups."""
        if summary_df.empty:
            return pd.DataFrame(
                columns=[
                    "pair_id",
                    "subject",
                    f"{split_label}_saline",
                    f"{split_label}_dcz",
                ]
            )

        stationary_df = summary_df.copy()
        if "subject" not in stationary_df.columns and "pair_id" in stationary_df.columns:
            stationary_df["subject"] = stationary_df["pair_id"].str[:6]
        for column in ["pair_id", "subject", "saline", "DCZ"]:
            if column not in stationary_df.columns:
                stationary_df[column] = pd.Series(dtype=float)
        return stationary_df[
            ["pair_id", "subject", "saline", "DCZ"]
        ].rename(
            columns={
                "saline": f"{split_label}_saline",
                "DCZ": f"{split_label}_dcz",
            }
        )

    def plot_stationary_ratio_four_conditions(summary_df):
        """Compare four stationary ratios for hM4Di and hM3Dq mice."""
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharey=True,
        )

        condition_cols = [
            "true_saline",
            "true_dcz",
            "false_saline",
            "false_dcz",
        ]
        condition_labels = [
            f"{true_label}\nsaline",
            f"{true_label}\nDCZ",
            f"{false_label}\nsaline",
            f"{false_label}\nDCZ",
        ]
        condition_colors = ["#2563eb", "#dc2626", "#93c5fd", "#fca5a5"]
        # Use the same four paired comparisons as the ROI summary plot.
        comparisons = [
            (0, 1, "true_saline", "true_dcz"),
            (2, 3, "false_saline", "false_dcz"),
            (0, 2, "true_saline", "false_saline"),
            (1, 3, "true_dcz", "false_dcz"),
        ]

        for ax, mice, group_name in zip(
            axes,
            [hM4Di_mice, hM3Dq_mice],
            ["hM4Di", "hM3Dq"],
        ):
            group_df = summary_df[summary_df["subject"].isin(mice)].copy()
            for column in condition_cols:
                if column not in group_df.columns:
                    group_df[column] = pd.Series(dtype=float)

            for x0, x1, left_col, right_col in comparisons:
                paired_df = group_df[[left_col, right_col]].dropna()
                for _, row in paired_df.iterrows():
                    ax.plot(
                        [x0, x1],
                        [row[left_col], row[right_col]],
                        color="gray",
                        alpha=0.25,
                        linewidth=1,
                        zorder=1,
                    )

            for x, (column, color) in enumerate(zip(condition_cols, condition_colors)):
                values = pd.to_numeric(group_df[column], errors="coerce").dropna()
                ax.scatter(
                    [x] * len(values),
                    values,
                    color=color,
                    edgecolor="black",
                    zorder=3,
                )

            all_values = pd.to_numeric(
                group_df[condition_cols].stack(),
                errors="coerce",
            ).dropna()
            if all_values.empty:
                y_max = 1.0
                y_range = 0.1
            else:
                y_max = float(all_values.max())
                y_min = float(all_values.min())
                y_range = y_max - y_min
                if not np.isfinite(y_range) or y_range == 0:
                    y_range = max(abs(y_max), 1.0) * 0.1

            h = y_range * 0.04
            for idx, (x0, x1, left_col, right_col) in enumerate(comparisons):
                p_value = comparison_pvalue(group_df, left_col, right_col)
                add_sig_bar(
                    ax,
                    x0,
                    x1,
                    y_max + y_range * (0.10 + 0.13 * idx),
                    h,
                    p_to_star(p_value),
                )

            ax.set_xticks(range(4))
            ax.set_xticklabels(condition_labels)
            ax.set_ylabel("fraction of time stationary")
            ax.set_title(f"{group_name}, n={group_df['pair_id'].nunique()}")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(
                bottom=0,
                top=y_max + y_range * 0.75,
            )

        fig.tight_layout()
        return fig

    # Draw speed traces only when both the stationary master switch and
    # the trace-specific switch are enabled.
    if include_stationary and include_stationary_speed:
        stationary_pair_ids = [
            pair_id
            for pair_id in pair_ids
            if pair_id in behav_df_dic_saline_true
            and pair_id in behav_df_dic_dcz_true
            and pair_id in behav_df_dic_saline_false
            and pair_id in behav_df_dic_dcz_false
            and any(
                has_speed_data(behav_df)
                for _, behav_df, _, _ in stationary_conditions_for_pair(pair_id)
            )
        ]
        stationary_speed_figures = [
            plot_four_condition_stationary_speed_trace(pair_id)
            for pair_id in stationary_pair_ids
        ]

        save_figures_svg(
            stationary_speed_figures,
            f"stationary_speed_trace_{true_label}_vs_{false_label}",
            save_dir=behavior_svg_dir,
            enabled=save_behavior_svg,
        )

    # Calculate and plot stationary ratios independently from speed traces.
    if include_stationary and include_stationary_time:
        true_stationary_time_ratio_summary = (
            paired_trial_stationary_time_ratio_comparison(True)
        )
        false_stationary_time_ratio_summary = (
            paired_trial_stationary_time_ratio_comparison(False)
        )
        stationary_split_summary = stationary_summary_for_merge(
            true_stationary_time_ratio_summary,
            "true",
        ).merge(
            stationary_summary_for_merge(
                false_stationary_time_ratio_summary,
                "false",
            ),
            on=["pair_id", "subject"],
            how="outer",
        )
        stationary_time_ratio_four_condition_fig = (
            plot_stationary_ratio_four_conditions(stationary_split_summary)
        )

        save_figure_svg(
            stationary_time_ratio_four_condition_fig,
            f"stationary_time_ratio_groups_{true_label}_vs_{false_label}",
            save_dir=behavior_svg_dir,
            enabled=save_behavior_svg,
        )

    # 7. Assemble only the requested outputs into one Marimo view.
    view_items = []
    if four_condition_pair_figures:
        view_items.extend(
            [
                _markdown(f"## {true_label} vs {false_label} paired behavior"),
                *four_condition_pair_figures,
            ]
        )
    if roi_time_ratio_four_condition_fig is not None:
        view_items.extend(
            [
                _markdown(f"## {true_label} vs {false_label} ROI time ratio groups"),
                roi_time_ratio_four_condition_fig,
            ]
        )
    if stationary_speed_figures:
        view_items.extend(
            [
                _markdown(f"## {true_label} vs {false_label} stationary speed traces"),
                *stationary_speed_figures,
            ]
        )
    if stationary_time_ratio_four_condition_fig is not None:
        view_items.extend(
            [
                _markdown(f"## {true_label} vs {false_label} stationary time ratio groups"),
                stationary_time_ratio_four_condition_fig,
            ]
        )
    if not view_items:
        view_items.append(_markdown(f"## {true_label} vs {false_label}: no output selected"))

    # Return intermediate dictionaries and summary tables as well as plots.
    # This makes it possible to inspect or reuse every analysis stage.
    return {
        "split_column": split_column,
        "stimulus": stimulus,
        "split_result": split_result,
        "behav_pair_map": behav_pair_map,
        "true_label": true_label,
        "false_label": false_label,
        "true_pair_ids": true_pair_ids,
        "false_pair_ids": false_pair_ids,
        "include_stationary": include_stationary,
        "include_occupancy": include_occupancy,
        "include_trajectory_speed": include_trajectory_speed,
        "include_roi_time": include_roi_time,
        "include_stationary_speed": include_stationary_speed,
        "include_stationary_time": include_stationary_time,
        "stationary_speed_threshold": stationary_speed_threshold,
        "behav_df_dic_dcz_true": behav_df_dic_dcz_true,
        "behav_df_dic_dcz_false": behav_df_dic_dcz_false,
        "behav_df_dic_saline_true": behav_df_dic_saline_true,
        "behav_df_dic_saline_false": behav_df_dic_saline_false,
        "trial_df_dic_dcz_true": trial_df_dic_dcz_true,
        "trial_df_dic_dcz_false": trial_df_dic_dcz_false,
        "trial_df_dic_saline_true": trial_df_dic_saline_true,
        "trial_df_dic_saline_false": trial_df_dic_saline_false,
        "four_condition_pair_ids": four_condition_pair_ids,
        "four_condition_pair_figures": four_condition_pair_figures,
        "true_roi_time_ratio_summary": true_roi_time_ratio_summary,
        "false_roi_time_ratio_summary": false_roi_time_ratio_summary,
        "roi_split_summary": roi_split_summary,
        "roi_time_ratio_four_condition_fig": roi_time_ratio_four_condition_fig,
        "stationary_speed_figures": stationary_speed_figures,
        "true_stationary_time_ratio_summary": true_stationary_time_ratio_summary,
        "false_stationary_time_ratio_summary": false_stationary_time_ratio_summary,
        "stationary_split_summary": stationary_split_summary,
        "stationary_time_ratio_four_condition_fig": stationary_time_ratio_four_condition_fig,
        "view": _vstack(view_items),
    }


def plot_pair_occupancy(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    roi_left,
    roi_right,
    roi_bottom,
    roi_top,
    bodypart="Center",
    timestamp_col=("timestamp", ""),
    xbins=15,
    ybins=15,
    cmap="viridis",
    figsize=(10, 5),
    quantiles=(0.01, 0.99),
    interpolation="gaussian",
):
    pair_dcz = behav_df_dic_dcz[pair_id]
    pair_saline = behav_df_dic_saline[pair_id]

    occ_dcz = behavior_utils.occupancy_map(
        pair_dcz[(bodypart, "x")],
        pair_dcz[(bodypart, "y")],
        pair_dcz[timestamp_col],
        xbins=xbins,
        ybins=ybins,
    )

    occ_saline = behavior_utils.occupancy_map(
        pair_saline[(bodypart, "x")],
        pair_saline[(bodypart, "y")],
        pair_saline[timestamp_col],
        xbins=xbins,
        ybins=ybins,
    )

    pair_extents = (
        np.nanmin([
            pair_dcz[(bodypart, "x")].min(),
            pair_saline[(bodypart, "x")].min(),
        ]),
        np.nanmax([
            pair_dcz[(bodypart, "x")].max(),
            pair_saline[(bodypart, "x")].max(),
        ]),
        np.nanmin([
            pair_dcz[(bodypart, "y")].min(),
            pair_saline[(bodypart, "y")].min(),
        ]),
        np.nanmax([
            pair_dcz[(bodypart, "y")].max(),
            pair_saline[(bodypart, "y")].max(),
        ]),
    )

    pair_occ_values = np.concatenate([
        occ_dcz.ravel(),
        occ_saline.ravel(),
    ])
    pair_occ_norm = _normalize_from_values(pair_occ_values, quantiles=quantiles)

    pair_occ_fig, pair_occ_axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    pair_occ_im0 = pair_occ_axes[0].imshow(
        occ_saline.T,
        origin="lower",
        extent=pair_extents,
        cmap=cmap,
        norm=pair_occ_norm,
        interpolation=interpolation,
    )
    pair_occ_axes[0].set_title(f"{pair_id} saline", fontsize=10)

    pair_occ_im1 = pair_occ_axes[1].imshow(
        occ_dcz.T,
        origin="lower",
        extent=pair_extents,
        cmap=cmap,
        norm=pair_occ_norm,
        interpolation=interpolation,
    )
    pair_occ_axes[1].set_title(f"{pair_id} DCZ", fontsize=10)

    for pair_occ_ax in pair_occ_axes:
        pair_roi_rect = patches.Rectangle(
            (roi_left, roi_bottom),
            roi_right - roi_left,
            roi_top - roi_bottom,
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
        )
        pair_occ_ax.add_patch(pair_roi_rect)
        pair_occ_ax.set_aspect("equal")
        pair_occ_ax.set_xlim(0, 640)
        pair_occ_ax.set_ylim(0, 480)
        pair_occ_ax.set_xticks([])
        pair_occ_ax.set_yticks([])
    pair_occ_axes[0].invert_yaxis()

    pair_occ_fig.colorbar(
        pair_occ_im1,
        ax=pair_occ_axes,
        label="occupancy (s/pixels)",
        shrink=0.7,
        fraction=0.04,
        pad=0.02,
        extend="both",
    )

    return pair_occ_fig


def plot_pair_traj_speed(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    cmap="inferno",
    figsize=(12, 6),
    quantiles=(0.01, 0.99),
    invert_y=True,
):
    pair_dcz = behav_df_dic_dcz[pair_id]
    pair_saline = behav_df_dic_saline[pair_id]

    pair_dcz_traj = prep_traj_speed_df(pair_dcz, bodypart=bodypart)
    pair_saline_traj = prep_traj_speed_df(pair_saline, bodypart=bodypart)

    pair_speed_values = pd.concat(
        [
            pair_dcz_traj["mean_speed"],
            pair_saline_traj["mean_speed"],
        ],
        ignore_index=True,
    ).dropna()

    pair_speed_norm = _normalize_from_values(
        pair_speed_values,
        quantiles=quantiles,
    )

    pair_traj_fig, pair_traj_axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    plot_traj_speed(
        pair_saline_traj,
        cmap=cmap,
        ax=pair_traj_axes[0],
        norm=pair_speed_norm,
    )
    pair_traj_axes[0].set_title(f"{pair_id} saline")

    plot_traj_speed(
        pair_dcz_traj,
        cmap=cmap,
        ax=pair_traj_axes[1],
        norm=pair_speed_norm,
    )
    pair_traj_axes[1].set_title(f"{pair_id} DCZ")

    for pair_traj_ax in pair_traj_axes:
        pair_traj_ax.set_aspect("equal")
        pair_traj_ax.autoscale()
        pair_traj_ax.set_xlim(0, 640)
        pair_traj_ax.set_ylim(0, 480)
        pair_traj_ax.set_xticks([])
        pair_traj_ax.set_yticks([])

    if invert_y:
        pair_traj_axes[0].invert_yaxis()

    pair_speed_sm = mpl.cm.ScalarMappable(
        norm=pair_speed_norm,
        cmap=cmap,
    )
    pair_traj_fig.colorbar(
        pair_speed_sm,
        ax=pair_traj_axes,
        label="mean speed (pixels/s)",
        shrink=0.6,
    )

    return pair_traj_fig


def plot_paired_behavior_figures(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    roi_left,
    roi_right,
    roi_bottom,
    roi_top,
    bodypart="Center",
):
    figures = []
    for pair_id in pair_ids:
        figures.append(
            plot_pair_occupancy(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                roi_left=roi_left,
                roi_right=roi_right,
                roi_bottom=roi_bottom,
                roi_top=roi_top,
                bodypart=bodypart,
            )
        )
        figures.append(
            plot_pair_traj_speed(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                bodypart=bodypart,
            )
        )
    return figures


def plot_stationary_time_ratio_groups(
    stationary_time_ratio_summary,
    hM4Di_mice,
    hM3Dq_mice,
    figsize=(8, 5),
):
    plot_df = stationary_time_ratio_summary.copy()
    if "subject" not in plot_df.columns:
        plot_df["subject"] = plot_df["pair_id"].astype(str).str[:6]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    def _plot_group(ax, mice, group_name):
        group_df = plot_df[
            plot_df["subject"].isin(mice)
        ].dropna(subset=["saline", "DCZ"]).copy()

        for _, row in group_df.iterrows():
            ax.plot(
                [0, 1],
                [row["saline"], row["DCZ"]],
                color="gray",
                alpha=0.4,
                linewidth=1,
            )

        ax.scatter(
            [0] * len(group_df),
            group_df["saline"],
            color="blue",
            edgecolor="black",
            label="saline",
            zorder=3,
        )
        ax.scatter(
            [1] * len(group_df),
            group_df["DCZ"],
            color="red",
            edgecolor="black",
            label="DCZ",
            zorder=3,
        )

        if len(group_df) > 0:
            try:
                p_value = stats.wilcoxon(
                    group_df["saline"],
                    group_df["DCZ"],
                ).pvalue
                p_text = f"p={p_value:.3g}"
            except ValueError:
                p_value = np.nan
                p_text = "p=NA"
        else:
            p_value = np.nan
            p_text = "p=NA"

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["saline", "DCZ"])
        ax.set_title(f"{group_name}, n={len(group_df)}, {p_text}")
        add_paired_significance_label(
            ax,
            group_df["saline"],
            group_df["DCZ"],
            p_to_star(p_value),
        )
        ax.grid(axis="y", alpha=0.3)
        return group_df

    hm4_df = _plot_group(axes[0], hM4Di_mice, "hM4Di")
    hm3_df = _plot_group(axes[1], hM3Dq_mice, "hM3Dq")
    axes[0].set_ylabel("fraction of time stationary")
    fig.tight_layout()
    return fig, hm4_df, hm3_df


def _speed_trace_df(
    behav_df,
    bodypart="Center",
    speed_col="mean_speed",
    timestamp_col=("timestamp", ""),
):
    speed = pd.to_numeric(
        behavior_utils.get_behavior_column(behav_df, (bodypart, speed_col)),
        errors="coerce",
    )
    try:
        x_values = pd.to_numeric(
            behavior_utils.get_behavior_column(behav_df, timestamp_col),
            errors="coerce",
        )
        x_label = "time (s)"
    except KeyError:
        x_values = pd.Series(np.arange(len(speed)), index=speed.index)
        x_label = "frame"

    trace_df = pd.DataFrame(
        {
            "x": x_values.to_numpy(),
            "speed": speed.to_numpy(),
        }
    )
    trace_df = trace_df.dropna(subset=["x", "speed"]).reset_index(drop=True)
    return trace_df, x_label


def _shade_stationary_segments(
    ax,
    x_values,
    stationary,
    color="gray",
    alpha=0.18,
):
    x_values = np.asarray(x_values)
    stationary = np.asarray(stationary, dtype=bool)
    if len(x_values) == 0 or len(stationary) == 0:
        return

    start_idx = None
    for idx, is_stationary in enumerate(stationary):
        if is_stationary and start_idx is None:
            start_idx = idx
        if start_idx is not None and (
            (not is_stationary) or idx == len(stationary) - 1
        ):
            end_idx = idx if is_stationary and idx == len(stationary) - 1 else idx - 1
            ax.axvspan(
                x_values[start_idx],
                x_values[end_idx],
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            start_idx = None


def plot_pair_stationary_speed_trace(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    speed_threshold,
    bodypart="Center",
    speed_col="mean_speed",
    timestamp_col=("timestamp", ""),
    figsize=(12, 4),
    sharey=True,
):
    pair_fig, pair_axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=sharey,
    )

    for ax, condition, behav_df_dic, color in [
        (pair_axes[0], "saline", behav_df_dic_saline, "blue"),
        (pair_axes[1], "DCZ", behav_df_dic_dcz, "red"),
    ]:
        trace_df, x_label = _speed_trace_df(
            behav_df_dic[pair_id],
            bodypart=bodypart,
            speed_col=speed_col,
            timestamp_col=timestamp_col,
        )
        stationary = trace_df["speed"] <= speed_threshold
        _shade_stationary_segments(
            ax,
            trace_df["x"],
            stationary,
            color="gray",
            alpha=0.18,
        )
        ax.plot(
            trace_df["x"],
            trace_df["speed"],
            color=color,
            linewidth=1,
            alpha=0.85,
        )
        ax.axhline(
            speed_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.set_title(f"{pair_id} {condition}")
        ax.set_xlabel(x_label)
        ax.grid(axis="y", alpha=0.25)

    pair_axes[0].set_ylabel(f"{bodypart} {speed_col} (pixels/s)")
    pair_fig.tight_layout()
    return pair_fig


def plot_paired_stationary_speed_traces(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    speed_threshold,
    bodypart="Center",
    speed_col="mean_speed",
    timestamp_col=("timestamp", ""),
):
    figures = []
    for pair_id in pair_ids:
        figures.append(
            plot_pair_stationary_speed_trace(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                speed_threshold=speed_threshold,
                bodypart=bodypart,
                speed_col=speed_col,
                timestamp_col=timestamp_col,
            )
        )
    return figures


def _speed_values(
    behav_df,
    bodypart="Center",
    speed_col="mean_speed",
):
    speed = pd.to_numeric(
        behavior_utils.get_behavior_column(behav_df, (bodypart, speed_col)),
        errors="coerce",
    )
    speed = speed.replace([np.inf, -np.inf], np.nan).dropna()
    return speed[speed >= 0]


def plot_pair_speed_distribution(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    speed_col="mean_speed",
    speed_threshold=None,
    bins=60,
    quantiles=(0.0, 0.99),
    figsize=(6, 4),
    kde=True,
):
    saline_speed = _speed_values(
        behav_df_dic_saline[pair_id],
        bodypart=bodypart,
        speed_col=speed_col,
    )
    dcz_speed = _speed_values(
        behav_df_dic_dcz[pair_id],
        bodypart=bodypart,
        speed_col=speed_col,
    )
    all_speed = pd.concat([saline_speed, dcz_speed], ignore_index=True)

    fig, ax = plt.subplots(figsize=figsize)
    if all_speed.empty:
        ax.text(
            0.5,
            0.5,
            "No valid speed values.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    x_min = all_speed.quantile(quantiles[0])
    x_max = all_speed.quantile(quantiles[1])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min = float(all_speed.min())
        x_max = float(all_speed.max()) + 1

    saline_plot = saline_speed[(saline_speed >= x_min) & (saline_speed <= x_max)]
    dcz_plot = dcz_speed[(dcz_speed >= x_min) & (dcz_speed <= x_max)]

    sns.histplot(
        saline_plot,
        bins=bins,
        binrange=(x_min, x_max),
        stat="density",
        element="step",
        fill=False,
        color="blue",
        linewidth=1.5,
        label=f"saline median={saline_speed.median():.2f}",
        ax=ax,
    )
    sns.histplot(
        dcz_plot,
        bins=bins,
        binrange=(x_min, x_max),
        stat="density",
        element="step",
        fill=False,
        color="red",
        linewidth=1.5,
        label=f"DCZ median={dcz_speed.median():.2f}",
        ax=ax,
    )

    if kde:
        if saline_plot.nunique() > 1:
            sns.kdeplot(
                saline_plot,
                color="blue",
                linewidth=2,
                ax=ax,
            )
        if dcz_plot.nunique() > 1:
            sns.kdeplot(
                dcz_plot,
                color="red",
                linewidth=2,
                ax=ax,
            )

    if speed_threshold is not None:
        ax.axvline(
            speed_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label=f"threshold={speed_threshold:g}",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"{bodypart} {speed_col} (pixels/s)")
    ax.set_ylabel("density")
    ax.set_title(f"{pair_id} speed distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_paired_speed_distributions(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    speed_col="mean_speed",
    speed_threshold=None,
    bins=60,
    quantiles=(0.0, 0.99),
    kde=True,
):
    figures = []
    for pair_id in pair_ids:
        figures.append(
            plot_pair_speed_distribution(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                bodypart=bodypart,
                speed_col=speed_col,
                speed_threshold=speed_threshold,
                bins=bins,
                quantiles=quantiles,
                kde=kde,
            )
        )
    return figures


def _speed_values_by_state(
    behav_df,
    state,
    bodypart="Center",
    speed_col="mean_speed",
    state_col=("speed_hmm_state", ""),
):
    speed = pd.to_numeric(
        behavior_utils.get_behavior_column(behav_df, (bodypart, speed_col)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    states = pd.to_numeric(
        _series_from_column(behav_df, state_col),
        errors="coerce",
    )

    state_speed = speed[
        speed.notna()
        & states.notna()
        & (states.astype(float) == float(state))
        & (speed >= 0)
    ]
    return state_speed


def plot_pair_state_speed_distribution(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    speed_col="mean_speed",
    state_col=("speed_hmm_state", ""),
    states=(0, 1),
    state_colors=None,
    speed_threshold=None,
    bins=60,
    quantiles=(0.0, 0.99),
    figsize=(10, 4),
    kde=True,
):
    state_colors = state_colors or {
        0: "#4c78a8",
        1: "#f58518",
    }

    condition_dfs = [
        ("saline", behav_df_dic_saline),
        ("DCZ", behav_df_dic_dcz),
    ]
    speed_by_condition_state = {}
    all_speed_parts = []
    for condition, behav_df_dic in condition_dfs:
        speed_by_condition_state[condition] = {}
        if pair_id not in behav_df_dic:
            continue

        for state in states:
            state_speed = _speed_values_by_state(
                behav_df_dic[pair_id],
                state=state,
                bodypart=bodypart,
                speed_col=speed_col,
                state_col=state_col,
            )
            speed_by_condition_state[condition][state] = state_speed
            all_speed_parts.append(state_speed)

    all_speed = pd.concat(
        all_speed_parts,
        ignore_index=True,
    ) if all_speed_parts else pd.Series(dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    if all_speed.empty:
        axes[0].text(
            0.5,
            0.5,
            "No valid state speed values.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        for ax in axes:
            ax.set_axis_off()
        return fig

    x_min = all_speed.quantile(quantiles[0])
    x_max = all_speed.quantile(quantiles[1])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min = float(all_speed.min())
        x_max = float(all_speed.max()) + 1

    for ax, (condition, _) in zip(axes, condition_dfs):
        condition_has_data = False
        for state in states:
            state_speed = speed_by_condition_state.get(condition, {}).get(
                state,
                pd.Series(dtype=float),
            )
            state_plot = state_speed[
                (state_speed >= x_min) & (state_speed <= x_max)
            ]
            color = state_colors.get(
                state,
                plt.get_cmap("tab10")(int(state) % 10),
            )

            if state_plot.empty:
                continue

            condition_has_data = True
            sns.histplot(
                state_plot,
                bins=bins,
                binrange=(x_min, x_max),
                stat="density",
                element="step",
                fill=False,
                color=color,
                linewidth=1.5,
                label=(
                    f"state {state} median={state_speed.median():.2f}, "
                    f"n={len(state_speed)}"
                ),
                ax=ax,
            )

            if kde and state_plot.nunique() > 1:
                sns.kdeplot(
                    state_plot,
                    color=color,
                    linewidth=2,
                    ax=ax,
                )

        if speed_threshold is not None:
            ax.axvline(
                speed_threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                label=f"threshold={speed_threshold:g}",
            )

        if not condition_has_data:
            ax.text(
                0.5,
                0.5,
                "No valid state speed values.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(f"{bodypart} {speed_col} (pixels/s)")
        ax.set_title(f"{pair_id} {condition} speed by HMM state")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("density")
    fig.tight_layout()
    return fig


def plot_paired_state_speed_distributions(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    speed_col="mean_speed",
    state_col=("speed_hmm_state", ""),
    states=(0, 1),
    state_colors=None,
    speed_threshold=None,
    bins=60,
    quantiles=(0.0, 0.99),
    kde=True,
):
    figures = []
    for pair_id in pair_ids:
        figures.append(
            plot_pair_state_speed_distribution(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                bodypart=bodypart,
                speed_col=speed_col,
                state_col=state_col,
                states=states,
                state_colors=state_colors,
                speed_threshold=speed_threshold,
                bins=bins,
                quantiles=quantiles,
                kde=kde,
            )
        )
    return figures


def _speed_series_for_acf(
    behav_df,
    bodypart="Center",
    speed_col="mean_speed",
):
    speed = pd.to_numeric(
        behavior_utils.get_behavior_column(behav_df, (bodypart, speed_col)),
        errors="coerce",
    )
    speed = speed.replace([np.inf, -np.inf], np.nan)
    speed = speed.interpolate(limit_direction="both").dropna()
    return speed.to_numpy()


def plot_pair_speed_acf(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    fps=30,
    max_lag_sec=60,
    bodypart="Center",
    speed_col="mean_speed",
    figsize=(6, 4),
):
    saline_speed = _speed_series_for_acf(
        behav_df_dic_saline[pair_id],
        bodypart=bodypart,
        speed_col=speed_col,
    )
    dcz_speed = _speed_series_for_acf(
        behav_df_dic_dcz[pair_id],
        bodypart=bodypart,
        speed_col=speed_col,
    )

    max_nlags = int(fps * max_lag_sec)
    saline_nlags = min(max_nlags, len(saline_speed) - 1)
    dcz_nlags = min(max_nlags, len(dcz_speed) - 1)
    nlags = min(saline_nlags, dcz_nlags)

    fig, ax = plt.subplots(figsize=figsize)
    if nlags < 1:
        ax.text(
            0.5,
            0.5,
            "Not enough speed samples for ACF.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    acf_saline = acf(saline_speed, nlags=nlags, fft=True, missing="drop")
    acf_dcz = acf(dcz_speed, nlags=nlags, fft=True, missing="drop")
    lag_sec = np.arange(nlags + 1) / fps

    ax.plot(lag_sec, acf_saline, label="saline", color="blue")
    ax.plot(lag_sec, acf_dcz, label="DCZ", color="red")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{pair_id} {bodypart} {speed_col} ACF")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_paired_speed_acfs(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    fps=30,
    max_lag_sec=60,
    bodypart="Center",
    speed_col="mean_speed",
):
    figures = []
    for pair_id in pair_ids:
        figures.append(
            plot_pair_speed_acf(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                fps=fps,
                max_lag_sec=max_lag_sec,
                bodypart=bodypart,
                speed_col=speed_col,
            )
        )
    return figures


def plot_speed_acfs_by_group(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    hM4Di_mice,
    hM3Dq_mice,
    fps=30,
    max_lag_sec=60,
    bodypart="Center",
    speed_col="mean_speed",
    figsize=(7, 4),
    alpha=0.35,
):
    pair_ids = list(pair_ids)

    def _plot_group(mice, group_name):
        group_pair_ids = [
            pair_id
            for pair_id in pair_ids
            if str(pair_id)[:6] in set(mice)
            and pair_id in behav_df_dic_saline
            and pair_id in behav_df_dic_dcz
        ]

        fig, ax = plt.subplots(figsize=figsize)
        if not group_pair_ids:
            ax.text(
                0.5,
                0.5,
                f"No {group_name} pairs.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig, group_pair_ids

        saline_label_added = False
        dcz_label_added = False
        for pair_id in group_pair_ids:
            saline_speed = _speed_series_for_acf(
                behav_df_dic_saline[pair_id],
                bodypart=bodypart,
                speed_col=speed_col,
            )
            dcz_speed = _speed_series_for_acf(
                behav_df_dic_dcz[pair_id],
                bodypart=bodypart,
                speed_col=speed_col,
            )

            max_nlags = int(fps * max_lag_sec)
            nlags = min(max_nlags, len(saline_speed) - 1, len(dcz_speed) - 1)
            if nlags < 1:
                continue

            lag_sec = np.arange(nlags + 1) / fps
            acf_saline = acf(saline_speed, nlags=nlags, fft=True, missing="drop")
            acf_dcz = acf(dcz_speed, nlags=nlags, fft=True, missing="drop")

            ax.plot(
                lag_sec,
                acf_saline,
                color="blue",
                alpha=alpha,
                linewidth=1,
                label="saline pairs" if not saline_label_added else None,
            )
            ax.plot(
                lag_sec,
                acf_dcz,
                color="red",
                alpha=alpha,
                linewidth=1,
                label="DCZ pairs" if not dcz_label_added else None,
            )
            saline_label_added = True
            dcz_label_added = True

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(
            f"{group_name} {bodypart} {speed_col} ACF, n pairs={len(group_pair_ids)}"
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, group_pair_ids

    hm4_fig, hm4_pair_ids = _plot_group(hM4Di_mice, "hM4Di")
    hm3_fig, hm3_pair_ids = _plot_group(hM3Dq_mice, "hM3Dq")
    return [hm4_fig, hm3_fig], hm4_pair_ids, hm3_pair_ids


def plot_speed_acf_auc_groups(
    speed_acf_auc_summary,
    hM4Di_mice,
    hM3Dq_mice,
    figsize=(8, 5),
    ylabel=None,
):
    plot_df = speed_acf_auc_summary.copy()
    if "subject" not in plot_df.columns:
        plot_df["subject"] = plot_df["pair_id"].astype(str).str[:6]

    max_lag_sec = (
        plot_df["max_lag_sec"].dropna().iloc[0]
        if "max_lag_sec" in plot_df.columns and plot_df["max_lag_sec"].notna().any()
        else 60
    )
    ylabel = ylabel or f"ACF AUC 0-{max_lag_sec:g}s"

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    def _plot_group(ax, mice, group_name):
        group_df = plot_df[
            plot_df["subject"].isin(mice)
        ].dropna(subset=["saline", "DCZ"]).copy()

        for _, row in group_df.iterrows():
            ax.plot(
                [0, 1],
                [row["saline"], row["DCZ"]],
                color="gray",
                alpha=0.4,
                linewidth=1,
            )

        ax.scatter(
            [0] * len(group_df),
            group_df["saline"],
            color="blue",
            edgecolor="black",
            label="saline",
            zorder=3,
        )
        ax.scatter(
            [1] * len(group_df),
            group_df["DCZ"],
            color="red",
            edgecolor="black",
            label="DCZ",
            zorder=3,
        )

        if len(group_df) > 0:
            try:
                p_value = stats.wilcoxon(
                    group_df["saline"],
                    group_df["DCZ"],
                ).pvalue
                p_text = f"p={p_value:.3g}"
            except ValueError:
                p_value = np.nan
                p_text = "p=NA"
        else:
            p_value = np.nan
            p_text = "p=NA"

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["saline", "DCZ"])
        ax.set_title(f"{group_name}, n={len(group_df)}, {p_text}")
        add_paired_significance_label(
            ax,
            group_df["saline"],
            group_df["DCZ"],
            p_to_star(p_value),
        )
        ax.grid(axis="y", alpha=0.3)
        return group_df

    hm4_df = _plot_group(axes[0], hM4Di_mice, "hM4Di")
    hm3_df = _plot_group(axes[1], hM3Dq_mice, "hM3Dq")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig, hm4_df, hm3_df


def _normalize_speed_trace(speed, quantiles=(0.01, 0.99)):
    speed = np.asarray(speed, dtype=float)
    normalized = np.full_like(speed, np.nan, dtype=float)
    finite = np.isfinite(speed)
    if not finite.any():
        return normalized

    vmin, vmax = np.nanquantile(speed[finite], quantiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        normalized[finite] = 0.5
        return normalized

    normalized[finite] = (speed[finite] - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


def plot_speed_hmm_posteriors_with_speed(
    df,
    hmm,
    speed_col=("Center", "mean_speed"),
    colors=None,
    ax=None,
    title=None,
    lw=2,
    speed_quantiles=(0.01, 0.99),
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 2.5), dpi=80)

    if hmm is None:
        ax.text(
            0.5,
            0.5,
            "Missing HMM model.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    if df.empty or speed_col not in df.columns:
        ax.text(
            0.5,
            0.5,
            f"Missing speed column: {speed_col}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    speed = pd.to_numeric(df[speed_col], errors="coerce")
    valid_speed = speed.replace([np.inf, -np.inf], np.nan).notna().to_numpy()
    valid_positions = np.flatnonzero(valid_speed)

    K = hmm.K
    colors = _state_colors(K, colors)
    posterior_full = np.full((len(df), K), np.nan, dtype=float)

    if len(valid_positions) > 0:
        data = speed.iloc[valid_positions].to_numpy(dtype=float).reshape(-1, 1)
        posterior = hmm.expected_states(data)[0]
        posterior_full[valid_positions] = posterior

    for state in range(K):
        ax.plot(
            posterior_full[:, state],
            label=f"State {state}",
            lw=lw,
            color=colors[state],
        )

    normalized_speed = _normalize_speed_trace(
        speed.to_numpy(dtype=float),
        quantiles=speed_quantiles,
    )
    if np.isfinite(normalized_speed).any():
        ax.plot(
            normalized_speed,
            "-k",
            label="speed (normalized)",
            alpha=0.7,
            linewidth=1,
        )

    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("frame #")
    ax.set_ylabel("p(state)")
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, ncol=max(1, K + 1), loc="upper right")
    return ax


def plot_speed_hmm_posteriors_for_dfs(
    hmm_speed_models,
    speed_col=("Center", "mean_speed"),
    colors=None,
    figsize=(20, 2.5),
    speed_quantiles=(0.01, 0.99),
):
    figures = []

    for model_name, model_info in hmm_speed_models.items():
        hmm = model_info.get("model")
        for data_info in model_info.get("data_info", []):
            pair_id = data_info.get("pair_id", "")
            condition = data_info.get("condition", "")
            df = data_info.get("df")

            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=80)
            plot_speed_hmm_posteriors_with_speed(
                df,
                hmm,
                speed_col=speed_col,
                colors=colors,
                ax=ax,
                title=f"{model_name} model | {pair_id} | {condition}",
                speed_quantiles=speed_quantiles,
            )
            fig.tight_layout()
            figures.append(fig)

    return figures


def plot_speed_hmm_posteriors_for_behavior_dicts(
    behav_df_dic_saline,
    behav_df_dic_dcz,
    hmm_effec,
    hmm_noeffec,
    speed_col=("Center", "mean_speed"),
    noeffec_subjects=("NUO005", "NUO008"),
    colors=None,
    figsize=(20, 2.5),
    speed_quantiles=(0.01, 0.99),
):
    figures = []

    for pair_id, df in behav_df_dic_saline.items():
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=80)
        plot_speed_hmm_posteriors_with_speed(
            df,
            hmm_noeffec,
            speed_col=speed_col,
            colors=colors,
            ax=ax,
            title=f"no-effect model | {pair_id} | saline",
            speed_quantiles=speed_quantiles,
        )
        fig.tight_layout()
        figures.append(fig)

    for pair_id, df in behav_df_dic_dcz.items():
        model_is_noeffect = any(
            str(subject) in str(pair_id) for subject in noeffec_subjects
        )
        hmm = hmm_noeffec if model_is_noeffect else hmm_effec
        model_name = "no-effect" if model_is_noeffect else "effect"

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=80)
        plot_speed_hmm_posteriors_with_speed(
            df,
            hmm,
            speed_col=speed_col,
            colors=colors,
            ax=ax,
            title=f"{model_name} model | {pair_id} | DCZ",
            speed_quantiles=speed_quantiles,
        )
        fig.tight_layout()
        figures.append(fig)

    return figures


def state_transition_matrix_from_df(
    df,
    state_col=("speed_hmm_state", ""),
    K=2,
):
    if df.empty or state_col not in df.columns:
        counts = np.zeros((K, K), dtype=float)
        return np.zeros((K, K), dtype=float), counts

    states = pd.to_numeric(df[state_col], errors="coerce").to_numpy()
    return state_transition_matrix_from_states(states, K=K)


def state_transition_matrix_from_states(states, K=2):
    counts = np.zeros((K, K), dtype=float)
    states = np.asarray(states, dtype=float)

    if len(states) < 2:
        return np.zeros((K, K), dtype=float), counts

    previous_states = states[:-1]
    next_states = states[1:]
    valid = np.isfinite(previous_states) & np.isfinite(next_states)

    for previous_state, next_state in zip(
        previous_states[valid].astype(int),
        next_states[valid].astype(int),
    ):
        if 0 <= previous_state < K and 0 <= next_state < K:
            counts[previous_state, next_state] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )
    return transition_matrix, counts


def infer_speed_hmm_states(
    df,
    hmm,
    speed_col=("Center", "mean_speed"),
):
    states = np.full(len(df), np.nan, dtype=float)

    if hmm is None or df.empty or speed_col not in df.columns:
        return states

    speed = pd.to_numeric(df[speed_col], errors="coerce")
    valid_speed = speed.replace([np.inf, -np.inf], np.nan).notna().to_numpy()
    valid_positions = np.flatnonzero(valid_speed)
    if len(valid_positions) == 0:
        return states

    data = speed.iloc[valid_positions].to_numpy(dtype=float).reshape(-1, 1)
    posterior = hmm.expected_states(data)[0]
    states[valid_positions] = np.argmax(posterior, axis=1)
    return states


def speed_hmm_switch_probability_from_df(
    df,
    hmm=None,
    speed_col=("Center", "mean_speed"),
    state_col=("speed_hmm_state", ""),
    K=2,
    use_existing_states=True,
):
    if use_existing_states and state_col in df.columns:
        transition_matrix, counts = state_transition_matrix_from_df(
            df,
            state_col=state_col,
            K=K,
        )
    elif hmm is not None:
        states = infer_speed_hmm_states(
            df,
            hmm,
            speed_col=speed_col,
        )
        transition_matrix, counts = state_transition_matrix_from_states(
            states,
            K=K,
        )
    else:
        transition_matrix, counts = state_transition_matrix_from_df(
            df,
            state_col=state_col,
            K=K,
        )

    if counts.sum() == 0:
        return np.nan, transition_matrix, counts

    switch_probability = transition_matrix[0, 1] + transition_matrix[1, 0]
    return switch_probability, transition_matrix, counts


def speed_hmm_switch_probability_summary(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    hmm_effec=None,
    hmm_noeffec=None,
    noeffec_subjects=("NUO005", "NUO008"),
    speed_col=("Center", "mean_speed"),
    state_col=("speed_hmm_state", ""),
    K=2,
    use_existing_states=True,
):
    rows = []

    for pair_id in pair_ids:
        if pair_id not in behav_df_dic_saline or pair_id not in behav_df_dic_dcz:
            continue

        subject = str(pair_id)[:6]
        group = (
            "noeffect"
            if any(str(noeffec_subject) in str(pair_id) for noeffec_subject in noeffec_subjects)
            else "effect"
        )
        hmm = hmm_noeffec if group == "noeffect" else hmm_effec

        saline_switch, saline_matrix, saline_counts = (
            speed_hmm_switch_probability_from_df(
                behav_df_dic_saline[pair_id],
                hmm=hmm,
                speed_col=speed_col,
                state_col=state_col,
                K=K,
                use_existing_states=use_existing_states,
            )
        )
        dcz_switch, dcz_matrix, dcz_counts = speed_hmm_switch_probability_from_df(
            behav_df_dic_dcz[pair_id],
            hmm=hmm,
            speed_col=speed_col,
            state_col=state_col,
            K=K,
            use_existing_states=use_existing_states,
        )

        rows.append(
            {
                "pair_id": pair_id,
                "subject": subject,
                "group": group,
                "saline": saline_switch,
                "DCZ": dcz_switch,
                "DCZ_minus_saline": dcz_switch - saline_switch,
                "saline_p01": saline_matrix[0, 1],
                "saline_p10": saline_matrix[1, 0],
                "DCZ_p01": dcz_matrix[0, 1],
                "DCZ_p10": dcz_matrix[1, 0],
                "saline_n_transitions": int(saline_counts.sum()),
                "DCZ_n_transitions": int(dcz_counts.sum()),
            }
        )

    return pd.DataFrame(rows)


def _transition_data_from_transition_figure(fig):
    transition_data = getattr(fig, "_speed_hmm_transition_data", None)
    if transition_data is not None:
        return transition_data

    transition_data = {}
    for ax in fig.axes:
        if not ax.images:
            continue

        title = ax.get_title().lower()
        if "saline" in title:
            condition = "saline"
        elif "dcz" in title:
            condition = "DCZ"
        else:
            continue

        transition_matrix = np.asarray(ax.images[0].get_array(), dtype=float)
        transition_data[condition] = {
            "transition_matrix": transition_matrix,
            "counts": np.full_like(transition_matrix, np.nan, dtype=float),
        }

    return transition_data


def speed_hmm_switch_probability_summary_from_transition_figures(
    transition_fig_dic,
    noeffec_subjects=("NUO005", "NUO008"),
):
    rows = []

    for pair_id, fig in transition_fig_dic.items():
        transition_data = _transition_data_from_transition_figure(fig)
        if "saline" not in transition_data or "DCZ" not in transition_data:
            continue

        saline_matrix = np.asarray(
            transition_data["saline"]["transition_matrix"],
            dtype=float,
        )
        dcz_matrix = np.asarray(
            transition_data["DCZ"]["transition_matrix"],
            dtype=float,
        )
        saline_counts = np.asarray(
            transition_data["saline"].get("counts", np.nan),
            dtype=float,
        )
        dcz_counts = np.asarray(
            transition_data["DCZ"].get("counts", np.nan),
            dtype=float,
        )

        if saline_matrix.shape[0] < 2 or dcz_matrix.shape[0] < 2:
            continue

        subject = str(pair_id)[:6]
        group = (
            "noeffect"
            if any(str(noeffec_subject) in str(pair_id) for noeffec_subject in noeffec_subjects)
            else "effect"
        )
        saline_switch = saline_matrix[0, 1] + saline_matrix[1, 0]
        dcz_switch = dcz_matrix[0, 1] + dcz_matrix[1, 0]

        rows.append(
            {
                "pair_id": pair_id,
                "subject": subject,
                "group": group,
                "saline": saline_switch,
                "DCZ": dcz_switch,
                "DCZ_minus_saline": dcz_switch - saline_switch,
                "saline_p01": saline_matrix[0, 1],
                "saline_p10": saline_matrix[1, 0],
                "DCZ_p01": dcz_matrix[0, 1],
                "DCZ_p10": dcz_matrix[1, 0],
                "saline_n_transitions": (
                    int(np.nansum(saline_counts))
                    if np.isfinite(saline_counts).any()
                    else np.nan
                ),
                "DCZ_n_transitions": (
                    int(np.nansum(dcz_counts))
                    if np.isfinite(dcz_counts).any()
                    else np.nan
                ),
                "source": "transition_fig_dic",
            }
        )

    return pd.DataFrame(rows)


def plot_speed_hmm_switch_probability_groups(
    switch_probability_summary,
    figsize=(8, 7),
):
    plot_df = switch_probability_summary.copy()
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=False)

    transition_specs = [
        ("saline_p01", "DCZ_p01", "P(0->1)"),
        ("saline_p10", "DCZ_p10", "P(1->0)"),
    ]
    group_specs = [
        ("noeffect", "no-effect group"),
        ("effect", "effect group"),
    ]

    def _plot_group_transition(
        ax,
        group_name,
        group_title,
        saline_col,
        dcz_col,
        transition_label,
    ):
        group_df = plot_df[
            plot_df["group"] == group_name
        ].dropna(subset=[saline_col, dcz_col]).copy()

        for _, row in group_df.iterrows():
            ax.plot(
                [0, 1],
                [row[saline_col], row[dcz_col]],
                color="gray",
                alpha=0.4,
                linewidth=1,
            )

        ax.scatter(
            [0] * len(group_df),
            group_df[saline_col],
            color="blue",
            edgecolor="black",
            label="saline",
            zorder=3,
        )
        ax.scatter(
            [1] * len(group_df),
            group_df[dcz_col],
            color="red",
            edgecolor="black",
            label="DCZ",
            zorder=3,
        )

        if len(group_df) > 0:
            try:
                paired_diff = group_df[saline_col] - group_df[dcz_col]
                if np.allclose(paired_diff, 0, equal_nan=False):
                    p_value = 1.0
                else:
                    p_value = stats.wilcoxon(
                        group_df[saline_col],
                        group_df[dcz_col],
                    ).pvalue
                p_text = f"p={p_value:.3g}"
            except ValueError:
                p_value = np.nan
                p_text = "p=NA"
        else:
            p_value = np.nan
            p_text = "p=NA"

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["saline", "DCZ"])
        ax.set_title(f"{group_title} {transition_label}, {p_text}")
        if len(group_df) > 0:
            add_paired_significance_label(
                ax,
                group_df[saline_col],
                group_df[dcz_col],
                p_to_star(p_value),
            )
        else:
            ax.set_ylim(bottom=0, top=1)
            ax.text(
                0.5,
                0.5,
                "No paired data.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.grid(axis="y", alpha=0.3)
        return group_df

    plotted_group_dfs = {}
    for row_idx, (group_name, group_title) in enumerate(group_specs):
        for col_idx, (saline_col, dcz_col, transition_label) in enumerate(
            transition_specs
        ):
            group_df = _plot_group_transition(
                axes[row_idx, col_idx],
                group_name,
                group_title,
                saline_col,
                dcz_col,
                transition_label,
            )
            plotted_group_dfs.setdefault(group_name, group_df)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("transition probability")

    axes[0, -1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, plotted_group_dfs["noeffect"], plotted_group_dfs["effect"]


def plot_speed_hmm_switch_probability_comparison(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    hmm_effec=None,
    hmm_noeffec=None,
    noeffec_subjects=("NUO005", "NUO008"),
    speed_col=("Center", "mean_speed"),
    state_col=("speed_hmm_state", ""),
    K=2,
    figsize=(8, 7),
    use_existing_states=True,
    transition_fig_dic=None,
):
    if transition_fig_dic is not None:
        summary = speed_hmm_switch_probability_summary_from_transition_figures(
            transition_fig_dic,
            noeffec_subjects=noeffec_subjects,
        )
    else:
        summary = speed_hmm_switch_probability_summary(
            pair_ids=pair_ids,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            hmm_effec=hmm_effec,
            hmm_noeffec=hmm_noeffec,
            noeffec_subjects=noeffec_subjects,
            speed_col=speed_col,
            state_col=state_col,
            K=K,
            use_existing_states=use_existing_states,
        )
    fig, noeffect_df, effect_df = plot_speed_hmm_switch_probability_groups(
        summary,
        figsize=figsize,
    )
    return fig, summary, noeffect_df, effect_df


def speed_hmm_state_fraction_from_df(
    df,
    state_col=("speed_hmm_state", ""),
    K=2,
):
    states = pd.to_numeric(
        _series_from_column(df, state_col),
        errors="coerce",
    ).to_numpy(dtype=float)
    states = states[np.isfinite(states)]

    state_counts = np.zeros(K, dtype=int)
    state_fractions = np.full(K, np.nan, dtype=float)
    if len(states) == 0:
        return state_fractions, state_counts

    for state in range(K):
        state_counts[state] = int(np.sum(states == state))
    state_fractions = state_counts / len(states)
    return state_fractions, state_counts


def speed_hmm_state_fraction_summary(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    state_col=("speed_hmm_state", ""),
    K=2,
    noeffec_subjects=("NUO005", "NUO008"),
):
    rows = []

    for pair_id in pair_ids:
        if pair_id not in behav_df_dic_saline or pair_id not in behav_df_dic_dcz:
            continue

        subject = str(pair_id)[:6]
        group = (
            "noeffect"
            if any(str(noeffec_subject) in str(pair_id) for noeffec_subject in noeffec_subjects)
            else "effect"
        )

        saline_fractions, saline_counts = speed_hmm_state_fraction_from_df(
            behav_df_dic_saline[pair_id],
            state_col=state_col,
            K=K,
        )
        dcz_fractions, dcz_counts = speed_hmm_state_fraction_from_df(
            behav_df_dic_dcz[pair_id],
            state_col=state_col,
            K=K,
        )

        row = {
            "pair_id": pair_id,
            "subject": subject,
            "group": group,
            "saline_n_state_frames": int(saline_counts.sum()),
            "DCZ_n_state_frames": int(dcz_counts.sum()),
        }
        for state in range(K):
            row[f"saline_state{state}_fraction"] = saline_fractions[state]
            row[f"DCZ_state{state}_fraction"] = dcz_fractions[state]
            row[f"saline_state{state}_count"] = int(saline_counts[state])
            row[f"DCZ_state{state}_count"] = int(dcz_counts[state])
            row[f"state{state}_DCZ_minus_saline"] = (
                dcz_fractions[state] - saline_fractions[state]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_speed_hmm_state_fraction_groups(
    state_fraction_summary,
    states=(0, 1),
    figsize=(8, 7),
):
    plot_df = state_fraction_summary.copy()
    fig, axes = plt.subplots(
        2,
        len(states),
        figsize=figsize,
        sharey=False,
        squeeze=False,
    )

    group_specs = [
        ("noeffect", "no-effect group"),
        ("effect", "effect group"),
    ]

    def _plot_group_state(ax, group_name, group_title, state):
        saline_col = f"saline_state{state}_fraction"
        dcz_col = f"DCZ_state{state}_fraction"
        group_df = plot_df[
            plot_df["group"] == group_name
        ].dropna(subset=[saline_col, dcz_col]).copy()

        for _, row in group_df.iterrows():
            ax.plot(
                [0, 1],
                [row[saline_col], row[dcz_col]],
                color="gray",
                alpha=0.4,
                linewidth=1,
            )

        ax.scatter(
            [0] * len(group_df),
            group_df[saline_col],
            color="blue",
            edgecolor="black",
            label="saline",
            zorder=3,
        )
        ax.scatter(
            [1] * len(group_df),
            group_df[dcz_col],
            color="red",
            edgecolor="black",
            label="DCZ",
            zorder=3,
        )

        if len(group_df) > 0:
            try:
                paired_diff = group_df[saline_col] - group_df[dcz_col]
                if np.allclose(paired_diff, 0, equal_nan=False):
                    p_value = 1.0
                else:
                    p_value = stats.wilcoxon(
                        group_df[saline_col],
                        group_df[dcz_col],
                    ).pvalue
                p_text = f"p={p_value:.3g}"
            except ValueError:
                p_value = np.nan
                p_text = "p=NA"
        else:
            p_value = np.nan
            p_text = "p=NA"

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["saline", "DCZ"])
        ax.set_title(f"{group_title} state {state} fraction, {p_text}")
        if len(group_df) > 0:
            add_paired_significance_label(
                ax,
                group_df[saline_col],
                group_df[dcz_col],
                p_to_star(p_value),
            )
        else:
            ax.set_ylim(bottom=0, top=1)
            ax.text(
                0.5,
                0.5,
                "No paired data.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_ylim(bottom=0, top=min(max(ax.get_ylim()[1], 1), 1.2))
        ax.grid(axis="y", alpha=0.3)
        return group_df

    plotted_group_dfs = {}
    for row_idx, (group_name, group_title) in enumerate(group_specs):
        for col_idx, state in enumerate(states):
            group_df = _plot_group_state(
                axes[row_idx, col_idx],
                group_name,
                group_title,
                state,
            )
            plotted_group_dfs.setdefault(group_name, group_df)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("fraction of state frames")

    axes[0, -1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, plotted_group_dfs["noeffect"], plotted_group_dfs["effect"]


def plot_speed_hmm_state_fraction_comparison(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    state_col=("speed_hmm_state", ""),
    K=2,
    states=None,
    noeffec_subjects=("NUO005", "NUO008"),
    figsize=(8, 7),
):
    if states is None:
        states = tuple(range(K))

    summary = speed_hmm_state_fraction_summary(
        pair_ids=pair_ids,
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        state_col=state_col,
        K=K,
        noeffec_subjects=noeffec_subjects,
    )
    fig, noeffect_df, effect_df = plot_speed_hmm_state_fraction_groups(
        summary,
        states=states,
        figsize=figsize,
    )
    return fig, summary, noeffect_df, effect_df


def _series_from_column(df, column):
    if isinstance(column, tuple) and column in df.columns:
        return df[column]

    if isinstance(column, str):
        tuple_column = (column, "")
        if tuple_column in df.columns:
            return df[tuple_column]
        if column in df.columns and not isinstance(df.columns, pd.MultiIndex):
            return df[column]
        if isinstance(df.columns, pd.MultiIndex) and column in df.columns.get_level_values(0):
            column_df = df[column]
            if isinstance(column_df, pd.Series):
                return column_df
            if column_df.shape[1] == 1:
                return column_df.iloc[:, 0]

    if column in df.columns:
        return df[column]

    raise KeyError(f"{column!r} not found in dataframe columns")


def get_switch_rate_from_states(states, fps=None):
    states = np.asarray(states, dtype=float)
    if len(states) < 2:
        return np.nan

    previous_states = states[:-1]
    next_states = states[1:]
    valid_transition = np.isfinite(previous_states) & np.isfinite(next_states)
    if not valid_transition.any():
        return np.nan

    n_switches = np.sum(
        previous_states[valid_transition] != next_states[valid_transition]
    )

    if fps is None:
        return n_switches / valid_transition.sum()

    valid_frames = np.isfinite(states).sum()
    if valid_frames == 0:
        return np.nan
    return n_switches / (valid_frames / fps)


def get_state_switch_rates_from_states(states, fps=None, K=2):
    states = np.asarray(states, dtype=float)
    metrics = {}

    if len(states) < 2:
        for state in range(K):
            metrics[f"state{state}_switch_rate"] = np.nan
            metrics[f"state{state}_n_switches"] = 0
            metrics[f"state{state}_n_frames"] = 0
        return metrics

    valid_state = np.isfinite(states)
    previous_states = states[:-1]
    next_states = states[1:]
    valid_transition = np.isfinite(previous_states) & np.isfinite(next_states)

    for state in range(K):
        state_frames = valid_state & (states == state)
        state_transitions = valid_transition & (previous_states == state)
        n_state_switches = np.sum(
            state_transitions & (next_states != previous_states)
        )

        if fps is None:
            denominator = state_transitions.sum()
        else:
            denominator = state_frames.sum() / fps

        if denominator > 0:
            state_switch_rate = n_state_switches / denominator
        else:
            state_switch_rate = np.nan

        metrics[f"state{state}_switch_rate"] = state_switch_rate
        metrics[f"state{state}_n_switches"] = int(n_state_switches)
        metrics[f"state{state}_n_frames"] = int(state_frames.sum())

    return metrics


def speed_hmm_switch_rate_from_df(
    df,
    fps=30,
    state_col=("speed_hmm_state", ""),
    K=2,
):
    states = pd.to_numeric(
        _series_from_column(df, state_col),
        errors="coerce",
    ).to_numpy()

    previous_states = states[:-1]
    next_states = states[1:]
    valid_transition = np.isfinite(previous_states) & np.isfinite(next_states)
    n_switches = int(
        np.sum(previous_states[valid_transition] != next_states[valid_transition])
    )
    valid_frames = int(np.isfinite(states).sum())

    metrics = {
        "switch_rate": get_switch_rate_from_states(states, fps=fps),
        "n_switches": n_switches,
        "n_frames": valid_frames,
        "n_valid_transitions": int(valid_transition.sum()),
    }
    if fps is not None:
        metrics["duration_sec"] = valid_frames / fps

    metrics.update(
        get_state_switch_rates_from_states(
            states,
            fps=fps,
            K=K,
        )
    )
    return metrics


def speed_hmm_switch_rate_summary(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    fps=30,
    state_col=("speed_hmm_state", ""),
    K=2,
    noeffec_subjects=("NUO005", "NUO008"),
):
    rows = []

    for pair_id in pair_ids:
        subject = str(pair_id)[:6]
        group = (
            "noeffect"
            if any(str(noeffec_subject) in str(pair_id) for noeffec_subject in noeffec_subjects)
            else "effect"
        )

        for condition, behav_df_dic in [
            ("saline", behav_df_dic_saline),
            ("DCZ", behav_df_dic_dcz),
        ]:
            if pair_id not in behav_df_dic:
                continue

            try:
                metrics = speed_hmm_switch_rate_from_df(
                    behav_df_dic[pair_id],
                    fps=fps,
                    state_col=state_col,
                    K=K,
                )
            except KeyError:
                continue

            rows.append(
                {
                    "pair_id": pair_id,
                    "subject": subject,
                    "group": group,
                    "condition": condition,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def plot_speed_hmm_switch_rate_distribution(
    switch_rate_summary,
    states=(0, 1),
    figsize=None,
    condition_order=("saline", "DCZ"),
    state_colors=None,
    bar_width=0.36,
    alpha=0.75,
):
    plot_df = switch_rate_summary.copy()
    state_cols = [f"state{state}_switch_rate" for state in states]
    for state_col in state_cols:
        if state_col in plot_df.columns:
            plot_df[state_col] = pd.to_numeric(
                plot_df[state_col],
                errors="coerce",
            )

    condition_order = list(condition_order)
    condition_rank = {
        condition: rank for rank, condition in enumerate(condition_order)
    }
    state_colors = state_colors or {
        0: "#4c78a8",
        1: "#f58518",
    }
    condition_tick_colors = {"saline": "blue", "DCZ": "red"}
    group_order = [
        ("noeffect", "no-effect group"),
        ("effect", "effect group"),
    ]

    if figsize is None:
        max_group_sessions = max(
            [
                len(plot_df[plot_df["group"] == group_name])
                for group_name, _ in group_order
            ]
            or [1]
        )
        figsize = (max(10, max_group_sessions * 0.55), 4.8)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, (group_name, group_title) in zip(axes, group_order):
        group_df = plot_df[plot_df["group"] == group_name].copy()

        if group_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No {group_title} data.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group_title)
            ax.set_ylim(bottom=0)
            continue

        group_df["_condition_rank"] = group_df["condition"].map(
            condition_rank
        ).fillna(len(condition_rank))
        group_df = group_df.sort_values(
            ["pair_id", "_condition_rank", "condition"]
        ).reset_index(drop=True)

        x_positions = np.arange(len(group_df))
        n_states = len(states)
        total_bar_width = min(0.8, bar_width * n_states)
        offsets = np.linspace(
            -total_bar_width / 2 + total_bar_width / (2 * n_states),
            total_bar_width / 2 - total_bar_width / (2 * n_states),
            n_states,
        )

        for state, offset in zip(states, offsets):
            state_col = f"state{state}_switch_rate"
            if state_col not in group_df.columns:
                continue

            ax.bar(
                x_positions + offset,
                group_df[state_col],
                width=total_bar_width / n_states,
                color=state_colors.get(state, plt.get_cmap("tab10")(state)),
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
                label=f"state {state}",
                align="center",
            )

        session_labels = (
            group_df["pair_id"].astype(str)
            + "\n"
            + group_df["condition"].astype(str)
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(session_labels, rotation=90, fontsize=7)
        for tick_label, condition in zip(
            ax.get_xticklabels(),
            group_df["condition"],
        ):
            tick_label.set_color(
                condition_tick_colors.get(str(condition), "black")
            )

        ax.set_title(group_title)
        ax.set_xlabel("session")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, title="HMM state")

    axes[0].set_ylabel("switch rate (switches/s)")
    axes[1].set_ylabel("")
    fig.tight_layout()
    return fig, plot_df


def plot_speed_hmm_state_switch_rate_groups(
    switch_rate_summary,
    states=(0, 1),
    figsize=(9, 6),
    sharey=False,
):
    plot_df = switch_rate_summary.copy()
    condition_order = ["saline", "DCZ"]
    colors = {"saline": "blue", "DCZ": "red"}
    group_order = [
        ("noeffect", "no-effect group"),
        ("effect", "effect group"),
    ]

    fig, axes = plt.subplots(
        len(group_order),
        len(states),
        figsize=figsize,
        sharey=False,
        squeeze=False,
    )

    for row_idx, (group_name, group_title) in enumerate(group_order):
        for col_idx, state in enumerate(states):
            ax = axes[row_idx, col_idx]
            value_col = f"state{state}_switch_rate"
            group_df = plot_df[
                plot_df["group"] == group_name
            ].copy()
            group_df[value_col] = pd.to_numeric(
                group_df[value_col],
                errors="coerce",
            )

            paired_df = (
                group_df.pivot_table(
                    index="pair_id",
                    columns="condition",
                    values=value_col,
                    aggfunc="mean",
                )
                .reindex(columns=condition_order)
                .dropna()
            )

            for _, pair_row in paired_df.iterrows():
                ax.plot(
                    [0, 1],
                    [pair_row["saline"], pair_row["DCZ"]],
                    color="gray",
                    alpha=0.4,
                    linewidth=1,
                )

            ax.scatter(
                [0] * len(paired_df),
                paired_df["saline"],
                color=colors["saline"],
                edgecolor="black",
                label="saline",
                zorder=3,
            )
            ax.scatter(
                [1] * len(paired_df),
                paired_df["DCZ"],
                color=colors["DCZ"],
                edgecolor="black",
                label="DCZ",
                zorder=3,
            )

            if len(paired_df) > 0:
                try:
                    paired_diff = paired_df["saline"] - paired_df["DCZ"]
                    if np.allclose(paired_diff, 0, equal_nan=False):
                        p_value = 1.0
                    else:
                        p_value = stats.wilcoxon(
                            paired_df["saline"],
                            paired_df["DCZ"],
                        ).pvalue
                except ValueError:
                    p_value = np.nan
                add_paired_significance_label(
                    ax,
                    paired_df["saline"],
                    paired_df["DCZ"],
                    p_to_star(p_value),
                )
            else:
                ax.set_ylim(bottom=0, top=1e-6)
                ax.text(
                    0.5,
                    0.5,
                    "No paired data.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(condition_order)
            ax.set_title(f"{group_title} state {state}")
            ax.grid(axis="y", alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel("state exit rate (switches/s)")

    if sharey:
        y_max = max(ax.get_ylim()[1] for ax in axes.ravel())
        for ax in axes.ravel():
            ax.set_ylim(bottom=0, top=max(y_max, 1e-6))

    axes[0, -1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, plot_df


def plot_state_transition_matrix(
    transition_matrix,
    counts=None,
    title="",
    ax=None,
    cmap="gray",
    fontsize=8,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    im = ax.imshow(
        transition_matrix,
        vmin=0,
        vmax=1,
        aspect="auto",
        cmap=cmap,
    )

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("State t")
    ax.set_ylabel("State t-1")

    K = transition_matrix.shape[0]
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"State {state}" for state in range(K)])
    ax.set_yticklabels([f"State {state}" for state in range(K)])

    for i in range(K):
        for j in range(K):
            val = transition_matrix[i, j]
            label = f"{val:.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="black" if val > 0.5 else "white",
                fontsize=fontsize,
            )

    return im


def plot_pair_speed_hmm_transition_matrices(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    state_col=("speed_hmm_state", ""),
    K=2,
    figsize=(7, 3),
    cmap="gray",
):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    transition_data = {}

    for ax, condition, behav_df_dic in [
        (axes[0], "saline", behav_df_dic_saline),
        (axes[1], "DCZ", behav_df_dic_dcz),
    ]:
        if pair_id not in behav_df_dic:
            ax.text(
                0.5,
                0.5,
                f"Missing {condition}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        transition_matrix, counts = state_transition_matrix_from_df(
            behav_df_dic[pair_id],
            state_col=state_col,
            K=K,
        )
        transition_data[condition] = {
            "transition_matrix": transition_matrix.copy(),
            "counts": counts.copy(),
        }
        plot_state_transition_matrix(
            transition_matrix,
            counts=counts,
            title=f"{pair_id} {condition}",
            ax=ax,
            cmap=cmap,
        )

    fig._speed_hmm_pair_id = pair_id
    fig._speed_hmm_transition_data = transition_data
    fig.tight_layout()
    return fig


def plot_paired_speed_hmm_transition_matrices(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    state_col=("speed_hmm_state", ""),
    K=2,
    figsize=(7, 3),
    cmap="gray",
):
    fig_dic = {}
    for pair_id in pair_ids:
        fig_dic[pair_id] = plot_pair_speed_hmm_transition_matrices(
            pair_id=pair_id,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            state_col=state_col,
            K=K,
            figsize=figsize,
            cmap=cmap,
        )
    return fig_dic


def _transition_matrix_model(transition_matrix):
    class _Transitions:
        pass

    class _Model:
        pass

    model = _Model()
    model.transitions = _Transitions()
    model.transitions.transition_matrix = np.asarray(transition_matrix, dtype=float)
    model.K = model.transitions.transition_matrix.shape[0]
    return model


def plot_pair_speed_hmm_transition_graphs(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    state_col=("speed_hmm_state", ""),
    K=2,
    figsize=(7, 3),
    colors=None,
    fontsize=8,
):
    import utils_test

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    transition_data = {}

    for ax, condition, behav_df_dic in [
        (axes[0], "saline", behav_df_dic_saline),
        (axes[1], "DCZ", behav_df_dic_dcz),
    ]:
        if pair_id not in behav_df_dic:
            ax.text(
                0.5,
                0.5,
                f"Missing {condition}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        transition_matrix, counts = state_transition_matrix_from_df(
            behav_df_dic[pair_id],
            state_col=state_col,
            K=K,
        )
        transition_data[condition] = {
            "transition_matrix": transition_matrix.copy(),
            "counts": counts.copy(),
        }
        transition_model = _transition_matrix_model(transition_matrix)
        utils_test.plot_transition_graph(
            transition_model,
            ax=ax,
            colors=colors,
            fontsize=fontsize,
        )
        ax.set_title(f"{pair_id} {condition}", fontsize=fontsize)

    fig._speed_hmm_pair_id = pair_id
    fig._speed_hmm_transition_data = transition_data
    fig.tight_layout()
    return fig


def plot_paired_speed_hmm_transition_graphs(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    state_col=("speed_hmm_state", ""),
    K=2,
    figsize=(7, 3),
    colors=None,
    fontsize=8,
):
    fig_dic = {}
    for pair_id in pair_ids:
        fig_dic[pair_id] = plot_pair_speed_hmm_transition_graphs(
            pair_id=pair_id,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            state_col=state_col,
            K=K,
            figsize=figsize,
            colors=colors,
            fontsize=fontsize,
        )
    return fig_dic


def speed_hmm_state_xy_heatmap(
    df,
    state,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    x_col="x",
    y_col="y",
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
):
    heatmap = np.zeros((xbins, ybins), dtype=float)

    required_cols = [(bodypart, x_col), (bodypart, y_col), state_col]
    if df.empty or any(column not in df.columns for column in required_cols):
        return heatmap

    x = pd.to_numeric(df[(bodypart, x_col)], errors="coerce").to_numpy()
    y = pd.to_numeric(df[(bodypart, y_col)], errors="coerce").to_numpy()
    states = pd.to_numeric(df[state_col], errors="coerce").to_numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(states)
        & (states.astype(float) == float(state))
    )
    if not valid.any():
        return heatmap

    x_min, x_max, y_min, y_max = extent
    heatmap, _, _ = np.histogram2d(
        x[valid],
        y[valid],
        bins=[xbins, ybins],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    if normalize and heatmap.sum() > 0:
        heatmap = heatmap / heatmap.sum()

    return heatmap


def speed_hmm_state_occupancy_map(
    df,
    state,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    timestamp_col=("timestamp", ""),
    x_col="x",
    y_col="y",
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
):
    occupancy = np.zeros((xbins, ybins), dtype=float)

    required_cols = [(bodypart, x_col), (bodypart, y_col), state_col, timestamp_col]
    if df.empty or any(column not in df.columns for column in required_cols):
        return occupancy

    x = pd.to_numeric(df[(bodypart, x_col)], errors="coerce").to_numpy()
    y = pd.to_numeric(df[(bodypart, y_col)], errors="coerce").to_numpy()
    states = pd.to_numeric(_series_from_column(df, state_col), errors="coerce").to_numpy()
    t = pd.to_numeric(_series_from_column(df, timestamp_col), errors="coerce").to_numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(states)
        & np.isfinite(t)
        & (states.astype(float) == float(state))
    )
    if valid.sum() < 2:
        return occupancy

    if extent is None:
        occupancy = behavior_utils.occupancy_map(
            x[valid],
            y[valid],
            t[valid],
            xbins=xbins,
            ybins=ybins,
        )
    else:
        x_min, x_max, y_min, y_max = extent
        t_state = t[valid]
        t_next = np.append(t_state, t_state[-1] + np.median(np.diff(t_state)))
        time_in_bin = np.diff(t_next)
        occupancy, _, _ = np.histogram2d(
            x[valid],
            y[valid],
            bins=[xbins, ybins],
            range=[[x_min, x_max], [y_min, y_max]],
            weights=time_in_bin,
        )

    if normalize and occupancy.sum() > 0:
        occupancy = occupancy / occupancy.sum()

    return occupancy


def plot_pair_speed_hmm_state_occupancy_diff_heatmaps(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    timestamp_col=("timestamp", ""),
    states=(0, 1),
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
    cmap="coolwarm",
    figsize=None,
    interpolation="nearest",
    diff_quantile=0.99,
):
    occupancy_maps = {}
    for state in states:
        for condition, behav_df_dic in [
            ("saline", behav_df_dic_saline),
            ("DCZ", behav_df_dic_dcz),
        ]:
            if pair_id not in behav_df_dic:
                occupancy = np.zeros((xbins, ybins), dtype=float)
            else:
                occupancy = speed_hmm_state_occupancy_map(
                    behav_df_dic[pair_id],
                    state=state,
                    bodypart=bodypart,
                    state_col=state_col,
                    timestamp_col=timestamp_col,
                    xbins=xbins,
                    ybins=ybins,
                    extent=extent,
                    normalize=normalize,
                )
            occupancy_maps[(state, condition)] = occupancy

    diff_maps = {
        state: occupancy_maps[(state, "DCZ")] - occupancy_maps[(state, "saline")]
        for state in states
    }
    diff_values = np.concatenate([diff_maps[state].ravel() for state in states])
    diff_values = diff_values[np.isfinite(diff_values)]
    max_abs = (
        np.nanquantile(np.abs(diff_values), diff_quantile)
        if len(diff_values) > 0
        else 1
    )
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1

    diff_norm = mpl.colors.TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs,
    )

    if figsize is None:
        figsize = (4 * len(states), 4)
    fig, axes = plt.subplots(
        1,
        len(states),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    if extent is None:
        plot_extent = None
        x_min = x_max = y_min = y_max = None
    else:
        plot_extent = extent
        x_min, x_max, y_min, y_max = extent

    last_im = None
    for col_idx, state in enumerate(states):
        ax = axes[0, col_idx]
        imshow_kwargs = {
            "origin": "lower",
            "cmap": cmap,
            "norm": diff_norm,
            "interpolation": interpolation,
        }
        if plot_extent is not None:
            imshow_kwargs["extent"] = plot_extent

        last_im = ax.imshow(diff_maps[state].T, **imshow_kwargs)
        ax.set_title(f"{pair_id} state {state} occupancy DCZ - saline")
        ax.set_aspect("equal")
        if plot_extent is not None:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].invert_yaxis()
    colorbar_label = (
        "DCZ - saline fraction of state occupancy time / bin"
        if normalize
        else "DCZ - saline state occupancy time / bin"
    )
    fig.colorbar(
        last_im,
        ax=axes,
        label=colorbar_label,
        shrink=0.75,
        fraction=0.04,
        pad=0.02,
        extend="both",
    )
    fig.tight_layout()
    return fig


def plot_paired_speed_hmm_state_occupancy_diff_heatmaps(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    timestamp_col=("timestamp", ""),
    states=(0, 1),
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
    cmap="coolwarm",
    figsize=None,
    interpolation="nearest",
    diff_quantile=0.99,
):
    fig_dic = {}
    for pair_id in pair_ids:
        fig_dic[pair_id] = plot_pair_speed_hmm_state_occupancy_diff_heatmaps(
            pair_id=pair_id,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            bodypart=bodypart,
            state_col=state_col,
            timestamp_col=timestamp_col,
            states=states,
            xbins=xbins,
            ybins=ybins,
            extent=extent,
            normalize=normalize,
            cmap=cmap,
            figsize=figsize,
            interpolation=interpolation,
            diff_quantile=diff_quantile,
        )
    return fig_dic


def plot_pair_speed_hmm_condition_state_occupancy_diff_heatmaps(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    timestamp_col=("timestamp", ""),
    state_a=1,
    state_b=0,
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
    cmap="coolwarm",
    figsize=None,
    interpolation="nearest",
    diff_quantile=0.99,
):
    condition_specs = [
        ("saline", behav_df_dic_saline),
        ("DCZ", behav_df_dic_dcz),
    ]

    diff_maps = {}
    for condition, behav_df_dic in condition_specs:
        if pair_id not in behav_df_dic:
            diff_maps[condition] = np.zeros((xbins, ybins), dtype=float)
            continue

        occupancy_a = speed_hmm_state_occupancy_map(
            behav_df_dic[pair_id],
            state=state_a,
            bodypart=bodypart,
            state_col=state_col,
            timestamp_col=timestamp_col,
            xbins=xbins,
            ybins=ybins,
            extent=extent,
            normalize=normalize,
        )
        occupancy_b = speed_hmm_state_occupancy_map(
            behav_df_dic[pair_id],
            state=state_b,
            bodypart=bodypart,
            state_col=state_col,
            timestamp_col=timestamp_col,
            xbins=xbins,
            ybins=ybins,
            extent=extent,
            normalize=normalize,
        )
        diff_maps[condition] = occupancy_a - occupancy_b

    diff_values = np.concatenate(
        [diff_maps[condition].ravel() for condition, _ in condition_specs]
    )
    diff_values = diff_values[np.isfinite(diff_values)]
    max_abs = (
        np.nanquantile(np.abs(diff_values), diff_quantile)
        if len(diff_values) > 0
        else 1
    )
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1

    diff_norm = mpl.colors.TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs,
    )

    if figsize is None:
        figsize = (8, 4)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    if extent is None:
        plot_extent = None
        x_min = x_max = y_min = y_max = None
    else:
        plot_extent = extent
        x_min, x_max, y_min, y_max = extent

    last_im = None
    for col_idx, (condition, _) in enumerate(condition_specs):
        ax = axes[0, col_idx]
        imshow_kwargs = {
            "origin": "lower",
            "cmap": cmap,
            "norm": diff_norm,
            "interpolation": interpolation,
        }
        if plot_extent is not None:
            imshow_kwargs["extent"] = plot_extent

        last_im = ax.imshow(diff_maps[condition].T, **imshow_kwargs)
        ax.set_title(
            f"{pair_id} {condition} occupancy state {state_a} - state {state_b}"
        )
        ax.set_aspect("equal")
        if plot_extent is not None:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].invert_yaxis()
    colorbar_label = (
        f"state {state_a} - state {state_b} fraction of occupancy time / bin"
        if normalize
        else f"state {state_a} - state {state_b} occupancy time / bin"
    )
    fig.colorbar(
        last_im,
        ax=axes,
        label=colorbar_label,
        shrink=0.75,
        fraction=0.04,
        pad=0.02,
        extend="both",
    )
    fig.tight_layout()
    return fig


def plot_paired_speed_hmm_condition_state_occupancy_diff_heatmaps(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    timestamp_col=("timestamp", ""),
    state_a=1,
    state_b=0,
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
    cmap="coolwarm",
    figsize=None,
    interpolation="nearest",
    diff_quantile=0.99,
):
    fig_dic = {}
    for pair_id in pair_ids:
        fig_dic[pair_id] = (
            plot_pair_speed_hmm_condition_state_occupancy_diff_heatmaps(
                pair_id=pair_id,
                behav_df_dic_saline=behav_df_dic_saline,
                behav_df_dic_dcz=behav_df_dic_dcz,
                bodypart=bodypart,
                state_col=state_col,
                timestamp_col=timestamp_col,
                state_a=state_a,
                state_b=state_b,
                xbins=xbins,
                ybins=ybins,
                extent=extent,
                normalize=normalize,
                cmap=cmap,
                figsize=figsize,
                interpolation=interpolation,
                diff_quantile=diff_quantile,
            )
        )
    return fig_dic


def plot_pair_speed_hmm_state_xy_heatmaps(
    pair_id,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    states=(0, 1),
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
    cmap=None,
    figsize=None,
    interpolation="nearest",
    difference=True,
    diff_quantile=0.99,
):
    cmap = cmap or ("coolwarm" if difference else "magma")

    conditions = [
        ("saline", behav_df_dic_saline),
        ("DCZ", behav_df_dic_dcz),
    ]

    heatmaps = {}
    heatmap_values = []
    for state in states:
        for condition, behav_df_dic in conditions:
            if pair_id not in behav_df_dic:
                heatmap = np.zeros((xbins, ybins), dtype=float)
            else:
                heatmap = speed_hmm_state_xy_heatmap(
                    behav_df_dic[pair_id],
                    state=state,
                    bodypart=bodypart,
                    state_col=state_col,
                    xbins=xbins,
                    ybins=ybins,
                    extent=extent,
                    normalize=normalize,
                )
            heatmaps[(state, condition)] = heatmap
            heatmap_values.append(heatmap.ravel())

    if difference:
        diff_heatmaps = {
            state: heatmaps[(state, "DCZ")] - heatmaps[(state, "saline")]
            for state in states
        }
        diff_values = np.concatenate(
            [diff_heatmaps[state].ravel() for state in states]
        )
        diff_values = diff_values[np.isfinite(diff_values)]
        max_abs = (
            np.nanquantile(np.abs(diff_values), diff_quantile)
            if len(diff_values) > 0
            else 1
        )
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1
        heatmap_norm = mpl.colors.TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs,
        )

        if figsize is None:
            figsize = (4 * len(states), 4)
        fig, axes = plt.subplots(
            1,
            len(states),
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        x_min, x_max, y_min, y_max = extent
        last_im = None
        for col_idx, state in enumerate(states):
            ax = axes[0, col_idx]
            last_im = ax.imshow(
                diff_heatmaps[state].T,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=heatmap_norm,
                interpolation=interpolation,
            )

            ax.set_title(f"{pair_id} state {state} DCZ - saline")
            ax.set_aspect("equal")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].invert_yaxis()
        colorbar_label = (
            "DCZ - saline fraction of state frames / bin"
            if normalize
            else "DCZ - saline state frame count / bin"
        )
        fig.colorbar(
            last_im,
            ax=axes,
            label=colorbar_label,
            shrink=0.75,
            fraction=0.04,
            pad=0.02,
            extend="both",
        )
        fig.tight_layout()
        return fig

    heatmap_values = np.concatenate(heatmap_values)
    heatmap_norm = _normalize_from_values(
        heatmap_values,
        quantiles=(0.0, 0.99),
        default=(0, 1),
    )

    if figsize is None:
        figsize = (8, 7)
    fig, axes = plt.subplots(
        len(states),
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_min, x_max, y_min, y_max = extent
    last_im = None
    for row_idx, state in enumerate(states):
        for col_idx, (condition, behav_df_dic) in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            heatmap = heatmaps[(state, condition)]
            last_im = ax.imshow(
                heatmap.T,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=heatmap_norm,
                interpolation=interpolation,
            )

            ax.set_title(f"{pair_id} {condition} state {state}")
            ax.set_aspect("equal")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].invert_yaxis()
    colorbar_label = (
        "fraction of state frames / bin" if normalize else "state frame count / bin"
    )
    fig.colorbar(
        last_im,
        ax=axes,
        label=colorbar_label,
        shrink=0.75,
        fraction=0.04,
        pad=0.02,
        extend="max",
    )
    return fig


def plot_paired_speed_hmm_state_xy_heatmaps(
    pair_ids,
    behav_df_dic_saline,
    behav_df_dic_dcz,
    bodypart="Center",
    state_col=("speed_hmm_state", ""),
    states=(0, 1),
    xbins=40,
    ybins=40,
    extent=(0, 640, 0, 480),
    normalize=True,
    cmap=None,
    figsize=None,
    interpolation="nearest",
    difference=True,
    diff_quantile=0.99,
):
    fig_dic = {}
    for pair_id in pair_ids:
        fig_dic[pair_id] = plot_pair_speed_hmm_state_xy_heatmaps(
            pair_id=pair_id,
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            bodypart=bodypart,
            state_col=state_col,
            states=states,
            xbins=xbins,
            ybins=ybins,
            extent=extent,
            normalize=normalize,
            cmap=cmap,
            figsize=figsize,
            interpolation=interpolation,
            difference=difference,
            diff_quantile=diff_quantile,
        )
    return fig_dic
