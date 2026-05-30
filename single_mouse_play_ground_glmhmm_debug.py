import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import lecilab_behavior_analysis.utils as utils
    import lecilab_behavior_analysis.plots as plots
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import lecilab_behavior_analysis.df_transforms as dft
    from sklearn.linear_model import LogisticRegression
    import seaborn as sns
    import statsmodels.api as sm
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.preprocessing import MinMaxScaler

    import utils_test

    plots.psychometric_plot = utils_test.psychometric_plot_easy_logistic

    import autograd.numpy.random as npr
    # npr.seed(0)
    import ssm
    from ssm.util import find_permutation
    from ssm.plots import gradient_cmap, white_to_color_cmap

    from scipy.optimize import minimize

    # %load_ext autoreload
    # %autoreload 2
    return Path, dft, minimize, np, pd, plots, plt, sns, utils, utils_test


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data import
    """)
    return


@app.cell
def _(utils):
    # load data from cluster
    tv_projects = utils.get_server_projects()
    print(tv_projects)
    return


@app.cell
def _():
    # project = "visual_and_COT_data"
    project = "COT_cannula_data"
    return (project,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mouse select
    """)
    return


@app.cell
def _(mo, project, utils):
    animals = utils.get_animals_in_project(project)

    default_mice = animals[:-1]

    mouse_select = mo.ui.multiselect(
        options=animals,
        value=default_mice,
        label="Mice"
    )

    mouse_select
    return (mouse_select,)


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Download / update this mouse")
    download_button
    return (download_button,)


@app.cell
def _(Path, dft, download_button, mouse_select, np, pd, project, utils):
    def parameters_for_fit_if_nonempty(df_stage):
        if df_stage.empty:
            return pd.DataFrame()
        return dft.parameters_for_fit(df_stage)

    selected_mice = list(mouse_select.value)
    parameters_output_dir = (
        Path(utils.get_outpath())
        / project
        / "processed"
        / "parameters_for_fit"
        / "auto"
    )

    stage_columns = {
        "df_test_vis_easy": "TwoAFC_visual_easy",
        "df_test_aud_easy": "TwoAFC_auditory_easy",
        "df_test_vis_hard": "TwoAFC_visual_hard",
        "df_test_aud_hard": "TwoAFC_auditory_hard",
    }
    empty_results = {
        output_name: pd.DataFrame() for output_name in stage_columns
    }

    def load_or_compute_mouse_parameters(_mouse):
        parameters_output_path = (
            parameters_output_dir / f"{_mouse}_parameters_for_fit.pkl"
        )
        if parameters_output_path.exists() and not download_button.value:
            print(f"Loaded saved parameters_for_fit data: {parameters_output_path}")
            return pd.read_pickle(parameters_output_path)

        _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
        _local_path.mkdir(parents=True, exist_ok=True)

        _csv_path = _local_path / f"{_mouse}.csv"

        if download_button.value or not _csv_path.exists():
            utils.rsync_cluster_data(
                project_name=project,
                file_path=f"sessions/{_mouse}/{_mouse}.csv",
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )

        if not _csv_path.exists():
            print(f"No CSV found for {_mouse}; skipped parameters_for_fit.")
            return empty_results.copy()

        _df_mouse = pd.read_csv(_csv_path, sep=";")
        if "run_mode" not in _df_mouse.columns:
            print(f"No run_mode column for {_mouse}; skipped Auto parameters_for_fit.")
            computed_results = empty_results.copy()
        else:
            _df_auto = _df_mouse[_df_mouse["run_mode"] == "Auto"]
            if _df_auto.empty or "current_training_stage" not in _df_auto.columns:
                computed_results = empty_results.copy()
                print(f"No Auto stage data found for {_mouse}; skipped parameters_for_fit.")
            else:
                computed_results = {}
                for output_name, stage_name in stage_columns.items():
                    df_stage = _df_auto[
                        _df_auto["current_training_stage"] == stage_name
                    ]
                    computed_results[output_name] = (
                        parameters_for_fit_if_nonempty(df_stage)
                    )

        parameters_output_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(computed_results, parameters_output_path)
        print(f"Saved Auto parameters_for_fit data: {parameters_output_path}")
        return computed_results

    combined_results = {output_name: [] for output_name in stage_columns}
    for _mouse in selected_mice:
        mouse_results = load_or_compute_mouse_parameters(_mouse)
        for output_name in stage_columns:
            df_mouse_result = mouse_results.get(output_name, pd.DataFrame())
            if df_mouse_result.empty:
                continue
            df_mouse_result = df_mouse_result.copy()
            if "subject" not in df_mouse_result.columns:
                df_mouse_result["subject"] = _mouse
            combined_results[output_name].append(df_mouse_result)

    df_test_vis_easy = (
        pd.concat(combined_results["df_test_vis_easy"], ignore_index=True)
        if combined_results["df_test_vis_easy"]
        else pd.DataFrame()
    )
    df_test_aud_easy = (
        pd.concat(combined_results["df_test_aud_easy"], ignore_index=True)
        if combined_results["df_test_aud_easy"]
        else pd.DataFrame()
    )
    df_test_vis_hard = (
        pd.concat(combined_results["df_test_vis_hard"], ignore_index=True)
        if combined_results["df_test_vis_hard"]
        else pd.DataFrame()
    )
    df_test_aud_hard = (
        pd.concat(combined_results["df_test_aud_hard"], ignore_index=True)
        if combined_results["df_test_aud_hard"]
        else pd.DataFrame()
    )


    # Data clean, hard stimulus in some easy sessions
    col = "visual_stimulus_ratio"

    valid_mask = (
        df_test_vis_easy[col].isna()
        | np.isclose(df_test_vis_easy[col], 6)
        | np.isclose(df_test_vis_easy[col], -6)
    )

    bad_mouse_sessions = df_test_vis_easy.loc[
        ~valid_mask,
        ["subject", "session"],
    ].drop_duplicates()

    df_test_vis_easy = df_test_vis_easy.merge(
        bad_mouse_sessions.assign(_bad_session=True),
        on=["subject", "session"],
        how="left",
    )

    df_test_vis_easy = df_test_vis_easy[
        df_test_vis_easy["_bad_session"].isna()
    ].drop(columns="_bad_session").copy()
    return (
        df_test_aud_easy,
        df_test_aud_hard,
        df_test_vis_easy,
        df_test_vis_hard,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stage select
    """)
    return


@app.cell
def _(
    df_test_aud_easy,
    df_test_aud_hard,
    df_test_vis_easy,
    df_test_vis_hard,
    mo,
):
    df_options = {
        "aud_easy": {
            "df": df_test_aud_easy,
            "stim_col": "total_evidence_strength",
            "modality": "auditory"
        },
        "aud_hard": {
            "df": df_test_aud_hard,
            "stim_col": "total_evidence_strength",
            "modality": "auditory"
        },
        "vis_easy": {
            "df": df_test_vis_easy,
            "stim_col": "visual_stimulus_ratio",
            "modality": "visual"
        },
        "vis_hard": {
            "df": df_test_vis_hard,
            "stim_col": "visual_stimulus_ratio",
            "modality": "visual"
        },
    }

    df_selector = mo.ui.multiselect(
        options=list(df_options.keys()),
        value=["aud_easy", "aud_hard", "vis_easy", "vis_hard"]
    )
    df_selector
    return df_options, df_selector


@app.cell
def _(df_options, df_selector, pd):
    selected_options = list(df_selector.value)
    selected_dfs = []

    for option_name in selected_options:
        selected = df_options[option_name]
        selected_df = selected["df"].copy()
        if selected_df.empty:
            continue
        selected_df["selected_df_option"] = option_name
        selected_df["session"] = selected_df["session"].astype(str) + "__" + option_name
        selected_df["model_stimulus"] = selected_df[selected["stim_col"]]
        selected_dfs.append(selected_df)

    if selected_dfs:
        df_test = pd.concat(selected_dfs, ignore_index=True)
    else:
        df_test = pd.DataFrame()

    stim_col = "model_stimulus"
    df_test
    return df_test, stim_col


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # performance
    """)
    return


@app.cell
def _(df_test, dft, pd, plt, sns):
    mouse_col = "analysis_mouse" if "analysis_mouse" in df_test.columns else "subject"

    df_test_date = dft.add_day_column_to_df(df_test)

    daily_performance_df = (
        df_test_date.groupby(
            [mouse_col, "year_month_day", "current_training_stage"],
            as_index=False,
        )["correct"]
        .mean()
    )

    daily_performance_df["year_month_day"] = pd.to_datetime(
        daily_performance_df["year_month_day"],
        errors="coerce",
    )

    daily_performance_df = daily_performance_df.dropna(
        subset=["year_month_day"]
    ).sort_values([mouse_col, "year_month_day"])

    first_day = daily_performance_df["year_month_day"].min()
    end_day = first_day + pd.DateOffset(days=75)

    daily_performance_first_month = daily_performance_df[
        (daily_performance_df["year_month_day"] >= first_day)
        & (daily_performance_df["year_month_day"] < end_day)
    ].copy()

    plt.figure(figsize=(10, 4), dpi=150)

    sns.lineplot(
        data=daily_performance_first_month,
        x="year_month_day",
        y="correct",
        hue="current_training_stage",
        units=mouse_col,
        estimator=None,
        marker="",
        alpha=0.6,
    )

    plt.axhline(0.75, linestyle="--", color="gray", linewidth=1, alpha=0.8)

    plt.ylim(0, 1)
    plt.xlabel("Day")
    plt.ylabel("Performance")
    plt.title("Daily performance by training stage")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Training stage", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    daily_performance_first_month
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # psychometric curve
    """)
    return


@app.cell
def _(df_test, plt, sns, stim_col, utils_test):
    for stage_name__, df_stage__ in df_test.groupby("selected_df_option"):
        plt.figure(figsize=(8, 8), dpi=150)

        if "vis" in stage_name__:
            # x_col = "visual_stimulus_ratio"
            valueType = "discrete"
        else:
            # x_col = "total_evidence_strength"
            valueType = "continue"

        if "easy" in stage_name__:
            bins = 2
        else:
            bins = 6


        for mouse_name_psycurve, df_mouse_psycurve in df_stage__.groupby('subject'):
            color_palette = sns.color_palette(
                "colorblind",
                df_stage__['subject'].nunique()
            )

        for (mouse_name_psycurve, df_mouse_psycurve), color in zip(
            df_stage__.groupby('subject'),
            sns.color_palette("colorblind", df_stage__['subject'].nunique())
        ):
            utils_test.psychometric_plot_easy_logistic(
                df_mouse_psycurve,
                x=stim_col,
                y="first_choice_numeric",
                valueType = valueType,
                bins = bins,
                point_kwargs={
                    "color": color,
                    "label": "",
                    "alpha": 0.3,
                },
                line_kwargs={
                    "color": color,
                    "label": mouse_name_psycurve,
                    "alpha": 0.3,
                },
            )

        utils_test.psychometric_plot_easy_logistic(
            df_stage__,
            x=stim_col,
            valueType = valueType,
            bins = bins,
            y="first_choice_numeric",
            point_kwargs={
                "color": "black",
                "label": "",
            },
            line_kwargs={
                "color": "black",
                "label": "All mice",
            },
        )

        plt.title(stage_name__)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # get action trace and stimulus trace
    """)
    return


@app.cell
def _(df_test, pd, stim_col, utils, utils_test):
    n_lags = 5

    source_col = "first_choice_numeric"

    if df_test.empty or stim_col not in df_test.columns:
        logit_model_lags_params = pd.Series(dtype=float)
    else:
        df_lag = utils_test.add_lag_features(
            df_test,
            source_col=source_col,
            n_lags=n_lags,
            group_col="session"
        )

        lag_cols = [f"first_choice_numeric_lag{i}" for i in range(1, n_lags + 1)]

        fit_df = df_lag.dropna(subset=lag_cols + ["first_choice_numeric", stim_col]).copy()

        if fit_df.empty:
            logit_model_lags_params = pd.Series(dtype=float)
        else:
            _, logit_model_lags = utils.logi_model_fit(fit_df, 
                                    X=[stim_col] + [i for i in lag_cols],
                                    y='first_choice_numeric',)
            logit_model_lags_params = logit_model_lags.params
    return


@app.cell
def _(df_test, pd, stim_col, utils, utils_test):
    _n_lags = 5
    _source_col = "first_choice_numeric"
    _lag_cols = [
        f"{_source_col}_lag{_lag}" for _lag in range(1, _n_lags + 1)
    ]
    _model_x_order = [stim_col] + _lag_cols
    logit_lag_x_order = ["bias"] + _model_x_order
    _mouse_col = None

    if "analysis_mouse" in df_test.columns:
        _mouse_col = "analysis_mouse"
    elif "subject" in df_test.columns:
        _mouse_col = "subject"

    _rows = []
    if (
        not df_test.empty
        and _mouse_col is not None
        and stim_col in df_test.columns
        and _source_col in df_test.columns
    ):
        for _mouse in sorted(df_test[_mouse_col].dropna().astype(str).unique()):
            _df_mouse = df_test[df_test[_mouse_col].astype(str) == _mouse].copy()
            if _df_mouse.empty:
                continue

            _df_lag = utils_test.add_lag_features(
                _df_mouse,
                source_col=_source_col,
                n_lags=_n_lags,
                group_col="session",
            )
            _fit_df = _df_lag.dropna(
                subset=_lag_cols + [_source_col, stim_col]
            ).copy()

            if _fit_df.empty:
                _status = "no valid trials after lagging"
                _params = pd.Series(dtype=float)
            else:
                try:
                    _, _model = utils.logi_model_fit(
                        _fit_df,
                        X=_model_x_order,
                        y=_source_col,
                    )
                    _params = _model.params
                    _status = "ok"
                except Exception as _exc:
                    _params = pd.Series(dtype=float)
                    _status = f"failed: {_exc}"

            _row = {"mouse": _mouse, "status": _status, "n_trials": len(_fit_df)}
            _row["bias"] = _params.get("const", pd.NA)
            for _x_name in _model_x_order:
                _row[_x_name] = _params.get(_x_name, pd.NA)
            _rows.append(_row)

    per_mouse_logit_lags_params = pd.DataFrame(_rows)
    return logit_lag_x_order, per_mouse_logit_lags_params


@app.cell
def _(logit_lag_x_order, pd, per_mouse_logit_lags_params):
    if per_mouse_logit_lags_params.empty:
        per_mouse_logit_lags_long_df = pd.DataFrame(
            columns=["mouse", "status", "n_trials", "glm_x", "weight"]
        )
    else:
        per_mouse_logit_lags_long_df = per_mouse_logit_lags_params.melt(
            id_vars=["mouse", "status", "n_trials"],
            value_vars=logit_lag_x_order,
            var_name="glm_x",
            value_name="weight",
        )
        per_mouse_logit_lags_long_df["weight"] = pd.to_numeric(
            per_mouse_logit_lags_long_df["weight"],
            errors="coerce",
        )
        per_mouse_logit_lags_long_df = per_mouse_logit_lags_long_df.dropna(
            subset=["weight"]
        )
    return (per_mouse_logit_lags_long_df,)


@app.cell
def _(
    Path,
    df_selector,
    logit_lag_x_order,
    mo,
    np,
    per_mouse_logit_lags_long_df,
    plt,
):
    if per_mouse_logit_lags_long_df.empty:
        per_mouse_logit_lags_plot = mo.md(
            "No per-mouse logit lag weights to plot for the current selection."
        )
    else:
        _stage_label = "_".join(list(df_selector.value)) or "no_stage"
        _plot_title = f"Per-mouse logit lag weights {_stage_label}"
        _fig, _ax = plt.subplots(figsize=(10, 5))
        _x_positions = np.arange(len(logit_lag_x_order))
        _box_values = [
            per_mouse_logit_lags_long_df[
                per_mouse_logit_lags_long_df["glm_x"] == _x_name
            ]["weight"].values
            for _x_name in logit_lag_x_order
        ]
        _ax.boxplot(
            _box_values,
            positions=_x_positions,
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9e6f2", "edgecolor": "#4f6f8f"},
            medianprops={"color": "#1f2933", "linewidth": 1.5},
            whiskerprops={"color": "#4f6f8f"},
            capprops={"color": "#4f6f8f"},
        )

        _mice = sorted(per_mouse_logit_lags_long_df["mouse"].unique())
        _offsets = np.linspace(-0.18, 0.18, len(_mice)) if len(_mice) > 1 else [0]

        for _mouse, _offset in zip(_mice, _offsets):
            _mouse_df = per_mouse_logit_lags_long_df[
                per_mouse_logit_lags_long_df["mouse"] == _mouse
            ]
            _weights_by_x = _mouse_df.set_index("glm_x")["weight"].to_dict()
            _ys = [_weights_by_x.get(_x_name, np.nan) for _x_name in logit_lag_x_order]
            _xs = _x_positions + _offset
            _valid = ~np.isnan(_ys)
            _ax.plot(
                _xs[_valid],
                np.array(_ys)[_valid],
                alpha=0.45,
                linewidth=1,
                marker="o",
                markersize=4,
                label=_mouse,
            )
            _ax.scatter(_xs[_valid], np.array(_ys)[_valid], s=28, alpha=0.8)

        _ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        _ax.set_xticks(_x_positions)
        _ax.set_xticklabels(logit_lag_x_order, rotation=35, ha="right")
        _ax.set_xlabel("GLM X")
        _ax.set_ylabel("weight")
        _ax.set_title(_plot_title)
        if len(_mice) <= 12:
            _ax.legend(title="mouse", bbox_to_anchor=(1.02, 1), loc="upper left")
        _fig.tight_layout()
        per_mouse_logit_lags_plot = _fig
        _save_dir = Path(r"E:\data\LeciLab\behavioral_data\tmp")
        if not _save_dir.is_absolute() and str(_save_dir).startswith("E:\\"):
            _save_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp")
        _save_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _save_dir / f"{_plot_title}.svg"
        _fig.savefig(_save_path, format="svg", bbox_inches="tight")
        print(f"Saved per-mouse logit lag plot: {_save_path}")

    per_mouse_logit_lags_plot
    return


@app.cell
def _(np):
    v0 = np.array([0.0, 1.0, 1.0, 0.0, -2.0])
    return (v0,)


@app.cell
def _():
    # df_copy = df_test.copy()
    # if (
    #     df_copy.empty
    #     or 'first_choice_numeric' not in df_copy.columns
    #     or 'correct_side_numeric' not in df_copy.columns
    # ):
    #     df_clean = df_copy
    # else:
    #     df_clean = df_copy.dropna(subset=['first_choice_numeric', 'correct_side_numeric'])
    # if df_clean.empty:
    #     theta_stim = {}
    #     theta_act = {}
    # else:
    #     res_stim = minimize(utils_test.neg_loglik_reg, v0, args=(df_clean, "stim"), method="L-BFGS-B")
    #     res_act = minimize(utils_test.neg_loglik_reg, v0, args=(df_clean, "act"), method="L-BFGS-B")

    #     theta_stim = utils_test.unpack_params(res_stim.x)
    #     theta_act = utils_test.unpack_params(res_act.x)
    return


@app.cell
def _():
    # print(theta_act)
    # print(theta_stim)
    return


@app.cell
def _(
    Path,
    df_options,
    download_button,
    minimize,
    mo,
    mouse_select,
    pd,
    plt,
    utils_test,
    v0,
):
    _stage_order = list(df_options.keys())
    _selected_mice = list(mouse_select.value)
    _saved_label = "__".join(_selected_mice) if _selected_mice else "no_mouse"
    _alpha_output_dir = Path(r"E:\data\LeciLab\behavioral_data\tmp\processing")
    if not _alpha_output_dir.is_absolute() and str(_alpha_output_dir).startswith("E:\\"):
        _alpha_output_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp/processing")
    _alpha_act_path = _alpha_output_dir / f"{_saved_label}_per_mouse_stage_alpha_df_act.pkl"
    _alpha_stim_path = _alpha_output_dir / f"{_saved_label}_per_mouse_stage_alpha_df_stim.pkl"

    def _make_alpha_plot(alpha_df, alpha_col, title, ylabel):
        if alpha_df.empty:
            return mo.md(f"No {title.lower()} values to compare.")

        _plot_df = alpha_df.dropna(subset=[alpha_col]).copy()
        if _plot_df.empty:
            return mo.md(f"No valid {title.lower()} values to compare.")

        _available_stages = set(_plot_df["stage"].astype(str).unique())
        _stages = [
            _stage for _stage in _stage_order if _stage in _available_stages
        ]
        _stages.extend(sorted(_available_stages - set(_stages)))
        _stage_positions = {stage: i for i, stage in enumerate(_stages)}
        _fig, _ax = plt.subplots(figsize=(8, 5))

        for _mouse, _df_mouse in _plot_df.groupby("mouse", sort=True):
            _df_mouse = _df_mouse.copy()
            _df_mouse["stage"] = _df_mouse["stage"].astype(str)
            _df_mouse = _df_mouse.sort_values(
                "stage",
                key=lambda _s: _s.map(_stage_positions),
            )
            _xs = [_stage_positions[_stage] for _stage in _df_mouse["stage"]]
            _ys = _df_mouse[alpha_col].astype(float).values
            _ax.plot(
                _xs,
                _ys,
                marker="o",
                linewidth=1.2,
                alpha=0.75,
                label=_mouse,
            )

        _ax.set_xticks(range(len(_stages)))
        _ax.set_xticklabels(_stages, rotation=35, ha="right")
        _ax.set_ylim(0, 1.1)
        _ax.set_xlabel("stage")
        _ax.set_ylabel(ylabel)
        _ax.set_title(title)
        if _plot_df["mouse"].nunique() <= 12:
            _ax.legend(title="mouse", bbox_to_anchor=(1.02, 1), loc="upper left")
        _fig.tight_layout()
        return _fig

    if (
        _alpha_act_path.exists()
        and _alpha_stim_path.exists()
        and not download_button.value
    ):
        per_mouse_stage_alpha_df_act = pd.read_pickle(_alpha_act_path)
        per_mouse_stage_alpha_df_stim = pd.read_pickle(_alpha_stim_path)
        print(
            "Loaded saved per-mouse stage alpha data: "
            f"{_alpha_act_path}, {_alpha_stim_path}"
        )
    else:
        _all_stage_dfs = []
        for _stage, _stage_info in df_options.items():
            _df_stage = _stage_info["df"].copy()
            if _df_stage.empty:
                continue
            _df_stage["selected_df_option"] = _stage
            _all_stage_dfs.append(_df_stage)

        if _all_stage_dfs:
            _alpha_df = pd.concat(_all_stage_dfs, ignore_index=True)
        else:
            _alpha_df = pd.DataFrame()

        _mouse_col = None
        if "analysis_mouse" in _alpha_df.columns:
            _mouse_col = "analysis_mouse"
        elif "subject" in _alpha_df.columns:
            _mouse_col = "subject"

        _required_cols = {
            "first_choice_numeric",
            "correct_side_numeric",
            "selected_df_option",
        }

        def _fit_alpha_by_mouse_stage(prior_type, alpha_col):
            _rows = []
            if (
                _alpha_df.empty
                or _mouse_col is None
                or not _required_cols.issubset(_alpha_df.columns)
            ):
                return pd.DataFrame(_rows)

            for (_mouse, _stage), _df_group in _alpha_df.groupby(
                [_mouse_col, "selected_df_option"],
                sort=True,
            ):
                _df_clean = _df_group.dropna(
                    subset=["first_choice_numeric", "correct_side_numeric"]
                ).copy()

                if _df_clean.empty:
                    _rows.append(
                        {
                            "mouse": _mouse,
                            "stage": _stage,
                            "n_trials": 0,
                            alpha_col: pd.NA,
                            "status": "no valid trials",
                        }
                    )
                    continue

                try:
                    _res = minimize(
                        utils_test.neg_loglik_reg,
                        v0,
                        args=(_df_clean, prior_type),
                        method="L-BFGS-B",
                    )
                    _theta = utils_test.unpack_params(_res.x)
                    _rows.append(
                        {
                            "mouse": _mouse,
                            "stage": _stage,
                            "n_trials": len(_df_clean),
                            alpha_col: _theta["alpha"],
                            "status": "ok" if _res.success else _res.message,
                        }
                    )
                except Exception as _exc:
                    _rows.append(
                        {
                            "mouse": _mouse,
                            "stage": _stage,
                            "n_trials": len(_df_clean),
                            alpha_col: pd.NA,
                            "status": f"failed: {_exc}",
                        }
                    )

            return pd.DataFrame(_rows)

        per_mouse_stage_alpha_df_act = _fit_alpha_by_mouse_stage(
            prior_type="act",
            alpha_col="alpha_act",
        )
        per_mouse_stage_alpha_df_stim = _fit_alpha_by_mouse_stage(
            prior_type="stim",
            alpha_col="alpha_stim",
        )
        _alpha_output_dir.mkdir(parents=True, exist_ok=True)
        per_mouse_stage_alpha_df_act.to_pickle(_alpha_act_path)
        per_mouse_stage_alpha_df_stim.to_pickle(_alpha_stim_path)
        print(
            "Saved per-mouse stage alpha data: "
            f"{_alpha_act_path}, {_alpha_stim_path}"
        )

    per_mouse_stage_alpha_plot_act = _make_alpha_plot(
        per_mouse_stage_alpha_df_act,
        "alpha_act",
        "Per-mouse action kernel alpha by stage",
        "action kernel alpha",
    )
    per_mouse_stage_alpha_plot_stim = _make_alpha_plot(
        per_mouse_stage_alpha_df_stim,
        "alpha_stim",
        "Per-mouse stimulus kernel alpha by stage",
        "stimulus kernel alpha",
    )

    mo.vstack([per_mouse_stage_alpha_plot_act, per_mouse_stage_alpha_plot_stim])
    return per_mouse_stage_alpha_df_act, per_mouse_stage_alpha_df_stim


@app.cell
def _(
    df_test,
    pd,
    per_mouse_stage_alpha_df_act,
    per_mouse_stage_alpha_df_stim,
    utils_test,
):
    df_test_kernel = pd.DataFrame([])
    _mouse_col = None
    if "analysis_mouse" in df_test.columns:
        _mouse_col = "analysis_mouse"
    elif "subject" in df_test.columns:
        _mouse_col = "subject"

    _required_cols = {
        "session",
        "selected_df_option",
        "first_choice_numeric",
        "correct_side_numeric",
    }

    def _alpha_lookup(alpha_df, alpha_col):
        _required_alpha_cols = {"mouse", "stage", alpha_col}
        if alpha_df.empty or not _required_alpha_cols.issubset(alpha_df.columns):
            return {}
        _alpha_df = alpha_df.dropna(subset=[alpha_col]).copy()
        _alpha_df["mouse"] = _alpha_df["mouse"].astype(str)
        _alpha_df["stage"] = _alpha_df["stage"].astype(str)
        return (
            _alpha_df.set_index(["mouse", "stage"])[alpha_col]
            .astype(float)
            .to_dict()
        )

    if (
        not df_test.empty
        and _mouse_col is not None
        and _required_cols.issubset(df_test.columns)
    ):
        _alpha_act_lookup = _alpha_lookup(
            per_mouse_stage_alpha_df_act,
            "alpha_act",
        )
        _alpha_stim_lookup = _alpha_lookup(
            per_mouse_stage_alpha_df_stim,
            "alpha_stim",
        )

        _kernel_dfs = []
        _missing_alpha_keys = set()
        for (_mouse, _stage, _session), df_test_session in df_test.groupby(
            [_mouse_col, "selected_df_option", "session"],
            sort=False,
        ):
            _lookup_key = (str(_mouse), str(_stage))
            _alpha_act = _alpha_act_lookup.get(_lookup_key)
            _alpha_stim = _alpha_stim_lookup.get(_lookup_key)
            if (
                _alpha_act is None
                or _alpha_stim is None
                or pd.isna(_alpha_act)
                or pd.isna(_alpha_stim)
            ):
                _missing_alpha_keys.add(_lookup_key)
                continue

            df_test_session = df_test_session.copy()
            df_test_session["action_trace"] = utils_test.recursive_kernel_prior(
                df_test_session["first_choice_numeric"].values,
                _alpha_act,
            )
            df_test_session["stimulus_trace"] = utils_test.recursive_kernel_prior(
                df_test_session["correct_side_numeric"].values,
                _alpha_stim,
            )
            _kernel_dfs.append(df_test_session)

        if _kernel_dfs:
            df_test_kernel = pd.concat(_kernel_dfs, ignore_index=True)
        if _missing_alpha_keys:
            print(f"Skipped sessions with missing alpha: {sorted(_missing_alpha_keys)}")
    return (df_test_kernel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare glm
    """)
    return


@app.cell
def _():
    import contextlib
    import io

    return contextlib, io


@app.cell
def _(contextlib, df_test_kernel, io, mo, plots, plt, stim_col, utils_test):


    fig_glm_filter = []

    for stage_name_glm_compare, df_stage_glm_compare in df_test_kernel.groupby("selected_df_option"):
        if "vis" in stage_name_glm_compare:
            X_filter_model = [
                stim_col,
                "visual_ratio_diff_interact",
                # "visual_ratio_bright_interact",
                "left_bright",
                'visual_stimulus_diff',
                "action_trace",
                # "previous_right_choice_wrong_numeric",
                "previous_left_choice_correct_numeric",
                "stimulus_trace",
            ]
        else:
            X_filter_model = [
                stim_col,
                "left_tones_amplitude_sum",
                "amplitude_strength_left_right",
                "action_trace",
                # "previous_right_choice_wrong_numeric",
                "previous_left_choice_correct_numeric",
                "stimulus_trace",
            ]

        with contextlib.redirect_stdout(io.StringIO()):
            corr_mat_list, norm_contribution_df = utils_test.filter_variables_for_model(
                df_fit=df_stage_glm_compare,
                X=X_filter_model,
                y="first_choice_numeric",
            )

        if corr_mat_list and not norm_contribution_df.empty:
            plots.plot_filter_model_variables(
                corr_mat_list=corr_mat_list,
                norm_contribution_df=norm_contribution_df,
            )
            fig_glm_filter_ = plt.gcf()
            fig_glm_filter_.suptitle(stage_name_glm_compare)
            fig_glm_filter.append(fig_glm_filter_)
            plt.close(fig_glm_filter_)

    mo.vstack(fig_glm_filter)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # glm-hmm
    """)
    return


@app.cell
def _():
    glmhmm_cache = {}
    return (glmhmm_cache,)


@app.cell
def _(mo):
    glmhmm_run_button = mo.ui.run_button(label="Run GLM-HMM comparison")
    glmhmm_num_states = mo.ui.number(
        start=1,
        stop=6,
        step=1,
        value=3,
        label="Number of states",
    )
    glmhmm_n_iters = mo.ui.number(
        start=5,
        stop=150,
        step=5,
        value=50,
        label="EM iterations",
    )
    mo.vstack([glmhmm_run_button, glmhmm_num_states, glmhmm_n_iters])
    return glmhmm_n_iters, glmhmm_num_states, glmhmm_run_button


@app.cell
def _(
    Path,
    df_selector,
    df_test_kernel,
    glmhmm_cache,
    glmhmm_n_iters,
    glmhmm_num_states,
    glmhmm_run_button,
    mo,
    mouse_select,
    pd,
    plt,
    stim_col,
    utils_test,
):


    cache_key = (
        tuple(mouse_select.value),
        tuple(df_selector.value),
        stim_col,
        int(glmhmm_num_states.value),
        int(glmhmm_n_iters.value),
    )
    cache_label = "_".join(tuple(mouse_select.value) + tuple(df_selector.value))
    cached_result = glmhmm_cache.get(cache_key)

    if cached_result is not None and not glmhmm_run_button.value:
        glmhmm_results = cached_result["results"]
        glmhmm_output = mo.vstack(
            [
                mo.md(f"Using cached GLM-HMM result for `{cache_key}`."),
                glmhmm_results,
                cached_result["figure"],
                mo.download(
                    data=glmhmm_results.to_csv(index=False).encode("utf-8"),
                    filename=f"glmhmm_{cache_label}.csv",
                    mimetype="text/csv",
                    label="Download cached GLM-HMM results",
                ),
            ]
        )
    elif not glmhmm_run_button.value:
        glmhmm_results = pd.DataFrame(
            [
                {
                    "status": (
                        "Click Run GLM-HMM comparison to fit the models. "
                        "Previously run configurations will appear from cache."
                    )
                }
            ]
        )
        glmhmm_output = glmhmm_results
    elif df_test_kernel.empty or stim_col not in df_test_kernel.columns:
        glmhmm_results = pd.DataFrame(
            [
                {
                    "status": (
                        "No valid data for the selected mice / df options. "
                        "Try another mouse or condition."
                    )
                }
            ]
        )
        glmhmm_output = glmhmm_results
    else:
        base_inputs = [stim_col, "bias", "action_trace", "stimulus_trace"]
        model_specs = [("all", base_inputs)]
        model_specs.extend(
            (
                f"drop {col}",
                [input_col for input_col in base_inputs if input_col != col],
            )
            for col in base_inputs
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        result_rows = []

        for model_names, model_inputs in model_specs:
            datas, inpts = utils_test.build_glmhmm_inputs_by_session(
                df_test_kernel,
                y_col="first_choice_numeric",
                stim_col=model_inputs,
            )

            valid_pairs = [
                (data, inpt)
                for data, inpt in zip(datas, inpts)
                if len(data) > 0 and len(inpt) > 0
            ]
            if not valid_pairs:
                result_rows.append(
                    {
                        "model": model_names,
                        "inputs": ", ".join(model_inputs),
                        "input_dim": 0,
                        "log_likelihood": None,
                        "n_sessions": 0,
                        "status": "no valid sessions",
                    }
                )
                continue

            datas, inpts = zip(*valid_pairs)
            datas = list(datas)
            inpts = list(inpts)
            input_dim = inpts[0].shape[1]

            try:
                _, ll, hmm_lls = utils_test.fit_glmhmm(
                    datas,
                    inpts,
                    num_states=int(glmhmm_num_states.value),
                    obs_dim=1,
                    input_dim=input_dim,
                    num_categories=2,
                    N_iters=int(glmhmm_n_iters.value),
                )
                ax.plot(hmm_lls, label=model_names)
                result_rows.append(
                    {
                        "model": model_names,
                        "inputs": ", ".join(model_inputs),
                        "input_dim": input_dim,
                        "log_likelihood": ll,
                        "n_sessions": len(datas),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                result_rows.append(
                    {
                        "model": model_names,
                        "inputs": ", ".join(model_inputs),
                        "input_dim": input_dim,
                        "log_likelihood": None,
                        "n_sessions": len(datas),
                        "status": f"failed: {exc}",
                    }
                )

        _selected_df_label = "_".join(map(str, df_selector.value)) or "no_stage"
        _plot_title = f"GLM-HMM convergence {_selected_df_label}"
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("log probability")
        ax.set_title(_plot_title)
        ax.legend()
        fig.tight_layout()

        _save_dir = Path(r"E:\data\LeciLab\behavioral_data\tmp")
        if not _save_dir.is_absolute() and str(_save_dir).startswith("E:\\"):
            _save_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp")
        _save_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _save_dir / f"{_plot_title}.svg"
        fig.savefig(_save_path, format="svg", bbox_inches="tight")
        print(f"Saved per-mouse GLM-HMM convergence plot: {_save_path}")

        glmhmm_results = pd.DataFrame(result_rows)
        glmhmm_cache[cache_key] = {
            "results": glmhmm_results.copy(),
            "figure": fig,
        }
        glmhmm_output = mo.vstack(
            [
                mo.md(f"Computed and cached GLM-HMM result for `{cache_key}`."),
                glmhmm_results,
                fig,
                mo.download(
                    data=glmhmm_results.to_csv(index=False).encode("utf-8"),
                    filename=f"glmhmm_{cache_label}.csv",
                    mimetype="text/csv",
                    label="Download GLM-HMM results",
                ),
            ]
        )


    glmhmm_output
    return


@app.cell
def _():
    input_cols_noStim = ["bias", "action_trace", "stimulus_trace"]
    return (input_cols_noStim,)


@app.cell
def _(
    df_test_kernel,
    glmhmm_n_iters,
    glmhmm_num_states,
    input_cols_noStim,
    pd,
    stim_col,
    utils_test,
):
    hmm_model_dic = {}
    _hmm_model_rows = []

    _subject_col = None
    if "analysis_mouse" in df_test_kernel.columns:
        _subject_col = "analysis_mouse"
    elif "subject" in df_test_kernel.columns:
        _subject_col = "subject"

    _input_cols = [stim_col] + input_cols_noStim
    _real_input_cols = [_col for _col in _input_cols if _col != "bias"]
    _required_cols = {
        "session",
        "selected_df_option",
        "first_choice_numeric",
        *_real_input_cols,
    }

    if df_test_kernel.empty:
        _hmm_model_rows.append({"status": "df_test_kernel is empty"})
    elif _subject_col is None:
        _hmm_model_rows.append(
            {"status": "No subject or analysis_mouse column found"}
        )
    elif not _required_cols.issubset(df_test_kernel.columns):
        _missing_cols = sorted(_required_cols - set(df_test_kernel.columns))
        _hmm_model_rows.append({"status": f"Missing columns: {_missing_cols}"})
    else:
        for (_subject, _stage), _df_subject_stage in df_test_kernel.groupby(
            [_subject_col, "selected_df_option"],
            sort=True,
        ):
            _model_key = f"{_subject}_{_stage}"
            _df_model = _df_subject_stage.dropna(
                subset=["first_choice_numeric"] + _real_input_cols
            ).copy()

            if _df_model.empty:
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": _subject,
                        "stage": _stage,
                        "n_trials": 0,
                        "n_sessions": 0,
                        "log_likelihood": pd.NA,
                        "status": "no valid trials after dropna",
                    }
                )
                continue

            _datas, _inpts = utils_test.build_glmhmm_inputs_by_session(
                _df_model,
                y_col="first_choice_numeric",
                stim_col=_input_cols,
            )
            _valid_pairs = [
                (_data, _inpt)
                for _data, _inpt in zip(_datas, _inpts)
                if len(_data) > 0 and len(_inpt) > 0
            ]

            if not _valid_pairs:
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": _subject,
                        "stage": _stage,
                        "n_trials": len(_df_model),
                        "n_sessions": 0,
                        "log_likelihood": pd.NA,
                        "status": "no valid sessions",
                    }
                )
                continue

            _datas, _inpts = zip(*_valid_pairs)
            _datas = list(_datas)
            _inpts = list(_inpts)
            _input_dim = _inpts[0].shape[1]

            try:
                _map_glmhmm, _ll, _hmm_lls = utils_test.fit_glmhmm(
                    _datas,
                    _inpts,
                    num_states=int(glmhmm_num_states.value),
                    obs_dim=1,
                    input_dim=_input_dim,
                    num_categories=2,
                    N_iters=int(glmhmm_n_iters.value),
                    prior_sigma=2,
                    kappa=15,
                )
                hmm_model_dic[_model_key] = {
                    "model": _map_glmhmm,
                    "log_likelihood": _ll,
                    "hmm_lls": _hmm_lls,
                    "datas": _datas,
                    "inpts": _inpts,
                    "df": _df_model,
                    "subject": _subject,
                    "stage": _stage,
                    "inputs": _input_cols,
                }
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": _subject,
                        "stage": _stage,
                        "n_trials": len(_df_model),
                        "n_sessions": len(_datas),
                        "input_dim": _input_dim,
                        "log_likelihood": _ll,
                        "status": "ok",
                    }
                )
            except Exception as _exc:
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": _subject,
                        "stage": _stage,
                        "n_trials": len(_df_model),
                        "n_sessions": len(_datas),
                        "input_dim": _input_dim,
                        "log_likelihood": pd.NA,
                        "status": f"failed: {_exc}",
                    }
                )

    hmm_model_summary = pd.DataFrame(_hmm_model_rows)
    hmm_model_summary
    return (hmm_model_dic,)


@app.cell
def _(hmm_model_dic, input_cols_noStim, pd, stim_col):
    weight_cols = (
        [stim_col]
        + [col for col in input_cols_noStim if col != "bias"]
        + ["bias"]
    )

    eps = 1e-9

    for hmm_model_name, model_info in hmm_model_dic.items():
        hmm = model_info["model"]

        if hmm.K != 3:
            print(f"[skip] {hmm_model_name}: expected 3 states, got {hmm.K}")
            continue

        weights = hmm.observations.params.squeeze()
        if weights.ndim != 2 or weights.shape[1] != len(weight_cols):
            print(
                f"[skip] {hmm_model_name}: weights shape {weights.shape} does "
                f"not match columns {weight_cols}"
            )
            continue

        weight_df_before = pd.DataFrame(
            weights,
            columns=weight_cols,
            index=[f"state{i}" for i in range(weights.shape[0])],
        )

        stim_abs = weight_df_before[stim_col].abs()
        bias_abs = weight_df_before["bias"].abs()

        stim_score = stim_abs / (bias_abs + eps)
        bias_score = bias_abs / (stim_abs + eps)

        old_stim_state_label = stim_score.idxmax()
        old_stim_state = int(old_stim_state_label.replace("state", ""))

        remaining_state_labels = [
            f"state{state}"
            for state in range(hmm.K)
            if state != old_stim_state
        ]

        old_bias_state_label = bias_score.loc[remaining_state_labels].idxmax()
        old_bias_state = int(old_bias_state_label.replace("state", ""))

        old_other_state = [
            state
            for state in range(hmm.K)
            if state not in {old_bias_state, old_stim_state}
        ][0]

        # new state0 <- relative-bias strongest old state
        # new state1 <- relative-stim strongest old state
        # new state2 <- remaining old state
        perm = [old_bias_state, old_stim_state, old_other_state]

        hmm.permute(perm)

        weights_after = hmm.observations.params.squeeze()
        weight_df_after = pd.DataFrame(
            weights_after,
            columns=weight_cols,
            index=[f"state{i}" for i in range(weights_after.shape[0])],
        )

        model_info["state_permutation"] = perm
        model_info["weight_df_before_reorder"] = weight_df_before
        model_info["weight_df_after_reorder"] = weight_df_after
        model_info["state_reorder_scores"] = pd.DataFrame(
            {
                "stim_abs": stim_abs,
                "bias_abs": bias_abs,
                "stim_relative_to_bias": stim_score,
                "bias_relative_to_stim": bias_score,
            }
        )

    # if (
    #     model in hmm_model_dic
    #     and "weight_df_after_reorder" in hmm_model_dic[model]
    # ):
    #     hmm_model_dic[model]["weight_df_after_reorder"]
    # else:
    #     pd.DataFrame()
    return


@app.cell
def _(Path, hmm_model_dic, input_cols_noStim, stim_col, utils_test):
    figs_all = []
    for model_name_ in hmm_model_dic:
        _fig, _df_with_state, _post_prob_list = (
            utils_test.plot_glmhmm_pipeline_figure(
                hmm_model_dic[model_name_]["model"],
                hmm_model_dic[model_name_]['df'],
                hmm_model_dic[model_name_]['datas'],
                hmm_model_dic[model_name_]['inpts'],
                input_cols=[
                    stim_col
                ] + [col for col in input_cols_noStim if col != "bias"],
                y_col="first_choice_numeric",
                psychometric_x=stim_col,
                title=(
                    f"GLM-HMM summary {model_name_} "
                ),
                psychometric_value_type=(
                    "continuous"
                    if "aud" in str(model_name_)
                    else "discrete"
                ),
            )
        )
        print(_df_with_state["glmhmm_state"].value_counts().sort_index())
        figs_all.append(_fig)
        _fig.savefig(
            Path(
                f"/mnt/e/data/LeciLab/behavioral_data/tmp/glm_hmm_model_allData/{hmm_model_dic[model_name_]['subject']}_{model_name_}.svg"
            ),
            format="svg",
            bbox_inches="tight"
        )
    return


@app.cell
def _(mo):
    selected_option_glmhmm_run_button = mo.ui.run_button(
        label="Plot trained GLM-HMM models"
    )
    selected_option_plot_session_mode = mo.ui.dropdown(
        options=["Random consecutive", "Manual sessions"],
        value="Random consecutive",
        label="Plot session selection",
    )
    selected_option_plot_n_sessions = mo.ui.number(
        start=1,
        stop=200,
        step=1,
        value=1,
        label="Consecutive sessions per model",
    )
    selected_option_plot_seed = mo.ui.number(
        start=0,
        stop=100000,
        step=1,
        value=0,
        label="Random session seed",
    )
    selected_option_manual_sessions = mo.ui.text(
        value="",
        label="Manual sessions",
        placeholder="session_a, session_b, session_c",
    )
    mo.vstack(
        [
            selected_option_plot_session_mode,
            selected_option_plot_n_sessions,
            selected_option_plot_seed,
            selected_option_manual_sessions,
            selected_option_glmhmm_run_button,
        ]
    )
    return (
        selected_option_manual_sessions,
        selected_option_plot_n_sessions,
        selected_option_plot_session_mode,
    )


@app.cell
def _(
    Path,
    df_test_kernel,
    hmm_model_dic,
    input_cols_noStim,
    selected_option_manual_sessions,
    selected_option_plot_n_sessions,
    selected_option_plot_session_mode,
    stim_col,
    utils_test,
):
    figs = []
    for model in hmm_model_dic:
        mouse_name = model[:6]
        stage_name = model[-8:]
        df_model = df_test_kernel[(df_test_kernel['subject'] == mouse_name) & (df_test_kernel['selected_df_option'] == stage_name)]
        if selected_option_plot_session_mode.value == 'Random consecutive':
            sample_sessions = (df_model['session'].drop_duplicates().sample(n=selected_option_plot_n_sessions.value))
            df_sample = df_model[df_model['session'].isin(sample_sessions)]
        else:
            sample_sessions = selected_option_manual_sessions.value
            df_sample = df_model[df_model['session'].isin(sample_sessions)]
        df_sample_clean = df_sample.dropna(subset=[stim_col] + [col for col in input_cols_noStim if col != "bias"] + ["first_choice_numeric"]).copy()
        _plot_datas, _plot_inpts = utils_test.build_glmhmm_inputs_by_session(
            df_sample_clean,
            y_col="first_choice_numeric",
            stim_col=[stim_col] + input_cols_noStim,
        )
        _fig, _df_with_state, _post_prob_list = (
            utils_test.plot_glmhmm_pipeline_figure(
                hmm_model_dic[model]["model"],
                df_sample_clean,
                _plot_datas,
                _plot_inpts,
                input_cols=[
                    stim_col
                ] + [col for col in input_cols_noStim if col != "bias"],
                y_col="first_choice_numeric",
                psychometric_x=stim_col,
                title=(
                    f"GLM-HMM summary {model} "
                    f"({str(sample_sessions.values)} plot sessions)"
                ),
                psychometric_value_type=(
                    "continuous"
                    if "aud" in str(stage_name)
                    else "discrete"
                ),
            )
        )
        print(_df_with_state["glmhmm_state"].value_counts().sort_index())
        figs.append(_fig)
        _fig.savefig(
            Path(
                f"/mnt/e/data/LeciLab/behavioral_data/tmp/glm_hmm_model_sampleData/{hmm_model_dic[model]['subject']}_{model}.svg"
            ),
            format="svg",
            bbox_inches="tight"
        )
    figs
    return


if __name__ == "__main__":
    app.run()
