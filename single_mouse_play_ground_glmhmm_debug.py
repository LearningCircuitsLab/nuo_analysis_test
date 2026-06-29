import marimo

__generated_with = "0.23.9"
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
def _(utils_test):
    import importlib
    importlib.reload(utils_test)
    return


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
    # project = "COT_cannula_GAD2_data"
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
def _(
    Path,
    dft,
    download_button,
    mouse_select,
    np,
    pd,
    project,
    utils,
    utils_test,
):
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


    # add fixation breaks 
    df_test_vis_easy = utils_test.add_number_of_pokes(utils_test.add_fixation_break_columns(df_test_vis_easy),port_number=2,)
    df_test_aud_easy = utils_test.add_number_of_pokes(utils_test.add_fixation_break_columns(df_test_aud_easy),port_number=2,)
    df_test_vis_hard = utils_test.add_number_of_pokes(utils_test.add_fixation_break_columns(df_test_vis_hard),port_number=2,)
    df_test_aud_hard = utils_test.add_number_of_pokes(utils_test.add_fixation_break_columns(df_test_aud_hard),port_number=2,)
    return (
        df_test_aud_easy,
        df_test_aud_hard,
        df_test_vis_easy,
        df_test_vis_hard,
    )


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

    df_test_vis = df_test[df_test['stimulus_modality'] == 'visual']
    df_test_aud = df_test[df_test['stimulus_modality'] == 'auditory']
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
        marker="o",
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


@app.cell
def _(df_test, dft, np, plots, plt):
    # random session performance plot
    _random_mouse = df_test["subject"].dropna().sample(n=1).iloc[0]
    _df_random_mouse = df_test[df_test["subject"] == _random_mouse]
    n_sessions = 5
    sessions = (
        _df_random_mouse["session"]
        .dropna()
        .sort_values()
        .unique()
    )
    start_idx = np.random.randint(0, len(sessions) - n_sessions + 1)
    random_sessions = sessions[start_idx:start_idx + n_sessions]
    _df_random_session = _df_random_mouse[
        _df_random_mouse["session"].isin(random_sessions)
    ]

    _window = 50
    _df_random_session = dft.get_performance_through_trials(
        _df_random_session,
        window=_window,
    )
    _session_changes = _df_random_session[
        _df_random_session.session != _df_random_session.session.shift(1)
    ].index

    _perf_hue = "_performance_line"
    _df_random_session[_perf_hue] = "performance"

    _fig, _perf_ax = plt.subplots(figsize=(7, 4))
    _perf_ax = plots.performance_vs_trials_plot(
        _df_random_session,
        _perf_ax,
        legend=True,
        session_changes=_session_changes,
        hue=_perf_hue,
        palette=["black"],
    )
    _df_random_roap = _df_random_session.copy()
    _df_random_roap["repeat_or_alternate"] = dft.get_repeat_or_alternate_series(
        _df_random_roap.correct_side
    )
    _df_random_roap = dft.get_repeat_or_alternate_performance(
        _df_random_roap,
        window=_window,
    )
    _perf_ax = plots.repeat_or_alternate_performance_plot(
        _df_random_roap,
        _perf_ax,
        session_changes=_session_changes,
    )
    _x_start = (
        _df_random_session.loc[
            _df_random_session["performance_w"].notna(),
            "total_trial",
        ].min()
        + 50
    )
    _perf_ax.set_xlim(left=_x_start)
    _perf_ax.set_title(f"{_random_mouse} | {random_sessions[0]}")
    _fig.tight_layout()
    _fig.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/random_mouse_performance_plot.svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # number of central pokes
    """)
    return


@app.cell
def _(df_test, np, plt):
    # port 2 pokes histogram, split by correct
    _df_p2h = df_test.copy()

    _npokes_correct = _df_p2h.loc[_df_p2h["correct"] == True, "port2_pokes_num"]
    _npokes_wrong = _df_p2h.loc[_df_p2h["correct"] == False, "port2_pokes_num"]

    _bins = np.arange(
        _df_p2h["port2_pokes_num"].min() - 0.5,
        # _df_p2h["port2_pokes_num"].max() + 1.5,
        25.5,
        1,
    )

    _fig, _ax = plt.subplots(figsize=(5, 4))

    _ax.hist(
        _npokes_correct,
        bins=_bins,
        color="green",
        alpha=0.45,
        label="correct",
        density=True
    )

    _ax.hist(
        _npokes_wrong,
        bins=_bins,
        color="red",
        alpha=0.45,
        label="wrong",
        density=True
    )

    _ax.set_xlabel("Number of pokes in Port2")
    _ax.set_ylabel("Frequency")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False)

    _fig.tight_layout()
    _fig.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/all_normal_data_port2_pokes_hist.svg")
    plt.show()
    return


@app.cell
def _(df_test, np, plt):
    # port 2 pokes histogram, split by correct
    _df_break = df_test.copy()
    _nbreaks_correct = _df_break.loc[_df_break["correct"] == True]
    _nbreaks_wrong = _df_break.loc[_df_break["correct"] == False]

    _bins = np.arange(
        _df_break["fixation_breaks"].min() - 0.5,
        # _df_break["fixation_breaks"].max() + 1.5,
        25.5,
        1,
    )

    _fig, _ax = plt.subplots(figsize=(5, 4))

    _ax.hist(
        _nbreaks_correct["fixation_breaks"],
        bins=_bins,
        color="green",
        alpha=0.45,
        label="correct",
        density=True
    )

    _ax.hist(
        _nbreaks_wrong["fixation_breaks"],
        bins=_bins,
        color="red",
        alpha=0.45,
    
        label="wrong",
        density=True
    )

    _ax.set_xlabel("Number of breaks in fixation")
    _ax.set_ylabel("Frequency")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False)

    _fig.tight_layout()
    _fig.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/all_normal_data_fixation_breaks_hist.svg")
    plt.show()
    return


@app.cell
def _(df_options, df_selector, np, plt, utils):
    # port 2 pokes histogram, split by correct, using df_selector / df_options
    _selected_df_names = df_selector.value

    _bins = np.arange(-0.5, 25.5, 1)

    _fig, _axes = plt.subplots(
        1,
        len(_selected_df_names),
        figsize=(5 * len(_selected_df_names), 4),
        sharey=True,
    )

    for _ax, _df_name in zip(_axes, _selected_df_names):
        _df_p2h = df_options[_df_name]["df"].copy()

        _df_p2h["port2_holds"] = _df_p2h.apply(
            lambda row: utils.get_trial_port_hold(row, 2),
            axis=1,
        )
        _df_p2h["port2_npokes"] = _df_p2h["port2_holds"].apply(len)

        _npokes_correct = _df_p2h.loc[_df_p2h["correct"] == True, "port2_npokes"]
        _npokes_wrong = _df_p2h.loc[_df_p2h["correct"] == False, "port2_npokes"]

        _ax.hist(
            _npokes_correct,
            bins=_bins,
            color="green",
            alpha=0.45,
            label="correct",
            density=True,
        )

        _ax.hist(
            _npokes_wrong,
            bins=_bins,
            color="red",
            alpha=0.45,
            label="wrong",
            density=True,
        )

        _ax.set_title(_df_name)
        _ax.set_xlabel("Number of pokes in Port2")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.legend(frameon=False)

    _axes[0].set_ylabel("Density")

    _fig.tight_layout()
    # _fig.savefig(
    #     "/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/all_normal_data_port2_pokes_hist.svg"
    # )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # trial start across time
    """)
    return


@app.cell
def _(df_test, plt, utils):
    # random 5 sessions trial start time plot, colored by previous_correct
    # each session starts from 0
    _n_sessions = 5

    _session_keys = (
        df_test[["subject", "session"]]
        .dropna()
        .drop_duplicates()
        .sample(n=_n_sessions)
    )

    _df_tst = df_test.merge(
        _session_keys,
        on=["subject", "session"],
        how="inner",
    ).copy()

    _df_tst = _df_tst.sort_values(["subject", "session", "trial"]).reset_index(drop=True)

    _df_tst = utils.add_time_from_session_start(_df_tst)

    _sdf = _df_tst.copy()
    _sdf["trial_from_start"] = _sdf.groupby(["subject", "session"]).cumcount()

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _ax.scatter(
        _sdf.loc[_sdf["previous_correct"] == True, "time_from_start"],
        _sdf.loc[_sdf["previous_correct"] == True, "trial_from_start"],
        s=8,
        color="green",
        alpha=0.6,
        label="previous_correct",
    )

    _ax.scatter(
        _sdf.loc[_sdf["previous_correct"] == False, "time_from_start"],
        _sdf.loc[_sdf["previous_correct"] == False, "trial_from_start"],
        s=8,
        color="red",
        alpha=0.6,
        label="previous_incorrect",
    )

    _ax.set_xlabel("Trial start time from session start (s)")
    _ax.set_ylabel("Trial number from session start")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False)

    _fig.tight_layout()
    _fig.savefig(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/random_5_sessions_trial_start_time_plot.png"
    )
    _fig.savefig(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/random_5_sessions_trial_start_time_plot.svg"
    )
    plt.show()
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
        plt.figure(figsize=(5, 5), dpi=150)

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
    # # no curve

    # for stage_name__, df_stage__ in df_test.groupby("selected_df_option"):
    #     plt.figure(figsize=(5, 5), dpi=150)

    #     if "vis" in stage_name__:
    #         # x_col = "visual_stimulus_ratio"
    #         valueType = "discrete"
    #     else:
    #         # x_col = "total_evidence_strength"
    #         valueType = "continue"

    #     if "easy" in stage_name__:
    #         bins = 2
    #     else:
    #         bins = 6


    #     for mouse_name_psycurve, df_mouse_psycurve in df_stage__.groupby('subject'):
    #         color_palette = sns.color_palette(
    #             "colorblind",
    #             df_stage__['subject'].nunique()
    #         )

    #     for (mouse_name_psycurve, df_mouse_psycurve), color in zip(
    #         df_stage__.groupby('subject'),
    #         sns.color_palette("colorblind", df_stage__['subject'].nunique())
    #     ):
    #         utils_test.psychometric_plot_easy_logistic(
    #             df_mouse_psycurve,
    #             x=stim_col,
    #             y="first_choice_numeric",
    #             valueType = valueType,
    #             bins = bins,
    #             point_kwargs={
    #                 "color": color,
    #                 "label": "",
    #                 "alpha": 0.3,
    #             },
    #             line_kwargs={
    #                 "color": color,
    #                 "label": "",
    #                 "alpha": 0,
    #             },
    #         )

    #     utils_test.psychometric_plot_easy_logistic(
    #         df_stage__,
    #         x=stim_col,
    #         valueType = valueType,
    #         bins = bins,
    #         y="first_choice_numeric",
    #         point_kwargs={
    #             "color": "black",
    #             "label": "All mice",
    #         },
    #         line_kwargs={
    #             "color": "black",
    #             "label": "",
    #             "alpha": 0,
    #         },
    #     )

    #     plt.title(stage_name__)
    #     plt.legend(frameon=False)
    #     plt.tight_layout()
    #     plt.show()
    return


@app.cell
def _():
    # for stage_name__, df_stage__ in df_test.groupby("selected_df_option"):
    #     plt.figure(figsize=(5, 5), dpi=150)

    #     if "vis" in stage_name__:
    #         valueType = "discrete"
    #     else:
    #         valueType = "continue"

    #     if "easy" in stage_name__:
    #         bins = 2
    #     else:
    #         bins = 6


    #     for mouse_name_psycurve, df_mouse_psycurve in df_stage__.groupby('subject'):
    #         color_palette = sns.color_palette(
    #             "colorblind",
    #             df_stage__['subject'].nunique()
    #         )

    #     for (mouse_name_psycurve, df_mouse_psycurve), color in zip(
    #         df_stage__.groupby('subject'),
    #         sns.color_palette("colorblind", df_stage__['subject'].nunique())
    #     ):
    #         utils_test.psychometric_plot_easy_logistic(
    #             df_mouse_psycurve,
    #             x=stim_col,
    #             y="repeat_choice_numeric",
    #             valueType = valueType,
    #             bins = bins,
    #             point_kwargs={
    #                 "color": color,
    #                 "label": "",
    #                 "alpha": 0.6,
    #             },
    #             line_kwargs={
    #                 "color": color,
    #                 "label": "",
    #                 "alpha": 0,
    #             },
    #         )

    #     utils_test.psychometric_plot_easy_logistic(
    #         df_stage__,
    #         x=stim_col,
    #         valueType = valueType,
    #         bins = bins,
    #         y="repeat_choice_numeric",
    #         point_kwargs={
    #             "color": "black",
    #             "label": "",
    #         },
    #         line_kwargs={
    #             "color": color,
    #             "label": "",
    #             "alpha": 0,
    #         },
    #     )

    #     plt.title(stage_name__)
    #     plt.legend(frameon=False)
    #     plt.tight_layout()
    #     plt.show()
    return


@app.cell
def _():
    # for stage_name__, df_stage__ in df_test.groupby("selected_df_option"):
    #     plt.figure(figsize=(5, 5), dpi=150)

    #     if "vis" in stage_name__:
    #         x_col = "visual_stimulus_diff"
    #         valueType = "continue"
    #     else:
    #         x_col = "total_evidence_strength"
    #         valueType = "continue"

    #     if "easy" in stage_name__:
    #         bins = 2
    #     else:
    #         bins = 6


    #     for mouse_name_psycurve, df_mouse_psycurve in df_stage__.groupby('subject'):
    #         color_palette = sns.color_palette(
    #             "colorblind",
    #             df_stage__['subject'].nunique()
    #         )

    #     for (mouse_name_psycurve, df_mouse_psycurve), color in zip(
    #         df_stage__.groupby('subject'),
    #         sns.color_palette("colorblind", df_stage__['subject'].nunique())
    #     ):
    #         utils_test.psychometric_plot_easy_logistic(
    #             df_mouse_psycurve,
    #             x=x_col,
    #             y="first_choice_numeric",
    #             valueType = valueType,
    #             bins = bins,
    #             point_kwargs={
    #                 "color": color,
    #                 "label": "",
    #                 "alpha": 0.6,
    #             },
    #             line_kwargs={
    #                 "color": color,
    #                 "label": "",
    #                 "alpha": 1,
    #             },
    #         )

    #     utils_test.psychometric_plot_easy_logistic(
    #         df_stage__,
    #         x=x_col,
    #         valueType = valueType,
    #         bins = bins,
    #         y="first_choice_numeric",
    #         point_kwargs={
    #             "color": "black",
    #             "label": "",
    #         },
    #         line_kwargs={
    #             "color": color,
    #             "label": "",
    #             "alpha": 1,
    #         },
    #     )

    #     plt.title(stage_name__)
    #     plt.legend(frameon=False)
    #     plt.tight_layout()
    #     plt.show()
    return


@app.cell
def _(df_test_vis_hard, plots, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(5, 5))
    for _i, linecolor in zip(df_test_vis_hard[df_test_vis_hard['previous_correct_numeric'] == 1].groupby('previous_port_before_stimulus'), ['brown', 'lightcoral']):
        plots.psychometric_plot(df = _i[1], x = 'visual_stimulus_ratio', y = 'first_choice_numeric', 
                                line_kwargs={'color': linecolor,
                                             'label': 'Left_previous_correct' if _i[0] == 'left' else 'Right_previous_correct'}, 
                                point_kwargs={'color': 'black', 'label': None},
                                ax=_ax)

    for _i, linecolor in zip(df_test_vis_hard[df_test_vis_hard['previous_correct_numeric'] == 0].groupby('previous_port_before_stimulus'), ['darkgreen', 'limegreen']):
        plots.psychometric_plot(df = _i[1], x = 'visual_stimulus_ratio', y = 'first_choice_numeric', 
                                line_kwargs={'color': linecolor,
                                             'label': 'Left_previous_incorrect' if _i[0] == 'left' else 'Right_previous_incorrect'}, 
                                point_kwargs={'color': 'black', 'label': None},
                                ax=_ax)
    _fig.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/psychometric_plot_previous_correction_previous.svg")
    _ax.legend()
    _ax.set_title("Previous Correction Previous")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # matrix with previous choices
    """)
    return


@app.cell
def _(mo):
    heatmap_dataset_selector = mo.ui.dropdown(
        options=["hard", "easy", "all"],
        value="hard",
        label="Dataset",
    )
    heatmap_value_selector = mo.ui.dropdown(
        options=["previous_same_choice_numeric", "roa_choice_numeric"],
        value="previous_same_choice_numeric",
        label="Heatmap value",
    )
    heatmap_condition_selector = mo.ui.dropdown(
        options=["previous_correct", "None"],
        value="previous_correct",
        label="Row condition",
    )

    mo.hstack(
        [
            heatmap_dataset_selector,
            heatmap_value_selector,
            heatmap_condition_selector,
        ]
    )
    return (
        heatmap_condition_selector,
        heatmap_dataset_selector,
        heatmap_value_selector,
    )


@app.cell
def _(
    df_test,
    df_test_aud_easy,
    df_test_aud_hard,
    df_test_vis_easy,
    df_test_vis_hard,
    heatmap_condition_selector,
    heatmap_dataset_selector,
    heatmap_value_selector,
    mo,
    pd,
    plt,
    sns,
):
    def add_previous_col(df, value_col, previous_col):
        df = df.copy()
        group_cols = [
            _col for _col in ["subject", "session"]
            if _col in df.columns
        ]
        if group_cols:
            df[previous_col] = (
                df
                .groupby(group_cols, sort=False)[value_col]
                .shift(1)
            )
        else:
            df[previous_col] = df[value_col].shift(1)

        return df

    def bool_mask(series, value):
        text_values = series.astype(str).str.lower()
        if value:
            return (
                series.eq(True)
                | series.eq(1)
                | text_values.isin(["true", "1"])
            )

        return (
            series.eq(False)
            | series.eq(0)
            | text_values.isin(["false", "0"])
        )

    def make_pivot_table(
        df,
        modality,
        value_col,
        condition_col=None,
        condition_value=None,
    ):
        if value_col not in df.columns:
            return pd.DataFrame()

        if condition_col is not None:
            if condition_col not in df.columns:
                return pd.DataFrame()

            df_plot = df[
                bool_mask(df[condition_col], condition_value)
            ].copy()
        else:
            df_plot = df.copy()

        if df_plot.empty:
            return pd.DataFrame()

        df_plot[value_col] = (
            pd.to_numeric(df_plot[value_col], errors="coerce")
            .astype(float)
        )
        df_plot = df_plot.dropna(subset=[value_col])
        if df_plot.empty:
            return pd.DataFrame()

        if modality == "visual":
            stim_col_plot = "visual_stimulus_ratio"
            previous_stim_col = "previous_visual_stimulus_ratio"
            if stim_col_plot not in df_plot.columns:
                return pd.DataFrame()

            df_plot[stim_col_plot] = df_plot[stim_col_plot].round(2)
            df_plot = add_previous_col(
                df_plot,
                value_col=stim_col_plot,
                previous_col=previous_stim_col,
            )

        else:
            stim_col_plot = "total_evidence_strength_binned"
            previous_stim_col = "previous_total_evidence_strength_binned"
            if "total_evidence_strength" not in df_plot.columns:
                return pd.DataFrame()

            df_plot["total_evidence_strength"] = pd.to_numeric(
                df_plot["total_evidence_strength"],
                errors="coerce",
            )
            df_plot = df_plot.dropna(subset=["total_evidence_strength"])
            if df_plot.empty:
                return pd.DataFrame()

            _n_bins = 2 if dataset_choice == "easy" else 6
            bin_groups = pd.cut(
                df_plot["total_evidence_strength"],
                bins=_n_bins,
            )
            labels = (
                df_plot["total_evidence_strength"]
                .groupby(bin_groups, observed=False)
                .mean()
            )
            fallback_labels = pd.Series(
                [interval.mid for interval in labels.index],
                index=labels.index,
            )
            labels = labels.fillna(fallback_labels)

            df_plot[stim_col_plot] = (
                bin_groups
                .map(labels)
                .astype(float)
                .round(2)
            )

            df_plot = add_previous_col(
                df_plot,
                value_col=stim_col_plot,
                previous_col=previous_stim_col,
            )

        return (
            df_plot
            .pivot_table(
                index=previous_stim_col,
                columns=stim_col_plot,
                values=value_col,
                aggfunc="mean",
                observed=True,
            )
            .astype(float)
        )

    def plot_heatmap(table, ax, title, xlabel, ylabel, value_col):
        if table.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            ax.set_title(title)
            return

        sns.heatmap(
            table,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": f"Mean {value_col}"},
            vmin=0,
            vmax=1,
            annot_kws={"color": "black"},
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        ax.set_title(title)

    dataset_choice = heatmap_dataset_selector.value
    value_col = heatmap_value_selector.value
    condition_col = heatmap_condition_selector.value

    dataset_map = {
        "hard": {
            "Visual": df_test_vis_hard,
            "Auditory": df_test_aud_hard,
        },
        "easy": {
            "Visual": df_test_vis_easy,
            "Auditory": df_test_aud_easy,
        },
        "all": {
            "Visual": df_test[
                df_test["stimulus_modality"] == "visual"
            ],
            "Auditory": df_test[
                df_test["stimulus_modality"] == "auditory"
            ],
        },
    }

    figures = []
    condition_col = None if condition_col == "None" else condition_col
    condition_values = [None] if condition_col is None else [True, False]

    for condition_value in condition_values:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4),
            constrained_layout=True,
        )

        visual_table = make_pivot_table(
            dataset_map[dataset_choice]["Visual"],
            modality="visual",
            value_col=value_col,
            condition_col=condition_col,
            condition_value=condition_value,
        )
        auditory_table = make_pivot_table(
            dataset_map[dataset_choice]["Auditory"],
            modality="auditory",
            value_col=value_col,
            condition_col=condition_col,
            condition_value=condition_value,
        )

        if condition_col is None:
            visual_title = "Visual"
            auditory_title = "Auditory"
        else:
            visual_title = f"Visual | {condition_col} = {condition_value}"
            auditory_title = f"Auditory | {condition_col} = {condition_value}"

        plot_heatmap(
            visual_table,
            axes[0],
            title=visual_title,
            xlabel="visual stimulus ratio",
            ylabel="previous visual stimulus ratio",
            value_col=value_col,
        )
        plot_heatmap(
            auditory_table,
            axes[1],
            title=auditory_title,
            xlabel="total evidence strength binned",
            ylabel="previous total evidence strength binned",
            value_col=value_col,
        )

        figures.append(fig)

    if condition_col is None:
        _heatmap_output = mo.vstack(
            [
                mo.md(f"## {dataset_choice}: mean `{value_col}`"),
                figures[0],
            ]
        )
    else:
        _heatmap_output = mo.vstack(
            [
                mo.md(
                    f"## {dataset_choice}: mean `{value_col}` "
                    f"by `{condition_col}`"
                ),
                mo.md(f"### {condition_col} = True"),
                figures[0],
                mo.md(f"### {condition_col} = False"),
                figures[1],
            ]
        )

    _heatmap_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # stimulus correlation to see antibias
    """)
    return


@app.cell
def _(df_test, np):
    df_test_copy = df_test.copy()
    for session_2 in df_test["session"].unique():
        df_session_2 = df_test[df_test["session"] == session_2].copy()

        df_session_2["previous_correct_side"] = (
            df_session_2["correct_side"].shift(1, fill_value=np.nan)
        )
        df_test_copy.loc[
            df_session_2.index,
            "previous_correct_side"
        ] = df_session_2["previous_correct_side"]
    return (df_test_copy,)


@app.cell
def _(df_test_copy, pd, plt, sns):
    df_test_copy_vis = df_test_copy[
        df_test_copy["stimulus_modality"] == "visual"
    ]

    df_test_copy_aud = df_test_copy[
        df_test_copy["stimulus_modality"] == "auditory"
    ]

    correlation_table_vis = pd.crosstab(
        df_test_copy_vis["previous_correct_side"],
        df_test_copy_vis["correct_side"],
        normalize="index",
    )

    correlation_table_aud = pd.crosstab(
        df_test_copy_aud["previous_correct_side"],
        df_test_copy_aud["correct_side"],
        normalize="index",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # 左边：visual
    sns.heatmap(
        correlation_table_vis,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=axes[0],
    )

    axes[0].set_xlabel("Current correct side")
    axes[0].set_ylabel("Previous correct side")
    axes[0].set_title("Visual")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0)

    # 右边：auditory
    sns.heatmap(
        correlation_table_aud,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=axes[1],
    )

    axes[1].set_xlabel("Current correct side")
    axes[1].set_ylabel("Previous correct side")
    axes[1].set_title("Auditory")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.show()
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
    _alpha_output_dir = Path(r"E:\data\LeciLab\behavioral_data\tmp\processing")
    if not _alpha_output_dir.is_absolute() and str(_alpha_output_dir).startswith("E:\\"):
        _alpha_output_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp/processing")
    _alpha_act_paths = {
        _mouse: (
            _alpha_output_dir
            / f"{_mouse}_per_mouse_stage_alpha_df_act.pkl"
        )
        for _mouse in _selected_mice
    }
    _alpha_stim_paths = {
        _mouse: (
            _alpha_output_dir
            / f"{_mouse}_per_mouse_stage_alpha_df_stim.pkl"
        )
        for _mouse in _selected_mice
    }

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

    _all_selected_alpha_files_exist = bool(_selected_mice) and all(
        _alpha_act_paths[_mouse].exists()
        and _alpha_stim_paths[_mouse].exists()
        for _mouse in _selected_mice
    )

    if _all_selected_alpha_files_exist and not download_button.value:
        per_mouse_stage_alpha_df_act = pd.concat(
            [
                pd.read_pickle(_alpha_act_paths[_mouse])
                for _mouse in _selected_mice
            ],
            ignore_index=True,
        )
        per_mouse_stage_alpha_df_stim = pd.concat(
            [
                pd.read_pickle(_alpha_stim_paths[_mouse])
                for _mouse in _selected_mice
            ],
            ignore_index=True,
        )
        print(
            "Loaded saved per-mouse stage alpha data for: "
            f"{_selected_mice}"
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

        for _mouse in _selected_mice:
            _mouse_alpha_act = (
                per_mouse_stage_alpha_df_act[
                    per_mouse_stage_alpha_df_act["mouse"] == _mouse
                ].copy()
                if "mouse" in per_mouse_stage_alpha_df_act.columns
                else pd.DataFrame()
            )
            _mouse_alpha_stim = (
                per_mouse_stage_alpha_df_stim[
                    per_mouse_stage_alpha_df_stim["mouse"] == _mouse
                ].copy()
                if "mouse" in per_mouse_stage_alpha_df_stim.columns
                else pd.DataFrame()
            )
            _mouse_alpha_act.to_pickle(_alpha_act_paths[_mouse])
            _mouse_alpha_stim.to_pickle(_alpha_stim_paths[_mouse])

        per_mouse_stage_alpha_df_act = pd.concat(
            [
                pd.read_pickle(_alpha_act_paths[_mouse])
                for _mouse in _selected_mice
            ],
            ignore_index=True,
        ) if _selected_mice else pd.DataFrame()
        per_mouse_stage_alpha_df_stim = pd.concat(
            [
                pd.read_pickle(_alpha_stim_paths[_mouse])
                for _mouse in _selected_mice
            ],
            ignore_index=True,
        ) if _selected_mice else pd.DataFrame()
        print(
            "Saved per-mouse stage alpha data for: "
            f"{_selected_mice}"
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
def _(df_test_kernel):
    df_test_kernel['visual_stimulus_ratio_scaled'] = df_test_kernel['visual_stimulus_ratio']/6
    df_test_kernel['visual_stimulus_ratio_scaled'].unique()
    return


@app.cell
def _(contextlib, df_test_kernel, io, mo, plots, plt, stim_col, utils_test):


    fig_glm_filter = []

    for stage_name_glm_compare, df_stage_glm_compare in df_test_kernel.groupby("selected_df_option"):
        if "vis" in stage_name_glm_compare:
            X_filter_model = [
                # stim_col,
                "visual_stimulus_ratio_scaled",
                # "visual_ratio_diff_interact",
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
        value=2,
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

        fig__, ax = plt.subplots(figsize=(8, 5))
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
        fig__.tight_layout()

        _save_dir = Path(r"E:\data\LeciLab\behavioral_data\tmp")
        if not _save_dir.is_absolute() and str(_save_dir).startswith("E:\\"):
            _save_dir = Path("/mnt/e/data/LeciLab/b ehavioral_data/tmp")
        _save_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _save_dir / f"{_plot_title}.svg"
        fig__.savefig(_save_path, format="svg", bbox_inches="tight")
        print(f"Saved per-mouse GLM-HMM convergence plot: {_save_path}")

        glmhmm_results = pd.DataFrame(result_rows)
        glmhmm_cache[cache_key] = {
            "results": glmhmm_results.copy(),
            "figure": fig__,
        }
        glmhmm_output = mo.vstack(
            [
                mo.md(f"Computed and cached GLM-HMM result for `{cache_key}`."),
                glmhmm_results,
                fig__,
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
def _(mo):
    model_training_type = mo.ui.dropdown(
        options=["by subjects", "by stages"],
        value="by subjects",
        label="Model training type",
    )
    glmhmm_fit_button = mo.ui.run_button(
        label="Run GLM-HMM fitting"
    )

    mo.vstack([model_training_type, glmhmm_fit_button])
    return glmhmm_fit_button, model_training_type


@app.cell
def _():
    input_cols_noStim = ["bias", "action_trace", "stimulus_trace"]
    return (input_cols_noStim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## train subjects x stage models
    """)
    return


@app.cell
def _(
    Path,
    df_test_kernel,
    glmhmm_fit_button,
    glmhmm_n_iters,
    glmhmm_num_states,
    input_cols_noStim,
    model_training_type,
    pd,
    stim_col,
    utils_test,
):
    import pickle as _pickle

    hmm_model_dic_by_subjects = {}
    _hmm_model_rows = []
    _model_output_dir = Path(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/processing/glmhmm_models"
    )

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

    if model_training_type.value != "by subjects":
        _hmm_model_rows.append({"status": "Select 'by subjects' to use these models"})
    elif df_test_kernel.empty:
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
            _model_path = (
                _model_output_dir / f"{_model_key}_glmhmm_model.pkl"
            )
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
                if glmhmm_fit_button.value:
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

                    _saved_model_info = {
                        "model": _map_glmhmm,
                        "log_likelihood": _ll,
                        "hmm_lls": _hmm_lls,
                        "subject": _subject,
                        "stage": _stage,
                        "inputs": _input_cols,
                        "input_dim": _input_dim,
                        "num_states": int(glmhmm_num_states.value),
                        "training_type": "by subjects",
                    }
                    _model_output_dir.mkdir(parents=True, exist_ok=True)
                    with open(_model_path, "wb") as _file:
                        _pickle.dump(
                            _saved_model_info,
                            _file,
                            protocol=_pickle.HIGHEST_PROTOCOL,
                        )
                    _status = "fit and saved"
                else:
                    if not _model_path.exists():
                        _hmm_model_rows.append(
                            {
                                "model_key": _model_key,
                                "subject": _subject,
                                "stage": _stage,
                                "n_trials": len(_df_model),
                                "n_sessions": len(_datas),
                                "input_dim": _input_dim,
                                "log_likelihood": pd.NA,
                                "status": (
                                    "saved model not found; click "
                                    "'Run GLM-HMM fitting'"
                                ),
                            }
                        )
                        continue

                    with open(_model_path, "rb") as _file:
                        _saved_model_info = _pickle.load(_file)

                    if (
                        _saved_model_info.get("input_dim") != _input_dim
                        or _saved_model_info.get("num_states")
                        != int(glmhmm_num_states.value)
                    ):
                        _hmm_model_rows.append(
                            {
                                "model_key": _model_key,
                                "subject": _subject,
                                "stage": _stage,
                                "n_trials": len(_df_model),
                                "n_sessions": len(_datas),
                                "input_dim": _input_dim,
                                "log_likelihood": pd.NA,
                                "status": (
                                    "saved model configuration differs; "
                                    "refit required"
                                ),
                            }
                        )
                        continue

                    _map_glmhmm = _saved_model_info["model"]
                    _ll = _saved_model_info.get("log_likelihood", pd.NA)
                    _hmm_lls = _saved_model_info.get("hmm_lls", [])
                    _status = "loaded"

                hmm_model_dic_by_subjects[_model_key] = {
                    "model": _map_glmhmm,
                    "log_likelihood": _ll,
                    "hmm_lls": _hmm_lls,
                    "datas": _datas,
                    "inpts": _inpts,
                    "df": _df_model,
                    "subject": _subject,
                    "stage": _stage,
                    "inputs": _input_cols,
                    "training_type": "by subjects",
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
                        "status": _status,
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

    hmm_model_summary_by_subjects = pd.DataFrame(_hmm_model_rows)
    return hmm_model_dic_by_subjects, hmm_model_summary_by_subjects


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## train 4 stage models
    """)
    return


@app.cell
def _(
    Path,
    df_test_kernel,
    glmhmm_fit_button,
    glmhmm_n_iters,
    glmhmm_num_states,
    input_cols_noStim,
    model_training_type,
    pd,
    stim_col,
    utils_test,
):
    import pickle as _pickle

    hmm_model_dic_by_stages = {}
    _hmm_model_rows = []
    _model_output_dir = Path(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/processing/glmhmm_models"
    )

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

    if model_training_type.value != "by stages":
        _hmm_model_rows.append({"status": "Select 'by stages' to use these models"})
    elif df_test_kernel.empty:
        _hmm_model_rows.append({"status": "df_test_kernel is empty"})
    elif _subject_col is None:
        _hmm_model_rows.append(
            {"status": "No subject or analysis_mouse column found"}
        )
    elif not _required_cols.issubset(df_test_kernel.columns):
        _missing_cols = sorted(_required_cols - set(df_test_kernel.columns))
        _hmm_model_rows.append({"status": f"Missing columns: {_missing_cols}"})
    else:
        for _stage, _df_stage in df_test_kernel.groupby(
            "selected_df_option",
            sort=True,
        ):
            _model_key = f"stage_{_stage}"
            _model_path = (
                _model_output_dir / f"{_model_key}_glmhmm_model.pkl"
            )
            _df_model = _df_stage.dropna(
                subset=["first_choice_numeric"] + _real_input_cols
            ).copy()

            if _df_model.empty:
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": "all_subjects",
                        "stage": _stage,
                        "n_trials": 0,
                        "n_sessions": 0,
                        "log_likelihood": pd.NA,
                        "status": "no valid trials after dropna",
                    }
                )
                continue

            # Keep sessions from different subjects as separate HMM sequences.
            _df_model["session"] = (
                _df_model[_subject_col].astype(str)
                + "__"
                + _df_model["session"].astype(str)
            )
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
                        "subject": "all_subjects",
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
                if glmhmm_fit_button.value:
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

                    _saved_model_info = {
                        "model": _map_glmhmm,
                        "log_likelihood": _ll,
                        "hmm_lls": _hmm_lls,
                        "subject": "all_subjects",
                        "stage": _stage,
                        "inputs": _input_cols,
                        "input_dim": _input_dim,
                        "num_states": int(glmhmm_num_states.value),
                        "training_type": "by stages",
                    }
                    _model_output_dir.mkdir(parents=True, exist_ok=True)
                    with open(_model_path, "wb") as _file:
                        _pickle.dump(
                            _saved_model_info,
                            _file,
                            protocol=_pickle.HIGHEST_PROTOCOL,
                        )
                    _status = "fit and saved"
                else:
                    if not _model_path.exists():
                        _hmm_model_rows.append(
                            {
                                "model_key": _model_key,
                                "subject": "all_subjects",
                                "stage": _stage,
                                "n_trials": len(_df_model),
                                "n_sessions": len(_datas),
                                "input_dim": _input_dim,
                                "log_likelihood": pd.NA,
                                "status": (
                                    "saved model not found; click "
                                    "'Run GLM-HMM fitting'"
                                ),
                            }
                        )
                        continue

                    with open(_model_path, "rb") as _file:
                        _saved_model_info = _pickle.load(_file)

                    if (
                        _saved_model_info.get("input_dim") != _input_dim
                        or _saved_model_info.get("num_states")
                        != int(glmhmm_num_states.value)
                    ):
                        _hmm_model_rows.append(
                            {
                                "model_key": _model_key,
                                "subject": "all_subjects",
                                "stage": _stage,
                                "n_trials": len(_df_model),
                                "n_sessions": len(_datas),
                                "input_dim": _input_dim,
                                "log_likelihood": pd.NA,
                                "status": (
                                    "saved model configuration differs; "
                                    "refit required"
                                ),
                            }
                        )
                        continue

                    _map_glmhmm = _saved_model_info["model"]
                    _ll = _saved_model_info.get("log_likelihood", pd.NA)
                    _hmm_lls = _saved_model_info.get("hmm_lls", [])
                    _status = "loaded"

                hmm_model_dic_by_stages[_model_key] = {
                    "model": _map_glmhmm,
                    "log_likelihood": _ll,
                    "hmm_lls": _hmm_lls,
                    "datas": _datas,
                    "inpts": _inpts,
                    "df": _df_model,
                    "subject": "all_subjects",
                    "stage": _stage,
                    "inputs": _input_cols,
                    "training_type": "by stages",
                }

                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": "all_subjects",
                        "stage": _stage,
                        "n_subjects": _df_model[_subject_col].nunique(),
                        "n_trials": len(_df_model),
                        "n_sessions": len(_datas),
                        "input_dim": _input_dim,
                        "log_likelihood": _ll,
                        "status": _status,
                    }
                )

            except Exception as _exc:
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "subject": "all_subjects",
                        "stage": _stage,
                        "n_subjects": _df_model[_subject_col].nunique(),
                        "n_trials": len(_df_model),
                        "n_sessions": len(_datas),
                        "input_dim": _input_dim,
                        "log_likelihood": pd.NA,
                        "status": f"failed: {_exc}",
                    }
                )

    hmm_model_summary_by_stages = pd.DataFrame(_hmm_model_rows)
    return hmm_model_dic_by_stages, hmm_model_summary_by_stages


@app.cell
def _(
    hmm_model_dic_by_stages,
    hmm_model_dic_by_subjects,
    hmm_model_summary_by_stages,
    hmm_model_summary_by_subjects,
    model_training_type,
):
    if model_training_type.value == "by stages":
        hmm_model_dic = hmm_model_dic_by_stages
        hmm_model_summary = hmm_model_summary_by_stages
    else:
        hmm_model_dic = hmm_model_dic_by_subjects
        hmm_model_summary = hmm_model_summary_by_subjects

    hmm_model_summary
    return (hmm_model_dic,)


@app.cell
def _(hmm_model_dic, np, plt):
    # model likelihood
    fig, ax = plt.subplots(figsize=(7, 4))

    for model_name, model_info_ in hmm_model_dic.items():
        hmm_lls = np.asarray(model_info_["hmm_lls"])

        ax.plot(
            np.arange(1, len(hmm_lls) + 1),
            hmm_lls,
            label=model_name,
        )

    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("GLM-HMM fitting convergence")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## permute states order
    """)
    return


@app.cell
def _(hmm_model_dic, input_cols_noStim, np, pd, stim_col):
    weight_cols = (
        [stim_col]
        + [col_ for col_ in input_cols_noStim if col_ != "bias"]
        + ["bias"]
    )

    for hmm_model_name, model_info in hmm_model_dic.items():
        hmm = model_info["model"]

        if hmm.K not in (2, 3):
            print(
                f"[skip] {hmm_model_name}: "
                f"expected 2 or 3 states, got {hmm.K}"
            )
            continue

        weights = np.asarray(
            hmm.observations.params
        ).squeeze()

        if weights.ndim != 2 or weights.shape[1] != len(weight_cols):
            print(
                f"[skip] {hmm_model_name}: weights shape {weights.shape} "
                f"does not match columns {weight_cols}"
            )
            continue

        weight_df_before = pd.DataFrame(
            weights,
            columns=weight_cols,
            index=[f"state{i}" for i in range(hmm.K)],
        )

        # 各类 weight 的绝对强度
        stim_abs = weight_df_before[stim_col].abs()

        # stimulus state
        old_stimulus_label = stim_abs.idxmax()
        old_stimulus_state = int(old_stimulus_label.replace("state", ""))

        if hmm.K == 2:
            old_non_stimulus_state = [
                state for state in range(hmm.K)
                if state != old_stimulus_state
            ][0]

            # New state1 is the stimulus-driven state.
            perm = [
                old_non_stimulus_state,
                old_stimulus_state,
            ]
            state_order = {
                0: "non_stimulus",
                1: "stimulus_driven",
            }

        else:
            missing_weight_cols = [
                col for col in ["bias", "action_trace", "stimulus_trace"]
                if col not in weight_df_before.columns
            ]
            if missing_weight_cols:
                print(
                    f"[skip] {hmm_model_name}: missing columns "
                    f"{missing_weight_cols} for 3-state ordering"
                )
                continue

            bias_abs = weight_df_before["bias"].abs()
            act_trace_abs = weight_df_before["action_trace"].abs()
            stim_trace_abs = weight_df_before["stimulus_trace"].abs()

            # 用 L2 norm 合并两个 history weights
            history_abs = np.sqrt(
                act_trace_abs**2 + stim_trace_abs**2
            )

            # remaining states
            remaining_labels = [
                f"state{i}"
                for i in range(hmm.K)
                if f"state{i}" != old_stimulus_label
            ]

            # bias vs history dominance
            dominance = bias_abs - history_abs

            old_bias_label = dominance.loc[remaining_labels].idxmax()
            old_history_label = dominance.loc[remaining_labels].idxmin()

            old_bias_state = int(old_bias_label.replace("state", ""))
            old_history_state = int(old_history_label.replace("state", ""))

            perm = [
                old_bias_state,
                old_stimulus_state,
                old_history_state,
            ]
            state_order = {
                0: "bias",
                1: "stimulus_driven",
                2: "history",
            }

        hmm.permute(perm)

        weights_after = np.asarray(
            hmm.observations.params
        ).squeeze()

        weight_df_after = pd.DataFrame(
            weights_after,
            columns=weight_cols,
            index=[f"state{i}" for i in range(hmm.K)],
        )

        model_info["weight_df_before_reorder"] = weight_df_before
        model_info["weight_df_after_reorder"] = weight_df_after
        model_info["state_order_after_reorder"] = state_order

        print(hmm_model_name)
        print(f"weight before reorder\n{weight_df_before}")
        print(f"permute{perm}")
        print(f"state order after reorder: {state_order}")
    return


@app.cell
def _(Path, hmm_model_dic, input_cols_noStim, pd, plt, stim_col, utils_test):
    stages = ["aud_easy", "aud_hard", "vis_easy", "vis_hard"]

    glmhmm_state_df_dir = Path(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/processing/glmhmm_state_df"
    )
    glmhmm_state_df_dir.mkdir(parents=True, exist_ok=True)

    if hmm_model_dic:
        state_index = sorted(
            {
                state
                for model_info in hmm_model_dic.values()
                for state in range(model_info["model"].K)
            }
        )
    else:
        saved_state_index = []
        for state_df_path in glmhmm_state_df_dir.glob("*_state*.pkl"):
            state_text = state_df_path.stem.rsplit("_state", 1)[-1]
            if state_text.isdigit():
                saved_state_index.append(int(state_text))

        state_index = sorted(set(saved_state_index)) or [0, 1, 2]

    expected_paths = {
        (stage, state): glmhmm_state_df_dir / f"{stage}_state{state}.pkl"
        for stage in stages
        for state in state_index
    }

    state_counts_by_stage = {}
    glmhmm_state_df_loaded = {
        stage: {state: [] for state in state_index}
        for stage in stages
    }

    if hmm_model_dic:
        for model_name_ in hmm_model_dic:
            _fig, _df_with_state, _post_prob_list = (
                utils_test.plot_glmhmm_pipeline_figure(
                    hmm_model_dic[model_name_]["model"],
                    hmm_model_dic[model_name_]["df"],
                    hmm_model_dic[model_name_]["datas"],
                    hmm_model_dic[model_name_]["inpts"],
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

            state_counts = (
                _df_with_state["glmhmm_state"]
                .value_counts()
                .reindex(state_index, fill_value=0)
                .sort_index()
            )

            print(model_name_)
            print(state_counts)

            stage_name_ = hmm_model_dic[model_name_].get("stage", None)

            if stage_name_ is None:
                # fallback: NUO001_vis_easy -> vis_easy
                stage_name_ = "_".join(model_name_.split("_")[1:])

            if stage_name_ not in stages:
                print(f"[skip] {model_name_}: unknown stage {stage_name_}")
                continue

            if stage_name_ not in state_counts_by_stage:
                state_counts_by_stage[stage_name_] = pd.Series(
                    0,
                    index=state_index,
                    dtype=int,
                )

            state_counts_by_stage[stage_name_] = (
                state_counts_by_stage[stage_name_] + state_counts
            )

            _df_with_state = _df_with_state.copy()
            _df_with_state["model_name"] = model_name_
            _df_with_state["selected_df_option"] = stage_name_

            if "subject" not in _df_with_state.columns:
                _df_with_state["subject"] = hmm_model_dic[model_name_].get(
                    "subject",
                    model_name_.split("_")[0],
                )

            for state in state_index:
                df_state = _df_with_state[
                    _df_with_state["glmhmm_state"] == state
                ].copy()

                glmhmm_state_df_loaded[stage_name_][state].append(df_state)

            _fig.savefig(
                Path(
                    f"/mnt/e/data/LeciLab/behavioral_data/tmp/glm_hmm_model_allData/"
                    f"{hmm_model_dic[model_name_]['subject']}_{model_name_}.svg"
                ),
                format="svg",
                bbox_inches="tight",
            )
            plt.close(_fig)

        for stage in stages:
            for state in state_index:
                df_concat = (
                    pd.concat(
                        glmhmm_state_df_loaded[stage][state],
                        ignore_index=True,
                    )
                    if glmhmm_state_df_loaded[stage][state]
                    else pd.DataFrame()
                )

                output_path = expected_paths[(stage, state)]
                df_concat.to_pickle(output_path)
                glmhmm_state_df_loaded[stage][state] = df_concat

        print(f"Saved GLM-HMM state dfs to: {glmhmm_state_df_dir}")

    else:
        missing_paths = [
            path for path in expected_paths.values()
            if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                "hmm_model_dic is empty and these saved state dfs are missing: "
                + ", ".join(str(path) for path in missing_paths)
            )

        for stage in stages:
            state_counts_by_stage[stage] = pd.Series(
                0,
                index=state_index,
                dtype=int,
            )

            for state in state_index:
                df_state = pd.read_pickle(expected_paths[(stage, state)])
                glmhmm_state_df_loaded[stage][state] = df_state
                state_counts_by_stage[stage].loc[state] = len(df_state)

        print(f"Loaded GLM-HMM state dfs from: {glmhmm_state_df_dir}")

    state_distribution_df = pd.DataFrame(state_counts_by_stage)
    state_distribution_df = state_distribution_df.reindex(columns=stages, fill_value=0)
    state_distribution_df.index = [
        f"state{s}" for s in state_distribution_df.index
    ]

    state_distribution_df
    return (state_distribution_df,)


@app.cell
def _(hmm_model_dic, mo, np, pd, plt):
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    _all_weights = []
    _state_rows = []
    _weight_names = None
    pc_num = 3

    for _model_name, _model_info in hmm_model_dic.items():
        _weights = np.asarray(_model_info["model"].observations.params).squeeze()

        _model_inputs = list(_model_info.get("inputs", []))
        _model_weight_names = [_col for _col in _model_inputs if _col != "bias"]
        if "bias" in _model_inputs:
            _model_weight_names.append("bias")

        if len(_model_weight_names) != _weights.shape[1]:
            print(
                f"[skip] {_model_name}: weights shape {_weights.shape} "
                f"does not match inputs {_model_weight_names}"
            )
            continue

        if _weight_names is None:
            _weight_names = _model_weight_names
        elif _model_weight_names != _weight_names:
            print(
                f"[skip] {_model_name}: input order "
                f"{_model_weight_names} does not match {_weight_names}"
            )
            continue

        for _state in range(_weights.shape[0]):
            _all_weights.append(_weights[_state])
            _state_rows.append(
                {
                    "model_name": _model_name,
                    "mouse": _model_info.get(
                        "subject",
                        _model_name.split("_")[0],
                    ),
                    "stage": _model_info.get(
                        "stage",
                        "_".join(_model_name.split("_")[1:]),
                    ),
                    "state": _state,
                }
            )

    state_pca_fig = None
    state_weight_df = pd.DataFrame()
    state_info = pd.DataFrame(_state_rows)
    pca_loadings = pd.DataFrame()

    if not _all_weights:
        _state_pca_output = mo.md(
            "No valid GLM-HMM state weights available for PCA."
        )

    else:
        _X = np.asarray(_all_weights, dtype=float)

        state_weight_df = pd.concat(
            [
                state_info.reset_index(drop=True),
                pd.DataFrame(_X, columns=_weight_names),
            ],
            axis=1,
        )

        # Standardize each weight feature across all states/models
        _scaler = _StandardScaler()
        _Xz = _scaler.fit_transform(_X)

        # Fit PCA once: retain PC1-PC3
        _n_components = min(pc_num, _Xz.shape[0], _Xz.shape[1])

        if _n_components < 3:
            _state_pca_output = mo.md(
                f"Cannot create a 3D PCA plot: only {_n_components} "
                "valid principal component(s) are available."
            )

        else:
            _pca = _PCA(n_components=_n_components)
            _X_pca = _pca.fit_transform(_Xz)

            pca_loadings = pd.DataFrame(
                _pca.components_.T,
                index=_weight_names,
                columns=[f"PC{i + 1}" for i in range(_n_components)],
            )

            _explained = _pca.explained_variance_ratio_
            _pc1_pct = 100 * _explained[0]
            _pc2_pct = 100 * _explained[1]
            _pc3_pct = 100 * _explained[2]

            state_info = state_info.copy()
            state_info["pc1"] = _X_pca[:, 0]
            state_info["pc2"] = _X_pca[:, 1]
            state_info["pc3"] = _X_pca[:, 2]
            state_info["label"] = [
                f"{_row.model_name}_s{_row.state}"
                for _, _row in state_info.iterrows()
            ]

            _unique_mice = sorted(state_info["mouse"].astype(str).unique())
            _unique_states = sorted(state_info["state"].unique())

            _mouse_cmap = plt.get_cmap("tab20")
            _mouse_colors = {
                _mouse: _mouse_cmap(_idx)
                for _idx, _mouse in enumerate(_unique_mice)
            }

            _marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*"]
            _state_markers = {
                _state: _marker_cycle[_idx % len(_marker_cycle)]
                for _idx, _state in enumerate(_unique_states)
            }

            _mouse_values = state_info["mouse"].astype(str).to_numpy()
            _state_values = state_info["state"].to_numpy()

            _mouse_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=_mouse_colors[_mouse],
                    markeredgecolor=_mouse_colors[_mouse],
                    linestyle="",
                    label=_mouse,
                    markersize=6,
                )
                for _mouse in _unique_mice
            ]

            _state_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=_state_markers[_state],
                    color="black",
                    markerfacecolor="white",
                    linestyle="",
                    label=f"State {_state}",
                    markersize=6,
                )
                for _state in _unique_states
            ]

            # 3D PCA plot
            state_pca_fig = plt.figure(figsize=(10, 6), dpi=180)
            _pca_ax = state_pca_fig.add_subplot(
                1,
                1,
                1,
                projection="3d",
            )

            for _mouse in _unique_mice:
                for _state in _unique_states:
                    _mask = (
                        (_mouse_values == _mouse)
                        & (_state_values == _state)
                    )

                    if not _mask.any():
                        continue

                    _pca_ax.scatter(
                        _X_pca[_mask, 0],
                        _X_pca[_mask, 1],
                        _X_pca[_mask, 2],
                        color=_mouse_colors[_mouse],
                        marker=_state_markers[_state],
                        alpha=0.8,
                        s=42,
                    )

            _pca_ax.set_xlabel(f"PC1 ({_pc1_pct:.1f}%)")
            _pca_ax.set_ylabel(f"PC2 ({_pc2_pct:.1f}%)")
            _pca_ax.set_zlabel(f"PC3 ({_pc3_pct:.1f}%)", labelpad=-2)
            _pca_ax.set_title(
                "3-PC PCA of standardized GLM-HMM state weights"
            )

            _mouse_legend = _pca_ax.legend(
                handles=_mouse_handles,
                title="Mouse",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
                fontsize=8,
            )
            _pca_ax.add_artist(_mouse_legend)

            _pca_ax.legend(
                handles=_state_handles,
                title="State",
                bbox_to_anchor=(1.02, 0.45),
                loc="upper left",
                frameon=False,
                fontsize=8,
            )

            state_pca_fig.tight_layout()

            _state_pca_output = mo.vstack(
                [
                    state_pca_fig,
                    mo.md("### PCA loadings (PC1-PC3)"),
                    pca_loadings,
                    mo.md("### State PCA coordinates"),
                    state_info,
                    mo.md("### State weights"),
                    state_weight_df,
                ]
            )

    _state_pca_output
    return


@app.cell
def _(plt, state_distribution_df):
    ax_state_distribution = state_distribution_df.T.plot(
        kind="bar",
        figsize=(8, 5),
        width=0.75,
    )

    plt.ylabel("Number of trials")
    plt.xlabel("Stage")
    plt.title("GLM-HMM state distribution by stage")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="State", frameon=False)
    plt.tight_layout()

    plt.savefig(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/glm_hmm_state_distribution_by_stage.svg",
        format="svg",
        bbox_inches="tight",
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## plot random glmhmm pipeline plots
    """)
    return


@app.cell
def _(hmm_model_dic, mo):
    _model_options = list(hmm_model_dic)
    selected_option_glmhmm_run_button = mo.ui.run_button(
        label="Plot trained GLM-HMM models"
    )
    selected_option_plot_model = mo.ui.dropdown(
        options=_model_options,
        value=_model_options[0] if _model_options else None,
        label="GLM-HMM model",
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
            selected_option_plot_model,
            selected_option_plot_session_mode,
            selected_option_plot_n_sessions,
            selected_option_plot_seed,
            selected_option_manual_sessions,
            selected_option_glmhmm_run_button,
        ]
    )
    return (
        selected_option_glmhmm_run_button,
        selected_option_manual_sessions,
        selected_option_plot_model,
        selected_option_plot_n_sessions,
        selected_option_plot_seed,
        selected_option_plot_session_mode,
    )


@app.cell
def _(
    Path,
    hmm_model_dic,
    input_cols_noStim,
    mo,
    selected_option_glmhmm_run_button,
    selected_option_manual_sessions,
    selected_option_plot_model,
    selected_option_plot_n_sessions,
    selected_option_plot_seed,
    selected_option_plot_session_mode,
    stim_col,
    utils_test,
):
    mo.stop(
        not selected_option_glmhmm_run_button.value,
        mo.md("Select a model and click **Plot trained GLM-HMM models**."),
    )

    _model = selected_option_plot_model.value
    mo.stop(_model is None, mo.md("No trained GLM-HMM model is available."))

    _model_info = hmm_model_dic[_model]
    _mouse_name = _model_info["subject"]
    _stage_name = _model_info["stage"]
    _df_model = _model_info["df"]

    if selected_option_plot_session_mode.value == "Random consecutive":
        _available_sessions = _df_model["session"].drop_duplicates()
        _n_sessions = min(
            int(selected_option_plot_n_sessions.value),
            len(_available_sessions),
        )
        _sample_sessions = _available_sessions.sample(
            n=_n_sessions,
            random_state=int(selected_option_plot_seed.value),
        ).tolist()
    else:
        _sample_sessions = [
            _session.strip()
            for _session in selected_option_manual_sessions.value.split(",")
            if _session.strip()
        ]

    mo.stop(not _sample_sessions, mo.md("No sessions are available to plot."))
    _df_sample = _df_model[_df_model["session"].isin(_sample_sessions)]
    _df_sample_clean = _df_sample.dropna(
        subset=[stim_col]
        + [col for col in input_cols_noStim if col != "bias"]
        + ["first_choice_numeric"]
    ).copy()
    mo.stop(_df_sample_clean.empty, mo.md("The selected sessions contain no valid trials."))

    _plot_datas, _plot_inpts = utils_test.build_glmhmm_inputs_by_session(
        _df_sample_clean,
        y_col="first_choice_numeric",
        stim_col=[stim_col] + input_cols_noStim,
    )
    _fig, _df_with_state, _post_prob_list = (
        utils_test.plot_glmhmm_pipeline_figure(
            _model_info["model"],
            _df_sample_clean,
            _plot_datas,
            _plot_inpts,
            input_cols=[stim_col]
            + [col for col in input_cols_noStim if col != "bias"],
            y_col="first_choice_numeric",
            psychometric_x=stim_col,
            title=(
                f"GLM-HMM summary {_model} "
                f"({_sample_sessions} plot sessions)"
            ),
            psychometric_value_type=(
                "continuous" if "aud" in str(_stage_name) else "discrete"
            ),
        )
    )
    print(_df_with_state["glmhmm_state"].value_counts().sort_index())

    _save_dir = Path(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/glm_hmm_model_sampleData"
    )
    _save_dir.mkdir(parents=True, exist_ok=True)
    _fig.savefig(
        _save_dir / f"{_mouse_name}_{_model}.svg",
        format="svg",
        bbox_inches="tight",
    )
    _fig.set_dpi(100)
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
