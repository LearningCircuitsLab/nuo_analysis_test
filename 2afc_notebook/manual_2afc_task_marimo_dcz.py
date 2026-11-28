import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    from pathlib import Path
    import sys
    import warnings


    REPO_ROOT = Path("/home/kudongdong/code/brainCircuitsLab")
    _package_path = REPO_ROOT / "lecilab-behavior-analysis"
    if _package_path.exists() and str(_package_path) not in sys.path:
        sys.path.insert(0, str(_package_path))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import behavior_utils
    import plot_test
    import utils_test
    import seaborn as sns

    from lecilab_behavior_analysis import utils as utils
    utils.IDIBAPS_TV_PROJECTS = "/archive/training_village/"
    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )
    from scipy.optimize import minimize

    warnings.filterwarnings("ignore")
    return (
        Path,
        behavior_utils,
        dft,
        minimize,
        np,
        pd,
        plot_test,
        plots,
        plt,
        session_summary_figure,
        sns,
        utils,
        utils_test,
    )


@app.cell
def _(behavior_utils, plot_test, utils_test):
    import importlib
    importlib.reload(behavior_utils)
    importlib.reload(plot_test)
    importlib.reload(utils_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # plot_testing.ipynb in marimo
    """)
    return


@app.cell
def _(utils):
    def safe_rsync_cluster_data(project_name, file_path, credentials, local_path):
        try:
            return utils.rsync_cluster_data(
                project_name=project_name,
                file_path=file_path,
                credentials=credentials,
                local_path=local_path,
            )
        except AttributeError as exc:
            if "NoneType" not in str(exc) or "decode" not in str(exc):
                raise
            print(
                f"Could not sync {file_path}; rsync failed. "
                "See the console output above for details."
            )
            return False

    return (safe_rsync_cluster_data,)


@app.cell
def _(utils):
    tv_projects = utils.get_server_projects()
    print(tv_projects)
    return (tv_projects,)


@app.cell
def _(mo, tv_projects):
    _default_project = 'COT_cannula_data'
    project_select = mo.ui.dropdown(
        options=tv_projects,
        value=_default_project,
        label="Project",
    )

    project_select
    return (project_select,)


@app.cell
def _(project_select, tv_projects):
    project = project_select.value
    project_idx = tv_projects.index(project)
    return (project,)


@app.cell
def _(project, utils):
    animals = utils.get_animals_in_project(project)
    animals = [_a for _a in animals if "test" not in _a]
    print(animals)
    return (animals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # mouse select
    """)
    return


@app.cell
def _(animals, mo):
    mouse_select = mo.ui.dropdown(
        options=animals,
        value=animals[0],
        label="Mouse",
    )

    mouse_select
    return


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Download / update this mouse")
    download_button
    return (download_button,)


@app.cell
def _(
    Path,
    animals,
    download_button,
    pd,
    project,
    safe_rsync_cluster_data,
    utils,
):
    local_path = Path(utils.get_outpath()) / project / "sessions" 
    local_path.mkdir(parents=True, exist_ok=True)


    df_all = pd.DataFrame()
    for _mouse in animals:
        _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
        _local_path.mkdir(parents=True, exist_ok=True)
        if download_button.value:
            safe_rsync_cluster_data(
                project_name=project,
                file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )
        df_all = pd.concat([df_all, pd.read_csv(_local_path / Path(f"{_mouse}.csv"), sep=";")], ignore_index=True)
    return (df_all,)


@app.cell
def _(mo):
    run_mode_select = mo.ui.dropdown(
        options=["Auto", "Manual"],
        value="Manual",
        label="Run mode",
    )

    run_mode_select
    return (run_mode_select,)


@app.cell
def _(df_all, run_mode_select):
    run_mode = run_mode_select.value
    df = df_all[df_all["run_mode"] == run_mode]
    df.iloc[-10:]
    return (df,)


@app.cell
def _():
    # fig_subject_progress = subject_progress_figure(
    #     df,
    #     perf_window=100,
    #     summary_matrix_plot=False,
    # )
    # fig_subject_progress
    return


@app.cell
def _(df, dft, np):
    df_day = dft.add_day_column_to_df(df.copy())
    df_day = dft.add_trial_of_day_column_to_df(df_day)
    df_day["total_trial"] = np.arange(1, df_day.shape[0] + 1)

    available_dates = sorted(
        df_day["year_month_day"].dropna().astype(str).unique().tolist()
    )
    return available_dates, df_day


@app.cell
def _(available_dates, mo):
    date_select_start, date_select_end = (
        mo.ui.dropdown(
            options=available_dates,
            value=default,
            label=f"Date_{label}",
        )
        for label, default in [
            ("start", available_dates[0]),
            ("end", available_dates[-1]),
        ]
    )
    mo.vstack([date_select_start, date_select_end])
    return date_select_end, date_select_start


@app.cell
def _(dft, plots, plt, utils):
    def compact_session_summary_figure(df, perf_window=50):
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1, 1],
            width_ratios=[1, 1],
        )
        text_ax = fig.add_subplot(gs[0, 0])
        perf_ax = fig.add_subplot(gs[0, 1])
        visual_ax = fig.add_subplot(gs[1, 0])
        auditory_ax = fig.add_subplot(gs[1, 1])

        plots.summary_text_plot(df, kind="session", ax=text_ax)
        if df.empty:
            fig.tight_layout()
            return fig

        df = dft.get_performance_through_trials(df.copy(), window=perf_window)
        session_changes = df[df.session != df.session.shift(1)].index
        if df.current_training_stage.nunique() > 1:
            perf_hue = "current_training_stage"
        else:
            perf_hue = "stimulus_modality"
        plots.performance_vs_trials_plot(
            df,
            ax=perf_ax,
            legend=True,
            session_changes=session_changes,
            hue=perf_hue,
        )

        if "task" in df.columns:
            df_task = df[df["task"] != "Habituation"]
        else:
            df_task = df.copy()

        psych_axes = {
            "visual": visual_ax,
            "auditory": auditory_ax,
        }
        for mod, ax_name in psych_axes.items():
            if len(df_task) > 0:
                if (
                    "stimulus_modality" in df_task.columns
                    and mod in df_task["stimulus_modality"].unique()
                ):
                    df_mod = df_task[df_task["stimulus_modality"] == mod].copy()
                    if (
                        df_mod["difficulty"].nunique() == 1
                        and df_mod["difficulty"].unique()[0] == "easy"
                    ):
                        df_mod["side_difficulty"] = df_mod.apply(
                            lambda row: utils.side_and_difficulty_to_numeric(row),
                            axis=1,
                        )
                        df_mod = dft.add_mouse_first_choice(df_mod)
                        df_mod["first_choice_numeric"] = df_mod[
                            "first_choice"
                        ].apply(utils.transform_side_choice_to_numeric)
                        plots.choice_by_difficulty_plot(df_mod, ax=ax_name)
                    else:
                        if mod == "visual":
                            xvar = "visual_stimulus_ratio"
                            value_type = "discrete"
                        elif mod == "auditory":
                            xvar = "total_evidence_strength"
                            value_type = "continue"
                        psych_df = dft.get_performance_by_difficulty_ratio(df_mod)
                        plots.psychometric_plot(
                            psych_df,
                            x=xvar,
                            y="first_choice_numeric",
                            ax=ax_name,
                            valueType=value_type,
                        )
                    ax_name.set_title("choices on " + mod + " trials", fontsize=10)
                    if ax_name.get_legend() is not None:
                        ax_name.get_legend().remove()
                else:
                    ax_name.text(
                        0.1,
                        0.5,
                        "No trials in " + mod,
                        fontsize=10,
                        color="k",
                    )
            else:
                ax_name.text(0.1, 0.5, "Habituation phase", fontsize=10, color="k")

        fig.tight_layout()
        return fig

    return (compact_session_summary_figure,)


@app.cell
def _(date_select, df_day, session_summary_figure):
    date = date_select.value
    sdf = df_day[df_day["year_month_day"].astype(str) == date].copy()
    fig_session_summary = session_summary_figure(sdf, perf_window=50)
    fig_session_summary
    return


@app.cell
def _(available_dates, compact_session_summary_figure, df_day, mo):

    last_three_date_outputs = []
    for _date in available_dates[-3:]:
        _sdf = df_day[df_day["year_month_day"].astype(str) == _date].copy()
        _fig = compact_session_summary_figure(_sdf, perf_window=50)
        last_three_date_outputs.append(mo.vstack([mo.md(f"## {_date}"), _fig]))

    mo.vstack(
        [
            mo.md("# Last three days compact summaries"),
            *last_three_date_outputs,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # manual mouse select
    """)
    return


@app.cell
def _(animals, mo):
    fit_mouse_select = mo.ui.multiselect(
        options=animals,
        value=animals,
        label="Mice for fit",
    )

    fit_mouse_select
    return (fit_mouse_select,)


@app.cell
def _(mo):
    update_processData_button = mo.ui.run_button(label="update selected mice processed data")
    update_processData_button
    return (update_processData_button,)


@app.cell
def _():
    hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    # hM3Dq_mice = ['NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    # hM4Di_mice = ['NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    return hM3Dq_mice, hM4Di_mice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # stimulus modality select
    """)
    return


@app.cell
def _(mo):
    stimulus_modality_select = mo.ui.dropdown(
        options=["visual", "auditory", "all"],
        value="all",
        label="Stimulus modality",
    )
    stimulus_modality_select
    return (stimulus_modality_select,)


@app.cell
def _():
    basic_cols = [
        "subject",
        "year_month_day",
        "session",
        "trial",
        "stimulus_modality",
        "TRIAL_START",
        "TRIAL_END",
        "first_choice_numeric"
    ]

    continuous_cols = [
        "reaction_time",
        "trial_duration",
        "inter_trial_interval",
    ]

    binary_cols = [
        "correct_numeric",
        "previous_same_choice_numeric",
        "previous_correct_numeric",
        "roa_choice_numeric",
    ]

    count_cols = [
        "fixation_breaks",
        "eager_pokes",
        "Port2In"
    ]

    features = continuous_cols + binary_cols + count_cols
    return basic_cols, features


@app.cell
def _(
    Path,
    basic_cols,
    date_select_end,
    date_select_start,
    dft,
    features,
    fit_mouse_select,
    pd,
    project,
    run_mode_select,
    safe_rsync_cluster_data,
    update_processData_button,
    utils,
):
    _run_mode = run_mode_select.value
    parameters_output_dir = (
        Path(utils.get_outpath()) / project / "processed" / "parameters_for_fit" / _run_mode.lower()
    )
    stim_columns = {
        "df_test_vis": "visual",
        "df_test_aud": "auditory",
    }
    _stim_feature_columns = {
        "visual": "visual_stimulus_ratio",
        "auditory": "total_evidence_strength",
    }

    selected_mice = list(fit_mouse_select.value)

    def load_or_compute_mouse_parameters(_mouse):
        parameters_output_path = parameters_output_dir / f"{_mouse}_parameters_for_fit.pkl"
        if parameters_output_path.exists() and not update_processData_button.value:
            print(f"Loaded saved parameters_for_fit data: {parameters_output_path}")
            return pd.read_pickle(parameters_output_path)

        _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
        _local_path.mkdir(parents=True, exist_ok=True)
        _csv_path = _local_path / f"{_mouse}.csv"
        if not _csv_path.exists():
            safe_rsync_cluster_data(
                project_name=project,
                file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )

        if not _csv_path.exists():
            print(f"No CSV found for {_mouse}; skipped parameters_for_fit.")
            return {output_name: pd.DataFrame() for output_name in stim_columns}

        df_all_mouse = pd.read_csv(_csv_path, sep=";")
        df_manual = df_all_mouse.loc[df_all_mouse["run_mode"] == _run_mode]
        del df_all_mouse
        computed_results = {}
        for output_name, stim_name in stim_columns.items():
            df_stim = df_manual[df_manual["stimulus_modality"] == stim_name]
            if df_stim.empty:
                computed_results[output_name] = df_stim.copy()
                print(f"No {stim_name} trials found for {_mouse}; skipped parameters_for_fit.")
            else:
                computed_results[output_name] = dft.parameters_for_fit(df_stim)
            columns_to_save = basic_cols + features + [_stim_feature_columns[stim_name]]
            if computed_results[output_name].empty:
                computed_results[output_name] = computed_results[output_name].reindex(columns=columns_to_save)
            else:
                computed_results[output_name] = computed_results[output_name].loc[:, columns_to_save].copy()

        parameters_output_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(computed_results, parameters_output_path)
        print(f"Saved parameters_for_fit data: {parameters_output_path}")
        return computed_results

    combined_results = {output_name: [] for output_name in stim_columns}
    for _mouse in selected_mice:
        mouse_results = load_or_compute_mouse_parameters(_mouse)
        for output_name in stim_columns:
            df_mouse_result = mouse_results.get(output_name, pd.DataFrame())
            if not df_mouse_result.empty:
                if "subject" not in df_mouse_result.columns:
                    df_mouse_result = df_mouse_result.copy()
                    df_mouse_result["subject"] = _mouse
                combined_results[output_name].append(df_mouse_result)

    _all_results = [df for frames in combined_results.values() for df in frames]
    df_test_selected_raw = (
        pd.concat(_all_results, ignore_index=True)
        if _all_results
        else pd.DataFrame(columns=basic_cols + features + list(_stim_feature_columns.values()))
    )

    _dates = pd.to_datetime(df_test_selected_raw["year_month_day"], errors="coerce").dt.normalize()

    _start_date = pd.to_datetime(date_select_start.value).normalize()
    _end_date = pd.to_datetime(date_select_end.value).normalize()

    df_test_selected_raw = df_test_selected_raw[_dates.between(_start_date, _end_date, inclusive="both")].copy()
    return (df_test_selected_raw,)


@app.cell
def _():
    # df_test_aud["observations"] = pd.NA

    # for session in df_test_aud["session"].dropna().unique():
    #     df_session = df_test_aud[df_test_aud["session"] == session]
    #     date_values = df_session["date"].dropna().unique()
    #     if len(date_values) == 0:
    #         print(f"No date found for session {session}; skipped observation.")
    #         continue

    #     date_ = pd.to_datetime(date_values[0]).strftime("%Y%m%d_%H%M%S")
    #     session_info_file_name = mouse + "_TwoAFC_" + date_ + ".json"
    #     session_info_path = local_path / session_info_file_name
    #     safe_rsync_cluster_data(
    #         project_name=project,
    #         file_path="sessions/{}/{}".format(mouse, session_info_file_name),
    #         local_path=str(local_path),
    #         credentials=utils.get_idibaps_cluster_credentials(),
    #     )

    #     if not session_info_path.exists():
    #         print(f"No session info JSON found for session {session}: {session_info_path}")
    #         continue

    #     with session_info_path.open("r") as session_info_file:
    #         session_info = json.load(session_info_file)
    #     if isinstance(session_info, str):
    #         session_info = json.loads(session_info)

    #     observation = session_info.get("observations", pd.NA)
    #     df_test_aud.loc[df_session.index, "observations"] = [
    #         observation
    #     ] * len(df_session)
    return


@app.cell
def _(pd):
    injection_info_file_path = "/mnt/e/data/LeciLab/behavioral_data/data_test/manual_test_record_nuo.xlsx"
    # injection_info_file_path = "/home/nuo/Documents/data/LeciLab/behavioral_data/manual_record/manual_test_record_nuo.xlsx"
    injection_info_df = pd.read_excel(injection_info_file_path)
    injection_info_df
    return (injection_info_df,)


@app.cell
def _(df_test_selected_raw, injection_info_df, mouse, pd):
    df_test_selected = df_test_selected_raw.copy()
    df_test_selected["observations"] = pd.NA

    date_col = injection_info_df.columns[0]

    injection_dates = pd.to_datetime(
        injection_info_df[date_col],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    injection_info_with_dates = injection_info_df.copy()
    injection_info_with_dates["_injection_date"] = injection_dates
    injection_mouse_columns = [
        column for column in injection_info_df.columns if column != date_col
    ]
    injection_lookup = (
        injection_info_with_dates.melt(
            id_vars="_injection_date",
            value_vars=injection_mouse_columns,
            var_name="_mouse",
            value_name="_observations",
        )
        .set_index(["_injection_date", "_mouse"])["_observations"]
        .to_dict()
    )

    trial_dates = pd.to_datetime(
        df_test_selected["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    trial_mice = (
        df_test_selected["subject"].astype(str)
        if "subject" in df_test_selected.columns
        else pd.Series(mouse, index=df_test_selected.index)
    )
    df_test_selected["observations"] = [
        injection_lookup.get((trial_date, trial_mouse), pd.NA)
        for trial_date, trial_mouse in zip(trial_dates, trial_mice)
    ]

    df_test_selected['observations']
    return (df_test_selected,)


@app.cell
def _(
    behavior_utils,
    df_test_selected,
    dft,
    hM3Dq_mice,
    hM4Di_mice,
    injection_info_df,
    pd,
):
    session_performance = df_test_selected.groupby(
        ["subject", "session"]
    )["correct_numeric"].transform("mean")

    is_saline = df_test_selected["observations"].str.contains(
        "saline",
        case=False,
        na=False,
    )

    drop_mask = is_saline & (session_performance < 0.7)

    df_test_selected_upd = df_test_selected[
        ~drop_mask
    ].copy()

    metric_paired_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df
    )
    df_test_selected_upd["observation_group"] = df_test_selected_upd[
        "observations"
    ].apply(behavior_utils.observation_group)
    df_test_selected_upd["_pair_date"] = pd.to_datetime(
        df_test_selected_upd["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    for _subject in df_test_selected_upd['subject'].unique():
        _df_sub = df_test_selected_upd.loc[df_test_selected_upd['subject'] == _subject]
        _df_sub = dft.add_engagement_column(_df_sub, engagement_sd_criteria=2)
        df_test_selected_upd.loc[df_test_selected_upd['subject'] == _subject, 'engaged'] = _df_sub['engaged']

    available_dates_ = df_test_selected_upd[
        ["subject", "_pair_date", "observation_group"]
    ].dropna().drop_duplicates()

    paired_dates_for_filter = metric_paired_dates.copy()
    paired_dates_for_filter["subject"] = paired_dates_for_filter["subject"].astype(str)
    paired_dates_for_filter["_saline_date"] = pd.to_datetime(
        paired_dates_for_filter["saline_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    paired_dates_for_filter["_DCZ_date"] = pd.to_datetime(
        paired_dates_for_filter["DCZ_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    saline_available = available_dates_.rename(
        columns={
            "_pair_date": "_saline_date",
            "observation_group": "_saline_observation_group",
        }
    )
    dcz_available = available_dates_.rename(
        columns={
            "_pair_date": "_DCZ_date",
            "observation_group": "_DCZ_observation_group",
        }
    )

    paired_dates_for_filter = (
        paired_dates_for_filter.merge(
            saline_available,
            on=["subject", "_saline_date"],
            how="inner",
        )
        .merge(
            dcz_available,
            on=["subject", "_DCZ_date"],
            how="inner",
        )
    )
    paired_dates_for_filter = paired_dates_for_filter[
        (paired_dates_for_filter["_saline_observation_group"] == "saline")
        & (paired_dates_for_filter["_DCZ_observation_group"] == "DCZ")
    ].copy()

    metric_paired_dates = metric_paired_dates[
        metric_paired_dates["pair_id"].isin(paired_dates_for_filter["pair_id"])
    ].copy()

    paired_date_rows = pd.concat(
        [
            metric_paired_dates[["subject", "saline_date"]]
            .rename(columns={"saline_date": "_pair_date"})
            .assign(observation_group="saline"),
            metric_paired_dates[["subject", "DCZ_date"]]
            .rename(columns={"DCZ_date": "_pair_date"})
            .assign(observation_group="DCZ"),
        ],
        ignore_index=True,
    )
    paired_date_rows["subject"] = paired_date_rows["subject"].astype(str)
    paired_date_rows["_pair_date"] = pd.to_datetime(
        paired_date_rows["_pair_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    paired_date_rows = paired_date_rows.dropna().drop_duplicates()

    df_test_selected_upd["subject"] = df_test_selected_upd["subject"].astype(str)
    df_test_selected_upd = df_test_selected_upd.merge(
        paired_date_rows,
        on=["subject", "_pair_date", "observation_group"],
        how="inner",
    )

    df_test_selected_upd["session_duration"] = df_test_selected_upd.groupby(
            ["subject", "session"]
        )["trial_duration"].transform("cumsum")
    df_test_selected_upd = df_test_selected_upd[df_test_selected_upd["session_duration"] <= 2400].copy()

    # df_test_selected_upd = df_test_selected_upd[
    #     df_test_selected_upd['year_month_day'].isin(
    #         ['2026-05-22', '2026-05-20']
    #     )
    # ]

    df_test_selected_hm4 = df_test_selected_upd[
        df_test_selected_upd["subject"].isin(hM4Di_mice)
    ].copy()

    df_test_selected_hm3 = df_test_selected_upd[
        df_test_selected_upd["subject"].isin(hM3Dq_mice)
    ].copy()
    return (
        df_test_selected_hm3,
        df_test_selected_hm4,
        df_test_selected_upd,
        metric_paired_dates,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare by animals
    """)
    return


@app.function
def observation_group(observation):
    observation = str(observation).lower()
    if "dcz" in observation:
        return "DCZ"
    if "saline" in observation:
        return "saline"
    return None


@app.function
def get_metric_per_mouse_sessions(
    df,
    metric_paired_dates,
    metric_col="correct",
    agg_func="mean",
    difference=False,
):
    import pandas as pd

    if agg_func == "len":
        agg_func = len

    df_metric = df.dropna(
        subset=[
            "subject",
            "session",
            "year_month_day",
            "stimulus_modality",
            "observations",
            metric_col,
        ]
    ).copy()

    df_metric["observation_group"] = df_metric["observations"].apply(
        observation_group
    )
    df_metric = df_metric.dropna(subset=["observation_group"])

    paired_columns = ["subject", "pair_id"]
    paired_dates = pd.concat(
        [
            metric_paired_dates[paired_columns + ["saline_date"]]
            .rename(columns={"saline_date": "year_month_day"})
            .assign(observation_group="saline"),
            metric_paired_dates[paired_columns + ["DCZ_date"]]
            .rename(columns={"DCZ_date": "year_month_day"})
            .assign(observation_group="DCZ"),
        ],
        ignore_index=True,
    ).dropna()
    paired_dates["subject"] = paired_dates["subject"].astype(str)
    paired_dates["year_month_day"] = pd.to_datetime(
        paired_dates["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    paired_dates = paired_dates.dropna()
    if not difference:
        paired_dates = paired_dates.drop(columns="pair_id").drop_duplicates()

    df_metric["subject"] = df_metric["subject"].astype(str)
    df_metric["year_month_day"] = pd.to_datetime(
        df_metric["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    df_metric = df_metric.merge(
        paired_dates,
        on=["subject", "year_month_day", "observation_group"],
        how="inner",
    )

    group_columns = [
        "subject",
        "year_month_day",
        "session",
        "stimulus_modality",
        "observation_group",
    ]
    if difference:
        group_columns.insert(1, "pair_id")

    session_summary = (
        df_metric.groupby(group_columns, sort=True)
        .agg(metric_value=(metric_col, agg_func))
        .reset_index()
    )

    if difference:
        difference_output = {
            "vis": {},
            "aud": {},
        }
        for (subject, pair_id, modality), df_pair in session_summary.groupby(
            ["subject", "pair_id", "stimulus_modality"],
            sort=True,
        ):
            if modality == "visual":
                modality_key = "vis"
            elif modality == "auditory":
                modality_key = "aud"
            else:
                continue

            saline_values = df_pair.loc[
                df_pair["observation_group"] == "saline",
                "metric_value",
            ]
            dcz_values = df_pair.loc[
                df_pair["observation_group"] == "DCZ",
                "metric_value",
            ]
            if saline_values.empty or dcz_values.empty:
                continue

            difference_output[modality_key].setdefault(subject, []).append(
                saline_values.mean() - dcz_values.mean()
            )

        return difference_output

    output = {
        "vis_saline": {},
        "vis_dcz": {},
        "aud_saline": {},
        "aud_dcz": {},
    }

    for (modality, observation), df_condition in session_summary.groupby(
        ["stimulus_modality", "observation_group"]
    ):
        modality_prefix = "vis" if modality == "visual" else "aud"
        observation_suffix = observation.lower()
        key = f"{modality_prefix}_{observation_suffix}"

        output[key] = {
            subject: df_mouse["metric_value"].tolist()
            for subject, df_mouse in df_condition.groupby("subject")
        }

    return output


@app.cell
def _(mo):
    compare_value_settings = {
        "performance": {
            "metric_col": "correct",
            "agg_func": "mean",
            "ylabel": "Session mean performance",
        },
        "number of central pokes": {
            "metric_col": "port2_pokes_num",
            "agg_func": "mean",
            "ylabel": "Session mean number of central pokes",
        },
        "reaction time": {
            "metric_col": "reaction_time",
            "agg_func": "mean",
            "ylabel": "Session mean reaction time",
        },
        "time interval": {
            "metric_col": "inter_trial_interval",
            "agg_func": "mean",
            "ylabel": "Session mean time interval",
        },
        "number of trials": {
            "metric_col": "trial",
            "agg_func": "len",
            "ylabel": "Session total number of trials",
        },
        "trial duration": {
            "metric_col": "trial_duration",
            "agg_func": "mean",
            "ylabel": "Session mean trial duration",
        },
    }
    compare_value_select = mo.ui.dropdown(
        options=list(compare_value_settings),
        value="performance",
        label="Compare value",
    )
    compare_value_select
    return compare_value_select, compare_value_settings


@app.cell
def _(
    compare_value_select,
    compare_value_settings,
    df_test_selected_hm3,
    df_test_selected_hm4,
    metric_paired_dates,
):
    compare_value_setting = compare_value_settings[compare_value_select.value]
    compare_metric_col = compare_value_setting["metric_col"]
    compare_agg_func = compare_value_setting["agg_func"]
    compare_y_label = compare_value_setting["ylabel"]

    output_per_mouse_dic_hm4 = get_metric_per_mouse_sessions(
        df_test_selected_hm4,
        metric_paired_dates,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
    )
    output_per_mouse_dic_hm3 = get_metric_per_mouse_sessions(
        df_test_selected_hm3,
        metric_paired_dates,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
    )
    output_per_mouse_dic_hm4_diff = get_metric_per_mouse_sessions(
        df_test_selected_hm4,
        metric_paired_dates,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
        difference=True
    )
    output_per_mouse_dic_hm3_diff = get_metric_per_mouse_sessions(
        df_test_selected_hm3,
        metric_paired_dates,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
        difference=True
    )
    return (
        compare_y_label,
        output_per_mouse_dic_hm3,
        output_per_mouse_dic_hm3_diff,
        output_per_mouse_dic_hm4,
        output_per_mouse_dic_hm4_diff,
    )


@app.cell
def _(
    compare_y_label,
    mo,
    output_per_mouse_dic_hm3,
    output_per_mouse_dic_hm3_diff,
    output_per_mouse_dic_hm4,
    output_per_mouse_dic_hm4_diff,
    plot_test,
):
    hm4_condition_session_fig = plot_test.plot_condition_session_values_by_mouse(
        output_per_mouse_dic_hm4,
        "hM4Di",
        ylabel=compare_y_label,
        title=f"hM4Di: {compare_y_label}",
    )
    hm3_condition_session_fig = plot_test.plot_condition_session_values_by_mouse(
        output_per_mouse_dic_hm3,
        "hM3Dq",
        ylabel=compare_y_label,
        title=f"hM3Di: {compare_y_label}",
    )
    condition_diff_session_fig = plot_test.plot_group_condition_values_by_mouse(
        output_per_mouse_dic_hm3_diff,
        output_per_mouse_dic_hm4_diff,
        ylabel=f"saline - DCZ ({compare_y_label})",
        title=f"saline - DCZ: {compare_y_label}",
    )
    mo.vstack(
        [
            hm4_condition_session_fig,
            hm3_condition_session_fig,
            condition_diff_session_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare the trial with time
    """)
    return


@app.cell
def _(df_test_selected_upd, mo, plots, plt, utils):
    trial_time_figures = []

    for subject, df_mouse in df_test_selected_upd.groupby("subject", sort=True):
        df_mouse = utils.add_time_from_session_start(df_mouse.copy())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, modality in zip(axes, ["visual", "auditory"]):
            df_modality = df_mouse[
                df_mouse["stimulus_modality"] == modality
            ]

            for observation, color in [
                ("saline", "steelblue"),
                ("dcz", "firebrick"),
            ]:
                df_observation = df_modality[
                    df_modality["observations"].str.contains(
                        observation,
                        case=False,
                        na=False,
                    )
                ]

                if df_observation.empty:
                    continue

                collection_start = len(ax.collections)
                plots.plot_trial_time_of_start(
                    df_observation,
                    ax=ax,
                )

                for collection in ax.collections[collection_start:]:
                    collection.set_facecolor(color)
                    collection.set_edgecolor(color)

            ax.scatter([], [], color="steelblue", s=16, label="saline")
            ax.scatter([], [], color="firebrick", s=16, label="DCZ")
            ax.set_title(modality.capitalize())
            ax.legend(frameon=False)

        fig.suptitle(str(subject))
        fig.tight_layout()
        trial_time_figures.append(fig)

    mo.vstack(trial_time_figures) if trial_time_figures else mo.md(
        "No paired saline/DCZ sessions to plot."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare by sessions
    """)
    return


@app.cell
def _(behavior_utils, metric_paired_dates, pd, plt):
    from scipy import stats

    def observation_to_color(observation):
        observation = str(observation).lower()
        if "dcz" in observation:
            return "red"
        if "saline" in observation:
            return "blue"
        return "gray"


    def p_value_to_label(p_value):
        if p_value < 0.0001:
            return "****"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    def add_significance_bar(ax, x1, x2, y, h, label):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
        ax.text(
            (x1 + x2) / 2,
            y + h,
            label,
            ha="center",
            va="bottom",
            color="black",
        )


    def plot_metric_by_date(
        df,
        metric_col,
        agg_func="mean",
        subject_col="subject",
        date_col="year_month_day",
        observation_col="observations",
        ylabel=None,
        title=None,
        figsize=(10, 5),
    ):
        required_columns = {
            subject_col,
            date_col,
            metric_col,
            observation_col,
        }
        if agg_func == "len":
            agg_func = len

        metric_by_subject = {}
        metric_summary = pd.DataFrame()

        fig, ax = plt.subplots(figsize=figsize)
        missing_columns = required_columns - set(df.columns)


        plot_df = df.dropna(subset=[subject_col, date_col]).copy()
        plot_df[subject_col] = plot_df[subject_col].astype(str)
        plot_df[date_col] = plot_df[date_col].astype(str)
        plot_df[observation_col] = plot_df[observation_col].astype(str)
        metric_summary = (
            plot_df.groupby([subject_col, date_col], sort=True)
            .agg(
                **{
                    metric_col: (metric_col, agg_func),
                    observation_col: (
                        observation_col,
                        lambda x: x.dropna().unique()[0],
                    ),
                }
            )
            .reset_index()
        )

        if metric_summary.empty:
            ax.text(
                0.5,
                0.5,
                "No valid subject/date data to plot.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            all_dates = sorted(metric_summary[date_col].unique())
            for subject, df_per_mouse in metric_summary.groupby(subject_col, sort=True):
                mouse_series = (
                    df_per_mouse.set_index(date_col)[metric_col]
                    .sort_index()
                )
                metric_by_subject[subject] = mouse_series
                ax.plot(
                    all_dates,
                    mouse_series.reindex(all_dates).values,
                    linewidth=1.8,
                    label=subject,
                    alpha=0.6,
                )
                ax.scatter(
                    df_per_mouse[date_col],
                    df_per_mouse[metric_col],
                    color=[
                        observation_to_color(observation)
                        for observation in df_per_mouse[observation_col]
                    ],
                    edgecolor="black",
                    linewidth=0.5,
                    s=45,
                    zorder=3,
                )

            ax.scatter([], [], color="blue", edgecolor="black", label="saline")
            ax.scatter([], [], color="red", edgecolor="black", label="DCZ")
            ax.set_xlabel("Date")
            ax.set_ylabel(ylabel or metric_col)
            ax.set_title(title or f"{metric_col} by date")
            ax.legend(title="Mouse")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return fig, metric_by_subject, metric_summary

    def plot_metric_by_observation(
        df,
        metric_col,
        agg_func="mean",
        subject_col="subject",
        date_col="year_month_day",
        observation_col="observations",
        ylabel=None,
        title=None,
        figsize=(6, 5),
        test_method="wilcoxon",
    ):
        required_columns = {
            subject_col,
            date_col,
            metric_col,
            observation_col,
        }
        if agg_func == "len":
            agg_func = len

        observation_summary = pd.DataFrame()

        fig, ax = plt.subplots(figsize=figsize)
        missing_columns = required_columns - set(df.columns)

        plot_df = df.dropna(subset=[subject_col, date_col]).copy()
        plot_df["observation_group"] = plot_df[observation_col].apply(
            observation_group
        )
        plot_df = plot_df.dropna(subset=["observation_group"])

        observation_summary = (
            plot_df.groupby(
                [subject_col, date_col, "observation_group"],
                sort=True,
            )
            .agg(**{metric_col: (metric_col, agg_func)})
            .reset_index()
        )

        plot_subjects = plot_df[subject_col].astype(str).unique()
        paired_dates = metric_paired_dates[
            metric_paired_dates[subject_col].astype(str).isin(plot_subjects)
        ].copy()
        paired_summary = behavior_utils.add_metric_values_to_pairs(
            observation_summary,
            paired_dates,
            metric_col=metric_col,
            subject_col=subject_col,
            date_col=date_col,
        )
        observation_summary.attrs["paired_comparison"] = paired_summary

        groups = ["saline", "DCZ"]
        colors = {"saline": "blue", "DCZ": "red"}
        data = [
            pd.to_numeric(
                paired_summary[group] if group in paired_summary else pd.Series(),
                errors="coerce",
            ).dropna()
            for group in groups
        ]

        if all(group_data.empty for group_data in data):
            ax.text(
                0.5,
                0.5,
                "No paired previous-saline/DCZ observations to compare.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            boxplot = ax.boxplot(
                data,
                labels=groups,
                patch_artist=True,
                showmeans=True,
            )
            for patch, group in zip(boxplot["boxes"], groups):
                patch.set_facecolor(colors[group])
                patch.set_alpha(0.2)
                patch.set_edgecolor(colors[group])

            for _, paired_row in paired_summary.iterrows():
                ax.plot(
                    [1, 2],
                    [paired_row["saline"], paired_row["DCZ"]],
                    color="gray",
                    alpha=0.35,
                    linewidth=1,
                    zorder=2,
                )

            for idx, (group, group_data) in enumerate(zip(groups, data), start=1):
                ax.scatter(
                    [idx] * len(group_data),
                    group_data,
                    color=colors[group],
                    edgecolor="black",
                    linewidth=0.5,
                    s=45,
                    alpha=0.8,
                    zorder=3,
                )

            if all(len(group_data) > 0 for group_data in data):
                if test_method == "wilcoxon":
                    try:
                        p_value = stats.wilcoxon(
                            data[0],
                            data[1],
                        ).pvalue
                    except ValueError:
                        p_value = 1.0
                elif test_method == "ttest_rel":
                    p_value = stats.ttest_rel(
                        data[0],
                        data[1],
                        nan_policy="omit",
                    ).pvalue
                elif test_method == "mannwhitneyu":
                    p_value = stats.mannwhitneyu(
                        data[0],
                        data[1],
                        alternative="two-sided",
                    ).pvalue
                elif test_method == "ttest_ind":
                    p_value = stats.ttest_ind(
                        data[0],
                        data[1],
                        equal_var=False,
                        nan_policy="omit",
                    ).pvalue
                else:
                    raise ValueError(
                        "test_method must be 'wilcoxon', 'ttest_rel', "
                        "'mannwhitneyu', or 'ttest_ind'"
                    )

                observation_summary.attrs["test_method"] = test_method
                observation_summary.attrs["p_value"] = p_value
                observation_summary.attrs["comparison_mode"] = (
                    "nearest_previous_saline_for_each_DCZ"
                )
                y_values = pd.concat(data)
                y_min = y_values.min()
                y_max = y_values.max()
                y_range = y_max - y_min
                if y_range == 0:
                    y_range = abs(y_max) * 0.1 if y_max != 0 else 1
                bar_y = y_max + y_range * 0.12
                bar_h = y_range * 0.06
                add_significance_bar(
                    ax,
                    1,
                    2,
                    bar_y,
                    bar_h,
                    p_value_to_label(p_value),
                )
                ax.set_ylim(top=bar_y + bar_h + y_range * 0.15)

            ax.set_xlabel("Observation")
            ax.set_ylabel(ylabel or metric_col)
            ax.set_title(title or f"{metric_col} by observation")
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return fig, observation_summary

    return plot_metric_by_date, plot_metric_by_observation


@app.cell
def _(plot_metric_by_date):
    def plot_group(
        df_group,
        group_name,
        metric_col="port2_pokes_num",
        metric_name=None,
        agg_func="mean",
        plot_func=plot_metric_by_date,
    ):
        metric_name = metric_name or metric_col.replace("_", " ")
        plot_agg_func = len if agg_func == "len" else agg_func
        ylabel = metric_name if agg_func == "len" else f"{agg_func} {metric_name}"

        return plot_func(
            df_group,
            metric_col=metric_col,
            agg_func=plot_agg_func,
            ylabel=ylabel,
            title=f"{metric_name} in {group_name} group",
        )[0]

    return (plot_group,)


@app.cell
def _(mo):
    observation_select = mo.ui.dropdown(
        options=["fixation_breaks", "eager_pokes", "port2_pokes_num", "reaction_time", "time_between_trials", "trial", "correct", "trial_duration"],
        value="fixation_breaks",
        label="observation",
    )

    observation_select
    return (observation_select,)


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    observation_select,
    plot_group,
    plot_metric_by_date,
):
    if observation_select.value == "trial":
        _aggfunc = "len"
    else:
        _aggfunc = "mean"
    mo.vstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col=observation_select.value, metric_name=observation_select.value, agg_func=_aggfunc, plot_func=plot_metric_by_date), 
            plot_group(df_test_selected_hm3, "hM3Di", metric_col=observation_select.value, metric_name=observation_select.value, agg_func=_aggfunc, plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    observation_select,
    plot_group,
    plot_metric_by_observation,
):
    if observation_select.value == "trial":
        _aggfunc = "len"
    else:
        _aggfunc = "mean"
    mo.hstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col=observation_select.value, metric_name=observation_select.value, agg_func=_aggfunc, plot_func=plot_metric_by_observation), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col=observation_select.value, metric_name=observation_select.value, agg_func=_aggfunc, plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## sampled for plotting
    """)
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_date,
):
    hm4_sub_bydate_fig_aud = plot_group(
        df_test_selected_hm4[
            (df_test_selected_hm4["subject"].isin(["NUO001", "NUO002", "NUO005"])) 
            # & (df_test_selected_hm4['stimulus_modality']=='auditory')
            ], "hM4Dq", metric_col="reaction_time", metric_name="reaction time", agg_func="mean", plot_func=plot_metric_by_date)
    hm4_sub_bydate_fig_aud.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm4_sub_bydate_fig_aud.svg", format="svg", bbox_inches="tight")
    hm3_sub_bydate_fig_aud = plot_group(
        df_test_selected_hm3[
            (df_test_selected_hm3["subject"].isin(["NUO010", "NUO012"])) 
            # & (df_test_selected_hm3['stimulus_modality']=='auditory')
            ], "hM3Dq", metric_col="reaction_time", metric_name="reaction time", agg_func="mean", plot_func=plot_metric_by_date)
    hm3_sub_bydate_fig_aud.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_sub_bydate_fig_aud.svg", format="svg", bbox_inches="tight")
    mo.vstack(
        [
            hm4_sub_bydate_fig_aud,
            hm3_sub_bydate_fig_aud
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    hm4_sub_byobserve_fig_aud = plot_group(
        df_test_selected_hm4[
            (df_test_selected_hm4["subject"].isin(hM4Di_mice)) 
            # & (df_test_selected_hm4['stimulus_modality']=='auditory')
            ], "hM4Dq", metric_col="correct", metric_name="performance", agg_func="mean", plot_func=plot_metric_by_observation)
    hm4_sub_byobserve_fig_aud.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm4_sub_byobserve_fig_aud.svg", format="svg", bbox_inches="tight")
    hm3_sub_byobserve_fig_aud = plot_group(df_test_selected_hm3[
        (df_test_selected_hm3["subject"].isin(hM3Dq_mice)) 
        # & (df_test_selected_hm3['stimulus_modality']=='auditory')
        ], "hM3Dq", metric_col="correct", metric_name="performance", agg_func="mean", plot_func=plot_metric_by_observation)
    hm3_sub_byobserve_fig_aud.savefig("/mnt/e/data/LeciLab/behavioral_data/tmp/for_fens_tmp/hm3_sub_byobserve_fig_aud.svg", format="svg", bbox_inches="tight")
    mo.hstack(
        [
            hm4_sub_byobserve_fig_aud,
            hm3_sub_byobserve_fig_aud
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # work bout
    """)
    return


@app.cell
def _(mo):
    work_bout_modality = mo.ui.dropdown(
        options=["visual", "auditory", "both"],
        value="visual",
        label="Modality",
        allow_select_none=False,
    )
    work_bout_modality
    return (work_bout_modality,)


@app.cell
def _(df_test_selected_upd, work_bout_modality):
    _selected_modality = work_bout_modality.value
    dcz_bout_modalities = (
        ("visual", "auditory") if _selected_modality == "both" else (_selected_modality,)
    )
    dcz_bout_trials = df_test_selected_upd.loc[
        df_test_selected_upd["stimulus_modality"].isin(dcz_bout_modalities)
    ]
    return dcz_bout_modalities, dcz_bout_trials


@app.cell
def _(hM3Dq_mice, hM4Di_mice):
    dcz_bout_cohorts = {"hM3Dq": hM3Dq_mice, "hM4Di": hM4Di_mice}
    dcz_bout_palette = {"saline": "blue", "DCZ": "red"}
    return dcz_bout_cohorts, dcz_bout_palette


@app.cell
def _(
    dcz_bout_cohorts,
    dcz_bout_modalities,
    dcz_bout_palette,
    dcz_bout_trials,
    plt,
    sns,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for _ax, (_group, _mice) in zip(_axes, dcz_bout_cohorts.items()):
        _df = dcz_bout_trials.loc[
            dcz_bout_trials["subject"].isin(_mice)
            & dcz_bout_trials["inter_trial_interval"].gt(0)
        ]
        if not _df.empty:
            sns.ecdfplot(
                data=_df,
                x="inter_trial_interval",
                hue="observation_group",
                hue_order=list(dcz_bout_palette),
                palette=dcz_bout_palette,
                ax=_ax,
            )
        else:
            _ax.text(0.5, 0.5, "No data", ha="center", transform=_ax.transAxes)
        _ax.set(
            title=_group, xscale="log", xlabel="Inter-trial interval (s)",
            ylabel="Cumulative probability", ylim=(0, 1),
        )
        _ax.grid(alpha=0.2)
    _fig.suptitle(f"Inter-trial intervals | {' + '.join(dcz_bout_modalities)}")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(dcz_bout_trials, np, pd):
    import ast

    _keys = ["subject", "year_month_day", "session", "observation_group"]
    _df = dcz_bout_trials.sort_values(_keys + ["trial"]).copy()
    _g = _df.groupby(_keys)

    _df["bout_id"] = (
        _g.cumcount().eq(0)
        | _g["engaged"].shift().ne(_df["engaged"])
    ).cumsum()

    work_bouts = (
        _df.groupby(_keys + ["bout_id"], observed=True)
        .agg(
            engaged=("engaged", "first"),
            n_trials=("trial", "size"),
            start_time=("TRIAL_START", "first"),
            end_time=("TRIAL_END", "last"),
        )
        .reset_index()
    )

    work_bouts["duration_s"] = (
        work_bouts["end_time"] - work_bouts["start_time"]
    )

    _bg = work_bouts.groupby(_keys, observed=True)

    work_bouts["break_onset"] = (
        work_bouts["engaged"].eq(False)
        & _bg["engaged"].shift().eq(True)
    )

    work_bouts["complete"] = (
        _bg.cumcount().gt(0)
        & _bg.cumcount(ascending=False).gt(0)
    )

    def has_poke(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return np.isfinite(np.asarray(value, dtype=float)).any()

    initiated_counts = (
        _df.assign(initiated=_df["Port2In"].apply(has_poke))
        .groupby(_keys, observed=True)["initiated"]
        .sum()
    )

    _rows = []

    for _key, _bout in work_bouts.groupby(_keys, observed=True):
        working = _bout[_bout["engaged"].eq(True)]
        complete_working = working[working["complete"]]
        complete_breaks = _bout[_bout["engaged"].eq(False) & _bout["complete"]]

        # Observation window: from the first trial start to the last trial end
        session_s = _bout["end_time"].max() - _bout["start_time"].min()
        session_s = session_s if session_s > 0 else np.nan
        session_h = session_s / 3600

        _rows.append({
            **dict(zip(_keys, _key)),
            "session_duration_s": session_s,

            # Observed True-to-False transitions per hour
            "break_frequency": _bout["break_onset"].sum() / session_h,

            # Mean duration of complete breaks within the session
            "break_duration": complete_breaks["duration_s"].mean(),

            # Mean trial count and duration of complete working bouts within the session
            "trials_per_bout": complete_working["n_trials"].mean(),
            "bout_duration": complete_working["duration_s"].mean(),

            # Observed working time, including incomplete bouts at the session boundaries
            "working_time_fraction": working["duration_s"].sum() / session_s,

            # Trials with a center-port poke per hour
            "trial_initiation_rate": initiated_counts.loc[_key] / session_h,

            "n_break_onsets": int(_bout["break_onset"].sum()),
            "n_initiated_trials": int(initiated_counts.loc[_key]),
            "working_time_s": working["duration_s"].sum(),
        })

    session_metrics = pd.DataFrame(_rows, columns=[
        *_keys, "session_duration_s", "break_frequency", "break_duration",
        "trials_per_bout", "bout_duration", "working_time_fraction",
        "trial_initiation_rate", "n_break_onsets", "n_initiated_trials",
        "working_time_s",
    ])
    session_metrics
    return session_metrics, work_bouts


@app.cell
def _(
    dcz_bout_cohorts,
    dcz_bout_modalities,
    dcz_bout_trials,
    metric_paired_dates,
    pd,
    plt,
    work_bouts,
):
    _keys = ["subject", "year_month_day", "session", "observation_group"]
    _rows = [
        (_modality, _group, _mice)
        for _modality in dcz_bout_modalities
        for _group, _mice in dcz_bout_cohorts.items()
    ]
    _palettes = {"saline": ("steelblue", "limegreen"), "DCZ": ("firebrick", "gold")}
    _fig, _axes = plt.subplots(len(_rows), 2, figsize=(11, 3.5 * len(_rows)), squeeze=False)

    for _axes_row, (_modality, _group, _mice) in zip(_axes, _rows):
        _candidates = dcz_bout_trials.loc[
            dcz_bout_trials["stimulus_modality"].eq(_modality)
        ]
        _dates = pd.to_datetime(_candidates["year_month_day"], errors="coerce").dt.normalize()
        _pairs = metric_paired_dates.loc[metric_paired_dates["subject"].isin(_mice)]
        for _pair in _pairs.sample(frac=1, random_state=4).itertuples(index=False):
            _trials = {
                _condition: _candidates.loc[
                    _candidates["subject"].eq(_pair.subject)
                    & _candidates["observation_group"].eq(_condition)
                    & _dates.eq(pd.Timestamp(getattr(_pair, f"{_condition}_date")).normalize())
                ]
                for _condition in _palettes
            }
            if all(not _df.empty for _df in _trials.values()):
                break
        else:
            for _ax, _condition in zip(_axes_row, _palettes):
                _ax.set_title(f"{_group} | {_condition}: no paired {_modality} sessions")
                _ax.set_axis_off()
            continue

        for _ax, (_condition, _df) in zip(_axes_row, _trials.items()):
            _session = _df[_keys].sort_values("session").iloc[[0]]
            _df = _df.merge(_session, on=_keys)
            _start = _df["TRIAL_START"].min()
            _bouts = work_bouts.loc[work_bouts["engaged"].eq(True)].merge(_session, on=_keys)
            for _bout in _bouts.itertuples(index=False):
                _ax.axvspan(_bout.start_time - _start, _bout.end_time - _start,
                            color="lightskyblue", alpha=0.2, linewidth=0)
            for _engaged, _color in zip([True, False], _palettes[_condition]):
                _points = _df.loc[_df["engaged"].eq(_engaged)]
                _ax.scatter(_points["TRIAL_START"] - _start, _points["trial"],
                            color=_color, s=5, label=f"Engaged: {_engaged}")
            _date = _session["year_month_day"].iat[0]
            _ax.set(title=f"{_group} | {_modality} | {_condition}\n{_pair.subject} | {_date}",
                    xlabel="Trial start time (s)", ylabel="Trial number", xlim=(0, None))
            _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    dcz_bout_metric_select = mo.ui.dropdown(
        options={
            "Bout duration (s)": "duration_s",
            "Trials per bout": "n_trials",
            "Break onset per bout": "break_onset",
        },
        value="Bout duration (s)",
        label="Bout metric",
        allow_select_none=False,
    )
    dcz_bout_metric_select
    return (dcz_bout_metric_select,)


@app.cell
def _(
    dcz_bout_cohorts,
    dcz_bout_metric_select,
    dcz_bout_modalities,
    dcz_bout_palette,
    mo,
    plt,
    sns,
    work_bouts,
):
    _metric = dcz_bout_metric_select.value
    _label = {"duration_s": "Bout duration (s)", "n_trials": "Trials per bout",
              "break_onset": "Break onset per bout"}[_metric]
    _figures = []
    for _group, _mice in dcz_bout_cohorts.items():
        _fig, _axes = plt.subplots(2, 2, figsize=(10, 7))
        for _j, _engaged in enumerate([True, False]):
            _df = work_bouts.loc[
                work_bouts["subject"].isin(_mice) & work_bouts["engaged"].eq(_engaged)
            ].dropna(subset=[_metric]).copy()
            _df[_metric] = _df[_metric].astype(float)
            _box_ax, _ecdf_ax = _axes[:, _j]
            _title = "Working bouts" if _engaged else "Break bouts"
            _box_ax.set_title(_title)
            _ecdf_ax.set_title(f"{_title} (ECDF)")
            if _df.empty:
                for _ax in (_box_ax, _ecdf_ax):
                    _ax.text(0.5, 0.5, "No data", ha="center", transform=_ax.transAxes)
                continue
            _order = list(dcz_bout_palette)
            sns.boxplot(data=_df, x="observation_group", y=_metric,
                        hue="observation_group", order=_order, hue_order=_order,
                        palette=dcz_bout_palette, fill=False, dodge=False,
                        legend=False, ax=_box_ax)
            sns.ecdfplot(data=_df, x=_metric, hue="observation_group", hue_order=_order,
                         palette=dcz_bout_palette, ax=_ecdf_ax)
            if _metric != "break_onset" and _df[_metric].gt(0).all():
                _ecdf_ax.set_xscale("log")
            _box_ax.set(xlabel="", ylabel=_label)
            _ecdf_ax.set(xlabel=_label, ylabel="Cumulative probability", ylim=(0, 1))
            _ecdf_ax.grid(alpha=0.2)
        _fig.suptitle(f"{_group} | {' + '.join(dcz_bout_modalities)}")
        _fig.tight_layout()
        plt.close(_fig)
        _figures.append(_fig)
    mo.vstack(_figures)
    return


@app.cell
def _(session_metrics, work_bouts):
    _keys = ["subject", "observation_group"]
    dcz_bout_mouse_metrics = session_metrics.groupby(_keys, observed=True, as_index=False).agg(
        n_sessions=("session", "size"),
        observed_duration_s=("session_duration_s", "sum"),
        n_break_onsets=("n_break_onsets", "sum"),
        n_initiated_trials=("n_initiated_trials", "sum"),
        working_time_s=("working_time_s", "sum"),
    )
    _complete = work_bouts.loc[work_bouts["complete"]]
    _working = _complete.loc[_complete["engaged"].eq(True)].groupby(_keys, observed=True).agg(
        trials_per_bout=("n_trials", "mean"), bout_duration=("duration_s", "mean"),
    )
    _breaks = (
        _complete.loc[_complete["engaged"].eq(False)]
        .groupby(_keys, observed=True)["duration_s"].mean().rename("break_duration")
    )
    dcz_bout_mouse_metrics = dcz_bout_mouse_metrics.join(_working, on=_keys).join(_breaks, on=_keys)
    _duration_s = dcz_bout_mouse_metrics["observed_duration_s"].where(
        dcz_bout_mouse_metrics["observed_duration_s"].gt(0)
    )
    dcz_bout_mouse_metrics["break_frequency"] = dcz_bout_mouse_metrics["n_break_onsets"] / (_duration_s / 3600)
    dcz_bout_mouse_metrics["trial_initiation_rate"] = dcz_bout_mouse_metrics["n_initiated_trials"] / (_duration_s / 3600)
    dcz_bout_mouse_metrics["working_time_fraction"] = dcz_bout_mouse_metrics["working_time_s"] / _duration_s
    return (dcz_bout_mouse_metrics,)


@app.cell
def _(
    behavior_utils,
    dcz_bout_cohorts,
    dcz_bout_modalities,
    dcz_bout_mouse_metrics,
    dcz_bout_palette,
    metric_paired_dates,
    mo,
    np,
    pd,
    plot_test,
    plt,
    session_metrics,
):
    from scipy import stats as _stats

    _metrics = {
        "break_frequency": "Break onsets / observed hour",
        "break_duration": "Mean break duration (s)",
        "trials_per_bout": "Mean trials per working bout",
        "bout_duration": "Mean working bout duration (s)",
        "working_time_fraction": "Working time fraction",
        "trial_initiation_rate": "Trials initiated / observed hour",
    }
    _conditions = list(dcz_bout_palette)
    _sessions = session_metrics.assign(
        year_month_day=pd.to_datetime(session_metrics["year_month_day"], errors="coerce").dt.normalize()
    )
    _daily = (
        _sessions.replace([np.inf, -np.inf], np.nan)
        .groupby(["subject", "year_month_day", "observation_group"], observed=True, as_index=False)
        [list(_metrics)].mean()
    )

    def _plot_paired(_ax, _paired, _label, _detail, _p):
        _ax.set(title=f"{_label}\n{_detail}", ylabel=_label)
        _ax.set_xticks([1, 2], _conditions)
        if _paired.empty:
            _ax.text(0.5, 0.5, "No complete pairs", ha="center", transform=_ax.transAxes)
            return
        _values = [_paired[_condition].to_numpy(dtype=float) for _condition in _conditions]
        _boxes = _ax.boxplot(_values, positions=[1, 2], patch_artist=True,
                             showmeans=True, showfliers=False)
        for _box, _condition in zip(_boxes["boxes"], _conditions):
            _box.set(facecolor=dcz_bout_palette[_condition], alpha=0.2)
        _ax.plot([1, 2], np.vstack(_values), color="gray", alpha=0.35, linewidth=1)
        for _x, (_condition, _ys) in enumerate(zip(_conditions, _values), 1):
            _ax.scatter(np.full(len(_ys), _x), _ys, color=dcz_bout_palette[_condition],
                        edgecolor="black", linewidth=0.5, alpha=0.8, s=32, zorder=3)
        _tag = plot_test.p_to_star(_p) if np.isfinite(_p) else "n/a"
        plot_test.add_paired_significance_label(_ax, *_values, label=_tag, x0=1, x1=2)
        _ax.grid(axis="y", alpha=0.3)

    dcz_bout_date_pairs = {}
    dcz_bout_mouse_pairs = {}
    _test_rows = []
    _views = []
    for _group, _mice in dcz_bout_cohorts.items():
        _dates = metric_paired_dates.loc[metric_paired_dates["subject"].isin(_mice)]
        _pooled = dcz_bout_mouse_metrics.loc[dcz_bout_mouse_metrics["subject"].isin(_mice)]
        for _level, _method in [("Date pairs", "ttest_rel"), ("Pooled bouts per mouse", "wilcoxon")]:
            _fig, _axes = plt.subplots(2, 3, figsize=(13, 8))
            for _ax, (_metric, _label) in zip(_axes.flat, _metrics.items()):
                if _method == "ttest_rel":
                    _paired = behavior_utils.add_metric_values_to_pairs(
                        _daily, _dates, metric_col=_metric, drug_name="DCZ"
                    )
                    _mice_pairs = (
                        _paired.groupby("subject", observed=True)[_conditions].mean()
                        if not _paired.empty else pd.DataFrame(columns=_conditions)
                    )
                    _detail = f"{len(_paired)} date pairs, {len(_mice_pairs)} mice"
                    dcz_bout_date_pairs[_group, _metric] = _paired
                else:
                    _mice_pairs = (
                        _pooled.pivot(index="subject", columns="observation_group", values=_metric)
                        .reindex(columns=_conditions).replace([np.inf, -np.inf], np.nan).dropna()
                    )
                    _paired = _mice_pairs
                    _detail = f"{len(_mice_pairs)} paired mice"
                    dcz_bout_mouse_pairs[_group, _metric] = _mice_pairs

                _diff = _mice_pairs["DCZ"] - _mice_pairs["saline"]
                _p, _note = np.nan, ""
                if len(_diff) < 2:
                    _note = "Fewer than 2 paired mice"
                elif _diff.eq(0).all():
                    _note = "All paired differences are zero"
                elif _method == "wilcoxon":
                    _p = _stats.wilcoxon(_diff).pvalue
                elif _diff.std() == 0:
                    _note = "Zero variance of paired differences"
                else:
                    _p = _stats.ttest_rel(_mice_pairs["DCZ"], _mice_pairs["saline"]).pvalue
                _plot_paired(_ax, _paired, _label, _detail, _p)
                _test_rows.append({
                    "group": _group, "level": _level, "metric": _metric, "test": _method,
                    "n_date_pairs": len(_paired) if _method == "ttest_rel" else pd.NA,
                    "n_mice": len(_mice_pairs), "p_value": _p, "note": _note,
                })
            _fig.suptitle(f"{_group}: saline vs DCZ | {_level} | {' + '.join(dcz_bout_modalities)}")
            _fig.tight_layout()
            plt.close(_fig)
            _views.append(_fig)

    dcz_bout_test_results = pd.DataFrame(_test_rows)
    mo.vstack([*_views, dcz_bout_test_results])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Psychometric curve
    """)
    return


@app.cell
def _(stimulus_modality_select):
    if stimulus_modality_select.value == "visual":
        stimulus_col = "visual_stimulus_ratio"
    elif stimulus_modality_select.value == "auditory":
        stimulus_col = "total_evidence_strength"
    else:
        stimulus_col = "total_evidence_strength"
    return (stimulus_col,)


@app.cell
def _(
    behavior_utils,
    df_test_selected_hm3,
    df_test_selected_hm4,
    hM3Dq_mice,
    hM4Di_mice,
    injection_info_df,
    mo,
    plot_test,
    stimulus_col,
):
    psychometric_hm4_paired_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=hM4Di_mice,
    )
    psychometric_hm3_paired_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=hM3Dq_mice,
    )

    psychometric_hm4_fig, psychometric_hm4_summary = (
        plot_test.plot_condition_psychometric_curves(
            df_test_selected_hm4,
            "hM4Di",
            paired_dates=psychometric_hm4_paired_dates,
            x_col=stimulus_col,
            y_col="first_choice_numeric",
            valueType="continue",
            bins=6,
            min_trials=20,
        )
    )

    psychometric_hm3_fig, psychometric_hm3_summary = (
        plot_test.plot_condition_psychometric_curves(
            df_test_selected_hm3,
            "hM3Dq",
            paired_dates=psychometric_hm3_paired_dates,
            x_col=stimulus_col,
            y_col="first_choice_numeric",
            valueType="continue",
            bins=6,
            min_trials=20,
        )
    )

    mo.hstack([psychometric_hm4_fig, psychometric_hm3_fig])
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_date,
):
    mo.vstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_date,
):
    mo.vstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_date,
):
    mo.vstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_selected_hm3,
    df_test_selected_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_selected_hm4, "hM4Di", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_selected_hm3, "hM3Dq", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell
def _():
    noeffec_group = ['NUO005', 'NUO008']
    effec_group = ['NUO010', 'NUO012']
    return effec_group, noeffec_group


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GLM-HMM for effective DCZ vs saline/no-effect sessions
    """)
    return


@app.cell
def _(df_test_selected_upd, effec_group, noeffec_group, pd):
    effec_glmhmm_stim_col = "total_evidence_strength"

    df_effec_glmhmm = df_test_selected_upd.copy()
    _subject = df_effec_glmhmm["subject"].astype(str)
    _observation = df_effec_glmhmm["observations"].astype(str).str.lower()

    _effec_dcz_mask = _subject.isin(effec_group) & _observation.str.contains(
        "dcz",
        na=False,
    )
    _saline_noeffec_mask = _observation.str.contains(
        "saline",
        na=False,
    ) | _subject.isin(noeffec_group)

    df_effec_glmhmm["selected_df_option"] = pd.NA
    df_effec_glmhmm.loc[_effec_dcz_mask, "selected_df_option"] = "effec_dcz"
    df_effec_glmhmm.loc[
        _saline_noeffec_mask & ~_effec_dcz_mask,
        "selected_df_option",
    ] = "saline_noeffec"
    df_effec_glmhmm = df_effec_glmhmm.dropna(subset=["selected_df_option"]).copy()
    df_effec_glmhmm["session"] = (
        df_effec_glmhmm["subject"].astype(str)
        + "__"
        + df_effec_glmhmm["session"].astype(str)
        + "__"
        + df_effec_glmhmm["selected_df_option"].astype(str)
    )

    effec_glmhmm_session_summary = (
        df_effec_glmhmm.groupby(
            ["selected_df_option", "subject", "session"],
            sort=True,
        )
        .size()
        .reset_index(name="n_trials")
    )
    effec_glmhmm_session_summary
    return df_effec_glmhmm, effec_glmhmm_stim_col


@app.cell
def _(np):
    v0 = np.array([0.0, 1.0, 1.0, 0.0, -2.0])
    return (v0,)


@app.cell
def _(
    Path,
    df_effec_glmhmm,
    effec_group,
    minimize,
    mo,
    noeffec_group,
    pd,
    plt,
    update_processData_button,
    utils_test,
    v0,
):
    _condition_order = ["saline_noeffec", "effec_dcz"]
    _saved_mice = list(effec_group) + list(noeffec_group)
    _saved_label = "__".join(_saved_mice) if _saved_mice else "no_mouse"
    _alpha_output_dir = Path(r"E:\data\LeciLab\behavioral_data\tmp\processing")
    if not _alpha_output_dir.is_absolute() and str(_alpha_output_dir).startswith("E:\\"):
        _alpha_output_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp/processing")
    _alpha_act_path = (
        _alpha_output_dir
        / f"{_saved_label}_manual_effec_dcz_vs_saline_noeffec_alpha_df_act.pkl"
    )
    _alpha_stim_path = (
        _alpha_output_dir
        / f"{_saved_label}_manual_effec_dcz_vs_saline_noeffec_alpha_df_stim.pkl"
    )

    def _make_alpha_plot(alpha_df, alpha_col, title, ylabel):
        if alpha_df.empty:
            return mo.md(f"No {title.lower()} values to compare.")

        _plot_df = alpha_df.dropna(subset=[alpha_col]).copy()
        if _plot_df.empty:
            return mo.md(f"No valid {title.lower()} values to compare.")

        _available_conditions = set(_plot_df["condition"].astype(str).unique())
        _conditions = [
            _condition
            for _condition in _condition_order
            if _condition in _available_conditions
        ]
        _conditions.extend(sorted(_available_conditions - set(_conditions)))
        _condition_positions = {
            condition: i for i, condition in enumerate(_conditions)
        }
        _fig, _ax = plt.subplots(figsize=(6, 5))

        for _mouse, _df_mouse in _plot_df.groupby("mouse", sort=True):
            _df_mouse = _df_mouse.copy()
            _df_mouse["condition"] = _df_mouse["condition"].astype(str)
            _df_mouse = _df_mouse.sort_values(
                "condition",
                key=lambda _s: _s.map(_condition_positions),
            )
            _xs = [
                _condition_positions[_condition]
                for _condition in _df_mouse["condition"]
            ]
            _ys = _df_mouse[alpha_col].astype(float).values
            _ax.plot(
                _xs,
                _ys,
                marker="o",
                linewidth=1.2,
                alpha=0.75,
                label=_mouse,
            )

        _ax.set_xticks(range(len(_conditions)))
        _ax.set_xticklabels(_conditions, rotation=35, ha="right")
        _ax.set_ylim(0, 1.1)
        _ax.set_xlabel("injection condition")
        _ax.set_ylabel(ylabel)
        _ax.set_title(title)
        if _plot_df["mouse"].nunique() <= 12:
            _ax.legend(title="mouse", bbox_to_anchor=(1.02, 1), loc="upper left")
        _fig.tight_layout()
        return _fig

    if (
        _alpha_act_path.exists()
        and _alpha_stim_path.exists()
        and not update_processData_button.value
    ):
        effec_group_alpha_df_act = pd.read_pickle(_alpha_act_path)
        effec_group_alpha_df_stim = pd.read_pickle(_alpha_stim_path)
        print(
            "Loaded saved effective DCZ vs saline/no-effect alpha data: "
            f"{_alpha_act_path}, {_alpha_stim_path}"
        )
    else:
        _mouse_col = None
        if "analysis_mouse" in df_effec_glmhmm.columns:
            _mouse_col = "analysis_mouse"
        elif "subject" in df_effec_glmhmm.columns:
            _mouse_col = "subject"

        _required_cols = {
            "first_choice_numeric",
            "correct_side_numeric",
            "selected_df_option",
        }

        def _fit_alpha_by_mouse_condition(prior_type, alpha_col):
            _rows = []
            if (
                df_effec_glmhmm.empty
                or _mouse_col is None
                or not _required_cols.issubset(df_effec_glmhmm.columns)
            ):
                return pd.DataFrame(_rows)

            for (_mouse, _condition), _df_group in df_effec_glmhmm.groupby(
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
                            "condition": _condition,
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
                            "condition": _condition,
                            "n_trials": len(_df_clean),
                            alpha_col: _theta["alpha"],
                            "status": "ok" if _res.success else _res.message,
                        }
                    )
                except Exception as _exc:
                    _rows.append(
                        {
                            "mouse": _mouse,
                            "condition": _condition,
                            "n_trials": len(_df_clean),
                            alpha_col: pd.NA,
                            "status": f"failed: {_exc}",
                        }
                    )

            return pd.DataFrame(_rows)

        effec_group_alpha_df_act = _fit_alpha_by_mouse_condition(
            prior_type="act",
            alpha_col="alpha_act",
        )
        effec_group_alpha_df_stim = _fit_alpha_by_mouse_condition(
            prior_type="stim",
            alpha_col="alpha_stim",
        )
        _alpha_output_dir.mkdir(parents=True, exist_ok=True)
        effec_group_alpha_df_act.to_pickle(_alpha_act_path)
        effec_group_alpha_df_stim.to_pickle(_alpha_stim_path)
        print(
            "Saved effective DCZ vs saline/no-effect alpha data: "
            f"{_alpha_act_path}, {_alpha_stim_path}"
        )

    effec_group_alpha_plot_act = _make_alpha_plot(
        effec_group_alpha_df_act,
        "alpha_act",
        "Effective DCZ vs saline/no-effect action kernel alpha",
        "action kernel alpha",
    )
    effec_group_alpha_plot_stim = _make_alpha_plot(
        effec_group_alpha_df_stim,
        "alpha_stim",
        "Effective DCZ vs saline/no-effect stimulus kernel alpha",
        "stimulus kernel alpha",
    )

    mo.vstack([effec_group_alpha_plot_act, effec_group_alpha_plot_stim])
    return effec_group_alpha_df_act, effec_group_alpha_df_stim


@app.cell
def _(
    df_effec_glmhmm,
    effec_group_alpha_df_act,
    effec_group_alpha_df_stim,
    pd,
    utils_test,
):
    df_effec_glmhmm_kernel = pd.DataFrame([])
    _mouse_col = None
    if "analysis_mouse" in df_effec_glmhmm.columns:
        _mouse_col = "analysis_mouse"
    elif "subject" in df_effec_glmhmm.columns:
        _mouse_col = "subject"

    _required_cols = {
        "session",
        "selected_df_option",
        "first_choice_numeric",
        "correct_side_numeric",
    }

    def _alpha_lookup(alpha_df, alpha_col):
        _required_alpha_cols = {"mouse", "condition", alpha_col}
        if alpha_df.empty or not _required_alpha_cols.issubset(alpha_df.columns):
            return {}
        _alpha_df = alpha_df.dropna(subset=[alpha_col]).copy()
        _alpha_df["mouse"] = _alpha_df["mouse"].astype(str)
        _alpha_df["condition"] = _alpha_df["condition"].astype(str)
        return (
            _alpha_df.set_index(["mouse", "condition"])[alpha_col]
            .astype(float)
            .to_dict()
        )

    if (
        not df_effec_glmhmm.empty
        and _mouse_col is not None
        and _required_cols.issubset(df_effec_glmhmm.columns)
    ):
        _alpha_act_lookup = _alpha_lookup(
            effec_group_alpha_df_act,
            "alpha_act",
        )
        _alpha_stim_lookup = _alpha_lookup(
            effec_group_alpha_df_stim,
            "alpha_stim",
        )

        _kernel_dfs = []
        _missing_alpha_keys = set()
        for (_mouse, _condition, _session), df_test_session in df_effec_glmhmm.groupby(
            [_mouse_col, "selected_df_option", "session"],
            sort=False,
        ):
            _lookup_key = (str(_mouse), str(_condition))
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
            df_effec_glmhmm_kernel = pd.concat(_kernel_dfs, ignore_index=True)
        if _missing_alpha_keys:
            print(f"Skipped sessions with missing alpha: {sorted(_missing_alpha_keys)}")
    return (df_effec_glmhmm_kernel,)


@app.cell
def _():
    input_cols_noStim = ["bias", "action_trace", "stimulus_trace"]
    return (input_cols_noStim,)


@app.cell
def _(mo):
    effec_glmhmm_num_states = mo.ui.number(
        start=1,
        stop=6,
        step=1,
        value=3,
        label="Effective-group GLM-HMM states",
    )
    effec_glmhmm_n_iters = mo.ui.number(
        start=5,
        stop=150,
        step=5,
        value=50,
        label="Effective-group GLM-HMM EM iterations",
    )
    mo.vstack([effec_glmhmm_num_states, effec_glmhmm_n_iters])
    return effec_glmhmm_n_iters, effec_glmhmm_num_states


@app.cell
def _(
    df_effec_glmhmm_kernel,
    effec_glmhmm_n_iters,
    effec_glmhmm_num_states,
    effec_glmhmm_stim_col,
    input_cols_noStim,
    pd,
    utils_test,
):
    effec_group_glmhmm_model_dic = {}
    _hmm_model_rows = []

    _input_cols = [effec_glmhmm_stim_col] + input_cols_noStim
    _real_input_cols = [_col for _col in _input_cols if _col != "bias"]
    _required_cols = {
        "session",
        "selected_df_option",
        "first_choice_numeric",
        *_real_input_cols,
    }

    if df_effec_glmhmm_kernel.empty:
        _hmm_model_rows.append({"status": "df_effec_glmhmm_kernel is empty"})
    elif not _required_cols.issubset(df_effec_glmhmm_kernel.columns):
        _missing_cols = sorted(
            _required_cols - set(df_effec_glmhmm_kernel.columns)
        )
        _hmm_model_rows.append({"status": f"Missing columns: {_missing_cols}"})
    else:
        for _condition, _df_condition in df_effec_glmhmm_kernel.groupby(
            "selected_df_option",
            sort=True,
        ):
            _model_key = _condition
            _df_model = _df_condition.dropna(
                subset=["first_choice_numeric"] + _real_input_cols
            ).copy()

            if _df_model.empty:
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "condition": _condition,
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
                        "condition": _condition,
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
                    num_states=int(effec_glmhmm_num_states.value),
                    obs_dim=1,
                    input_dim=_input_dim,
                    num_categories=2,
                    N_iters=int(effec_glmhmm_n_iters.value),
                    prior_sigma=2,
                    kappa=15,
                )
                effec_group_glmhmm_model_dic[_model_key] = {
                    "model": _map_glmhmm,
                    "log_likelihood": _ll,
                    "hmm_lls": _hmm_lls,
                    "datas": _datas,
                    "inpts": _inpts,
                    "df": _df_model,
                    "condition": _condition,
                    "inputs": _input_cols,
                }
                _hmm_model_rows.append(
                    {
                        "model_key": _model_key,
                        "condition": _condition,
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
                        "condition": _condition,
                        "n_trials": len(_df_model),
                        "n_sessions": len(_datas),
                        "input_dim": _input_dim,
                        "log_likelihood": pd.NA,
                        "status": f"failed: {_exc}",
                    }
                )

    effec_group_glmhmm_model_summary = pd.DataFrame(_hmm_model_rows)
    effec_group_glmhmm_model_summary
    return (effec_group_glmhmm_model_dic,)


@app.cell
def _(
    effec_glmhmm_stim_col,
    effec_group_glmhmm_model_dic,
    input_cols_noStim,
    pd,
):
    weight_cols = (
        [effec_glmhmm_stim_col]
        + [col for col in input_cols_noStim if col != "bias"]
        + ["bias"]
    )

    eps = 1e-9

    for hmm_model_name, model_info in effec_group_glmhmm_model_dic.items():
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

        stim_abs = weight_df_before[effec_glmhmm_stim_col].abs()
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
    effec_group_glmhmm_state_reorder_done = True
    return (effec_group_glmhmm_state_reorder_done,)


@app.cell
def _(
    Path,
    effec_glmhmm_stim_col,
    effec_group_glmhmm_model_dic,
    effec_group_glmhmm_state_reorder_done,
    input_cols_noStim,
    mo,
    utils_test,
):
    _ = effec_group_glmhmm_state_reorder_done
    effec_group_glmhmm_session_figs = []
    _save_dir = Path(
        "/mnt/e/data/LeciLab/behavioral_data/tmp/glm_hmm_model_effec_dcz_vs_saline_noeffec_sessions"
    )
    _save_dir.mkdir(parents=True, exist_ok=True)

    for model_name_ in effec_group_glmhmm_model_dic:
        _model_info = effec_group_glmhmm_model_dic[model_name_]
        _condition = _model_info["condition"]
        _df_model = _model_info["df"]

        for _session, _df_session in _df_model.groupby("session", sort=True):
            _df_session_clean = _df_session.dropna(
                subset=[
                    effec_glmhmm_stim_col,
                    *[col for col in input_cols_noStim if col != "bias"],
                    "first_choice_numeric",
                ]
            ).copy()
            if _df_session_clean.empty:
                continue

            _plot_datas, _plot_inpts = utils_test.build_glmhmm_inputs_by_session(
                _df_session_clean,
                y_col="first_choice_numeric",
                stim_col=[effec_glmhmm_stim_col] + input_cols_noStim,
            )
            _fig, _df_with_state, _post_prob_list = (
                utils_test.plot_glmhmm_pipeline_figure(
                    _model_info["model"],
                    _df_session_clean,
                    _plot_datas,
                    _plot_inpts,
                    input_cols=[
                        effec_glmhmm_stim_col
                    ] + [col for col in input_cols_noStim if col != "bias"],
                    y_col="first_choice_numeric",
                    psychometric_x=effec_glmhmm_stim_col,
                    title=(
                        f"GLM-HMM summary {model_name_} "
                        f"({_session})"
                    ),
                    psychometric_value_type="continuous",
                )
            )
            print(
                model_name_,
                _session,
                _df_with_state["glmhmm_state"].value_counts().sort_index(),
            )
            effec_group_glmhmm_session_figs.append(
                mo.vstack([mo.md(f"## {_condition}: `{_session}`"), _fig])
            )
            _safe_session = (
                str(_session)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
                .replace(" ", "_")
            )
            _fig.savefig(
                _save_dir / f"{model_name_}_{_safe_session}.svg",
                format="svg",
                bbox_inches="tight",
            )

    if effec_group_glmhmm_session_figs:
        effec_group_glmhmm_session_output = mo.vstack(
            effec_group_glmhmm_session_figs
        )
    else:
        effec_group_glmhmm_session_output = mo.md(
            "No effective DCZ vs saline/no-effect GLM-HMM session figures to show."
        )
    effec_group_glmhmm_session_output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
