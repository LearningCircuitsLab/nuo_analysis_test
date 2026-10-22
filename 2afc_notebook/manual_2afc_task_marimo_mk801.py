import marimo

__generated_with = "0.25.1"
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
    import scipy.stats as stats

    from lecilab_behavior_analysis import utils as utils
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
        np,
        pd,
        plot_test,
        plots,
        plt,
        sns,
        stats,
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
    _default_project = 'COT_cannula_GAD2_data'
    project_select = mo.ui.dropdown(
        options=tv_projects,
        value=_default_project,
        label="Project",
    )

    project_select
    return (project_select,)


@app.cell
def _(project_select):
    project = project_select.value
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # date select
    """)
    return


@app.cell
def _(df_all, dft, np):
    df_day = dft.add_day_column_to_df(df_all.copy())
    df_day = dft.add_trial_of_day_column_to_df(df_day)
    df_day["total_trial"] = np.arange(1, df_day.shape[0] + 1)

    available_dates = sorted(
        df_day["year_month_day"].dropna().astype(str).unique().tolist()
    )
    return (available_dates,)


@app.cell
def _(available_dates, mo):
    # Manual selection of date range for analysis
    # mix1 (0.01mg/ml from Alexis) using time: 2026-07-18 to 2026-07-31
    # mix2 (0.01mg/ml from Nuo) using time: 2026-08-18 to 2026-09-02
    # mix3 (0.02mg/ml from Nuo) using time: 2026-09-03 to now
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
def _(
    Path,
    date_select_end,
    date_select_start,
    dft,
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
        df_manual = df_all_mouse[df_all_mouse["run_mode"] == _run_mode]
        computed_results = {}
        for output_name, stim_name in stim_columns.items():
            df_stim = df_manual[df_manual["stimulus_modality"] == stim_name]
            if df_stim.empty:
                computed_results[output_name] = df_stim.copy()
                print(f"No {stim_name} trials found for {_mouse}; skipped parameters_for_fit.")
            else:
                computed_results[output_name] = dft.parameters_for_fit(df_stim)

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

    _df_test_vis_raw = (
        pd.concat(combined_results["df_test_vis"], ignore_index=True)
        if combined_results["df_test_vis"]
        else pd.DataFrame()
    )

    _df_test_aud_raw = (
        pd.concat(combined_results["df_test_aud"], ignore_index=True)
        if combined_results["df_test_aud"]
        else pd.DataFrame()
    )

    df_test_selected_raw = pd.concat(
        [_df_test_vis_raw, _df_test_aud_raw],
        ignore_index=True,
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
    injection_info_file_path = "/home/nuo/Documents/data/LeciLab/behavioral_data/manual_record/manual_test_record_nuo_mk801.xlsx"
    injection_info_df = pd.read_excel(injection_info_file_path)
    injection_info_df
    return (injection_info_df,)


@app.cell
def _(behavior_utils, injection_info_df):
    observation_dates = behavior_utils.get_injection_observation_dates(
        injection_info_df,
    )

    print(observation_dates)

    print(observation_dates.shape)
    print(observation_dates.columns.tolist())
    print(
        observation_dates["observation_group"]
        .value_counts(dropna=False)
    )
    print(observation_dates["date"].dtype)
    print(observation_dates["date"].isna().sum())
    return


@app.cell
def _(df_test_selected_raw, injection_info_df, pd):
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
        df_test_selected["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    trial_mice = df_test_selected["subject"].astype(str)
    df_test_selected["observations"] = [
        injection_lookup.get((trial_date, trial_mouse), pd.NA)
        for trial_date, trial_mouse in zip(trial_dates, trial_mice)
    ]
    return (df_test_selected,)


@app.cell
def _(behavior_utils, df_test_selected, dft, injection_info_df, pd):
    session_performance = df_test_selected.groupby(
        ["subject", "session"]
    )["correct"].transform("mean")

    is_saline = df_test_selected["observations"].str.contains(
        "saline",
        case=False,
        na=False,
    )

    drop_mask = is_saline & (session_performance < 0.65)

    df_test_selected_upd = df_test_selected[
        ~drop_mask
    ].copy()

    for _subject in df_test_selected_upd['subject'].unique():
        _df_sub = df_test_selected_upd.loc[df_test_selected_upd['subject'] == _subject]
        _df_sub = dft.add_engagement_column(_df_sub, engagement_sd_criteria=2)
        df_test_selected_upd.loc[df_test_selected_upd['subject'] == _subject, 'engaged'] = _df_sub['engaged']
    metric_paired_dates = behavior_utils.get_paired_injection_dates(injection_info_df, drug_name="mk801")
    df_test_selected_upd["observation_group"] = df_test_selected_upd[
        "observations"
    ].apply(behavior_utils.observation_group)
    df_test_selected_upd["pair_date"] = pd.to_datetime(
        df_test_selected_upd["year_month_day"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    available_dates_ = df_test_selected_upd[
        ["subject", "pair_date", "observation_group"]
    ].dropna().drop_duplicates()

    paired_dates_for_filter = metric_paired_dates.copy()
    paired_dates_for_filter["subject"] = paired_dates_for_filter["subject"].astype(str)
    paired_dates_for_filter["_saline_date"] = pd.to_datetime(
        paired_dates_for_filter["saline_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    paired_dates_for_filter["_mk801_date"] = pd.to_datetime(
        paired_dates_for_filter["mk801_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    saline_available = available_dates_.rename(
        columns={
            "pair_date": "_saline_date",
            "observation_group": "_saline_observation_group",
        }
    )
    mk801_available = available_dates_.rename(
        columns={
            "pair_date": "_mk801_date",
            "observation_group": "_mk801_observation_group",
        }
    )

    paired_dates_for_filter = (
        paired_dates_for_filter.merge(
            saline_available,
            on=["subject", "_saline_date"],
            how="inner",
        )
        .merge(
            mk801_available,
            on=["subject", "_mk801_date"],
            how="inner",
        )
    )
    paired_dates_for_filter = paired_dates_for_filter[
        (paired_dates_for_filter["_saline_observation_group"] == "saline")
        & (paired_dates_for_filter["_mk801_observation_group"] == "mk801")
    ].copy()

    metric_paired_dates = metric_paired_dates[
        metric_paired_dates["pair_id"].isin(paired_dates_for_filter["pair_id"])
    ].copy()

    paired_date_rows = pd.concat(
        [
            metric_paired_dates[["subject", "saline_date"]]
            .rename(columns={"saline_date": "pair_date"})
            .assign(observation_group="saline"),
            metric_paired_dates[["subject", "mk801_date"]]
            .rename(columns={"mk801_date": "pair_date"})
            .assign(observation_group="mk801"),
        ],
        ignore_index=True,
    )
    paired_date_rows["subject"] = paired_date_rows["subject"].astype(str)
    paired_date_rows["pair_date"] = pd.to_datetime(
        paired_date_rows["pair_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    paired_date_rows = paired_date_rows.dropna().drop_duplicates()

    df_test_selected_upd["subject"] = df_test_selected_upd["subject"].astype(str)
    df_test_selected_upd = df_test_selected_upd.merge(
        paired_date_rows,
        on=["subject", "pair_date", "observation_group"],
        how="inner",
    )


    df_test_selected_upd["repeat_rate_session"] = (
        df_test_selected_upd
        .groupby(["subject", "session"])["previous_same_choice_numeric"]
        .transform("mean")
    )
    df_test_selected_upd["repeating_bias_session"] = (df_test_selected_upd["repeat_rate_session"] - 0.5)

    df_test_selected_upd["session_duration"] = (
        df_test_selected_upd["TRIAL_END"] - df_test_selected_upd.groupby(
            ["subject", "session"]
        )["TRIAL_START"].transform("min")
    )

    df_test_selected_upd = df_test_selected_upd[df_test_selected_upd["session_duration"] <= 2400].copy()
    return df_test_selected_upd, metric_paired_dates


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # mixed correlation matrix
    """)
    return


@app.cell
def _():
    continuous_cols = [
        "reaction_time",
        "trial_duration",
        "inter_trial_interval",
    ]

    binary_cols = [
        "correct_numeric",
        "previous_same_choice_numeric",
        "roa_choice_numeric",
    ]

    count_cols = [
        "fixation_breaks",
        "eager_pokes",
    ]

    features = continuous_cols + binary_cols + count_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare by animals
    """)
    return


@app.cell
def get_metric_per_mouse_sessions(behavior_utils, pd):
    def get_metric_per_mouse_sessions(
        df,
        metric_paired_dates,
        metric_col="correct",
        agg_func="mean",
        difference=False,
    ):

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
            behavior_utils.observation_group
        )
        df_metric = df_metric.dropna(subset=["observation_group"])

        paired_columns = ["subject", "pair_id"]
        paired_dates = pd.concat(
            [
                metric_paired_dates[paired_columns + ["saline_date"]]
                .rename(columns={"saline_date": "year_month_day"})
                .assign(observation_group="saline"),
                metric_paired_dates[paired_columns + ["mk801_date"]]
                .rename(columns={"mk801_date": "year_month_day"})
                .assign(observation_group="mk801"),
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
                mk801_values = df_pair.loc[
                    df_pair["observation_group"] == "mk801",
                    "metric_value",
                ]
                if saline_values.empty or mk801_values.empty:
                    continue

                difference_output[modality_key].setdefault(subject, []).append(
                    saline_values.mean() - mk801_values.mean()
                )

            return difference_output

        output = {
            "vis_saline": {},
            "vis_mk801": {},
            "aud_saline": {},
            "aud_mk801": {},
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

    return (get_metric_per_mouse_sessions,)


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
        "repeating bias": {
            "metric_col": "previous_same_choice_numeric",
            "agg_func": "mean",
            "ylabel": "Session repeating bias",
        },
        "fixation breaks": {
            "metric_col": "fixation_breaks",
            "agg_func": "mean",
            "ylabel": "Session fixation breaks",
        }
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
    df_test_selected_upd,
    get_metric_per_mouse_sessions,
    metric_paired_dates,
):
    compare_value_setting = compare_value_settings[compare_value_select.value]
    compare_metric_col = compare_value_setting["metric_col"]
    compare_agg_func = compare_value_setting["agg_func"]
    compare_y_label = compare_value_setting["ylabel"]

    output_per_mouse_dic = get_metric_per_mouse_sessions(
        df_test_selected_upd,
        metric_paired_dates,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
    )
    output_per_mouse_dic_diff = get_metric_per_mouse_sessions(
        df_test_selected_upd,
        metric_paired_dates,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
        difference=True
    )
    return (
        compare_agg_func,
        compare_metric_col,
        compare_y_label,
        output_per_mouse_dic,
    )


@app.cell
def _(compare_y_label, mo, output_per_mouse_dic, plot_test):
    condition_session_fig = plot_test.plot_condition_session_values_by_mouse(
        output_per_mouse_dic,
        drug_name="mk801",
        group_name = "MK801 vs Saline",
        ylabel=compare_y_label,
        title=f"{compare_y_label}",
    )
    mo.vstack(
        [
            condition_session_fig
        ]
    )
    return


@app.cell
def _(add_significance_bar, np, p_value_to_label, plt, stats):
    def plot_group_by_subjects(
        df,
        metric_col="correct",
        agg_func="mean",
        drug_name="mk801",
        ylabel=None,
        title=None,
        test_method="wilcoxon",
    ):
        groups = ["saline", drug_name]
        colors = {
            "saline": "blue",
            drug_name: "red",
        }

        plot_df = (
            df.dropna(
                subset=[
                    "subject",
                    "observation_group",
                    metric_col,
                ]
            )
            .loc[
                lambda x: x["observation_group"].isin(groups)
            ]
            .copy()
        )

        # 每只鼠、每个条件直接汇总所有 trial
        if agg_func == "len":
            mouse_summary = (
                plot_df.groupby(
                    ["subject", "observation_group"],
                    as_index=False,
                )
                .agg(
                    new_col=(metric_col, "size"),
                )
            )
        else:
            mouse_summary = (
                plot_df.groupby(
                    ["subject", "observation_group"],
                    as_index=False,
                )
                .agg(
                    new_col=(metric_col, agg_func),
                    n_trials=(metric_col, "size"),
                )
            )

        if metric_col=="previous_same_choice_numeric":
            mouse_summary["new_col"] -= 0.5

        # 每只鼠一行，只保留同时有 saline 和 drug 的鼠
        paired_summary = (
            mouse_summary
            .pivot(
                index="subject",
                columns="observation_group",
                values="new_col",
            )
            .dropna(subset=groups)
            .reset_index()
        )

        data = [
            paired_summary[group].dropna()
            for group in groups
        ]

        fig, ax = plt.subplots(figsize=(6, 5))

        boxplot = ax.boxplot(
            data,
            tick_labels=groups,
            patch_artist=True,
            showmeans=True,
        )

        for patch, group in zip(boxplot["boxes"], groups):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.2)
            patch.set_edgecolor(colors[group])

        # 一只鼠一条配对线
        for _, row in paired_summary.iterrows():
            ax.plot(
                [1, 2],
                [row["saline"], row[drug_name]],
                color="gray",
                alpha=0.35,
                linewidth=1,
                zorder=2,
            )

        for position, group in enumerate(groups, start=1):
            ax.scatter(
                [position] * len(paired_summary),
                paired_summary[group],
                color=colors[group],
                edgecolor="black",
                linewidth=0.5,
                s=45,
                alpha=0.8,
                zorder=3,
            )
            # 配对统计检验：每一行代表同一只鼠
        saline_values = paired_summary["saline"].to_numpy()
        drug_values = paired_summary[drug_name].to_numpy()

        p_value = np.nan

        if len(paired_summary) >= 2:
            if test_method == "wilcoxon":
                try:
                    p_value = stats.wilcoxon(
                        saline_values,
                        drug_values,
                    ).pvalue
                except ValueError:
                    # 例如所有配对差值均为 0
                    p_value = 1.0

            elif test_method == "ttest_rel":
                p_value = stats.ttest_rel(
                    saline_values,
                    drug_values,
                    nan_policy="omit",
                ).pvalue

            else:
                raise ValueError(
                    "test_method must be 'wilcoxon' or 'ttest_rel'"
                )

            # 保存统计结果
            paired_summary.attrs["test_method"] = test_method
            paired_summary.attrs["p_value"] = p_value

            # significance bar 的位置
            y_values = np.concatenate([
                saline_values,
                drug_values,
            ])

            y_min = np.nanmin(y_values)
            y_max = np.nanmax(y_values)
            y_range = y_max - y_min

            if y_range == 0:
                y_range = abs(y_max) * 0.1 if y_max != 0 else 0.1

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

            ax.set_ylim(
                top=bar_y + bar_h + y_range * 0.15
            )

        ax.set_xlabel("Observation")
        ax.set_ylabel(ylabel or metric_col)
        ax.set_title(title or f"{metric_col} by observation")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return fig, paired_summary


    return (plot_group_by_subjects,)


@app.cell
def _(
    compare_agg_func,
    compare_metric_col,
    compare_value_select,
    compare_y_label,
    df_test_selected_upd,
    plot_group_by_subjects,
):
    plot_group_by_subjects(df_test_selected_upd, metric_col=compare_metric_col, agg_func=compare_agg_func, ylabel=compare_y_label, title=compare_value_select.value)
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

    for _subject, _df_mouse in df_test_selected_upd.groupby("subject", sort=True):
        _df_mouse = utils.add_time_from_session_start(_df_mouse.copy())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, modality in zip(axes, ["visual", "auditory"]):
            df_modality = _df_mouse[
                _df_mouse["stimulus_modality"] == modality
            ]

            for observation, color in [
                ("saline", "steelblue"),
                ("mk801", "firebrick"),
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
            ax.scatter([], [], color="firebrick", s=16, label="mk801")
            ax.set_title(modality.capitalize())
            ax.legend(frameon=False)

        fig.suptitle(str(_subject))
        fig.tight_layout()
        trial_time_figures.append(fig)

    mo.vstack(trial_time_figures) if trial_time_figures else mo.md(
        "No paired saline/mk801 sessions to plot."
    )
    return


@app.cell
def _(df_test_selected_upd):
    df_test_selected_upd.head()
    return


@app.cell
def _(df_test_selected_upd, np, pd):
    # accumulated trial numbers
    time_points = np.arange(0, 2401, 30)  # 每5分钟一个点
    rows = []

    df_trials = df_test_selected_upd.copy()

    # 每个 trial 相对于 session 开始的时间
    df_trials["session_time"] = (
        df_trials["TRIAL_END"]
        - df_trials.groupby(
            ["subject", "session"]
        )["TRIAL_START"].transform("min")
    )

    df_trials = df_trials[
        df_trials["session_time"].between(0, 2400)
    ].copy()

    # 每个 session 在每个时间点已经完成多少个 trial
    for (
        subject,
        session,
        treatment,
    ), df_session in df_trials.groupby(
        ["subject", "session", "observation_group"]
    ):
        trial_times = np.sort(
            df_session["session_time"].dropna().to_numpy()
        )

        cumulative_trials = np.searchsorted(
            trial_times,
            time_points,
            side="right",
        )

        rows.extend(
            {
                "subject": subject,
                "session": session,
                "treatment": treatment,
                "session_time": time_point,
                "cumulative_trials": n_trials,
            }
            for time_point, n_trials in zip(
                time_points,
                cumulative_trials,
            )
        )

    session_trial_counts = pd.DataFrame(rows)
    return (session_trial_counts,)


@app.cell
def _(session_trial_counts):
    session_trial_counts
    return


@app.cell
def _(plt, session_trial_counts):
    all_session_trial_counts = (
        session_trial_counts
        .groupby(
            ["treatment", "session_time"],
            as_index=False,
        )["cumulative_trials"]
        .sum()
    )

    all_session_trial_counts_drug = all_session_trial_counts[
        all_session_trial_counts["treatment"] == "mk801"
    ]

    all_session_trial_counts_saline = all_session_trial_counts[
        all_session_trial_counts["treatment"] == "saline"
    ]

    plt.scatter(
        all_session_trial_counts_saline["session_time"],
        all_session_trial_counts_saline["cumulative_trials"],
        color="blue",
        label="saline",
        s=3
    )

    plt.scatter(
        all_session_trial_counts_drug["session_time"],
        all_session_trial_counts_drug["cumulative_trials"],
        color="red",
        label="mk801",
        s=3
    )

    plt.xlabel("Session time")
    plt.ylabel("Cumulative trials")
    plt.legend(frameon=False)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare by sessions
    """)
    return


@app.cell
def _(np, pd, plt):
    def plot_metric_by_evidence_bins(
        df,
        metric_col,
        agg_func="mean",
        ylabel=None,
        title=None,
        drug_name="mk801",
        n_bins=3,
        test_method="ttest_ind",
    ):
        from scipy import stats

        groups = ["saline", drug_name]

        colors = {
            "saline": "blue",
            drug_name: "red",
        }

        plot_df = df.copy()

        # 将 evidence 转换为数值
        plot_df["abs_total_evidence_strength"] = pd.to_numeric(
            plot_df["abs_total_evidence_strength"],
            errors="coerce",
        )

        # 只保留 saline、drug 和有效 evidence
        plot_df = plot_df[
            plot_df["observation_group"].isin(groups)
        ].dropna(
            subset=[
                "observation_group",
                "abs_total_evidence_strength",
            ]
        ).copy()

        is_count = (
            agg_func in {"len", "size", "count"}
            or agg_func is len
        )

        # 非计数指标必须转换为数值
        if not is_count:
            plot_df[metric_col] = pd.to_numeric(
                plot_df[metric_col],
                errors="coerce",
            )

            plot_df = plot_df.dropna(
                subset=[metric_col]
            ).copy()

        # 没有有效数据
        if plot_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))

            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.set_axis_off()

            return (
                fig,
                pd.DataFrame(),
                pd.DataFrame(),
            )

        # 按 evidence 数值范围等宽分箱
        plot_df["evidence_bin"] = pd.cut(
            plot_df["abs_total_evidence_strength"],
            bins=n_bins,
            include_lowest=True,
        )

        # 保留有数据的 bin
        bin_order = [
            interval
            for interval in plot_df[
                "evidence_bin"
            ].cat.categories
            if (
                plot_df["evidence_bin"] == interval
            ).any()
        ]

        # 每个 bin 的横坐标使用区间中心
        bin_centers = {
            interval: interval.mid
            for interval in bin_order
        }

        # 每个 bin 的数据范围和 trial 数
        bin_info = (
            plot_df.groupby(
                "evidence_bin",
                observed=True,
            )["abs_total_evidence_strength"]
            .agg(
                evidence_mean="mean",
                evidence_min="min",
                evidence_max="max",
                n_trials="size",
            )
            .reset_index()
        )

        if is_count:
            # 每个 treatment、每个 evidence bin 的总 trial 数
            summary = (
                plot_df.groupby(
                    [
                        "observation_group",
                        "evidence_bin",
                    ],
                    observed=True,
                )
                .size()
                .rename("value")
                .reset_index()
            )

            summary["n_trials"] = summary["value"]

            # 总计数不能直接计算普通 SEM/95% CI；
            # sqrt(N) 作为 Poisson count error
            summary["error"] = np.sqrt(
                summary["value"]
            )

        else:
            # pooled trials 的均值、标准差和样本数
            summary = (
                plot_df.groupby(
                    [
                        "observation_group",
                        "evidence_bin",
                    ],
                    observed=True,
                )[metric_col]
                .agg(
                    value="mean",
                    std="std",
                    n_trials="size",
                )
                .reset_index()
            )

            # 标准误
            summary["sem"] = (
                summary["std"]
                / np.sqrt(summary["n_trials"])
            )

            # 95% confidence interval 的半宽
            summary["error"] = (
                stats.t.ppf(
                    0.975,
                    df=summary["n_trials"] - 1,
                )
                * summary["sem"]
            )

            summary["ci95_lower"] = (
                summary["value"]
                - summary["error"]
            )

            summary["ci95_upper"] = (
                summary["value"]
                + summary["error"]
            )

        # 为每个 bin 添加横坐标中心
        summary["bin_center"] = (
            summary["evidence_bin"]
            .map(bin_centers)
            .astype(float)
        )

        # 每个 evidence bin 内进行 saline–drug 比较
        comparison_rows = []

        if not is_count:
            for evidence_bin, df_bin in plot_df.groupby(
                "evidence_bin",
                observed=True,
            ):
                saline_values = df_bin.loc[
                    df_bin["observation_group"] == "saline",
                    metric_col,
                ].dropna()

                drug_values = df_bin.loc[
                    df_bin["observation_group"] == drug_name,
                    metric_col,
                ].dropna()

                if saline_values.empty or drug_values.empty:
                    p_value = np.nan

                elif test_method == "mannwhitneyu":
                    p_value = stats.mannwhitneyu(
                        saline_values,
                        drug_values,
                        alternative="two-sided",
                    ).pvalue

                elif test_method == "ttest_ind":
                    p_value = stats.ttest_ind(
                        saline_values,
                        drug_values,
                        equal_var=False,
                        nan_policy="omit",
                    ).pvalue

                else:
                    raise ValueError(
                        "test_method must be "
                        "'mannwhitneyu' or 'ttest_ind'"
                    )

                comparison_rows.append({
                    "evidence_bin": evidence_bin,
                    "bin_center": bin_centers[
                        evidence_bin
                    ],
                    "n_saline": len(saline_values),
                    f"n_{drug_name}": len(
                        drug_values
                    ),
                    "p_value": p_value,
                })

        comparisons = pd.DataFrame(
            comparison_rows
        )

        def _p_label(p_value):
            if pd.isna(p_value):
                return "n/a"
            if p_value < 0.001:
                return "***"
            if p_value < 0.01:
                return "**"
            if p_value < 0.05:
                return "*"
            return "ns"

        # 绘图
        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        for group in groups:
            group_summary = (
                summary[
                    summary["observation_group"]
                    == group
                ]
                .sort_values("bin_center")
            )

            if group_summary.empty:
                continue

            ax.errorbar(
                group_summary["bin_center"],
                group_summary["value"],
                yerr=group_summary[
                    "error"
                ].fillna(0),
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=4,
                color=colors[group],
                label=group,
            )

        # 添加显著性标记
        if not comparisons.empty:
            y_upper = (
                summary["value"]
                + summary["error"].fillna(0)
            )

            y_lower = (
                summary["value"]
                - summary["error"].fillna(0)
            )

            y_max = y_upper.max()
            y_min = y_lower.min()
            y_range = y_max - y_min

            if (
                not np.isfinite(y_range)
                or y_range == 0
            ):
                y_range = (
                    abs(y_max) * 0.1
                    if np.isfinite(y_max)
                    and y_max != 0
                    else 1
                )

            significance_y = (
                y_max + y_range * 0.08
            )

            for _, row in comparisons.iterrows():
                ax.text(
                    row["bin_center"],
                    significance_y,
                    _p_label(
                        row["p_value"]
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            ax.set_ylim(
                top=y_max + y_range * 0.20
            )

        # 横坐标位置
        ax.set_xticks(
            [
                bin_centers[interval]
                for interval in bin_order
            ]
        )

        # 显示每个 bin 的 evidence 范围
        ax.set_xticklabels(
            [
                (
                    f"{interval.left:.2f}–"
                    f"{interval.right:.2f}"
                )
                for interval in bin_order
            ],
            rotation=30,
            ha="right",
        )

        ax.set_xlabel(
            "Absolute total evidence strength"
        )

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        elif is_count:
            ax.set_ylabel(
                f"Total number of {metric_col}"
            )

        else:
            ax.set_ylabel(
                f"Mean {metric_col} (95% CI)"
            )

        if title is not None:
            ax.set_title(title)

        else:
            ax.set_title(
                f"{metric_col} by evidence "
                "strength and treatment"
            )

        ax.legend(frameon=False)
        ax.grid(
            axis="y",
            alpha=0.25,
        )
        ax.spines[
            ["top", "right"]
        ].set_visible(False)

        fig.tight_layout()

        # 保存附加信息
        summary.attrs["bin_info"] = (
            bin_info
        )

        summary.attrs["binning"] = (
            "equal-width evidence bins"
        )

        summary.attrs["error_bars"] = (
            "95% CI across pooled trials"
            if not is_count
            else "Poisson sqrt(N) count error"
        )

        return (
            fig,
            summary,
            comparisons,
        )

    return (plot_metric_by_evidence_bins,)


@app.cell
def _(
    compare_agg_func,
    compare_metric_col,
    compare_value_select,
    compare_y_label,
    df_test_selected_upd,
    plot_metric_by_evidence_bins,
):
    plot_metric_by_evidence_bins(
        df_test_selected_upd,
        metric_col=compare_metric_col,
        agg_func=compare_agg_func,
        ylabel=compare_y_label, 
        title=compare_value_select.value,
        drug_name="mk801",
        n_bins=5,
    )
    return


@app.cell
def _(np, pd, plt, stats):
    def plot_metric_by_difficulty(
        df,
        metric_col,
        agg_func="mean",
        ylabel=None,
        title=None,
        drug_name="mk801",
        test_method="ttest_ind",
    ):

        difficulty_order = ["easy", "medium", "hard"]
        groups = ["saline", drug_name]

        colors = {
            "saline": "blue",
            drug_name: "red",
        }

        plot_df = df.copy()

        plot_df = plot_df[
            plot_df["difficulty"].isin(difficulty_order)
            & plot_df["observation_group"].isin(groups)
        ].copy()

        is_count = agg_func in {"len", "size", "count"}

        if not is_count:
            plot_df[metric_col] = pd.to_numeric(
                plot_df[metric_col],
                errors="coerce",
            )
            plot_df = plot_df.dropna(subset=[metric_col])

        plot_df["difficulty"] = pd.Categorical(
            plot_df["difficulty"],
            categories=difficulty_order,
            ordered=True,
        )

        # 所有小鼠的 trial 合并后，按 difficulty 和 treatment 聚合
        if is_count:
            summary = (
                plot_df.groupby(
                    ["observation_group", "difficulty"],
                    observed=True,
                )
                .size()
                .rename("value")
                .reset_index()
            )

            # 对总计数使用 Poisson 误差，而不是 SEM
            summary["error"] = np.sqrt(summary["value"])
            summary["n_trials"] = summary["value"]

        summary = (
            plot_df.groupby(
                ["observation_group", "difficulty"],
                observed=True,
            )[metric_col]
            .agg(
                value="mean",
                std="std",
                n_trials="size",
            )
            .reset_index()
        )

        summary["sem"] = (
            summary["std"]
            / np.sqrt(summary["n_trials"])
        )

        # 95% CI 的半宽
        summary["error"] = (
            stats.t.ppf(
                0.975,
                df=summary["n_trials"] - 1,
            )
            * summary["sem"]
        )

        summary["ci95_lower"] = (
            summary["value"] - summary["error"]
        )

        summary["ci95_upper"] = (
            summary["value"] + summary["error"]
        )

        difficulty_positions = {
            difficulty: position
            for position, difficulty in enumerate(difficulty_order)
        }

        summary["x_position"] = (
            summary["difficulty"]
            .astype(str)
            .map(difficulty_positions)
        )

        # 每个 difficulty 内比较 saline 和 MK801
        comparison_rows = []

        if not is_count:
            for difficulty in difficulty_order:
                df_difficulty = plot_df[
                    plot_df["difficulty"] == difficulty
                ]

                saline_values = df_difficulty.loc[
                    df_difficulty["observation_group"] == "saline",
                    metric_col,
                ].dropna()

                drug_values = df_difficulty.loc[
                    df_difficulty["observation_group"] == drug_name,
                    metric_col,
                ].dropna()

                if saline_values.empty or drug_values.empty:
                    p_value = np.nan

                elif test_method == "mannwhitneyu":
                    p_value = stats.mannwhitneyu(
                        saline_values,
                        drug_values,
                        alternative="two-sided",
                    ).pvalue

                elif test_method == "ttest_ind":
                    p_value = stats.ttest_ind(
                        saline_values,
                        drug_values,
                        equal_var=False,
                        nan_policy="omit",
                    ).pvalue

                else:
                    raise ValueError(
                        "test_method must be "
                        "'mannwhitneyu' or 'ttest_ind'"
                    )

                comparison_rows.append({
                    "difficulty": difficulty,
                    "x_position": difficulty_positions[difficulty],
                    "n_saline": len(saline_values),
                    f"n_{drug_name}": len(drug_values),
                    "p_value": p_value,
                })

        comparisons = pd.DataFrame(comparison_rows)

        def _p_label(p_value):
            if pd.isna(p_value):
                return "n/a"
            if p_value < 0.001:
                return "***"
            if p_value < 0.01:
                return "**"
            if p_value < 0.05:
                return "*"
            return "ns"

        fig, ax = plt.subplots(figsize=(7, 5))

        for group in groups:
            group_summary = (
                summary[
                    summary["observation_group"] == group
                ]
                .sort_values("x_position")
            )

            ax.errorbar(
                group_summary["x_position"],
                group_summary["value"],
                yerr=group_summary["error"].fillna(0),
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=4,
                color=colors[group],
                label=group,
            )

        # 每个 difficulty 上方显示显著性
        if not comparisons.empty:
            y_upper = (
                summary["value"]
                + summary["error"].fillna(0)
            )
            y_lower = (
                summary["value"]
                - summary["error"].fillna(0)
            )

            y_max = y_upper.max()
            y_min = y_lower.min()
            y_range = y_max - y_min

            if y_range == 0:
                y_range = abs(y_max) * 0.1 if y_max != 0 else 1

            for _, row in comparisons.iterrows():
                ax.text(
                    row["x_position"],
                    y_max + y_range * 0.08,
                    _p_label(row["p_value"]),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            ax.set_ylim(
                top=y_max + y_range * 0.20
            )

        ax.set_xticks(range(len(difficulty_order)))
        ax.set_xticklabels(
            ["Easy", "Medium", "Hard"]
        )

        ax.set_xlabel("Difficulty")

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif is_count:
            ax.set_ylabel(f"Total number of {metric_col}")
        else:
            ax.set_ylabel(f"Mean {metric_col} ± SEM")

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(
                f"{metric_col} by evidence "
                "strength and treatment"
            )
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        return fig, summary, comparisons

    return (plot_metric_by_difficulty,)


@app.cell
def _(
    compare_metric_col,
    compare_value_select,
    compare_y_label,
    df_test_selected_upd,
    plot_metric_by_difficulty,
):
    plot_metric_by_difficulty(
        df_test_selected_upd,
        metric_col=compare_metric_col,
        agg_func="mean",
        ylabel=compare_y_label,
        title=compare_value_select.value,
        drug_name="mk801",
        test_method="ttest_ind",
    )
    return


@app.cell
def _(np, pd, plt):
    def plot_metric_by_evidence_bin(
        df,
        metric_col,
        agg_func="mean",
        ylabel=None,
        title=None,
        drug_name="mk801",
        n_bins=5,
        test_method="ttest_ind",
    ):
        from scipy import stats

        groups = ["saline", drug_name]

        colors = {
            "saline": "blue",
            drug_name: "red",
        }

        plot_df = df.copy()

        # 转换 evidence 为数值
        plot_df["abs_total_evidence_strength"] = pd.to_numeric(
            plot_df["abs_total_evidence_strength"],
            errors="coerce",
        )

        # 只保留 saline、drug 和有效 evidence
        plot_df = plot_df[
            plot_df["observation_group"].isin(groups)
        ].dropna(
            subset=[
                "observation_group",
                "abs_total_evidence_strength",
            ]
        ).copy()

        is_count = (
            agg_func in {"len", "size", "count"}
            or agg_func is len
        )

        # 非计数指标需要转换为数值
        if not is_count:
            plot_df[metric_col] = pd.to_numeric(
                plot_df[metric_col],
                errors="coerce",
            )

            plot_df = plot_df.dropna(
                subset=[metric_col]
            ).copy()

        if plot_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig, pd.DataFrame(), pd.DataFrame()

        # 按 evidence 排序后，根据 trial 数进行等频分箱
        # rank(method="first") 可以使每个 bin 的 trial 数尽量相等
        plot_df["evidence_rank"] = (
            plot_df["abs_total_evidence_strength"]
            .rank(method="first")
        )

        actual_n_bins = min(
            n_bins,
            len(plot_df),
        )

        plot_df["evidence_bin_number"] = pd.qcut(
            plot_df["evidence_rank"],
            q=actual_n_bins,
            labels=False,
        )

        # 每个 bin 的实际中心、范围和 trial 数
        bin_info = (
            plot_df.groupby(
                "evidence_bin_number",
                observed=True,
            )["abs_total_evidence_strength"]
            .agg(
                center="mean",
                left="min",
                right="max",
                n_trials="size",
            )
        )

        bin_order = bin_info.index.tolist()
        bin_centers = bin_info["center"].to_dict()

        # 所有小鼠的 trial 合并后计算
        if is_count:
            summary = (
                plot_df.groupby(
                    [
                        "observation_group",
                        "evidence_bin_number",
                    ],
                    observed=True,
                )
                .size()
                .rename("value")
                .reset_index()
            )

            # 总 trial 数没有 SEM；
            # sqrt(N) 表示 Poisson count error
            summary["error"] = np.sqrt(
                summary["value"]
            )
            summary["n_trials"] = summary["value"]

        else:
            summary = (
                plot_df.groupby(
                    [
                        "observation_group",
                        "evidence_bin_number",
                    ],
                    observed=True,
                )[metric_col]
                .agg(
                    value="mean",
                    std="std",
                    n_trials="size",
                )
                .reset_index()
            )

            summary["sem"] = (
                summary["std"]
                / np.sqrt(summary["n_trials"])
            )

            summary["error"] = (
                stats.t.ppf(
                    0.975,
                    df=summary["n_trials"] - 1,
                )
                * summary["sem"]
            )

        # 横坐标使用每个 bin 内实际 evidence 的平均值
        summary["bin_center"] = (
            summary["evidence_bin_number"]
            .map(bin_centers)
            .astype(float)
        )

        # 每个 evidence bin 内比较 saline 和 drug
        comparison_rows = []

        if not is_count:
            for evidence_bin_number, df_bin in plot_df.groupby(
                "evidence_bin_number",
                observed=True,
            ):
                saline_values = df_bin.loc[
                    df_bin["observation_group"] == "saline",
                    metric_col,
                ].dropna()

                drug_values = df_bin.loc[
                    df_bin["observation_group"] == drug_name,
                    metric_col,
                ].dropna()

                if saline_values.empty or drug_values.empty:
                    p_value = np.nan

                elif test_method == "mannwhitneyu":
                    p_value = stats.mannwhitneyu(
                        saline_values,
                        drug_values,
                        alternative="two-sided",
                    ).pvalue

                elif test_method == "ttest_ind":
                    p_value = stats.ttest_ind(
                        saline_values,
                        drug_values,
                        equal_var=False,
                        nan_policy="omit",
                    ).pvalue

                else:
                    raise ValueError(
                        "test_method must be "
                        "'mannwhitneyu' or 'ttest_ind'"
                    )

                comparison_rows.append({
                    "evidence_bin_number": (
                        evidence_bin_number
                    ),
                    "bin_center": (
                        bin_centers[evidence_bin_number]
                    ),
                    "n_saline": len(saline_values),
                    f"n_{drug_name}": len(drug_values),
                    "p_value": p_value,
                })

        comparisons = pd.DataFrame(comparison_rows)

        def _p_label(p_value):
            if pd.isna(p_value):
                return "n/a"
            if p_value < 0.001:
                return "***"
            if p_value < 0.01:
                return "**"
            if p_value < 0.05:
                return "*"
            return "ns"

        fig, ax = plt.subplots(figsize=(7, 5))

        # 绘制 saline 和 drug 的均值 ± SEM
        for group in groups:
            group_summary = (
                summary[
                    summary["observation_group"] == group
                ]
                .sort_values("bin_center")
            )

            if group_summary.empty:
                continue

            ax.errorbar(
                group_summary["bin_center"],
                group_summary["value"],
                yerr=group_summary["error"].fillna(0),
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=4,
                color=colors[group],
                label=group,
            )

        # 添加每个 bin 的显著性结果
        if not comparisons.empty:
            y_upper = (
                summary["value"]
                + summary["error"].fillna(0)
            )
            y_lower = (
                summary["value"]
                - summary["error"].fillna(0)
            )

            y_max = y_upper.max()
            y_min = y_lower.min()
            y_range = y_max - y_min

            if not np.isfinite(y_range) or y_range == 0:
                y_range = (
                    abs(y_max) * 0.1
                    if np.isfinite(y_max) and y_max != 0
                    else 1
                )

            significance_y = y_max + y_range * 0.08

            for _, row in comparisons.iterrows():
                ax.text(
                    row["bin_center"],
                    significance_y,
                    _p_label(row["p_value"]),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            ax.set_ylim(
                top=y_max + y_range * 0.20
            )

        # 横坐标位于每个 bin 的实际 evidence 平均值
        ax.set_xticks(
            [
                bin_centers[bin_number]
                for bin_number in bin_order
            ]
        )

        # 标签显示每个 bin 的实际 evidence 范围
        ax.set_xticklabels(
            [
                (
                    f"{bin_info.loc[bin_number, 'left']:.2f}–"
                    f"{bin_info.loc[bin_number, 'right']:.2f}"
                )
                for bin_number in bin_order
            ]
        )

        ax.set_xlabel(
            "Absolute total evidence strength"
        )

        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif is_count:
            ax.set_ylabel(
                f"Total number of {metric_col}"
            )
        else:
            ax.set_ylabel(
                f"Mean {metric_col} (95% CI)"
            )

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(
                f"{metric_col} by evidence strength "
                f"and treatment"
            )

        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        # 将 bin 信息附加到输出，方便检查
        summary.attrs["bin_info"] = bin_info
        summary.attrs["binning"] = (
            "equal number of pooled trials"
        )

        return fig, summary, comparisons

    return (plot_metric_by_evidence_bin,)


@app.cell
def _(
    compare_metric_col,
    compare_value_select,
    compare_y_label,
    df_test_selected_upd,
    plot_metric_by_evidence_bin,
):
    plot_metric_by_evidence_bin(
        df_test_selected_upd,
        metric_col=compare_metric_col,
        title=compare_value_select.value,
        agg_func="mean",
        ylabel=compare_y_label,
        drug_name="mk801",
        test_method="ttest_ind",
        n_bins=5
    )
    return


@app.cell
def _(behavior_utils, metric_paired_dates, pd, plt, stats):
    def observation_to_color(observation):
        observation = str(observation).lower()
        if "mk801" in observation:
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
            ax.scatter([], [], color="red", edgecolor="black", label="mk801")
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
            behavior_utils.observation_group
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
            drug_name="mk801",
        )
        observation_summary.attrs["paired_comparison"] = paired_summary

        groups = ["saline", "mk801"]
        colors = {"saline": "blue", "mk801": "red"}
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
                "No paired previous-saline/mk801 observations to compare.",
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
                    [paired_row["saline"], paired_row["mk801"]],
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
                    "nearest_previous_saline_for_each_mk801"
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

    return (
        add_significance_bar,
        p_value_to_label,
        plot_metric_by_date,
        plot_metric_by_observation,
    )


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
        options=["fixation_breaks", "eager_pokes", "port2_pokes_num", "reaction_time", "time_between_trials", "trial", "correct", "repeating_bias_session"],
        value="fixation_breaks",
        label="observation",
    )

    observation_select
    return (observation_select,)


@app.cell
def _(
    df_test_selected_upd,
    observation_select,
    plot_group,
    plot_metric_by_date,
):
    if observation_select.value == "trial":
        _aggfunc = "len"
    else:
        _aggfunc = "mean"
    plot_group(df_test_selected_upd, "", metric_col=observation_select.value, metric_name=observation_select.value, agg_func=_aggfunc, plot_func=plot_metric_by_date)
    return


@app.cell
def _(
    df_test_selected_upd,
    observation_select,
    pd,
    plot_group,
    plot_metric_by_observation,
):
    _df = df_test_selected_upd.copy()
    _df["evidence_bin"] = pd.cut(df_test_selected_upd["total_evidence_strength"].abs(), bins=6, labels=False)

    _df_max = _df[_df["evidence_bin"] == _df["evidence_bin"].max()].copy()
    if observation_select.value == "trial":
        _aggfunc = "len"
    else:
        _aggfunc = "mean"
    plot_group(_df_max, "lapse", metric_col=observation_select.value, metric_name=observation_select.value, agg_func=_aggfunc, plot_func=plot_metric_by_observation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # work bout
    """)
    return


@app.cell
def _(df_test_selected_upd, plt, sns):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _palette = {"saline": "blue", "mk801": "red"}
    sns.ecdfplot(
        data=df_test_selected_upd,
        x="inter_trial_interval",
        hue="observation_group",
        stat="proportion",
        ax=_ax,
        palette=_palette
    )

    _ax.set_xscale("log")
    _ax.set_xlabel("Inter-trial interval (s)")
    _ax.set_ylabel("Cumulative probability")
    _ax.set_ylim(0, 1)
    _ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_test_selected_upd, np, pd):
    import ast

    _keys = ["subject", "year_month_day", "session", "observation_group"]
    _df = df_test_selected_upd.sort_values(_keys + ["trial"]).copy()
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
        ~work_bouts["engaged"]
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

        # 观测窗口：第一个 trial 开始至最后一个 trial 结束
        session_s = _bout["end_time"].max() - _bout["start_time"].min()
        session_s = session_s if session_s > 0 else np.nan
        session_h = session_s / 3600

        _rows.append({
            **dict(zip(_keys, _key)),
            "session_duration_s": session_s,

            # 观察到的 True → False 次数 / 小时
            "break_frequency": _bout["break_onset"].sum() / session_h,

            # 每个 session 内完整 break 的平均时长
            "break_duration": complete_breaks["duration_s"].mean(),

            # 每个 session 内完整工作段的平均 trial 数和时长
            "trials_per_bout": complete_working["n_trials"].mean(),
            "bout_duration": complete_working["duration_s"].mean(),

            # 包含首尾不完整工作段的已观测时间
            "working_time_fraction": working["duration_s"].sum() / session_s,

            # 有中央 poke 的 trial 数 / 小时
            "trial_initiation_rate": initiated_counts.loc[_key] / session_h,

            "n_break_onsets": int(_bout["break_onset"].sum()),
            "n_initiated_trials": int(initiated_counts.loc[_key]),
            "working_time_s": working["duration_s"].sum(),
        })

    session_metrics = pd.DataFrame(_rows)
    session_metrics
    return session_metrics, work_bouts


@app.cell
def _(work_bouts):
    work_bouts
    return


@app.cell
def _(mo, work_bouts):
    # mean_bout_duration_s = work_bouts.groupby(["subject", "year_month_day", "observation_group", "engaged"])["duration_s"].mean().reset_index().rename(columns={"duration_s": "mean_bout_duration_s"})
    # sum_bout_n_trials = work_bouts.groupby(["subject", "year_month_day", "observation_group", "engaged"])["n_trials"].sum().reset_index().rename(columns={"n_trials": "sum_bout_n_trials"})
    # sum_bout_break_onset = work_bouts.groupby(["subject", "year_month_day", "observation_group", "engaged"], observed=True)["break_onset"].sum().reset_index().rename(columns={"break_onset": "sum_bout_break_onset"})
    mean_bout_duration_s = work_bouts.groupby(["bout_id", "observation_group", "engaged"])["duration_s"].mean().reset_index().rename(columns={"duration_s": "mean_bout_duration_s"})
    sum_bout_n_trials = work_bouts.groupby(["bout_id", "observation_group", "engaged"])["n_trials"].sum().reset_index().rename(columns={"n_trials": "sum_bout_n_trials"})
    sum_bout_break_onset = work_bouts.groupby(["bout_id", "observation_group", "engaged"], observed=True)["break_onset"].sum().reset_index().rename(columns={"break_onset": "sum_bout_break_onset"})

    merit_select = mo.ui.dropdown(
        options=[
            "mean_bout_duration_s",
            "sum_bout_n_trials",
            "sum_bout_break_onset",
        ],
        value="mean_bout_duration_s",
        label="choose meric",
        allow_select_none=False,
    )

    merit_select
    return (
        mean_bout_duration_s,
        merit_select,
        sum_bout_break_onset,
        sum_bout_n_trials,
    )


@app.cell
def _(
    mean_bout_duration_s,
    merit_select,
    plt,
    sns,
    sum_bout_break_onset,
    sum_bout_n_trials,
):
    _metric_configs = {
        "mean_bout_duration_s": (
            mean_bout_duration_s,
            "Mean bout duration (s)",
        ),
        "sum_bout_n_trials": (
            sum_bout_n_trials,
            "Total trials",
        ),
        "sum_bout_break_onset": (
            sum_bout_break_onset,
            "Total break-onset",
        ),
    }

    _column = merit_select.value
    _data, _ylabel = _metric_configs[_column]

    _order = ["saline", "mk801"]
    _palette = {"saline": "blue", "mk801": "red"}

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 10))

    for _j, _engaged in enumerate([True, False]):
        _box_ax = _axes[0, _j]
        _ecdf_ax = _axes[1, _j]
        _title = "Working bouts" if _engaged else "Break bouts"

        _plot_df = (
            _data.loc[
                _data["engaged"].eq(_engaged)
                & _data["observation_group"].isin(_order)
            ]
            .dropna(subset=[_column])
        )

        if _plot_df.empty:
            for _ax in [_box_ax, _ecdf_ax]:
                _ax.set_title(_title)
                _ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center",
                    transform=_ax.transAxes,
                )
            continue

        # 第一排：空心箱线图
        sns.boxplot(
            data=_plot_df,
            x="observation_group",
            y=_column,
            hue="observation_group",
            order=_order,
            hue_order=_order,
            palette=_palette,
            fill=False,
            linewidth=1.5,
            dodge=False,
            legend=False,
            ax=_box_ax,
        )

        _box_ax.set_title(_title)
        _box_ax.set_xlabel("Observation group")
        _box_ax.set_ylabel(_ylabel)

        # 第二排：累计分布图
        sns.ecdfplot(
            data=_plot_df,
            x=_column,
            hue="observation_group",
            hue_order=_order,
            palette=_palette,
            stat="proportion",
            linewidth=2,
            ax=_ecdf_ax,
        )

        # 计数指标可能为 0，symlog 可以保留这些值
        if _column in {"mean_bout_duration_s", "sum_bout_n_trials"}:
            _ecdf_ax.set_xscale("log")
        _ecdf_ax.set_title(f"{_title} (ECDF)")
        _ecdf_ax.set_xlabel(_ylabel)
        _ecdf_ax.set_ylabel("Cumulative probability")
        _ecdf_ax.set_ylim(0, 1)
        _ecdf_ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(
    behavior_utils,
    metric_paired_dates,
    np,
    pd,
    plt,
    session_metrics,
    stats,
):
    _metrics = {
        "break_frequency": "Break onsets / observed hour",
        "break_duration": "Mean break duration (s)",
        "trials_per_bout": "Mean trials per working bout",
        "bout_duration": "Mean working bout duration (s)",
        "working_time_fraction": "Working time fraction",
        "trial_initiation_rate": "Trials initiated / observed hour",
    }

    _groups = ["saline", "mk801"]
    _colors = {"saline": "blue", "mk801": "red"}

    # 1. 统一动物编号和日期格式
    _sessions = session_metrics.dropna(
        subset=["subject", "year_month_day", "observation_group"]
    ).copy()

    _sessions["subject"] = _sessions["subject"].astype(str)
    _sessions["year_month_day"] = pd.to_datetime(
        _sessions["year_month_day"],
        errors="coerce",
    ).dt.normalize()

    _sessions[list(_metrics)] = _sessions[list(_metrics)].replace(
        [np.inf, -np.inf], np.nan
    )

    # 同鼠、同日、同条件多个 session：取 session 指标的算术平均
    _daily = (
        _sessions.groupby(
            ["subject", "year_month_day", "observation_group"],
            observed=True,
            as_index=False,
        )[list(_metrics)]
        .mean()
    )

    # 2. 绘图及统计
    _fig, _axes = plt.subplots(2, 3, figsize=(18, 11))

    bout_date_pairs = {}
    bout_mouse_pairs = {}
    _test_rows = []

    for _ax, (_metric, _label) in zip(_axes.flat, _metrics.items()):

        # 使用已有日期配对表；函数会删除当前指标不完整的配对
        _paired = behavior_utils.add_metric_values_to_pairs(
            observation_summary=_daily,
            paired_dates=metric_paired_dates,
            metric_col=_metric,
            subject_col="subject",
            date_col="year_month_day",
            drug_name="mk801",
        )

        bout_date_pairs[_metric] = _paired

        if _paired.empty:
            _ax.set_title(_label)
            _ax.text(
                0.5, 0.5, "No complete date pairs",
                ha="center", va="center",
                transform=_ax.transAxes,
            )
            _ax.set_axis_off()

            _test_rows.append({
                "metric": _metric,
                "n_date_pairs": 0,
                "n_mice": 0,
                "t_statistic": np.nan,
                "p_value": np.nan,
                "note": "No complete date pairs",
            })
            continue

        # 3. 图上展示全部日期对
        _box = _ax.boxplot(
            [
                _paired["saline"].to_numpy(),
                _paired["mk801"].to_numpy(),
            ],
            tick_labels=_groups,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
        )

        for _patch, _group in zip(_box["boxes"], _groups):
            _patch.set_facecolor(_colors[_group])
            _patch.set_edgecolor(_colors[_group])
            _patch.set_alpha(0.2)

        for _, _row in _paired.iterrows():
            _ax.plot(
                [1, 2],
                [_row["saline"], _row["mk801"]],
                color="gray",
                alpha=0.35,
                linewidth=1,
                zorder=2,
            )

        for _position, _group in enumerate(_groups, start=1):
            _ax.scatter(
                np.full(len(_paired), _position),
                _paired[_group],
                color=_colors[_group],
                edgecolor="black",
                linewidth=0.5,
                s=45,
                alpha=0.8,
                zorder=3,
            )

        # 4. 检验使用每只鼠的完整日期对均值
        # 先删除不完整日期对，再同时平均两个条件，保持配对关系
        _mouse_pairs = (
            _paired.groupby("subject", observed=True)[_groups]
            .mean()
        )
        bout_mouse_pairs[_metric] = _mouse_pairs

        _n_mice = len(_mouse_pairs)
        _t = np.nan
        _p = np.nan
        _note = ""

        if _n_mice < 2:
            _note = "Fewer than 2 paired mice"
        else:
            _diff = _mouse_pairs["mk801"] - _mouse_pairs["saline"]

            if _diff.std(ddof=1) == 0:
                _note = "Zero variance of paired differences"
            else:
                _test = stats.ttest_rel(
                    _mouse_pairs["mk801"].to_numpy(),
                    _mouse_pairs["saline"].to_numpy(),
                    alternative="two-sided",
                )
                _t = float(_test.statistic)
                _p = float(_test.pvalue)

        _test_rows.append({
            "metric": _metric,
            "n_date_pairs": len(_paired),
            "n_mice": _n_mice,
            "t_statistic": _t,
            "p_value": _p,
            "note": _note,
        })

        # p 值直接显示在标题中
        _p_text = (
            f"Paired t-test across mice: p = {_p:.3g}"
            if np.isfinite(_p)
            else _note or "Test unavailable"
        )

        _ax.set_title(
            f"{_label}\n"
            f"{len(_paired)} date pairs, {_n_mice} mice\n"
            f"{_p_text}",
            fontsize=10,
        )
        _ax.set_xlabel("")
        _ax.set_ylabel(_label)
        _ax.grid(axis="y", alpha=0.3)
        _ax.set_axisbelow(True)

        if _metric == "working_time_fraction":
            _ax.set_ylim(-0.02, 1.02)

    _fig.tight_layout()
    plt.show()

    # 每个指标的配对数量、动物数量和检验结果
    bout_date_test_results = pd.DataFrame(_test_rows)
    bout_date_test_results
    return


@app.cell
def _(session_metrics, work_bouts):
    _subject_keys = ["subject", "observation_group"]

    # 1. 每只动物、每个实验组：累加各 session 的时间和计数
    # 使用你文件里现有的 session_duration_s 列名
    metrics_df = (
        session_metrics
        .groupby(_subject_keys, observed=True, as_index=False)
        .agg(
            n_sessions=("session", "size"),
            observed_duration_s=("session_duration_s", "sum"),
            n_break_onsets=("n_break_onsets", "sum"),
            n_initiated_trials=("n_initiated_trials", "sum"),
            working_time_s=("working_time_s", "sum"),
        )
    )

    # 2. 完整工作 bout：每个 bout 权重相同
    _working_means = (
        work_bouts.loc[
            work_bouts["engaged"].eq(True)
            & work_bouts["complete"]
        ]
        .groupby(_subject_keys, observed=True, as_index=False)
        .agg(
            mean_trials_per_bout=("n_trials", "mean"),
            mean_bout_duration_s=("duration_s", "mean"),
        )
    )

    # 3. 完整休息 bout：每个 bout 权重相同
    _break_means = (
        work_bouts.loc[
            work_bouts["engaged"].eq(False)
            & work_bouts["complete"]
        ]
        .groupby(_subject_keys, observed=True, as_index=False)
        .agg(
            mean_break_duration_s=("duration_s", "mean"),
        )
    )

    # 4. 合并；没有完整 bout 时，对应均值保留 NaN
    metrics_df = (
        metrics_df
        .merge(_working_means, on=_subject_keys, how="left")
        .merge(_break_means, on=_subject_keys, how="left")
    )

    # 5. 计算频率和时间占比
    _observed_s = metrics_df["observed_duration_s"].where(
        metrics_df["observed_duration_s"] > 0
    )
    _observed_h = _observed_s / 3600

    metrics_df["break_frequency_per_h"] = (
        metrics_df["n_break_onsets"] / _observed_h
    )

    metrics_df["working_time_fraction"] = (
        metrics_df["working_time_s"] / _observed_s
    )

    metrics_df["trial_initiation_rate_per_h"] = (
        metrics_df["n_initiated_trials"] / _observed_h
    )

    metrics_df
    return (metrics_df,)


@app.cell
def _(np, plt, stats):
    def plot_bouts_group_by_subjects(
        df,
        metric_col="correct",
        agg_func="mean",
        drug_name="mk801",
        ylabel=None,
        title=None,
        test_method="wilcoxon",
        ax=None,
    ):
        if test_method not in ["wilcoxon", "ttest_rel"]:
            raise ValueError(
                "test_method must be 'wilcoxon' or 'ttest_rel'"
            )

        groups = ["saline", drug_name]
        colors = {
            "saline": "blue",
            drug_name: "red",
        }

        # 提取有效数据
        plot_df = df[
            ["subject", "observation_group", metric_col]
        ].copy()

        plot_df[metric_col] = plot_df[metric_col].replace(
            [np.inf, -np.inf], np.nan
        )

        plot_df = (
            plot_df
            .dropna(subset=["subject", "observation_group", metric_col])
            .loc[lambda x: x["observation_group"].isin(groups)]
        )

        # 每只鼠、每个条件汇总
        # metrics_df 已经每鼠每条件一行，因此 mean 不会改变数值
        aggregation = "size" if agg_func == "len" else agg_func

        mouse_summary = (
            plot_df.groupby(
                ["subject", "observation_group"],
                observed=True,
                as_index=False,
            )
            .agg(value=(metric_col, aggregation))
        )

        # 保留原函数的特殊处理
        if metric_col == "previous_same_choice_numeric":
            mouse_summary["value"] -= 0.5

        # 每只鼠一行，只保留两个条件都有有效值的鼠
        paired_summary = (
            mouse_summary
            .pivot(
                index="subject",
                columns="observation_group",
                values="value",
            )
            .reindex(columns=groups)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=groups)
            .reset_index()
        )

        n_pairs = len(paired_summary)

        paired_summary.attrs.update({
            "test_method": test_method,
            "n_pairs": n_pairs,
            "p_value": np.nan,
            "test_note": "",
        })

        # 支持独立画图，也支持外部传入子图
        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        ax.set_xlabel("Observation")
        ax.set_ylabel(ylabel or metric_col)
        ax.set_title(
            f"{title or metric_col}\n"
            f"n = {n_pairs} paired mice"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # 没有配对数据
        if n_pairs == 0:
            ax.set_xticks([1, 2], groups)
            ax.set_xlim(0.5, 2.5)
            ax.text(
                0.5, 0.5, "No paired data",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            paired_summary.attrs["test_note"] = "No paired data"

            if own_figure:
                fig.tight_layout()

            return fig, paired_summary

        # 箱线图
        data = [
            paired_summary[group].to_numpy(dtype=float)
            for group in groups
        ]

        boxplot = ax.boxplot(
            data,
            tick_labels=groups,
            patch_artist=True,
            showmeans=True,
        )

        for patch, group in zip(boxplot["boxes"], groups):
            patch.set_facecolor(colors[group])
            patch.set_edgecolor(colors[group])
            patch.set_alpha(0.2)

        # 每只鼠一条配对线
        for _, row in paired_summary.iterrows():
            ax.plot(
                [1, 2],
                [row["saline"], row[drug_name]],
                color="gray",
                alpha=0.35,
                linewidth=1,
                zorder=2,
            )

        # 每只鼠的散点
        for position, group in enumerate(groups, start=1):
            ax.scatter(
                np.full(n_pairs, position),
                paired_summary[group],
                color=colors[group],
                edgecolor="black",
                linewidth=0.5,
                s=45,
                alpha=0.8,
                zorder=3,
            )

        # 配对检验
        saline_values, drug_values = data
        differences = drug_values - saline_values

        p_value = np.nan
        test_note = ""

        if n_pairs < 2:
            test_note = "Too few pairs for testing"
        elif np.all(differences == 0):
            # 全部差值为零时，跳过检验并明确标记
            test_note = "All paired differences = 0"
        elif test_method == "wilcoxon":
            p_value = stats.wilcoxon(
                differences,
                alternative="two-sided",
            ).pvalue
        else:
            p_value = stats.ttest_rel(
                drug_values,
                saline_values,
            ).pvalue

        if not np.isfinite(p_value) and not test_note:
            test_note = "Test unavailable"

        paired_summary.attrs["p_value"] = p_value
        paired_summary.attrs["test_note"] = test_note

        # 显著性标记
        if np.isfinite(p_value):
            if p_value < 0.0001:
                significance = "****"
            elif p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"

            y_values = np.concatenate(data)
            y_min = y_values.min()
            y_max = y_values.max()
            y_range = y_max - y_min

            if y_range == 0:
                y_range = abs(y_max) * 0.1 if y_max != 0 else 0.1

            bar_y = y_max + y_range * 0.12
            bar_h = y_range * 0.06

            ax.plot(
                [1, 1, 2, 2],
                [bar_y, bar_y + bar_h, bar_y + bar_h, bar_y],
                color="black",
                linewidth=1,
            )
            ax.text(
                1.5,
                bar_y + bar_h,
                significance,
                ha="center",
                va="bottom",
            )
            ax.set_ylim(
                top=bar_y + bar_h + y_range * 0.15
            )
        else:
            ax.text(
                0.5, 0.98,
                test_note,
                ha="center",
                va="top",
                fontsize=9,
                transform=ax.transAxes,
            )

        if own_figure:
            fig.tight_layout()

        return fig, paired_summary

    return (plot_bouts_group_by_subjects,)


@app.cell
def _(metrics_df, pd, plot_bouts_group_by_subjects, plt):
    metrics = {
        "break_frequency_per_h": "Break onsets / observed hour",
        "mean_break_duration_s": "Mean break duration (s)",
        "mean_trials_per_bout": "Mean trials per working bout",
        "mean_bout_duration_s": "Mean working bout duration (s)",
        "working_time_fraction": "Working time fraction",
        "trial_initiation_rate_per_h": "Trials initiated / observed hour",
    }

    _fig, _axes = plt.subplots(2, 3, figsize=(18, 10))

    bout_paired_results = {}
    _test_rows = []

    for _ax, (_metric, _label) in zip(_axes.flat, metrics.items()):
        _, _paired = plot_bouts_group_by_subjects(
            df=metrics_df,
            metric_col=_metric,
            agg_func="mean",
            drug_name="mk801",
            ylabel=_label,
            title=_label,
            test_method="wilcoxon",
            ax=_ax,
        )

        bout_paired_results[_metric] = _paired

        _test_rows.append({
            "metric": _metric,
            "n_pairs": _paired.attrs["n_pairs"],
            "test_method": _paired.attrs["test_method"],
            "p_value": _paired.attrs["p_value"],
            "note": _paired.attrs["test_note"],
        })

        if _metric == "working_time_fraction" and len(_paired) > 0:
            # 上方保留显著性标记所需的空间
            _ax.set_ylim(bottom=0)

    _fig.tight_layout()
    plt.show()

    # 查看每个指标的样本量和 p 值
    bout_test_results = pd.DataFrame(_test_rows)
    bout_test_results
    return


@app.cell
def _():
    return


@app.cell
def _(df_test_selected_upd, np):
    keys = ["subject", "session"]

    # 1. Trial level
    trials = (
        df_test_selected_upd
        .sort_values(keys + ["trial"])
        .copy()
    )

    assert trials["engaged"].notna().all()
    trials["engaged"] = trials["engaged"].astype(bool)

    g = trials.groupby(keys, observed=True)

    trials["interval_end"] = (
        g["TRIAL_START"].shift(-1)
        .fillna(trials["TRIAL_END"])
    )

    trials["bout_id"] = (
        g["engaged"].shift().ne(trials["engaged"])
        | g.cumcount().eq(0)
    ).cumsum()

    # 2. Bout level
    bouts = (
        trials.groupby(keys + ["bout_id"], observed=True)
        .agg(
            engaged=("engaged", "first"),
            n_trials=("trial", "size"),
            start_time=("TRIAL_START", "first"),
            end_time=("interval_end", "last"),
        )
        .reset_index()
    )

    bouts["duration_s"] = (
        bouts["end_time"] - bouts["start_time"]
    )

    bg = bouts.groupby(keys, observed=True)

    bouts["break_onset"] = (
        ~bouts["engaged"]
        & bg["engaged"].shift().eq(True)
    )

    bouts["complete"] = (
        bg.cumcount().gt(0)
        & bg.cumcount(ascending=False).gt(0)
    )

    # 3. Session level
    session_metrics = (
        bouts.assign(
            working=bouts["engaged"].astype(int),
            breaking=(~bouts["engaged"]).astype(int),
            complete_working=(
                bouts["complete"] & bouts["engaged"]
            ).astype(int),
            complete_break=(
                bouts["complete"] & ~bouts["engaged"]
            ).astype(int),
            working_duration=bouts["duration_s"].where(
                bouts["engaged"], 0
            ),
            break_duration=bouts["duration_s"].where(
                bouts["complete"] & ~bouts["engaged"]
            ),
            bout_duration=bouts["duration_s"].where(
                bouts["complete"] & bouts["engaged"]
            ),
            bout_trials=bouts["n_trials"].where(
                bouts["complete"] & bouts["engaged"]
            ),
        )
        .groupby(keys, observed=True)
        .agg(
            n_trials=("n_trials", "sum"),
            n_working_bouts=("working", "sum"),
            n_break_bouts=("breaking", "sum"),
            n_break_onsets=("break_onset", "sum"),
            n_complete_working_bouts=("complete_working", "sum"),
            n_complete_break_bouts=("complete_break", "sum"),
            mean_break_duration_s=("break_duration", "mean"),
            mean_trials_per_bout=("bout_trials", "mean"),
            mean_bout_duration_s=("bout_duration", "mean"),
            working_duration_s=("working_duration", "sum"),
            session_start=("start_time", "min"),
            session_end=("end_time", "max"),
        )
        .reset_index()
    )

    # 4. Derived metrics
    session_metrics["observed_duration_s"] = (
        session_metrics["session_end"]
        - session_metrics["session_start"]
    )

    hours = session_metrics["observed_duration_s"] / 3600
    hours = hours.replace(0, np.nan)

    session_metrics["break_frequency_per_h"] = (
        session_metrics["n_break_onsets"] / hours
    )

    session_metrics["trial_rate_per_h"] = (
        session_metrics["n_trials"] / hours
    )

    session_metrics["working_time_fraction"] = (
        session_metrics["working_duration_s"]
        / session_metrics["observed_duration_s"].replace(0, np.nan)
    )

    session_metrics = session_metrics.drop(
        columns=[
            "working_duration_s",
            "session_start",
            "session_end",
        ]
    )
    return (session_metrics,)


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
    df_test_selected_upd,
    fit_mouse_select,
    injection_info_df,
    plot_test,
    stimulus_col,
):
    psychometric_paired_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=fit_mouse_select.value,
        drug_name="mk801"
    )

    psychometric_fig, psychometric_summary = (
        plot_test.plot_condition_psychometric_curves(
            df_test_selected_upd,
            group_name="MK801 vs Saline",
            paired_dates=psychometric_paired_dates,
            x_col=stimulus_col,
            y_col="first_choice_numeric",
            valueType="continue",
            bins=6,
            min_trials=20,
            drug_name="mk801",
        )
    )

    psychometric_fig
    return


@app.cell
def _(df_test_selected_upd):
    import statsmodels.api as sm
    repeating_bias_saline = []
    repeating_bias_mk801 = []
    for _session in df_test_selected_upd['session'].unique():
        _df_session = df_test_selected_upd[df_test_selected_upd['session'] == _session]
        if _df_session.empty:
            continue
        _model_df = df_test_selected_upd[["first_choice_numeric", "total_evidence_strength", "previous_first_choice_numeric"]].dropna()
        _X = sm.add_constant(
            _model_df[
                ["total_evidence_strength", "previous_first_choice_numeric"]
            ]
        )
        _model = sm.GLM(
            _model_df["first_choice_numeric"],
            _X,
            family=sm.families.Binomial(),
        ).fit()
        if "saline" in _df_session['observation_group'].unique():
            repeating_bias_saline.append(_model.params["previous_first_choice_numeric"])
        elif "mk801" in _df_session['observation_group'].unique():
            repeating_bias_mk801.append(_model.params["previous_first_choice_numeric"])
    return


@app.cell
def _(df_test_selected_upd, utils_test):
    df_test_selected_saline = df_test_selected_upd[df_test_selected_upd["observation_group"] == "saline"]
    df_test_selected_mk801 = df_test_selected_upd[df_test_selected_upd["observation_group"] == "mk801"]
    utils_test.psychometric_plot_easy_logistic(
        df_test_selected_saline.dropna(subset=["total_evidence_strength", "first_choice_numeric"]),
        x="total_evidence_strength",
        y="first_choice_numeric",
        point_kwargs={
            "color": "blue",
            "alpha": 0.12,
            "label": "_nolegend_",
            "markersize": 3,
        },
        line_kwargs={
            "color": "blue",
            "alpha": 0.55,
            "linewidth": 1.4,
            "label": "saline"
        },
        valueType="continue",
        bins=6,
        log=False
    )
    utils_test.psychometric_plot_easy_logistic(
        df_test_selected_mk801.dropna(subset=["total_evidence_strength", "first_choice_numeric"]),
        x="total_evidence_strength",
        y="first_choice_numeric",
        point_kwargs={
            "color": "red",
            "alpha": 0.12,
            "label": "_nolegend_",
            "markersize": 3,
        },
        line_kwargs={
            "color": "red",
            "alpha": 0.55,
            "linewidth": 1.4,
            "label": "mk801"
        },
        valueType="continue",
        bins=6,
        log=False
    )
    return


if __name__ == "__main__":
    app.run()
