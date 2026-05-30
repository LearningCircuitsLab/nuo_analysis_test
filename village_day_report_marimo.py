import marimo

__generated_with = "0.23.4"
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

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import behavior_utils
    import plot_test

    from lecilab_behavior_analysis import utils as utils
    from lecilab_behavior_analysis import df_transforms as dft
    from lecilab_behavior_analysis import plots
    from lecilab_behavior_analysis.figure_maker import (
        session_summary_figure,
        subject_progress_figure,
    )

    warnings.filterwarnings("ignore")
    return (
        Path,
        behavior_utils,
        colors,
        dft,
        mpl,
        np,
        pd,
        plot_test,
        plots,
        plt,
        session_summary_figure,
        utils,
    )


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
    return (mouse_select,)


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
    mouse_select,
    pd,
    project,
    safe_rsync_cluster_data,
    utils,
):
    mouse = mouse_select.value

    local_path = Path(utils.get_outpath()) / project / "sessions" / mouse
    local_path.mkdir(parents=True, exist_ok=True)

    if download_button.value:
        for _mouse in animals:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / _mouse
            _local_path.mkdir(parents=True, exist_ok=True)

            safe_rsync_cluster_data(
                project_name=project,
                file_path="sessions/{}/{}.csv".format(_mouse, _mouse),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )


    df_all = pd.read_csv(local_path / Path(f"{mouse}.csv"), sep=";")
    return df_all, mouse


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
    _default_date = available_dates[-1]
    date_select = mo.ui.dropdown(
        options=available_dates,
        value=_default_date,
        label="Date",
    )

    date_select
    return (date_select,)


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
        value=[animals[0]],
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
    hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    return hM3Dq_mice, hM4Di_mice


@app.cell
def _(
    Path,
    dft,
    fit_mouse_select,
    mouse,
    pd,
    project,
    safe_rsync_cluster_data,
    update_processData_button,
    utils,
):
    parameters_output_dir = (
        Path(utils.get_outpath()) / project / "processed" / "parameters_for_fit" / "manual"
    )
    stim_columns = {
        "df_test_vis": "visual",
        "df_test_aud": "auditory",
    }

    selected_mice = list(fit_mouse_select.value)
    if not selected_mice:
        selected_mice = [mouse]

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
        df_manual = df_all_mouse[df_all_mouse["run_mode"] == "Manual"]
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

    df_test_vis = (
        pd.concat(combined_results["df_test_vis"], ignore_index=True)
        if combined_results["df_test_vis"]
        else pd.DataFrame()
    )
    df_test_aud_raw = (
        pd.concat(combined_results["df_test_aud"], ignore_index=True)
        if combined_results["df_test_aud"]
        else pd.DataFrame()
    )
    return (df_test_aud_raw,)


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
    injection_info_df = pd.read_excel(injection_info_file_path)
    injection_info_df
    return (injection_info_df,)


@app.cell
def _(df_test_aud_raw, injection_info_df, mouse, pd):
    df_test_aud = df_test_aud_raw.copy()
    df_test_aud["observations"] = pd.NA

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
        df_test_aud["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    trial_mice = (
        df_test_aud["subject"].astype(str)
        if "subject" in df_test_aud.columns
        else pd.Series(mouse, index=df_test_aud.index)
    )
    df_test_aud["observations"] = [
        injection_lookup.get((trial_date, trial_mouse), pd.NA)
        for trial_date, trial_mouse in zip(trial_dates, trial_mice)
    ]

    df_test_aud['observations']
    return (df_test_aud,)


@app.cell
def _(pd, utils):
    def add_number_of_pokes(df_raw: pd.DataFrame, port_number: int):
        df = df_raw.copy()
        port_hold_column = f"port{port_number}_holds"
        df[port_hold_column] = df.apply(lambda row: utils.get_trial_port_hold(row, port_number), axis=1)
        df[f"port{port_number}_pokes_num"] = df[port_hold_column].apply(lambda x: len(x))
        return df

    return (add_number_of_pokes,)


@app.cell
def _(add_number_of_pokes, df_test_aud, dft, hM3Dq_mice, hM4Di_mice):
    session_performance = df_test_aud.groupby(
        ["subject", "session"]
    )["correct"].transform("mean")

    is_saline = df_test_aud["observations"].str.contains(
        "saline",
        case=False,
        na=False,
    )

    drop_mask = is_saline & (session_performance < 0.7)

    df_test_aud_upd = df_test_aud[
        ~drop_mask
    ].copy()

    df_test_aud_upd = dft.calculate_time_between_trials_and_reaction_time(dft.add_day_column_to_df(
        add_number_of_pokes(df_test_aud_upd, port_number=2)
    ))

    # df_test_aud_upd = df_test_aud_upd[
    #     df_test_aud_upd['year_month_day'].isin(
    #         ['2026-05-22', '2026-05-20']
    #     )
    # ]

    df_test_aud_hm4 = df_test_aud_upd[
        df_test_aud_upd["subject"].isin(hM4Di_mice)
    ].copy()

    df_test_aud_hm3 = df_test_aud_upd[
        df_test_aud_upd["subject"].isin(hM3Dq_mice)
    ].copy()
    return df_test_aud_hm3, df_test_aud_hm4


@app.cell
def _(pd, plt):
    from scipy import stats

    def observation_to_color(observation):
        observation = str(observation).lower()
        if "dcz" in observation:
            return "red"
        if "saline" in observation:
            return "blue"
        return "gray"

    def observation_group(observation):
        observation = str(observation).lower()
        if "dcz" in observation:
            return "DCZ"
        if "saline" in observation:
            return "saline"
        return None

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

        observation_summary["_date_for_pairing"] = pd.to_datetime(
            observation_summary[date_col],
            errors="coerce",
        )
        saline_summary = observation_summary[
            observation_summary["observation_group"] == "saline"
        ].copy()
        dcz_summary = observation_summary[
            observation_summary["observation_group"] == "DCZ"
        ].copy()

        paired_rows = []
        for _, dcz_row in dcz_summary.iterrows():
            previous_saline = saline_summary[
                (saline_summary[subject_col] == dcz_row[subject_col])
                & (
                    saline_summary["_date_for_pairing"]
                    < dcz_row["_date_for_pairing"]
                )
            ].sort_values("_date_for_pairing")
            if previous_saline.empty:
                continue

            saline_row = previous_saline.iloc[-1]
            paired_rows.append(
                {
                    subject_col: dcz_row[subject_col],
                    "saline_date": saline_row[date_col],
                    "DCZ_date": dcz_row[date_col],
                    "days_between": (
                        dcz_row["_date_for_pairing"]
                        - saline_row["_date_for_pairing"]
                    ).days,
                    "saline": saline_row[metric_col],
                    "DCZ": dcz_row[metric_col],
                }
            )

        paired_summary = pd.DataFrame(paired_rows)
        if not paired_summary.empty:
            paired_summary["saline"] = pd.to_numeric(
                paired_summary["saline"],
                errors="coerce",
            )
            paired_summary["DCZ"] = pd.to_numeric(
                paired_summary["DCZ"],
                errors="coerce",
            )
            paired_summary = paired_summary.dropna(subset=["saline", "DCZ"])
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

        return plot_func(
            df_group,
            metric_col=metric_col,
            agg_func=agg_func,
            ylabel=f"{agg_func} {metric_name}",
            title=f"{metric_name} in {group_name} group",
        )[0]

    return (plot_group,)


@app.cell
def _(df_test_aud_hm3, df_test_aud_hm4, mo, plot_group, plot_metric_by_date):
    mo.vstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="correct", metric_name="performance", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="correct", metric_name="performance", agg_func="mean", plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_aud_hm3,
    df_test_aud_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="correct", metric_name="performance", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="correct", metric_name="performance", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell
def _(df_test_aud_hm3, df_test_aud_hm4, mo, plot_group, plot_metric_by_date):
    mo.vstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_aud_hm3,
    df_test_aud_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="port2_pokes_num", metric_name="port2_pokes_num", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell
def _(df_test_aud_hm3, df_test_aud_hm4, mo, plot_group, plot_metric_by_date):
    mo.vstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_date)
        ]
    )
    return


@app.cell
def _(
    df_test_aud_hm3,
    df_test_aud_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="reaction_time", metric_name="reaction_time", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell
def _(
    df_test_aud_hm3,
    df_test_aud_hm4,
    mo,
    plot_group,
    plot_metric_by_observation,
):
    mo.hstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_observation), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_observation)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # import dlc data
    """)
    return


@app.cell
def _(pd):
    def flatten_dlc_columns(columns):
        flat_columns = []
        for column in columns:
            if isinstance(column, tuple) and len(column) >= 3:
                if column[0] == "scorer" and column[1] == "bodyparts":
                    column_name = "frame"
                else:
                    column_name = f"{column[1]}_{column[2]}"
            else:
                column_name = str(column)

            if column_name in flat_columns:
                column_name = f"{column_name}_{flat_columns.count(column_name)}"
            flat_columns.append(column_name)
        return flat_columns

    def read_dlc_csv(dlc_file):
        dlc_df = pd.read_csv(dlc_file, header=[0, 1, 2])
        dlc_df.columns = flatten_dlc_columns(dlc_df.columns)
        if "frame" in dlc_df.columns:
            dlc_df["frame"] = pd.to_numeric(
                dlc_df["frame"],
                errors="coerce",
            ).astype("Int64")
        return dlc_df


    return


@app.cell
def _(mo):
    download_dlcData_button = mo.ui.run_button(label="Download / update the mice dlc data")
    download_dlcData_button
    return (download_dlcData_button,)


@app.cell
def _():
    dlc_mice_selected = ['NUO005', 'NUO008', 'NUO010', 'NUO012']
    return (dlc_mice_selected,)


@app.cell
def _(mo):
    load_paired_dlc_button = mo.ui.run_button(label="Load paired DLC data")
    load_paired_dlc_button
    return


@app.cell
def _(Path, dlc_mice_selected, download_dlcData_button, pd, project, utils):

    if download_dlcData_button.value:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            _local_path.mkdir(parents=True, exist_ok=True)

            utils.rsync_specific_file(
                file_path="/home/kudongdong/data/behavior_DLC/TrainingVillage_2AFC_superanimal-LeciLab-2025-10-16/v2/{}_*.csv".format(dlc_mouse_selected),
                local_path=str(_local_path),
                credentials=utils.get_idibaps_cluster_credentials(),
            )

    else:
        behav_df_dic = {}
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            for dlc_file_ in sorted(_local_path.glob("*DLC*.csv")):
                random_df = pd.read_csv(dlc_file_, header=[1, 2])
                behav_df_dic[dlc_file_.name[:29]] = random_df

    video_data_path = f'/storage/training_village/{project}/videos/'
    if download_dlcData_button.value:
        for dlc_mouse_selected in dlc_mice_selected:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / dlc_mouse_selected / "DLC"
            for dlc_file_ in sorted(_local_path.glob("*DLC*.csv")):
                utils.rsync_specific_file(
                    file_path=Path(video_data_path, dlc_mouse_selected, dlc_file_.name[:29]+".csv"),
                    local_path=str(_local_path),
                    credentials=utils.get_idibaps_cluster_credentials(),
                )
    else:
        video_df_dic = {}
        for video_file_ in behav_df_dic:
            _local_path = Path(utils.get_outpath()) / project / "sessions" / video_file_[:6] / "DLC"
            video_df = pd.read_csv(Path(_local_path, video_file_+".csv"), sep=';', index_col='frame')
            video_df = video_df[2:]
            video_df.index = range(0, len(video_df))
            video_df_dic[video_file_] = video_df
    return behav_df_dic, video_df_dic


@app.cell
def _(behavior_utils, plot_test):
    import importlib
    importlib.reload(behavior_utils)
    importlib.reload(plot_test)
    return


@app.cell
def _(behav_df_dic, behavior_utils, video_df_dic):
    dlc_df_test_dcz = behav_df_dic["NUO010_TwoAFC_20260523_114007"]
    video_df_test_dcz = video_df_dic["NUO010_TwoAFC_20260523_114007"]
    dlc_df_test_saline = behav_df_dic["NUO010_TwoAFC_20260522_131557"]
    video_df_test_saline = video_df_dic["NUO010_TwoAFC_20260522_131557"]

    dlc_df_test_dcz['timestamp'] = video_df_test_dcz['timestamp']
    dlc_df_test_dcz = behavior_utils.preprocess_positions(dlc_df_test_dcz, likelihood_thr=0.65, distance_thr=200, max_iter=100)
    dlc_df_test_saline['timestamp'] = video_df_test_saline['timestamp']
    dlc_df_test_saline = behavior_utils.preprocess_positions(dlc_df_test_saline, likelihood_thr=0.65, distance_thr=200, max_iter=100)

    dlc_df_test_dcz = behavior_utils.compute_distance_speed(dlc_df_test_dcz, window_size = 5)
    dlc_df_test_saline = behavior_utils.compute_distance_speed(dlc_df_test_saline, window_size = 5)
    return dlc_df_test_dcz, dlc_df_test_saline


@app.cell
def _(behavior_utils, colors, dlc_df_test_dcz, dlc_df_test_saline, np, plt):
    occupancy_map_dcz = behavior_utils.occupancy_map(
        dlc_df_test_dcz[("Center", "x")],
        dlc_df_test_dcz[("Center", "y")],
        dlc_df_test_dcz[("timestamp", "")],
    )

    occupancy_map_saline = behavior_utils.occupancy_map(
        dlc_df_test_saline[("Center", "x")],
        dlc_df_test_saline[("Center", "y")],
        dlc_df_test_saline[("timestamp", "")],
    )

    occ_extents = (
        np.nanmin([
            dlc_df_test_dcz[("Center", "x")].min(),
            dlc_df_test_saline[("Center", "x")].min(),
        ]),
        np.nanmax([
            dlc_df_test_dcz[("Center", "x")].max(),
            dlc_df_test_saline[("Center", "x")].max(),
        ]),
        np.nanmin([
            dlc_df_test_dcz[("Center", "y")].min(),
            dlc_df_test_saline[("Center", "y")].min(),
        ]),
        np.nanmax([
            dlc_df_test_dcz[("Center", "y")].max(),
            dlc_df_test_saline[("Center", "y")].max(),
        ]),
    )

    occ_all_values = np.concatenate([
        occupancy_map_dcz.ravel(),
        occupancy_map_saline.ravel(),
    ])

    occ_all_values = occ_all_values[np.isfinite(occ_all_values)]

    occ_vmin = np.nanquantile(occ_all_values, 0.01)
    occ_vmax = np.nanquantile(occ_all_values, 0.99)

    occ_norm = colors.Normalize(vmin=occ_vmin, vmax=occ_vmax)

    occ_fig, occ_axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    occ_im0 = occ_axes[0].imshow(
        occupancy_map_saline.T,
        origin="upper",
        extent=occ_extents,
        cmap="viridis",
        norm=occ_norm,
        interpolation="gaussian",
    )
    occ_axes[0].set_title("Saline", fontsize=10)

    occ_im1 = occ_axes[1].imshow(
        occupancy_map_dcz.T,
        origin="upper",
        extent=occ_extents,
        cmap="viridis",
        norm=occ_norm,
        interpolation="gaussian",
    )
    occ_axes[1].set_title("DCZ", fontsize=10)

    for occ_ax in occ_axes:
        occ_ax.set_aspect("equal")
        occ_ax.set_xticks([])
        occ_ax.set_yticks([])

    occ_fig.colorbar(
        occ_im1,
        ax=occ_axes,
        label="occupancy (s/pixels)",
        shrink=0.7,
        fraction=0.04,
        pad=0.02,
        extend="both",
    )

    plt.show()
    return


@app.cell
def _(dlc_df_test_dcz, dlc_df_test_saline, mpl, pd, plot_test, plt):
    def prep_traj_speed_df(dlc_df):
        return (
            dlc_df["Center"][["x", "y", "mean_speed"]]
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(limit_direction="both")
            .dropna()
            .copy()
        )


    dcz_traj = prep_traj_speed_df(dlc_df_test_dcz)
    saline_traj = prep_traj_speed_df(dlc_df_test_saline)

    all_speed = pd.concat(
        [
            dcz_traj["mean_speed"],
            saline_traj["mean_speed"],
        ],
        ignore_index=True,
    ).dropna()

    vmin = all_speed.quantile(0.01)
    vmax = all_speed.quantile(0.99)
    norm_ = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    plot_test.plot_traj_speed(saline_traj, cmap="inferno", ax=axes[0], norm=norm_)
    axes[0].set_title("saline")

    plot_test.plot_traj_speed(dcz_traj, cmap="inferno", ax=axes[1], norm=norm_)
    axes[1].set_title("DCZ")

    for ax in axes:
        ax.set_aspect("equal")
        ax.autoscale()
        # ax.set_xticks([])
        # ax.set_yticks([])
    axes[0].invert_yaxis()

    sm = mpl.cm.ScalarMappable(norm=norm_, cmap="inferno")
    fig.colorbar(sm, ax=axes, label="mean speed (pixels/s)", shrink=0.6)

    plt.show()
    return dcz_traj, saline_traj


@app.cell
def _(saline_traj):
    saline_traj['y'].max()
    return


@app.cell
def _():
    roi_botttom = 350
    roi_top = 500
    roi_left = 250
    roi_right = 400
    return


@app.cell
def _():
    return


@app.cell
def _(dcz_traj, plt):
    plt.plot(dcz_traj["mean_speed"])
    return


@app.cell
def _(plt, saline_traj):
    plt.plot(saline_traj["mean_speed"])
    return


if __name__ == "__main__":
    app.run()
