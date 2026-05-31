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
    import matplotlib.patches as patches
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
        dft,
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
        value=animals[:-1],
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
    # hM3Dq_mice = ['NUO062', 'NUO063', 'NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    # hM4Di_mice = ['NUO057', 'NUO058', 'NUO059', 'NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
    hM3Dq_mice = ['NUO064', 'NUO065', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    hM4Di_mice = ['NUO060', 'NUO061', 'NUO001', 'NUO002', 'NUO003', 'NUO005', 'NUO006']
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
def _(behavior_utils, injection_info_df, pd, plt):
    from scipy import stats

    metric_paired_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df
    )

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

    return plot_metric_by_date, plot_metric_by_observation, stats


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Psychometric curve
    """)
    return


@app.cell
def _(
    behavior_utils,
    df_test_aud_hm3,
    df_test_aud_hm4,
    hM3Dq_mice,
    hM4Di_mice,
    injection_info_df,
    mo,
    plot_test,
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
            df_test_aud_hm4,
            "hM4Di",
            paired_dates=psychometric_hm4_paired_dates,
            x_col="total_evidence_strength",
            y_col="first_choice_numeric",
            valueType="continue",
            bins=6,
            min_trials=20,
        )
    )

    psychometric_hm3_fig, psychometric_hm3_summary = (
        plot_test.plot_condition_psychometric_curves(
            df_test_aud_hm3,
            "hM3Dq",
            paired_dates=psychometric_hm3_paired_dates,
            x_col="total_evidence_strength",
            y_col="first_choice_numeric",
            valueType="continue",
            bins=6,
            min_trials=20,
        )
    )

    mo.hstack([psychometric_hm4_fig, psychometric_hm3_fig])
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
def _(df_test_aud_hm3, df_test_aud_hm4, mo, plot_group, plot_metric_by_date):
    mo.vstack(
        [
            plot_group(df_test_aud_hm4, "hM4Di", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_date), 
            plot_group(df_test_aud_hm3, "hM3Dq", metric_col="time_between_trials", metric_name="time_between_trials", agg_func="mean", plot_func=plot_metric_by_date)
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
def _(
    Path,
    behavior_utils,
    dlc_mice_selected,
    download_dlcData_button,
    injection_info_df,
    pd,
    project,
    utils,
):
    behav_df_dic = {}
    video_df_dic = {}

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

    paired_dlc_dates = behavior_utils.get_paired_injection_dates(
        injection_info_df,
        mice_selected=dlc_mice_selected,
    )
    (
        behav_df_dic_saline,
        behav_df_dic_dcz,
        behav_pair_map,
        missing_behav_pairs,
    ) = behavior_utils.split_paired_behavior_dicts(behav_df_dic, paired_dlc_dates)
    (
        video_df_dic_saline,
        video_df_dic_dcz,
        video_pair_map,
        missing_video_pairs,
    ) = behavior_utils.split_paired_behavior_dicts(video_df_dic, paired_dlc_dates)

    behav_pair_map
    return (
        behav_df_dic_dcz,
        behav_df_dic_saline,
        behav_pair_map,
        video_df_dic_dcz,
        video_df_dic_saline,
    )


@app.cell
def _(behavior_utils, plot_test):
    import importlib
    importlib.reload(behavior_utils)
    importlib.reload(plot_test)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_utils,
    np,
    video_df_dic_dcz,
    video_df_dic_saline,
):
    def add_timestamp_from_end(behav_df, video_df):
        behav_df[("timestamp", "")] = np.nan

        n = min(len(behav_df), len(video_df))

        behav_df.iloc[
            -n:,
            behav_df.columns.get_loc(("timestamp", "")),
        ] = video_df["timestamp"].to_numpy()[-n:]

        return behav_df


    for pair_id in behav_pair_map["pair_id"].unique():
        behav_df_dic_dcz[pair_id] = add_timestamp_from_end(
            behav_df_dic_dcz[pair_id],
            video_df_dic_dcz[pair_id],
        )
        behav_df_dic_dcz[pair_id].dropna(subset=[("timestamp", "")], inplace=True)
        behav_df_dic_dcz[pair_id] = behavior_utils.preprocess_positions(
            behav_df_dic_dcz[pair_id],
            likelihood_thr=0.7,
            distance_thr=200,
            max_iter=100,
            speed_thr=600
        )

        behav_df_dic_saline[pair_id] = add_timestamp_from_end(
            behav_df_dic_saline[pair_id],
            video_df_dic_saline[pair_id],
        )
        behav_df_dic_saline[pair_id].dropna(subset=[("timestamp", "")], inplace=True)
        behav_df_dic_saline[pair_id] = behavior_utils.preprocess_positions(
            behav_df_dic_saline[pair_id],
            likelihood_thr=0.7,
            distance_thr=200,
            max_iter=100,
            speed_thr=600
        )

        behav_df_dic_dcz[pair_id] = behavior_utils.compute_distance_speed(
            behav_df_dic_dcz[pair_id],
            window_size=5,
        )
        behav_df_dic_saline[pair_id] = behavior_utils.compute_distance_speed(
            behav_df_dic_saline[pair_id],
            window_size=5,
        )
    return


@app.cell
def _():
    roi_bottom = 80
    roi_top = 190
    roi_left = 235
    roi_right = 400
    return roi_bottom, roi_left, roi_right, roi_top


@app.cell
def _(behav_pair_map):
    behav_pair_map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # save tmp svg
    """)
    return


@app.cell
def _(Path):
    save_behavior_svg = True
    behavior_svg_dir = Path("/mnt/e/data/LeciLab/behavioral_data/tmp")
    return behavior_svg_dir, save_behavior_svg


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    save_behavior_svg,
):
    pair_figures = plot_test.plot_paired_behavior_figures(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    plot_test.save_figures_svg(
        pair_figures,
        "paired_behavior",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(pair_figures)
    return


@app.cell
def _(
    behav_df_dic_saline,
    behavior_utils,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
):
    behavior_utils.get_roi_time_ratio(
        behav_df_dic_saline['NUO005_pair_01'],
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_utils,
    roi_bottom,
    roi_left,
    roi_right,
    roi_top,
    video_df_dic_dcz,
    video_df_dic_saline,
):
    roi_time_ratio_summary = behavior_utils.paired_roi_time_ratio_comparison(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        video_df_dic_saline=video_df_dic_saline,
        video_df_dic_dcz=video_df_dic_dcz,
        pair_map=behav_pair_map,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
        bodypart="Center",
    )

    roi_time_ratio_summary
    return (roi_time_ratio_summary,)


@app.cell
def _(
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    plot_test,
    plt,
    roi_time_ratio_summary,
    save_behavior_svg,
    stats,
):
    roi_group_df = roi_time_ratio_summary.copy()

    if "subject" not in roi_group_df.columns:
        roi_group_df["subject"] = roi_group_df["pair_id"].str[:6]


    def plot_roi_group(ax, df, mice, group_name):
        group_df = df[
            df["subject"].isin(mice)
        ].dropna(subset=["saline", "DCZ"]).copy()

        for _, roi_row in group_df.iterrows():
            ax.plot(
                [0, 1],
                [roi_row["saline"], roi_row["DCZ"]],
                color="gray",
                alpha=0.4,
            )

        ax.scatter(
            [0] * len(group_df),
            group_df["saline"],
            color="blue",
            edgecolor="black",
            label="saline",
        )

        ax.scatter(
            [1] * len(group_df),
            group_df["DCZ"],
            color="red",
            edgecolor="black",
            label="DCZ",
        )

        if len(group_df) > 0:
            try:
                roi_p = stats.wilcoxon(
                    group_df["saline"],
                    group_df["DCZ"],
                ).pvalue
                p_text = f"p={roi_p:.3g}"
            except ValueError:
                roi_p = float("nan")
                p_text = "p=NA"
        else:
            roi_p = float("nan")
            p_text = "p=NA"

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["saline", "DCZ"])
        ax.set_ylabel("fraction of time in ROI")
        ax.set_title(f"{group_name}, n={len(group_df)}, {p_text}")
        plot_test.add_paired_significance_label(
            ax,
            group_df["saline"],
            group_df["DCZ"],
            plot_test.p_to_star(roi_p),
        )
        ax.grid(axis="y", alpha=0.3)

        return group_df


    roi_group_fig, roi_group_axes = plt.subplots(
        1,
        2,
        figsize=(8, 5),
        sharey=True,
    )

    roi_hm4_df = plot_roi_group(
        roi_group_axes[0],
        roi_group_df,
        hM4Di_mice,
        "hM4Di",
    )

    roi_hm3_df = plot_roi_group(
        roi_group_axes[1],
        roi_group_df,
        hM3Dq_mice,
        "hM3Dq",
    )

    roi_group_fig.tight_layout()
    plot_test.save_figure_svg(
        roi_group_fig,
        "roi_time_ratio_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    plt.show()
    return


@app.cell
def _(behav_df_dic_dcz, behav_df_dic_saline, behav_pair_map, behavior_utils):
    stationary_speed_threshold = 10  # pixels/s调

    stationary_time_ratio_summary = behavior_utils.paired_stationary_time_ratio_comparison(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        speed_threshold=stationary_speed_threshold,
        pair_map=behav_pair_map,
        bodypart="Center",
        speed_col="mean_speed",
    )

    stationary_time_ratio_summary
    return stationary_speed_threshold, stationary_time_ratio_summary


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
    stationary_speed_threshold,
):
    stationary_speed_figures = plot_test.plot_paired_stationary_speed_traces(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        speed_threshold=stationary_speed_threshold,
        bodypart="Center",
        speed_col="mean_speed",
    )

    plot_test.save_figures_svg(
        stationary_speed_figures,
        "stationary_speed_trace",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    stationary_speed_figures
    return


@app.cell
def _(
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    plot_test,
    save_behavior_svg,
    stationary_time_ratio_summary,
):
    stationary_group_fig, stationary_hm4_df, stationary_hm3_df = (
        plot_test.plot_stationary_time_ratio_groups(
            stationary_time_ratio_summary,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
        )
    )
    plot_test.save_figure_svg(
        stationary_group_fig,
        "stationary_time_ratio_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    stationary_group_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
    stationary_speed_threshold,
):
    speed_distribution_figures = plot_test.plot_paired_speed_distributions(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        bodypart="Center",
        speed_col="mean_speed",
        speed_threshold=stationary_speed_threshold,  
        bins=60,
        kde=True,
    )

    plot_test.save_figures_svg(
        speed_distribution_figures,
        "speed_distribution",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(speed_distribution_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    hM3Dq_mice,
    hM4Di_mice,
    mo,
    plot_test,
    save_behavior_svg,
):
    speed_acf_group_figures, hm4_acf_pair_ids, hm3_acf_pair_ids = (
        plot_test.plot_speed_acfs_by_group(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
            fps=30,
            max_lag_sec=60,
            bodypart="Center",
            speed_col="mean_speed",
        )
    )

    plot_test.save_figures_svg(
        speed_acf_group_figures,
        "speed_acf_group",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(speed_acf_group_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    behavior_utils,
    hM3Dq_mice,
    hM4Di_mice,
    plot_test,
    save_behavior_svg,
):
    speed_acf_auc_summary = behavior_utils.paired_speed_acf_auc_comparison(
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        fps=30,
        max_lag_sec=60,
        pair_map=behav_pair_map,
        bodypart="Center",
        speed_col="mean_speed",
    )

    speed_acf_auc_fig, speed_acf_auc_hm4_df, speed_acf_auc_hm3_df = (
        plot_test.plot_speed_acf_auc_groups(
            speed_acf_auc_summary,
            hM4Di_mice=hM4Di_mice,
            hM3Dq_mice=hM3Dq_mice,
        )
    )

    plot_test.save_figure_svg(
        speed_acf_auc_fig,
        "speed_acf_auc_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    speed_acf_auc_fig
    return


@app.cell
def _():
    noeffec_group = ['NUO005', 'NUO008']
    effec_group = ['NUO010', 'NUO012']
    return


@app.cell
def _(behav_df_dic_dcz, behav_df_dic_saline, pd):
    behav_df_effec = pd.DataFrame([])
    behav_df_noeffec = pd.DataFrame([])
    for behav_df_name_ in behav_df_dic_dcz:
        behav_df_subject = behav_df_dic_dcz[behav_df_name_]
        behav_df_subject[("session", "")] = behav_df_name_
        if ("NUO005" in behav_df_name_) or ("NUO008" in behav_df_name_):
            behav_df_noeffec = pd.concat([behav_df_noeffec, behav_df_subject])
        else:
            behav_df_effec = pd.concat([behav_df_effec, behav_df_subject])
    for behav_df_name_ in behav_df_dic_saline:
        behav_df_subject = behav_df_dic_saline[behav_df_name_]
        behav_df_subject[("session", "")] = behav_df_name_
        behav_df_noeffec = pd.concat([behav_df_noeffec, behav_df_subject])
    return


@app.cell
def _(behav_df_dic_dcz, behav_df_dic_saline, np, pd):
    import ssm

    def iter_speed_hmm_entries(model_name):
        if model_name == "effect":
            for pair_id, behav_df in behav_df_dic_dcz.items():
                if ("NUO005" in pair_id) or ("NUO008" in pair_id):
                    continue
                yield pair_id, "DCZ", behav_df
        else:
            for pair_id, behav_df in behav_df_dic_dcz.items():
                if ("NUO005" in pair_id) or ("NUO008" in pair_id):
                    yield pair_id, "DCZ", behav_df
            for pair_id, behav_df in behav_df_dic_saline.items():
                yield pair_id, "saline", behav_df

    def build_speed_hmm_datas_from_entries(
        entries,
        speed_col=("Center", "mean_speed"),
        min_len=20,
    ):
        datas = []
        data_info = []

        for pair_id, condition, behav_df in entries:
            if behav_df.empty or speed_col not in behav_df.columns:
                continue

            speed = pd.to_numeric(behav_df[speed_col], errors="coerce")
            valid_speed = speed.replace([np.inf, -np.inf], np.nan).notna()
            valid_positions = np.flatnonzero(valid_speed.to_numpy())
            if len(valid_positions) < min_len:
                continue

            data = speed.iloc[valid_positions].to_numpy(dtype=float).reshape(-1, 1)
            datas.append(data)
            data_info.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "df": behav_df,
                    "valid_positions": valid_positions,
                }
            )

        return datas, data_info

    def state_mean_speeds(hmm, datas):
        state_rows = []
        for state in range(hmm.K):
            weighted_sum = 0.0
            weighted_count = 0.0
            for data in datas:
                posterior = hmm.expected_states(data)[0]
                state_weight = posterior[:, state]
                weighted_sum += np.sum(state_weight * data[:, 0])
                weighted_count += np.sum(state_weight)

            state_rows.append(
                {
                    "state": state,
                    "mean_speed": weighted_sum / weighted_count
                    if weighted_count > 0
                    else np.nan,
                    "state_weight": weighted_count,
                }
            )

        return pd.DataFrame(state_rows)

    def fit_speed_hmm(datas, num_states=2, n_iters=100):
        hmm = ssm.HMM(num_states, 1, observations="gaussian")
        hmm_lls = hmm.fit(
            datas,
            method="em",
            num_iters=n_iters,
            tolerance=1e-4,
        )

        state_summary_before = state_mean_speeds(hmm, datas)
        perm = [
            int(state)
            for state in state_summary_before.sort_values("mean_speed")[
                "state"
            ].to_list()
        ]
        hmm.permute(perm)

        state_summary_after = state_mean_speeds(hmm, datas)
        log_likelihood = sum(float(hmm.log_probability(data)) for data in datas)

        return hmm, log_likelihood, hmm_lls, perm, state_summary_after

    def add_hmm_states_to_dfs(hmm, datas, data_info, model_name):
        state_rows = []
        for data, info in zip(datas, data_info):
            posterior = hmm.expected_states(data)[0]
            states = np.argmax(posterior, axis=1)
            behav_df = info["df"]

            behav_df[("speed_hmm_state", "")] = np.nan
            behav_df[("speed_hmm_model", "")] = None
            state_col_loc = behav_df.columns.get_loc(("speed_hmm_state", ""))
            model_col_loc = behav_df.columns.get_loc(("speed_hmm_model", ""))

            behav_df.iloc[info["valid_positions"], state_col_loc] = states
            behav_df.iloc[info["valid_positions"], model_col_loc] = model_name

            state_rows.append(
                {
                    "model": model_name,
                    "pair_id": info["pair_id"],
                    "condition": info["condition"],
                    "n_valid_frames": len(states),
                    "state0_frames": int(np.sum(states == 0)),
                    "state1_frames": int(np.sum(states == 1)),
                    "state1_fraction": float(np.mean(states == 1)),
                }
            )

        return pd.DataFrame(state_rows)

    hmm_speed_models = {}
    hmm_speed_rows = []
    hmm_speed_state_summaries = []

    for model_name in ["effect", "noeffect"]:
        entries = list(iter_speed_hmm_entries(model_name))
        datas, data_info = build_speed_hmm_datas_from_entries(entries)
        if not datas:
            hmm_speed_rows.append(
                {
                    "model": model_name,
                    "n_sequences": 0,
                    "n_frames": 0,
                    "log_likelihood": np.nan,
                    "state0_mean_speed": np.nan,
                    "state1_mean_speed": np.nan,
                    "state_permutation": None,
                    "status": "no valid speed data",
                }
            )
            continue

        hmm, ll, hmm_lls, perm, state_summary = fit_speed_hmm(datas)
        state_summary_by_df = add_hmm_states_to_dfs(
            hmm,
            datas,
            data_info,
            model_name,
        )
        hmm_speed_state_summaries.append(state_summary_by_df)
        hmm_speed_models[model_name] = {
            "model": hmm,
            "datas": datas,
            "data_info": data_info,
            "log_likelihood": ll,
            "hmm_lls": hmm_lls,
            "state_permutation": perm,
            "state_summary": state_summary,
            "state_summary_by_df": state_summary_by_df,
        }

        hmm_speed_rows.append(
            {
                "model": model_name,
                "n_sequences": len(datas),
                "n_frames": sum(len(data) for data in datas),
                "log_likelihood": ll,
                "state0_mean_speed": state_summary.loc[
                    state_summary["state"] == 0,
                    "mean_speed",
                ].iloc[0],
                "state1_mean_speed": state_summary.loc[
                    state_summary["state"] == 1,
                    "mean_speed",
                ].iloc[0],
                "state_permutation": perm,
                "status": "ok",
            }
        )

    hmm_effec = hmm_speed_models.get("effect", {}).get("model")
    hmm_noeffec = hmm_speed_models.get("noeffect", {}).get("model")
    hmm_speed_summary = pd.DataFrame(hmm_speed_rows)
    hmm_speed_state_summary = (
        pd.concat(hmm_speed_state_summaries, ignore_index=True)
        if hmm_speed_state_summaries
        else pd.DataFrame()
    )
    hmm_speed_summary
    return hmm_effec, hmm_noeffec


@app.cell
def _(
    behavior_svg_dir,
    hmm_effec,
    hmm_noeffec,
    plot_test,
    plt,
    save_behavior_svg,
):
    import utils_test

    hmm_transition_fig, hmm_transition_axes = plt.subplots(
        1,
        2,
        figsize=(7, 3),
    )

    if hmm_effec is not None:
        utils_test.plot_transition_matrix(
            hmm_effec,
            title="effect speed HMM",
            ax=hmm_transition_axes[0],
            cmap="gray",
        )
    else:
        hmm_transition_axes[0].set_title("effect speed HMM missing")
        hmm_transition_axes[0].axis("off")

    if hmm_noeffec is not None:
        utils_test.plot_transition_matrix(
            hmm_noeffec,
            title="no-effect speed HMM",
            ax=hmm_transition_axes[1],
            cmap="gray",
        )
    else:
        hmm_transition_axes[1].set_title("no-effect speed HMM missing")
        hmm_transition_axes[1].axis("off")

    hmm_transition_fig.tight_layout()
    plot_test.save_figure_svg(
        hmm_transition_fig,
        "speed_hmm_model_transition_matrix",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )
    hmm_transition_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behavior_svg_dir,
    hmm_effec,
    hmm_noeffec,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_speed_posterior_figures = (
        plot_test.plot_speed_hmm_posteriors_for_behavior_dicts(
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            hmm_effec=hmm_effec,
            hmm_noeffec=hmm_noeffec,
            speed_col=("Center", "mean_speed"),
            speed_quantiles=(0.01, 0.99),
        )
    )

    plot_test.save_figures_svg(
        hmm_speed_posterior_figures,
        "speed_hmm_posterior",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(hmm_speed_posterior_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
    stationary_speed_threshold,
):
    state_speed_distribution_figures = plot_test.plot_paired_state_speed_distributions(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        bodypart="Center",
        speed_col="mean_speed",
        state_col=("speed_hmm_state", ""),
        states=(0, 1),
        speed_threshold=stationary_speed_threshold,
        bins=60,
        kde=True,
    )

    plot_test.save_figures_svg(
        state_speed_distribution_figures,
        "state_speed_distribution",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(state_speed_distribution_figures)
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
):
    (
        speed_hmm_state_fraction_fig,
        speed_hmm_state_fraction_summary,
        noeffec_group_state_fraction_df,
        effec_group_state_fraction_df,
    ) = plot_test.plot_speed_hmm_state_fraction_comparison(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        state_col=("speed_hmm_state", ""),
        K=2,
        states=(0, 1),
        figsize=(8, 7),
    )

    plot_test.save_figure_svg(
        speed_hmm_state_fraction_fig,
        "speed_hmm_state_fraction_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_state_fraction_groups",
    )

    speed_hmm_state_fraction_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_speed_transition_fig_dic = (
        plot_test.plot_paired_speed_hmm_transition_matrices(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            state_col=("speed_hmm_state", ""),
            K=2,
        )
    )

    plot_test.save_figures_svg(
        hmm_speed_transition_fig_dic,
        "speed_hmm_pair_transition_matrix",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(hmm_speed_transition_fig_dic.values()))
    return (hmm_speed_transition_fig_dic,)


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    hmm_speed_transition_fig_dic,
    plot_test,
    save_behavior_svg,
):
    (
        speed_hmm_switch_probability_fig,
        speed_hmm_switch_probability_summary,
        noeffec_group_switch_df,
        effec_group_switch_df,
    ) = plot_test.plot_speed_hmm_switch_probability_comparison(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        transition_fig_dic=hmm_speed_transition_fig_dic,
        K=2,
        figsize=(8, 7),
    )

    plot_test.save_figure_svg(
        speed_hmm_switch_probability_fig,
        "speed_hmm_switch_probability_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_switch_probability_groups",
    )

    speed_hmm_switch_probability_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
):
    speed_hmm_switch_rate_summary_df = plot_test.speed_hmm_switch_rate_summary(
        pair_ids=behav_pair_map["pair_id"].unique(),
        behav_df_dic_saline=behav_df_dic_saline,
        behav_df_dic_dcz=behav_df_dic_dcz,
        fps=30,
        state_col=("speed_hmm_state", ""),
        K=2,
    )

    (
        speed_hmm_switch_rate_distribution_fig,
        _speed_hmm_switch_rate_distribution_df,
    ) = plot_test.plot_speed_hmm_switch_rate_distribution(
        speed_hmm_switch_rate_summary_df,
        states=(0, 1),
    )

    plot_test.save_figure_svg(
        speed_hmm_switch_rate_distribution_fig,
        "speed_hmm_switch_rate_by_session",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_switch_rate_by_session",
    )

    speed_hmm_switch_rate_distribution_fig
    return (speed_hmm_switch_rate_summary_df,)


@app.cell
def _(
    behavior_svg_dir,
    plot_test,
    save_behavior_svg,
    speed_hmm_switch_rate_summary_df,
):
    (
        speed_hmm_state_switch_rate_fig,
        _speed_hmm_state_switch_rate_plot_df,
    ) = plot_test.plot_speed_hmm_state_switch_rate_groups(
        speed_hmm_switch_rate_summary_df,
        states=(0, 1),
    )

    plot_test.save_figure_svg(
        speed_hmm_state_switch_rate_fig,
        "speed_hmm_state_switch_rate_groups",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
        subfolder="speed_hmm_state_switch_rate",
    )
    speed_hmm_state_switch_rate_fig
    return


@app.cell
def _(
    behav_df_dic_dcz,
    behav_df_dic_saline,
    behav_pair_map,
    behavior_svg_dir,
    mo,
    plot_test,
    save_behavior_svg,
):
    hmm_state_xy_heatmap_fig_dic = (
        plot_test.plot_paired_speed_hmm_state_xy_heatmaps(
            pair_ids=behav_pair_map["pair_id"].unique(),
            behav_df_dic_saline=behav_df_dic_saline,
            behav_df_dic_dcz=behav_df_dic_dcz,
            bodypart="Center",
            state_col=("speed_hmm_state", ""),
            states=(0, 1),
            xbins=40,
            ybins=40,
            extent=(0, 640, 0, 480),
            normalize=True,
        )
    )

    plot_test.save_figures_svg(
        hmm_state_xy_heatmap_fig_dic,
        "speed_hmm_state_xy_heatmap",
        save_dir=behavior_svg_dir,
        enabled=save_behavior_svg,
    )

    mo.vstack(list(hmm_state_xy_heatmap_fig_dic.values()))
    return


@app.cell
def _(behav_df_dic_dcz):
    behav_df_dic_dcz['NUO005_pair_01'][("speed_hmm_state", "")]
    return


if __name__ == "__main__":
    app.run()
