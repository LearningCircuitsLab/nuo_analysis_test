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


    import autograd.numpy.random as npr
    # npr.seed(0)
    import ssm
    from ssm.util import find_permutation
    from ssm.plots import gradient_cmap, white_to_color_cmap

    from scipy.optimize import minimize

    # %load_ext autoreload
    # %autoreload 2
    return Path, dft, minimize, np, pd, plt, utils, utils_test


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


@app.cell
def _(mo, project, utils):
    animals = utils.get_animals_in_project(project)

    mouse_select = mo.ui.dropdown(
        options=animals,
        value="NUO001",
        label="Mouse"
    )

    mouse_select
    return (mouse_select,)


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="Download / update this mouse")
    download_button
    return (download_button,)


@app.cell
def _(Path, download_button, mouse_select, pd, project, utils):
    mouse = mouse_select.value

    local_path = Path(utils.get_outpath()) / project / "sessions" / mouse
    local_path.mkdir(parents=True, exist_ok=True)

    csv_path = local_path / f"{mouse}.csv"

    if download_button.value or not csv_path.exists():
        utils.rsync_cluster_data(
            project_name=project,
            file_path=f"sessions/{mouse}/{mouse}.csv",
            local_path=str(local_path),
            credentials=utils.get_idibaps_cluster_credentials(),
        )

    df = pd.read_csv(csv_path, sep=";")
    df.head()
    return (df,)


@app.cell
def _(df, dft):
    df_test_vis_easy = df[df['current_training_stage'] == 'TwoAFC_visual_easy']
    df_test_vis_easy = dft.parameters_for_fit(df_test_vis_easy)
    df_test_aud_easy = df[df['current_training_stage'] == 'TwoAFC_auditory_easy']
    df_test_aud_easy = dft.parameters_for_fit(df_test_aud_easy)

    df_test_vis_hard = df[df['current_training_stage'] == 'TwoAFC_visual_hard']
    df_test_vis_hard = dft.parameters_for_fit(df_test_vis_hard)
    df_test_aud_hard = df[df['current_training_stage'] == 'TwoAFC_auditory_hard']
    df_test_aud_hard = dft.parameters_for_fit(df_test_aud_hard)
    return (
        df_test_aud_easy,
        df_test_aud_hard,
        df_test_vis_easy,
        df_test_vis_hard,
    )


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

    df_selector = mo.ui.dropdown(
        options=list(df_options.keys()),
        value="aud_easy"
    )
    df_selector
    return df_options, df_selector


@app.cell
def _(df_options, df_selector):
    selected = df_options[df_selector.value]

    df_test = selected["df"]
    stim_col = selected["stim_col"]
    modality = selected["modality"]
    return df_test, stim_col


@app.function
def add_lag_features(df, source_col, n_lags=10, group_col="session"):
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"{source_col}_lag{lag}"] = (
            df.groupby(group_col)[source_col].shift(lag)
        )
    return df


@app.cell
def _(df_test, stim_col, utils):
    n_lags = 5

    source_col = "first_choice_numeric"

    df_lag = add_lag_features(
        df_test,
        source_col=source_col,
        n_lags=n_lags,
        group_col="session"
    )

    lag_cols = [f"first_choice_numeric_lag{i}" for i in range(1, n_lags + 1)]

    fit_df = df_lag.dropna(subset=lag_cols + ["first_choice_numeric"]).copy()

    _, logit_model_lags = utils.logi_model_fit(fit_df, 
                            X=[stim_col] + [i for i in lag_cols],
                            y='first_choice_numeric',)
    logit_model_lags_params = logit_model_lags.params
    return (logit_model_lags_params,)


@app.cell
def _(logit_model_lags_params):
    logit_model_lags_params
    return


@app.cell
def _(np):
    v0 = np.array([0.0, 1.0, 1.0, 0.0, -2.0])
    return (v0,)


@app.cell
def _(df_test, minimize, utils_test, v0):
    df_copy = df_test.copy()
    df_clean = df_copy.dropna(subset=['first_choice_numeric', 'correct_side_numeric'])
    res_stim = minimize(utils_test.neg_loglik_reg, v0, args=(df_clean, "stim"), method="L-BFGS-B")
    res_act = minimize(utils_test.neg_loglik_reg, v0, args=(df_clean, "act"), method="L-BFGS-B")

    theta_stim = utils_test.unpack_params(res_stim.x)
    theta_act = utils_test.unpack_params(res_act.x)
    return theta_act, theta_stim


@app.cell
def _(theta_act, theta_stim):
    print(theta_act)
    print(theta_stim)
    return


@app.cell
def _(df_test, pd, theta_act, utils_test):
    df_test_kernel = pd.DataFrame([])
    for session in df_test['session'].unique():
        df_test_session = df_test[df_test['session'] == session]
        df_test_session['action_trace'] = utils_test.recursive_kernel_prior(df_test_session['first_choice_numeric'].values, theta_act['alpha'])
        df_test_session['stimulus_trace'] = utils_test.recursive_kernel_prior(df_test_session['correct_side_numeric'].values, theta_act['alpha'])
        df_test_kernel = pd.concat([df_test_kernel, df_test_session])
    return (df_test_kernel,)


@app.cell
def _(df_test_kernel, plt, stim_col, utils_test):
    base_inputs = [stim_col, "bias", "action_trace", "stimulus_trace"]
    datas, inpts = utils_test.build_glmhmm_inputs_by_session(df_test_kernel, y_col="first_choice_numeric", stim_col=base_inputs)
    map_glmhmm, ll, hmm_lls = utils_test.fit_glmhmm(datas, inpts, num_states=3, obs_dim=1, input_dim=4, num_categories=2, N_iters=150)
    plt.plot(hmm_lls, label = 'all')
    # drop-one-out models
    for col in base_inputs:
        reduced_inputs = [c for c in base_inputs if c != col]
        datas, inpts = utils_test.build_glmhmm_inputs_by_session(df_test_kernel, y_col="first_choice_numeric", stim_col=reduced_inputs)
        map_glmhmm, ll, hmm_lls = utils_test.fit_glmhmm(datas, inpts, num_states=3, obs_dim=1, input_dim=3, num_categories=2, N_iters=150)
        plt.plot(hmm_lls, label = 'drop' + col)


    return


if __name__ == "__main__":
    app.run()
