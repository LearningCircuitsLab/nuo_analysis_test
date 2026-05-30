import marimo

__generated_with = "0.23.4"
app = marimo.App()


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
    return (
        LinearSegmentedColormap,
        LogisticRegression,
        Path,
        dft,
        gradient_cmap,
        minimize,
        np,
        pd,
        plots,
        plt,
        sm,
        sns,
        ssm,
        utils,
    )


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Import
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
    ## single mouse data import
    """)
    return


@app.cell
def _(Path, pd, project, utils):
    # see the available animals
    animals = utils.get_animals_in_project(project)
    print(animals)
    # download the data for a specific animal
    mouse = "NUO003"
    local_path = Path(utils.get_outpath()) / Path(project) / Path("sessions") / Path(mouse)
    # create the directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    # download the session data
    utils.rsync_cluster_data(
        project_name=project,
        file_path="sessions/{}/{}.csv".format(mouse, mouse),
        local_path=str(local_path),
        credentials=utils.get_idibaps_cluster_credentials(),
    )
    # load the data
    df = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")
    return (df,)


@app.cell
def _(df, dft, pd):
    df_test_multi = df[df['current_training_stage'] == 'TwoAFC_multisensory_easy']
    df_test_multi_vis = df_test_multi[df_test_multi['stimulus_modality'] == 'visual']
    df_test_multi_vis = dft.parameters_for_fit(df_test_multi_vis)
    df_test_multi_aud = df_test_multi[df_test_multi['stimulus_modality'] == 'auditory']
    df_test_multi_aud = dft.parameters_for_fit(df_test_multi_aud)
    df_test_multi = pd.concat([df_test_multi_vis, df_test_multi_aud]).sort_index()
    return df_test_multi, df_test_multi_vis


@app.cell
def _(df, dft):
    df_test_vis_hard = df[df['current_training_stage'] == 'TwoAFC_visual_hard']
    df_test_vis_hard = dft.parameters_for_fit(df_test_vis_hard)
    df_test_aud_hard = df[df['current_training_stage'] == 'TwoAFC_auditory_hard']
    df_test_aud_hard = dft.parameters_for_fit(df_test_aud_hard)
    return (df_test_aud_hard,)


@app.cell
def _(df, dft):
    df_test_vis_easy = df[df['current_training_stage'] == 'TwoAFC_visual_easy']
    df_test_vis_easy = dft.parameters_for_fit(df_test_vis_easy)
    df_test_aud_easy = df[df['current_training_stage'] == 'TwoAFC_auditory_easy']
    df_test_aud_easy = dft.parameters_for_fit(df_test_aud_easy)
    return (df_test_aud_easy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## multiple mice data import
    """)
    return


@app.cell
def _(Path, pd, project, utils):
    # run this if you want to pull data from several animals
    # acv_animals = [f"ACV{str(i).zfill(3)}" for i in range(1, 11)]
    nuo_animals = [f'NUO{str(i).zfill(3)}' for i in range(1, 13)]
    acv_animals = ['ACV002', 'ACV004', 'ACV008', 'ACV010']
    #ACV mice 2afc multisensory hard stage data
    df_dic = {}
    for mouse_1 in nuo_animals:
        local_path_1 = Path(utils.get_outpath()) / Path(project) / Path('sessions') / Path(mouse_1)
        local_path_1.mkdir(parents=True, exist_ok=True)
        utils.rsync_cluster_data(project_name=project, file_path='sessions/{}/{}.csv'.format(mouse_1, mouse_1), local_path=str(local_path_1), credentials=utils.get_idibaps_cluster_credentials())
        df_single = pd.read_csv(local_path_1 / Path(f'{mouse_1}.csv'), sep=';')  # create the directory if it doesn't exist
        df_dic[mouse_1] = df_single  # download the session data  # load the data
    return (df_dic,)


@app.cell
def _(df_dic, dft, pd):
    df_dic_easy_vis = {}
    df_dic_easy_aud = {}
    df_test_visual = pd.DataFrame()
    df_test_auditory = pd.DataFrame()
    for mouse_2 in df_dic:
        df_dic[mouse_2] = dft.analyze_df(df_dic[mouse_2])
        if 'TwoAFC_visual_easy' in df_dic[mouse_2]['current_training_stage'].values:
            df_dic_easy_vis[mouse_2] = df_dic[mouse_2][df_dic[mouse_2]['current_training_stage'] == 'TwoAFC_visual_easy']
            df_dic_easy_vis[mouse_2] = dft.parameters_for_fit(df_dic_easy_vis[mouse_2])
            df_test_visual = pd.concat([df_test_visual, df_dic_easy_vis[mouse_2]], ignore_index=True)
        if 'TwoAFC_auditory_easy' in df_dic[mouse_2]['current_training_stage'].values:
            df_dic_easy_aud[mouse_2] = df_dic[mouse_2][df_dic[mouse_2]['current_training_stage'] == 'TwoAFC_auditory_easy']
            df_dic_easy_aud[mouse_2] = dft.parameters_for_fit(df_dic_easy_aud[mouse_2])
            df_test_auditory = pd.concat([df_test_auditory, df_dic_easy_aud[mouse_2]], ignore_index=True)
    return df_dic_easy_aud, df_dic_easy_vis


@app.cell
def _(df_dic, dft, pd):
    df_dic_hard_vis = {}
    df_dic_hard_aud = {}
    df_test_visual_1 = pd.DataFrame()
    df_test_auditory_1 = pd.DataFrame()
    for mouse_3 in df_dic:
        df_dic[mouse_3] = dft.analyze_df(df_dic[mouse_3])
        if 'TwoAFC_visual_hard' in df_dic[mouse_3]['current_training_stage'].values:
            df_dic_hard_vis[mouse_3] = df_dic[mouse_3][df_dic[mouse_3]['current_training_stage'] == 'TwoAFC_visual_hard']
            df_dic_hard_vis[mouse_3] = dft.parameters_for_fit(df_dic_hard_vis[mouse_3])
            df_test_visual_1 = pd.concat([df_test_visual_1, df_dic_hard_vis[mouse_3]], ignore_index=True)
        if 'TwoAFC_auditory_hard' in df_dic[mouse_3]['current_training_stage'].values:
            df_dic_hard_aud[mouse_3] = df_dic[mouse_3][df_dic[mouse_3]['current_training_stage'] == 'TwoAFC_auditory_hard']
            df_dic_hard_aud[mouse_3] = dft.parameters_for_fit(df_dic_hard_aud[mouse_3])
            df_test_auditory_1 = pd.concat([df_test_auditory_1, df_dic_hard_aud[mouse_3]], ignore_index=True)
    return (
        df_dic_hard_aud,
        df_dic_hard_vis,
        df_test_auditory_1,
        df_test_visual_1,
    )


@app.cell
def _(df_dic, dft, pd):
    df_dic_hard_multi_vis = {}
    df_dic_hard_multi_aud = {}
    df_test_visual_multi = pd.DataFrame()
    df_test_auditory_multi = pd.DataFrame()
    for mouse_4 in df_dic:
        df_dic[mouse_4] = dft.analyze_df(df_dic[mouse_4])
        df_dic_hard_multi_vis[mouse_4] = df_dic[mouse_4][(df_dic[mouse_4]['current_training_stage'] == 'TwoAFC_multisensory_hard') & (df_dic[mouse_4]['stimulus_modality'] == 'visual')]
        df_dic_hard_multi_vis[mouse_4] = dft.parameters_for_fit(df_dic_hard_multi_vis[mouse_4])
        df_test_visual_multi = pd.concat([df_test_visual_multi, df_dic_hard_multi_vis[mouse_4]], ignore_index=True)
        df_dic_hard_multi_aud[mouse_4] = df_dic[mouse_4][(df_dic[mouse_4]['current_training_stage'] == 'TwoAFC_multisensory_hard') & (df_dic[mouse_4]['stimulus_modality'] == 'auditory')]
        df_dic_hard_multi_aud[mouse_4] = dft.parameters_for_fit(df_dic_hard_multi_aud[mouse_4])
        df_test_auditory_multi = pd.concat([df_test_auditory_multi, df_dic_hard_multi_aud[mouse_4]], ignore_index=True)
    return (
        df_dic_hard_multi_aud,
        df_dic_hard_multi_vis,
        df_test_auditory_multi,
        df_test_visual_multi,
    )


@app.cell
def _(df_dic, dft, pd):
    df_dic_vis = {}
    df_dic_aud = {}
    df_all_visual = pd.DataFrame()
    df_all_auditory = pd.DataFrame()
    for mouse_5 in df_dic:
        df_dic[mouse_5] = dft.analyze_df(df_dic[mouse_5])
        df_dic_vis[mouse_5] = df_dic[mouse_5][df_dic[mouse_5]['stimulus_modality'].str.contains('visual', na=False) & (df_dic[mouse_5]['task'] != 'Habituation')]
        df_dic_vis[mouse_5] = dft.parameters_for_fit(df_dic_vis[mouse_5])
        df_all_visual = pd.concat([df_all_visual, df_dic_vis[mouse_5]], ignore_index=True)
        df_dic_aud[mouse_5] = df_dic[mouse_5][df_dic[mouse_5]['stimulus_modality'].str.contains('auditory', na=False) & (df_dic[mouse_5]['task'] != 'Habituation')]
        df_dic_aud[mouse_5] = dft.parameters_for_fit(df_dic_aud[mouse_5])
        df_all_auditory = pd.concat([df_all_auditory, df_dic_aud[mouse_5]], ignore_index=True)
    return df_all_auditory, df_all_visual, df_dic_aud, df_dic_vis


@app.cell
def _(df_dic):
    for mouse_6 in df_dic:
        if 'TwoAFC_multisensory_hard' in df_dic[mouse_6]['current_training_stage'].unique():
            print(mouse_6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # psychometric curve test
    """)
    return


@app.cell
def _(df_all_visual, df_dic_vis, plots, plt, sns):
    plt.figure(figsize=(8, 8))
    for (df_name, df_1, color) in zip(df_dic_vis.keys(), df_dic_vis.values(), sns.color_palette('colorblind', len(df_dic_vis))):
        plots.psychometric_plot(df_1, x='visual_stimulus_ratio', y='first_choice_numeric', point_kwargs={'color': color, 'label': '', 'alpha': 0.3}, line_kwargs={'color': color, 'label': df_name, 'alpha': 0.3})
    plots.psychometric_plot(df_all_visual, x='visual_stimulus_ratio', y='first_choice_numeric', point_kwargs={'color': 'black', 'label': ''}, line_kwargs={'color': 'black', 'label': 'All mice'})
    plt.show()
    return


@app.cell
def _(df_dic_hard_vis, df_test_visual_1, plots, plt, sns):
    plt.figure(figsize=(8, 8))
    for (df_name_1, df_2, color_1) in zip(df_dic_hard_vis.keys(), df_dic_hard_vis.values(), sns.color_palette('colorblind', len(df_dic_hard_vis))):
        plots.psychometric_plot(df_2, x='visual_stimulus_ratio', y='first_choice_numeric', point_kwargs={'color': color_1, 'label': '', 'alpha': 0.3}, line_kwargs={'color': color_1, 'label': df_name_1, 'alpha': 0.3})
    plots.psychometric_plot(df_test_visual_1, x='visual_stimulus_ratio', y='first_choice_numeric', point_kwargs={'color': 'black', 'label': ''}, line_kwargs={'color': 'black', 'label': 'All mice'})
    plt.show()
    return


@app.cell
def _(np):
    def rescale_to_range(x, new_min=0, new_max=1):
        x = np.array(x, dtype=float)
        old_min, old_max = np.min(x), np.max(x)
        x_scaled = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        return x_scaled

    return (rescale_to_range,)


@app.cell
def _(
    df_all_auditory,
    df_all_visual,
    df_dic_aud,
    plots,
    plt,
    rescale_to_range,
    sns,
):
    plt.figure(figsize=(8, 8))
    for (df_name_2, df_3, color_2) in zip(df_dic_aud.keys(), df_dic_aud.values(), sns.color_palette('colorblind', len(df_dic_aud))):
        df_3['total_evidence_strength_scaled'] = rescale_to_range(df_3['total_evidence_strength'], new_min=df_all_visual['visual_stimulus_ratio'].min(), new_max=df_all_visual['visual_stimulus_ratio'].max())
        plots.psychometric_plot(df_3, x='total_evidence_strength_scaled', y='first_choice_numeric', valueType='continue', point_kwargs={'color': color_2, 'label': '', 'alpha': 0.3}, line_kwargs={'color': color_2, 'label': df_name_2, 'alpha': 0.3})
    df_all_auditory['total_evidence_strength_scaled'] = rescale_to_range(df_all_auditory['total_evidence_strength'], new_min=df_all_visual['visual_stimulus_ratio'].min(), new_max=df_all_visual['visual_stimulus_ratio'].max())
    plots.psychometric_plot(df_all_auditory, x='total_evidence_strength_scaled', y='first_choice_numeric', valueType='continue', point_kwargs={'color': 'black', 'label': ''}, line_kwargs={'color': 'black', 'label': 'All mice'})
    plt.show()
    return


@app.cell
def _(
    df_dic_hard_aud,
    df_test_auditory_1,
    df_test_visual_1,
    plots,
    plt,
    rescale_to_range,
    sns,
):
    plt.figure(figsize=(8, 8))
    for (df_name_3, df_4, color_3) in zip(df_dic_hard_aud.keys(), df_dic_hard_aud.values(), sns.color_palette('colorblind', len(df_dic_hard_aud))):
        df_4['total_evidence_strength_scaled'] = rescale_to_range(df_4['total_evidence_strength'], new_min=df_test_visual_1['visual_stimulus_ratio'].min(), new_max=df_test_visual_1['visual_stimulus_ratio'].max())
        plots.psychometric_plot(df_4, x='total_evidence_strength_scaled', y='first_choice_numeric', valueType='continue', point_kwargs={'color': color_3, 'label': '', 'alpha': 0.3}, line_kwargs={'color': color_3, 'label': df_name_3, 'alpha': 0.3})
    df_test_auditory_1['total_evidence_strength_scaled'] = rescale_to_range(df_test_auditory_1['total_evidence_strength'], new_min=df_test_visual_1['visual_stimulus_ratio'].min(), new_max=df_test_visual_1['visual_stimulus_ratio'].max())
    plots.psychometric_plot(df_test_auditory_1, x='total_evidence_strength_scaled', y='first_choice_numeric', valueType='continue', point_kwargs={'color': 'black', 'label': ''}, line_kwargs={'color': 'black', 'label': 'All mice'})
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## psychometric curve in different stimuls intensity
    """)
    return


@app.cell
def _(df_dic_vis):
    df_test_vis = df_dic_vis['NUO001']
    return (df_test_vis,)


@app.cell
def _(df_test_vis, np, pd, plots, plt, sns):
    # self.settings.side_port_wrong_intensities_extremes = [0.01, 0.1666]
    bins = np.linspace(0.01, 0.1666, num=11)
    labels = [f'{round(bins[i], 4)}-{round(bins[i + 1], 4)}' for i in range(len(bins) - 1)]
    df_test_vis['wrong_bright_bin'] = pd.cut(df_test_vis['wrong_bright'], bins=bins, labels=labels)
    (fig, ax) = plt.subplots(1, 3, figsize=(16, 5))
    colors = sns.color_palette('viridis', len(labels))
    for (label, color_4) in zip(labels, colors):
        for (i, difficulty) in enumerate(df_test_vis['difficulty'].unique()):
            df_test_vis_difficulty = df_test_vis[df_test_vis['difficulty'] == difficulty]
            df_test_vis_bin = df_test_vis_difficulty[df_test_vis_difficulty['wrong_bright_bin'] == label]
            plots.psychometric_plot(df_test_vis_bin, x='visual_stimulus_ratio', y='first_choice_numeric', ax=ax[i], point_kwargs={'label': None, 'color': color_4, 'alpha': 0.5}, line_kwargs={'label': None, 'color': color_4, 'alpha': 0.5})
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=sns.color_palette('viridis', as_cmap=True)), ax=ax, orientation='vertical', shrink=0.3)
    cbar.set_ticks([])
    cbar.set_label('0.01 → 0.1666')
    return


@app.cell
def _(df_test_vis, np, pd, plots, plt, sns):
    # self.settings.side_port_wrong_intensities_extremes = [0.01, 0.1666]
    bins_1 = np.linspace(0.01, 0.833, num=11)
    labels_1 = [f'{round(bins_1[i], 4)}-{round(bins_1[i + 1], 4)}' for i in range(len(bins_1) - 1)]
    df_test_vis['viStim_diff_bin'] = pd.cut(df_test_vis['visual_stimulus_diff'].abs(), bins=bins_1, labels=labels_1)
    (fig_1, ax_1) = plt.subplots(1, 3, figsize=(16, 5))
    colors_1 = sns.color_palette('viridis', len(labels_1))
    for (i_1, difficulty_1) in enumerate(df_test_vis['difficulty'].unique()):
        df_test_vis_difficulty_1 = df_test_vis[df_test_vis['difficulty'] == difficulty_1]
        for (label_1, color_5) in zip(labels_1, colors_1):
            df_test_vis_bin_1 = df_test_vis_difficulty_1[df_test_vis_difficulty_1['viStim_diff_bin'] == label_1]
            if len(df_test_vis_bin_1) != 0:
                plots.psychometric_plot(df_test_vis_bin_1, x='visual_stimulus_ratio', y='first_choice_numeric', ax=ax_1[i_1], point_kwargs={'label': None, 'color': color_5, 'alpha': 0.5}, line_kwargs={'label': None, 'color': color_5, 'alpha': 0.5})
            else:
                pass
    cbar_1 = plt.colorbar(plt.cm.ScalarMappable(cmap=sns.color_palette('viridis', as_cmap=True)), ax=ax_1, orientation='vertical', shrink=0.3)
    cbar_1.set_ticks([])
    cbar_1.set_label('0.01 → 0.833')
    return


@app.cell
def _(df_dic_vis, np, pd, plots, plt, sns):
    for mouse_7 in df_dic_vis:
        df_test_vis_1 = df_dic_vis[mouse_7]
        df_test_vis_1['year_month_day'] = pd.to_datetime(df_test_vis_1['year_month_day'])
        latest_day = df_test_vis_1['year_month_day'].max()
        df_test_vis_recent = df_test_vis_1[df_test_vis_1['year_month_day'] >= latest_day - pd.Timedelta(days=7)]
        bins_2 = np.linspace(0.01, 0.1666, num=11)  # self.settings.side_port_wrong_intensities_extremes = [0.01, 0.1666]
        labels_2 = [f'{round(bins_2[i], 4)}-{round(bins_2[i + 1], 4)}' for i in range(len(bins_2) - 1)]
        df_test_vis_recent['wrong_bright_bin'] = pd.cut(df_test_vis_recent['wrong_bright'], bins=bins_2, labels=labels_2)
        (fig_2, ax_2) = plt.subplots(1, 1, figsize=(8, 5))
        colors_2 = sns.color_palette('viridis', len(labels_2))
        for (label_2, color_6) in zip(labels_2, colors_2):
            df_test_vis_bin_2 = df_test_vis_recent[df_test_vis_recent['wrong_bright_bin'] == label_2]
            plots.psychometric_plot(df_test_vis_bin_2, x='visual_stimulus_ratio', y='first_choice_numeric', ax=ax_2, point_kwargs={'label': None, 'color': color_6, 'alpha': 0.5}, line_kwargs={'label': None, 'color': color_6, 'alpha': 0.5})
        cbar_2 = plt.colorbar(plt.cm.ScalarMappable(cmap=sns.color_palette('viridis', as_cmap=True)), ax=ax_2, orientation='vertical', shrink=0.3)
        cbar_2.set_ticks([])
        cbar_2.set_label('0.01 → 0.1666')
        plt.title(f'Mouse: {mouse_7} - Last 7 days')
        fig_2.savefig(f'/home/kudongdong/data/LeciLab/temporary_plots/psychometric_wrong_bright_{mouse_7}_last7days.png')
    return


@app.cell
def _(
    df_dic_hard_aud,
    df_dic_hard_vis,
    df_test_visual_1,
    rescale_to_range,
    utils,
):
    lapse_left_list_vis = []
    lapse_right_list_vis = []
    beta_list_vis = []
    x0_list_vis = []
    for mouse_8 in df_dic_hard_vis:
        (y, params) = utils.psychometric_curve_fitting_params(df_dic_hard_vis[mouse_8], x='visual_stimulus_ratio', y='first_choice_numeric')
        (lapse_left, lapse_right, beta, x0) = params
        lapse_left_list_vis.append(lapse_left)
        lapse_right_list_vis.append(lapse_right)
        beta_list_vis.append(beta)
        x0_list_vis.append(x0)
    lapse_left_list_aud = []
    lapse_right_list_aud = []
    beta_list_aud = []
    x0_list_aud = []
    for mouse_8 in df_dic_hard_aud:
        df_dic_hard_aud[mouse_8]['total_evidence_strength_scaled'] = rescale_to_range(df_dic_hard_aud[mouse_8]['total_evidence_strength'], new_min=df_test_visual_1['visual_stimulus_ratio'].min(), new_max=df_test_visual_1['visual_stimulus_ratio'].max())
        (y, params) = utils.psychometric_curve_fitting_params(df_dic_hard_aud[mouse_8], x='total_evidence_strength_scaled', y='first_choice_numeric', valueType='continue', bins=6)
        (lapse_left, lapse_right, beta, x0) = params
        lapse_left_list_aud.append(lapse_left)
        lapse_right_list_aud.append(lapse_right)
        beta_list_aud.append(beta)
        x0_list_aud.append(x0)
    return (
        beta_list_aud,
        beta_list_vis,
        lapse_left_list_aud,
        lapse_left_list_vis,
        lapse_right_list_aud,
        lapse_right_list_vis,
        x0_list_aud,
        x0_list_vis,
    )


@app.cell
def _(
    beta_list_aud,
    beta_list_vis,
    lapse_left_list_aud,
    lapse_left_list_vis,
    lapse_right_list_aud,
    lapse_right_list_vis,
    pd,
    x0_list_aud,
    x0_list_vis,
):
    params_vis = pd.DataFrame({
                'lapse_left': lapse_left_list_vis,
                'lapse_right': lapse_right_list_vis,
                'beta': beta_list_vis,
                'x0': x0_list_vis
                })
    params_vis['condition'] = 'Visual'

    params_aud = pd.DataFrame({
                'lapse_left': lapse_left_list_aud,
                'lapse_right': lapse_right_list_aud,
                'beta': beta_list_aud,
                'x0': x0_list_aud
                })
    params_aud['condition'] = 'Auditory'

    params_df = pd.concat([params_vis, params_aud], ignore_index=True)
    return (params_df,)


@app.cell
def _(params_df, plots):
    plots.plot_param_comparison(
        df=params_df,
        params=['beta', 'x0', 'lapse_left', 'lapse_right'],
        conditions=['Visual', 'Auditory'],
        test='paired_t',
        title='Psychometric Curve on Left Choice',
        palette=["#D4361D", "#875215"]
    )
    return


@app.cell
def _(df_test_visual_1):
    df_test_visual_1['visual_stimulus']
    return


@app.cell
def _(
    df_dic_hard_aud,
    df_dic_hard_vis,
    df_test_visual_1,
    rescale_to_range,
    utils,
):
    lapse_hard_list_vis = []
    lapse_easy_list_vis = []
    beta_list_vis_1 = []
    x0_list_vis_1 = []
    for mouse_9 in df_dic_hard_vis:
        (y_1, params_1) = utils.psychometric_curve_fitting_params(df_dic_hard_vis[mouse_9], x='abs_visual_stimulus_ratio', y='correct_numeric')
        (lapse_hard, lapse_easy, beta_1, x0_1) = params_1
        lapse_hard_list_vis.append(lapse_hard)
        lapse_easy_list_vis.append(lapse_easy)
        beta_list_vis_1.append(beta_1)
        x0_list_vis_1.append(x0_1)
    lapse_hard_list_aud = []
    lapse_easy_list_aud = []
    beta_list_aud_1 = []
    x0_list_aud_1 = []
    for mouse_9 in df_dic_hard_aud:
        df_dic_hard_aud[mouse_9]['total_evidence_strength_scaled'] = rescale_to_range(df_dic_hard_aud[mouse_9]['total_evidence_strength'], new_min=df_test_visual_1['visual_stimulus_ratio'].min(), new_max=df_test_visual_1['visual_stimulus_ratio'].max())
        df_dic_hard_aud[mouse_9]['abs_total_evidence_strength_scaled'] = df_dic_hard_aud[mouse_9]['total_evidence_strength_scaled'].abs()
        (y_1, params_1) = utils.psychometric_curve_fitting_params(df_dic_hard_aud[mouse_9], x='abs_total_evidence_strength_scaled', y='correct_numeric', valueType='continue', bins=3)
        (lapse_hard, lapse_easy, beta_1, x0_1) = params_1
        lapse_hard_list_aud.append(lapse_hard)
        lapse_easy_list_aud.append(lapse_easy)
        beta_list_aud_1.append(beta_1)
        x0_list_aud_1.append(x0_1)
    return (
        beta_list_aud_1,
        beta_list_vis_1,
        lapse_easy_list_aud,
        lapse_easy_list_vis,
        lapse_hard_list_aud,
        lapse_hard_list_vis,
        x0_list_aud_1,
        x0_list_vis_1,
    )


@app.cell
def _(
    beta_list_aud_1,
    beta_list_vis_1,
    lapse_easy_list_aud,
    lapse_easy_list_vis,
    lapse_hard_list_aud,
    lapse_hard_list_vis,
    pd,
    x0_list_aud_1,
    x0_list_vis_1,
):
    params_vis_1 = pd.DataFrame({'lapse_hard': lapse_hard_list_vis, 'lapse_easy': lapse_easy_list_vis, 'beta': beta_list_vis_1, 'x0': x0_list_vis_1})
    params_vis_1['condition'] = 'Visual'
    params_aud_1 = pd.DataFrame({'lapse_hard': lapse_hard_list_aud, 'lapse_easy': lapse_easy_list_aud, 'beta': beta_list_aud_1, 'x0': x0_list_aud_1})
    params_aud_1['condition'] = 'Auditory'
    params_df_1 = pd.concat([params_vis_1, params_aud_1], ignore_index=True)
    return (params_df_1,)


@app.cell
def _(params_df_1, plots):
    plots.plot_param_comparison(df=params_df_1, params=['beta', 'x0', 'lapse_hard', 'lapse_easy'], conditions=['Visual', 'Auditory'], test='paired_t', title='Psychometric Curve on Correct Choice', palette=['#D4361D', '#875215'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correct wrong psychometric curve
    """)
    return


@app.cell
def _(df_test_visual_multi, plots, plt):
    (fig_3, ax_3) = plt.subplots(1, 2, figsize=(10, 5))
    for (i_2, linecolor) in zip(df_test_visual_multi[df_test_visual_multi['previous_port_before_stimulus'] == 'left'].groupby('previous_correct_numeric'), ['red', 'green']):
        plots.psychometric_plot(i_2[1], x='visual_stimulus_ratio', y='first_choice_numeric', line_kwargs={'color': linecolor, 'label': 'True_previous' if i_2[0] == 1 else 'False_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_3[0])
    for (i_2, linecolor) in zip(df_test_visual_multi[df_test_visual_multi['previous_port_before_stimulus'] == 'right'].groupby('previous_correct_numeric'), ['red', 'green']):
        plots.psychometric_plot(i_2[1], x='visual_stimulus_ratio', y='first_choice_numeric', line_kwargs={'color': linecolor, 'label': 'True_previous' if i_2[0] == 1 else 'False_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_3[1])
    ax_3[0].legend()
    ax_3[0].set_title('Left Choice Previous multimodality hard')
    ax_3[1].legend()
    ax_3[1].set_title('Right Choice Previous multimodality hard')
    return


@app.cell
def _(df_test_auditory_multi, plots, plt):
    (fig_4, ax_4) = plt.subplots(1, 2, figsize=(10, 5))
    for (i_3, linecolor_1) in zip(df_test_auditory_multi[df_test_auditory_multi['previous_port_before_stimulus'] == 'left'].groupby('previous_correct_numeric'), ['red', 'green']):
        plots.psychometric_plot(i_3[1], x='total_evidence_strength', y='first_choice_numeric', valueType='continue', line_kwargs={'color': linecolor_1, 'label': 'True_previous' if i_3[0] == 1 else 'False_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_4[0])
    for (i_3, linecolor_1) in zip(df_test_auditory_multi[df_test_auditory_multi['previous_port_before_stimulus'] == 'right'].groupby('previous_correct_numeric'), ['red', 'green']):
        plots.psychometric_plot(i_3[1], x='total_evidence_strength', y='first_choice_numeric', valueType='continue', line_kwargs={'color': linecolor_1, 'label': 'True_previous' if i_3[0] == 1 else 'False_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_4[1])
    ax_4[0].legend()
    ax_4[0].set_title('Left Choice Previous multimodality hard')
    ax_4[1].legend()
    ax_4[1].set_title('Right Choice Previous multimodality hard')
    return


@app.cell
def _(df_test_visual_multi, plots, plt):
    (fig_5, ax_5) = plt.subplots(1, 2, figsize=(10, 5))
    for (i_4, linecolor_2) in zip(df_test_visual_multi[df_test_visual_multi['previous_correct_numeric'] == 1].groupby('previous_port_before_stimulus'), ['gold', 'lightskyblue']):
        plots.psychometric_plot(df=i_4[1], x='visual_stimulus_ratio', y='first_choice_numeric', line_kwargs={'color': linecolor_2, 'label': 'Left_previous' if i_4[0] == 'left' else 'Right_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_5[0])
    for (i_4, linecolor_2) in zip(df_test_visual_multi[df_test_visual_multi['previous_correct_numeric'] == 0].groupby('previous_port_before_stimulus'), ['gold', 'lightskyblue']):
        plots.psychometric_plot(df=i_4[1], x='visual_stimulus_ratio', y='first_choice_numeric', line_kwargs={'color': linecolor_2, 'label': 'Left_previous' if i_4[0] == 'left' else 'Right_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_5[1])
    ax_5[0].legend()
    ax_5[0].set_title('Correct Choice Previous multimodality hard')
    ax_5[1].legend()
    ax_5[1].set_title('Incorrect Choice Previous multimodality hard')
    return


@app.cell
def _(df_test_auditory_multi, plots, plt):
    (fig_6, ax_6) = plt.subplots(1, 2, figsize=(10, 5))
    for (i_5, linecolor_3) in zip(df_test_auditory_multi[df_test_auditory_multi['previous_correct_numeric'] == 1].groupby('previous_port_before_stimulus'), ['gold', 'lightskyblue']):
        plots.psychometric_plot(i_5[1], x='total_evidence_strength', y='first_choice_numeric', valueType='continue', line_kwargs={'color': linecolor_3, 'label': 'Left_previous' if i_5[0] == 'left' else 'Right_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_6[0])
    for (i_5, linecolor_3) in zip(df_test_auditory_multi[df_test_auditory_multi['previous_correct_numeric'] == 0].groupby('previous_port_before_stimulus'), ['gold', 'lightskyblue']):
        plots.psychometric_plot(i_5[1], x='total_evidence_strength', y='first_choice_numeric', valueType='continue', line_kwargs={'color': linecolor_3, 'label': 'Left_previous' if i_5[0] == 'left' else 'Right_previous'}, point_kwargs={'color': 'black', 'label': None}, ax=ax_6[1])
    ax_6[0].legend()
    ax_6[0].set_title('Correct Choice Previous multimodality hard')
    ax_6[1].legend()
    ax_6[1].set_title('Incorrect Choice Previous multimodality hard')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matrix format
    """)
    return


@app.cell
def _(df_test_visual_1, np, plt, sns):
    (fig_7, axes) = plt.subplots(1, 2, figsize=(15, 6))
    for i_6 in df_test_visual_1[df_test_visual_1['previous_port_before_stimulus'] == 'left'].groupby('previous_correct_numeric'):
        df_preleft = i_6[1]
        df_preleft['visual_stimulus_ratio'] = df_preleft['visual_stimulus_ratio'].apply(lambda x: round(x, 2))
        for mouse_10 in df_preleft['subject'].unique():
            for session in df_preleft[df_preleft.subject == mouse_10]['session'].unique():
                df_mouse_session = df_preleft[np.logical_and(df_preleft['subject'] == mouse_10, df_preleft['session'] == session)]
                df_mouse_session['previous_visual_stimulus_ratio'] = df_mouse_session['visual_stimulus_ratio'].shift(1, fill_value=np.nan)
                df_preleft.loc[df_mouse_session.index, 'previous_visual_stimulus_ratio'] = df_mouse_session['previous_visual_stimulus_ratio']
        pivot_table = df_preleft.pivot_table(index='previous_visual_stimulus_ratio', columns='visual_stimulus_ratio', values='first_choice_numeric', aggfunc='mean', observed=True)
        sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Probability of Left Choice'}, vmin=0, vmax=1, annot_kws={'color': 'black'}, ax=axes[i_6[0]])
        axes[i_6[0]].set_xlabel('Visual Stimulus Ratio')
        axes[i_6[0]].set_ylabel('Previous Visual Stimulus Ratio')
        axes[i_6[0]].set_title('Heatmap of Probability of Left Choice (Previous Left {})'.format('Correct' if i_6[0] == 1 else 'Incorrect'))
    plt.yticks(rotation=0)
    # rotate the y-axis labels
    plt.show()  # plot the heatmap
    return


@app.cell
def _(df_test_visual_1, np, plt, sns):
    (fig_8, axes_1) = plt.subplots(1, 2, figsize=(15, 6))
    for i_7 in df_test_visual_1[df_test_visual_1['previous_port_before_stimulus'] == 'right'].groupby('previous_correct_numeric'):
        df_preright = i_7[1]
        df_preright['visual_stimulus_ratio'] = df_preright['visual_stimulus_ratio'].apply(lambda x: round(x, 2))
        for mouse_11 in df_preright['subject'].unique():
            for session_1 in df_preright[df_preright.subject == mouse_11]['session'].unique():
                df_mouse_session_1 = df_preright[np.logical_and(df_preright['subject'] == mouse_11, df_preright['session'] == session_1)]
                df_mouse_session_1['previous_visual_stimulus_ratio'] = df_mouse_session_1['visual_stimulus_ratio'].shift(1, fill_value=np.nan)
                df_preright.loc[df_mouse_session_1.index, 'previous_visual_stimulus_ratio'] = df_mouse_session_1['previous_visual_stimulus_ratio']
        pivot_table_1 = df_preright.pivot_table(index='previous_visual_stimulus_ratio', columns='visual_stimulus_ratio', values='first_choice_numeric', aggfunc='mean', observed=True)
        sns.heatmap(pivot_table_1, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Probability of left Choice'}, vmin=0, vmax=1, annot_kws={'color': 'black'}, ax=axes_1[i_7[0]])
        axes_1[i_7[0]].set_xlabel('Visual Stimulus Ratio')
        axes_1[i_7[0]].set_ylabel('Previous Visual Stimulus Ratio')
        axes_1[i_7[0]].set_title('Heatmap of Probability of left Choice (Previous right {})'.format('Correct' if i_7[0] == 1 else 'Incorrect'))
    plt.yticks(rotation=0)
    # rotate the y-axis labels
    plt.show()  # plot the heatmap
    return


@app.cell
def _(df_test_auditory_1, np, pd, plt, sns):
    (fig_9, axes_2) = plt.subplots(1, 2, figsize=(15, 6))
    for i_8 in df_test_auditory_1[df_test_auditory_1['previous_port_before_stimulus'] == 'left'].groupby('previous_correct_numeric'):
        df_preleft_1 = i_8[1]
        bins_3 = 6
        bin_groups = pd.cut(df_preleft_1['total_evidence_strength'], bins=bins_3)
        labels_3 = df_preleft_1['total_evidence_strength'].groupby(bin_groups).mean()
        df_preleft_1['total_evidence_strength_binned'] = pd.cut(df_preleft_1['total_evidence_strength'], bins=bins_3, labels=labels_3).astype(float)
        df_preleft_1['total_evidence_strength_binned'] = df_preleft_1['total_evidence_strength_binned'].apply(lambda x: round(x, 2))
        for mouse_12 in df_preleft_1['subject'].unique():
            for session_2 in df_preleft_1[df_preleft_1.subject == mouse_12]['session'].unique():
                df_mouse_session_2 = df_preleft_1[np.logical_and(df_preleft_1['subject'] == mouse_12, df_preleft_1['session'] == session_2)]
                df_mouse_session_2['previous_total_evidence_strength_binned'] = df_mouse_session_2['total_evidence_strength_binned'].shift(1, fill_value=np.nan)
                df_preleft_1.loc[df_mouse_session_2.index, 'previous_total_evidence_strength_binned'] = df_mouse_session_2['previous_total_evidence_strength_binned']
        pivot_table_2 = df_preleft_1.pivot_table(index='previous_total_evidence_strength_binned', columns='total_evidence_strength_binned', values='first_choice_numeric', aggfunc='mean', observed=True)
        sns.heatmap(pivot_table_2, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Probability of Left Choice'}, vmin=0, vmax=1, annot_kws={'color': 'black'}, ax=axes_2[i_8[0]])
        axes_2[i_8[0]].set_xlabel('Total Evidence Strength')
        axes_2[i_8[0]].set_ylabel('Previous Total Evidence Strength')
        axes_2[i_8[0]].set_title('Heatmap of Probability of Left Choice (Previous Left {})'.format('Correct' if i_8[0] == 1 else 'Incorrect'))
    plt.yticks(rotation=0)
    # rotate the y-axis labels
    plt.show()  # plot the heatmap
    return


@app.cell
def _(df_test_auditory_1, np, pd, plt, sns):
    (fig_10, axes_3) = plt.subplots(1, 2, figsize=(15, 6))
    for i_9 in df_test_auditory_1[df_test_auditory_1['previous_port_before_stimulus'] == 'right'].groupby('previous_correct_numeric'):
        df_preright_1 = i_9[1]
        bins_4 = 6
        bin_groups_1 = pd.cut(df_preright_1['total_evidence_strength'], bins=bins_4)
        labels_4 = df_preright_1['total_evidence_strength'].groupby(bin_groups_1).mean()
        df_preright_1['total_evidence_strength_binned'] = pd.cut(df_preright_1['total_evidence_strength'], bins=bins_4, labels=labels_4).astype(float)
        df_preright_1['total_evidence_strength_binned'] = df_preright_1['total_evidence_strength_binned'].apply(lambda x: round(x, 2))
        for mouse_13 in df_preright_1['subject'].unique():
            for session_3 in df_preright_1[df_preright_1.subject == mouse_13]['session'].unique():
                df_mouse_session_3 = df_preright_1[np.logical_and(df_preright_1['subject'] == mouse_13, df_preright_1['session'] == session_3)]
                df_mouse_session_3['previous_total_evidence_strength_binned'] = df_mouse_session_3['total_evidence_strength_binned'].shift(1, fill_value=np.nan)
                df_preright_1.loc[df_mouse_session_3.index, 'previous_total_evidence_strength_binned'] = df_mouse_session_3['previous_total_evidence_strength_binned']
        pivot_table_3 = df_preright_1.pivot_table(index='previous_total_evidence_strength_binned', columns='total_evidence_strength_binned', values='first_choice_numeric', aggfunc='mean', observed=True)
        sns.heatmap(pivot_table_3, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Probability of left Choice'}, vmin=0, vmax=1, annot_kws={'color': 'black'}, ax=axes_3[i_9[0]])
        axes_3[i_9[0]].set_xlabel('Total Evidence Strength')
        axes_3[i_9[0]].set_ylabel('Previous Total Evidence Strength')
        axes_3[i_9[0]].set_title('Heatmap of Probability of left Choice (Previous right {})'.format('Correct' if i_9[0] == 1 else 'Incorrect'))
    plt.yticks(rotation=0)
    # rotate the y-axis labels
    plt.show()  # plot the heatmap
    return


@app.cell
def _(df_test_visual_1, np, pd, plt, sns):
    # let's use the absolute value of the lowest visual stimulus as a proxy for the brightness of the visual stimulus
    df_test_visual_1['visual_stimulus_lowest'] = df_test_visual_1['visual_stimulus'].apply(lambda x: abs(eval(x)[0]) if eval(x)[0] < eval(x)[1] else abs(eval(x)[1]))
    # visual stimulus ratio to only two decimal places
    df_test_visual_1['visual_stimulus_ratio'] = df_test_visual_1['visual_stimulus_ratio'].apply(lambda x: round(x, 2))
    # create 10 bins for the absolute value of the lowest visual stimulus
    min_value = df_test_visual_1['visual_stimulus_lowest'].min()
    max_value = df_test_visual_1['visual_stimulus_lowest'].max()
    bins_5 = np.linspace(min_value, max_value, 5)
    df_test_visual_1['visual_stimulus_lowest_binned'] = pd.cut(df_test_visual_1['visual_stimulus_lowest'], bins=bins_5, labels=[f'{b:.2f}' for b in bins_5[:-1]])
    # create a pivot table with the visual stimulus ratio and absolute value of the lowest visual stimulus
    pivot_table_abs = df_test_visual_1.pivot_table(index='visual_stimulus_lowest_binned', columns='visual_stimulus_ratio', values='first_choice_numeric', aggfunc='mean', observed=True)
    plt.figure(figsize=(5, 3))
    sns.heatmap(pivot_table_abs, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Probability of Left Choice'}, vmin=0, vmax=1, annot_kws={'color': 'black'})
    plt.xlabel('Evidence to the left (in luminance ratio)')
    plt.ylabel('Absolute Value of\nLowest Visual Stimulus')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    # plot the heatmap
    # plt.title("Heatmap of Probability of Left Choice")
    # rotate the y-axis labels
    # save in Desktop
    # plt.savefig('/home/hmv/lab/heatmap_visual_stimulus.pdf', bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(LogisticRegression, df_test_visual_1, pd, plt):
    # transform visual_stimulus_lowest_binned to a numeric value for plotting
    df_test_visual_1['visual_stimulus_lowest_binned_num'] = pd.to_numeric(df_test_visual_1['visual_stimulus_lowest_binned'], errors='coerce')
    (fig_11, axs) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # make two plots, one for when the animals comes from the left and one for when it comes from the right
    for (ax_7, side) in zip(axs.ravel(), ['left', 'right']):
    # Plot for when the animal comes from the left
        df_side = df_test_visual_1[df_test_visual_1['previous_port_before_stimulus'] == side]
        for i_10 in df_side.groupby('visual_stimulus_ratio'):
            df_i = i_10[1].sort_values(by='visual_stimulus_lowest_binned_num')
            df_i = df_i.dropna(subset=['visual_stimulus_lowest_binned_num'])
            X = df_i['visual_stimulus_lowest_binned_num'].values.reshape(-1, 1)  # drop nan
            y_2 = df_i['first_choice_numeric'].values.astype(int)
            model = LogisticRegression()
            model.fit(X, y_2)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            ax_7.plot(X, y_prob, label=f'Visual Stimulus ratio: {i_10[0]}')
        ax_7.set_xlabel('Absolute Value of Lowest Visual Stimulus')
        ax_7.set_ylabel('Probability of Left Choice')
        ax_7.legend()
        ax_7.set_title(f'Last Choice Before Stimulus: {side.capitalize()}')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## compare stimulus kernel model and action kernel model
    """)
    return


@app.cell
def _(np):
    def recursive_kernel_prior(history, alpha, pi0=0.5):
        """
        history: 1D array-like of 0/1 values for updating the prior

        alpha: float in (0, 1)
        pi0: initial prior, usually set to 0.5

        returns
        -------
        pi : np.ndarray, shape (T,)
             pi[t] represents the prior before the t-th trial
        """
        history = np.asarray(history, dtype=float)
        T = len(history)
        pi = np.empty(T, dtype=float)

        # Before the first trial, there is no history, so use the initial value
        pi[0] = pi0

        # From the second trial onwards, update using the previous trial's history
        for t in range(1, T):
            pi[t] = (1 - alpha) * pi[t - 1] + alpha * history[t - 1]

        return pi

    def logit(p, eps=1e-9):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def choice_prob_from_prior(stim, pi, beta, w, bias, lapse):
        dv = bias + beta * stim + w * logit(pi)
        p = sigmoid(dv)
        p = (1 - lapse) * p + lapse * 0.5
        return np.clip(p, 1e-9, 1 - 1e-9)

    return choice_prob_from_prior, recursive_kernel_prior


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    θ=(α,β,w,bias,lapse)

    πt​=recursive_kernel_prior(stim_r,α)
    """)
    return


@app.cell
def _(choice_prob_from_prior, np, recursive_kernel_prior):
    def session_loglik_kernel(theta, sess_df, prior_type="stim"):
        alpha = theta["alpha"]
        beta  = theta["beta"]
        w     = theta["w"]
        bias  = theta["bias"]
        lapse = theta["lapse"]

        if sess_df['stimulus_modality'].iloc[0] == "visual":
            x = sess_df["visual_stimulus_ratio"].to_numpy(dtype=float)
        elif sess_df['stimulus_modality'].iloc[0] == "auditory":
            x = sess_df["total_evidence_strength"].to_numpy(dtype=float)
        else:
            raise ValueError("Unknown stimulus modality: {}".format(sess_df['stimulus_modality'].iloc[0]))

        if sess_df["first_choice_numeric"].isna().any():
            raise ValueError("NaN in first_choice_numeric")


        if prior_type == "stim":
            history = sess_df["correct_side_numeric"]
            if history.isnull().any():
                raise ValueError("NaN in correct_side_numeric")
        elif prior_type == "act":
            history = sess_df["first_choice_numeric"]
        y = sess_df["first_choice_numeric"].to_numpy(dtype=int)

        pi = recursive_kernel_prior(history, alpha=alpha, pi0=0.5)
        p = choice_prob_from_prior(x, pi, beta=beta, w=w, bias=bias, lapse=lapse)

        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return ll

    return (session_loglik_kernel,)


@app.cell
def _(session_loglik_kernel):
    def dataset_loglik(theta, df, prior_type='stim'):
        total_ll = 0.0
        for (_, sess_df) in df.groupby('session', sort=False):
            total_ll = total_ll + session_loglik_kernel(theta, sess_df, prior_type=prior_type)
        return total_ll

    return (dataset_loglik,)


@app.cell
def _(dataset_loglik, np):
    def unpack_params(v):
        alpha = 1 / (1 + np.exp(-v[0]))
        beta  = v[1]
        w     = v[2]
        bias  = v[3]
        lapse = 1 / (1 + np.exp(-v[4]))
        return {"alpha": alpha, "beta": beta, "w": w, "bias": bias, "lapse": lapse}

    def neg_loglik_reg(v, df, prior_type):
        theta = unpack_params(v)
        ll = dataset_loglik(theta, df, prior_type=prior_type)

        # add regularization to make sure the parameters don't grow too large, which can lead to overfitting and numerical instability
        reg = 1 * (theta["beta"]**2 + theta["w"]**2 + theta["bias"]**2)

        return -(ll - reg)

    v0 = np.array([0.0, 1.0, 1.0, 0.0, -2.0])
    return neg_loglik_reg, unpack_params, v0


@app.cell
def _(
    df_dic_easy_aud,
    minimize,
    neg_loglik_reg,
    pd,
    session_loglik_kernel,
    unpack_params,
    v0,
):
    theta_act_df_easy_aud = pd.DataFrame([])
    theta_stim_df_easy_aud = pd.DataFrame([])
    ll_stim_easy_aud_dic = {}
    ll_act_easy_aud_dic = {}
    for animal in df_dic_easy_aud:
        df_5 = df_dic_easy_aud[animal]
        df_clean = df_5.dropna(subset=['first_choice_numeric', 'correct_side_numeric'])
        res_stim = minimize(neg_loglik_reg, v0, args=(df_clean, 'stim'), method='L-BFGS-B')
        res_act = minimize(neg_loglik_reg, v0, args=(df_clean, 'act'), method='L-BFGS-B')
        theta_stim = unpack_params(res_stim.x)
        theta_act = unpack_params(res_act.x)
        ll_stim_easy_aud_dic[animal] = session_loglik_kernel(theta_stim, df_clean, prior_type='stim')
        ll_act_easy_aud_dic[animal] = session_loglik_kernel(theta_act, df_clean, prior_type='act')
        theta_stim_df_easy_aud[animal] = pd.DataFrame([theta_stim]).T[0]
        theta_act_df_easy_aud[animal] = pd.DataFrame([theta_act]).T[0]
    return (theta_act_df_easy_aud,)


@app.cell
def _(
    df_dic_easy_vis,
    minimize,
    neg_loglik_reg,
    pd,
    session_loglik_kernel,
    unpack_params,
    v0,
):
    theta_act_df_easy_vis = pd.DataFrame([])
    theta_stim_df_easy_vis = pd.DataFrame([])
    ll_stim_easy_vis_dic = {}
    ll_act_easy_vis_dic = {}
    for animal_1 in df_dic_easy_vis:
        df_6 = df_dic_easy_vis[animal_1]
        df_clean_1 = df_6.dropna(subset=['first_choice_numeric', 'correct_side_numeric'])
        res_stim_1 = minimize(neg_loglik_reg, v0, args=(df_clean_1, 'stim'), method='L-BFGS-B')
        res_act_1 = minimize(neg_loglik_reg, v0, args=(df_clean_1, 'act'), method='L-BFGS-B')
        theta_stim_1 = unpack_params(res_stim_1.x)
        theta_act_1 = unpack_params(res_act_1.x)
        ll_stim_easy_vis_dic[animal_1] = session_loglik_kernel(theta_stim_1, df_clean_1, prior_type='stim')
        ll_act_easy_vis_dic[animal_1] = session_loglik_kernel(theta_act_1, df_clean_1, prior_type='act')
        theta_stim_df_easy_vis[animal_1] = pd.DataFrame([theta_stim_1]).T[0]
        theta_act_df_easy_vis[animal_1] = pd.DataFrame([theta_act_1]).T[0]
    return (theta_act_df_easy_vis,)


@app.cell
def _(
    df_dic_hard_aud,
    minimize,
    neg_loglik_reg,
    pd,
    session_loglik_kernel,
    unpack_params,
    v0,
):
    theta_act_df_hard_aud = pd.DataFrame([])
    theta_stim_df_hard_aud = pd.DataFrame([])
    ll_stim_hard_aud_dic = {}
    ll_act_hard_aud_dic = {}
    for animal_2 in df_dic_hard_aud:
        df_7 = df_dic_hard_aud[animal_2]
        df_clean_2 = df_7.dropna(subset=['first_choice_numeric', 'correct_side_numeric'])
        res_stim_2 = minimize(neg_loglik_reg, v0, args=(df_clean_2, 'stim'), method='L-BFGS-B')
        res_act_2 = minimize(neg_loglik_reg, v0, args=(df_clean_2, 'act'), method='L-BFGS-B')
        theta_stim_2 = unpack_params(res_stim_2.x)
        theta_act_2 = unpack_params(res_act_2.x)
        ll_stim_hard_aud_dic[animal_2] = session_loglik_kernel(theta_stim_2, df_clean_2, prior_type='stim')
        ll_act_hard_aud_dic[animal_2] = session_loglik_kernel(theta_act_2, df_clean_2, prior_type='act')
        theta_stim_df_hard_aud[animal_2] = pd.DataFrame([theta_stim_2]).T[0]
        theta_act_df_hard_aud[animal_2] = pd.DataFrame([theta_act_2]).T[0]
    return (theta_act_df_hard_aud,)


@app.cell
def _(
    df_dic_hard_vis,
    minimize,
    neg_loglik_reg,
    pd,
    session_loglik_kernel,
    unpack_params,
    v0,
):
    theta_act_df_hard_vis = pd.DataFrame([])
    theta_stim_df_hard_vis = pd.DataFrame([])
    ll_stim_hard_vis_dic = {}
    ll_act_hard_vis_dic = {}
    for animal_3 in df_dic_hard_vis:
        df_8 = df_dic_hard_vis[animal_3]
        df_clean_3 = df_8.dropna(subset=['first_choice_numeric', 'correct_side_numeric'])
        res_stim_3 = minimize(neg_loglik_reg, v0, args=(df_clean_3, 'stim'), method='L-BFGS-B')
        res_act_3 = minimize(neg_loglik_reg, v0, args=(df_clean_3, 'act'), method='L-BFGS-B')
        theta_stim_3 = unpack_params(res_stim_3.x)
        theta_act_3 = unpack_params(res_act_3.x)
        ll_stim_hard_vis_dic[animal_3] = session_loglik_kernel(theta_stim_3, df_clean_3, prior_type='stim')
        ll_act_hard_vis_dic[animal_3] = session_loglik_kernel(theta_act_3, df_clean_3, prior_type='act')
        theta_stim_df_hard_vis[animal_3] = pd.DataFrame([theta_stim_3]).T[0]
        theta_act_df_hard_vis[animal_3] = pd.DataFrame([theta_act_3]).T[0]
    return (theta_act_df_hard_vis,)


@app.cell
def _(
    pd,
    plt,
    sns,
    theta_act_df_easy_aud,
    theta_act_df_easy_vis,
    theta_act_df_hard_aud,
    theta_act_df_hard_vis,
):
    df_alpha = pd.DataFrame({
        'easy_aud': theta_act_df_easy_aud.loc['alpha'],
        'hard_aud': theta_act_df_hard_aud.loc['alpha'],
        'easy_vis': theta_act_df_easy_vis.loc['alpha'],
        'hard_vis': theta_act_df_hard_vis.loc['alpha'],
    })
    df_long = df_alpha.melt(var_name='condition', value_name='alpha')

    plt.figure(figsize=(6,5))

    sns.barplot(
        data=df_long,
        x='condition',
        y='alpha',
        errorbar='se',
        capsize=0.2,
        color='lightgray'
    )

    sns.stripplot(
        data=df_long,
        x='condition',
        y='alpha',
        color='black',
        size=6,
        jitter=True,
        alpha=0.7
    )

    plt.ylabel('alpha')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_test_aud_easy, minimize, neg_loglik_reg, unpack_params, v0):
    df_test = df_test_aud_easy.dropna(subset=['correct_side_numeric', 'first_choice_numeric'])
    res_stim_4 = minimize(neg_loglik_reg, v0, args=(df_test, 'stim'), method='L-BFGS-B')
    res_act_4 = minimize(neg_loglik_reg, v0, args=(df_test, 'act'), method='L-BFGS-B')
    theta_stim_4 = unpack_params(res_stim_4.x)
    theta_act_4 = unpack_params(res_act_4.x)
    print(theta_stim_4)
    print(theta_act_4)
    return df_test, theta_act_4, theta_stim_4


@app.cell
def _(df_test, session_loglik_kernel, theta_act_4, theta_stim_4):
    ll_stim = session_loglik_kernel(theta_stim_4, df_test, prior_type='stim')
    ll_act = session_loglik_kernel(theta_act_4, df_test, prior_type='act')
    print('Log-likelihood with stimulus history prior: {:.2f}'.format(ll_stim))
    print('Log-likelihood with action history prior: {:.2f}'.format(ll_act))
    return


@app.cell
def _(
    df_test,
    minimize,
    neg_loglik_reg,
    np,
    session_loglik_kernel,
    unpack_params,
    v0,
):
    sessions = df_test['session'].unique()
    stim_scores = []
    act_scores = []
    for heldout in sessions:
        train_df = df_test[df_test['session'] != heldout].copy()
        test_df = df_test[df_test['session'] == heldout].copy()
        res_stim_5 = minimize(neg_loglik_reg, v0, args=(train_df, 'stim'), method='L-BFGS-B')
        res_act_5 = minimize(neg_loglik_reg, v0, args=(train_df, 'act'), method='L-BFGS-B')
        theta_stim_5 = unpack_params(res_stim_5.x)
        theta_act_5 = unpack_params(res_act_5.x)
        ll_stim_1 = session_loglik_kernel(theta_stim_5, test_df, prior_type='stim')
        ll_act_1 = session_loglik_kernel(theta_act_5, test_df, prior_type='act')
        stim_scores.append(ll_stim_1)
        act_scores.append(ll_act_1)
    print('stim total loglik:', np.sum(stim_scores))
    print('act total loglik :', np.sum(act_scores))
    print('stim - act       :', np.sum(stim_scores) - np.sum(act_scores))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## put different stims in the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## find the optimal parameters of time kernel
    """)
    return


@app.cell
def _(df_dic_hard_vis, utils):
    comb_dict = utils.verify_params_time_kernel(dic = df_dic_hard_vis, y='first_choice_numeric')
    sorted_items = sorted(comb_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    sorted_items[:5]
    return


@app.cell
def _(df_dic_hard_vis, utils):
    comb_dict_vis_correct = utils.verify_params_time_kernel(dic = df_dic_hard_vis, y='correct_numeric', max_lag_range=range(1,100), tau_range=range(1,2))
    sorted_items_vis_correct = sorted(comb_dict_vis_correct.items(), key=lambda item: abs(item[1]), reverse=True)
    sorted_items_vis_correct[:5]
    return (comb_dict_vis_correct,)


@app.cell
def _(df_dic_hard_aud, utils):
    comb_dict_1 = utils.verify_params_time_kernel(dic=df_dic_hard_aud, y='first_choice_numeric')
    sorted_items_1 = sorted(comb_dict_1.items(), key=lambda item: abs(item[1]), reverse=True)
    sorted_items_1[:5]
    return


@app.cell
def _(df_dic_hard_aud, utils):
    comb_dict_aud_correct = utils.verify_params_time_kernel(dic = df_dic_hard_aud, y='correct_numeric', max_lag_range=range(1,100), tau_range=range(1,2))
    sorted_items_aud_correct = sorted(comb_dict_aud_correct.items(), key=lambda item: abs(item[1]), reverse=True)
    sorted_items_aud_correct[:5]
    return (comb_dict_aud_correct,)


@app.cell
def _(comb_dict_aud_correct, comb_dict_vis_correct, plt):
    x_vis, y_vis = zip(*sorted(comb_dict_vis_correct.items()))
    x_aud, y_aud = zip(*sorted(comb_dict_aud_correct.items()))

    x_peak_vis = max(comb_dict_vis_correct, key=comb_dict_vis_correct.get)
    y_peak_vis = comb_dict_vis_correct[x_peak_vis]

    x_peak_aud = max(comb_dict_aud_correct, key=comb_dict_aud_correct.get)
    y_peak_aud = comb_dict_aud_correct[x_peak_aud]

    plt.figure(figsize=(7,5))
    plt.plot([x[0] for x in x_vis], y_vis, color="#D4361D", label='Visual', linewidth=2)
    plt.scatter(x_peak_vis[0], y_peak_vis, color='#D4361D', s=50, zorder=3)

    plt.plot([x[0] for x in x_aud], y_aud, color="#875215", label='Auditory', linewidth=2)
    plt.scatter(x_peak_aud[0], y_peak_aud, color='#875215', s=50, zorder=3)

    plt.xlabel('max_lag', fontsize=12)
    plt.ylabel('fitting_beta', fontsize=12)
    plt.title('Effect of Max Lag on Fitted β in Correct-Choice Model', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return x_peak_aud, x_peak_vis


@app.cell
def _(df_dic_hard_aud, df_dic_hard_vis, dft, pd, x_peak_aud, x_peak_vis):
    max_lag_vis_correct = x_peak_vis[0]
    max_lag_aud_correct = x_peak_aud[0]
    df_test_visual_2 = pd.DataFrame()
    for mouse_14 in df_dic_hard_vis:
        df_dic_hard_vis[mouse_14] = dft.get_time_kernel_impact(df_dic_hard_vis[mouse_14], y='correct_numeric', max_lag=max_lag_vis_correct, tau=1)
        df_test_visual_2 = pd.concat([df_test_visual_2, df_dic_hard_vis[mouse_14]], ignore_index=True)
    df_test_auditory_2 = pd.DataFrame()
    for mouse_14 in df_dic_hard_aud:
        df_dic_hard_aud[mouse_14] = dft.get_time_kernel_impact(df_dic_hard_aud[mouse_14], y='correct_numeric', max_lag=max_lag_aud_correct, tau=1)
        df_test_auditory_2 = pd.concat([df_test_auditory_2, df_dic_hard_aud[mouse_14]], ignore_index=True)
    return max_lag_aud_correct, max_lag_vis_correct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## get performance
    """)
    return


@app.cell
def _(df_dic_aud, df_dic_vis, dft, pd):
    df_test_visual_3 = pd.DataFrame()
    df_test_auditory_3 = pd.DataFrame()
    for mouse_15 in df_dic_vis:
        for session_4 in df_dic_vis[mouse_15]['session'].unique():
            df_mouse_session_4 = df_dic_vis[mouse_15][df_dic_vis[mouse_15]['session'] == session_4]
            df_mouse_session_4 = dft.get_performance_through_trials(df_mouse_session_4)
            df_dic_vis[mouse_15].loc[df_mouse_session_4.index, 'performance_w'] = df_mouse_session_4['performance_w'].values
        df_test_visual_3 = pd.concat([df_test_visual_3, df_dic_vis[mouse_15]], ignore_index=True)
    for mouse_15 in df_dic_aud:
        for session_4 in df_dic_aud[mouse_15]['session'].unique():
            df_mouse_session_4 = df_dic_aud[mouse_15][df_dic_aud[mouse_15]['session'] == session_4]
            df_mouse_session_4 = dft.get_performance_through_trials(df_mouse_session_4)
            df_dic_aud[mouse_15].loc[df_mouse_session_4.index, 'performance_w'] = df_mouse_session_4['performance_w'].values
        df_test_auditory_3 = pd.concat([df_test_auditory_3, df_dic_aud[mouse_15]], ignore_index=True)
    return df_test_auditory_3, df_test_visual_3


@app.cell
def _(df_dic_hard_multi_aud, df_dic_hard_multi_vis, dft, pd):
    df_test_visual_multi_1 = pd.DataFrame()
    df_test_auditory_multi_1 = pd.DataFrame()
    for mouse_16 in df_dic_hard_multi_vis:
        for session_5 in df_dic_hard_multi_vis[mouse_16]['session'].unique():
            df_mouse_session_5 = df_dic_hard_multi_vis[mouse_16][df_dic_hard_multi_vis[mouse_16]['session'] == session_5]
            df_mouse_session_5 = dft.get_performance_through_trials(df_mouse_session_5)
            df_dic_hard_multi_vis[mouse_16].loc[df_mouse_session_5.index, 'performance_w'] = df_mouse_session_5['performance_w'].values
        df_test_visual_multi_1 = pd.concat([df_test_visual_multi_1, df_dic_hard_multi_vis[mouse_16]], ignore_index=True)
    for mouse_16 in df_dic_hard_multi_aud:
        for session_5 in df_dic_hard_multi_aud[mouse_16]['session'].unique():
            df_mouse_session_5 = df_dic_hard_multi_aud[mouse_16][df_dic_hard_multi_aud[mouse_16]['session'] == session_5]
            df_mouse_session_5 = dft.get_performance_through_trials(df_mouse_session_5)
            df_dic_hard_multi_aud[mouse_16].loc[df_mouse_session_5.index, 'performance_w'] = df_mouse_session_5['performance_w'].values
        df_test_auditory_multi_1 = pd.concat([df_test_auditory_multi_1, df_dic_hard_multi_aud[mouse_16]], ignore_index=True)
    return df_test_auditory_multi_1, df_test_visual_multi_1


@app.cell
def _(sns):
    colors_3 = sns.color_palette('viridis', 10)
    return (colors_3,)


@app.cell
def _(colors_3, df_dic_hard_multi_vis, plt):
    for (n, mouse_17) in enumerate(df_dic_hard_multi_vis):
        for session_6 in df_dic_hard_multi_vis[mouse_17]['session'].unique():
            df_mouse_session_6 = df_dic_hard_multi_vis[mouse_17][df_dic_hard_multi_vis[mouse_17]['session'] == session_6]
            plt.scatter(df_mouse_session_6['difficulty'], df_mouse_session_6['performance_w'], color=colors_3[n], alpha=0.3, label=mouse_17 if session_6 == df_dic_hard_multi_vis[mouse_17]['session'].unique()[0] else None)  # df_mouse_session.groupby('difficulty')['performance_w'].mean().plot.scatter(color=colors[n], alpha=0.3)
    return


@app.cell
def _(sns):
    colors_4 = sns.color_palette('inferno', 10)
    return (colors_4,)


@app.cell
def _(colors_4, df_dic_hard_multi_aud):
    for (n_1, mouse_18) in enumerate(df_dic_hard_multi_aud):
        for session_7 in df_dic_hard_multi_aud[mouse_18]['session'].unique():
            df_mouse_session_7 = df_dic_hard_multi_aud[mouse_18][df_dic_hard_multi_aud[mouse_18]['session'] == session_7]
            df_mouse_session_7.groupby('difficulty')['performance_w'].mean().plot(color=colors_4[n_1], alpha=0.3)
    return


@app.cell
def _(df_test_auditory_multi_1, df_test_visual_multi_1):
    print(df_test_visual_multi_1.groupby('difficulty')['performance_w'].mean())
    print(df_test_auditory_multi_1.groupby('difficulty')['performance_w'].mean())
    return


@app.cell
def _(df_test_visual_multi_1):
    df_test_visual_multi_1.groupby('difficulty')['performance_w'].mean().plot(color='#D4361D')
    return


@app.cell
def _(df_test_visual_multi_1, plt):
    plt.scatter(df_test_visual_multi_1.groupby('difficulty')['performance_w'].mean())
    return


@app.cell
def _(df_test_auditory_3, df_test_visual_3, plt):
    plt.figure(figsize=(8, 5))
    zoomin_index = slice(6000, 12000)
    plt.plot(df_test_visual_3['performance_w'].loc[zoomin_index], label='Visual', color='#D4361D', alpha=0.5)
    plt.plot(df_test_auditory_3['performance_w'].loc[zoomin_index], label='Auditory', color='#875215', alpha=0.5)
    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Performance (Rolling Mean)', fontsize=12)
    plt.title('Session Performance Over Trials', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HMM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### choice(0/1) as observation
    """)
    return


@app.cell
def _(gradient_cmap, sns):
    sns.set_style('white')
    sns.set_context('talk')
    color_names = ['windows blue', 'amber', 'red', 'faded green', 'dusty purple', 'orange']
    colors_5 = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors_5)
    return cmap, colors_5


@app.cell
def _(np):
    # some animals only have one of the states
    def make_full_perm(sorted_state, K): 
        used = list(sorted_state.astype(int))
        missing = [k for k in range(K) if k not in used]
        return np.array(used + missing, dtype=int)

    return


@app.cell
def _():
    num_states = 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### single session simple HMM
    """)
    return


@app.cell
def _(np, ssm):
    def fit_best_hmm(obs, K, D, n_restarts=20, num_em_iters=100, observations="gaussian"):
        best_hmm, best_ll = None, -np.inf

        for r in range(n_restarts):
            hmm = ssm.HMM(K, D, observations=observations)
            hmm.fit(obs, method="em", num_em_iters=num_em_iters)
            ll = hmm.log_probability(obs)
            if ll > best_ll:
                best_ll = ll
                best_hmm = hmm
        return best_hmm, best_ll

    return (fit_best_hmm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### bernoulli observation
    """)
    return


@app.cell
def _(df_dic_vis, fit_best_hmm, plt):
    vis_session = 326
    num_em_iters = 100
    K = 3  # states number
    D = 1  # each observation is 2-dimensional (correctness 0/1
    mouse_19 = 'NUO009'
    (fig_12, ax_8) = plt.subplots(1, 1, figsize=(4, 4))
    df_mouse_session_8 = df_dic_vis[mouse_19][df_dic_vis[mouse_19]['session'] == vis_session]
    obs_vis = df_mouse_session_8['first_choice_numeric'].to_numpy().astype(int).reshape(-1, 1)
    # fit HMM with multiple restarts to avoid local minima
    (hmm_vis, hmm_lls_vis) = fit_best_hmm(obs_vis, K, D, num_em_iters=num_em_iters, observations='bernoulli')
    hmm_z_vis_unorder = hmm_vis.most_likely_states(obs_vis)
    # order by mean performance of each state
    sorted_state = df_mouse_session_8.groupby(hmm_z_vis_unorder)['performance_w'].mean().sort_values().index.to_numpy()
    hmm_vis.permute(sorted_state)
    hmm_z_vis = hmm_vis.most_likely_states(obs_vis)
    df_mouse_session_8['hmm_states'] = hmm_z_vis
    transition_mat_vis = hmm_vis.transitions.transition_matrix
    im = ax_8.imshow(transition_mat_vis, vmin=0, vmax=1, aspect='auto', cmap='gray')
    ax_8.set_title(f'Visual: {(mouse_19, vis_session)}', fontsize=8)
    plt.tight_layout()
    plt.show()
    return df_mouse_session_8, mouse_19, vis_session


@app.cell
def _(cmap, colors_5, df_mouse_session_8, mouse_19, plt, vis_session):
    (fig_13, ax_9) = plt.subplots(1, 1, figsize=(6, 4))
    lim_vis = 1.05 * df_mouse_session_8['performance_w'].max()
    ax_9.imshow(df_mouse_session_8['hmm_states'].to_numpy()[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors_5) - 1, extent=[df_mouse_session_8.index.min(), df_mouse_session_8.index.max(), 0, lim_vis])
    ax_9.plot(df_mouse_session_8['performance_w'], '-k')
    ax_9.set_title(f'Visual: {(mouse_19, vis_session)}', fontsize=8)
    ax_9.tick_params(axis='x', labelsize=6)
    ax_9.tick_params(axis='y', labelsize=6)
    ax_9.set_ylabel('Performance', fontsize=6)
    ax_9.set_xlabel('Trial', fontsize=6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### input-driven HMM
    """)
    return


@app.cell
def _(df_dic_vis, dft, mouse_19, np, ssm):
    vis_session_1 = df_dic_vis[mouse_19]['session'].unique()[-5:]
    # vis_session = [326]
    mouse_20 = 'NUO009'
    df_mouse_session_9 = df_dic_vis[mouse_20][df_dic_vis[mouse_20]['session'].isin(vis_session_1)]
    df_mouse_session_9 = dft.calculate_time_between_trials_and_reaction_time(df_mouse_session_9)
    df_mouse_session_9 = df_mouse_session_9.dropna(subset=['reaction_time', 'previous_correct_numeric', 'visual_stimulus_diff'])
    # Now create a new HMM and fit it to the data with EM
    N_iters = 100
    num_states_1 = 2  # number of discrete states
    obs_dim = 1  # data dimension
    input_dim = 3  # input dimension
    num_categories = 2  # number of output types/categories
    hmm_vis_1 = ssm.HMM(num_states_1, obs_dim, input_dim, observations='bernoulli', transitions='inputdriven')
    obs_vis_1 = df_mouse_session_9['first_choice_numeric'].to_numpy().astype(int).reshape(-1, 1)
    inpt_vis = np.column_stack([df_mouse_session_9['visual_stimulus_diff'].to_list(), df_mouse_session_9['reaction_time'].to_list(), df_mouse_session_9['previous_correct_numeric'].to_list()])
    # Fit
    hmm_vis_lps = hmm_vis_1.fit(obs_vis_1, input=inpt_vis, method='em', num_iters=N_iters)
    return (
        df_mouse_session_9,
        hmm_vis_1,
        inpt_vis,
        mouse_20,
        obs_vis_1,
        vis_session_1,
    )


@app.cell
def _(
    df_mouse_session_9,
    hmm_vis_1,
    inpt_vis,
    mouse_20,
    obs_vis_1,
    plt,
    vis_session_1,
):
    (fig_14, ax_10) = plt.subplots(1, 1, figsize=(4, 4))
    hmm_z_vis_unorder_1 = hmm_vis_1.most_likely_states(obs_vis_1, input=inpt_vis)
    # order by mean performance of each state
    sorted_state_1 = df_mouse_session_9.groupby(hmm_z_vis_unorder_1)['performance_w'].mean().sort_values().index.to_numpy()
    hmm_vis_1.permute(sorted_state_1)
    hmm_z_vis_1 = hmm_vis_1.most_likely_states(obs_vis_1, input=inpt_vis)
    transition_mat_vis_1 = hmm_vis_1.transitions.transition_matrix
    im_1 = ax_10.imshow(transition_mat_vis_1, vmin=0, vmax=1, aspect='auto', cmap='gray')
    ax_10.set_title(f'Visual: {(mouse_20, vis_session_1)}', fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(hmm_vis_1, inpt_vis, np, obs_vis_1):
    (Ez, _, _) = hmm_vis_1.expected_states(obs_vis_1, input=inpt_vis)
    z_soft = np.full(len(Ez), -1)  # -1 = uncertain
    z_soft[Ez[:, 0] > 0.8] = 0
    z_soft[Ez[:, 1] > 0.8] = 1
    return Ez, z_soft


@app.cell
def _(
    cmap,
    colors_5,
    df_mouse_session_9,
    mouse_20,
    plt,
    vis_session_1,
    z_soft,
):
    (fig_15, ax_11) = plt.subplots(1, 1, figsize=(6, 4))
    lim_vis_1 = 1.05 * df_mouse_session_9['performance_w'].max()
    ax_11.imshow(z_soft[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors_5) - 1, extent=[df_mouse_session_9.index.min(), df_mouse_session_9.index.max(), 0, lim_vis_1])
    ax_11.plot(df_mouse_session_9['performance_w'], '-k')
    ax_11.set_title(f'Visual: {(mouse_20, vis_session_1)}', fontsize=8)
    ax_11.tick_params(axis='x', labelsize=6)
    ax_11.tick_params(axis='y', labelsize=6)
    ax_11.set_ylabel('Performance', fontsize=6)
    ax_11.set_xlabel('Trial', fontsize=6)
    return


@app.cell
def _(Ez, mouse_20, plt, vis_session_1):
    (fig_16, ax_12) = plt.subplots(1, 1, figsize=(10, 2))
    im_2 = ax_12.imshow(Ez.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax_12.set_yticks(range(Ez.shape[1]))
    ax_12.set_yticklabels([f'State {k}' for k in range(Ez.shape[1])])
    ax_12.set_xlabel('Trial')
    ax_12.set_title(f'Posterior state probabilities (Ez)\n{mouse_20}, session {vis_session_1}', fontsize=8)
    cbar_3 = plt.colorbar(im_2, ax=ax_12)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### input-driven HMM more sessions
    """)
    return


@app.cell
def _(df_dic_vis, dft, np, ssm):
    vis_session_2 = 326
    mouse_21 = 'NUO009'
    df_mouse_session_10 = df_dic_vis[mouse_21][df_dic_vis[mouse_21]['session'].isin(df_dic_vis[mouse_21]['session'].unique()[:10])]
    df_mouse_session_10 = dft.calculate_time_between_trials_and_reaction_time(df_mouse_session_10)
    df_mouse_session_10 = df_mouse_session_10.dropna(subset=['reaction_time', 'previous_correct_numeric', 'visual_stimulus_diff'])
    # Now create a new HMM and fit it to the data with EM
    N_iters_1 = 100
    num_states_2 = 2  # number of discrete states
    obs_dim_1 = 1  # data dimension
    input_dim_1 = 3  # input dimension
    num_categories_1 = 2  # number of output types/categories
    hmm_vis_2 = ssm.HMM(num_states_2, obs_dim_1, input_dim_1, observations='bernoulli', transitions='inputdriven')
    obs_vis_2 = df_mouse_session_10['first_choice_numeric'].to_numpy().astype(int).reshape(-1, 1)
    inpt_vis_1 = np.column_stack([df_mouse_session_10['visual_stimulus_diff'].to_list(), df_mouse_session_10['reaction_time'].to_list(), df_mouse_session_10['previous_correct_numeric'].to_list()])
    # Fit
    hmm_vis_lps_1 = hmm_vis_2.fit(obs_vis_2, input=inpt_vis_1, method='em', num_iters=N_iters_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### gaussian observation
    """)
    return


@app.cell
def _(df_dic_vis, fit_best_hmm, np, plt):
    vis_session_3 = 326
    aud_session = 290
    num_em_iters_1 = 100
    K_1 = 2  # states number
    D_1 = 1  # each observation is 2-dimensional (correctness 0/1
    mouse_22 = 'NUO009'
    (fig_17, ax_13) = plt.subplots(1, 1, figsize=(4, 4))
    df_mouse_session_11 = df_dic_vis[mouse_22][df_dic_vis[mouse_22]['session'] == vis_session_3]
    # df_mouse_session = dft.calculate_time_between_trials_and_reaction_time(df_mouse_session)
    # df_mouse_session = df_mouse_session.dropna(subset=['reaction_time'])
    obs_vis_3 = np.column_stack([df_mouse_session_11['first_choice_numeric'].to_list()])
    (hmm_vis_3, hmm_lls_vis_1) = fit_best_hmm(obs_vis_3, K_1, D_1, num_em_iters=num_em_iters_1)
    hmm_z_vis_unorder_2 = hmm_vis_3.most_likely_states(obs_vis_3)
    # # normalize the observations, or the HMM will just use the reaction time to separate states
    # obs_vis = (obs_vis - obs_vis.mean(axis=0)) / obs_vis.std(axis=0)
    # fit HMM with multiple restarts to avoid local minima
    sorted_state_2 = df_mouse_session_11.groupby(hmm_z_vis_unorder_2)['performance_w'].mean().sort_values().index.to_numpy()
    hmm_vis_3.permute(sorted_state_2)
    # order by mean performance of each state
    hmm_z_vis_2 = hmm_vis_3.most_likely_states(obs_vis_3)
    df_mouse_session_11['hmm_states'] = hmm_z_vis_2
    transition_mat_vis_2 = hmm_vis_3.transitions.transition_matrix
    im_3 = ax_13.imshow(transition_mat_vis_2, vmin=0, vmax=1, aspect='auto', cmap='gray')
    ax_13.set_title(f'Visual: {(mouse_22, vis_session_3)}', fontsize=8)
    plt.tight_layout()
    # df_mouse_session = df_dic_aud[mouse][df_dic_aud[mouse]['session'] == aud_session]
    # obs_aud = np.column_stack([
    #     df_mouse_session['first_choice_numeric'].to_list(),
    #     # df_mouse_session['reaction_time'].to_list(),
    #     ])
    # obs_aud = (obs_aud - obs_aud.mean(axis=0)) / obs_aud.std(axis=0)
    # # fit HMM with multiple restarts to avoid local minima
    # hmm_aud, hmm_lls_aud = fit_best_hmm(obs_aud, K, D, num_em_iters=num_em_iters)
    # hmm_z_aud_unorder = hmm_aud.most_likely_states(obs_aud)
    # # order by mean performance of each state
    # sorted_state = df_mouse_session.groupby(hmm_z_aud_unorder)['performance_w'].mean().sort_values().index.to_numpy()
    # hmm_aud.permute(sorted_state)
    # hmm_z_aud = hmm_aud.most_likely_states(obs_aud)
    # df_mouse_session['hmm_states'] = hmm_z_aud 
    # transition_mat_aud = hmm_aud.transitions.transition_matrix
    # im = ax[1].imshow(transition_mat_aud, vmin=0, vmax=1, aspect="auto", cmap="gray")
    # ax[1].set_title(f'Auditory: {mouse, aud_session}', fontsize=8)
    plt.show()
    return df_mouse_session_11, mouse_22, vis_session_3


@app.cell
def _(cmap, colors_5, df_mouse_session_11, mouse_22, plt, vis_session_3):
    (fig_18, ax_14) = plt.subplots(1, 1, figsize=(6, 4))
    lim_vis_2 = 1.05 * df_mouse_session_11['performance_w'].max()
    ax_14.imshow(df_mouse_session_11['hmm_states'].to_numpy()[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=len(colors_5) - 1, extent=[df_mouse_session_11.index.min(), df_mouse_session_11.index.max(), 0, lim_vis_2])
    ax_14.plot(df_mouse_session_11['performance_w'], '-k')
    ax_14.set_title(f'Visual: {(mouse_22, vis_session_3)}', fontsize=8)
    ax_14.tick_params(axis='x', labelsize=6)
    ax_14.tick_params(axis='y', labelsize=6)
    ax_14.set_ylabel('Performance', fontsize=6)
    ax_14.set_xlabel('Trial', fontsize=6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GLM-HMM
    """)
    return


@app.cell
def _(gradient_cmap, sns):
    sns.set_style('white')
    sns.set_context('talk')
    color_names_1 = ['windows blue', 'amber', 'red', 'faded green', 'dusty purple', 'orange']
    colors_6 = sns.xkcd_palette(color_names_1)
    cmap_1 = gradient_cmap(colors_6)
    return cmap_1, colors_6


@app.cell
def _(np, ssm):
    def fit_best_glmhmm(obs, 
                        inputs,
                        num_states = 3,        # number of discrete states
                        obs_dim = 1,           # number of observed dimensions
                        num_categories = 2,    # number of categories for output
                        input_dim = 2,         # input dimensions 
                        n_restarts = 10,        # number of EM iterations
                        N_iters = 100,         # number of EM iterations
                        prior_sigma=2,
                        prior_alpha=2,
                        kappa=1
                        ):

        best_hmm, best_ll = None, -np.inf

        for r in range(n_restarts):
            map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                 observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                 transitions="sticky", 
                 transition_kwargs=dict(alpha=prior_alpha,kappa=kappa),
                 )
            hmm_lls = map_glmhmm.fit(obs, inputs=inputs, method="em", num_iters=N_iters, tolerance=10**-4)
            ll = map_glmhmm.log_probability(obs, inputs=inputs)
            if ll > best_ll:
                best_ll = ll
                best_hmm = map_glmhmm
                best_hmm_lls = hmm_lls
        return best_hmm, best_ll, best_hmm_lls

    return (fit_best_glmhmm,)


@app.cell
def _(ssm):
    def fit_glmhmm(obs, 
                    inputs,
                    num_states = 3,        # number of discrete states
                    obs_dim = 1,           # number of observed dimensions
                    num_categories = 2,    # number of categories for output
                    input_dim = 2,         # input dimensions 
                    N_iters = 100,         # number of EM iterations
                    prior_sigma=2,
                    prior_alpha=2,
                    kappa=1
                    ):

        map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                transitions="sticky", 
                transition_kwargs=dict(alpha=prior_alpha,kappa=kappa),
                )
        hmm_lls = map_glmhmm.fit(obs, inputs=inputs, method="em", num_iters=N_iters, tolerance=10**-4)
        ll = map_glmhmm.log_probability(obs, inputs=inputs)
        return map_glmhmm, ll, hmm_lls

    return (fit_glmhmm,)


@app.cell
def _(np, pd):
    def build_glmhmm_inputs_by_session(
        df: pd.DataFrame,
        y_col="first_choice_numeric",
        stim_col=["visual_stimulus_ratio"],
    ):
        """
        Returns:
            datas: list of (T_i, 1)
            inpts: list of (T_i, M)
        """
        datas, inpts = [], []

        stim_col = [c for c in stim_col if c != "bias"]

        for sess, df_sess in df.groupby('session'):
            df_sess = df_sess[stim_col + [y_col]].dropna()

            y = df_sess[y_col].to_numpy(dtype=int).reshape(-1, 1)
            X = df_sess[stim_col].to_numpy(dtype=float)

            if "bias" in stim_col:
                X = np.column_stack([X, np.ones(len(X))])

            datas.append(y)
            inpts.append(X)

        return datas, inpts

    return (build_glmhmm_inputs_by_session,)


@app.cell
def _(plt):
    def plot_transition_matrix(map_glmhmm, title="", ax=None, cmap="gray", fontsize=8):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        A = map_glmhmm.transitions.transition_matrix
        im = ax.imshow(A, vmin=0, vmax=1, aspect="auto", cmap=cmap)

        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel("State t")
        ax.set_ylabel("State t-1")

        K = A.shape[0]
        for i in range(K):
            for j in range(K):
                val = A[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if val > 0.5 else "white",
                        fontsize=fontsize)

        plt.tight_layout()
        return ax, im

    return (plot_transition_matrix,)


@app.cell
def _(np, pd):
    def get_posterior_probs_concat(map_glmhmm, datas, inpts) -> np.ndarray:
        """Return posterior_probs_con with shape (N_total, K)."""
        post_prob_list = [map_glmhmm.expected_states(data=d, input=x)[0] for d, x in zip(datas, inpts)]
        return post_prob_list

    def add_df_glmhmm_state(df: pd.DataFrame, post_prob_list: list[np.ndarray]):
        """MAP state per time point."""
        posterior_probs_con = np.concatenate(post_prob_list, axis=0)
        df['glmhmm_state'] = np.argmax(posterior_probs_con, axis=1)
        return df

    return add_df_glmhmm_state, get_posterior_probs_concat


@app.cell
def _(np, pd, plt):
    def plot_posteriors_with_performance(df: pd.DataFrame, post_prob_list: list[np.ndarray], colors, ax=None, lw=2, perf_scale=0.01):
        posterior_probs_con = np.concatenate(post_prob_list, axis=0)
        K = posterior_probs_con.shape[1]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 2.5), dpi=80)

        for k in range(K):
            ax.plot(posterior_probs_con[:, k], label=f"State {k+1}", lw=lw, color=colors[k])

        ax.plot(np.asarray(df['performance_w']) * perf_scale, "-k", label="Performance", alpha=0.7)

        ax.set_ylim(-0.01, 1.01)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("trial #")
        ax.set_ylabel("p(state)")
        ax.legend(frameon=False)
        return ax

    return (plot_posteriors_with_performance,)


@app.cell
def _(colors_6, dft, pd, plt, utils):
    def plot_param_by_state(hmm, df: pd.DataFrame, X, y, state_col='glmhmm_state', marker='o', linestyle='-', ax=None):
        if ax is None:
            (fig, ax) = plt.subplots(1, 1, figsize=(10, 6))
        df_param_states = pd.DataFrame([])
        for state in range(hmm.K):
            df_session_state = df[df[state_col] == state]
            df_session_state = dft.calculate_time_between_trials_and_reaction_time(df_session_state)
            if len(df_session_state) > 1:
                (_, logit_model_multi) = utils.logi_model_fit(df_session_state, X=X, y=y)
                df_param_states[state] = logit_model_multi.params
        df_param_states.drop('const', inplace=True)
        for state in df_param_states.columns:
            ax.plot(df_param_states.index, df_param_states[state], marker=marker, linestyle=linestyle, color=colors_6[state], label=f'State {state}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Weights')
        ax.legend(title='State', frameon=False)
        return ax

    return (plot_param_by_state,)


@app.cell
def _(plots, plt):
    def plot_psychometric_by_state(df, map_glmhmm, colors, x="visual_stimulus_ratio", y="first_choice_numeric", state_col="glmhmm_state", ax=None, valueType = "discrete", 
                                   point_kwargs=None, line_kwargs=None, log = True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for state in range(map_glmhmm.K):
            df_s = df[df[state_col] == state].copy()
            if len(df_s) == 0:
                continue
            plots.psychometric_plot(
                df_s, x=x, y=y, ax=ax, valueType=valueType, log=log,
                point_kwargs={"color": colors[state], "label": "", **(point_kwargs or {})},
                line_kwargs={"color": colors[state], "label": {"state " + str(state)}, **(line_kwargs or {})}
            )
        ax.legend(frameon=False)
        return ax

    return (plot_psychometric_by_state,)


app._unparsable_cell(
    r"""
    for session in df_test_aud_easy['session'].unique():
            df_mouse_session = df_test_aud_easy[df_test_aud_easy['session'] == session]
            df_mouse_session = dft.get_performance_through_trials(df_mouse_session)
            df_test_aud_easy-.loc[df_mouse_session.index, 'performance_w'] = df_mouse_session['performance_w'].values
    """,
    name="_"
)


@app.cell
def _(df_test_aud_hard):
    df_test_aud_hard['difficulty_numeric'] = df_test_aud_hard['difficulty'].map({'easy': 3, 'medium': 2, 'hard': 1})
    df_test_aud_hard['difficulty_numeric'] = df_test_aud_hard.apply(
                lambda row: row["difficulty_numeric"] if row['correct_side'] == 'left' else -row["difficulty_numeric"],
                axis=1
            )
    return


@app.cell
def _(df_test_aud_hard):
    # sample_sessions = []
    # for session in df_test_multi_vis['session'].unique():
    #     if (df_test_multi_vis[df_test_multi_vis['session'] == session]['performance_w'].mean()>70):
    #         sample_sessions.append(session)
    #     else:
    #         pass
    sample_sessions = df_test_aud_hard['session'].unique()[-200:]
    sample_sessions
    return (sample_sessions,)


@app.cell
def _(build_glmhmm_inputs_by_session, df_test_aud_hard, fit_glmhmm, plt):
    datas, inpts = build_glmhmm_inputs_by_session(df_test_aud_hard, y_col="first_choice_numeric", stim_col=["difficulty_numeric", "bias", ])
    map_glmhmm, ll, hmm_lls = fit_glmhmm(datas, inpts, num_states=3, obs_dim=1, input_dim=1, num_categories=2, N_iters=150)
    plt.plot(hmm_lls)
    return


@app.cell
def _(alpha, df_test_aud_hard, recursive_kernel_prior):
    recursive_kernel_prior(df_test_aud_hard[''], alpha=alpha, pi0=0.5)
    return


app._unparsable_cell(
    r"""
    res_stim = minimize(neg_loglik, v0, args=(train_df, "stim"), method="L-BFGS-B")
        res_act  = minimize(neg_loglik, v0, args=(train_df, "act"), method="L-BFGS-B")

        theta_stim = unpack_params(res_stim.x)
        theta_act  = unpack_params(res_act.x)
    """,
    name="_"
)


@app.cell
def _(
    add_df_glmhmm_state,
    build_glmhmm_inputs_by_session,
    fit_glmhmm,
    get_posterior_probs_concat,
):
    def held_out_session_glmhmm_ll(df, heldout_session, colors):
        train_df = df[df["session"] != heldout_session].copy()
        test_df  = df[df["session"] == heldout_session].copy()

        datas, inpts = build_glmhmm_inputs_by_session(train_df, y_col="first_choice_numeric", stim_col=["difficulty_numeric"])
        map_glmhmm, ll, hmm_lls = fit_glmhmm(datas, inpts, num_states=3, obs_dim=1, input_dim=1, num_categories=2, N_iters=100)

        test_datas, test_inpts = build_glmhmm_inputs_by_session(test_df, y_col="first_choice_numeric", stim_col=["difficulty_numeric"])
        post_prob_list = get_posterior_probs_concat(map_glmhmm, test_datas, test_inpts)
        test_df = add_df_glmhmm_state(test_df, post_prob_list)

    return


@app.cell
def _(
    build_glmhmm_inputs_by_session,
    df_test_multi_vis,
    fit_glmhmm,
    sample_sessions,
):
    df_test_multi_vis_copy = df_test_multi_vis.copy()
    df_test_multi_vis_copy.dropna(subset=['difficulty_numeric', 'first_choice_numeric', 'previous_first_choice_numeric'], inplace=True)
    df_test_multi_session = df_test_multi_vis_copy[df_test_multi_vis_copy['session'].isin(sample_sessions)]
    (datas_1, inpts_1) = build_glmhmm_inputs_by_session(df_test_multi_session, y_col='first_choice_numeric', stim_col=['visual_stimulus_ratio', 'previous_first_choice_numeric'])
    # Fit GLM-HMM with MAP estimation:
    (map_glmhmm_1, ll_1, hmm_lls_1) = fit_glmhmm(datas_1, inpts_1, num_states=3, obs_dim=1, input_dim=2, kappa=15, prior_sigma=2)
    return datas_1, df_test_multi_session, inpts_1, map_glmhmm_1


@app.cell
def _(
    add_df_glmhmm_state,
    colors_6,
    datas_1,
    df_test_multi_session,
    get_posterior_probs_concat,
    inpts_1,
    map_glmhmm_1,
    plot_param_by_state,
    plot_posteriors_with_performance,
    plot_psychometric_by_state,
    plot_transition_matrix,
    plt,
):
    from matplotlib.gridspec import GridSpec
    fig_19 = plt.figure(figsize=(10, 16), dpi=300)
    # ===== 创建大 figure 和 layout =====
    gs = GridSpec(nrows=3, ncols=2, figure=fig_19, height_ratios=[1.0, 0.7, 1.2], width_ratios=[1.0, 1.0])
    ax_trans = fig_19.add_subplot(gs[0, 0])
    ax_param = fig_19.add_subplot(gs[0, 1])
    ax_post = fig_19.add_subplot(gs[1, :])  # 第二行是长条，扁一点
    ax_psy = fig_19.add_subplot(gs[2, 0])
    plot_transition_matrix(map_glmhmm_1, title='', ax=ax_trans, cmap='gray', fontsize=8)
    post_prob_list = get_posterior_probs_concat(map_glmhmm_1, datas_1, inpts_1)
    df_test_multi_session_1 = add_df_glmhmm_state(df_test_multi_session, post_prob_list)  # 转移矩阵
    plot_posteriors_with_performance(df_test_multi_session_1, post_prob_list, colors_6, ax=ax_post)  # 参数
    ax_post.set_title('Posterior p(state) + performance', fontsize=10)  # posterior（跨两列）
    plot_param_by_state(map_glmhmm_1, df_test_multi_session_1, X=['difficulty_numeric', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric', ax=ax_param)  # psychometric（跨两列）
    ax_param.set_title('Logistic parameters by state', fontsize=10)
    # ===== 1. Transition matrix =====
    plot_psychometric_by_state(df_test_multi_session_1, map_glmhmm_1, colors_6, x='difficulty_numeric', y='first_choice_numeric', ax=ax_psy, log=False)
    ax_psy.set_title('Psychometric by state', fontsize=10)
    # ===== 2. Posterior probabilities（长条）=====
    plt.tight_layout()
    # ===== 3. Param by state =====
    # ===== 4. Psychometric curves =====
    plt.show()
    return (GridSpec,)


@app.cell
def _(
    build_glmhmm_inputs_by_session,
    df_test_multi,
    fit_glmhmm,
    plt,
    sample_sessions,
):
    # find the optimal number of states by comparing the log-likelihood of models with different number of states
    sessions_1 = df_test_multi['session'].unique()[-1]
    df_test_multi_session_2 = df_test_multi[df_test_multi['session'].isin(sample_sessions)]
    (datas_2, inpts_2) = build_glmhmm_inputs_by_session(df_test_multi_session_2, y_col='first_choice_numeric', stim_col=['difficulty_numeric', 'bias', 'previous_first_choice_numeric'])
    state_range = range(2, 6)
    model_results = []
    for num_states_3 in state_range:
        (map_glmhmm_2, ll_2, hmm_lls_2) = fit_glmhmm(datas_2, inpts_2, num_states=num_states_3, obs_dim=1, input_dim=2, kappa=15, prior_sigma=2)
        model_results.append((num_states_3, ll_2))
    plt.figure(figsize=(6, 4))
    plt.plot([r[0] for r in model_results], [r[1] for r in model_results], marker='o')
    plt.xlabel('Number of states')
    plt.ylabel('Log-likelihood')
    # plot the log-likelihood for different number of states
    plt.title('Model selection for GLM-HMM')
    plt.xticks(state_range)
    plt.show()
    return df_test_multi_session_2, map_glmhmm_2, num_states_3


@app.cell
def _(df_test_multi_session_2):
    df_test_multi_session_vis = df_test_multi_session_2[df_test_multi_session_2['stimulus_modality'] == 'visual'].sample(n=500, random_state=42)
    df_test_multi_session_aud = df_test_multi_session_2[df_test_multi_session_2['stimulus_modality'] == 'auditory'].sample(n=500, random_state=42)
    df_test_multi_session_vis.shape
    return


@app.cell
def _(colors_6, counts, df_test_multi_session_2, np, plt):
    df_9 = df_test_multi_session_2.copy()
    state_col = 'glmhmm_state'
    mod_col = 'stimulus_modality'
    props = counts.div(counts.sum(axis=1), axis=0)
    modalities = counts.columns.tolist()
    palette = {'auditory': '#9bd3fb', 'visual': '#fca355'}
    (fig_20, ax_15) = plt.subplots(figsize=(12, 8), dpi=300)
    x = np.arange(len(counts.index))
    bottom = np.zeros(len(x))
    # 2) 同时算比例（仅用于标注百分比）
    n_2 = 0
    for m in modalities:
    # 3) modality 顺序（按你想显示的顺序）
        h = counts[m].to_numpy()
        bars = ax_15.bar(x, h, width=0.5, bottom=bottom, color=palette.get(m, None), edgecolor=colors_6, linewidth=5, label=m)
    # 4) 颜色（你可以替换成自己论文用的颜色）
        for (i_11, (bar, count_val)) in enumerate(zip(bars, h)):
            if count_val == 0:  # 蓝
                continue  # 橙
            pct = props.loc[counts.index[i_11], m] * 100
            ax_15.text(bar.get_x() + bar.get_width() / 2, bottom[i_11] + count_val / 2, f'{pct:.0f}%', ha='center', va='center', fontsize=10, color='black')
        bottom = bottom + h
    ax_15.set_xticks(x)
    ax_15.set_xticklabels([f'State {s}' for s in counts.index])
    ax_15.set_ylabel('Number of trials')
    ax_15.set_xlabel('GLM-HMM state')
    ax_15.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    # 6) 轴与样式
    fig_20.savefig('/home/kudongdong/data/LeciLab/temporary_plots/glmhmm_multisensory_psychometric_proportion.svg', format='svg')  # 5) 在每个堆叠块中标百分比
    return (m,)


@app.cell
def _(colors_6, counts, df_test_multi_session_2, m, np, plt):
    df_10 = df_test_multi_session_2.copy()
    state_col_1 = 'glmhmm_state'
    mod_col_1 = 'stimulus_modality'
    props_1 = counts.div(counts.sum(axis=1), axis=0)
    modalities_1 = counts.columns.tolist()
    # 2) 同时算比例（仅用于标注百分比）
    palette_1 = {'auditory': '#9bd3fb', 'visual': '#fca355'}
    (fig_21, ax_16) = plt.subplots(figsize=(12, 8), dpi=300)
    # 3) modality 顺序（按你想显示的顺序）
    x_1 = np.arange(len(counts.index))
    w = 0.3
    # 4) 颜色（你可以替换成自己论文用的颜色）
    h_1 = counts[m].to_numpy()
    ax_16.bar(x_1 - w / 2, counts['visual'].to_numpy(), width=w, color=palette_1.get('visual', None), edgecolor=colors_6, linewidth=5, label='visual')  # 蓝
    ax_16.bar(x_1 + w / 2, counts['auditory'].to_numpy(), width=w, color=palette_1.get('auditory', None), edgecolor=colors_6, linewidth=5, label='auditory')  # 橙
    ax_16.set_xticks(x_1)
    ax_16.set_xticklabels([f'State {s}' for s in counts.index])
    ax_16.set_ylabel('Number of trials')
    ax_16.set_xlabel('GLM-HMM state')
    ax_16.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()  # bar width
    plt.show()
    # 6) 轴与样式
    fig_21.savefig('/home/kudongdong/data/LeciLab/temporary_plots/glmhmm_multisensory_psychometric_proportion.svg', format='svg')
    return


@app.cell
def _(df_test_auditory_3, df_test_visual_3):
    for modality in ['visual', 'auditory']:
        df_preright_2 = df_test_visual_3 if modality == 'visual' else df_test_auditory_3
    return


@app.cell
def _(
    colors_6,
    df_test_multi_session_2,
    map_glmhmm_2,
    plot_psychometric_by_state,
    plt,
):
    (fig_22, ax_17) = plt.subplots(1, 1, figsize=(8, 6))
    plot_psychometric_by_state(df_test_multi_session_2, map_glmhmm_2, colors_6, x='difficulty_numeric', y='first_choice_numeric', ax=ax_17, log=False)
    fig_22.savefig('/home/kudongdong/data/LeciLab/temporary_plots/glmhmm_multisensory_psychometric.svg', format='svg')
    return


@app.cell
def _(df_test_multi):
    df_test_multi_vis_1 = df_test_multi[df_test_multi['stimulus_modality'] == 'visual']
    df_test_multi_aud_1 = df_test_multi[df_test_multi['stimulus_modality'] == 'auditory']
    sample_sessions_1 = []
    for session_8 in df_test_multi['session'].unique():
        if (df_test_multi_vis_1[df_test_multi_vis_1['session'] == session_8]['performance_w'].mean() > 75) & (df_test_multi_aud_1[df_test_multi_aud_1['session'] == session_8]['performance_w'].mean() > 75):
            sample_sessions_1.append(session_8)
        else:
            pass
    df_test_multi_vis_1 = df_test_multi_vis_1[df_test_multi_vis_1['session'].isin(sample_sessions_1)]
    df_test_multi_aud_1 = df_test_multi_aud_1[df_test_multi_aud_1['session'].isin(sample_sessions_1)]
    return df_test_multi_aud_1, df_test_multi_vis_1


@app.cell
def _(
    build_glmhmm_inputs_by_session,
    df_test_multi_aud_1,
    df_test_multi_vis_1,
    fit_glmhmm,
):
    (datas_vis, inpts_vis) = build_glmhmm_inputs_by_session(df_test_multi_vis_1, session_col='session', y_col='first_choice_numeric', stim_col='difficulty_numeric')
    (datas_aud, inpts_aud) = build_glmhmm_inputs_by_session(df_test_multi_aud_1, session_col='session', y_col='first_choice_numeric', stim_col='difficulty_numeric')
    (map_glmhmm_vis, ll_vis, hmm_lls_vis_2) = fit_glmhmm(datas_vis, inpts_vis, num_states=3, obs_dim=1, input_dim=2, kappa=20, prior_sigma=10)
    # Fit GLM-HMM with MAP estimation:
    (map_glmhmm_aud, ll_aud, hmm_lls_aud) = fit_glmhmm(datas_aud, inpts_aud, num_states=3, obs_dim=1, input_dim=2, kappa=20, prior_sigma=10)
    return (
        datas_aud,
        datas_vis,
        inpts_aud,
        inpts_vis,
        map_glmhmm_aud,
        map_glmhmm_vis,
    )


@app.cell
def _(
    add_df_glmhmm_state,
    colors_6,
    datas_aud,
    datas_vis,
    df_test_multi_aud_1,
    df_test_multi_vis_1,
    get_posterior_probs_concat,
    inpts_aud,
    inpts_vis,
    map_glmhmm_aud,
    map_glmhmm_vis,
    plot_posteriors_with_performance,
):
    post_prob_list_1 = get_posterior_probs_concat(map_glmhmm_vis, datas_vis, inpts_vis)
    df_test_multi_vis_2 = add_df_glmhmm_state(df_test_multi_vis_1, post_prob_list_1)
    plot_posteriors_with_performance(df_test_multi_vis_2, post_prob_list_1, colors_6, ax=None)
    post_prob_list_1 = get_posterior_probs_concat(map_glmhmm_aud, datas_aud, inpts_aud)
    df_test_multi_aud_2 = add_df_glmhmm_state(df_test_multi_aud_1, post_prob_list_1)
    plot_posteriors_with_performance(df_test_multi_aud_2, post_prob_list_1, colors_6, ax=None)
    return df_test_multi_aud_2, df_test_multi_vis_2


@app.cell
def _(
    df_test_multi_aud_2,
    df_test_multi_vis_2,
    map_glmhmm_aud,
    map_glmhmm_vis,
    plot_param_by_state,
    plt,
):
    (fig_23, ax_18) = plt.subplots(1, 1, figsize=(6, 4))
    plot_param_by_state(map_glmhmm_vis, df_test_multi_vis_2, X=['difficulty_numeric', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric', ax=ax_18, linestyle='-')
    plot_param_by_state(map_glmhmm_aud, df_test_multi_aud_2, X=['difficulty_numeric', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric', ax=ax_18, linestyle='--')
    return


@app.cell
def _(
    colors_6,
    df_test_multi_aud_2,
    df_test_multi_vis_2,
    map_glmhmm_aud,
    map_glmhmm_vis,
    plot_psychometric_by_state,
    plt,
):
    (fig_24, ax_19) = plt.subplots(1, 1, figsize=(6, 4))
    plot_psychometric_by_state(df_test_multi_vis_2, map_glmhmm_vis, colors_6, x='difficulty_numeric', y='first_choice_numeric', ax=ax_19, log=False)
    plot_psychometric_by_state(df_test_multi_aud_2, map_glmhmm_aud, colors_6, x='difficulty_numeric', y='first_choice_numeric', ax=ax_19, log=False, line_kwargs={'linestyle': '--'})
    return


@app.cell
def _(build_glmhmm_inputs_by_session, df_dic_vis, fit_best_glmhmm):
    mouse_23 = 'ACV009'
    df_mouse = df_dic_vis[mouse_23]
    df_mouse.dropna(subset=['visual_stimulus_ratio', 'first_choice_numeric'], inplace=True)
    (datas_vis_1, inpts_vis_1) = build_glmhmm_inputs_by_session(df_mouse, session_col='session', y_col='first_choice_numeric', stim_col='visual_stimulus_ratio')
    # Fit GLM-HMM with MAP estimation:
    (map_glmhmm_3, ll_3, hmm_lls_3) = fit_best_glmhmm(datas_vis_1, inpts_vis_1, num_states=3, obs_dim=1, input_dim=2)
    return datas_vis_1, df_mouse, inpts_vis_1, map_glmhmm_3, mouse_23


@app.cell
def _(map_glmhmm_3, plot_transition_matrix):
    plot_transition_matrix(map_glmhmm_3, title='', ax=None, cmap='gray', fontsize=8)
    return


@app.cell
def _(
    add_df_glmhmm_state,
    colors_6,
    datas_vis_1,
    df_mouse,
    get_posterior_probs_concat,
    inpts_vis_1,
    map_glmhmm_3,
    plot_posteriors_with_performance,
):
    post_prob_list_2 = get_posterior_probs_concat(map_glmhmm_3, datas_vis_1, inpts_vis_1)
    df_mouse_1 = add_df_glmhmm_state(df_mouse, post_prob_list_2)
    plot_posteriors_with_performance(df_mouse_1, post_prob_list_2, colors_6, ax=None)
    return df_mouse_1, post_prob_list_2


@app.cell
def _(df_mouse_1, map_glmhmm_3, plot_param_by_state):
    plot_param_by_state(map_glmhmm_3, df_mouse_1, X=['visual_stimulus_ratio', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric', ax=None)
    return


@app.cell
def _(colors_6, df_mouse_1, map_glmhmm_3, plot_psychometric_by_state):
    plot_psychometric_by_state(df_mouse_1, map_glmhmm_3, colors_6, x='visual_stimulus_ratio', y='first_choice_numeric', ax=None)
    return


@app.cell
def _(
    GridSpec,
    colors_6,
    df_mouse_1,
    map_glmhmm_3,
    mouse_23,
    plot_param_by_state,
    plot_posteriors_with_performance,
    plot_psychometric_by_state,
    plot_transition_matrix,
    plt,
    post_prob_list_2,
):
    fig_25 = plt.figure(figsize=(14, 10), dpi=300)
    gs_1 = GridSpec(nrows=3, ncols=2, figure=fig_25, height_ratios=[1.0, 0.45, 1.2], width_ratios=[1.0, 1.0])
    # ===== 创建大 figure 和 layout =====
    ax_trans_1 = fig_25.add_subplot(gs_1[0, 0])
    ax_param_1 = fig_25.add_subplot(gs_1[0, 1])
    ax_post_1 = fig_25.add_subplot(gs_1[1, :])
    ax_psy_1 = fig_25.add_subplot(gs_1[2, 0])  # 第二行是长条，扁一点
    plot_transition_matrix(map_glmhmm_3, title=f'{mouse_23}', ax=ax_trans_1, cmap='gray', fontsize=8)
    plot_param_by_state(map_glmhmm_3, df_mouse_1, X=['visual_stimulus_ratio', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric', ax=ax_param_1)
    ax_param_1.set_title('Logistic parameters by state', fontsize=10)
    plot_posteriors_with_performance(df_mouse_1, post_prob_list_2, colors_6, ax=ax_post_1)  # 转移矩阵
    ax_post_1.set_title('Posterior p(state) + performance', fontsize=10)  # 参数
    plot_psychometric_by_state(df_mouse_1, map_glmhmm_3, colors_6, x='visual_stimulus_ratio', y='first_choice_numeric', ax=ax_psy_1)  # posterior（跨两列）
    ax_psy_1.set_title('Psychometric by state', fontsize=10)  # psychometric
    plt.tight_layout()
    # ===== 1. Transition matrix =====
    # ===== 2. Param by state =====
    # ===== 3. Posterior probabilities（长条）=====
    # ===== 4. Psychometric curves =====
    plt.show()
    return


@app.cell
def _(
    GridSpec,
    plot_param_by_state,
    plot_posteriors_with_performance,
    plot_psychometric_by_state,
    plot_transition_matrix,
    plt,
):
    def plot_glmhmm_pipeline_figure(map_glmhmm, df_mouse, post_prob_list, colors, X, y='first_choice_numeric', stim_x='visual_stimulus_ratio', title_prefix='', figsize=(14, 10), dpi=200):
        """
        Layout:
          Row 0: [transition matrix] [param-by-state]
          Row 1: [posterior strip spanning both columns]  (long bar)
          Row 2: [psychometric by state spanning both columns] (or keep 2 cols if you prefer)
        """
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = GridSpec(nrows=3, ncols=2, figure=fig, height_ratios=[1.0, 0.45, 1.2], width_ratios=[1.0, 1.0])
        ax_A = fig.add_subplot(gs[0, 0])
        ax_B = fig.add_subplot(gs[0, 1])
        ax_C = fig.add_subplot(gs[1, :])
        ax_D = fig.add_subplot(gs[2, :])
        plot_transition_matrix(map_glmhmm, title=f'{title_prefix} Transition matrix', ax=ax_A, cmap='gray', fontsize=8)
        plot_param_by_state(map_glmhmm, df_mouse, X=X, y=y, ax=ax_B)
        ax_B.set_title(f'{title_prefix} Logistic params by state', fontsize=10)
        plot_posteriors_with_performance(df_mouse, post_prob_list, colors, ax=ax_C)
        ax_C.set_title(f'{title_prefix} Posterior p(state) + performance', fontsize=10)
        plot_psychometric_by_state(df_mouse, map_glmhmm, colors, x=stim_x, y=y, ax=ax_D)
        ax_D.set_title(f'{title_prefix} Psychometric by state', fontsize=10)
        return fig  # 中间长条更扁  # transition matrix  # param plot  # posterior strip (span)  # psychometric (span)  # 1) Transition matrix  # 2) Param-by-state  # 3) Posterior strip (long bar)  # 4) Psychometric by state

    return


@app.cell
def _(build_glmhmm_inputs_by_session, df_dic_aud, fit_glmhmm):
    mouse_24 = 'ACV009'
    df_mouse_aud = df_dic_aud[mouse_24]
    df_mouse_aud.dropna(subset=['total_evidence_strength', 'first_choice_numeric'], inplace=True)
    (datas_aud_1, inpts_aud_1) = build_glmhmm_inputs_by_session(df_mouse_aud, session_col='session', y_col='first_choice_numeric', stim_col='total_evidence_strength')
    # Fit GLM-HMM with MAP estimation:
    (map_glmhmm_aud_1, ll_aud_1, hmm_lls_aud_1) = fit_glmhmm(datas_aud_1, inpts_aud_1, num_states=3, obs_dim=1, input_dim=2)
    return datas_aud_1, df_mouse_aud, inpts_aud_1, map_glmhmm_aud_1


@app.cell
def _(map_glmhmm_aud_1, plot_transition_matrix):
    plot_transition_matrix(map_glmhmm_aud_1, title='', ax=None, cmap='gray', fontsize=8)
    return


@app.cell
def _(
    add_df_glmhmm_state,
    colors_6,
    datas_aud_1,
    df_mouse_aud,
    get_posterior_probs_concat,
    inpts_aud_1,
    map_glmhmm_aud_1,
    plot_posteriors_with_performance,
):
    post_prob_list_3 = get_posterior_probs_concat(map_glmhmm_aud_1, datas_aud_1, inpts_aud_1)
    df_mouse_aud_1 = add_df_glmhmm_state(df_mouse_aud, post_prob_list_3)
    plot_posteriors_with_performance(df_mouse_aud_1, post_prob_list_3, colors_6, ax=None)
    return (df_mouse_aud_1,)


@app.cell
def _(df_mouse_aud_1, map_glmhmm_aud_1, plot_param_by_state):
    plot_param_by_state(map_glmhmm_aud_1, df_mouse_aud_1, X=['total_evidence_strength', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric', ax=None)
    return


@app.cell
def _(colors_6, df_mouse_aud_1, map_glmhmm_aud_1, plot_psychometric_by_state):
    plot_psychometric_by_state(df_mouse_aud_1, map_glmhmm_aud_1, colors_6, x='total_evidence_strength', y='first_choice_numeric', ax=None, valueType='continuous')
    return


@app.cell
def _(df_dic_vis, fit_best_glmhmm, np):
    mouse_25 = 'ACV009'
    df_mouse_sessions = df_dic_vis[mouse_25][df_dic_vis[mouse_25]['session'].isin(df_dic_vis[mouse_25]['session'].unique()[-10:])]
    datas_3 = []
    inpts_3 = []
    for sess in df_mouse_sessions['session'].unique():
        df_mouse_sessions[df_mouse_sessions['session'] == sess].dropna(subset=['visual_stimulus_ratio', 'first_choice_numeric'], inplace=True)
        df_mouse_session_12 = df_mouse_sessions[df_mouse_sessions['session'] == sess]
        obs_sess = df_mouse_session_12['first_choice_numeric'].to_numpy().astype(int).reshape(-1, 1)
        inpt_sess = np.column_stack([df_mouse_session_12['visual_stimulus_ratio'].to_list(), np.ones(len(df_mouse_session_12))])
        datas_3.append(obs_sess)
        inpts_3.append(inpt_sess)
    # Fit GLM-HMM with MAP estimation:
    (map_glmhmm_4, map_ll) = fit_best_glmhmm(datas_3, inpts_3, num_states=3, obs_dim=1, input_dim=2)  # bias term
    return datas_3, df_mouse_sessions, inpts_3, map_glmhmm_4, mouse_25


@app.cell
def _(map_glmhmm_4, mouse_25, plt):
    (fig_26, ax_20) = plt.subplots(1, 1, figsize=(4, 4))
    A = map_glmhmm_4.transitions.transition_matrix
    im_4 = ax_20.imshow(A, vmin=0, vmax=1, aspect='auto', cmap='gray')  # shape (K, K)
    ax_20.set_title(f'Visual: {mouse_25}, 10 sessions', fontsize=8)
    ax_20.set_xlabel('State t')
    ax_20.set_ylabel('State t-1')
    K_2 = A.shape[0]
    for i_12 in range(K_2):
        for j in range(K_2):
            val = A[i_12, j]
            ax_20.text(j, i_12, f'{val:.2f}', ha='center', va='center', color='black' if val > 0.5 else 'white', fontsize=8)
    # write the values of the transition matrix on the heatmap
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(datas_3, inpts_3, map_glmhmm_4, np):
    # Get expected states:
    posterior_probs = [map_glmhmm_4.expected_states(data=data, input=inpt)[0] for (data, inpt) in zip(datas_3, inpts_3)]
    posterior_probs_con = np.concatenate(posterior_probs, axis=0)
    return (posterior_probs_con,)


@app.cell
def _(colors_6, df_mouse_sessions, num_states_3, plt, posterior_probs_con):
    fig_27 = plt.figure(figsize=(20, 2.5), dpi=80, facecolor='w', edgecolor='k')
    for k in range(num_states_3):
        plt.plot(posterior_probs_con[:, k], label='State ' + str(k + 1), lw=2, color=colors_6[k])
    plt.plot(df_mouse_sessions['performance_w'].values * 0.01, '-k', label='Performance', alpha=0.7)
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.xlabel('trial #', fontsize=15)
    plt.ylabel('p(state)', fontsize=15)
    return


@app.cell
def _(df_mouse_sessions, np, posterior_probs_con):
    state_seq = np.argmax(posterior_probs_con, axis=1)
    df_mouse_sessions["glmhmm_state"] = state_seq
    return


@app.cell
def _(df_mouse_sessions, dft, map_glmhmm_4, pd, plt, utils):
    fig_28 = plt.figure(figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    df_param_states = pd.DataFrame([])
    for state in range(map_glmhmm_4.K):
        df_session_state = df_mouse_sessions[df_mouse_sessions['glmhmm_state'] == state]
        df_session_state = dft.calculate_time_between_trials_and_reaction_time(df_session_state)
        if len(df_session_state) > 1:
            (_, logit_model_multi) = utils.logi_model_fit(df_session_state, X=['visual_stimulus_ratio', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric')
            df_param_states[state] = logit_model_multi.params
    df_param_states.drop('const', inplace=True)
    plt.plot(df_param_states, marker='o')
    plt.xticks(rotation=45, ha='right')
    return


@app.cell
def _(colors_6, df_mouse_sessions, map_glmhmm_4, plots, plt):
    (fig_29, ax_21) = plt.subplots(1, 1, figsize=(8, 6))
    for state_1 in range(map_glmhmm_4.K):
        df_session_state_1 = df_mouse_sessions[df_mouse_sessions['glmhmm_state'] == state_1]
        if len(df_session_state_1) > 0:
            plots.psychometric_plot(df_session_state_1, x='visual_stimulus_ratio', y='first_choice_numeric', point_kwargs={'color': colors_6[state_1], 'label': ''}, line_kwargs={'color': colors_6[state_1], 'label': state_1})
    return


@app.cell
def _(df_dic_aud, fit_best_glmhmm, np):
    mouse_26 = 'ACV009'
    df_mouse_sessions_1 = df_dic_aud[mouse_26][df_dic_aud[mouse_26]['session'].isin(df_dic_aud[mouse_26]['session'].unique()[-10:])]
    datas_4 = []
    inpts_4 = []
    for sess_1 in df_mouse_sessions_1['session'].unique():
        df_mouse_sessions_1[df_mouse_sessions_1['session'] == sess_1].dropna(subset=['total_evidence_strength', 'first_choice_numeric'], inplace=True)
        df_mouse_session_13 = df_mouse_sessions_1[df_mouse_sessions_1['session'] == sess_1]
        obs_sess_1 = df_mouse_session_13['first_choice_numeric'].to_numpy().astype(int).reshape(-1, 1)
        inpt_sess_1 = np.column_stack([df_mouse_session_13['total_evidence_strength'].to_list(), np.ones(len(df_mouse_session_13))])
        datas_4.append(obs_sess_1)
        inpts_4.append(inpt_sess_1)
    # Fit GLM-HMM with MAP estimation:
    (map_glmhmm_5, map_ll_1) = fit_best_glmhmm(datas_4, inpts_4, num_states=3, obs_dim=1, input_dim=2)  # bias term
    return datas_4, df_mouse_sessions_1, inpts_4, map_glmhmm_5, mouse_26


@app.cell
def _(map_glmhmm_5, mouse_26, plt):
    (fig_30, ax_22) = plt.subplots(1, 1, figsize=(4, 4))
    A_1 = map_glmhmm_5.transitions.transition_matrix
    im_5 = ax_22.imshow(A_1, vmin=0, vmax=1, aspect='auto', cmap='gray')  # shape (K, K)
    ax_22.set_title(f'Auditory: {mouse_26}, 10 sessions', fontsize=8)
    ax_22.set_xlabel('State t')
    ax_22.set_ylabel('State t-1')
    K_3 = A_1.shape[0]
    for i_13 in range(K_3):
        for j_1 in range(K_3):
            val_1 = A_1[i_13, j_1]
            ax_22.text(j_1, i_13, f'{val_1:.2f}', ha='center', va='center', color='black' if val_1 > 0.5 else 'white', fontsize=8)
    # write the values of the transition matrix on the heatmap
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(datas_4, inpts_4, map_glmhmm_5, np):
    # Get expected states:
    posterior_probs_1 = [map_glmhmm_5.expected_states(data=data, input=inpt)[0] for (data, inpt) in zip(datas_4, inpts_4)]
    posterior_probs_con_1 = np.concatenate(posterior_probs_1, axis=0)
    return (posterior_probs_con_1,)


@app.cell
def _(colors_6, df_mouse_sessions_1, map_glmhmm_5, plt, posterior_probs_con_1):
    fig_31 = plt.figure(figsize=(20, 2.5), dpi=80, facecolor='w', edgecolor='k')
    for k_1 in range(map_glmhmm_5.K):
        plt.plot(posterior_probs_con_1[:, k_1], label='State ' + str(k_1 + 1), lw=2, color=colors_6[k_1])
    plt.plot(df_mouse_sessions_1['performance_w'].values * 0.01, '-k', label='Performance', alpha=0.7)
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.xlabel('trial', fontsize=15)
    plt.ylabel('p(state)', fontsize=15)
    return


@app.cell
def _(df_mouse_sessions_1, np, posterior_probs_con_1):
    state_seq_1 = np.argmax(posterior_probs_con_1, axis=1)
    df_mouse_sessions_1['glmhmm_state'] = state_seq_1
    return


@app.cell
def _(df_mouse_sessions_1, dft, map_glmhmm_5, pd, plt, utils):
    fig_32 = plt.figure(figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    df_param_states_1 = pd.DataFrame([])
    for state_2 in range(map_glmhmm_5.K):
        df_session_state_2 = df_mouse_sessions_1[df_mouse_sessions_1['glmhmm_state'] == state_2]
        df_session_state_2 = dft.calculate_time_between_trials_and_reaction_time(df_session_state_2)
        if len(df_session_state_2) > 1:
            (_, logit_model_multi_1) = utils.logi_model_fit(df_session_state_2, X=['total_evidence_strength', 'roa_choice_numeric', 'reaction_time', 'previous_first_choice_numeric'], y='first_choice_numeric')
            df_param_states_1[state_2] = logit_model_multi_1.params
    df_param_states_1.drop('const', inplace=True)
    plt.plot(df_param_states_1, marker='o')
    plt.xticks(rotation=45, ha='right')
    return


@app.cell
def _(colors_6, df_mouse_sessions_1, map_glmhmm_5, plots, plt):
    (fig_33, ax_23) = plt.subplots(1, 1, figsize=(8, 6))
    for state_3 in range(map_glmhmm_5.K):
        df_session_state_3 = df_mouse_sessions_1[df_mouse_sessions_1['glmhmm_state'] == state_3]
        if len(df_session_state_3) > 0:
            plots.psychometric_plot(df_session_state_3, x='total_evidence_strength', y='first_choice_numeric', point_kwargs={'color': colors_6[state_3], 'label': ''}, line_kwargs={'color': colors_6[state_3], 'label': state_3}, valueType='continuous')
    return


@app.cell
def _(cmap_1, colors_6, df_dic_hard_aud, df_dic_hard_vis, plt):
    (fig_34, ax_24) = plt.subplots(4, 5, figsize=(16, 8))
    (raw, col) = (0, 0)
    for mouse_27 in df_dic_hard_vis:
        lim_vis_3 = 1.05 * df_dic_hard_vis[mouse_27]['performance_w'].max()
        ax_24[raw, col].imshow(df_dic_hard_vis[mouse_27]['hmm_states'].to_numpy()[None, :], aspect='auto', cmap=cmap_1, vmin=0, vmax=len(colors_6) - 1, extent=[df_dic_hard_vis[mouse_27].index.min(), df_dic_hard_vis[mouse_27].index.max(), 0, lim_vis_3])
        ax_24[raw, col].plot(df_dic_hard_vis[mouse_27]['performance_w'], '-k')
        ax_24[raw, col].set_title(f'Visual: {mouse_27}', fontsize=8)
        ax_24[raw, col].tick_params(axis='x', labelsize=6)
        ax_24[raw, col].tick_params(axis='y', labelsize=6)
        ax_24[raw, col].set_ylabel('Performance', fontsize=6)
        ax_24[raw, col].set_xlabel('Trial', fontsize=6)
        lim_aud = 1.05 * df_dic_hard_aud[mouse_27]['performance_w'].max()
        ax_24[raw + 1, col].imshow(df_dic_hard_aud[mouse_27]['hmm_states'].to_numpy()[None, :], aspect='auto', cmap=cmap_1, vmin=0, vmax=len(colors_6) - 1, extent=[df_dic_hard_aud[mouse_27].index.min(), df_dic_hard_aud[mouse_27].index.max(), 0, lim_aud])
        ax_24[raw + 1, col].plot(df_dic_hard_aud[mouse_27]['performance_w'], '-k')
        ax_24[raw + 1, col].set_title(f'Auditory: {mouse_27}', fontsize=8)
        ax_24[raw + 1, col].tick_params(axis='x', labelsize=6)
        ax_24[raw + 1, col].tick_params(axis='y', labelsize=6)
        ax_24[raw + 1, col].set_ylabel('Performance', fontsize=6)
        ax_24[raw + 1, col].set_xlabel('Trial', fontsize=6)
        col = col + 1
        if col % 5 == 0:
            raw = raw + 2
            col = 0
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(colors_6, df_dic_hard_aud, df_dic_hard_vis, np, num_states_3, plt, ssm):
    (fig_35, ax_25) = plt.subplots(4, 5, figsize=(16, 8))
    (raw_1, col_1) = (0, 0)
    all_vis_state_list = np.array([])
    all_vis_durations = np.array([])
    all_aud_state_list = np.array([])
    all_aud_durations = np.array([])
    for mouse_28 in df_dic_hard_vis:
        (vis_state_list, vis_durations) = ssm.util.rle(df_dic_hard_vis[mouse_28]['hmm_states'].to_numpy())
        all_vis_state_list = np.concatenate((all_vis_state_list, vis_state_list))
        all_vis_durations = np.concatenate((all_vis_durations, vis_durations))
        durs_stacked = []
        for s in range(num_states_3):
            durs_stacked.append(vis_durations[vis_state_list == s])
        ax_25[raw_1, col_1].hist(durs_stacked, label=['state ' + str(s) for s in range(num_states_3)], color=[c for c in colors_6[:len(durs_stacked)]])
        ax_25[raw_1, col_1].set_title(f'Visual: {mouse_28}', fontsize=8)
        ax_25[raw_1, col_1].set_xlabel('Duration of State (trials)', fontsize=6)
        ax_25[raw_1, col_1].set_ylabel('Frequency', fontsize=6)
        (aud_state_list, aud_durations) = ssm.util.rle(df_dic_hard_aud[mouse_28]['hmm_states'].to_numpy())
        all_aud_state_list = np.concatenate((all_aud_state_list, aud_state_list))
        all_aud_durations = np.concatenate((all_aud_durations, aud_durations))
        durs_stacked = []
        for s in range(num_states_3):
            durs_stacked.append(aud_durations[aud_state_list == s])
        ax_25[raw_1 + 1, col_1].hist(durs_stacked, label=['state ' + str(s) for s in range(num_states_3)], color=[c for c in colors_6[:len(durs_stacked)]])
        ax_25[raw_1 + 1, col_1].set_title(f'Auditory: {mouse_28}', fontsize=8)
        ax_25[raw_1 + 1, col_1].set_xlabel('Duration of State (trials)', fontsize=6)
        ax_25[raw_1 + 1, col_1].set_ylabel('Frequency', fontsize=6)
        col_1 = col_1 + 1
        if col_1 % 5 == 0:
            raw_1 = raw_1 + 2
            col_1 = 0
    plt.tight_layout()
    plt.show()
    return (
        all_aud_durations,
        all_aud_state_list,
        all_vis_durations,
        all_vis_state_list,
    )


@app.cell
def _(
    all_aud_durations,
    all_aud_state_list,
    all_vis_durations,
    all_vis_state_list,
    colors_6,
    num_states_3,
    plt,
):
    vis_durs_stacked = []
    aud_durs_stacked = []
    for s_1 in range(num_states_3):
        vis_durs_stacked.append(all_vis_durations[all_vis_state_list == s_1])
        aud_durs_stacked.append(all_aud_durations[all_aud_state_list == s_1])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(vis_durs_stacked, label=['state ' + str(s) for s in range(2)], color=[c for c in colors_6[:len(vis_durs_stacked)]])
    plt.title('Visual: Duration of States', fontsize=12)
    plt.xlabel('Duration of State (trials)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(aud_durs_stacked, label=['state ' + str(s) for s in range(2)], color=[c for c in colors_6[:len(aud_durs_stacked)]])
    plt.title('Auditory: Duration of States', fontsize=12)
    plt.xlabel('Duration of State (trials)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## filter the correlated values
    """)
    return


@app.cell
def _(df_dic_hard_vis, plots, plt, utils):
    fig_36 = plt.figure(figsize=(4, 8))
    X_1 = ['visual_stimulus_ratio', 'visual_ratio_diff_interact', 'visual_ratio_bright_interact', 'previous_left_choice_correct_numeric', 'previous_right_choice_wrong_numeric', 'previous_first_choice_numeric', 'previous_last_choice_numeric', 'previous_port_before_stimulus_numeric', 'left_bright']
    (corr_mat_list, norm_contribution_df) = utils.filter_variables_for_model(dic_fit=df_dic_hard_vis, X=X_1, y='first_choice_numeric')
    plots.plot_filter_model_variables(corr_mat_list=corr_mat_list, norm_contribution_df=norm_contribution_df)
    fig_36.savefig('/mnt/e/document/汇报/LeCi Lab/fens_abstract/variable_contribution_visual.svg')
    return


@app.cell
def _(df_dic_hard_vis, plots):
    X_2 = ['visual_stimulus_ratio', 'previous_left_choice_correct_numeric', 'previous_right_choice_wrong_numeric', 'previous_port_before_stimulus_numeric', 'left_bright']
    plots.plot_model_vars_params_compare(dic_fit=df_dic_hard_vis, X=X_2, y='first_choice_numeric')  # 'visual_ratio_diff_interact',  # 'visual_ratio_bright_interact',  # 'previous_first_choice_numeric',  # 'previous_last_choice_numeric', 
    return


@app.cell
def _(df_dic_hard_vis, max_lag_vis_correct, plots, utils):
    X_3 = ['abs_visual_stimulus_ratio', 'wrong_bright', 'previous_same_choice_correct_numeric', 'previous_same_choice_numeric', 'previous_correct_numeric', 'previous_port_before_stimulus_numeric', 'previous_first_choice_numeric', 'previous_last_choice_numeric', 'time_kernel_impact_correct']
    (corr_mat_list_1, norm_contribution_df_1) = utils.filter_variables_for_model(dic_fit=df_dic_hard_vis, X=X_3, y='correct_numeric', max_lag=max_lag_vis_correct, tau=1)
    plots.plot_filter_model_variables(corr_mat_list=corr_mat_list_1, norm_contribution_df=norm_contribution_df_1)
    return


@app.cell
def _(df_dic_hard_vis, plots):
    X_4 = ['abs_visual_stimulus_ratio', 'wrong_bright', 'previous_same_choice_correct_numeric', 'previous_same_choice_numeric', 'previous_correct_numeric', 'time_kernel_impact_correct']
    plots.plot_model_vars_params_compare(dic_fit=df_dic_hard_vis, X=X_4, y='correct_numeric')  # 'previous_port_before_stimulus_numeric',  # 'previous_first_choice_numeric',  # 'previous_last_choice_numeric', 
    return


@app.cell
def _(df_dic_hard_aud, plots, utils):
    X_5 = ['total_percentage_of_tones_left', 'number_of_tones_left', 'percentage_of_timebins_with_evidence_left', 'total_evidence_strength', 'left_tones_amplitude_sum', 'amplitude_strength_left_right', 'previous_port_before_stimulus_numeric', 'previous_left_choice_correct_numeric', 'previous_right_choice_wrong_numeric', 'previous_first_choice_numeric', 'previous_last_choice_numeric']
    (corr_mat_list_2, norm_contribution_df_2) = utils.filter_variables_for_model(dic_fit=df_dic_hard_aud, X=X_5, y='first_choice_numeric')
    plots.plot_filter_model_variables(corr_mat_list=corr_mat_list_2, norm_contribution_df=norm_contribution_df_2)
    return


@app.cell
def _(df_dic_hard_aud, plots):
    X_6 = ['total_evidence_strength', 'left_tones_amplitude_sum', 'amplitude_strength_left_right', 'previous_port_before_stimulus_numeric', 'previous_right_choice_wrong_numeric', 'previous_last_choice_numeric']
    plots.plot_model_vars_params_compare(dic_fit=df_dic_hard_aud, X=X_6, y='first_choice_numeric')  # 'total_percentage_of_tones_left',  # 'number_of_tones_left',  # 'percentage_of_timebins_with_evidence_left',  # 'previous_left_choice_correct_numeric',  # 'previous_first_choice_numeric', 
    return


@app.cell
def _(df_dic_hard_aud, max_lag_aud_correct, plots, utils):
    X_7 = ['total_percentage_of_tones_high', 'number_of_tones_high', 'percentage_of_timebins_with_evidence_high', 'abs_total_evidence_strength', 'high_tones_amplitude_sum', 'amplitude_strength', 'previous_same_choice_correct_numeric', 'previous_same_choice_numeric', 'previous_correct_numeric', 'previous_port_before_stimulus_numeric', 'time_kernel_impact_correct']
    (corr_mat_list_3, norm_contribution_df_3) = utils.filter_variables_for_model(dic_fit=df_dic_hard_aud, X=X_7, y='correct_numeric', max_lag=max_lag_aud_correct, tau=1)
    plots.plot_filter_model_variables(corr_mat_list=corr_mat_list_3, norm_contribution_df=norm_contribution_df_3)
    return


@app.cell
def _(df_dic_hard_aud, plots):
    X_8 = ['percentage_of_timebins_with_evidence_high', 'abs_total_evidence_strength', 'amplitude_strength', 'previous_same_choice_correct_numeric', 'previous_same_choice_numeric', 'previous_port_before_stimulus_numeric', 'time_kernel_impact_correct']
    plots.plot_model_vars_params_compare(dic_fit=df_dic_hard_aud, X=X_8, y='correct_numeric')  # 'total_percentage_of_tones_high',  # 'number_of_tones_high',  # 'high_tones_amplitude_sum',  # 'previous_correct_numeric', 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I kept what you did for comparison here
    """)
    return


@app.cell
def _(LogisticRegression, df_test, plt):
    # It is interesting to compare the effects of the relative difference between the two visual stimuli,
    # and the absolute difference between them.
    for i_14 in df_test.groupby('visual_stimulus_ratio'):
    # Maybe what we can do is to train another logistic regression model, adding as well the absolute difference
    # between the two visual stimuli, and see how it affects the probability of a left choice.
    # Do you know what I mean?
        df_i_1 = i_14[1].sort_values(by='visual_stimulus_diff')
        X_9 = df_i_1['visual_stimulus_diff'].values.reshape(-1, 1)
        y_3 = df_i_1['left_choice'].values.astype(int)
        model_1 = LogisticRegression()
        model_1.fit(X_9, y_3)
        y_pred_1 = model_1.predict(X_9)
        y_prob_1 = model_1.predict_proba(X_9)[:, 1]
        plt.plot(X_9, y_prob_1, label=f'Visual Stimulus ratio: {i_14[0]}')
        plt.legend()
    plt.xlabel('Visual Stimulus Difference')
    plt.ylabel('Probability of Left Choice')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # change with time
    """)
    return


@app.cell
def _(df_dic_hard_vis, plots, plt, sns):
    (fig_37, ax_26) = plt.subplots(2, 5, figsize=(20, 10))
    for (df_name_4, df_11, n_3) in zip(df_dic_hard_vis.keys(), df_dic_hard_vis.values(), range(len(df_dic_hard_vis))):
        row = n_3 // 5
        col_2 = n_3 % 5
        for (session_9, color_7) in zip(df_11['session'].unique(), sns.color_palette('crest', len(df_11['session'].unique()))):
            df_session = df_11[df_11['session'] == session_9]
            plots.psychometric_plot(df_session, x='visual_stimulus_ratio', y='first_choice_numeric', ax=ax_26[row, col_2], point_kwargs={'color': color_7, 'label': ''}, line_kwargs={'color': color_7, 'label': ''})
        ax_26[row, col_2].set_title(f'Psychometric Curve for {df_name_4}')
    plt.tight_layout()
    # Add a colorbar to indicate the session
    cbar_4 = plt.colorbar(plt.cm.ScalarMappable(cmap=sns.color_palette('crest', as_cmap=True)), orientation='horizontal', ax=ax_26, shrink=0.3)
    cbar_4.set_ticks([])
    cbar_4.set_label('before → after')
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, pd, plt, sns, utils):
    def plot_model_param_evolution(
        df_dic: dict,
        x_col: str,
        y_col: str,
        block_size: int = 1000,
        title: str = "Model parameter evolution over trial blocks",
        figsize: tuple = (14, 4),
        palette: str = "colorblind"
    ):
        fig, ax = plt.subplots(1, 4, figsize=figsize)
        colors = sns.color_palette(palette, len(df_dic))

        lapse_left_dict, lapse_right_dict, slope_dict, bias_dict = {}, {}, {}, {}

        for (df_name, df), color in zip(df_dic.items(), colors):
            df = df.copy()
            df["trial_group"] = np.arange(len(df)) // block_size

            lapse_left, lapse_right, slope, bias = [], [], [], []

            for group in df["trial_group"].unique():
                df_group = df[df["trial_group"] == group]
                _, params = utils.fit_lapse_logistic_independent(df_group[x_col], df_group[y_col])
                lapse_left.append(params[0])
                lapse_right.append(params[1])
                slope.append(params[2])
                bias.append(params[3])

            lapse_left_dict[df_name] = lapse_left
            lapse_right_dict[df_name] = lapse_right
            slope_dict[df_name] = slope
            bias_dict[df_name] = bias

            for a, param, name in zip(ax, [lapse_left, lapse_right, slope, bias],
                                      ["Lapse Left", "Lapse Right", "Beta", "X0"]):
                a.plot(param, c=color, alpha=0.5)
                a.set_ylabel(name)

        lapse_left_df = pd.DataFrame({k: pd.Series(v) for k, v in lapse_left_dict.items()})
        lapse_right_df = pd.DataFrame({k: pd.Series(v) for k, v in lapse_right_dict.items()})
        slope_df = pd.DataFrame({k: pd.Series(v) for k, v in slope_dict.items()})
        bias_df = pd.DataFrame({k: pd.Series(v) for k, v in bias_dict.items()})

        for a, df, name in zip(ax,
                               [lapse_left_df, lapse_right_df, slope_df, bias_df],
                               ["Lapse Left", "Lapse Right", "Beta", "X0"]):
            a.plot(df.mean(axis=1), color="k", linestyle="-", label="Overall Mean")
            a.set_xlabel(f"{block_size}-Trial Blocks")
            a.legend(frameon=False)
            a.set_title(name)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    return (plot_model_param_evolution,)


@app.cell
def _(df_dic_hard_vis, plot_model_param_evolution):
    plot_model_param_evolution(
        df_dic=df_dic_hard_vis,
        x_col="visual_stimulus_ratio",
        y_col="first_choice_numeric",
        title="Visual Left Choice Model parameter change over 1000-trial blocks"
    )
    return


@app.cell
def _(df_dic_hard_vis, plot_model_param_evolution):
    plot_model_param_evolution(
        df_dic=df_dic_hard_vis,
        x_col="abs_visual_stimulus_ratio",
        y_col="correct_numeric",
        title="Visual Correct Choice Model parameter change over 1000-trial blocks"
    )
    return


@app.cell
def _(df_dic_hard_aud, plot_model_param_evolution):
    plot_model_param_evolution(
        df_dic=df_dic_hard_aud,
        x_col="total_evidence_strength",
        y_col="first_choice_numeric",
        title="Auditory Left Choice Model parameter change over 1000-trial blocks"
    )
    return


@app.cell
def _(df_dic_hard_aud, plot_model_param_evolution):
    plot_model_param_evolution(
        df_dic=df_dic_hard_aud,
        x_col="abs_total_evidence_strength",
        y_col="correct_numeric",
        title="Auditory Correct Choice Model parameter change over 1000-trial blocks"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # timebin evidence of auditory task
    """)
    return


@app.cell
def _():
    sound_properties_for_cot_mat = {
        "duration": .5,
        "high_amplitude_mean": 70,
        "low_amplitude_mean": 70,
        "amplitude_std": 2,
        "subduration": 0.03,
        "suboverlap": 0.01,
        "ambiguous_beginning_time": 0.05,
    }
    return (sound_properties_for_cot_mat,)


@app.cell
def _(
    LinearSegmentedColormap,
    df_dic_hard_aud,
    pd,
    plt,
    sns,
    sound_properties_for_cot_mat,
):
    # auditory stimuli plot
    aud_stim_sample = df_dic_hard_aud['ACV001']['auditory_stimulus'][67103]
    high_mat = pd.DataFrame(eval(aud_stim_sample)['high_tones'])[::-1]
    low_mat = pd.DataFrame(eval(aud_stim_sample)['low_tones'])[::-1]
    colors_7 = [(1, 1, 1), (0.2, 0.6, 0.2)]
    n_bins = 100  # White to seagreen
    cmap_name = 'white_to_seagreen'  # Discretize the interpolation into bins
    cmap_2 = LinearSegmentedColormap.from_list(cmap_name, colors_7, N=n_bins)
    plt.figure(figsize=(10, 3))
    mat_to_plot = pd.concat([high_mat[::-1], low_mat[::-1]])
    sns.heatmap(mat_to_plot, cmap=cmap_2, vmin=sound_properties_for_cot_mat['low_amplitude_mean'] * 0.8, vmax=sound_properties_for_cot_mat['high_amplitude_mean'] * 1.2, cbar_kws={'label': 'Amplitude (dB)'})
    plt.hlines(6, 0, high_mat.shape[1], colors='gray', linestyles='dashed')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Cloud of Tones')
    plt.show()
    return


@app.cell
def _(df_dic_hard_aud, np, pd, sm, sns, utils):
    timebin_evidence_df = pd.DataFrame()
    for (df_name_5, df_12, color_8) in zip(df_dic_hard_aud.keys(), df_dic_hard_aud.values(), sns.color_palette('colorblind', len(df_dic_hard_aud))):
        X_10 = np.array([utils.get_timebin_evidence(eval(t)) for t in df_12['auditory_stimulus']])
        y_4 = df_12['first_choice_numeric']
        X_model = sm.add_constant(X_10)
        glm = sm.Logit(y_4, X_model).fit()
        timebin_evidence_df[df_name_5] = glm.params[1:]
    return (timebin_evidence_df,)


@app.cell
def _(plt, sns, timebin_evidence_df):
    fig_38 = plt.figure(figsize=(12, 5))
    timebin_evidence_df.index = range(len(timebin_evidence_df))
    for (df_name_6, col_3, color_9) in zip(timebin_evidence_df.columns, timebin_evidence_df, sns.color_palette('colorblind', len(timebin_evidence_df))):
        plt.plot(timebin_evidence_df[col_3], color=color_9, linestyle='--', alpha=0.7)
    plt.plot(timebin_evidence_df.mean(axis=1), color='black', label='Mean Coefficient', linewidth=2)
    plt.xlabel('Time Bin')
    plt.ylabel('Coefficient')
    plt.title('Time Bin Evidence Coefficients for Left Choice')
    plt.legend()
    return


@app.cell
def _(df_dic_hard_aud, np, pd, sm, sns, utils):
    timebin_evidence_df_1 = pd.DataFrame()
    for (df_name_7, df_13, color_10) in zip(df_dic_hard_aud.keys(), df_dic_hard_aud.values(), sns.color_palette('colorblind', len(df_dic_hard_aud))):
        X_11 = np.abs(np.array([utils.get_timebin_evidence(eval(t)) for t in df_13['auditory_stimulus']]))
        y_5 = df_13['correct_numeric']
        X_model_1 = sm.add_constant(X_11)
        glm_1 = sm.Logit(y_5, X_model_1).fit()
        timebin_evidence_df_1[df_name_7] = glm_1.params[1:]
    return (timebin_evidence_df_1,)


@app.cell
def _(plt, sns, timebin_evidence_df_1):
    fig_39 = plt.figure(figsize=(12, 5))
    timebin_evidence_df_1.index = range(len(timebin_evidence_df_1))
    for (df_name_8, col_4, color_11) in zip(timebin_evidence_df_1.columns, timebin_evidence_df_1, sns.color_palette('colorblind', len(timebin_evidence_df_1))):
        plt.plot(timebin_evidence_df_1[col_4], color=color_11, linestyle='--', alpha=0.7)
    plt.plot(timebin_evidence_df_1.mean(axis=1), color='black', label='Mean Coefficient', linewidth=2)
    plt.xlabel('Time Bin')
    plt.ylabel('Coefficient')
    plt.title('Time Bin Evidence Coefficients for Correct Choice')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
