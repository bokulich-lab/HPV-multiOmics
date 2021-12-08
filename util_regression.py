# eval1: Random forest regression predictive accuracy (table)
import pandas as pd
import numpy as np
import os
import qiime2
from qiime2.plugins import sample_classifier as sc
from q2_sample_classifier.visuals import (_regplot_from_dataframe,
                                          _linear_regress)
import matplotlib.pyplot as plt
import seaborn as sns


def train_regressors(df_data4regr, ls_targets, ls_features,
                     str_target_descr, transform_target2log,
                     seed):
    """
    Function training regressors for targets in `ls_targets` with
    features in `ls_features` from `df_data4regr`.
    If `transform_target2log` is `True` then features
    are log transformed.
    """
    # target preparation: read
    df_data = df_data4regr.copy(deep=True)

    # transform all prefixes in features
    if str_target_descr.startswith('metabolites'):
        df_data.columns = [x.replace('F_metabo_lipid_', '')
                           for x in df_data.columns]
        df_data.columns = [x.replace('F_metabo_other_', '')
                           for x in df_data.columns]
    elif str_target_descr.startswith('biomarkers'):
        df_data.columns = [x.replace('F_proteo_', '')
                           for x in df_data.columns]

    # transform all prefixes in features
    df_targets = df_data[ls_targets]

    # transform target
    if transform_target2log:
        df_targets = df_targets.apply(np.log)

    # transform to Q2 metadata
    if len(ls_targets) == 1 and ((ls_targets[0] == 'F_pcov_pH')
                                 or (ls_targets[0] == 'T_infl_score_flt')):
        md_targets = qiime2.NumericMetadataColumn(df_targets[ls_targets[0]])
    else:
        md_targets = qiime2.Metadata(df_targets)

    # feature preparation: transform to Q2 artifact
    art_features = qiime2.Artifact.import_data('FeatureTable[Frequency]',
                                               df_data[ls_features])

    # train regressors
    rf_res2 = {}

    for t in ls_targets:
        if t not in rf_res2.keys():
            if len(ls_targets) == 1:
                md_tar = md_targets
            else:
                md_tar = md_targets.get_column(t)

        rf_res2[t] = sc.actions.regress_samples_ncv(
            art_features,
            md_tar,
            cv=10,
            random_state=seed,
            n_jobs=4,
            missing_samples='ignore')

    return rf_res2, md_targets


def get_regr_accuracy_results(str_target_descr, ls_targets,
                              md_targets, rf_results, output_dir):
    """
    Function extracting accuracy results for
    regression models saved in rf_results
    """

    cols = ['Target', "Mean squared error",
            "r-value", "r-squared",
            "P-value", "Std Error", "Slope", "Intercept"]

    accuracy_results = pd.DataFrame()

    for t in ls_targets:
        acc = pd.to_numeric(rf_results[t].predictions.view(pd.Series))
        acc, exp = acc.align(md_targets.get_column(
            t).to_series(), join='inner', axis=0)
        res = _linear_regress(exp, acc)
        res['Target'] = t
        accuracy_results = pd.concat([accuracy_results, res], axis=0)
        _regplot_from_dataframe(exp, acc)

    accuracy_results = accuracy_results[cols].groupby(
        ['Target']).mean().sort_values('r-squared', ascending=False)
    path2save = os.path.join(
        output_dir, '{}-rf-tabular-results.tsv'.format(str_target_descr))
    print('Accuracy results saved to: {}'.format(path2save))
    accuracy_results.to_csv(path2save, sep='\t')

    return accuracy_results


def _mod_regplot_from_dataframe(x, y, plot_style="whitegrid", arb=True,
                                color="grey", ax=None):
    '''Seaborn regplot with true 1:1 ratio set by arb (bool).'''
    sns.set_style(plot_style)
    reg = sns.regplot(x, y, color=color, ax=ax)
    plt.xlabel('')
    plt.ylabel('')
    if arb is True:
        x0, x1 = reg.axes.get_xlim()
        y0, y1 = reg.axes.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        reg.axes.plot(lims, lims, ':k')
    return reg


def plot_regr_scatterplots_top20targets(accuracy_results, rf_results,
                                        md_targets, str_target_descr,
                                        output_dir):
    """
    Function plotting regression scatterplots for the top 20 most
    accurately predicted targets.
    """

    fig, axes = plt.subplots(5, 4, figsize=(15, 10))
    n = 0

    for ax1, target in zip(axes.flatten(), accuracy_results.index[:20]):
        n += 1
        acc = pd.to_numeric(rf_results[target].predictions.view(pd.Series))
        acc, exp = acc.align(md_targets.get_column(
            target).to_series(), join='inner', axis=0)
        fig = _mod_regplot_from_dataframe(exp, acc, ax=ax1)
        ax1.title.set_text(target.capitalize())
        ax1.title.set_fontsize(10)
        if n > 16:
            ax1.set_xlabel('True Value')
        else:
            ax1.set_xlabel('')
        if n == 9:
            ax1.set_ylabel('Predicted Value')
        else:
            ax1.set_ylabel('')
    plt.tight_layout()
    path2save = os.path.join(
        output_dir, '{}-rf-predictions-scatterplots.pdf'.format(
            str_target_descr))
    fig.get_figure().savefig(path2save, bbox_inches="tight")
    fig.get_figure().savefig(path2save.replace('.pdf', '.png'),
                             bbox_inches="tight")
    print('Scatterplots saved in: {}'.format(path2save))


def plot_regr_top20_features(accuracy_results, rf_results, taxa,
                             str_target_descr, output_dir):
    """
    Function plotting top 15 features for the top 20 most
    accurately predicted targets in accuracy_results
    """

    fig, axes = plt.subplots(5, 4, figsize=(15, 10))  # used to be 15, 10 
    n = 0
    for ax1, target in zip(axes.flatten(), accuracy_results.index[:20]):
        n += 1
        imp = rf_results[target].feature_importance.view(pd.DataFrame)[:15]
        imp['Importance'] = pd.to_numeric(imp['importance'])
        imp = imp.sort_values('Importance', ascending=True)

        # remove prefix in index
        imp.index = [x.replace('F_micro_', '') for x in imp.index]
        imp.index = [x.replace('F_proteo_', '') for x in imp.index]
        imp.index = [x.replace('F_metabo_lipid_', '') for x in imp.index]
        imp.index = [x.replace('F_metabo_other_', '') for x in imp.index]

        imp.index = [i[:6] + ' : ' + taxa.loc[i] +
                     '*' if i in taxa.index else i for i in imp.index]
        imp.plot(y='Importance', kind='barh', ax=ax1,
                 grid=False, legend=False, fontsize=7, width=1)

        ax1.title.set_text(target.capitalize())
        ax1.title.set_fontsize(10)
        plt.setp(ax1.get_yticklabels(), Fontsize=6)
        if n > 16:
            ax1.set_xlabel('Mean Importance')
        # color labels if microbial
        for ytick in ax1.get_yticklabels():
            if '*' in ytick.get_text():
                ytick.set_color('r')
    plt.tight_layout()
    path2save = os.path.join(
        output_dir, '{}-rf-importance.pdf'.format(str_target_descr))
    print('Saved feature plots to: {}'.format(path2save))
    fig.savefig(path2save, bbox_inches="tight")
    fig.savefig(path2save.replace('.pdf', '.png'), bbox_inches="tight")


def train_n_eval_regressors(ls_targets, str_target_desc, transform_target2log,
                            ls_features, df_data,
                            taxa,
                            seed,
                            output_dir):

    output_dir = os.path.join(output_dir, 'regressors')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ! Train regressor
    print('\nTraining regressors for {} number of {}...'.format(
        len(ls_targets), str_target_desc))

    rf_res, md_targets = train_regressors(df_data,
                                          ls_targets,
                                          ls_features,
                                          str_target_desc,
                                          transform_target2log,
                                          seed)

    if len(rf_res) == 1:
        # ! Eval1: plot true vs predicted
        target = ls_targets[0]
        acc = pd.to_numeric(rf_res[target].predictions.view(pd.Series))
        acc, exp = acc.align(md_targets.to_series(), join='inner', axis=0)
        _linear_regress(exp, acc)
        fig = _mod_regplot_from_dataframe(exp, acc)
        if target == 'T_infl_score_flt':
            label_addon = 'Inflammation Score'
            xlim = (-0.5, 7.5)
        elif target == 'F_pcov_pH':
            label_addon = 'log10 pH'
            xlim = (1.45, 2.05)
        fig.set_xlabel('True {}'.format(label_addon))
        fig.set_ylabel('Predicted {}'.format(label_addon))
        plt.tight_layout()
        plt.xlim(xlim)
        path2save = os.path.join(
            output_dir, '{}-rf-predictions-scatterplots.pdf'.format(
                str_target_desc))
        fig.get_figure().savefig(path2save, bbox_inches="tight")
        fig.get_figure().savefig(path2save.replace('.pdf', '.png'),
                                 bbox_inches="tight")
        print('Scatterplots saved in: {}'.format(path2save))

    else:
        # ! Eval1: predictive metrics
        print('\nCalculating predictive metrics for trained regressors...')
        accuracy_results = get_regr_accuracy_results(str_target_desc,
                                                     ls_targets,
                                                     md_targets,
                                                     rf_res,
                                                     output_dir)
        # ! Eval2: top20 targets scatterplot
        print('\nPlotting scatterplot for top20 targets...')
        plot_regr_scatterplots_top20targets(accuracy_results,
                                            rf_res,
                                            md_targets,
                                            str_target_desc,
                                            output_dir)

        # ! Eval3: top15 features of top20 targets
        print('\nPlotting top15 features of top20 predicted targets...')
        plot_regr_top20_features(accuracy_results,
                                 rf_res,
                                 taxa,
                                 str_target_desc,
                                 output_dir)
