import sys
import os
import re
from io import StringIO
import pandas as pd
import numpy as np
import math
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix)
from matplotlib.pylab import plt
import seaborn as sns
import matplotlib as mpl
import qiime2
from qiime2.plugins import sample_classifier as sc
from q2_sample_classifier.visuals import (
    _add_sample_size_to_xtick_labels, _custom_palettes)


class Capturing(list):
    """
    Class saving stdout from operations
    copied from
    https://stackoverflow.com/questions/16571150/
    how-to-capture-stdout-output-from-a-python-function-call
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def plot_importance_topx_features(top_x, df_top, taxa,
                                  str_target, output_dir):
    """
    Plot importance of `top_x` features in `df_top`
    and save plot in `output_dir`
    """
    plt.style.use('seaborn-whitegrid')
    # set colors for all features
    df_top['color'] = 'orange'  # would be pcov if in top features
    df_top.loc[df_top.index.str.startswith(
        'F_micro_'), 'color'] = '#800680'  # purple
    df_top.loc[df_top.index.str.startswith('F_proteo_'),
               'color'] = '#E31A1B'  # red
    df_top.loc[df_top.index.str.startswith(
        'F_metabo_lipid_'), 'color'] = '#6BC8FB'  # lightblue
    df_top.loc[df_top.index.str.startswith(
        'F_metabo_other_'), 'color'] = '#1F77B4'  # darker blue

    # rename all features - dropping prefixes
    for prefix in ['F_micro_', 'F_metabo_lipid_', 'F_metabo_other_',
                   'F_proteo_', 'F_pcov_']:
        df_top.index = [i.replace(prefix, '')
                        for i in df_top.index]
    # special naming of microbiome features:
    df_top.index = [i[:6] + ' : ' + taxa.loc[i] +
                    '*' if i in taxa.index else i for i in
                    df_top.index]

    # ensure df_top is ordered correctly and select top_x
    df_topx = df_top.sort_values('importance',
                                 ascending=False)[:top_x]

    # plot
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.barh(range(top_x - 1, -1, -1),
            df_topx.importance,
            color=df_topx['color'])
    ax.set_ylim(-0.6, top_x - 0.4)
    ls_var_names = df_topx.index.tolist()
    plt.yticks(range(len(ls_var_names)), list(
        reversed(ls_var_names)), fontsize=9)
    plt.grid(True)
    # color labels if microbial
    for ytick in ax.get_yticklabels():
        if '*' in ytick.get_text():
            ytick.set_color('r')

    ax.set_xlabel('Mean Gini feature importance', fontsize=9)
    ax.get_figure().savefig(os.path.join(output_dir,
                                         str_target +
                                         '-feature-importance.pdf'),
                            bbox_inches="tight")


def plot_abundances_top40_features(df_data_orig, df_top_features,
                                   taxa,
                                   str_target, ls_class_order,
                                   dic_palette, output_dir):
    """
    Plot abundances of top 40 most predictive features
    """
    fig, axes = plt.subplots(10, 4, figsize=(16, 20))

    # remove all prefixed
    df_data = df_data_orig.copy(deep=True)
    df_top = df_top_features.copy(deep=True)
    for prefix in ['F_micro_', 'F_metabo_lipid_', 'F_metabo_other_',
                   'F_proteo_', 'F_pcov_']:
        df_data.columns = [i.replace(prefix, '')
                           for i in df_data.columns]
        df_top.index = [i.replace(prefix, '')
                        for i in df_top.index]

    # rename micro features to taxa microbiome features
    df_data.columns = [i[:6] + ' : ' + taxa.loc[i] +
                       '*' if i in taxa.index else i for i in
                       df_data.columns]
    df_top.index = [i[:6] + ' : ' + taxa.loc[i] +
                    '*' if i in taxa.index else i for i in
                    df_top.index]

    # plot feature abundances
    for ax1, feature in zip(axes.flatten(), df_top.index[:40]):

        sns.boxplot(x=feature, y=str_target,
                    data=df_data, showfliers=False, ax=ax1,
                    order=ls_class_order, color='w')
        sns.swarmplot(y=str_target, x=feature, data=df_data,
                      hue=str_target, palette=dic_palette,
                      ax=ax1, order=ls_class_order,
                      orient='h')
        ax1.get_legend().remove()
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        # rename
        if feature in taxa.index:
            ax1.set_title(feature[:6] + ' : ' + taxa.loc[feature] + '*')
        else:
            ax1.set_title(feature)
        ax1.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(os.path.join(
        output_dir, str_target + '-predictors-abundance.pdf'),
        bbox_inches="tight")
    return fig


def calculate_aucs(result_q2c, df_data, str_target):
    """
    Function that calculates AUC of ROC and precision-recall curve
    with macro averaging.
    """
    df_predprob = result_q2c.probabilities.view(pd.DataFrame)
    df_predprob.sort_index(inplace=True)
    ls_cols = df_predprob.columns.tolist()

    df_true = df_data[str_target].copy(deep=True)
    df_true.sort_index(inplace=True)
    df_true = pd.get_dummies(df_true)[ls_cols]

    auc_roc = roc_auc_score(
        df_true, df_predprob, average='macro')
    auc_prc = average_precision_score(
        df_true, df_predprob, average='macro')

    return auc_roc, auc_prc


def run_omics_separately(ls_micro, ls_metabo, ls_proteome,
                         str_target,
                         df_data,
                         output_res_combined,
                         res_combined,
                         seed):
    """
    Function training separate classifiers with omics features
    defined in `ls_micro`, `ls_metabo` and `ls_proteome` to
    predict `str_target` on `df_data` and returning dataframe with accuracy
    & AUC metrics saved (`df_omics`).
    model run with combined omics features is fead in via saved
    output from previous run `output_res_combined` & `res_combined`.
    """

    dic_features_omics = {'Microbiome': ['orange', ls_micro],
                          'Metabolome': ['blue', ls_metabo],
                          'Immunoproteome': ['red', ls_proteome]}
    # drop omics if no features were used
    dic_features_omics = {k: (v1, v2) for (
        k, (v1, v2)) in dic_features_omics.items() if len(v2) > 0}
    # init df with metrics
    df_omics = pd.DataFrame(index=list(dic_features_omics.keys()), columns=[
        'Accuracy', 'SD', 'ROC_AUC_macro', 'PRC_AUC_macro', 'color'])

    for key, (color, features) in dic_features_omics.items():
        print('{} - Number of features: {}'.format(key, len(features)))

        # transform features to Q2 artifact
        art_features = qiime2.Artifact.import_data(
            'FeatureTable[Frequency]', df_data[features])

        # transform target to Q2 metadatacolumn
        md_target = qiime2.CategoricalMetadataColumn(
            df_data[str_target])

        # train classifier and capture accuracy output
        with Capturing() as output_res:
            res_omics = sc.actions.classify_samples_ncv(
                art_features,
                md_target,
                cv=10,
                random_state=seed,
                n_jobs=4,
                n_estimators=500,
                missing_samples='ignore')
        print(output_res)
        # extract estimator accuracy and stddev from output
        accuracy = float(
            re.search(r"(\d*(\.)\d*).±", output_res[0]).group(1))
        stddev = float(
            re.search(r"±.(\d*(\.)\d*)", output_res[0]).group(1))
        df_omics.loc[key, 'Accuracy'] = accuracy
        df_omics.loc[key, 'SD'] = stddev
        df_omics.loc[key, 'color'] = color

        # calculate AUC for ROC and precision-recall curve
        roc_auc, prc_auc = calculate_aucs(res_omics, df_data, str_target)
        df_omics.loc[key, 'ROC_AUC_macro'] = roc_auc
        df_omics.loc[key, 'PRC_AUC_macro'] = prc_auc

    # add combined results to df_omics
    # accuracy:
    df_omics.loc['Combined', 'Accuracy'] = float(
        re.search(r"(\d*(\.)\d*).±", output_res_combined[0]).group(1))
    df_omics.loc['Combined', 'SD'] = float(
        re.search(r"±.(\d*(\.)\d*)", output_res_combined[0]).group(1))
    df_omics.loc['Combined', 'color'] = 'purple'
    # AUCs of ROC and PRcurve
    roc_auc, prc_auc = calculate_aucs(res_combined, df_data, str_target)
    df_omics.loc['Combined', 'ROC_AUC_macro'] = roc_auc
    df_omics.loc['Combined', 'PRC_AUC_macro'] = prc_auc

    return df_omics


def plot_metric_omics(df_omics_metrics, sample_count, str_target, str_metric):
    """
    Function returning plot of each omics experiment's metric ('str_metric`)
    as provided in `df_omics`.
    """

    # process df_omics
    df_omics = df_omics_metrics.copy(deep=True)
    df_omics[1] = df_omics[str_metric]
    df_omics[2] = df_omics.loc['Combined', str_metric]

    fig, ax = plt.subplots(figsize=(4, 3))

    # get lineplot of omics accuracy performance
    nb_omics_rows = df_omics.shape[0] - 1
    df_omics[[1, 2]][:nb_omics_rows].T.plot(
        alpha=0.5, color=df_omics['color'],
        ax=ax, legend=False)

    # add error bars - only available for accuracy
    if str_metric == 'Accuracy':
        for y in df_omics.index:
            for x in [1, 2]:
                if (y != 'Combined' and x == 1) or (
                        y == 'Combined' and x == 2):
                    color2use = df_omics.loc[y, 'color']
                    yval = df_omics.loc[y, x]
                    # calc standard error
                    err = df_omics.loc[y, 'SD'] / math.sqrt(sample_count)
                    # error bar (standard error)
                    plt.plot([x, x], [yval - err, yval + err],
                             c=color2use, alpha=0.5)
                    # top tick
                    plt.plot([x-0.1, x+0.1], [yval + err, yval + err],
                             c=color2use, alpha=0.5)
                    # bottom tick
                    plt.plot([x-0.1, x+0.1], [yval - err, yval - err],
                             c=color2use, alpha=0.5)
        plt.xlim(0.8, 2.2)

    # add plot characteristics
    custom_lines = [mpl.lines.Line2D([0], [0], color=c, lw=4)
                    for c in df_omics['color'][:4]]
    ax.legend(custom_lines, df_omics.index[:4],
              #   loc=4,
              #   borderaxespad=0.,
              fontsize=10)
    ax.set_title(str_target)
    ax.set_ylabel(str_metric)
    ax.set_xlabel('N Data Sets')
    ax.set_xticklabels(['', 1, '', '', '', '', 3])

    return fig


def _plot_adjusted_heatmap_from_confusion_matrix(cm, palette,
                                                 vmin=None, vmax=None):
    """
    Function adjusted from q2_sample_classifier.visuals to suit
    viusal proportions needed for this study.
    """
    palette = _custom_palettes()[palette]
    plt.figure()
    scaler, labelsize, dpi, cbar_min = 10, 8, 100, .15
    sns.set(rc={'xtick.labelsize': labelsize, 'ytick.labelsize': labelsize,
            'figure.dpi': dpi})
    fig, (ax, cax) = plt.subplots(ncols=2, constrained_layout=True)
    heatmap = sns.heatmap(cm, vmin=vmin, vmax=vmax, cmap=palette, ax=ax,
                          cbar_ax=cax, cbar_kws={'label': 'Proportion',
                                                 'shrink': 0.7},
                          square=True, xticklabels=True, yticklabels=True)
    # Resize the plot dynamically based on number of classes
    hm_pos = ax.get_position()
    scale = len(cm) / scaler
    # prevent cbar from getting unreadably small
    cbar_height = max(cbar_min, scale)
    ax.set_position([hm_pos.x0, hm_pos.y0, scale, scale])
    cax.set_position([hm_pos.x0 + scale * .95, hm_pos.y0, scale / len(cm),
                     cbar_height])
    # Make the heatmap subplot (not the colorbar) the active axis object so
    # labels apply correctly on return
    plt.sca(ax)
    return heatmap


def train_n_eval_classifier(target2predict, ls_features, df_data, taxa,
                            ls_class_order, dic_color_palette,
                            seed, output_dir):

    output_dir = os.path.join(output_dir, 'classifiers')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ! Train classifier
    print('\nTraining classifier for {}...'.format(target2predict))
    # transform features to Q2 artifact
    print('Shape of feature table: {}'.format(
        df_data[ls_features].shape))
    art_features = qiime2.Artifact.import_data(
        'FeatureTable[Frequency]', df_data[ls_features])

    # transform target to Q2 metadatacolumn
    md_target = qiime2.CategoricalMetadataColumn(
        df_data[target2predict])

    # train classifier
    with Capturing() as output_res_combined:
        res_combined = sc.actions.classify_samples_ncv(
            art_features,
            md_target,
            cv=10,
            random_state=seed,
            n_jobs=4,
            n_estimators=500,
            missing_samples='ignore')
    print(output_res_combined)

    # ! Evaluate classifier
    print('\nEvaluating combined classifier...')
    # ROC curve & confusion matrix
    performance_qzv, = sc.actions.confusion_matrix(
        predictions=res_combined.predictions,
        probabilities=res_combined.probabilities,
        truth=md_target)
    path2save = os.path.join(
        output_dir, '{}-accuracy.qzv'.format(target2predict))
    performance_qzv.save(path2save)
    print('Confusion matrix and ROC curve saved as Q2'
          'artifact here: {}'.format(path2save))

    # Plot confusion matrix separately
    df_pred = res_combined.predictions.view(pd.Series)
    df_pred.sort_index(inplace=True)

    df_true = df_data[target2predict].copy(deep=True)
    df_true.sort_index(inplace=True)

    # # df_true = pd.get_dummies(df_true)[ls_cols]
    ls_classes = df_true.unique().tolist()
    cm = confusion_matrix(df_true, df_pred)
    # normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt_confusion_matrix = _plot_adjusted_heatmap_from_confusion_matrix(
        cm,
        'sirocco')
    x_tick_labels = _add_sample_size_to_xtick_labels(df_pred, ls_classes)
    y_tick_labels = _add_sample_size_to_xtick_labels(df_true, ls_classes)

    plt.ylabel('True label')  # , fontsize=9)
    plt.xlabel('Predicted label')  # , fontsize=9)
    plt_confusion_matrix.set_xticklabels(
        x_tick_labels, rotation=90, ha='center')
    plt_confusion_matrix.set_yticklabels(y_tick_labels, rotation=0, ha='right')

    path2save = os.path.join(output_dir, '{}-confusion-matrix.pdf'.format(
        target2predict))
    plt_confusion_matrix.get_figure().savefig(path2save,
                                              bbox_inches='tight')

    # Top features
    # df_top_features = res_combined.feature_importance.view(pd.DataFrame)
    plot_importance_topx_features(25,
                                  res_combined.feature_importance.view(
                                      pd.DataFrame),
                                  taxa,
                                  target2predict, output_dir)
    plt.show()

    # ! Evaluating separate-omics
    print('\nEvaluating separate-omics classifiers...')
    # extract accuracy for each omics run
    ls_micro = [
        x for x in ls_features if x.startswith('F_micro_')]
    ls_metabo = [
        x for x in ls_features if x.startswith('F_metabo_')]
    ls_proteome = [
        x for x in ls_features if x.startswith('F_proteo_')]

    df_omics_metrics = run_omics_separately(
        ls_micro=ls_micro,
        ls_metabo=ls_metabo,
        ls_proteome=ls_proteome,
        str_target=target2predict,
        df_data=df_data,
        output_res_combined=output_res_combined,
        res_combined=res_combined,
        seed=seed
    )

    # predictive accuracy and AUCs of individual vs. combined omics datasets
    sample_count = res_combined.predictions.view(pd.Series).shape[0]
    for metric in ['Accuracy', 'ROC_AUC_macro', 'PRC_AUC_macro']:
        fig_omics_metrics = plot_metric_omics(
            df_omics_metrics, sample_count, target2predict, metric)
        plt.tight_layout()
        plt.show()
        fig_omics_metrics.savefig(os.path.join(
            output_dir, '{}-omics-{}.pdf'.format(target2predict, metric)),
            bbox_inches="tight")

    # ! Evaluating feature abundances
    print('\nEvaluating feature abundances for combined classifier...')
    plot_abundances_top40_features(
        df_data_orig=df_data,
        df_top_features=res_combined.feature_importance.view(pd.DataFrame),
        taxa=taxa,
        str_target=target2predict,
        ls_class_order=ls_class_order,
        dic_palette=dic_color_palette,
        output_dir=output_dir)
    plt.show()
