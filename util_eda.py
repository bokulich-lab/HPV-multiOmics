import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, FunctionTransformer)
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA

import matplotlib as mpl
from matplotlib.pylab import plt

plt.rcParams['axes.grid'] = True
plt.style.use('seaborn-pastel')
mpl.rcParams['figure.dpi'] = 250


def plot_data_avail_per_target(df_data, ls_targets):
    """
    Function plotting data omics availability
    in `df_data` per target class in `ls_targets`
    """
    ls_microbiome_cols = [
        x for x in df_data.columns if x.startswith('F_micro')]
    ls_metabolome_cols = [
        x for x in df_data.columns if x.startswith('F_metabo')]
    ls_proteome_cols = [x for x in df_data.columns if x.startswith('F_proteo')]

    fontsize = 12

    reverse_ls_targets = list(reversed(ls_targets))
    for target in reverse_ls_targets:
        # get unique values for target
        unique_classes = df_data[target].unique().tolist()

        # init frac dataframe
        df_frac = pd.DataFrame(
            columns=['Microbiome', 'Metabolome', 'Immunoproteome'],
            index=unique_classes)

        for targ_class in unique_classes:
            # targ_class = 'High'
            if str(targ_class) == 'nan':
                class_w_targ = df_data[df_data[target].isna()]
            else:
                class_w_targ = df_data[df_data[target] == targ_class]

            nb_samples_of_targ_class = class_w_targ.shape[0]
            if nb_samples_of_targ_class != 0:
                micro_not_avail = class_w_targ \
                    .loc[:, ls_microbiome_cols] \
                    .isnull().all(axis=1).sum()
                df_frac.loc[targ_class, 'Microbiome'] = 1 - (
                    micro_not_avail / nb_samples_of_targ_class)

                metabo_not_avail = class_w_targ \
                    .loc[:, ls_metabolome_cols] \
                    .isnull().all(axis=1).sum()
                df_frac.loc[targ_class, 'Metabolome'] = 1 - (
                    metabo_not_avail / nb_samples_of_targ_class)

                proteo_not_avail = class_w_targ \
                    .loc[:, ls_proteome_cols] \
                    .isnull().all(axis=1).sum()
                df_frac.loc[targ_class, 'Immunoproteome'] = 1 - (
                    proteo_not_avail / nb_samples_of_targ_class)

        # plot fractions
        df_frac.T.plot.bar(rot=0, title=target, grid=True,
                           figsize=(8, 5), fontsize=fontsize)
        plt.title(target, fontsize=fontsize)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylabel('Fraction of samples with values', fontsize=fontsize)
        plt.show()

        # print counts
        print('Absolute counts:')
        print(df_data[target].value_counts(dropna=False))


def return_pcoa_metrics_microbiome(dic_beta_result, dict_variance,
                                   df_data, ls_targets):
    """
    Function save all PCoA beta diversity metrics
    with target values from `ls_targets` & `df_data`
    into one dataframe and save explained variance in dic_var.
    """
    dict_variance = {}
    # save explained variance into dictionary
    dict_variance['Microbiome_jaccard'] = (
        dic_beta_result['jaccard_pcoa_res'].proportion_explained[0],
        dic_beta_result['jaccard_pcoa_res'].proportion_explained[1])
    dict_variance['Microbiome_braycurtis'] = (
        dic_beta_result['braycurtis_pcoa_res'].proportion_explained[0],
        dic_beta_result['braycurtis_pcoa_res'].proportion_explained[1])

    # save all diversity metrics with target values into one df
    # beta PCoA data: from braycurtis
    df_pcoa_bc = dic_beta_result['braycurtis_pcoa_res'].samples.copy(deep=True)
    df_pcoa_bc.columns = [x + '_braycurtis' for x in df_pcoa_bc.columns]
    df_pcoa_bc.index.name = 'SampleID'

    # merge with PCoA data: from Jaccard
    df_pcoa_jac = dic_beta_result['jaccard_pcoa_res'].samples.copy(deep=True)
    df_pcoa_jac.columns = [x + '_jaccard' for x in df_pcoa_jac.columns]
    df_pcoa_jac.index.name = 'SampleID'
    df_pcoa_both = df_pcoa_bc.merge(
        df_pcoa_jac, left_index=True, right_index=True, how='left')

    # join target dataset
    df_pcoa_micro = df_pcoa_both.merge(
        df_data[ls_targets], left_index=True,
        right_index=True, how='left')
    # df_pcoa_micro.to_csv(os.path.join(output_dir, 'df_pcoa_micro.csv'))
    # df_pcoa_micro.shape

    return df_pcoa_micro, dict_variance


def calc_pca_metrics_metabolome(dict_variance, df_data, ls_targets):
    """
    Function performing PCA on log-transformed and scaled metabolome features
    in `df_data` and returning dataframe with PCA metrics and targets
    saved.
    """
    # get metabolites in df_data
    ls_metabolome_cols = [x for x in df_data.columns
                          if x.startswith('F_metabo')]
    df_metabolites = df_data[ls_metabolome_cols].copy(deep=True)

    # apply log transformation on metabolites
    transformer = FunctionTransformer(np.log1p, validate=True)
    arr_metabolites_log = transformer.transform(df_metabolites)

    df_metabolites_log = pd.DataFrame(arr_metabolites_log,
                                      columns=df_metabolites.columns,
                                      index=df_metabolites.index)
    # standardize metabolites
    arr_metabolites_scaled = StandardScaler().fit_transform(df_metabolites_log)
    df_metabolites_scaled = pd.DataFrame(arr_metabolites_scaled,
                                         columns=df_metabolites_log.columns,
                                         index=df_metabolites_log.index)

    # perform PCA on scaled metabolites
    pca_2d = PCA(n_components=2)
    pca_metabolites = pca_2d.fit_transform(df_metabolites_scaled)

    metab_prop_explained_pc1 = pca_2d.explained_variance_ratio_[0]
    metab_prop_explained_pc2 = pca_2d.explained_variance_ratio_[1]

    dict_variance['Metabolome'] = (
        metab_prop_explained_pc1, metab_prop_explained_pc2)

    df_pca_metabolites = pd.DataFrame(data=pca_metabolites,
                                      columns=['PC1', 'PC2'],
                                      index=df_metabolites_scaled.index)

    # join targets
    df_pca_metab = df_pca_metabolites.merge(
        df_data[ls_targets], left_index=True,
        right_index=True, how='left')

    # df_pca_metab.to_csv(os.path.join(output_dir, 'df_pca_metab.csv'))
    # df_pca_metab.shape
    return df_pca_metab, dict_variance


def calc_pca_metrics_proteome(dict_variance, df_data, ls_targets):
    """
    Function performing PCA on log-transformed and scaled proteome features
    in `df_data` and returning dataframe with PCA metrics and targets
    saved.
    """
    # get metabolies in df_data
    ls_proteome_cols = [x for x in df_data.columns
                        if x.startswith('F_proteo_')]
    df_proteome = df_data[ls_proteome_cols].copy(deep=True)

    # apply log transformation on proteome
    transformer = FunctionTransformer(np.log1p, validate=True)
    arr_prot_log = transformer.transform(df_proteome)

    df_prot_log = pd.DataFrame(arr_prot_log,
                               columns=df_proteome.columns,
                               index=df_proteome.index)
    # standardise proteome
    arr_prot_scaled = StandardScaler().fit_transform(df_prot_log)
    df_prot_scaled = pd.DataFrame(arr_prot_scaled, columns=df_prot_log.columns,
                                  index=df_prot_log.index)

    # perform PCA on scaled proteome
    pca_2d_prot = PCA(n_components=2)
    pca_prot = pca_2d_prot.fit_transform(df_prot_scaled)

    proteo_prop_explained_pc1 = pca_2d_prot.explained_variance_ratio_[0]
    proteo_prop_explained_pc2 = pca_2d_prot.explained_variance_ratio_[1]

    dict_variance['Immunoproteome'] = (proteo_prop_explained_pc1,
                                       proteo_prop_explained_pc2)

    df_pca_prot = pd.DataFrame(data=pca_prot, columns=['PC1', 'PC2'],
                               index=df_prot_scaled.index)

    # join targets
    df_pca_proteo = df_pca_prot.merge(
        df_data[ls_targets], left_index=True, right_index=True, how='left')
    # df_pca_proteo.to_csv(os.path.join(output_dir, 'df_pca_proteo.csv'))
    # df_pca_proteo.shape

    return df_pca_proteo, dict_variance


def merge_all_pca_metrics(df_pcoa_micro, beta_div2_choose,
                          df_pca_metab, df_pca_proteo):
    """
    Function merging all omics pc(o)a metric dataframes
    into one. For microbiome beta diversity metric selected in
    `beta_div2_choose` is displayed (options: "jaccard" and
    "braycurtis") are currently available.
    """
    # get microbiome
    df_pca_all = df_pcoa_micro.copy(deep=True)
    df_pca_all.reset_index(inplace=True)
    df_pca_all.rename(columns={'PC1_'+beta_div2_choose: 'PC1',
                               'PC2_'+beta_div2_choose: 'PC2'}, inplace=True)
    col2drop = [x for x in df_pca_all.columns if (
        x.startswith('PC1_') or x.startswith('PC2_'))]
    df_pca_all.drop(columns=col2drop, inplace=True)
    df_pca_all['Omics'] = 'Microbiome'
    # print(df_pca_all.shape)

    # add metabolome
    df_pca_metab['Omics'] = 'Metabolome'
    df_pca_metab_ed = df_pca_metab.reset_index()
    df_pca_metab_ed.rename(columns={'sample-id': 'SampleID'}, inplace=True)
    df_pca_all = pd.concat([df_pca_all, df_pca_metab_ed])
    # print(df_pca_all.shape)

    # add proteome
    df_pca_proteo['Omics'] = 'Immunoproteome'
    df_pca_proteo_ed = df_pca_proteo.reset_index()
    df_pca_proteo_ed.rename(columns={'sample-id': 'SampleID'}, inplace=True)
    df_pca_all = pd.concat([df_pca_all, df_pca_proteo_ed])
    # print(df_pca_all.shape)

    return df_pca_all


def run_manova(df_data, ls_dep_features, str_indep_feature):
    """
    Function running manova for given dependent (`ls_dep_features`)
    and independent features (`str_indep_feature`) returning p-value of
    Pillai\'s trace.
    """
    df_selected = df_data[ls_dep_features+[str_indep_feature]].copy(deep=True)

    # clean up column names for R-type formula (only as a backup)
    for item in [')', '(', '+', '-', '_', ',', ':', '/',
                 ' ', ';', "'", "[", "]"]:
        df_selected.columns = df_selected.columns.str.replace(item, '')
        ls_dep_features = [x.replace(item, '') for x in ls_dep_features]
        str_indep_feature = str_indep_feature.replace(item, '')

    # create R-type formula
    str_formula_dependent = '(' + ls_dep_features[0]
    for meta in ls_dep_features[1:]:
        str_formula_dependent += ' + ' + str(meta)
    str_formula = str_formula_dependent + ')  ~ ' + str_indep_feature

    maov = MANOVA.from_formula(str_formula, data=df_selected)

    pvalue_pillai = maov.mv_test(
    ).results['Intercept']['stat'].loc["Pillai\'s trace", "Pr > F"]

    return pvalue_pillai
