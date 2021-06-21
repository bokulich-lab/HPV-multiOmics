import os
import pandas as pd
import qiime2
from qiime2.plugins import (feature_table as ft,
                            taxa as q2taxa,
                            rescript)


def read_patient_data(path2data,
                      source_patient_md='sample_md.tsv',):
    """
    Function reading sample metadata from path2data, processing and tagging
    patient covariates with prefix `F_pcov_`.
    """

    print('\nGetting patient covariates (tagged with F_pcov_)...')

    # ! read sample data - shape: (102, 96)
    sample_md = qiime2.Metadata.load(
        os.path.join(path2data, source_patient_md))
    # drop negative controls - shape: (99, 96)
    sample_md = sample_md.filter_ids(
        [i for i in sample_md.get_ids() if i not in ['AK15-3975', 'Negctrl',
                                                     'NTC']])
    # transform to df
    df_sample_md = sample_md.to_dataframe()
    # add feature Ethnicity: "Latina"
    df_sample_md['Latina'] = df_sample_md['Ethnicity'].dropna().apply(
        lambda x: 1 if '3' in x else 0)
    # select only patient covariates
    ls_patient_covariates = ['Age', 'pH', 'BMI', 'Latina',
                             'L. crispatus', 'L. gasseri', 'L. iners',
                             'L. jensenii']
    # select fields required for target definition
    ls_other_variables = ['original-sample-id', 'Group']
    df_all = df_sample_md[ls_patient_covariates+ls_other_variables].copy(
        deep=True)
    # add prefix identifying patient covariates: F_pcov_
    ls_cols2replace = [x for x in df_all.columns if (
        x not in ls_other_variables)]
    ls_newname = ['F_pcov_' + x for x in ls_cols2replace]
    df_all.rename(columns=dict(zip(ls_cols2replace, ls_newname)), inplace=True)
    print('# of added patient covariates: {}'.format(
        len(ls_patient_covariates)))

    return df_all


def add_immunoproteo_data(df_all,
                          path2data,
                          source_immuno1='aging-plus-immune-checkpoint.txt',
                          source_immuno2='patient_data_table.qza'):
    """
    Function reading immunoproteome data from path2data, processing and tagging
    these features with prefix `F_proteo` and left joining with df_all.
    Additionally function returns selected columns of source_immuno2 to be
    used cancer biomarkers.
    """

    print('\nGetting immmunoproteome data (tagged with F_proteo_)')
    # ! read & transform immuno1 - shape: (78, 24)
    aging = qiime2.Metadata.load(os.path.join(path2data, source_immuno1))
    aging_table = aging.to_dataframe().drop(
        ['IL-6', 'IL-10', 'Leptin'], axis=1)
    aging_table.index.name = 'sample-id'
    # add prefix identifying proteo features: F_proteo_
    aging_table.columns = ['F_proteo_' +
                           x for x
                           in aging_table.columns]
    print('Shape of added immuno1 data: {}'.format(aging_table.shape))
    # merge w df_all on index (sample-ID)
    df_all = df_all.merge(aging_table, how='left',
                          left_index=True, right_index=True)

    # ! read & transform immuno2 - shape: (73, 44)
    patient_data = qiime2.Artifact.load(
        os.path.join(path2data, source_immuno2))
    df_patient_data = patient_data.view(pd.DataFrame).drop(
        ['Age', 'pH', 'BMI', 'L. crispatus', 'L. gasseri',
         'L. iners', 'L. jensenii'], axis=1)
    df_patient_data.index.name = 'sample-id'
    # add prefix identifying proteo features: F_proteo_
    ls_biomarkers_cancer = df_patient_data.columns.tolist()
    df_patient_data.columns = ['F_proteo_' +
                               x for x
                               in df_patient_data.columns]
    print('Shape of added immuno2 data: {}'.format(df_patient_data.shape))
    # merge w df_all on index (sample-ID)
    df_all = df_all.merge(df_patient_data, how='left',
                          left_index=True, right_index=True)

    print('\nShape of new df_all: {}'.format(df_all.shape))
    return df_all, ls_biomarkers_cancer


def add_microbiome_data(df_all,
                        path2data,
                        source_microbiome='table-w-phylum-filt-rarefied.qza'):
    """
    Function reading microbiome data from path2data, processing features and
    tagging them with prefix `F_micro` and left joining with df_all.
    """

    print('\nGetting microbiome data (tagged with F_micro_)...')

    # ! read - shape: (100, 849)
    table = qiime2.Artifact.load(os.path.join(path2data, source_microbiome))

    # # todo: outsource below 2 steps
    # # filter out sequencing controls - shape: (99, 849)
    # table, = ft.actions.filter_samples(
    #     table, metadata=sample_md, where="[Group]!='SequencingControl'")
    # # evenly subsample prior to training models - shape: (98, VAR)
    # table, = ft.actions.rarefy(table, 50000)
    # todo: closing outsourcing

    # transform to df
    df_table = table.view(pd.DataFrame)
    # add prefix identifying microbiome features: F_micro_
    df_table.columns = ['F_micro_' +
                        x for x in df_table.columns]
    df_table.index.name = 'sample-id'
    print('Shape of added microbiome data: {}'.format(df_table.shape))
    # merge w df_all on index (sample-ID)
    df_all = df_all.merge(df_table, how='left',
                          left_index=True, right_index=True)

    return df_all


def add_metabolome_data(
        df_all,
        path2data,
        source_metabolome='CC_metabolome_scaledPeaks_correctIDs.qza',
        source_metab_md='metabolite_metadata_CC.txt'):
    """
    Function reading metabolome data from path2data, processing features and
    tagging them with prefix `F_metabo` and left joining with df_all.
    """
    print('\nGetting metabolome data (tagged with F_metabo_)')
    # ! read - shape: (78, 475)
    metabolites = qiime2.Artifact.load(
        os.path.join(path2data, source_metabolome))

    # transform to df
    df_metabolites = metabolites.view(pd.DataFrame)
    # Filter out sequencing controls - shape: (77, 475)
    df_metabolites = df_metabolites[~df_metabolites.index.isin(['AK15-3975',
                                                                'Negctrl',
                                                                'NTC'])].copy(
        deep=True)

    # read metabolites metadata to find out which are lipids and which are not
    df_metadata = qiime2.Metadata.load(
        os.path.join(path2data, source_metab_md)).to_dataframe()
    ls_lipids = df_metadata[df_metadata['SUPER PATHWAY']
                            == 'Lipid'].index.tolist()

    # add prefix identifying metabolome features: F_metabo_
    df_metabolites.columns = ['F_metabo_lipid_'+x if x in ls_lipids
                              else 'F_metabo_other_'+x
                              for x
                              in df_metabolites.columns]
    df_metabolites.index.name = 'sample-id'

    print('Shape of added metabolite data: {}'.format(df_metabolites.shape))
    # merge w df_all on index (sample-ID)
    df_all = df_all.merge(df_metabolites, how='left',
                          left_index=True, right_index=True)
    return df_all


def perform_taxonomic_classification(
    path2data,
    source_gtdb='data-raw/2018.04-cervical-cancer/taxonomy-gtdb-bespoke.qza',
    source_gg_tax='data-raw/2018.04-cervical-cancer/taxonomy-gg.qza',
    source_unifor='data-raw/2018.04-cervical-cancer/taxonomy-gtdb-uniform.qza',
    source_all_micro_seq='data-raw/2018.04-cervical-cancer/table-w-phylum.qza',
    source_sequence_matching='data-raw/2018.04-cervical-cancer/rep-seqs.qza'
):
    """
    Function assigning consensus taxonomy to microbiome
    sequences.
    """
    path2merged_tax = os.path.join(path2data, 'merged_taxonomy.tsv')
    path2new_cons = os.path.join(path2data, 'taxonomy-new-consensus.qza')

    if os.path.isfile(path2merged_tax) & os.path.isfile(path2new_cons):
        print('Reading existing taxonomic classification')
        merged_taxonomy = pd.read_csv(path2merged_tax,
                                      sep='\t')
        merged_taxonomy.set_index('Feature ID', inplace=True)
        taxonomy_qza = qiime2.Artifact.load(path2new_cons)

    else:
        print('Performing taxonomic classification')
        # read taxonomies
        bespoke_taxonomy = qiime2.Artifact.load(source_gtdb)
        old_taxonomy = qiime2.Artifact.load(source_gg_tax)
        uniform_taxonomy = qiime2.Artifact.load(source_unifor)

        # read sequences & seq matching
        table = qiime2.Artifact.load(source_all_micro_seq)
        mean_abundances = ft.actions.relative_frequency(
            table).relative_frequency_table.view(
                pd.DataFrame).mean()
        # matching sequence ids to actual sequences
        seqs = qiime2.Artifact.load(source_sequence_matching).view(pd.Series)

        # merge all to one df
        merged_taxonomy = pd.concat([mean_abundances,
                                    old_taxonomy.view(pd.DataFrame),
                                    bespoke_taxonomy.view(pd.DataFrame),
                                    uniform_taxonomy.view(pd.DataFrame), seqs],
                                    axis=1, sort=True).dropna().sort_values(
                                        by=0, ascending=False)
        merged_taxonomy = merged_taxonomy.drop('Confidence', 1)
        merged_taxonomy.columns = ['Mean Relative Frequency', 'Greengenes',
                                   'GTDB-bespoke-weights',
                                   'GTDB-uniform-weights', 'Sequence']
        merged_taxonomy.index.name = 'id'

        # process individual taxonomy assignments
        taxa_copy = merged_taxonomy.copy()
        taxa_copy['Greengenes'] = taxa_copy['Greengenes'].apply(
            lambda x: x.replace('Gardnerella', 'Bifidobacterium').replace(
                'Enterobacteriales', 'Enterobacterales'))
        taxa_copy['GTDB-bespoke-weights'] = taxa_copy[
            'GTDB-bespoke-weights'].apply(
            lambda x: x.replace('Fannyhessea', 'Atopobium'))
        taxa_copy['GTDB-uniform-weights'] = taxa_copy[
            'GTDB-uniform-weights'].apply(
                lambda x: x.replace('Fannyhessea', 'Atopobium'))

        # perform consensus taxonomy
        taxa_copy.index.name = 'Feature ID'
        taxa_to_merge = []
        for c in taxa_copy.columns[1:4]:
            t = taxa_copy[c]
            t.name = 'Taxon'
            taxa_to_merge.append(qiime2.Artifact.import_data(
                'FeatureData[Taxonomy]', t))

        super_merged_taxa, = rescript.actions.merge_taxa(
            taxa_to_merge, mode='super', rank_handle_regex='^[dkpcofgs]__')

        super_taxa = super_merged_taxa.view(pd.Series)
        super_taxa.name = 'Consensus Taxonomy'

        merged_taxonomy = pd.concat([merged_taxonomy, super_taxa], axis=1)
        merged_taxonomy = merged_taxonomy[['Mean Relative Frequency',
                                           'Greengenes',
                                           'GTDB-bespoke-weights',
                                           'GTDB-uniform-weights',
                                           'Consensus Taxonomy',
                                           'Sequence']]
        merged_taxonomy.index.name = 'Feature ID'
        merged_taxonomy.to_csv(os.path.join(
            path2data, 'merged_taxonomy.tsv'), sep='\t')

        # extract consensus taxonomy
        taxonomy = merged_taxonomy[['Consensus Taxonomy']]
        taxonomy.columns = ['Taxon']
        taxonomy.index.name = 'Feature ID'
        taxonomy_qza = qiime2.Artifact.import_data('FeatureData[Taxonomy]',
                                                   taxonomy)
        taxonomy_qza.save(os.path.join(
            path2data, 'taxonomy-new-consensus.qza'))

    # return merged taxonomy and consensus taxonomy
    return merged_taxonomy, taxonomy_qza


def add_targets(df_dataset, path2data, taxonomy_qza,
                source_t_inflammation='inflammation score data.txt'):
    """
    Function adding categorical targets to df_dataset,
    namely: T_inflammation_score,
    T_disease_state, T_lactobacillus_dominance and T_pH
    """
    print('Adding targets')
    print('\nAdd T_inflammation_score')
    # numeric values
    inflammation_float = pd.read_csv(os.path.join(
        path2data, source_t_inflammation), sep='\t', index_col=0)
    # inflammation_float.index = [str(i) for i in inflammation_float.index]
    inflammation_float.drop(columns='Group', inplace=True)
    inflammation_float.drop(columns='Patient ID', inplace=True)
    inflammation_float.rename(
        columns={'Genital inflammatory score': 'T_infl_score_flt'},
        inplace=True)
    # group to categorical target
    inflammation_float['T_inflammation'] = ['None' if s == 0 else 'Low' if s <
                                            5 else 'High'
                                            for s in inflammation_float[
                                                'T_infl_score_flt']
                                            ]
    # merge
    df_inclT = df_dataset.merge(inflammation_float, how='left',
                                left_on='original-sample-id',
                                right_index=True)
    print(df_inclT['T_inflammation'].value_counts(dropna=False))

    print('\nAdd T_disease_state')
    df_inclT.rename(columns={'Group': 'T_disease_state'}, inplace=True)
    print(df_inclT['T_disease_state'].value_counts(dropna=False))

    print('\nAdd T_lactobacillus_dominance')
    # collapse feature table at genus level to assess Lactobacillus dominance
    ls_microb_features = [
        x for x in df_dataset.columns if x.startswith('F_micro_')]
    ls_microb_features_orig = [
        x.replace('F_micro_', '') for x in ls_microb_features]
    df_table = df_dataset[ls_microb_features].copy(deep=True)
    df_table.rename(columns=dict(
        zip(ls_microb_features, ls_microb_features_orig)), inplace=True)

    table = extract_microbiome_artifact(df_dataset)
    collapsed_table, = q2taxa.actions.collapse(table, taxonomy_qza, level=6)

    collapsed_table, = ft.actions.relative_frequency(collapsed_table)

    # LD is defined as >= 0.8 relative frequence Lactobacillus
    genus = ('Bacteria;Firmicutes;Bacilli;'
             'Lactobacillales;Lactobacillaceae;Lactobacillus')
    genus = collapsed_table.view(pd.DataFrame)[genus]
    lacto_dominance = (genus >= 0.8).replace([False, True], ['NLD', 'LD'])
    lacto_dominance.name = 'T_lactobacillus_dominance'
    lacto_dominance.index.name = 'sample-id'
    df_lacto_dominance = lacto_dominance.to_frame()

    df_inclT = df_inclT.merge(df_lacto_dominance, how='left', left_index=True,
                                                      right_index=True)
    print(df_inclT['T_lactobacillus_dominance'].value_counts(dropna=False))

    print('\nAdd T_pH')
    df_inclT['T_pH'] = df_inclT['F_pcov_pH'].replace(
        [4.5, 5., 5.5, 7., 6., 7.5, 6.5], ['Low', 'Low', 'High',
                                           'High', 'High', 'High', 'High'])
    print(df_inclT['T_pH'].value_counts(dropna=False))
    print('\nShape of df with targets: {}'.format(df_inclT.shape))

    return df_inclT


def extract_microbiome_artifact(df_dataset):
    """
    Function that extracts all microbiome features
    (tagged with prefix 'F_micro_')
    from df_dataset and returns a QIIME2 FeatureTable[Frequency] Artifact
    """
    ls_microb_features = [
        x for x in df_dataset.columns if x.startswith('F_micro_')]
    ls_microb_features_orig = [
        x.replace('F_micro_', '') for x in ls_microb_features]

    df_table = df_dataset[ls_microb_features].copy(deep=True)

    df_table.rename(columns=dict(
        zip(ls_microb_features, ls_microb_features_orig)), inplace=True)

    table = qiime2.Artifact.import_data(
        'FeatureTable[Frequency]', df_table)

    return table
