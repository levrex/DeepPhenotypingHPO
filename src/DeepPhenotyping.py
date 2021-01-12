import argparse
from argparse import RawTextHelpFormatter
import ast 
from bokeh.palettes import Colorblind8
import numpy as np
import obonet
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from phenopy import generate_annotated_hpo_network
from phenopy.score import Scorer
import re
from sklearn.cluster import KMeans
import sys


# example: python src/DeepPhenotyping.py -sample actin -visual TSNE

def load_hpo_tree(url):
    """ 
    Load the HPO tree
    
    Input:
        url = pathway to the hpo obo-file
    
    Output:
        graph = hpo graph
        id_to_name = dictionary that maps each HPO-concept to a phenotypic description
        name_to_id = dictionary that maps each phenotypic description to an HPO-concept
    """
    
    graph = obonet.read_obo(url)

    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
    return graph, id_to_name, name_to_id

def infer_HPO_translate(row, column_to_hpo):
    """
    Translate the phenotypic descriptions to HPO-concepts
    
    Output:
        hpo_list = phenotypic profiles per patient
    """
    hpo_list = []
    cols = list(row.index)
    for i in cols:
        val = row[i]
        #print(val)
        if val == 1:
            col = i.lower().strip()
            if (col in column_to_hpo.keys()) == False :
                print(col, col in column_to_hpo.keys())
            if col in column_to_hpo.keys():
                if type(column_to_hpo[col]) == list:
                    hpo_list.extend(column_to_hpo[col])
                else :
                    hpo_list.append(column_to_hpo[col])
    return hpo_list

def infer_HPO(row):
    """
    Creating phenotypic profile lists
    
    Output:
        hpo_list = phenotypic profiles per patient
    """
    hpo_list = []
    cols = list(row.index)
    for i in cols:
        val = row[i]
        if val == 1:
            hpo_list.append(i)
    return hpo_list

def initialize_hrss():
    """
    Initialize the Hybrid Relative Semantic Similarity Scorer.
    """
    # data directory
    phenopy_data_directory = os.path.join(os.getenv('HOMEPATH'), '.phenopy\\data')

    # files used in building the annotated HPO network
    obo_file = os.path.join(phenopy_data_directory, 'hp.obo')
    disease_to_phenotype_file = os.path.join(phenopy_data_directory, 'phenotype.hpoa')

    # if you have a custom ages_distribution_file, you can set it here.
    #ages_distribution_file = os.path.join(phenopy_data_directory, 'xa_age_stats_oct052019.tsv')

    hpo_network, alt2prim, disease_records = \
        generate_annotated_hpo_network(obo_file,
                                       disease_to_phenotype_file,
                                       )

    scorer = Scorer(hpo_network)
    return scorer

def getMostSpecificFeatures(explore_df, cluster, columns, weight_list, save_as=''):
    """
    Acquire most specific features according to the phenotypic specificity
    
    Input:
        explore_df = dataframe consisting of patient phenotypic profiles
        cluster = specify cluster of interest
        weight_list = list with weights for each phenotype
        save_as = title used for saving the elbow plot figure
            (no title implies that the figure won't be saved)
    """  
    #print(explore_df['cluster'].unique())
    N = len(explore_df[explore_df['cluster']==int(cluster)])
    
    #print(columns)
    prob = func.PhenotypicSpecifictyOccurrence(explore_df[columns], weights=weight_list, cluster=int(cluster), topN=10, pat_frac=.001, rename=False, divide_occurrence=True) # labels=lbls, 
    pheno_ACTG = list(prob.index) # topN=10, 
    #plt.clf()
    #print(prob)
    if save_as != '':
        #print('LOADING 2')
        plt.clf()
        prob.plot(kind='barh',figsize=(9,9), title='Most specific phenotypic features in Cluster %s (n=%s)' % (cluster, str(N)))
        fig = plt.gcf()
        fig.savefig('figures/phenospec_top10_%s' % (save_as))
    else :
        prob.plot(kind='barh',figsize=(9,9), title='Most specific phenotypic features in Cluster %s (n=%s)' % (cluster, str(N)))
    return

def calculate_hrss(hp_patients):
    """
    Provide a phenotypic profile to calculate the distance/ similarity between patients.
    
    Input:
        hp_patients = phenotypic profile per patient
        
    """
    scorer = initialize_hrss()
    l = []
    for j in range(len(hp_patients)):
        for i in range(len(hp_patients)):
            l.append(scorer.score_term_sets_basic(hp_patients[i], hp_patients[j]))

    l_cleaned = [x if str(x) != 'nan' else 0 for x in l ]

    l2 = []
    n = len(hp_patients)
    for j in range(len(hp_patients)):
        l2.append(l_cleaned[n*j:n+n*j])
    df_hrss = pd.DataFrame(l2)
    return df_hrss


def process_data(DATA, METRIC, VISUAL='TSNE', FIG=True, PHE=False):
    """

        python src/DeepPhenotyping.py -data data/deepdata/distance_matrix_IARS_hrss.csv -metric none -visual TSNE -fig 1 -phe 1 
    """
    # get title of file
    try:
        m = re.search('([^/]*)\.(?:csv|txt|tsv|xlsx|xls)$', DATA)
        title = m.group(1)
    except:
        title = 'NA'
        
    df_all = pd.read_csv(DATA, index_col=0)
    
    
    hp_patients = list(df_all.apply(lambda x: infer_HPO(x) , axis= 1))

    # Calculate distance / similarity between patients
    if METRIC == 'hrss': # SEMANTIC MATCHING
        if os.path.isfile('data/deepdata/distance_matrix_hrss.csv'):
            df_hrss = pd.read_csv('data/deepdata/distance_matrix_hrss.csv', index_col=0)
        else :
            df_hrss = calculate_hrss(hp_patients)
            df_hrss.to_csv('data/deepdata/distance_matrix_hrss.csv') # save HRSS data
        X_trans = df_hrss[df_hrss.columns[:113]]
    elif METRIC in ['jaccard', 'dice', 'hamming']: # EXACT MATCHING
        cols = list(df_all.loc[:, ~df_all.columns.isin(['Category', 'Id', 'Protein', 'Origin'])].columns)
        df = df_all.copy()
        pairwise = pd.DataFrame(
            squareform(pdist(df[cols], metric=METRIC)),
            columns = df['Id'],
            index = df['Id']

        )
        X_trans = pairwise.values 
    
    elif METRIC == 'none' : # Indicate that you don't need to perform another distance calculation
        X_trans = df_all.values
    
    if FIG:
        title_fig = title
    else :
        title_fig = ''
        
    k = func.elbowMethod(X_trans, method='kmeans', n=max(9, len(X_trans)), save_as=title_fig) 
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

    df_all['cluster'] = kmeans.fit_predict(X_trans)
    
    print('Nr of clusters = %s' % (k))
    
    if 'Id' not in df_all.columns:
        print('No Id column found. We will use index instead!')
        df_all =df_all.rename_axis('Id').reset_index()
    if 'Category' not in df_all.columns:
        print('No Category found. We will assign random categories!')
        df_all['Category'] = np.random.randint(0, 3, df_all.shape[0])
    #print(df_all)
    if VISUAL=='TSNE':
        func.makeTSNE_Cluster2(X_trans,df_all['Id'],  df_all['cluster'].astype(str), df_all['Category'].astype(str),
                          title='%s' % (title), clusters=k, pal=[Colorblind8, Colorblind8], seed=2)
    elif VISUAL=='PCA':
        func.makePCA(X_trans,df_all['Id'],  df_all['cluster'].astype(str),df_all['Category'].astype(str),
                      title='%s' % (title), clusters=k, pal=[Colorblind8, Colorblind8], seed=2)
    
    func.clusterMembership(df_all['Category'], df_all['cluster'], save_as = title_fig)
    
    # Explore the clusters that were made
    #explore_df =  pd.concat([df_all, df_all['cluster']], axis=1)
    explore_df = df_all.copy()
    l_ignore = ['Category', 'Id', 'Protein', 'Origin'] # , 'cluster'
    columns = list(explore_df.loc[:, ~explore_df.columns.isin(l_ignore)])
    
    l_ignore.append('cluster')
    columns_raw = list(explore_df.loc[:, ~explore_df.columns.isin(l_ignore)])
    
    if PHE:
        df_hpo = pd.read_table(r'hpo/util/annotation/phenotype_to_genes.txt', sep='\t|<tab>', engine='python') 
        weight_list =  func.getNumberOfGenes(df_all[columns_raw].columns, df_hpo=df_hpo)
        for i in list(df_all['cluster'].unique()):
            title_fig = 'actin_cluster_%s' % (str(i))
            getMostSpecificFeatures(df_all, int(i), columns, weight_list, save_as=title_fig)
  
    
def example_actin(METRIC, VISUAL='TSNE', FIG=True, PHE=False):
    if METRIC != 'occurrence':
        group_level = False
    else :
        group_level = True
    
    if group_level == False: # patient level
        # Prepare data
        df_actb = pd.read_csv(r'data/ACTB/ACTB_data.csv', sep=',', header=0)
        df_actg1 = pd.read_csv(r'data/ACTG1/ACTG1_data.csv', sep=',', header=0)
        df_actg1['Category'] = df_actg1['Category'] + ' (ACTG1)'
        df_actb['Category'] = df_actb['Category'] + ' (ACTB)'
        df_actb['Origin'] = 'ACTB'
        df_actg1['Origin'] = 'ACTG1'
        
        df_all = pd.concat([df_actg1, df_actb], ignore_index=True) # MERGE
        
        # Cleaning data
        df_all = df_all.rename(columns={"Retrognatia/micrognathia" : "Retrognathia/micrognathia", "Leukocytose": "Leukocytosis", "Trombocytopenie": "thrombocytopenia", "Other Psychiatric diagnosis (shizofrenia, depression, other, not autism, or ADHD/ADD)" : "Other Psychiatric diagnosis (schizophrenia, depression, other, not autism, or ADHD/ADD)", "Mid face hypoplasia": "Midface hypoplasia"})

        # Change categories
        df_all['Category'] = df_all['Category'].replace({'Other/Myosin binding (ACTG1)': 'Other (ACTG1)', 'Other/Fimbrin binding (ACTG1)': 'Other (ACTG1)', 'N-terminus (ACTG1)': 'Other (ACTG1)', 'N-terminus (ACTB)': 'Other (ACTB)'})

        # link each column in table to HPO
        file = open('columns_linked_to_hpo_actin.py', "r")
        contents = file.read()
        col2hpo = ast.literal_eval(contents[16:]) # skip first 16 characters -> to read directly
        file.close()
        


        # get HPO per patient
        hp_patients = list(df_all.apply(lambda x: infer_HPO_translate(x, col2hpo) , axis= 1))

        # Calculate distance / similarity between patients
        if METRIC == 'hrss': # SEMANTIC MATCHING
            if os.path.isfile('data/deepdata/distance_matrix_hrss.csv'):
                df_hrss = pd.read_csv('data/deepdata/distance_matrix_hrss.csv', index_col=0)
            else :
                df_hrss = calculate_hrss(hp_patients)
                df_hrss.to_csv('data/deepdata/distance_matrix_hrss.csv') # save HRSS data
            X_trans = df_hrss[df_hrss.columns[:113]]
        elif METRIC in ['jaccard', 'dice', 'hamming']: # EXACT MATCHING
            cols = list(df_all.loc[:, ~df_all.columns.isin(['Category', 'Id', 'Protein', 'Origin'])].columns)
            df = df_all.copy()
            pairwise = pd.DataFrame(
                squareform(pdist(df[cols], metric=METRIC)),
                columns = df['Id'],
                index = df['Id']

            )
            X_trans = pairwise.values 
        elif METRIC == 'none':
            X_trans = df_all.values
    else :
        df_all = pd.read_csv(r'data/deep_group/occurance_ratio_N6_and_ACTB_and_ACTG1.txt', sep='\t', header=0)
        df_all = df_all.apply(lambda x: x.astype('str').str.replace(',','.'))
        l_remove = ['actg1nterminus', 'actbnterminus', 'actbabs', 'actbtotal', 'NAT6']
        df_all =df_all[~(df_all['HPOcode'].isin(l_remove))].reset_index(drop=True)
        df_all['HPOcode'] = df_all['HPOcode'].replace({'actbgain': 'Gain of Function (ACTB)', 'actbloss': 'Loss of Function (ACTB)', 'actbother': 'Other (ACTB)', 'actg1gain': 'Gain of Function (ACTG1)', 'actg1loss': 'Loss of Function (ACTG1)', 'actg1other': 'Other (ACTG1)'})
        X_trans = [i[1:].astype(float) for i in df_all[columns].values]
    
    if FIG:
        title_fig = 'actin' 
    else :
        title_fig = ''
        
    k = func.elbowMethod(X_trans, method='kmeans', n=30, save_as=title_fig) 
    print('Nr of clusters = %s' % (k))
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df_all['cluster'] = kmeans.fit_predict(X_trans)
    
    if VISUAL=='TSNE':
        func.makeTSNE_Cluster2(X_trans,df_all['Id'],  df_all['cluster'].astype(str), df_all['Category'],
                          title='All_%s' % (METRIC), clusters=k, pal=[Colorblind8, Colorblind8], seed=2)
    elif VISUAL=='PCA':
        func.makePCA(X_trans,df_all['Id'],  df_all['cluster'].astype(str),df_all['Category'],
                      title='All_%s' % (METRIC), clusters=k, pal=[Colorblind8, Colorblind8], seed=2)
    
    func.clusterMembership(df_all['Category'], df_all['cluster'], save_as = title_fig)
    
    # Explore the clusters that were made
    #explore_df =  pd.concat([df_all, df_all['cluster']], axis=1)
    explore_df = df_all.copy()
    l_ignore = ['Category', 'Id', 'Protein', 'Origin'] # , 'cluster'
    columns = list(explore_df.loc[:, ~explore_df.columns.isin(l_ignore)])
    
    if PHE:
        df_hpo = pd.read_table(r'hpo/util/annotation/phenotype_to_genes.txt', sep='\t|<tab>', engine='python') 
        weight_list = func.getNumberOfGenesTranslation(df_all.columns[3:-1], col2hpo=col2hpo, df_hpo=df_hpo)
        for i in list(df_all['cluster'].unique()):
            #print('actin cluster %s' % (str(i)))
            title_fig = 'actin_cluster_%s' % (str(i))
            getMostSpecificFeatures(df_all, int(i), columns, weight_list, save_as=title_fig)
            
    print('Figures are written to figures/')    
    print('\nFinished Deep Phenotyping!')
    return    
    
def main(arguments):
    """
    Apply Case study Extractor on provided link (URL). 
    
    
    """
    d = vars(arguments)
    DATA = d['data']
    SAMPLE = d['sample']
    METRIC = d['metric']
    VISUAL = d['visual'] # dimension reduction
    FIG = bool(d['fig'])
    PHE = bool(d['phe'])
    #print(eqal)
    print('Vars:', d)
    
    if DATA != 'None':
        process_data(DATA, METRIC, VISUAL, FIG, PHE)
    elif SAMPLE == 'actin':
        example_actin(METRIC, VISUAL, FIG, PHE)
    #graph, id_to_name, name_to_id = load_hpo_tree(url = 'phenopy_mod/.phenopy/data/hp.obo')
    
    
    
if __name__ == '__main__':
    # CaseStudyExtractor.py -l www.google.com
    parser = argparse.ArgumentParser(description='Deep Phenotyping', formatter_class=RawTextHelpFormatter)
    
    ## Scraping variables
    parser.add_argument('-data', type=str, nargs='?', default='None',
                        help='path to file with phenotypic profiles')
    parser.add_argument('-metric',  type=str, nargs='?', default='hrss',
                        help='distance metric (jaccard, dice, hamming, hrss, occurrence, none)')
    parser.add_argument('-sample',  type=str, nargs='?', const=0, default='None',
                        help='Automatically perform deep phenotyping on available sample data (e.g. None or actin)')
    parser.add_argument('-visual',  type=str, nargs='?', const=0, default='TSNE',
                        help='Dimension reduction method to visualize the data (e.g: TSNE, PCA)')
    parser.add_argument('-phe',  type=int, nargs='?', default=0,
                        help='Calculate top 10 specific phenotypes per cluster (e.g: 0 (False) or 1 (True))')
    parser.add_argument('-fig',  type=int, nargs='?', default=1,
                        help='Whether or not to save the intermediary figures (e.g: 0 or 1 (default))')
    

    if len(sys.argv) > 2:
        import DeepPhenotyping_functions as func # initializing this script takes alot of time, hence we put it in the main function
    args = parser.parse_args()
    
    main(args)
    #print(args.accumulate(args.integers))
