import sys
sys.path.append('../PhenoTool/')

import ast 
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, Panel, Tabs
from bokeh.plotting import figure, show, output_notebook
from bokeh.transform import factor_cmap
from bokeh.models import  CategoricalColorMapper, LinearColorMapper
from bokeh.io import output_file, show
from bs4 import BeautifulSoup
from clinphen_src import get_phenotypes
from clinphen_src import src_dir
import collections
from io import BytesIO
#from KMedoids import KMedoids
import matplotlib.pyplot as plt
from math import log2
from nltk.tokenize import sent_tokenize
import numpy as np
from numpy.linalg import norm 
import networkx as nx
import os
import pandas as pd
from PIL import Image
import re
import requests
from selenium import webdriver
from sklearn.cluster import KMeans
from sklearn import metrics # 
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#import skfuzzy as fuzz ## Ultimately Fuzzy clustering is not used.
from txt2hpo.extract import Extractor
import time
import urllib.request
from urllib.parse import urlparse
from unidecode import unidecode
from webdriver_manager.chrome import ChromeDriverManager

global d_checks
global user
global pwd

import importlib as imp
import EntityLinking as enlink
#imp.reload(enlink)


# return Failed! or Succesful!
d_checks = {'Main Text Extraction' : '', 'Caption Extraction': '',  'Table Extraction': '', 'Table Legend Extraction': '', 'Supplementary Extraction': '', 'Figure Extraction' : '', 'Phenotyping Tables' : '', 'Extensive Annotation': '', 'Phenotypic Profiling' : ''}

def get_checks():
    global d_checks
    return d_checks

def set_login(USERNAME, PASSWORD):
    global user
    global pwd
    user = USERNAME
    pwd = PASSWORD
    return 

def get_login():
    global user
    global pwd
    return user, pwd
    
class AnnotateHPO(object):
    """
    Extract HPO terms from text.
    
    Generates a rule based method from an OBO model
    """
    def __init__(self, matcher):
        self.matcher = matcher
        self.matched_sents = [] 
        self.matched_hpo = []
        self.matched_def = []
        self.id_to_name = {}
        self.name_to_id = {}
    
    def setMatcher(self, matcher):
        self.matcher = matcher
        
    def getMatcher(self):
        return self.matcher
        
    def simpleCleaning(self, sent): 
        """
        Remove special characters that are not relevant to 
        the interpretation of the text

        Input:
            sent = free written text
        Output :
            processed sentence (lemmatized depending on preference)
        """
        sticky_chars = r'([!#,.:";@\-\+\\/&=$\]\[<>\'^\*`\(\)])'
        sent = re.sub(sticky_chars, r' ', sent)
        sent = sent.lower()
        return sent
    
    def collect_sents(self, matcher, doc, i, matches):
        """
        Register all information whenever a match is found!
        """
        match_id, start, end = matches[i]
        hpo_id = doc.vocab.strings[match_id]
        span = doc[start:end]
        sent = span.sent
        start = span.start_char - sent.start_char
        end = span.end_char - sent.start_char
        match_ents = [{
            "start": start,
            "end": end,
            "label": "HP",
        }]
        self.matched_sents.append({"text": sent.text, "ents": match_ents})
        self.matched_hpo.append(hpo_id)
        self.matched_def.append(sent.text[start:end])
        
    
    def addPatterns(self, graph):
        """
        Initialize Name to Id & Id to name dictionaries
        
        Add HPO definitions + synonyms to the provided matcher
        """
        count = 0
        
        self.id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
        self.name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
        
        for id_, data in graph.nodes(data=True):
            count += 1
            name = data.get('name').split(' ')
            pattern = [ast.literal_eval('{"ORTH": "' + str(i.lower()) +'"}') for i in name]
            self.matcher.add(self.name_to_id[data.get('name')], self.collect_sents, pattern)
            if 'synonym' in graph.nodes[id_]:
                for i in graph.nodes[id_]['synonym']: # append
                    pattern = [ast.literal_eval('{"ORTH": "' + str(i.lower()) +'"}') for i in i.split('"')[1].split(' ')]
                    self.matcher.add(self.name_to_id[data.get('name')], self.collect_sents, pattern)
                    
    def employ(self, doc):
        """
        Resets the previous matches and employs the model on fresh data
        
        doc = text subjected to nlp from spacy
        """
        self.matched_sents = [] 
        self.matched_hpo = []
        self.matched_def = []
        matches = self.matcher(doc)
        return matches
    
    def prune(self):
        """
        Prune overlapping entities, only keep the longest possible entity
        """
        remove_list = []
        for i in self.matched_sents: 
            i_start = i['ents'][0]['start']
            i_end = i['ents'][0]['end']
            i_length = i_end - i_start
            ix = 0
            for j in self.matched_sents:
                if i['text'] == j['text']:
                    j_start = j['ents'][0]['start']
                    j_end = j['ents'][0]['end']
                    j_length = j_end - j_start
                    if j not in remove_list and j_length < i_length: # don't do duplicates (not very clean)
                        if j_start <= i_start and j_end >= i_start:
                            remove_list.append(ix)
                        elif j_start <= i_end and j_end >= i_end:
                            remove_list.append(ix)
                        elif j_start >= i_start and j_end <= i_end:
                            remove_list.append(ix)
                ix += 1
        self.matched_sents = [i for j, i in enumerate(self.matched_sents) if j not in remove_list]
        self.matched_hpo = [i for j, i in enumerate(self.matched_hpo) if j not in remove_list]
        self.matched_def = [i for j, i in enumerate(self.matched_def) if j not in remove_list]
        return
    
    def show_marked_text(self):
        """
        Return a html that colors the found entities
        """
        matched_sents= self.matched_sents[::-1]
        doc = []
        current = matched_sents[0]['text']
        markup = current
        for i in range(len(matched_sents)):
            if matched_sents[i]['text'] != current:
                current = matched_sents[i]['text']
                doc.append(markup)
                markup = current
            else :
                end, start = matched_sents[i]['ents'][0]['end'], matched_sents[i]['ents'][0]['start']
                start_str = '<span style="color:red">'
                end_str = '</span>'
                markup = markup[:end] + end_str + markup[end:]
                markup = markup[:start] + start_str + markup[start:]
        return markup

def plot_graphs(data, k_medoids):
    """
    Render K-Medoids on a 2 dimensional plane
    
    Function from https://github.com/SachinKalsi/kmedoids/blob/master/demo.ipynb
    """
    colors = {0:'b*', 1:'g^',2:'ro',3:'c*', 4:'m^', 5:'yo', 6:'ko', 7:'w*'}
    index = 0
    for key in k_medoids.clusters.keys():
        temp_data = k_medoids.clusters[key]
        x = [data[i][0] for i in temp_data]
        y = [data[i][1] for i in temp_data]
        plt.plot(x, y, colors[index])
        index += 1
    plt.title('Cluster formations')
    plt.show()

    medoid_data_points = []
    for m in k_medoids.medoids:
        medoid_data_points.append(data[m])   
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x_ = [i[0] for i in medoid_data_points]
    y_ = [i[1] for i in medoid_data_points]
    plt.plot(x, y, 'yo')
    plt.plot(x_, y_, 'r*')
    plt.title('Mediods are highlighted in red')
    plt.show()
    
def getUMLS(data):
    umls = 'NaN'
    for item in data:
        if ':' in item:
            key, val = item.split(":", 1)
            if key == 'UMLS': 
                umls = val
                break
    return umls

def HPO_to_UMLS(graph):
    hpo_to_umls = {}
    for hpo in graph.nodes.keys():
        try : 
            data = graph.nodes[hpo]['xref']
            
        except :
            data = 'UMLS: NaN'
        hpo_to_umls[hpo] = getUMLS(data)
        umls_to_hpo = {v: k for k, v in hpo_to_umls.items()}
    return hpo_to_umls, umls_to_hpo

def makeTSNE_Cluster(values, l_id, l_lbl, l_order, title, clusters, pal, perp=30, seed=1234):
    
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    
    
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=perp, random_state=seed)
    tsne_2d = tsne_model.fit_transform(values)
    
    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    pred_y = kmeans.fit_predict(tsne_2d)
    print(kmeans)

    color_mapper = CategoricalColorMapper(factors=l_lbl, palette=pal)

    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    tsne_df['pt'] = l_id
    tsne_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    
    tsne_df2 = pd.DataFrame({'x' : kmeans.cluster_centers_[:, 0], 'y': kmeans.cluster_centers_[:, 1]})
    tsne_df2['pt'] = ["Cluster center %i" % (i) for i in range(len(kmeans.cluster_centers_))]
    #tsne_df['pt'] = l_id
    #tsne_df2['label'] = pred_y
                         

    #mapper = factor_cmap('cat', palette=Spectral6[:4], factors=tsne_df['cat'])
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=tsne_df, legend_field="label", size=10,  color={'field': 'label', 'transform': color_mapper}) # fill_color=mapper
    plot_tfidf.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"pt": "@pt"}
    
    bp.output_file('TSNE/Kmeans_phenoMap_tsne_%s.html' % (title), mode='inline')
    bp.save(plot_tfidf)
    print('\nTSNE figure saved under location: TSNE/Kmeans_phenoMap_tsne_%s.html' % (title))
    return 

def makeTSNE_Cluster2(values, l_id, l_clust, l_lbl, title, clusters, pal, perp=30, l_order=[], seed=1234, color_on_spot=False):
    """
    
    Perform a t-SNE dimension reduction and render an interactive bokeh plot
    
    Input:
        values = datapoints on which PCA should be performed
        l_id = list of (patient) identifiers (patient id)
        l_clust = list of cluster labels associated to datapoints
        l_lbl = list of labels associated to datapoints (often categories)
        l_order = list to indicate how labels should be sorted for coloring
                - if no list is provided than this is determined at random
        title = String containing title of plot
        clusters = 
        pal = pallete
        perp = perplexity for t-SNE
        seed = random seed
        color_on_spot = color after dimension reduction!
    """
    p1 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    
    
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=perp, random_state=seed)
    tsne_2d = tsne_model.fit_transform(values)
    
    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    pred_y = kmeans.fit_predict(tsne_2d)
    print(kmeans)

    color_mapper = CategoricalColorMapper(factors=list(set(l_clust)), palette=pal[0])
    
    if l_order != []:
        color_mapper2 = CategoricalColorMapper(factors=l_order, palette=pal[1])
    else :
        color_mapper2 = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal[1])
    
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    tsne_df['pt'] = l_id
    tsne_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    if color_on_spot:
        tsne_df['cluster'] = pred_y.astype(str)
    else :
        tsne_df['cluster'] = l_clust
    
    tsne_df2 = pd.DataFrame({'x' : kmeans.cluster_centers_[:, 0], 'y': kmeans.cluster_centers_[:, 1]})
    tsne_df2['pt'] = ["Cluster center %i" % (i) for i in range(len(kmeans.cluster_centers_))]
    #tsne_df['pt'] = l_id
    #tsne_df2['label'] = pred_y
                         

    #mapper = factor_cmap('cat', palette=Spectral6[:4], factors=tsne_df['cat'])
    # plotting. the corresponding word appears when you hover on the data point.
    p1.scatter(x='x', y='y', source=tsne_df, legend_field="cluster", size=10,  color={'field': 'cluster', 'transform': color_mapper}) # fill_color=mapper
    #p1.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
    
    hover = p1.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster"}
    #hover = p2.select(dict(type=HoverTool)) # or p1
    #hover.tooltips={"pt": "@pt", "lbl": "@lbl", "cluster" : "@cluster"}
    
    tab1 = Panel(child=p1, title="cluster")
    
    p2 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    p2.scatter(x='x', y='y', source=tsne_df, legend_field="label", size=10,  color={'field': 'label', 'transform': color_mapper2}) # fill_color=mapper
    #p2.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
    
    hover2 = p2.select(dict(type=HoverTool)) # or p1
    hover2.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster"}
    
    tab2 = Panel(child=p2, title="lbl")
    
    tabs = Tabs(tabs=[ tab1, tab2 ])
    
    
    
    
    
    bp.output_file('TSNE/Kmeans_phenoMap_tsne_%s.html' % (title), mode='inline')

    bp.save(tabs)
    print('\nTSNE figure saved under location: TSNE/Kmeans_phenoMap_tsne_%s.html' % (title))
    return 

def makeTSNE(values, l_id, l_lbl, title, pal, perp=30, seed=1234):
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=perp, random_state=seed)
    tsne_2d = tsne_model.fit_transform(values)
    
    color_mapper = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal)

    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    tsne_df['pt'] = l_id
    tsne_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    
    #mapper = factor_cmap('cat', palette=Spectral6[:4], factors=tsne_df['cat'])
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=tsne_df, legend_field="label", radius=0.5,  color={'field': 'label', 'transform': color_mapper}) # fill_color=mapper
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"pt": "@pt"}
    bp.output_file('TSNE/phenoMap_tsne_%s.html' % (title), mode='inline')
    bp.save(plot_tfidf)
    print('\nTSNE figure saved under location: TSNE/phenoMap_tsne_%s.html' % (title))
    return 

def makePCA(values, l_id, l_lbl, title, pal, radius=0.05, l_order=[], seed=1234):
    """
    Perform Principal Component Analysis for dimension reduction
    
    values = datapoints on which PCA should be performed
    l_id = list of (patient) identifiers associated to datapoints 
    l_lbl = list of labels associated to datapoints
    title = String containing title of plot
    pal = pallete
    radius = radius of the points to draw 
    l_order = list to indicate how labels should be sorted for coloring
    seed = random seed
    """
    # dimensionality reduction. converting the vectors to 2d vectors
    pca_model = PCA(n_components=2, random_state=seed) # , verbose=1, random_state=0
    pca_2d = pca_model.fit_transform(values)
    print('Explained PCA:\tPC1=', pca_model.explained_variance_ratio_[0], '\tPC2=',pca_model.explained_variance_ratio_[1])
    if l_order != []:
        color_mapper = CategoricalColorMapper(factors=l_order, palette=pal)
    else :
        color_mapper = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal)

    # putting everything in a dataframe
    pca_df = pd.DataFrame(pca_2d, columns=['x', 'y'])
    pca_df['pt'] = l_id
    pca_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients (Explained PCA:\tPC1=%.2f\tPC2=%.2f)" %(title, pca_model.explained_variance_ratio_[0], pca_model.explained_variance_ratio_[1]),        tools="pan,wheel_zoom,box_zoom,reset,hover,save", x_axis_type=None, y_axis_type=None, min_border=1)
    
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=pca_df, legend_field="label", radius=radius,  color={'field': 'label', 'transform': color_mapper}) # fill_color=mapper
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"pt": "@pt"}
    bp.output_file('PCA/phenoMap_pca_%s.html' % (title), mode='inline')
    bp.save(plot_tfidf)
    print('PCA figure saved under location: PCA/phenoMap_pca_%s.html' % (title))
    return

def k_distances2(x, k):
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    p=-2*x.dot(x.T)+np.sum(x**2, axis=1).T+ np.repeat(np.sum(x**2, axis=1),dim0,axis=0).reshape(dim0,dim0)
    p = [max(_,0) for _ in p.flatten()] # prevent negative values -> because of sqrt
    p = np.array(p).reshape(len(x.T), -1) 
    p = np.sqrt(p)
    p.sort(axis=1)
    p=p[:,:k]
    pm= p.flatten()
    pm= np.sort(pm)
    return p, pm

def lossVsClusters(X_trans, n=20):
    """
    Define optimal number of clusters with elbow method, by plotting
    loss vs no of clusters.
    
    Input:
        X_trans = Distance matrix based on HPO binary data (X)
        n = search space (number of clusters to consider)
    
    Return:
        k = Number of clusters that minimize loss
    """
    k_medoids = [KMedoids(n_cluster=i) for i in range(2,n)]
    k_medoids = [k_medoid.fit(X_trans) for k_medoid in k_medoids]
    loss = [k_medoid.calculate_distance_of_clusters() for k_medoid in k_medoids]

    # Plot elbow curve (to know best cluster count)
    plt.figure(figsize=(13,8))
    plt.plot(range(2,n),loss)
    plt.xticks(range(2,n))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Loss')
    plt.title('Loss Vs No. Of clusters')
    plt.show()

def defineOptimalNrOfPatients(df, n=0):
    """
    Define the minimal required nr of patients with the elbow method.
    Whereby the optimal nr of patients is assessed by drawing a line between
    the first and last observation and acquiring the greatest distance
    perpendicular to the draw line.
    
    Input:
        df = dataframe with phenotypic profiles for all patient
        n = search space (number of patients to consider)
    
    Return:
        k = Minimal number of patients a phenotype should be present in
    """
    if n == 0:
        n = len(df)
    distances = []
    l_cols = [col for col in df.columns if col != 'cluster']
    l_y = []
    for i in range(n):
        y = len(df[l_cols].replace(0, np.nan).dropna(axis=1, thresh=i).columns)
        if i != 1:
            l_y.append(y)

    fig1, ax1 = plt.subplots(1,2,figsize=(12,6))

    for i in range(1, len(l_y)+1):
        p1=np.array((1,l_y[0]))
        p2=np.array((n,l_y[len(l_y)-1]))
        p3=np.array((i+1, l_y[i-1]))
        distances.append(norm(np.cross(p2-p1, p1-p3))/norm(p2-p1))

    k = distances.index(max(distances))+1

    ax1[0].plot(range(1, len(l_y)+1), l_y)
    ax1[0].set_title('Presence of phenotypes in patient population')
    ax1[0].set_xlabel('Number of patients')
    ax1[0].set_ylabel('Phenotypes')

    ax1[1].plot(range(1, len(l_y)+1), distances, color='r')
    ax1[1].plot([k, k], [max(distances),0],
                 color='navy', linestyle='--')
    ax1[1].set_title('Perpendicular distance to line between first and last observation')
    ax1[1].set_xlabel('Number of clusters')
    ax1[1].set_ylabel('Distance')
    plt.show()
    return k
    
def clusterMembership(category_list, cluster_list, save_as=''):
    """ 
    Plot cluster membership for each subtype/ category
    
    Input:
        category_list = list of predefined categories/ subtypes
        cluster_list = list of clusters
        save_as = title used for saving the elbow plot figure
            (no title implies that the figure won't be saved)
    """ 
    d = {'cat': category_list, 'cluster': cluster_list}
    df_bar = pd.DataFrame(data=d)
    fig, ax = plt.subplots(figsize=(15,7))
    df_bar.groupby(['cluster', 'cat']).size().unstack().plot(ax=ax, kind = 'bar') 
    if save_as != '':
        fig = plt.gcf()
        fig.savefig('figures/cluster_memb_%s' % (save_as))
        plt.clf()
    else:
        plt.show()
    return
    
def elbowMethod(X_trans, method='kmeans', n=20, save_as=''):
    """
    Define optimal number of clusters with elbow method, optimized for 
    Within cluster sum of errors(wcss).
    
    Input:
        X_trans = Distance matrix based on HPO binary data (X)
        method = clustering method
        n = search space (number of clusters to consider)
        save_as = title used for saving the elbow plot figure
            (no title implies that the figure won't be saved)
    
    Return:
        k = Number of clusters that corresponds to optimized WCSS
    """
    methods = {'kmeans' : KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0), 
              'fuzz' : []}
    lbl = {'kmeans' : ['Number of clusters', 'WCCS'], 
          'fuzz' : ['Number of clusters', 'fuzzy partition coefficient']}
    
    wcss = []
    distances = []
    fig1, ax1 = plt.subplots(1,2,figsize=(12,6))
    
    kmeans = methods[method]
    
    for i in range(1, n+1):
        if method == 'kmeans':
            kmeans.n_clusters = i
            kmeans.fit(X_trans)
            wcss.append(kmeans.inertia_)
        elif method == 'fuzz':
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_trans, i, 2, error=0.005, maxiter=1000, init=123)
            wcss.append(fpc)

    for i in range(1, len(wcss)+1):
        p1=np.array((1,wcss[0]))
        p2=np.array((n,wcss[len(wcss)-1]))
        p3 =np.array((i+1, wcss[i-1]))
        distances.append(norm(np.cross(p2-p1, p1-p3))/norm(p2-p1))
        
    k = distances.index(max(distances))+1

    ax1[0].plot(range(1, len(wcss)+1), wcss)
    ax1[0].set_title('Elbow Method')
    ax1[0].set_xlabel(lbl[method][0])
    ax1[0].set_ylabel(lbl[method][1])
    
    ax1[1].plot(range(1, len(wcss)+1), distances, color='r')
    ax1[1].plot([k, k], [max(distances),0],
                 color='navy', linestyle='--')
    ax1[1].set_title('Perpendicular distance to line between first and last observation')
    ax1[1].set_xlabel('Number of clusters')
    ax1[1].set_ylabel('Distance')
    
    if save_as != '':
        plt.savefig('figures/elbow_plot_%s' % (save_as))
    else : 
        plt.show()
    return k

def makeDistanceHeatmap(df, col_id='Id', col_label='Category', dist='dice', title='Test'):
    """
    df = dataframe 
    col_id = column name where Patient Ids are stored
    col_label = column name where Labels are stored
    dist = distance metric
    title=  title of figure
    
    First: transforms the dataframe to a distance matrix
    Next: plot a heatmap
    """
    
    cols = list(df.loc[:, ~df.columns.isin([col_id, col_label])].columns)
    
    n = len(df)

    data = {
      'Id1':df[['Id']*n].values.flatten(),
      'Id2':  np.array([df['Id'] for i in range(n)]).flatten(),
      'Score':  squareform(pdist(df[cols], metric=dist)).flatten(),
      'Category1': df[['Category']*n].values.flatten(),
      'Category2': np.array([df['Category'] for i in range(n)]).flatten(),
    }
    
    mapper = LinearColorMapper(
        palette='Magma256',
        low=min(data['Score']),
        high=max(data['Score'])
    )

    factors =df['Id']
    #print(factors)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    hm = bp.figure(title="Categorical Heatmap", tools=TOOLS,toolbar_location='below',
                   tooltips=[('Patients', '@Id1 vs @Id2'), ('value', '@Score'), ('Categories:', '@Category1 vs @Category2')],
                x_range=factors, y_range=factors)

    hm.rect(x='Id1', y='Id2', 
           source=data, fill_color={'field': 'Score', 'transform': mapper}, width=1, height=1, line_color=None)

    bp.output_file('TSNE/Heatmap_%s_%s.html' % (title, str(dist)), mode='inline')
    bp.save(hm)
    return



def classificationReport(y_test, y_pred, threshold = 0.5):
    """
    Return an overview of the most important classification scoring
    metrics (with respect to chosen threshold). 

    This report consists of the following components:
        - Confusion matrix (heatmap)
        - PPV
        - NPV
        - Sensitivity
        - Specificity
        - Accuracy
        - F1-score

    Input:
        y_test = actual label
        y_pred = predicted label as a probability
        threshold = cut-off deciding the minimal confidence required to
            infer RA-status.
    """
    y_pred = [ 1 if i >= threshold else 0 for i in y_pred]  
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix

    plt = print_confusion_matrix(cnf_matrix, classes=['Non-RA', 'RA'],
                          title='Confusion matrix')
    #plt.figure()
    ax = plt.gca()
    ax.grid(False)
    #plt.savefig("figures/validation/confusion_matrix_SVM_"+ str(threshold) + ".png")

    print('\n|Overview of performance metrics|')
    print('Threshold:\t', round(threshold,2))
    print('F1:\t\t',round(metrics.f1_score(y_test, y_pred),2))
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    print('AUC-PR:\t\t',round(metrics.auc(recall, precision),2))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print('AUC-ROC:\t',round(metrics.auc(fpr, tpr),2))
    scores = scoresCM(cnf_matrix)
    print('Sensitivity:\t', round(scores[0],2))
    print('Specificity:\t', round(scores[1],2))
    print('PPV:\t\t', round(scores[2],2))
    print('NPV:\t\t', round(scores[3],2))
    print('Accuracy:\t', round(scores[7],2))

    print('\n|Confusion Matrix|')
    return

def scoresCM(CM):
    """
    Derive performance characteristics from the confusion matrix
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return [TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC]

def print_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    """
    This function only prints the confusion matrix.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def getav(value):
    return value[1][0]

def sortedPredictionList(b_pred, y_test):
    """
    This function sorts the list of true labels by the
    list of predictions. The sorted list of true labels
    can be used to create a ROC-curve for a non-probability
    classifier (a.k.a. a binary classifier like decision tree).
    Input:
        b_pred = list of hard-predictions (0 or 1) 
            or probabilities (0-1)
        y_test = list of actual labels, binarized to 
            1 or 0. 
    Example for generating 'l_sorted_true':
        Before sort:
           pred: 0 1 1 0 1 0 1 0 1 1 1 0 1 0
           true: 0 1 0 0 1 0 0 1 1 0 1 0 1 1
        After sort:
           pred: 1 1 1 1 1 1 1 1 0 0 0 0 0 0 
        -> true: 1 1 0 1 0 0 1 1 0 0 1 1 0 0
    Output:
        l_sorted_true = list of true label sorted on the 
            predictions label:
    """
    d_perf_dt = {}
    count = 0
    for i in range(0,len(y_test)):
        d_perf_dt[count] = [b_pred[count], y_test[count]]
        count += 1
    orderedDict = collections.OrderedDict(sorted(d_perf_dt.items(), key=lambda k: getav(k), reverse=True))
    l_sorted_pred= []
    l_sorted_true = []
    for x in orderedDict.items():
        l_sorted_pred.append(x[1][0])
        l_sorted_true.append(x[1][1])
    return l_sorted_true, l_sorted_pred

def score_binary(l_true, l_pred):
    """
    Calculates the dummy true en false positive rate for 
    a classifier that doesn't calculate probabilities 
    Input:
        l_true = list of true label sorted on the 
            predictions label.
            The function sortedPredictionList can
            be used to generate such a list!
    Output:
        TPR = list with true positive rates 
        FPR = list with false positive rates 
        PRC = list with precision (PPV)
    """
    dummi = l_true
    dummi = [2 if x==0 else x for x in dummi]
    dummi = [x -1 for x in dummi]
    l_pred.insert(0,0)
    l_true.insert(0,0)
    dummi.insert(0,0)
    # Compute basic statistics:
    TP = pd.Series(l_true).cumsum()
    FP = pd.Series(dummi).cumsum()
    P = sum(l_true)
    N = sum(dummi)
    TPR = TP.divide(P) # sensitivity / hit rate / recall
    FPR = FP.divide(N)  # fall-out
    PRC = TP.divide(TP + FP) # precision
    F1 = 2 * (PRC * TPR) / (PRC + TPR)
    d_conf = {'tpr': TPR, 'fpr': FPR, 'prc': PRC, 'threshold': l_pred, 'f1': F1}
    #d_conf = {'tpr': TPR, 'fpr': FPR, 'prc': PRC, 'threshold': l_pred}
    return d_conf 

def calculateAUC(x, y):
    """
    Calculate AUC by parts by calculating the surface area of a 
    trapzoid.
    x = x-axes 
    y = y-axes (interpolated with x)
    """
    auc = 0
    for i in range(1,len(y)):
        last_x = x[i-1]
        last_y = y[i-1]
        cur_x = x[i]
        cur_y = y[i]
        auc += np.trapz([last_y, cur_y], [last_x, cur_x])
    return auc

def plot_performance(y_test, preds, clf_name):
    """
    Plot Receiver Operator Characteristic curve and
        Precision recall curve
        
    Input :
        y_test = list with actual labels
        preds = list with predicted labels
        clf_name = name of classifiers
    """
    fpr, tpr, _ = roc_curve(y_test, preds)
    prec, recall, _ = precision_recall_curve(y_test, preds)
    
    ## calculate precision recall curve
    recall_scale = np.linspace(0, 1, 100)
    y_true1, y_pred1 = sortedPredictionList(preds, y_test)
    d_conf = score_binary(y_true1, y_pred1)
    PRC, TPR, FPR = d_conf['prc'], d_conf['tpr'], d_conf['fpr'] 

   
    PRC[0] = 0.0
    inter_prec = np.interp(recall_scale, TPR, PRC)
    inter_prec[0] = 1.0 
    pr_auc = calculateAUC(recall_scale, inter_prec) #  -1 * 
    
    ## calculate ROC curve
    roc_auc = calculateAUC(TPR, FPR)

    fig1, ax1 = plt.subplots(1,2,figsize=(12,6))

    lw = 2
    ## ROC
    ax1[0].plot(TPR, FPR, color='darkorange',
             lw=lw, label='%s ROC curve (AUC = %0.2f)' % (clf_name, roc_auc))
    ax1[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax1[0].set_xlim([0.0, 1.0])
    ax1[0].set_ylim([0.0, 1.05])
    ax1[0].set_xlabel('False Positive Rate', fontsize=18)
    ax1[0].set_ylabel('True Positive Rate', fontsize=18)
    ax1[0].legend(loc="lower right", fontsize=11)
    ax1[0].tick_params(axis='both', which='major', labelsize=10)
    ax1[0].tick_params(axis='both', which='minor', labelsize=8)

    ## PR
    ax1[1].plot(recall_scale, inter_prec, color='darkorange',
             lw=lw, label='%s PR curve (AUC = %0.2f)' % (clf_name, pr_auc))
    ax1[1].set_xlim([0.0, 1.0])
    ax1[1].set_ylim([0.0, 1.05])
    ax1[1].set_xlabel('Recall', fontsize=18)
    ax1[1].set_ylabel('Precision', fontsize=18)
    ax1[1].legend(loc="lower right", fontsize=11)
    ax1[1].tick_params(axis='both', which='major', labelsize=10)
    ax1[1].tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig('figures/Performance_%s.png' % (clf_name))
    plt.show()
    return

def print_perf(TP, FP, TN, FN):
    """
    Derive performance metrics from provided confusion matrix values 
    """
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # F1-score (harmonic mean between Precision and recall)
    F1 = 2 * ((PPV * TPR) / (PPV + TPR))

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    print('Sensitivity:\t', round(TPR,2))
    print('Specificity:\t', round(TNR,2))
    print('PPV:\t\t', round(PPV,2))
    print('NPV:\t\t', round(NPV,2))
    print('F1:\t\t', round(F1,2))
    print('Accuracy:\t', round(ACC,2))
    return

def generate_hpo_updgrade_list(graph):
    """
    Generate a dictionary that links deprecated hpo to the current hpo-definition
    
    Input:
        graph = OBO graph
    
    Output:
        d_trans = dictionary with deprecated HPO codes as keys and new hpo as value
    """
    d_trans = {}
    for id_, data in graph.nodes(data=True):
        if 'alt_id' in graph.nodes[id_]:
            for alt_id in graph.nodes[id_]['alt_id']:
                if alt_id not in d_trans:
                    d_trans[alt_id] = id_
    return d_trans

def update_deprecated_OBO(l_hpo, d_trans):
    """
    l_hpo = list with hpo values
    
    Description:
    - use dict to translate deprecated hpo to current version hpo definitions 
    
    """
    for idx, hpo in enumerate(l_hpo):
        if hpo in d_trans:
            l_hpo[idx] = d_trans[hpo]
    return l_hpo

def is_phenotypic_abnormality(graph, l_hpo):
    """
    Only keep HPO annotations that refer to phenotypic abnormality
    
    Input:
        graph = obo graph (HPO DAG)
        l_hpo = list of human phenotype ontology codes
    """ 
    new_hpo = []
    PHENOTYPIC_ABNORMALITY_ID = "HP:0000118"
    for idx, hpo in enumerate(l_hpo):
        if [PHENOTYPIC_ABNORMALITY_ID] in list(nx.dfs_successors(graph,hpo).values()):
            new_hpo.append(hpo)
    return new_hpo

def calculateIC(hpo_network, n_anno): # CURRENTLY NOT USED !
    """ 
    Calculate infromation content & depth per node!
    """ 
    (disease_records,phenotype_to_diseases,) = load_d2p(disease_to_phenotype_file, hpo_network, alt2prim, )

    custom_annos = []
    for node_id, data in hpo_network.nodes(data=True):
            # annotate with information content value
            hpo_network.nodes[node_id]['ic'] = calculate_information_content(
                node_id,
                hpo_network,
                phenotype_to_diseases,
                n_anno,
                custom_annos,
            )
    return hpo_network

def visualizeIC_Features(df, ic_weights, depth_weights, cluster, cutoff=3): # CURRENTLY NOT USED !
    """
    Input:
    cutoff = specify the required prevalency of the feature 
    
        Default cut-off = 0.2
            feature should at least be prevalent in 20% of the patients of this cluster
            
    Description:
        Visualize the top-%  features with highest Information Content

    Output:
        barplot highlighting prevalence of each feature
    """
    #plt.figure
    N = len(df[df['cluster']==cluster])
    series = df[df['cluster']==cluster].loc[:, ~df.columns.isin(['cluster'])].sum() #[:10]
    #print([weights[col.lower().strip()] for col in series.keys()])
    prob = series + ([ic_weights[col.lower().strip()] for col in series.keys()]) # series + 
    prob = prob * ([depth_weights[col.lower().strip()] for col in series.keys()])
    #prob = np.log2(prob)
    #print(weights[max(weights)])
    prob = prob / max(prob)
    #print(prob)
    prob = prob.rename(lambda x: x[:25] + '...' if len(x)> 25 else x) # long labels are automatically shortened
    prob = prob.sort_values(ascending=True)
    # / N
    mask = prob > cutoff
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob.plot(kind='barh',figsize=(9,9), title='Most occuring phenotypic features in Cluster %s (n=%s)' % (cluster, str(N)))
    plt.show()
    return


def visualizeMostOccuringFeatures(df, cluster, labels={}, cutoff=0.2):
    """
    Input:
        df = dataframe with phenotypic profile per patient & assigned clusters
        cluster = cluster of interest
        cutoff = specify the required prevalency of the feature 
    
            Default cut-off = 0.2
                feature should at least be prevalent in 20% of the patients of this cluster
            
    Description:
        Visualize the top-% most occurring features of a cluster/group
            
    Beware: making the cut-off too large will not show pathognomic features but
        making the cut-off too small and this plot will show individual characteristics
        rather than a phenotypic profile for the disease. Also, don't forget
        to consider the impact of the group-size when defining a cut-off.
    
    Output:
        barplot highlighting prevalence of each feature
    """
    
    #plt.figure
    N = len(df[df['cluster']==cluster])
    if labels != {}:
        df = df.rename(columns=labels)
    series = df[df['cluster']==cluster].loc[:, ~df.columns.isin(['cluster'])].sum().sort_values(ascending=True) #[:10]
    series = series.rename(lambda x: x[:25] + '...' if len(x)> 25 else x) # long labels are automatically shortened
    prob = series / N
    mask = prob > cutoff
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    ax = prob.plot(kind='barh',figsize=(9,9), title='Most occuring phenotypic features in Cluster %s (n=%s)' % (cluster, str(N)))

    
    plt.show(ax)
    return

def plotPhenotypesPatients(df, name=''):
    """
    Input:
        df = dataframe with phenotypic profile per patient & assigned clusters
    
    Description:
        Plot cumulative difference between nr of patients and nr of features
    
    Output:
        barplot highlighting phenotypic specificity of each feature
    """
    l_cols = [col for col in df.columns if col != 'cluster']
    l_y = []
    l_x = []
    for i in range(1, len(df)-1):
        
        y = len(df[l_cols].replace(0, np.nan).dropna(axis=1, thresh=i).columns)
        #print(str(i-1), '->', str(i),y)
        if i != 1:
            l_y.append(past_y - y)
            l_x.append(i)
        past_y = y
    #print(l_y)
    plt.plot(l_x, l_y)
    plt.ylabel('Difference in number of phenotypic features')
    plt.xlabel('Number of%spatients' % (' ' + name + ' '))
    plt.xlim((0,40))
    return l_y

def PhenotypicSpecifictyOccurrence(df, weights, cluster=None, labels={}, pat_frac = 0.2, rename=True, topN=None, cutoff=None, divide_occurrence=True):
    """
    Input:
        df = dataframe with phenotypic profile per patient & assigned clusters
        cluster = cluster of interest
        weights = weights for every phenotype (list)
        
        labels = labels to 
        pat_frac = phenotype should be present in at least % of patients
            - feature should at least be prevalent in 20% of the patients of this cluster
        rename = boolean indicating whether or not to limit character length of y-axis labels
        topN = the top 
        cutoff = specify the required specificity
        
            Default cut-off = 0.2
        divide_occurrence = boolean indicating whether to divide by total patients or occurrence
    
    Description:
        Visualize the top-% or top-N most specific features of a cluster/group. The specificity is 
        calculated by dividing the occurence of patients with said phenotypes by the nr of assoc genes. 
    
    Output:
        barplot highlighting phenotypic specificity of each feature
    """
    
    
    N = len(df[df['cluster']==cluster])
    l_cols = [col for col in df.columns if col != 'cluster']
    print(N)
    
    if cluster == None:
        cluster = list(df['cluster'].unique())[0]
    
    df = df[df['cluster']==cluster]
    df = pd.concat([df[l_cols].replace(0, np.nan).dropna(axis=1, thresh=round(pat_frac*N)), df['cluster']], axis=1)
    weights = [weights[l_cols.index(i)] for ix, i in enumerate(list([col for col in df.columns if col != 'cluster']))]
    
    
    if labels != {}:
        df = df.rename(columns=labels)
    
    series = df.loc[:, ~df.columns.isin(['cluster'])].sum() 
    
    if rename == True:
        series = series.rename(lambda x: x[:25] + '...' if len(x)> 25 else x) # long labels are automatically shortened
    if divide_occurrence:
        occ = np.divide(series, N)
        prob = np.divide(occ, weights)
    else :
        prob = np.divide(series, weights)
    prob = prob.sort_values(ascending=True)
    
    if topN != None:  # return top N phenotypic features
        prob = prob.nlargest(topN).sort_values(ascending=True)
    elif cutoff != None: # return top phenotypic features above phenotypic specificity cut-off
        mask = prob > cutoff
        tail_prob = prob.loc[~mask].sum()
        prob = prob.loc[mask]
    
    return prob

def getNumberOfGenes(l_cols,  df_hpo):
    """
    
    Assess number of associated genes for each patient
    
    Input: 
    - l_cols = list with columns
    - df_hpo = dataframe with hpo linked to associated genes
    
    
    Formula is as follows: 
    
    On Group level:
        nr of patients / nr of associated genes
    
    On individual level:
        1/ nr of associated genes
    
    """
    weight_list = []
    
    for val in l_cols:
        col = val.strip()
        weight = df_hpo[df_hpo['#Format: HPO-id']==col]['entrez-gene-id'].nunique()
        if weight != 0:
            weight_list.append(weight)
        else :
            print('No assoc genes found for %s ' % (col))
            weight_list.append(1)
    return weight_list

def getNumberOfGenesTranslation(l_cols, col2hpo, df_hpo, penalty=100):
    """
    Assess number of associated genes for each patient. 
     
    In contrast to getNumberOfGenes(), this function applies a dictionary translation
    whereby phenotypic features are converted to HPO-codes
    
    Input: 
    - l_cols = list with columns
    - col2hpo = translates columns to hpo
    - df_hpo = dataframe with hpo linked to associated genes
    - penalty = score to assign whenever there are no phenotypes / genes found
    
    
    Formula is as follows: 
    
    On Group level:
        nr of patients / nr of associated genes
    
    On individual level:
        1/ nr of associated genes
    
    """
    weight_list = []
    
    for val in l_cols:
        col = val.lower().strip()
        #assoc_genes = df_hpo[df_hpo['#Format: HPO-id']==hpo]['entrez-gene-id'].nunique()
        #coef_list.append(1/assoc_genes) # +0.00000001
        if col in col2hpo.keys():
            #print(col2hpo[col])
            
            if type(col2hpo[col]) == list:
                weight_list.append(sum([df_hpo[df_hpo['#Format: HPO-id']==hpo]['entrez-gene-id'].nunique() for hpo in col2hpo[col]]))
            else :
                weight_list.append(df_hpo[df_hpo['#Format: HPO-id']==col2hpo[col]]['entrez-gene-id'].nunique())
            #break
        else :
            print(col, 'not found')
            weight_list.append(penalty)
    return weight_list


## Functions preprocessing

def remove_html_tags(soup):
    """  
    Remove HTML tags and hyperlinks from text file
    """
        
    special_classes = ['accordion-tabbed__tab-mobile', 'dropBlock__body', 'inline-table', 'article-table-content']
    for cl in special_classes:
        for div in soup.find_all('div', {'class': cl}):  #
            div.replaceWith('')
            
    special_classes = ['dpauthors', 'dporcid', 'dptop', 'dptitle', 'dpfn', 'MathJax_Message']        
    for cl in special_classes:
        for div in soup.find_all('div', {'class': cl}):  #
            div.decompose()

    for element in ['ul', 'i', 'li',  "script", "style", "meta", "link", "sup", "select", "option", "figcaption", 'dt', 'dl', 'dd', 'table', 'thead', 'noscript' ]: # 'figure', 'table', 'a', 
        for div in soup.find_all(element):  # , {'class':'Google-Scholar'}
            #print(div)
            div.decompose()
    #print('1.1:', len(soup.text))
    for s in soup.select('div'):
        s.get_attribute_list = ''
    #print('1.2:', len(soup.text))
    return soup

## Extract Supplement

def predict_article_figure_img(match):
    """
    Input:
        match = link found in article
        
    
    Predict if link references image from article
    """
    article = False 
    link = match.get('src')
    if link == None:
        link = match.get('alt')
    if link != None: 
        link = link.lower()
        score  = 1*('article' in link) + 1 * ('image' in link) + 1 * ('figure' in link) + 1 * ('img' in link) + 1 * ('fig' in link) + 1 * ('pic' in link) - 1 * ('produkte' in link) - 1 * ('product' in link) - 1 * ('icon' in link) - 1 * ('logo' in link) - 1 * ('.gif' in link) - 1 * ('thumbnail' in link) - 1 * ('powerpoint' in link) - 1 * ('docx' in link)
        if score > 0 :
            article = True
    #print(article, link)
    return article

def predict_supplement(classes, texts):
    """
    Input:
        txt = text for supplement
        
    Predict which class refers to the supplement location.
    It checks if all expected words are found in class name.
    
    If no supplement is found than Suppl remains 0
    
    In the future you might want to check if there is 
    a link in the txt
    """
    max_score = 0
    max_ix = 0
    ix = 0
    suppl = False 
    candidates = []
    for txt in texts:
        score  = 1*('supplemental' in txt) + 1*('supplementary' in txt) + 1*('appendix' in txt) + 1*('download' in txt)  + 1*('link' in txt) + 1*('http:' in txt) + 1*('href' in txt)
        if score > 0 :
            max_ix = ix
            max_score= score
            suppl = True
            candidates.append(classes[ix])
        ix += 1
    return candidates, suppl

def import_file(file_link, title, file_title):
    """
    Import Supplementary & save in specified location
    
    Input:
        file_link = link to supplementary file
        title = title article
        file_title = title of supplementary file
    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent}  # , 'Host': 'example.org'
    cookieProcessor = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookieProcessor)
    request=urllib.request.Request(file_link,None, headers) #The assembled request
    try :
        response = opener.open(request,timeout=100)
        data = response.read()
    except :
        return
    
    try : 
        file = open("results/%s/0_raw/supplement/%s" % (title, file_title), "wb")
        file.write(data)
        file.close()
    except: 
        print('The file %s is not in a regular table format' % (file_title))
    return

def extract_supplement(soup, title, domain, save_supplement=False): 
    """
    Extract supplementary files from article by checking all links
    
    Input: 
        soup = extraction from case study
        save_supplement = save the actual supplementary files
    """
    accepted_formats = ['pdf', 'docx', 'doc', 'jpg', 'jpeg', 'odt', 
                       'pptx', 'png', 'csv', 'xlsx', 'xls', 'tsv', 'ppt', 'txt']
    
    dir_figs = 'results/%s/0_raw/supplement' % (title)
    txt = ''
    classes = []
    text = []
    for ix, element in enumerate(soup.findAll('a')):
        classes.append(element.get('href'))
        classes.append(element.get('srcset'))
        text.append(str(element))
    #cl, suppl = predict_supplement(classes, text)
    cl = classes
    l_supplement = []
    suppl = True
    if suppl: 
        for cand in cl:
            l_supplement.append(cand)
    if l_supplement == []:
        return
    links = l_supplement
    
    for link in links:
        if link is not None:
            if len(link.rsplit('.', 1)) > 1: 
                if link.rsplit('.', 1)[1] not in accepted_formats:
                    continue
            else :
                continue
        else :
            continue
        file_link = link
        try:
            if file_link[:2] == '//': # 
                file_link = 'https:' + figure_link
            elif 'http' not in file_link:
                continue
        except: 
            if file_link != '':
                continue
        file_title = file_link.rpartition('/')[2]
        file_title = re.sub(r"[^\.0-9a-zA-Z]+", "", file_title)
        if save_supplement and file_link !=  '':
            import_file(file_link, title, file_title)
    
    return 

## Webscraping

def scrapingCaseStudyHTML(path_to_html):
    """
    Scraping case study from downloaded HTML
    
    Input: 
        link = link to case study
        path_to_driver = path to google chrome webdriver
    
    Output:
        soup = scraped html file consisting of content from case study
    """
    with open(path_to_html, "r", encoding="utf-8") as file:
        html = file.read()
    soup = BeautifulSoup(html, "lxml")
    return soup


def getLoginData(login_file): 
    """
    Acquire the login information from the login_details.txt.
    This file contains a dictionary
    
    Input:
        login_file = path to file with login details
    """
    file = open(login_file, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return dictionary

def scrapingCaseStudyLOGIN(URL, user='', pwd='', LIBRARY='https://login.proxy.library.uu.nl/login?auth=uushibboleth&url=', TIME=10):
    """
    This function automatically gains access to the case studies behind paywalls. 
    However it does require an account to an academic library, thus you have 
    to provide a username and password. Default: University Utrecht library.
    
    Input: 
        URL = link to case study
        user = username 
        pwd = password
        LIBRARY = link to library to get access to paper (default: University Utrecht)
        TIME = appoint time to ensure a successfull connection in case you have a weak wifi (default:10)
    
    Output:
        soup = scraped html file consisting of content from case study
    """
    if user == '':
        user, pwd = get_login()
    link = LIBRARY + URL 

    driver = webdriver.Chrome(ChromeDriverManager().install())

    driver.get(link)
    time.sleep(round(TIME/2)) # make sure page has completed loading
    driver.find_element_by_name('Ecom_User_ID').send_keys(user)
    driver.find_element_by_name('Ecom_Password').send_keys(pwd)
    driver.find_element_by_id('loginButton2').click()
    time.sleep(TIME) # make sure page has completed loading
    
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml") # , "lxml"

    driver.quit()
    return soup

def scrapingCaseStudy(link, TIME):
    """
    Scraping case study with a google chrome webdriver, the content
    is then formatted by BeautifulSoup and the returned by the function 
    
    The function sleeps for 10 seconds, to ensure that the page is fully loaded,
    you might want to decrease the timer in the future?
    
    Input: 
        link = link to case study
        TIME = appoint time to ensure a successfull connection in case you have a weak wifi (default:10)
    
    Output:
        soup = scraped html file consisting of content from case study
    """
    #driver = webdriver.Chrome(path_to_driver) 
    driver = webdriver.Chrome(ChromeDriverManager().install())

    driver.get(link)
    time.sleep(TIME)
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml") # , "lxml"

    driver.quit()
    return soup

## Setting up environment
def createFolderStructure(title):
    """
    Create environment for case study pipeline, to 
    ensure that the tool saves the results in a convenient manner.
    
    Input: 
        link = link to case study
    """
    if not os.path.exists('results/%s' % (title)) : # make new folder
        os.makedirs('results/%s' % (title)) 
        os.makedirs('results/%s/0_raw' % (title)) 
        os.makedirs('results/%s/0_raw/figures' % (title)) 
        os.makedirs('results/%s/0_raw/tables' % (title)) 
        os.makedirs('results/%s/0_raw/supplement' % (title)) 
        os.makedirs('results/%s/1_extractions' % (title)) 
        os.makedirs('results/%s/2_phenotypes' % (title)) 
        os.makedirs('results/%s/3_annotations' % (title))
    return 

## Parse Case study

def parseCaseStudy(soup, title, link, save_supplement=False, screenshots=False, remove_accent=False):
    """
    Parse the HTML content from a case study, all raw content 
    is saved locally!
    
    ToDo:
        - add acronym support?
        - improve robustness by adding alternative table interpreters
    
    Input: 
        soup = content of case study as interpreted by BeautifulSoup
        title = title of case study paper
        link = URL link to case study
        save_supplement = boolean indicating whether or not to 
            save the supplementary files
        screenshots = boolean indicating whether or not to make 
            screenshots of the supplemented files.
                - Useful if you aren't able to extract figures
        
    Output:
        new_soup = processed main text from case study
    """
    title_classes = extract_custom_headers(soup)
    print('Perserve following custom titles & headers:')
    #print(str(title_classes) + '\n')
    
    # asses Domain
    parsed_uri = urlparse(link)
    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
    print('Domain:', domain)
    
    # Extract Figures / Tables
    tables = soup.findAll("table")
    
    extract_undefined_captions(soup, title)
    if save_supplement == True:
        extract_supplement(soup, title, domain, save_supplement) 
    save_tables(title, tables, remove_inc=True)
    extract_figures(soup, title, domain, save_figures=True, screenshots=screenshots)
    
    # Get links within table
    for index, table in enumerate(tables):
        check_for_link(table, domain, index, title) # add this functionality to save tables?
    print('1:', len(soup.text))
    soup = remove_html_tags(soup)
    print('2:', len(soup.text))
    
    print('3:', len(soup.text))
    
    #print(soup)
    new_soup = regex_cleaning(str(soup), title_classes, remove_accent)
    #print(new_soup)
    new_soup = 'DOI: ' + link + '<br><br>' + new_soup
    print('Saving extracted files in following location:\nDeepPhenotypingHPO/PhenoTool/results/%s/0_raw/' % (title))
    with open("results/%s/0_raw/Main_text_%s.html" % (title, title), "w", encoding="utf-8") as file:
        file.write(new_soup)
    return new_soup

## Extract figures

def remove_tag_caption(soup):
    """  
    Remove HTML tags and hyperlinks from figure caption
    """
    for element in ['ul', 'i', 'li', 'a', "script", "style", "meta", "link", "sup", "select", "option", "em"]: # a  # "script", "style", "meta", "link", "sup", "select", "option", "figcaption" 'span', 
        for div in soup.find_all(element):  # , {'class':'Google-Scholar'}
            #print(div)
            div.decompose()
    #print(soup)
    return soup


def screen_capture(URL, title, fig_title, user='', pwd='', LIBRARY='https://login.proxy.library.uu.nl/login?auth=uushibboleth&url='):
    """
    This function automatically gains access to the case studies behind paywalls. 
    However it does require an account to an academic library, thus you have 
    to provide a username and password. Default: University Utrecht library.
    
    Input: 
        URL = link to figure
        title = title article
        fig_title = title of figure
        user = username 
        pwd = password
        LIBRARY = link to library to get access to paper (default: University Utrecht)
    
    Output:
        soup = scraped html file consisting of content from case study
    """
    if user == '':
        user, pwd = get_login()
    
    link = LIBRARY + URL 

    driver = webdriver.Chrome(ChromeDriverManager().install())

    driver.get(link)
    time.sleep(5) # make sure page has completed loading
    driver.find_element_by_name('Ecom_User_ID').send_keys(user)
    driver.find_element_by_name('Ecom_Password').send_keys(pwd)
    driver.find_element_by_id('loginButton2').click()
    time.sleep(5) # make sure page has completed loading
    
    data = driver.page_source
    
    driver.save_screenshot("results/%s/0_raw/figures/%s.png" % (title, fig_title))
    driver.quit()
    #try : 
    #    #image = Image.open(BytesIO(data))
    #    #image.save("results/%s/0_raw/figures/%s" % (title, fig_title), 'png')
    #except: 
    #    return
    
    return 

def import_figure(figure_link, title, fig_title):
    """
    Import Figure & save in specified location
    
    Input:
        figure_link = link to figure
        title = title article
        fig_title = title of figure
    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent}  # , 'Host': 'example.org'
    cookieProcessor = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookieProcessor)
    try: 
        request=urllib.request.Request(figure_link,None, headers) #The assembled request
    except:
        return
    try :
        response = opener.open(request,timeout=100)
        data = response.read()
    except:
        return
    
    try : 
        image = Image.open(BytesIO(data))
        image.save("results/%s/0_raw/figures/%s" % (title, fig_title), 'png')
    except: 
        return
        #print('The file %s is not in a regular image format' % (fig_title))
    return

def import_table(table_link, title, table_title):
    """
    Import Table & save in specified location
    
    Input:
        table_link = link to figure
        title = title article
        table_title = title of figure
    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent}  # , 'Host': 'example.org'
    cookieProcessor = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookieProcessor)
    try: 
        request=urllib.request.Request(table_link,None, headers) #The assembled request
    except:
        return
    try :
        response = opener.open(request,timeout=100)
        data = response.read()
    except:
        return
    
    try : 
        file = open("results/%s/0_raw/tables/%s" % (title, table_title), "wb")
        file.write(data)
        file.close()
    except: 
        print('The file %s is not in a regular table format' % (table_title))
    return
    
def save_captions(txt, title, doc='captions_figures.txt'):
    """
    Save caption data
    
    Input:
        txt = text from caption
        title = title of article
        doc = user-defined title of the captions file
    """
    txt = removeAccent(txt)
    file = open("results/%s/0_raw/%s" % (title, doc), "w")
    file.write(txt)
    file.close()
    return

def predict_caption(classes, txt):
    """
    Input:
        classes = list with class names
        txt = text for caption
        
    Predict which class refers to the caption of a table / figure
    It checks if all expected words are found in class name
    
    The only check we do now with regard to the txt of caption, is
    too see which one has the longest caption. This is to
    prevent mistaking the title of the caption for the description. 
    """
    max_score = 0
    max_ix = 0
    max_len = 0
    ix = 0
    caption_found = False
    
    
    
    for cl in classes:
        #print(cl)
        if cl in ['caption', 'figcaption']: # early call
            return cl, True
        
        score  = 1*(cl.count('caption')) + 1*(cl.count('desc'))  + 1*('image' in cl) + 1*('figure' in cl) + 1*('text' in cl)# 1*('tbl' in cl) + 1*('table' in cl) + 1*('wrap' in cl) + 
        #scores.append(score)
        if score > max_score or (score == max_score and len(txt[ix]) > max_len) and score != 0:
            max_ix = ix
            max_len = len(txt[ix])
            max_score= score
            caption_found = True
        ix += 1
    #print(classes[max_ix])
    return classes[max_ix], caption_found

def extract_legend(soup, title): 
    """
    Extract legends from article
    
    Input: 
        soup = extraction from case study
        title = title of the article
    """
    dir_figs = 'results/%s/0_raw' % (title)
    
    interpret_legend(soup, title)
    return 

def interpret_legend(soup, title):
    """
    This function recognizes legends in the case study.
    
    In addition, it attempts to capture the corresponding caption.
    
    ToDo: look at extract_undefined_captions() function 
        - how to process bulk of data effectively
    """
    txt = ''
    #print(soup.find_all('p'))
    for ix, div in enumerate(soup.find_all('body')):
        #print(ix, '\t', div)
        #print(div)
        classes = []
        text = []
        for elem in div.findAll(class_=True):
            classes.extend(elem["class"])
            text.extend([elem.text for i in range(len(elem["class"]))])
        if classes == []:
            continue
        else :
            txt += '\nITEM %s:\n' % (ix)
            
        if soup.find_all('legend') != []: # maybe not best fix - puts everything on a single line?
            captions = div.findAll(i)
            
        else :
            cl, leg = predict_legend_table(classes, text)
            #print(ix, cl, leg)
            captions = []
        
        if leg: 
            for i in ['div', 'span', 'p', 'legend']:
                if captions == []:
                    captions = div.findAll(i, attrs={'class': cl}) # section-paragraph
                    #print(captions)
                else :
                    break
        if captions == []:
            continue

        captions = html_to_text(str(captions[0])) # format to text
        clean = re.compile('<.*?>')
        cleansed = re.sub(clean, '', captions)
        txt += cleansed
        
        
    if txt != '':
        save_captions(txt, title, 'legends_table.txt')
    return 

def extract_figures(soup, title, domain, save_figures=False, screenshots=False): 
    """
    Extract figures from article
    
    Input: 
        soup = extraction from case study
        title = title of the article
        domain = domain/ root of the website
        save_figures = only the captions are extracted unless save_figures=True
        screenshots = if you have difficulty extracting the figures, you could 
            make screenshots to ensure that the figures are captured
    """
    dir_figs = 'results/%s/0_raw/figures' % (title)
    
    interpret_figure(soup, title, domain, save_figures, screenshots)
    interpret_img(soup, title, domain, save_figures, screenshots) 
    return 

def interpret_img(soup, title, domain, save_figures=False, screenshots=False):
    """
    Captions can often not be retrieved from IMG directly
    
    # does image sometimes have a caption though??
    yes -> alt
    ToDo: check if it can also find images when they have multiple attributes 
    """
    links = []
    for link in soup.findAll('img'): # , attrs={'href': re.compile("^(?://|http\://|/)")}
        
        #print(link)
        if predict_article_figure_img(link):
            links.append(link)
    extract_images_with_links(links, title, domain, save_figures, screenshots)
    return


def interpret_figure(soup, title, domain, save_figures=False, screenshots=False):
    """
    This function recognizes figures in the case study.
    
    In addition, it attempts to capture the corresponding caption.
    """
    txt = ''
    #print(soup.find_all('figure'))
    for ix, div in enumerate(soup.find_all('figure')):
        #print(div)
        classes = []
        text = []
        for elem in div.findAll(class_=True):
            classes.extend(elem["class"])
            text.extend([elem.text for i in range(len(elem["class"]))])
        if classes == []:
            continue
        else :
            txt += '\nFIGURE %s:\n' % (ix)

        cl, fig = predict_caption_figure(classes, text)
        captions = []
        if fig: 
            for i in ['div', 'span', 'p', 'caption']:
                if captions == []:
                    captions = div.findAll(i, attrs={'class': cl}) # section-paragraph
                else :
                    break
        if captions == []:
            continue
        
        links = div.findAll('a', attrs={'href': re.compile("^(?://|http\://|/)")}) # maybe http as well
        links.extend(div.findAll('source', attrs={'srcset': re.compile("^(?://|http\://|/)")}))
        #print(links)
        extract_images_with_links(links, title, domain, save_figures, screenshots)
        #txt = regex_cleaning(str(captions[0]))
        #captions = str(remove_tag_caption(captions[0])) # hopefully this works everytime
        captions = html_to_text(str(captions[0])) # format to text
        clean = re.compile('<.*?>')
        cleansed = re.sub(clean, '', captions)
        txt += cleansed
    if txt != '':
        save_captions(txt, title, 'captions_figures.txt')
    return 

def extract_images_with_links(links, title, domain, save_figures=False, screenshots=False):
    """
    Format the paths to images to actual working links!
    
    Input:
        links = candidate links for figures
        title = title of article
        domain = 
        save_figures = save all figures you can find
        screenshots = make screenshots of the figures
    
    """ 
    for link in links:
        #print(link.get('src'))
        
        try:
            #print(link.attrs)
            if 'href' in link.attrs:
                figure_link = link.get('href')
            elif 'srcset' in link.attrs:
                figure_link = link.get('srcset')
            elif 'src' in link.attrs:
                figure_link = link.get('src')
            #elif 'alt' in link.attrs:
            #    figure_link = link.get('alt')
                #print('Valid link: ', figure_link)
            if figure_link[:2] == '//': # 'https://' not in 
                figure_link = 'https:' + figure_link
            elif figure_link[:1] == '/':
                figure_link = domain[:-1] + figure_link
            #print('Valid link: ', figure_link)
        except: 
            print('not a valid link')
            continue # break
        #print(figure_link)
        fig_title = figure_link.rpartition('/')[2]
        fig_title = re.sub(r"[^\.0-9a-zA-Z]+", "", fig_title)
        if screenshots: # takes alot of time
            screen_capture(figure_link, title, fig_title)
        if save_figures:
            import_figure(figure_link, title, fig_title)
    return

def extract_table_captions(soup, title): 
    """
    Extract tables from article
    
    Input: 
        soup = extraction from case study
        title = title of the document
    """
    dir_figs = 'results/%s/0_raw/figures' % (title)
    txt = ''
    # tbl
    for ix, div in enumerate(soup.find_all('div', attrs={'class': "inline-table__tail"})):
        txt += '\nTable %s:\n' % (ix)
        # section-paragraph
        captions = div.findAll('div', attrs={'class': "section-paragraph"}) # section-paragraph
        #captions = remove_tag_caption(captions[0]) 
        captions = html_to_text(str(captions[0])) # format to text
        clean = re.compile('<.*?>')
        cleansed = re.sub(clean, '', captions)
        txt += cleansed
    if txt != '':
        save_captions(txt, title, 'captions_tables.txt')
    return 

def get_figure_caption(soup):
    """
    Check for low-hanging fruit (is figure caption directly available?)
    
    Input:
        soup = content of case study (BeautifulSoup object)
        
    Output:
        captions = list of captions
        boolean (captions == []) -> indicating whether or not a 
             caption has been found
    """
    captions = []
    classes = []
    text = []
    for ix, div in enumerate(soup.findAll('figure')):
        for element in div.findAll(class_=True):
            #print('Name:', element.name, 'Attrs:', element.attrs)
            classes.extend(element["class"])
            text.extend([element.text for i in range(len(element["class"]))])
        if classes == []:
            continue
        cl, fig = predict_caption_figure(classes, text)
        #print('CL:', cl)
        for i in ['div', 'span', 'p', 'caption']:
            if captions == []:
                captions = div.findAll(i, attrs={'class': cl}) # section-paragraph
            else :
                break   
    #print(len(captions))
    return captions, captions==[]
    

def extract_undefined_captions(soup, title): 
    """
    This function searches for captions within the 
    text. Due to the heterogeneity of the case studies, this
    is pretty broad screening, with prediction capabilities.
    
    Maybe extend list of allowed types (based on emperical observation)
    """
    txt = ''
    captions, captions_found = get_figure_caption(soup)
    allowed_types = ['section', 'div', 'body', 'header', 'table']
    
    for sect in allowed_types: 
        captions = captions # TEMPORARY FIX -> doesn't keep order in tact!
        for ix, div in enumerate(soup.findAll(sect)): #for ix, div in enumerate(soup.find_all('a')):
            text = []
            links = []
            classes = []
            #print('Name:', div.name)

            for element in div.findAll(class_=True):

                if element.name in allowed_types:
                    #print(element["class"], 'Name:', element.name, 'Attrs:', element.attrs)
                    classes.extend(element["class"])
                    text.extend([element.text for i in range(len(element["class"]))])
                    #for tag in element:
                    #    print(tag)

            if classes == []:
                continue


            ## only predict if you don't know for certain!
            cl, caption_found = predict_caption(classes, text)

            if caption_found: 
                [captions.append(x) for x in soup.findAll(sect, attrs={'class': cl}) if x not in captions] 
                #for match in captions:
                #    match.decompose()
                #print('CL captured:', cl)
    #print('CAPTIONS:', captions)  
    print('Special elements found:', len(captions))
    clean = re.compile('<.*?>')
    for index, cap in enumerate(captions):
        #try:
        #    cleansed = cap.get_text()
        #except : 
        #cap = remove_tag_caption(cap) # hopefully this works everytime

        capt_text = html_to_text(str(cap.get_text())) # format to text
        cleansed = re.sub(clean, '', capt_text)

        if cleansed not in [None, '']:
            txt += 'Entity%s\n' % (index) + cleansed + '\n'
            index += 1
            #cap.decompose()
    if txt != '': # only save if you have content
        save_captions(txt, title, 'captions.txt')

def get_caption_from_link(soup, title, domain, save_figures=True):
    """
    Sometimes the captions of figures / tables are mentioned in the links
    """
    txt = ''
    for ix, div in enumerate(soup.find_all('body')): #for ix, div in enumerate(soup.find_all('a')):
        classes = []
        text = []
        links = []
        for element in div.findAll('a', class_=True):
            classes.extend(element["class"])
            text.extend([element.text for i in range(len(element["class"]))])
        if classes == []:
            continue
        #print(text)
        cl = predict_caption_figure(classes, text)
        #print(cl)
        captions = div.findAll('a', attrs={'class': cl}) # section-paragraph
        if captions == []:
            continue
        clean = re.compile('<.*?>')
        for cap in captions:
            try:
                cleansed = cap.get('title')
            except : 
                cap = html_to_text(str(cap)) # format to text
                cleansed = re.sub(clean, '', cap)
            if cleansed not in [None, '']:
                txt += cleansed + '\n'
    if txt != '': # only save if you have content
        save_captions(txt, title, 'captions_figures.txt')

def predict_caption_figure(classes, txt):
    """
    Input:
        classes = list with class names
        txt = text for caption
        
    Predict which class refers to the caption of a figure.
    It checks if all expected words are found in class name
    
    The only check we do now with regard to the txt of caption, is
    too see which one has the longest caption. This is to
    prevent mistaking the title of the caption for the description. 
    """
    #scores = []
    max_score = 0
    max_ix = 0
    max_len = 0
    ix = 0
    fig = False
    for cl in classes:
        score  = 1*('figure' in cl) + 1*('caption' in cl) + 1*('body' in cl) + 1*('text' in cl) + 1*('description' in cl) + \
                1*('figure' in txt[ix].lower()) + 1*('caption' in txt[ix].lower()) + 1*('description' in txt[ix].lower())
        #scores.append(score)
        if score > max_score or (score == max_score and len(txt[ix]) > max_len) & score != 0:
            max_ix = ix
            max_len = len(txt[ix])
            max_score= score
            fig = True
        ix += 1
        
    return classes[max_ix], fig

def predict_legend_table(classes, txt):
    """
    Input:
        classes = list with class names
        txt = text for legend
        
    Predict which class refers to the legend of a table
    It checks if all expected words are found in class name
    
    Output:
        classes[max_ix] = class that is most likely to be a legend
        legend = boolean indicating whether a legend class has actually been found
    """
    #scores = []
    max_score = 0
    max_ix = 0
    max_len = 0
    ix = 0
    legend = False 
    for cl in classes:
        score  = 1*('legend' in cl)
        if score > max_score :
            max_ix = ix
            max_score= score
            legend = True
        ix += 1
    return classes[max_ix], legend

def extract_custom_headers(soup):
    """  
    Automatically predict the custom headers based on the provided html file.
    
    Input:
        soup = raw html file
    """
    title_classes = []
    for ix, div in enumerate(soup.find_all('body')):
        for elem in div.findAll(class_=True):
            if 'title' in ' '.join(elem['class']) or 'head' in ' '.join(elem['class']):
                title_classes.append(' '.join(elem['class']))
    title_classes = list(set(title_classes))
    return title_classes

## Functions Screening Documents

def find_acronyms(txt):
    """
    Build a dictionary with all of the acronyms mentioned in the text.
    
    Checks if first letter of entity and first letter of acronym are matching
    
    Assumptions for Acronym expansion:
     - acronym is larger than 1 character
     - entities have a max length of 4 words
     - acronyms start with same letter as entity
     - the only special character allowed within an
         entity is the '-' character
    
    Input:
        txt = format free text fields from case study
    """
    d_acronyms = {}
    p = re.compile(r'(((\w)(?:\w|-)* )(\w* ){0,3})\(((?i)(\3)[^)\s]+)\)')
    matches = re.findall(p, txt)
    
    for match in matches:
        acronym = match[4]
        expanded = match[0].rstrip() # remove trailing space
        d_acronyms[acronym] = expanded
    return d_acronyms

def select_relevant_text(parsed_list, d_phenotype, bin_size, min_power, frames):
    """
    Description:
    
    Tag the relevant text - where a higher prevalence of phenotypes
    is found.
    
    """
    #print(parsed_list)
    reading_frames = {}
    for i in range(frames):
        reading_frames[i] = 0
    bin_ix = 0
    txt = ''
    start_str = '<span style="color:red">'
    end_str = '</span>'
    for ix, sent in enumerate(parsed_list): 
        passing = False
        for frame in reading_frames.keys():
            frame_ix = reading_frames[frame]
            if d_phenotype[frame][frame_ix] > min_power:
                passing = True 
        if passing== 1:
            txt += ' ' + start_str + parsed_list[ix] + end_str + ' ' 
        else :
            txt += parsed_list[ix] + ' '
        for frame in reading_frames.keys():
            frame_ix = reading_frames[frame]
            if (ix+1+frame) % bin_size == 0 :
                reading_frames[frame] += 1
    return txt

## Cleaning functions
def simpleCleaning(sent): 
    """
    Remove special characters that are not relevant to 
    the interpretation of the text

    Input:
        sent = free written text
    Output :
        processed sentence
    """
    sticky_chars = r'([!#,.:";@\-\+\\/&=$\]\[<>\'^\*`\(\)])'
    sent = re.sub(sticky_chars, r' ', sent)
    sent = re.sub(r'([\t|\n|\r])', r' ', sent)
    sent = sent.lower()
    return sent

def removeAccent(text):
    """
    This function removes the accent of characters from the text.
    Variables:
        text = text to be processed
        
    Sometimes spaces dissappear - BUG !! 
    Sometimes whole text dissappears
    """
    #try:
    #    text = unicode(text, 'utf-8')
    #except NameError: # unicode is a default on python 3 
    #    pass
    unaccented_text = unidecode(text)
    #text = unicodedata.normalize('NFD', text)
    #text = text.encode('ascii', 'ignore')
    #text = text.decode("utf-8")
    return unaccented_text

def html_to_text(soup):
    """
    Formatting HTML to text, ensuring that the spaces / newlines are preserved
    
    Todo: Maybe also for dl/dt?
    
    Input:
        soup = extracted text from html
    """
    new_soup = re.sub(r"<br>", "\n", soup)
    new_soup = re.sub(r"<hr/>", "\n", new_soup)
    new_soup = re.sub(r"</p>", "\n", new_soup)
    new_soup = re.sub(r"</td>", "</td>\n", soup)
    return new_soup

def html_add_spaces(soup):
    """
    Formatting HTML to text, ensuring that the spaces / newlines are preserved
    
    Todo: Maybe also for dl/dt?
    
    Input:
        soup = extracted text from html
    """
    new_soup = re.sub(r">", "> ", soup)
    new_soup = re.sub(r"<", " <", new_soup)
    return new_soup

def regex_cleaning(soup, title_classes=[], remove_accent=False):
    """ 
    Clean Html text with regular expression rules. Headers are 
    preserved to ensure readability. Custom header classes can be 
    provided by the user.
    
    Input:
        soup = extracted text from html
        title_classes = list with custom header flags
        remove_accent = remove special characters
    """
    # preserve headers
    new_soup = re.sub(r"\<h([1-6])[^\>]*>", r"@h\1@", soup) 
    new_soup = re.sub(r"\</h([1-6])[^\>]*>", r"@/h\1@", new_soup)
    
    if title_classes != []: ## preserve special div title classes
        new_soup = re.sub(r"(\<div class=\"(?:%s)\"[^>]*>[^<]*)(</div>)" % ('|'.join(title_classes)), r"\1@/h5@", new_soup)
        new_soup = re.sub(r"\<div class=\"(?:%s)\"[^>]*>" % ('|'.join(title_classes)), r"@h5@", new_soup)

    #print(len(new_soup))
    #new_soup = re.sub(r"\<[^\>]+\>", "", new_soup)
    new_soup = re.sub(r"(\<\!\-\-[^\]]*]\>[^\]]*\<\!\[endif\]\-\-\>)", '', new_soup) # remove comments!
    new_soup = re.sub(r"‐", " ", new_soup)
    new_soup = re.sub(r"<br>", "\n", new_soup)
    new_soup = re.sub(r"<hr/>", "\n", new_soup)
    new_soup = re.sub(r"</p>", "\n", new_soup)
    #print(len(new_soup))
    new_soup = re.sub(r"\<em\>|\</em\>", " ", new_soup) # remove em tags ? (overbodig?)
    new_soup = re.sub(r"\<[^\>]+\>", "", new_soup) #remove All html tags
    new_soup = re.sub(r"\s{3,}", r'<br>', new_soup) # change excessive spaces into a single newline
    #print(len(new_soup))
    new_soup = re.sub(r"\.([A-Z])", r'. \1', new_soup) # add whitespace where a new sentence is started
    new_soup = re.sub("\n", r"<br>\n", new_soup) # format newlines to <br> -> WEIRD FLEX
    new_soup = re.sub(r";", ":", new_soup) # Otherwise excel reads it incorrectly!!
    #print(len(new_soup))
    # restore preserved headers
    new_soup = re.sub(r"@h([1-6])@", r"<h\1>", new_soup)
    new_soup = re.sub(r"@/h([1-6])@", r"</h\1>\n", new_soup)
    #new_soup = re.sub(r'[^A-z]([0-9]*)([A-z]*)', '\1 \2', new_soup) # seperate numbers from characters!
    #print(len(new_soup))
    if remove_accent:
        new_soup = removeAccent(new_soup)
    return new_soup

## PRODUCT 0
def first_screening(parsed_list, first_intercept, bin_size, min_power, frames=5):
    """
    Scan a document for phenotypes. 
    
    Divide the parsed document over multiple bins / regions. Text is then 
    highlighted if the prevalence of phenotypes within the bin is larger 
    than expected. Discerning the relevant regions from the background.
    
    Input: 
        parsed_list = parsed document
        first_intercept = list with intercepted points (assoc. to phenotypes)
        bin_size = size of the bin (size reflects nr of subsents)
        min_power = minimal number of phenotypes that should be mentioned
    """
    bin_ix = 0
    bin_sum = 0
    
    reading_frames = {}
    for i in range(frames):
        reading_frames[i] = 0
    
    #reading_frames = {0: 0, 1:0, 2:0, 3:0, 4:0} #[0, 1, 2, 3, 4]
    d_phenotype = {}

    for frame in reading_frames: # initialize
        d_phenotype[frame] = [0]

    for ix, sent in enumerate(parsed_list): 
        if ix in first_intercept:
            for frame in reading_frames.keys():
                frame_ix = reading_frames[frame]
                val = d_phenotype[frame][frame_ix]
                if type(first_intercept) == list:
                    d_phenotype[frame][frame_ix] = val + first_intercept.count(ix)
                elif type(first_intercept) == dict:
                    d_phenotype[frame][frame_ix] = val + len(first_intercept[ix])
        for frame in reading_frames.keys():
            frame_ix = reading_frames[frame]
            if (ix+1+frame) % bin_size == 0 :
                l_values = d_phenotype[frame]
                l_values.append(0)
                d_phenotype[frame] = l_values 
                reading_frames[frame] += 1
    for frame in reading_frames.keys():
        frame_ix = reading_frames[frame]
        if (ix+1+frame) % bin_size == 0 :
            l_values = d_phenotype[frame]
            l_values.append(0)
            d_phenotype[frame] = l_values 
            reading_frames[frame] += 1
    txt = select_relevant_text(parsed_list, d_phenotype, bin_size, min_power, frames)
    return txt, d_phenotype

## Functions for phenotyper
def load_common_phenotypes(commonFile):
    returnSet = set()
    for line in open(commonFile): returnSet.add(line.strip())
    return returnSet

def clinphen(inputFile, srcDir, extensive=True, custom_thesaurus="", rare=False):
    """
    Employ ClinPhen to infer HPO-codes based on format-free text 
    
    first_intercept = First intercepted phenotypes (from a simple Screening)
    
    Extensive: perform an extensive search
    """
    #srcDir
    hpo_main_names = srcDir + "/hpo_term_names.txt"

    def getNames():
        returnMap = {}
        for line in open(hpo_main_names):
            lineData = line.strip().split("\t")
            returnMap[lineData[0]] = lineData[1]
        return returnMap
    hpo_to_name = getNames()

    inputStr = ""
    for line in open(inputFile): inputStr += line
    if extensive:
        if not custom_thesaurus: returnString = get_phenotypes.extract_phenotypes(inputStr, hpo_to_name, extensive)
        else: returnString = get_phenotypes.extract_phenotypes_custom_thesaurus(inputStr, custom_thesaurus, hpo_to_name)
        if not rare: return returnString
    else:
        if not custom_thesaurus: returnString, first_intercept, lines = get_phenotypes.extract_phenotypes(inputStr, hpo_to_name, extensive)
        else: returnString = get_phenotypes.extract_phenotypes_custom_thesaurus(inputStr, custom_thesaurus, hpo_to_name)
        if not rare: return returnString, first_intercept, lines
    #print('qq')
    items = returnString.split("\n")
    returnList = []
    common = load_common_phenotypes(srcDir + "/common_phenotypes.txt")
    for item in items:
        HPO = item.split("\t")[0]
        if HPO in common: continue
        returnList.append(item)
    if extensive == True:
        return "\n".join(returnList)
    elif extensive == False :
        return "\n".join(returnList), first_intercept, lines
    
def clinphen_str(inputStr, srcDir, extensive=True, custom_thesaurus="", rare=False): # Move this to DeepPhenotyping_functions
    """
    Employ ClinPhen to infer HPO-codes based on format-free text 
    
    first_intercept = First intercepted phenotypes (from a simple Screening)
    
    Extensive: perform an extensive search
    """
    #srcDir
    hpo_main_names = srcDir + "/hpo_term_names.txt"

    def getNames():
        returnMap = {}
        for line in open(hpo_main_names):
            lineData = line.strip().split("\t")
            returnMap[lineData[0]] = lineData[1]
        return returnMap
    hpo_to_name = getNames()

    if extensive:
        if not custom_thesaurus: returnString = get_phenotypes.extract_phenotypes(inputStr, hpo_to_name, extensive)
        else: returnString = get_phenotypes.extract_phenotypes_custom_thesaurus(inputStr, custom_thesaurus, hpo_to_name)
        if not rare: return returnString
    else:
        if not custom_thesaurus: returnString, first_intercept, lines = get_phenotypes.extract_phenotypes(inputStr, hpo_to_name, extensive)
        else: returnString = get_phenotypes.extract_phenotypes_custom_thesaurus(inputStr, custom_thesaurus, hpo_to_name)
        if not rare: return returnString, first_intercept, lines
    items = returnString.split("\n")
    returnList = []
    common = load_common_phenotypes(srcDir + "/common_phenotypes.txt")
    for item in items:
        HPO = item.split("\t")[0]
        if HPO in common: continue
        returnList.append(item)
    if extensive == True:
        return "\n".join(returnList)
    elif extensive == False :
        return "\n".join(returnList), first_intercept, lines

def txt2hpo_str(inputStr):
    extract = Extractor()
    returnList = extract.hpo(inputStr.lower()).hpids
    return returnList # "\n".join(

def ncr_str(inputStr):
    """
    Utilize the Neural Concept Recognizer ( a deep learning phenotyper).
    
    Return df with HPO and exact location
    """ 
    params = (
        ('text', inputStr),
    )
    response = requests.get('https://ncr.ccm.sickkids.ca/curr/annotate/', params=params)
    #print(response)
    p = re.compile(r'\<Response \[[0-9]*\]\>')
    
    if p.match(inputStr): # only if you don't have a negative response
        # Potentially causes error if " or ' in text!
        d_ncr = {
        'HPO ID' : [i['hp_id'] for i in ast.literal_eval(response.text)['matches']],
        'names' : [i['names'] for i in ast.literal_eval(response.text)['matches']],  
        'start' : [i['start'] for i in ast.literal_eval(response.text)['matches']],    
        'end' : [i['end'] for i in ast.literal_eval(response.text)['matches']],   
        'score' : [i['score'] for i in ast.literal_eval(response.text)['matches']],
        'line' : [inputStr for i in ast.literal_eval(response.text)['matches']] # add more lines
        }
        df_hpo =pd.DataFrame.from_dict(d_ncr)
    else: 
        df_hpo = pd.DataFrame(columns=['HPO ID', 'names', 'start', 'end', 'score', 'line'])
    return df_hpo 

def ncr_str_chunk(lines, batch_size=15):
    """
    Utilize the Neural Concept Recognizer on a large text by chunking
    
    Return df with HPO and exact location. Now it will also save the line
    
    be careful - ensure that url is not too long (do not make the batchsize too big!)
    
    Return:
        first_intercept = dictionary containing the intercepted phenotypes
        l_batches = list with all batches
    """ 
    first_intercept = {}
    batch_size = 15 
    if len(lines) <= batch_size:
        batch_size = len(lines)-1
    
    cur_batch = 0
    batch = ''
    l_batches =[]
    #print(len(lines))
    for ix, line in enumerate(lines): 
        if ix != 0 and (ix % batch_size == 0 or ix == len(lines)-1):
            #print(batch)
            params = (
                ('text', batch),
                )
            response = requests.get('https://ncr.ccm.sickkids.ca/curr/annotate/', params=params)
            
            if response.status_code == requests.codes.ok:
                d_val = ast.literal_eval(response.text)['matches']
                #print(d_val)
                if d_val != {}: # New: also save line
                    for d in d_val: 
                        line_nr, cur_line = calculateLine(batch, d, cur_batch, batch_size)
                        d['line'] = cur_line
                        d['match'] = batch[d['start']:d['end']]
                        #l_batches.append(cur_line) ## Remove later??
                        if line_nr in first_intercept:
                            first_intercept[line_nr].append(d)
                        else :
                            first_intercept[line_nr] = [d]
            else : 
                print('Error', response, 'Batch size may be too large!')
            #print(first_intercept)
            #print(eql)
            cur_batch += 1
            #l_batches.append(batch)
            
            batch = ''
            print('Batch', str(cur_batch), str(round(len(lines)/batch_size)))
        batch += line + '\n'
        l_batches.append(line)
    #print(first_intercept)
    #print(eql)
    return first_intercept, l_batches

def scispacy_str(nlp, hpo, umls_to_hpo, lines):
    """ 
    nlp = Natural Language Processing pipeline
    hpo = entity linker (HPO)
    """
    l_hpo = []
    for ix, line in enumerate(lines): 
        ents = nlp(line.lower()).ents # .text
        #print(ents)
        l_hpo.extend(inferHPO(ents, umls_to_hpo, hpo))
    return l_hpo

def inferHPO(row, umls_to_hpo, hpo):
    """ 
    row = all found entities (in spacy Span format)
    hpo = entity linker (HPO)
    
    Description:
        Infer hpo codes based on the found entities.
    """
    hpo_list = []
    for entity in row:
        for umls_ent in entity._.kb_ents:
            try :
                hpo_list.append(umls_to_hpo[hpo.kb.cui_to_entity[umls_ent[0]].concept_id])
            except :
                continue
    return hpo_list

def clinphen_extensive_search(lines):
    # clinphen_extensive_search
    #lines = [item for sublist in lines for item in sublist]
    new_d = {}
    index= 0
    for ix, line in enumerate(lines): 
        #items = clinphen_str(line,'data') # func
        items, first, txt = clinphen_str(line,'data', extensive=False)
        df_hpo = pd.DataFrame([n.split('\t') for n in items.split('\n')])
        df_hpo.columns = df_hpo.iloc[0]
        df_hpo = df_hpo.reindex(df_hpo.index.drop(0))
        l_row_pheno = []
        for index, row in df_hpo.iterrows():
            stepsize = round(len(line.split(' '))/len(df_hpo))
            #start = txt.find(line)
            #end = txt.find(line) + len(line) - 1
            d_val = {}
            d_val['hp_id'] = row['HPO ID']
            d_val['start'] = stepsize*index
            d_val['names'] =[row['Phenotype name']]
            d_val['end'] = stepsize*(index+1)
            d_val['score'] = '1'  # doesn't calculate a conf
            d_val['line'] = line
            d_val['match'] = row['Phenotype name']
            l_row_pheno.append(d_val) 
        new_d[ix] = l_row_pheno
        if ix % 10 == 0:
            print('Iteration ', str(ix), '/', str(len(lines)))
    return new_d

def clinphen_extensive_search_back(first_intercept, lines):
    # clinphen_extensive_search
    lines = [item for sublist in lines for item in sublist]
    new_d = {}
    index= 0
    for ix, line in enumerate(lines): 
        if ix in first_intercept: 
            items = clinphen_str(line,'data') # func
            df_hpo = pd.DataFrame([n.split('\t') for n in items.split('\n')])
            df_hpo.columns = df_hpo.iloc[0]
            df_hpo = df_hpo.reindex(df_hpo.index.drop(0))
            l_row_pheno = []
            for index, row in df_hpo.iterrows():
                stepsize = len(row['Example sentence'])/len(df_hpo)
                d_val = {}
                d_val['hp_id'] = row['HPO ID']
                d_val['start'] = stepsize*index
                d_val['names'] =[row['Phenotype name']]
                d_val['end'] = stepsize*(index+1)
                d_val['score'] = '1'  # doesn't calculate a conf
                d_val['line'] = line
                d_val['match'] = row['Phenotype name']
                l_row_pheno.append(d_val)
            new_d[ix] = l_row_pheno
        else :
            continue
    return new_d

def txt2hpo_str_chunk(lines):
    """
    Utilize the txt2hpo tool on a large text by chunking
    
    Return df with HPO and exact location
    """ 
    first_intercept = {}
    extract = Extractor()
    for ix, line in enumerate(lines): 
        returnList = extract.hpo(line.lower()).entries
        l_row_pheno = []
        for i in returnList:
            d_val = {}
            d_val['hp_id'] = i['hpid'][0]
            d_val['start'] = i['index'][0]
            d_val['end'] = i['index'][1]
            d_val['score'] = '1' # doesn't calculate a conf
            d_val['line'] = line
            d_val['match'] = i['matched']
            l_row_pheno.append(d_val)
        first_intercept[ix] = l_row_pheno
        #print(ix)
    return first_intercept

## Functions - Parsing Tables

def tableDataText(table):    
    """Parses a html segment started with tag <table> followed 
    by multiple <tr> (table rows) and inner <td> (table data) tags. 
    It returns a list of rows with inner columns. 
    Accepts only one <th> (table header/data) in the first row.
    """
    def rowgetDataText(tr, coltag='td'): # td (data) or th (header)       
        return [td.get_text(strip=True).replace('\n','').replace('\r','').replace(';','') for td in tr.find_all(coltag)]  
    rows = []
    trs = table.find_all('tr')
    headerow = rowgetDataText(trs[0], 'th')
    if headerow: # if there is a header row include first
        rows.append(headerow)
        trs = trs[1:]
    for tr in trs: # for every table row
        rows.append(rowgetDataText(tr, 'td') ) # data row   
    return rows

def parseTable(table, remove_inc=True):
    """
    Convert Table to pandas Dataframe
    
    Input:
        table = html table from article
        remove_inc = remove rows with incosistent length
            (be careful: these can be helpful to categorize table)
    """
    #print('y')
    list_table = tableDataText(table)
    dftable = pd.DataFrame(list_table[1:], columns=list_table[0])
    if remove_inc:
        med = np.median(dftable.isnull().sum(axis=1).values)
        dftable = dftable.dropna(thresh=len(dftable.columns)-med)
    return dftable

def check_for_link(table, domain, index, title):
    """
    Sometimes the table tag refers to supplement with a link. 
    In that case, we want to download the  supplemented tables 
    (if these are in xlsx or xls format).
    
    Input:
        table = beautifulsoup / html content that consists of a table
    
    Output:
        This function outputs a boolean that indicates whether or not 
        links were found in the html box 
    """
    links = table.findAll('a', attrs={'href': re.compile("^(?://|http\://|/)")}) # maybe http as well
    link_ix = 0
    for link in links:
        try:
            table_link = link.get('href')
        except:
            break
        if table_link[:2] == '//': # 'https://' not in 
            table_link = 'https:' + table_link
        elif table_link[:1] == '/':
            table_link = domain[:-1] + table_link
        elif 'http' not in table_link:
            break
        table_title = table_link.rpartition('/')[2]
        import_table(table_link, title, "Table_%s_%s_%s" % (str(index), str(link_ix), str(table_title)))
        link_ix += 1
    if link_ix != 0:
        return True
    else :
        return False
   
def save_tables(title, tables, remove_inc=True):
    """
    Save tables found in html file as csv files
    
    Input:
        title = title of article
        tables = list of html tables
        remove_inc = remove rows with incosistent length (be careful: these can improve readability, or perhaps include valuable information)
    """

    for index, table in enumerate(tables):
        with open("results/%s/0_raw/tables/Raw_table_%s_%s.html" % (title, str(index), title), "w", encoding="utf-8") as file:
            table = str(table).replace(';',':') # easier to read for excel
            file.write(table)
        try : 
            raw_table = parseTable(table, remove_inc)
        except:
            try :
                raw_table = readTableDirectly(table)
                raw_table.to_csv("results/%s/0_raw/tables/Table_%s_%s.csv" % (title, str(index), title), sep='|', index=False, encoding='utf-8-sig',)
                continue
            except:
                continue
        raw_table = raw_table.astype('str')
        raw_table.to_csv("results/%s/0_raw/tables/Table_%s_%s.csv" % (title, str(index), title), sep='|', index=False)
        index += 1
    return

def readTableDirectly(table):
    """
    Parses an HTML table directly
    
    Input:
        table = html table
    Output:
        dfs = table converted to a pandas Dataframe
    """
    table = str(table).replace(';',':') # easier to read for excel
    dfs = pd.read_html(table)[0]
    dfs.columns = dfs.columns.to_flat_index()
    return dfs
    
def calculateLine(batch, d_val, cur_batch, batch_size):
    """
    Reverse engineer the line number from a batch of text
    
    Input:
        batch = text from a single batch
        d_val = dictionary with found phenotypes
        cur_batch = current batch 
        batch_size = size of each batch 
    
    Output:
        line_nr = global number of line 
        line = line from a batch
    """
    start = d_val['start']
    text = batch[:start]
    newlines_found = len(segmentation(text))
    line_nr = batch_size * cur_batch + newlines_found -1 # klopt deze -1
    line = segmentation(batch)[newlines_found-1]
    return line_nr, line
    
    
def segmentation(text):
    """
    Segment the content of the case study
    
    Input:
        text = content of the case study (parsed HTML)
    Output:
        sentences = list of sentences
    Do not split on <br> -> otherwise you'll remove the structure.
    """
    text = text.replace(';',':') # remove semicolon to ensure excel can open the file.
    #text = text.replace('\n','\n [NEWLINE]') # keep \n for new checks
    #text = text.replace('\t','[NEWLINE]')
    paragraphs = [p for p in re.split('\n|\t', text) if p]
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    return sentences

## Functions Scanning Tables

### Normal Table
def find_binary_columns(table):
    """
    Description: 
    Find out which columns qualify as being binary columns. 
    In other words, check if values fall into one of the three categories:
     - YES category = phenotype present
     - NO category = phenotype absent
     - UNKNOWN category = phenotype was not mentioned
    
    Input:
        table = Pandas Dataframe 
    Output:
        l_qualify = list of columns that qualify
    """
    l_yes = ['y', 'true', 't', 'yes', '1', 'present', 'p', 'pres', '+'] 
    l_no = ['n', 'false', 'f', 'no', '0', 'absent', 'a', 'abs', '-', '−']
    l_unknown = ['u', 'unknown', 'na', 'nan', '', 'nd', 'not determined'] # , ''
    l_not = []
    for col in table.columns:
        l_val = table[col].values
        for i in l_val:
            val = str(i).lower()
            val = re.sub("[\(\[].*?[\)\]]", "", val)
            val = val.strip()
            if (val not in l_yes and val not in l_no and val not in l_unknown):
                l_not.append(col)
                continue
    l_qualify = list(set(table.columns) - set(l_not))
    return l_qualify


def link_col_hpo(l_qualify, phenotyper='clinphen'):
    """
    Link list with columns to HPO. Columns are preprocessed
    prior to screening
    
    Input:
        l_qualify = list of columns that qualify as boolean vectors
    Output: 
        d_col = dictionary with column names linked to assoc. phenotype ID (HPO)
    """
    d_col = {}
    for col in l_qualify:
        l_hpo = []
        d_col[col] = phenotyper_interp(col, phenotyper)
    return d_col

def col_hpo(row, d_col):
    """
    Acquire all positive cases in binary column and infer 
    the corresponding phenotype (mentioned in the header), if
    it was captured by the phenotyping extraction tool
    
    Input:
        row = row in table from case study
        phenotyper = HPO-extraction tool
    Output:
        d_col = dictionary with values from column
    """
    l_yes = ['y', 'true', 't', 'yes', '1', 'present', 'p', 'pres', '+'] 
    l_hpo = []
    #print(d_col) ### 
    for ix, i in enumerate(row):
        col_name = list(row.keys())[ix]
        val = str(i).lower()
        val = re.sub("[\(\[].*?[\)\]]", "", val)
        val = val.strip()
        #print(val, col_name) ### 
        
        if val in l_yes:
            l_hpo.extend(d_col[col_name])
    return l_hpo

def row_hpo(row, phenotyper='clinphen'):
    """
    Scan all phenotypes mentioned in row with the provided phenotyper tool
    
    Input:
        row = row in table from case study
        phenotyper = HPO-extraction tool
    Output:
        l_hpo = list of HPO-ids
    """
    l_hpo = [] 
    values = [str(i) for i in row.values] # first convert to string
    row_content = ' '.join(list(values))
    l_hpo = phenotyper_interp(row_content, phenotyper)
    return l_hpo

def scan_table(table, phenotyper='clinphen'): ## Add function for scanning rows -> text. Then add this to function list
    """
    Scan table in case study for phenotypes. 
    
    It checks both phenotypes mentioned in rows, as well as phenotypes in
    columns.
    """
    l_qualify = find_binary_columns(table)
    print('Binary columns:', l_qualify)
    d_col = link_col_hpo(l_qualify, phenotyper)
    #print(d_col)
    try :
        table['row_hpo'] = table.apply(lambda x : row_hpo(x, phenotyper), axis=1)
        table['col_hpo'] = table.apply(lambda x : col_hpo(x, d_col), axis=1)
    except :
        print('Table has unconventional format')
    return table

### Transposed Table

def is_transposed(table):
    """
    Predict whether or not we are dealing with a transposed table,
    by checking for several flags
    
    ToDo: Might be improved
    
    Input:
    table = parsed pandas Dataframe table pulled from case study
    
    Output:
        Boolean indicating whether or not the table is transposed
    """
    flags = ['patient', 'case', 'casenr', 'patnr', 'caseid', 'family', 'pat']
    l_cols = table.columns
    score = 0
    for col in l_cols:
        val = str(col).lower()
        score +=  1 * (any([True if i in val else False for i in flags ])==True)
    if score > 2*(len(l_cols)/3): # at least 66% should be about patients
        return True
    else : 
        return False

def find_binary_rows(table):
    """
    Description: 
    Find out which columns qualify as being binary columns. 
    In other words, check if values fall into one of the three categories:
     - YES category = phenotype present
     - NO category = phenotype absent
     - UNKNOWN category = phenotype was not mentioned
    
    Input:
        table = Pandas Dataframe 
    Output:
        l_qualify = list of columns that qualify
    """
    l_yes = ['y', 'true', 't', 'yes', '1', 'present', 'p', 'pres', '+'] 
    l_no = ['n', 'false', 'f', 'no', '0', 'absent', 'a', 'abs', '-', '−']
    l_unknown = ['u', 'unknown', 'na', 'nan', '', 'nd', 'not determined'] # , '
    l_not = []
    l_rows = []
    
    for row in table.iterrows():
        row_id = row[1][0]
        l_val = row[1][1:]
        l_rows.append(row_id)
        #print(row_id, ': ', l_val)
        for i in l_val:
            val = str(i).lower()
            val = re.sub("[\(\[].*?[\)\]]", "", val)
            val = val.strip()
            if (val not in l_yes and val not in l_no and val not in l_unknown):
                l_not.append(row_id)
                continue
    l_qualify = list(set(l_rows) - set(l_not))
    return l_qualify, l_rows

def phenotyper_interp(content, phenotyper='clinphen'):
    if phenotyper == 'clinphen':
        items = clinphen_str(simpleCleaning(content),'data')
        df_hpo = pd.DataFrame([n.split('\t') for n in items.split('\n')])
        df_hpo.columns = df_hpo.iloc[0]
        df_hpo = df_hpo.reindex(df_hpo.index.drop(0))
        l_hpo = list(df_hpo['HPO ID'])
    elif phenotyper == 'txt2hpo':
        l_hpo = txt2hpo_str(simpleCleaning(content))
    elif phenotyper == 'ncr':
        df_hpo = ncr_str(simpleCleaning(content))
        l_hpo =list(df_hpo['HPO ID'])
    return l_hpo

def link_row_hpo(l_qualify, phenotyper='clinphen'):
    """
    Link list with rows to HPO. Rows are preprocessed
    prior to screening
    
    Input:
        l_qualify = list of rows that qualify as boolean vectors
    Output: 
        d_row = dictionary with row names linked to assoc. phenotype ID (HPO)
    """
    d_row = {}
    for row_name in l_qualify:
        #l_hpo = []
        d_row[row_name] = phenotyper_interp(row_name, phenotyper)
        #elif phenotyper == 'scispacy':
        #    d_col[col] = scispacy_str(simpleCleaning(col))
    return d_row

def trans_row_hpo(col, l_rows_id, d_row):
    """
    Infer based HPO-codes based on the priorly identified binary rows
    
    Input:
        col = all values from the column
        l_rows_id = list with all row names
        d_row = dictionary with row names that could be linked to a HPO-code
    Output: 
        l_hpo = list of HPO-codes
    """
    l_yes = ['y', 'true', 't', 'yes', '1', 'present', 'p', 'pres', '+'] 
    l_hpo = []
    for ix, i in enumerate(col):
        if l_rows_id[ix] in d_row:
            val = str(i).lower()
            val = re.sub("[\(\[].*?[\)\]]", "", val)
            val = val.strip()
            #print(val, row_id) ### 
        
            if val in l_yes:
                l_hpo.extend(d_row[l_rows_id[ix]])
    return l_hpo

def trans_col_hpo(col, phenotyper='clinphen'):
    """
    Screen every column in a transposed table for phenotypes. 
    A list of HPO-codes is returned by the function
    
    Input:
        col = all values from the column
        phenotyper = phenotype extraction tool
    Output: 
        l_hpo = list of HPO-codes
    """
    l_hpo = [] 
    values = [str(i) for i in col.values] # first convert to string
    col_content = ' '.join(list(values))
    l_hpo = phenotyper_interp(col_content, phenotyper)
    return l_hpo

def scan_transposed_table(table, phenotyper='clinphen'): 
    """
    Scan table (transposed) in case study for phenotypes. 
    
    - Extracts Patient ids from header  
    - Searches for binary rows rather than binary rows
    - 
    
    It checks both phenotypes mentioned in rows, as well as phenotypes in
    columns.
    """
    l_qualify, l_rows_id = find_binary_rows(table)
    print('Binary rows:', l_qualify)
    d_row = link_row_hpo(l_qualify, phenotyper) # only values - skip first column when interpreting
    #table['row_hpo'] = table[table.columns[0]].copy()
    
    try :
        new_row = table.apply(lambda x : trans_row_hpo(x, l_rows_id, d_row), axis=0)
        new_row.update(pd.Series(['row_hpo'], index=[new_row.index[0]]))
        table = table.append(new_row, ignore_index=True)
        new_row = table.apply(lambda x : trans_col_hpo(x, phenotyper), axis=0)
        new_row.update(pd.Series(['col_hpo'], index=[new_row.index[0]]))
        table = table.append(new_row, ignore_index=True)
    except :
        print('Table has unconventional format')
    return table

def scan_main(main_txt, pheno='clinphen'):
    """
    Segment and scan the main text.
    
    Input:
        main_txt = content of the case study (processed html)
        pheno = phenotyper extraction tool
    Output:
        first_intercept = dictionary or list with intercepted phenotypes
        lines = list with the segmented lines of the article
        l_batches = list with batches
    """
    l_batches = []
    if pheno == 'ncr':
        
        lines = segmentation(main_txt)
        first_intercept, l_batches = ncr_str_chunk(lines, batch_size=15)
        #print(first_intercept)
    elif pheno == 'clinphen':
        lines = segmentation(main_txt)
        first_intercept = clinphen_extensive_search(lines) # time expensi
        #items, first_intercept, lines = clinphen_str(main_txt,'data', extensive=False)
        #print(first_intercept)
        #print(eql)
        #print(len(items.split('\n')), len(first_intercept))
        #print(items)
        #print(eqd)
        #if lines == []: # if it can't be chunked further
        #    lines = main_txt
        #first_intercept = clinphen_extensive_search(first_intercept, lines)
        #lines = [item for sublist in lines for item in sublist]
        #print(df_output.head())
    elif pheno == 'txt2hpo':
        lines = segmentation(main_txt)
        first_intercept = txt2hpo_str_chunk(lines)
    return first_intercept, lines, l_batches

## 3. Phenotyping

def phenotypeCaseStudy(new_soup, title, pheno='clinphen', stringent=True, l_patterns=[]):
    """
    Perform phenotyping on the case study content. 
    
    ToDo: append phenotypes from supplementary?
    
    Input:
        new_soup = content of the case study (processed html)
        title = title of the article
        pheno = phenotyper extraction tool
        stringent = boolean indicating whether or not to perform stringent 
            entity linking
        l_patterns = list of patterns to identify entities
    Output: 
        df_hpo = pandas Dataframe with all phenotypes 
           captured from the main text of the article
        d_parsed = dictionary containing all parsed documents
    """
    d_parsed = {} # collect all parsed documents
    first_intercept, lines, l_batches = scan_main(new_soup, pheno)
    d_parsed['Main text'] = lines

    file_name = 'Annotated_%s_%s.html' % (pheno, title)
    df_hpo = main_annotation(title, pheno, file_name, lines, first_intercept, stringent=stringent, l_patterns=l_patterns, l_batches=l_batches) 
    
    print('Phenotypes in main text:', len(df_hpo))
    
    # Screen captions
    caption_file = "results/%s/0_raw/captions.txt" % (title) 
    if os.path.exists(caption_file):
        with open(caption_file, "r", encoding="utf-8") as file:
            captions_txt = file.read()
        first_intercept, lines, l_batches = scan_main(captions_txt, pheno)
        d_parsed['Captions'] = lines
        
        file_name = 'Annotated_%s_captions.html' % (pheno)
        df_hpo2 = main_annotation(title, pheno, file_name, lines, first_intercept, section='Captions', stringent=stringent, l_patterns=l_patterns, l_batches=l_batches)
        df_hpo = df_hpo.append(df_hpo2, ignore_index=True)
        
    # Screen tables 
    df_hpo2, d_parsed = table_annotation(title, pheno, d_parsed, stringent, l_patterns)
    #df_hpo2 = None
    if df_hpo2 is not None:
        if df_hpo2.empty == False:
            df_hpo = df_hpo.append(df_hpo2, ignore_index=True)
        
    print('Phenotypes in captions & table:', len(df_hpo))
    save_all_phenotypes(title, pheno, df_hpo)
    return df_hpo, d_parsed


def annotateCaseStudy(title, pheno='clinphen', entity_linking=True):
    """
    Annotate the case study content, linking each phenotype
    to the right patient
    
    Also acquire locations from the InterceptedPhenotypes table
    
    Input:
        title = title of the article
        pheno = phenotyper extraction tool
        entity_linking = whether or not to create convenient profiles for deep phenotyping
    Output: 
        df_pheno = pandas Dataframe with all phenotypes 
           captured from the table.
    """
    d_pat = collectPhenoProfiles(title, phenotyper=pheno, entity_linking=entity_linking)
    #print(d_pat.values())
    df_pheno = getPatientProfileTable(d_pat)

    df_pheno.to_csv("results/%s/3_annotations/PhenoProfiles_%s.csv" % (title, pheno), sep='|', index=True)
    return df_pheno, d_pat

def table_annotation(title, pheno='clinphen', d_parsed={}, stringent=True, l_patterns = []):
    """
    Annotate the tables and highlight the intercepted phenotypes in 
    an html table. This function is 
    
    In addition, this function also processes the pandas
    dataframe Tables. Respecting the table structure during 
    screening & linking the phenotypes mentioned to the 
    patient ID in corresponding row.
    
    Input:
        title = title of the article
        pheno = phenotyper extraction tool
        d_parsed = dictionary containing all parsed documents
        stringent = whether or not to perform a stringent entity linking
        l_patterns = list of patterns to identify entities
    Output: 
        df_hpo = pandas Dataframe with all phenotypes 
           captured from the tables within the article
        d_parsed = dictionary containing all parsed documents
    """
    result_files = os.listdir("results/%s/0_raw/tables/" % (title))
    tabloid_files = [s for s in result_files if ('Raw_table' in s)]
    table_files = [s for s in result_files if ('Table' in s)]
    
    df_hpo = pd.DataFrame()
    
    # Process HTML tables
    for index, tab in enumerate(tabloid_files):
        print('Screening Table File ' + str(index))
        with open("results/%s/0_raw/tables/%s" % (title, tab), "r", encoding="utf-8") as file:
            html = file.read()
            new_soup = BeautifulSoup(html, "lxml")

        first_intercept, lines, l_batches = scan_main(str(new_soup), pheno=pheno)
        d_parsed['Table File ' + str(index)] = lines
        new_lines = annotate_text(lines, first_intercept, 'Table File ' + str(index), stringent, l_patterns, l_batches) # hopefully this works

        ## collect phenotypes in a dataframe - with additional information
        file_name = 'Annotated_%s_table%s.html' % (pheno, str(index))
        df_hpo2 = main_annotation(title, pheno, file_name, lines, first_intercept, 'Table File ' + str(index), stringent, l_patterns, l_batches)

        if df_hpo2 is not None:
            if df_hpo2.empty == False:
                df_hpo = df_hpo.append(df_hpo2, ignore_index=True)

        with open("results/%s/2_phenotypes/Annotated_tabloid_%s_%s.html" % (title, str(index), pheno), "w", encoding="utf-8") as file:
            file.write(' '.join(new_lines))
            
    # Process pandas dataframe tables
   
    for index, tab in enumerate(table_files):
        print('Process Table ', index)
        try:
            table = pd.read_csv("results/%s/0_raw/tables/%s" % (title, tab), sep='|')
            table = table.fillna('') # very important to fill na prior to function
            if is_transposed(table):
                print('Scan Table (Transposed)')
                annotated_table = scan_transposed_table(table, phenotyper=pheno) #  tables/
            else :
                print('Scan Table (Default)')
                annotated_table = scan_table(table, phenotyper=pheno) #  tables/
            # Save as dataframe
            annotated_table.to_csv("results/%s/2_phenotypes/Table_%s_%s_%s.csv" % (title, str(index), pheno, title), sep='|', index=False)
        except pd.errors.ParserError:
            print('Table %s is not screened due to unconvenient format' % (str(index)))
    return df_hpo, d_parsed

def main_annotation(title, pheno, file_name, lines, first_intercept, section='Main text', stringent=True, l_patterns=[], l_batches=[]):
    """
    Annotate the main text.
    
    Input:
        title = title of the article 
        pheno = phenotypic extraction tool used
        lines = list with the segmented lines of the article
        first_intercept = intercepted phenotypes, with location 
        section = in which file the phenotypes are screened
        stringent = whether or not to perform a stringent entity linking
        l_patterns = list of patterns to identify entities
    """
    if type(first_intercept) != list:
        ## save annotated text
        new_lines = annotate_text(lines, first_intercept, section, stringent, l_patterns, l_batches)

        with open("results/%s/2_phenotypes/%s" % (title, file_name), "w", encoding="utf-8") as file:
            file.write(' '.join(new_lines))

        ## save all found phenotypes
        df_hpo = generate_phenotable(first_intercept, lines)
        
        #df_append = pd.read_csv('results/%s/2_phenotypes/InterceptedPhenotypes_%s.csv' % (title, pheno), sep='|')

        #df_hpo.merge(df_append, left_on='HPO', right_on='hp_id', how='left')
        
    return df_hpo 

def save_all_phenotypes(title, pheno, df_hpo):
    """
    Save all intercepted HPO's 
    
    Input:
        title = title of the article 
        pheno = phenotypic extraction tool used
        df_hpo = pandas Dataframe consisting of the complete collection of extracted phenotypes!
    """
    df_hpo.to_csv("results/%s/3_annotations/Overview_Intercepted_Phenotypes_%s.csv" % (title, pheno), sep='|', index=False)
    return

def add_more_context(key, lines, extra_lines=[3, 3]):
    """
    Add more lines, to facilitate evaluation of the algorithm.
    Since more context is provided it is easier to deduce the misclassifications.
    
    Input :
        key = line number where phenotype is intercepted
        lines = segmented content of case study
        extra_lines = lines to append (before:after) 
    Output: 
        List of neighbouring lines, providing more content to the user
    """
    return '. '.join(lines[key-extra_lines[0]:key+extra_lines[1]])

def generate_phenotable(first_intercept, lines):
    """
    Collect information of intercepted phenotypes in a
    convenient pandas dataframe
    
    The found phenotypes are indexed and the context is
    extracted.
    
    Input:
        first_intercept = dictionary with all intercepted phenotypes
        lines = segmented content of case study 
    """
    df = pd.DataFrame()

    for i, key in enumerate(first_intercept): 
        for item in first_intercept[key]:
            item['index'] = i
            item['context'] = add_more_context(key, lines)
            df = df.append(item, ignore_index=True)
    return df

def predict_identifier_patient(table):
    """
    Input:
        col_name = name of column
        l_values = values assoc. with column

    return: 
        predicted column = column predicted to contain the patient ID
        patient_identifier = indicates if column is even present (boolean)
    """
    max_score = 0
    max_ix = 0
    ix = 0
    patient_identifier = False 
    
    for col in table.columns:
        l_values = table[col]
        col_name = col.lower()
        score  = 1*('id' in col_name) + 1*('patient' in col_name) + 1*('case' in col_name) + 1*('patnr' in col_name) + \
                 1*('patid' in col_name) + 1*('casenr' in col_name) + 1*('caseid' in col_name)  + 1*('individual' in col_name) + 1*('proband' in col_name) # + 1*(len(np.unique(l_values))==len(l_values))
        if score > max_score:
            max_ix = ix
            max_score= score
            patient_identifier = True
        ix += 1
    return table.columns[max_ix],  patient_identifier

def get_patprof_tables(title, tab_files, d_pat):
    """
    Expand the patient phenotypic profiles in d_pat with 
    extra information from the table 
    
    ToDo: Recognize transposed tables
    
    For Transposed tables
        - this function assumes the last two rows to contain the 
            HPO codes.
    
    Input:
        title = title of the article
        tab_files = list of paths to table files to read
        d_pat = dictionary with phenotypic profile (list of HPO's) per patient
    Output:
        d_pat = updated phenotypic profile dictionary
    """
    for tab in tab_files:
        table = pd.read_csv("results/%s/2_phenotypes/%s" % (title, tab), sep='|')
        table = table.fillna('') # very important to fill na prior to function
        #print(table.head())
        if is_transposed(table):
            print('Found transposed Table!')
            
            l_cols = table.columns
            # Loop through all columns
            for col_name in l_cols[1:]:
                if col_name not in d_pat.keys():
                    d_pat[col_name] = ast.literal_eval(table[col_name].iloc[-1])
                else : 
                    d_pat[col_name].extend(ast.literal_eval(table[col_name].iloc[-1]))
                d_pat[col_name].extend(ast.literal_eval(table[col_name].iloc[-2]))
        else : 
            cl, pat_identifier = predict_identifier_patient(table)
            if pat_identifier:
                print('Patient Identifier column found: ', cl)
                for ix, patient in enumerate(table[cl]):
                    key = table[cl].iloc[ix]
                    if key in d_pat.keys():
                        d_pat[key].extend(ast.literal_eval(table['row_hpo'].iloc[ix]))
                    else : 
                        d_pat[key] = ast.literal_eval(table['row_hpo'].iloc[ix])
                    d_pat[key].extend(ast.literal_eval(table['col_hpo'].iloc[ix]))
            else : 
                print("No column seems to represent a patient identifier")
    print('Table', d_pat)        
    return d_pat

def get_intercepted_pheno(title, phenotyper, d_pat, sensitive=True):
    """
    Acquire intercepted phenotypes from output file which 
    was generated by the phenotypeCaseStudy function.
    
    Input:
        title = title of the article
        phenotyper = HPO phenotype extraction tool used
        d_pat = dictionary with phenotypic profile per patient
        sensitive = also includes the phenotypes that couldnt be 
            assigned to any patient
                - Turn this on, if the rudimentary entity linking is lacking
    Output:
        d_pat = updated phenotypic profile dictionary
    """
    #try: 
    df_intercept = pd.read_csv('results/%s/3_annotations/Overview_Intercepted_Phenotypes_%s.csv' % (title, phenotyper), sep='|')
    for pat_id in df_intercept['pat_id'].unique():
        # 
        pat_id = re.sub(r"\([^\>]+\)", "", pat_id) # remove everything between brackets
        pat_id.strip()
        if (sensitive or pat_id != 'None') or ([pat_id] == list(df_intercept['pat_id'].unique())):
            print('No patients found in text, thus we presume the case study concerns a single patient.')
            d_pat[pat_id] = list(df_intercept[((df_intercept['pat_id']==pat_id) & (df_intercept['flags']=='set()') & (df_intercept['negated']==0))]['hp_id'].values)
        
    # (df_intercept['relevant']==1) & flags
    #d_pat['Undefined_case'] = [item['hp_id'] for sublist in df_intercept for item in sublist]
    #except: 
    #    print('Error: Please run the phenotypeCaseStudy before proceeding!') 
    return d_pat

def merge_duplicate_keys(d_pat):
    """
    This function searches for duplicate keys (patient identifiers)
    and automatically merges the features (logical OR) that concern 
    the same key.
    
    Input:
        d_pat = dictionary with phenotypic profile per patient
    Output:
        d_pat = updated phenotypic profile dictionary
    """
    d_copy = d_pat.copy()
    for pat_id in d_copy.keys():
        l_keys = list(d_copy.keys())
        l_keys.remove(pat_id)
        if len(l_keys) > 0:
            mask = np.array([pat_id.lower() in k.lower() for k in l_keys])

            for sim in np.asarray(l_keys)[mask]:
                print(type(sim))
                print('Found (presumed) duplicate key: ', pat_id, '<-->' , sim)
                l1 = set(d_pat[pat_id])
                l2 = set(d_pat[sim])
                l1.update(l2)
                d_pat[pat_id] = list(l1)
                del d_pat[sim]
    return d_pat

def collectPhenoProfiles(title, phenotyper, entity_linking=True):
    """
    Search all files in the results directory & link
    to patients with Identifier.
    
    ToDo: add entity linking for main text
    
    Input:
        title = article title
        phenotyper = phenotyping extraction tool
        entity_linking = whether or not to link the phenotypes to the patients
        
    Output:
        d_pat = updated phenotypic profile dictionary
    """
    result_files = os.listdir("results/%s/2_phenotypes/" % (title))
    table_files = [s for s in result_files if ('Table' in s and phenotyper in s)]
    d_pat = {}
    
    # check if phenotypic abnormality
    
    # Process intercepted 
    d_pat = get_intercepted_pheno(title, phenotyper, d_pat ) #??
    # Consult Tables (NOT EFFICIENT!) -> running scan_table again!
    d_pat = get_patprof_tables(title, table_files, d_pat) # , d_pat
    
    # Merge duplicate patient identifiers
    d_pat = merge_duplicate_keys(d_pat)
    #print(d_pat)
    
    #d_pat = get_patprof_main(title, d_pat, phenotyper, entity_linking)
    
    if d_pat == {}:
        print('no phenotypes found in table')
    
    # toDo: add function entity_linking -> regex rules
    # d_pat = get_patprof_main(title, main_file, d_pat)
    if entity_linking:
        return d_pat
    else :
        return {'all' : [item for sublist in d_pat.values() for item in sublist]}
    

def getPatientProfileTable(d_pat):
    """ 
    Create patient profile table, where a phenotypic profile 
    is linked to a specific individual. 
    
    This is the data that we need to perform deep phenotyping!
    
    Important note: phenotypes that were not mentioned are imputed with 0 
        (assumed to be absent in patient)
    
    Input:
        d_pat = dictionary where patients are linked to phenotypes
    Output:
        df_pheno = pandas Dataframe with all unique HPO's as columns and
            where the patients each have a dedicated row featuring their 
            known phenotypic profile.
    """
    hpo_pool = []
    for pat in d_pat.keys():
        hpo_pool.extend(d_pat[pat])
    columns = list(set(hpo_pool))

    df_pheno = pd.DataFrame(columns=list(set(hpo_pool)))
    
    

    for ix, pat in enumerate(d_pat.keys()):
        l_val = [1 if val in d_pat[pat] else 0 for ix, val in enumerate(df_pheno.columns) ]
        df_pheno.loc[pat] = l_val
    return df_pheno

## 4. Entity Linking Main Text / Annotation
def get_superclass(graph, hpo_id, name_to_id, id_to_name):
    """ 
    Acquire parent features of provided HPO-id
    
    Input:
        graph = HPO DAG tree
        hpo_id = HPO ID
        l_col = values of column (implies superclass = same value)
        name_to_id = dictionary where names are linked to hpo-id codes
        id_to_name = dictionary where hpo-id codes are linked to names
    Output:
        dataframe with inferred parent classes (ancestor nodes of provided hpo-id)
    """
    paths = nx.all_simple_paths(
    graph,
    source=name_to_id[hpo_id],
    target=name_to_id['Phenotypic abnormality'] # phenotypic abnormality
    )
    
    hpo_list = []
    desc_list = []
    
    for path in paths:
        #print('•', ' ⟶ '.join(node for node in path))
        for node in path:
            hpo_list.append(node)
            desc_list.append(id_to_name[node])
            
    data = {'HPO id': hpo_list, 'Phenotype': desc_list}
    return pd.DataFrame.from_dict(data).iloc[::-1].reset_index(drop=True).reset_index()

def remove_overlapping_entity(l_found):
    """
    Remove entities that have the same start/ end values (overlapping entities)
    Ultimately the largest entity is kept (longest name).
    
    Input;
        l_found = List with overlapping entities/ phenotypes
    Output:
        List without the overlapping entities/ phenotypes
    
    Potential errors: An entity within another entity 
        but without a duplicate start or end will still cause errors!!
    
    """
    df = pd.DataFrame.from_records(l_found, columns=['match', 'start', 'end', 'desc'])
    
    s = df.match.str.len().sort_values().index # sort on str size of match
    df = df.reindex(s)
    df = df.drop_duplicates(subset='start', keep="last") # remove same start
    df = df.drop_duplicates(subset='end', keep="last") # remove same end
    return df.values.tolist()
    
def get_location_phenotype(txt, d_pheno):
    """
    Get the location of each phenotype (match) in the text and return
    instructions for colouring the entities
    
    Input:
        txt = sentence (string)
        d_pheno = list of phenotypes intercepted in a specific sentence
    
    Output:
        l_found = list with start / end position per phenotype in sentence
            a.k.a. regions to color!
        l_new_found = list with entity, start, stop and description (dict)
    """
    l_ent= [i['match'] for i in d_pheno]
    l_pheno = [i for i in d_pheno]
    l_found = []
    l_new_found = []
    for ent in l_ent:
        l_found.extend([[ent, m.start(), m.end()] for m in re.finditer(ent, txt)])
    #print(txt)
    #print(l_found[::-1])
    l_found = sorted(l_found, key=lambda x: x[1])
    #print(l_found)
    for pheno in l_found[::-1]: # add other details to pheno as well
        ix = l_ent.index(pheno[0])
        #print(l_ent.index(pheno[0]))
        l = pheno
        l.append(l_pheno[ix])
        l_new_found.append(l)
    l_new_found = remove_overlapping_entity(l_new_found)
    l_new_found = sorted(l_new_found, key=lambda x: x[1])[::-1]
    #print(l_new_found)
    return l_new_found

def annotate_text(parsed_list, d_phenotype, section='Main text', stringent=True, l_patterns=[], l_batches=[], BATCHES=True):
    """
    Description:
    Annotate the entities found with HPO: 
    
    On mouse over the user will see the hpo, confidence score and assoc. patient
    
    Input:
        parsed_list = segmented content of the case study
        d_phenotype = dictionary with all intercepted phenotypes 
        section = file where screening is performed
        stringent = boolean indicating whether or not to perform a 
            stringent entity linking
        l_patterns = list of patterns to identify entities
        BATCHES = work with offset if there are batches
        
    Output:
        new_lines = annotated content
    """
    batch_size = 15 
    new_lines = []
    start_str = '<span style="color:red">'
    end_str = '</span>'
    
    # ENTITY LINKING
    d_patient_ids = enlink.identify_patient(parsed_list, stringent, l_patterns)
    d_phenotype = enlink.mass_flagging(parsed_list, d_phenotype, d_patient_ids)
    
    for ix, sent in enumerate(parsed_list): 
        txt = parsed_list[ix]
        init_txt = txt
        passing = False
        if ix in d_phenotype:
            l_found = get_location_phenotype(txt, d_phenotype[ix])
            for j in l_found:
                start_int = j[1]
                end_int = j[2]
                desc = j[3]
                if section != 'Main text': # Check for table or caption!
                    desc['section'] = section
                if 'negated' in desc.keys():
                    start_str = '<span style="color:red" title="%s" >'  % ('HPO: ' + desc['hp_id'] + '\nCONF: ' + str(desc['score']) + '\nPAT: ' + desc['pat_id'] + '\nNEG: ' + str(desc['negated']) + '\nREL: ' + str(desc['relevant']) + '\nSECTION: ' + desc['section']) # i['pat_id']
                else : 
                    start_str = '<span style="color:red" title="%s" >'  % ('HPO: ' + desc['hp_id'] + '\nCONF: ' + str(desc['score']) + '\nPAT: ' + desc['pat_id'])
                txt = txt[:start_int] + ' ' + start_str + txt[start_int:end_int]  + end_str + ' ' + txt[end_int:]
        new_lines.append(txt)
    return new_lines

## Validation (For example: ARS)

def evaluateDictionaryHPO(annotated_table, graph):
    """
    Create a dictionary where every patient is linked to a list of
    phenotypes (HPO-codes).
    
    Input:
        annotated_table = pandas Dataframe (generated table from case-study)
        graph = HPO directed acyclic graph in OBO format
    Output: 
        d_inffered = dictionary with phenotypes linked to each patient
    """
    url = '../phenopy_mod/.phenopy/data/hp.obo'

    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
    
    s1 = annotated_table.apply(lambda x: extractAnnotations(x, annotated_table.columns), axis=1)

    d_inferred = {}
    for ix, pat in enumerate(annotated_table[annotated_table.columns[0]]):
        d_inferred[pat] = s1[ix]
    d_inferred = inferParentHPO(d_inferred, graph, id_to_name, name_to_id)
    return d_inferred
    
def inferParentHPO(d_inferred, graph, id_to_name, name_to_id):
    """
    Infer parent features (ancestor nodes) - 
    to get a detailed description of the patient phenotypes
    
    Input:
        d_inferred = dictionary containing all the phenotypes that were found
        graph = OBO HPO Directed acyclic graph
        id_to_name = dictionary where every HPO-id is matched to a name
        name_to_id = dictionary where every name is matched with a HPO-id
    Output: 
        d_inffered = dictionary with phenotypes and parent phenotypes 
            linked to each patient
    """
    d_new_inferred = {}
    for pat in d_inferred.keys():
        inferred = d_inferred[pat]
        new_inferred = []
        for hpo_id in inferred:
            new_inferred.append(hpo_id)
            df_super = get_superclass(graph, id_to_name[hpo_id], name_to_id, id_to_name)
            new_inferred.extend(list(df_super['HPO id']))
        d_new_inferred[pat] = new_inferred
    return d_new_inferred

def extractAnnotations(row, columns):
    """
    Extract the phenotypes as found in the annotatedTable
    
    Output:
        inferred = list of found HPO's 
    """
    columns = columns[1:]
    binary_vector = row[1:]
    inferred = []
    for ix, val in enumerate(binary_vector):
        if val == 1:
            inferred.append(columns[ix])
    return inferred

def evaluateDictionaryPatients(d_inferred, df_valid, graph):
    """
    Compare the predicted phenotypes (d_inferred) with the 
    validation set.
    
    Manual Annotation vs Automatic Extraction
    
    Input: 
        d_inferred = dictionary containing phenotypic profile per patient (wide format)
        df_valid = pandas Dataframe with phenotypic profiles per patient (long format)
        graph = OBO HPO Directed acyclic graph
    Output:
        confusion matrix with TP, FN, FP and TN
    """
    y_test, y_pred = [], []
    TP, FN, FP, TN = 0, 0, 0, 0
    d_tracker = {}
    for pat in d_inferred.keys(): # Loop through all patients
        inferred = d_inferred[pat]

        gold = list(df_valid[df_valid['Patient'].str.contains(pat.split(' ')[0])]['HPO-id']) 
        inferred = is_phenotypic_abnormality(graph, list(inferred))
        gold = is_phenotypic_abnormality(graph, list(gold))
        #gold = update_deprecated_OBO(list(gold), d_trans)
        #inferred = update_deprecated_OBO(list(inferred), d_trans)
        #print(len(inferred))
        TP += len(np.intersect1d(list(set(gold)), list(set(inferred))))
        
        FN += len(set(gold) - set(inferred))
        FP +=  len(set(inferred) - set(gold))

        # update tracker
        for i in np.intersect1d(list(set(gold)), list(set(inferred))):
            d_tracker[i] = 'TP'
        for i in set(gold) - set(inferred):
            d_tracker[i] = 'FN'
        for i in set(inferred) - set(gold):
            d_tracker[i] = 'FP'
        
        for i in range(len(np.intersect1d(list(set(gold)), list(set(inferred))))):
            y_test.append(1)
            y_pred.append(1)
        for i in range(len(set(gold) - set(inferred))):
            y_test.append(1)
            y_pred.append(0)
        for i in range(len(set(inferred) - set(gold))):
            y_test.append(0)
            y_pred.append(1)
        #print(gold, inferred)
        print('FN: ', len(set(gold) - set(inferred)))
        print('FP: ', len(set(inferred) - set(gold)))
        print(pat, len(inferred), len(gold))
    return np.array([[TP, FN], [FP, TN]]), d_tracker

## GENERATING REPORT
def write_HTML_report(title, phenotyper):
    """
    
    
    title = title of article
    
    Generate a HTML report file, which provides a summary of the extraction process.
    
    Highlighting and warning the user for error propagation
    """
    htmFile = open("results/%s/report.html" % (title),"w")
    htmFile.write("""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
            "http://www.w3.org/TR/html4/loose.dtd">
    <html>
    <head>
        <title>Report - HPO extraction</title>
    </head>
    <body>
        <h1>Report - HPO extraction</h1>
        <hr>
    """)
    
    d_checks = bookKeepingProgress(title)
    d_steps = {0: ['Main Text Extraction', 'Table Extraction', 'Caption Extraction', 'Table Legend Extraction', 'Figure Extraction', 'Supplementary Extraction'], 1:[], 2:['Phenotyping Tables'], 3:['Extensive Annotation', 'Phenotypic Profiling'] } # bookkeeping which rows consists of which steps
    l_headers = ['<h2>1. Raw</h2>', '<h2>2. First screening</h2>', '<h2>3. Phenotypes</h2>', 
              '<h2>4. Entity linked annotations</h2>']
    
    string = ["Phenotyper: %s" % (phenotyper)]
    
    for i in range(len(d_steps)):
        string.append(l_headers[i])
        for component in d_steps[i]:
            if d_checks[component]:
                string.append('%s: <b><span style="color:lime">%s</span></b>' % (component, 'Succesfull!'))
            else :
                string.append('%s: <b><span style="color:red">%s</span></b>' % (component, 'Not Found!'))
    
    #string = ["Phenotyper: %s" % (phenotyper), '<h2>1. Raw</h2>', '' % d_checks['tb'], 
    #          ]

    for s in string:
        htmFile.write( "<p> %s</p>" %s)

    htmFile.write("""
    </body>
    </html>""")

    htmFile.close()
    return


def bookKeepingProgress(title):
    """
    Input:
        title = title of article
        phenotyper = tool used to extract phenotypes from text
        
    """ 
    global d_checks
    
    
    # 1. check if files in directory
    # 2. screen files and establish whether the extraction was succesful
    # 3. check if main text - at least has expected structure - or features at least 1 individual / phenotype
    
    # 0_raw
    # check_table_extract()
    # check_caption_extract()
    # check_suppl_extract()
    # check_figure_extract()
    
    # 1_extractions
    # d_stored = { key : [path, file] }
    d_stored = {
        'Main Text Extraction' : ["results/%s/%s/" % (title, '0_raw'), 'Main_text'],
        'Table Extraction' : ["results/%s/%s/" % (title, '0_raw/tables'), 'Table'],
        'Table Legend Extraction' : ["results/%s/%s/" % (title, '0_raw'), 'legends'],
        'Caption Extraction' : ["results/%s/%s/" % (title, '0_raw'), 'captions.txt'],
        'Supplementary Extraction' : ["results/%s/%s/" % (title, '0_raw/supplement'), '.'],
        'Figure Extraction' : ["results/%s/%s/" % (title, '0_raw/figures'), '.'],
        'Phenotyping Tables' : ["results/%s/%s/" % (title, '2_phenotypes') , 'Table'],
        'Extensive Annotation' : ["results/%s/%s/" % (title, '3_annotations') , 'Overview_Intercepted'],
        'Phenotypic Profiling' : ["results/%s/%s/" % (title, '3_annotations') , 'PhenoProfiles'],
               }
    for key in d_stored.keys():
        d_checks[key] = file_exists(d_stored[key][0], d_stored[key][1])
    # 2_phenotypes
    #check_pheno_tables()
    # 3_annotations
    return d_checks
    
def file_exists(path, keyword):
    """
    Input:
        path = path to file
        keyword = flag to identify files of interest
    
    
    ToDo: Maybe perform a more extensive check.
    Where you actually evaluate the content produced in addition 
    to only checking if file is accessible
    
    """
    result_files = os.listdir(path)
    table_files = [s for s in result_files if (keyword in s and keyword in s)]
    if table_files != []:
        return True
    else : 
        return False