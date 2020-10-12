import sys
sys.path.append('PhenoTool/')

import ast 
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, Panel, Tabs
from bokeh.plotting import figure, show, output_notebook
from bokeh.transform import factor_cmap
from bokeh.models import  CategoricalColorMapper, LinearColorMapper
from bokeh.io import output_file, show
from clinphen_src import get_phenotypes
from clinphen_src import src_dir
import collections
from io import BytesIO
import matplotlib.pyplot as plt
from math import log2
import numpy as np
from numpy.linalg import norm 
import networkx as nx
import os
import pandas as pd
from PIL import Image
import re
from sklearn.cluster import KMeans
from sklearn import metrics # 
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from txt2hpo.extract import Extractor
import unicodedata
import urllib.request
import requests



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

def makeTSNE_Cluster(values, l_id, l_lbl, title, clusters, pal, perp=30, seed=1234):
    
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

    color_mapper = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal)

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

def makeTSNE_Cluster2(values, l_id, l_clust, l_lbl, title, clusters, pal, perp=30, seed=1234):
    """
    l_id = patient id
    l_clust = cluster label
    l_lbl = category label
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
    color_mapper2 = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal[1])
    
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    tsne_df['pt'] = l_id
    tsne_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    tsne_df['cluster'] = l_clust
    
    tsne_df2 = pd.DataFrame({'x' : kmeans.cluster_centers_[:, 0], 'y': kmeans.cluster_centers_[:, 1]})
    tsne_df2['pt'] = ["Cluster center %i" % (i) for i in range(len(kmeans.cluster_centers_))]
    #tsne_df['pt'] = l_id
    #tsne_df2['label'] = pred_y
                         

    #mapper = factor_cmap('cat', palette=Spectral6[:4], factors=tsne_df['cat'])
    # plotting. the corresponding word appears when you hover on the data point.
    p1.scatter(x='x', y='y', source=tsne_df, legend_field="cluster", size=10,  color={'field': 'cluster', 'transform': color_mapper}) # fill_color=mapper
    p1.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
    
    hover = p1.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster"}
    #hover = p2.select(dict(type=HoverTool)) # or p1
    #hover.tooltips={"pt": "@pt", "lbl": "@lbl", "cluster" : "@cluster"}
    
    tab1 = Panel(child=p1, title="cluster")
    
    p2 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    p2.scatter(x='x', y='y', source=tsne_df, legend_field="label", size=10,  color={'field': 'label', 'transform': color_mapper2}) # fill_color=mapper
    p2.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
    
    hover2 = p2.select(dict(type=HoverTool)) # or p1
    hover2.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster"}
    
    tab2 = Panel(child=p2, title="lbl")
    
    tabs = Tabs(tabs=[ tab1, tab2 ])
    
    
    
    
    
    bp.output_file('TSNE/Kmeans_phenoMap_tsne_%s.html' % (title), mode='inline')

    bp.save(tabs)
    print('\nTSNE figure saved under location: TSNE/Kmeans_phenoMap_tsne_%s.html' % (title))
    return 

def makeTSNE_Cluster3(values, l_id, l_clust, l_lbl, l_origin, title, clusters, pal, perp=30, seed=1234, center=False):
    """
    l_id = patient id
    l_clust = cluster label
    l_lbl = category label
    l_origin = label that refers to initial data
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
    color_mapper2 = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal[1])
    color_mapper3 = CategoricalColorMapper(factors=list(set(l_origin)), palette=pal[2])
    
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    tsne_df['pt'] = l_id
    tsne_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    tsne_df['cluster'] = l_clust
    tsne_df['origin'] = l_origin
    
    tsne_df2 = pd.DataFrame({'x' : kmeans.cluster_centers_[:, 0], 'y': kmeans.cluster_centers_[:, 1]})
    tsne_df2['pt'] = ["Cluster center %i" % (i) for i in range(len(kmeans.cluster_centers_))]
    #tsne_df['pt'] = l_id
    #tsne_df2['label'] = pred_y
                         

    #mapper = factor_cmap('cat', palette=Spectral6[:4], factors=tsne_df['cat'])
    # plotting. the corresponding word appears when you hover on the data point.
    p1.scatter(x='x', y='y', source=tsne_df, legend_field="cluster", size=10,  color={'field': 'cluster', 'transform': color_mapper}) # fill_color=mapper
    
    
    hover = p1.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster", "origin": "@origin"}
    #hover = p2.select(dict(type=HoverTool)) # or p1
    #hover.tooltips={"pt": "@pt", "lbl": "@lbl", "cluster" : "@cluster"}
    
    tab1 = Panel(child=p1, title="cluster")
    
    p2 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    p2.scatter(x='x', y='y', source=tsne_df, legend_field="label", size=10,  color={'field': 'label', 'transform': color_mapper2}) # fill_color=mapper
    
    
    hover2 = p2.select(dict(type=HoverTool)) # or p1
    hover2.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster", "origin": "@origin"}
    
    tab2 = Panel(child=p2, title="lbl")
    
    p3 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    p3.scatter(x='x', y='y', source=tsne_df, legend_field="origin", size=10,  color={'field': 'origin', 'transform': color_mapper3}) # fill_color=mapper
    
    
    if center:
        p3.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
        p2.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
        p1.scatter(x='x', y='y', source=tsne_df2, size=15,  marker="diamond", color='pink') # fill_color=mapper
    
    hover3 = p3.select(dict(type=HoverTool)) # or p1
    hover3.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster", "origin": "@origin"}
    
    tab3 = Panel(child=p3, title="origin")
    
    tabs = Tabs(tabs=[ tab1, tab2, tab3 ])
    
    
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

def makePCA2(values, l_id, l_clust, l_lbl, l_origin, title, clusters, pal, perp=30, seed=1234, center=False):
    """
    l_id = patient id
    l_clust = cluster label
    l_lbl = category label
    l_origin = label that refers to initial data
    """
    p1 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    
    
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

    # dimensionality reduction. converting the vectors to 2d vectors
    pca_model = PCA(n_components=2, random_state=seed)
    pca_2d = pca_model.fit_transform(values)
    
    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    pred_y = kmeans.fit_predict(pca_2d) #
    
    print('Explained PCA:\tPC1=', pca_model.explained_variance_ratio_[0], '\tPC2=',pca_model.explained_variance_ratio_[1])
    color_mapper = CategoricalColorMapper(factors=list(set(l_clust)), palette=pal[0])
    color_mapper2 = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal[1])
    color_mapper3 = CategoricalColorMapper(factors=list(set(l_origin)), palette=pal[2])
    
    # putting everything in a dataframe
    pca_df = pd.DataFrame(pca_2d, columns=['x', 'y'])
    pca_df['pt'] = l_id
    pca_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    pca_df['cluster'] = l_clust
    pca_df['origin'] = l_origin
    
    pca_df2 = pd.DataFrame({'x' : kmeans.cluster_centers_[:, 0], 'y': kmeans.cluster_centers_[:, 1]})
    pca_df2['pt'] = ["Cluster center %i" % (i) for i in range(len(kmeans.cluster_centers_))]

    p1.scatter(x='x', y='y', source=pca_df, legend_field="cluster", size=10,  color={'field': 'cluster', 'transform': color_mapper}) 
    
    hover = p1.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster", "origin": "@origin"}
    
    tab1 = Panel(child=p1, title="cluster")
    
    p2 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    p2.scatter(x='x', y='y', source=pca_df, legend_field="label", size=10,  color={'field': 'label', 'transform': color_mapper2})
        
    hover2 = p2.select(dict(type=HoverTool)) # or p1
    hover2.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster", "origin": "@origin"}
    
    tab2 = Panel(child=p2, title="lbl")
    
    p3 = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    if center: 
        p1.scatter(x='x', y='y', source=pca_df2, size=15,  marker="diamond", color='pink')
        p2.scatter(x='x', y='y', source=pca_df2, size=15,  marker="diamond", color='pink')
        p3.scatter(x='x', y='y', source=pca_df2, size=15,  marker="diamond", color='pink') 
        
    p3.scatter(x='x', y='y', source=pca_df, legend_field="origin", size=10,  color={'field': 'origin', 'transform': color_mapper3}) 
    
    
    hover3 = p3.select(dict(type=HoverTool)) # or p1
    hover3.tooltips={"pt": "@pt", "lbl": "@label", "cluster" : "@cluster", "origin": "@origin"}
    
    tab3 = Panel(child=p3, title="origin")
    
    tabs = Tabs(tabs=[ tab1, tab2, tab3 ])
    
    
    bp.output_file('PCA/phenoMap_pca_%s.html' % (title), mode='inline')

    bp.save(tabs)
    print('\nPCA figure saved under location: PCA/phenoMap_pca_%s.html' % (title))
    return 


def makePCA(values, l_id, l_lbl, title, pal, radius=0.05, seed=1234):
    # dimensionality reduction. converting the vectors to 2d vectors
    pca_model = PCA(n_components=2, random_state=seed) # , verbose=1, random_state=0
    pca_2d = pca_model.fit_transform(values)
    print('Explained PCA:\tPC1=', pca_model.explained_variance_ratio_[0], '\tPC2=',pca_model.explained_variance_ratio_[1])
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

def elbowMethod(X_trans, method='kmeans', n=20):
    """
    Define optimal number of clusters with elbow method, optimized for 
    Within cluster sum of errors(wcss).
    
    Input:
        X_trans = Distance matrix based on HPO binary data (X)
        method = clustering method
        n = search space (number of clusters to consider)
    
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


def visualizeMostOccuringFeatures(df, cluster, cutoff=0.2):
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
    series = df[df['cluster']==cluster].loc[:, ~df.columns.isin(['cluster'])].sum().sort_values(ascending=True) #[:10]
    series = series.rename(lambda x: x[:25] + '...' if len(x)> 25 else x) # long labels are automatically shortened
    prob = series / N
    mask = prob > cutoff
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob.plot(kind='barh',figsize=(9,9), title='Most occuring phenotypic features in Cluster %s (n=%s)' % (cluster, str(N)))
    plt.show()
    return

def visualizePhenotypicSpecificity(df, weights, cluster, cutoff=0.2, pat=2):
    """
    Input:
        df = dataframe with phenotypic profile per patient & assigned clusters
        cluster = cluster of interest
        weights = weights for every phenotype (list)
        cutoff = specify the required specificity
    
            Default cut-off = 0.2
                feature should at least be prevalent in 20% of the patients of this cluster
        pat = minimal number of patients       
    
    Description:
        Visualize the top-% most specific features of a cluster/group. The specificity is 
        calculated by dividing the nr of patient with said phenotypes by the nr of assoc genes. 
    
    Output:
        barplot highlighting phenotypic specificity of each feature
    """
    N = len(df[df['cluster']==cluster])
    l_cols = [col for col in df.columns if col != 'cluster']
    #print(len(l_cols))
    #print(df['cluster'])
    df = df[df['cluster']==cluster]
    df = pd.concat([df[l_cols].replace(0, np.nan).dropna(axis=1, thresh=pat), df['cluster']], axis=1)
    #print(df['cluster'])
    #df['cluster'].replace(np.nan, 0, inplace=True)
    weights = [weights[l_cols.index(i)] for ix, i in enumerate(list([col for col in df.columns if col != 'cluster']))]
    series = df.loc[:, ~df.columns.isin(['cluster'])].sum() #[:10]
    series = series.rename(lambda x: x[:25] + '...' if len(x)> 25 else x) # long labels are automatically shortened
    prob = np.divide(series, weights)
    prob = prob.sort_values(ascending=True)
    
    mask = prob > cutoff
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    print('Raw probabilities:\n', prob)
    
    prob.plot(kind='barh',figsize=(9,9), title='Most occuring phenotypic features in Cluster %s (n=%s)' % (cluster, str(N)))
    plt.show()
    return

def calculatePhenotypicSpecificity(l_cols, col2hpo, df_hpo, penalty=100):
    """
    Calculate patient similarity with gene occurrence
    
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
        
    special_classes = ['accordion-tabbed__tab-mobile', 'dropBlock__body', 'inline-table']
    for cl in special_classes:
        for div in soup.find_all('div', {'class': cl}):  #
            div.replaceWith('')
            
    special_classes = ['dpauthors', 'dporcid', 'dptop', 'dptitle', 'dpfn']        
    for cl in special_classes:
        for div in soup.find_all('div', {'class': cl}):  #
            div.decompose()

    for element in ['ul', 'i', 'span', 'li', 'a', "script", "style", "meta", "link", "sup", "select", "option", "figcaption"]: # a 
        for div in soup.find_all(element):  # , {'class':'Google-Scholar'}
            #print(div)
            div.decompose()

    for s in soup.select('div'):
        s.get_attribute_list = ''       
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
    if link != None: 
        score  = 1*('article' in link) 
        if score > 0 :
            article = True
    return article

def predict_supplement(classes, txt):
    """
    Input:
        classes = list with class names
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
    for cl in classes:
        score  = 1*('supplementary' in cl) + 1*('appendix' in cl)
        if score > max_score :
            max_ix = ix
            max_score= score
            suppl = True
        ix += 1
    return classes[max_ix], suppl

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
    response = opener.open(request,timeout=100)
    data = response.read()
    
    try : 
        file = open("results/%s/0_raw/supplement/%s" % (title, file_title), "wb")
        file.write(data)
        file.close()
    except: 
        print('The file %s is not in a regular table format' % (file_title))
    return

def extract_supplement(soup, title, domain, save_supplement=False): 
    """
    Extract supplementary files from article
    
    Input: 
        soup = extraction from case study
        save_supplement = save the actual supplementary files
    """
    dir_figs = 'results/%s/0_raw/supplement' % (title)
    txt = ''
    for ix, div in enumerate(soup.find_all('body')):
        classes = []
        text = []
        links = []
        for element in div.findAll(class_=True):
            classes.extend(element["class"])
            text.extend([element.text for i in range(len(element["class"]))])
        if classes == []:
            break
        cl, suppl = predict_supplement(classes, text)
        if suppl: # check if supplement is found
            l_supplement = div.findAll('div', attrs={'class': cl}) # section-paragraph
        else :
            l_supplement = []
        if l_supplement == []:
            break
        for i in range(len(l_supplement)):
            links.extend(l_supplement[i].find_all('a'))
        for link in links:
            try:
                if 'href' in link.attrs:
                    file_link = link.get('href')
                elif 'srcset' in link.attrs:
                    file_link = link.get('srcset')
                if file_link[:2] == '//': # 'https://' not in 
                    file_link = 'https:' + figure_link
                elif file_link[:1] == '/':
                    file_link = domain[:-1] + file_link
                elif 'http' not in file_link:
                    print(file_link, 'not a valid link')
                    break
            except: 
                print(str(file_link), 'not a valid link')
                break
            file_title = file_link.rpartition('/')[2]
            file_title = re.sub(r"[^\.0-9a-zA-Z]+", "", file_title)
            print(file_link)
            if save_supplement:
                import_file(file_link, title, file_title)
    return 

## Extract figures

def remove_tag_caption(soup):
    """  
    Remove HTML tags and hyperlinks from figure caption
    """
    for element in ['ul', 'i', 'span', 'li', 'a', "script", "style", "meta", "link", "sup", "select", "option", "em"]: # a  # "script", "style", "meta", "link", "sup", "select", "option", "figcaption"
        for div in soup.find_all(element):  # , {'class':'Google-Scholar'}
            #print(div)
            div.decompose()
    #print(soup)
    return soup

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

    request=urllib.request.Request(figure_link,None, headers) #The assembled request
    response = opener.open(request,timeout=100)
    data = response.read()
    
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
    request=urllib.request.Request(table_link,None, headers) #The assembled request
    response = opener.open(request,timeout=100)
    data = response.read()
    
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
    for cl in classes:
        score  = 1*('caption' in cl) + 1*('description' in cl) # 1*('tbl' in cl) + 1*('table' in cl) + 1*('wrap' in cl) + 
        #scores.append(score)
        if score > max_score or (score == max_score and len(txt[ix]) > max_len):
            max_ix = ix
            max_len = len(txt[ix])
            max_score= score
        ix += 1
    #print(classes[max_ix])
    return classes[max_ix]


def extract_figures(soup, title, domain, save_figures=False): 
    """
    Extract figures from article
    
    Input: 
        soup = extraction from case study
        title = title of the article
        domain = domain/ root of the website
        save_figures = only the captions are extracted unless save_figures=True
    """
    dir_figs = 'results/%s/0_raw/figures' % (title)
    
    interpret_figure(soup, title, domain, save_figures)
    interpret_img(soup, title, domain, save_figures) 
    return 

def interpret_img(soup, title, domain, save_figures=False):
    """
    Captions can often not be retrieved from IMG directly
    
    # does image sometimes have a caption though??

    """
    links = []
    for link in soup.findAll('img'): # , attrs={'href': re.compile("^(?://|http\://|/)")}
        #print(link)
        if predict_article_figure_img(link):
            links.append(link)
    extract_images_with_links(links, title, domain, save_figures)
    return


def interpret_figure(soup, title, domain, save_figures=False):
    txt = ''
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

        cl = predict_caption_figure(classes, text)
        #print(cl)
        captions = div.findAll('div', attrs={'class': cl}) # section-paragraph

        if captions == []:
            continue

        links = div.findAll('a', attrs={'href': re.compile("^(?://|http\://|/)")}) # maybe http as well
        links.extend(div.findAll('source', attrs={'srcset': re.compile("^(?://|http\://|/)")}))
        #print(links)
        extract_images_with_links(links, title, domain, save_figures)
        captions = remove_tag_caption(captions[0]) # hopefully this works everytime
        clean = re.compile('<.*?>')
        cleansed = re.sub(clean, '', str(captions))
        txt += cleansed
    if txt != '':
        save_captions(txt, title, 'captions_figures.txt')
    return 

def extract_images_with_links(links, title, domain, save_figures=False):
    """
    Format the paths to images to actual working links!
    
    Input:
        links = candidate links for figures
        title = title of article
    
    """ 
    for link in links:
        #print(link.get('src'))
        
        try:
            if 'href' in link.attrs:
                figure_link = link.get('href')
            elif 'srcset' in link.attrs:
                figure_link = link.get('srcset')
            elif 'src' in link.attrs:
                figure_link = link.get('src')
                #print('Valid link: ', figure_link)
            if figure_link[:2] == '//': # 'https://' not in 
                figure_link = 'https:' + figure_link
            elif figure_link[:1] == '/':
                figure_link = domain[:-1] + figure_link
            #print('Valid link: ', figure_link)
        except: 
            print('not a valid link')
            continue # break
        fig_title = figure_link.rpartition('/')[2]
        fig_title = re.sub(r"[^\.0-9a-zA-Z]+", "", fig_title)
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
        captions = remove_tag_caption(captions[0]) # hopefully this works everytime
        clean = re.compile('<.*?>')
        cleansed = re.sub(clean, '', str(captions))
        txt += cleansed
    if txt != '':
        save_captions(txt, title, 'captions_tables.txt')
    return 

def extract_captions(soup, title): 
    txt = ''
    for ix, div in enumerate(soup.find_all('body')): #for ix, div in enumerate(soup.find_all('a')):
        classes = []
        text = []
        links = []
        for element in div.findAll(class_=True):
            classes.extend(element["class"])
            text.extend([element.text for i in range(len(element["class"]))])
        if classes == []:
            continue
        #print(text)
        cl = predict_caption(classes, text)
        #print(classes)
        #print(cl)
        captions = soup.findAll('div', attrs={'class': cl}) # section-paragraph
        if captions == []:
            continue
        clean = re.compile('<.*?>')
        #print(captions)
        index = 0
        for cap in captions:
            #print(cap)
            try:
                cleansed = cap.get_text()
            except : 
                cap = remove_tag_caption(cap) # hopefully this works everytime
                cleansed = re.sub(clean, '', str(cap))
            if cleansed not in [None, '']:
                txt += 'Entity%s\n' % (index) + cleansed + '\n'
                index += 1
        #print(txt)
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
                cap = remove_tag_caption(cap) # hopefully this works everytime
                cleansed = re.sub(clean, '', str(cap))
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
        if score > max_score or (score == max_score and len(txt[ix]) > max_len):
            max_ix = ix
            max_len = len(txt[ix])
            max_score= score
            fig = True
        ix += 1
        
    return classes[max_ix], fig

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
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return text

def regex_cleaning(soup, title_classes=[]):
    """ 
    Clean Html text with regular expression rules. Headers are 
    preserved to ensure readability. Custom header classes can be 
    provided by the user.
    
    Input:
        soup = extracted text from html
        title_classes = list with custom header flags
    """
    # preserve headers
    new_soup = re.sub(r"\<h([1-6])>", r"@h\1@", soup) 
    new_soup = re.sub(r"\</h([1-6])>", r"@/h\1@", new_soup)
    
    if title_classes != []: ## preserve special div title classes
        new_soup = re.sub(r"(\<div class=\"(?:%s)\"[^>]*>[^<]*)(</div>)" % ('|'.join(title_classes)), r"\1@/h5@", new_soup)
        new_soup = re.sub(r"\<div class=\"(?:%s)\"[^>]*>" % ('|'.join(title_classes)), r"@h5@", new_soup)

    new_soup = re.sub(r"<br>", "\n", new_soup)
    new_soup = re.sub(r"<hr/>", "\n", new_soup)
    new_soup = re.sub(r"\<em\>|\</em\>", "", new_soup) # remove em tags ? (overbodig?)
    new_soup = re.sub(r"\<[^\>]+\>", "", new_soup) #remove All html tags
    new_soup = re.sub(r"\s{3,}", r'<br>', new_soup) # change excessive spaces into a single newline
    new_soup = re.sub(r"\.([A-Z])", r'. \1', new_soup) # add whitespace where a new sentence is started
    new_soup = re.sub(r"\\n", r"<br>", new_soup) # format newlines to <br>
    #
    
    # restore preserved headers
    new_soup = re.sub(r"@h([1-6])@", r"<h\1>", new_soup)
    new_soup = re.sub(r"@/h([1-6])@", r"</h\1>", new_soup)
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
                if type(d_phenotype) == list:
                    d_phenotype[frame][frame_ix] = val + first_intercept.count(ix)
                elif type(d_phenotype) == dict:
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
    
    d_ncr = {
    'HPO ID' : [i['hp_id'] for i in ast.literal_eval(response.text)['matches']],
    'names' : [i['names'] for i in ast.literal_eval(response.text)['matches']],  
    'start' : [i['start'] for i in ast.literal_eval(response.text)['matches']],    
    'end' : [i['end'] for i in ast.literal_eval(response.text)['matches']],   
    'score' : [i['score'] for i in ast.literal_eval(response.text)['matches']],   
    }
    df_hpo =pd.DataFrame.from_dict(d_ncr)
    
    
    
    return df_hpo #, first_intercept, lines

def ncr_str_chunk(lines):
    """
    Utilize the Neural Concept Recognizer on a large text by chunking
    
    Return df with HPO and exact location
    """ 
    first_intercept = {}
    for ix, line in enumerate(lines): 
        params = (
            ('text', line),
            )
        response = requests.get('https://ncr.ccm.sickkids.ca/curr/annotate/', params=params)
        if response.status_code == requests.codes.ok:
            d_val = ast.literal_eval(response.text)['matches']
            if d_val != {}: 
                first_intercept[ix] = d_val
        #print(ix)
    return first_intercept

def scispacy_str(nlp, umls_to_hpo, lines):
    """ 
    nlp = Natural Language Processing pipeline
    """
    l_hpo = []
    for ix, line in enumerate(lines): 
        ents = nlp(line.lower()).ents # .text
        l_hpo.extend(inferHPO(ents, umls_to_hpo))
    return l_hpo

def inferHPO(row, umls_to_hpo):
    """ 
    row = all found entities (in spacy Span format)
    
    Description:
        Infer hpo codes based on the found entities.
    """
    hpo_list = []
    for entity in row:
        for umls_ent in entity._.kb_ents:
            try :
                hpo_list.append(umls_to_hpo[hpo.kb.cui_to_entity[umls_ent[0]].concept_id])
            except :
                print('not found')
    return hpo_list

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
    #print(table)
    trs = table.find_all('tr')
    #print(trs)
    headerow = rowgetDataText(trs[0], 'th')
    if headerow: # if there is a header row include first
        rows.append(headerow)
        trs = trs[1:]
    for tr in trs: # for every table row
        #print(tr)
        #tr = tr.replace('\n','').replace('\r','')
        rows.append(rowgetDataText(tr, 'td') ) # data row   
    #print(rows)
    caption = (len(rows) == 1)
    return rows, caption

def parseTable(table, remove_inc=True):
    """
    Convert Table to pandas Dataframe
    
    Input:
        table = html table from article
        remove_inc = remove rows with incosistent length
            (be careful: these can be helpful to categorize table)
    """
    #print('y')
    list_table, caption = tableDataText(table)
    if caption == True:
        return list_table, caption
    dftable = pd.DataFrame(list_table[1:], columns=list_table[0])
    if remove_inc:
        med = np.median(dftable.isnull().sum(axis=1).values)
        dftable = dftable.dropna(thresh=len(dftable.columns)-med)
    return dftable, caption 

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
    Save tables found in html file as csv files & collect associated captions.
    
    Input:
        title = title of article
        tables = list of html tables
        remove_inc = remove rows with incosistent length (be careful: these can improve readability, or perhaps include valuable information)
    """
    index = 0
    open("results/%s/0_raw/captions_tables.txt" % (title), "w").close()
    file = open("results/%s/0_raw/captions_tables.txt" % (title), "a")
    caption = False
    for table in tables:
        try : 
            raw_table, caption = parseTable(table, remove_inc)
        except:
            try :
                raw_table = readTableDirectly(table)
                raw_table.to_csv("results/%s/0_raw/tables/Table_%s_%s.csv" % (title, str(index), title), sep='|', index=False, encoding='utf-8-sig',)
                continue
            except:
                index += 1
                continue
        if caption:
            print('Caption Found!')
            file.write(regex_cleaning(str(raw_table[0][1]))+ '\n')
        else :
            raw_table = raw_table.astype('str')
            raw_table.to_csv("results/%s/0_raw/tables/Table_%s_%s.csv" % (title, str(index), title), sep='|', index=False)
        index += 1
    file.close()
    return

def readTableDirectly(table):
    dfs = pd.read_html(str(table))[0]
    dfs.columns = dfs.columns.to_flat_index()
    return dfs
    
## Functions Scanning Tables

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
    l_yes = ['y', 'true', 't', 'yes', '1', 'present', 'p', 'pres'] 
    l_no = ['n', 'false', 'f', 'no', '0', 'absent', 'a', 'abs']
    l_unknown = ['u', 'unknown', 'na', 'nan', ''] # , '
    l_not = []
    for col in table.columns:
        l_val = table[col].values
        for i in l_val:
            val = i.lower()
            val = re.sub("[\(\[].*?[\)\]]", "", val)
            val = val.strip()
            if (val not in l_yes and val not in l_no and val not in l_unknown):
                l_not.append(col)
                break
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
        if phenotyper == 'clinphen':
            items = clinphen_str(simpleCleaning(col),'data')
            df_hpo = pd.DataFrame([n.split('\t') for n in items.split('\n')])
            df_hpo.columns = df_hpo.iloc[0]
            df_hpo = df_hpo.reindex(df_hpo.index.drop(0))
            d_col[col] = list(df_hpo['HPO ID'])
        elif phenotyper == 'txt2hpo':
            d_col[col] = txt2hpo_str(simpleCleaning(col))
        elif phenotyper == 'ncr':
            df_hpo = ncr_str(simpleCleaning(col)) # misschien iets soortgelijks als hpo -> waarbij je ook locatie hebt
            d_col[col] =list(df_hpo['HPO ID'])
    return d_col

def col_hpo(row, d_col):
    l_yes = ['y', 'true', 't', 'yes', '1', 'present', 'p', 'pres'] 
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
    l_hpo = [] 
    #print(list(row.values))
    row_content = ' '.join(list(row.values))
    if phenotyper == 'clinphen':
        items = clinphen_str(row_content,'data')
        df_hpo = pd.DataFrame([n.split('\t') for n in items.split('\n')])
        df_hpo.columns = df_hpo.iloc[0]
        df_hpo = df_hpo.reindex(df_hpo.index.drop(0))
        l_hpo = list(df_hpo['HPO ID'])
    elif phenotyper == 'txt2hpo':
        l_hpo = txt2hpo_str(row_content)
    elif phenotyper == 'ncr':
        df_hpo = ncr_str(col) # misschien iets soortgelijks als hpo -> waarbij je ook locatie hebt
        d_col[col] =list(df_hpo['HPO ID'])
    return l_hpo

def scan_table(table, phenotyper='clinphen'): ## Add function for scanning rows -> text. Then add this to function list
    l_qualify = find_binary_columns(table)
    print('Binary columns:', l_qualify)
    d_col = link_col_hpo(l_qualify, phenotyper)
    #print(d_col)
    table['row_hpo'] = table.apply(lambda x : row_hpo(x, phenotyper), axis=1)
    table['col_hpo'] = table.apply(lambda x : col_hpo(x, d_col), axis=1)
    return table

## 3. Phenotyping

def generate_phenotable(first_intercept):
    """
    Collect information of intercepted phenotypes in a
    convenient pandas dataframe
    
    Input:
        first_intercept = dictionary with all intercepted phenotypes
    
    """
    df = pd.DataFrame()

    for i, key in enumerate(first_intercept): 
        for item in first_intercept[key]:
            item['index'] = i
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
                 1*('patid' in col_name) + 1*('casenr' in col_name) + 1*('caseid' in col_name) + 1*(len(np.unique(l_values))==len(l_values)) + 1*('individual' in col_name) + 1*('proband' in col_name)
        if score > max_score:
            max_ix = ix
            max_score= score
            patient_identifier = True
        ix += 1
    return table.columns[max_ix],  patient_identifier

def get_patprof_tables(title, tab_files, d_pat):
    """
    Expand the patient phenotypic profiles with provided table
    """
    for tab in tab_files:
        table = pd.read_csv("results/%s/2_phenotypes/%s" % (title, tab), sep='|')
        table = table.fillna('') # very important to fill na prior to function
        #print(table.head())
        cl, pat_identifier = predict_identifier_patient(table)
        if pat_identifier:
            print('Patient Identifier column found: ', cl)
        else : 
            print("No column seems to represent a patient identifier")
        for ix, patient in enumerate(table[cl]):
            key = table[cl].iloc[ix]
            if key in d_pat.keys():
                d_pat[key].extend(ast.literal_eval(table['row_hpo'].iloc[ix]))
            else : 
                d_pat[key] = ast.literal_eval(table['row_hpo'].iloc[ix])
            #print(table['row_hpo'].iloc[ix])
            d_pat[key].extend(ast.literal_eval(table['col_hpo'].iloc[ix]))
            #print(table[['Case ID', 'row_hpo', 'col_hpo']].iloc[ix])
    return d_pat

def collectPhenoProfiles(title, phenotyper):
    """
    Search all files in the results directory & link
    to patients with Identifier.
    """
    result_files = os.listdir("results/%s/2_phenotypes/" % (title))
    #print(result_files)
    table_files = [s for s in result_files if ('Table' in s and phenotyper in s)]
    d_pat = {}
    
    # check if phenotypic abnormality
    # Consult Tables
    d_pat = get_patprof_tables(title, table_files, d_pat) # , d_pat
    if d_pat == {}:
        print('no phenotypes found in table')
    # Consult Main text
    # toDo: add function entity_linking -> regex rules
    # d_pat = get_patprof_main(title, main_file, d_pat)
    return d_pat

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

def annotate_text(parsed_list, d_phenotype):
    """
    Description:
    Annotate the entities found with HPO: 
    
    On mouse over the user will see the hpo, confidence score and assoc. patient
    
    """
    bin_ix = 0
    new_lines = []
    start_str = '<span style="color:red">'
    end_str = '</span>'
    
    for ix, sent in enumerate(parsed_list): 
        txt = parsed_list[ix]
        passing = False
        if ix in d_phenotype:
            d_sort = sorted(d_phenotype[ix], key = lambda j: j['end'], reverse=True)
            for i in d_sort:
                start_str = '<span style="color:red" title="%s" >'  % ('HPO: ' + i['hp_id'] + '\nCONF: ' + i['score'])
                start_int = i['start']
                end_int = i['end']
                txt = txt[:start_int] + ' ' + start_str + txt[start_int:end_int]  + end_str + ' ' + txt[end_int:] 
        new_lines.append(txt)
    return new_lines