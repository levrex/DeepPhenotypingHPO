import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from bokeh.transform import factor_cmap
from bokeh.models import  CategoricalColorMapper, LinearColorMapper
import collections
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics # 
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import ast 
import re



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

from bokeh.io import output_file, show
from bokeh.models import Panel, Tabs
from bokeh.plotting import figure


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