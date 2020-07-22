from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from bokeh.transform import factor_cmap
from bokeh.models import  CategoricalColorMapper, LinearColorMapper
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from numpy.linalg import norm 
import matplotlib.pyplot as plt


def makeTSNE_Cluster(values, l_id, l_lbl, title, clusters, pal, perp=30, seed=1234):
    
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    
    
    #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=perp, random_state=seed)
    tsne_2d = tsne_model.fit_transform(values)
    
    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
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

def makePCA(values, l_id, l_lbl, title, pal, seed=1234):
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A phenoMap of %s patients" % (title),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)

    # dimensionality reduction. converting the vectors to 2d vectors
    pca_model = PCA(n_components=2, random_state=seed) # , verbose=1, random_state=0
    pca_2d = pca_model.fit_transform(values)
    print('Explained PCA:\tPC1=', pca_model.explained_variance_ratio_[0], '\tPC2=',pca_model.explained_variance_ratio_[1])
    color_mapper = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal)

    # putting everything in a dataframe
    pca_df = pd.DataFrame(pca_2d, columns=['x', 'y'])
    pca_df['pt'] = l_id
    pca_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=pca_df, legend_field="label", radius=0.05,  color={'field': 'label', 'transform': color_mapper}) # fill_color=mapper
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

    