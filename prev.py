from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Text
from bokeh.models import BoxZoomTool, HoverTool, PanTool, ResetTool, WheelZoomTool
from bokeh.models.markers import Circle
from bokeh.palettes import viridis
from bokeh.transform import factor_cmap

from multiprocessing import Pool

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

import textwrap

stemmer = PorterStemmer()
stopWords = set(stopwords.words('english'))
stopWords.add("article")

def remove_delegations(text_list):
    return [word for word in text_list if not word in delegations]

def remove_stopwords(text_list):
    return [word for word in text_list if not word.lower() in stopWords]
    
def stem_and_lower(text_list):
    return [stemmer.stem(word.lower()) for word in text_list]

def preprocess(text):
    tokenized = word_tokenize(text)
    tokenized_delegations_removed = remove_delegations(tokenized)
    tokenized_stopwords_removed = remove_stopwords(tokenized_delegations_removed)
    tokenized_stemmed_lowered = stem_and_lower(tokenized_stopwords_removed)
    
    return ' '.join(tokenized_stemmed_lowered)


def make_dtm(texts):
    length_before = len(texts)
    
    print('\tText preprocessing')
    with Pool(3) as p:
        texts_preprocessed = p.map(preprocess, texts)
        
    assert (length_before == len(texts_preprocessed)), 'After preprocessing there should be the same number of items as before'
    
    print('\tApply CountVectorizer')
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(texts_preprocessed)
    
    assert (length_before == dtm.shape[0]), 'After preprocessing there should be the same number of items as before'
    print(f'\tDTM shape: {dtm.shape}')
    
    return dtm, vectorizer

def do_k_means(k, comments_dtm):
    kmeans = KMeans(n_clusters=k, random_state=13, n_jobs = 3, verbose = 2)
    kmeans.fit(comments_dtm)
    predictions = kmeans.predict(comments_dtm)
    
    return predictions

def do_pca(comments_dtm):
    pca = PCA(n_components = 2, random_state=13)
    coordinates = pca.fit_transform(comments_dtm.todense())
    
    components = pca.components_
    
    return coordinates, components
    

def K_means_pca_plot(k, texts):
    print('Make DTM')
    dtm, vectorizer = make_dtm(texts)
    print('Do KMeans')
    predictions = do_k_means(k, dtm)
    print('Do PCA')
    coordinates, components = do_pca(dtm)
    
    print('Make_plot')
    output_file(f'PCA_plot_{k}.html')
    
    xs = coordinates[:,0]
    x_scale = 1.0/(xs.max() - xs.min())
#     print(x_scale)
    ys = coordinates[:,1]
    y_scale = 1.0/(ys.max() - ys.min())
    
    vocab = vectorizer.vocabulary_
    inverse_vocab = {value:key for key,value in vocab.items()}

    wordlist = [inverse_vocab[index] for index in sorted(list(inverse_vocab.keys()))]
    
    data = {'PC1': x_scale*xs, 'PC2':  y_scale*ys , 'cluster': [str(x) for x in predictions], 'text': texts}
    source = ColumnDataSource(data)

    data2 = {'C1': components[0,:], 'C2': components[1,:], 'words':wordlist}
    source2 = ColumnDataSource(data2)
    
    color_factors = [str(x) for x in sorted(list(set(predictions)))]
    cmap = factor_cmap(field_name='cluster', palette =  viridis(k), factors = color_factors, nan_color='gray')
    
    plot = Plot(plot_width=600, plot_height=600, tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool()])

    glyph = Circle(x="PC1", y="PC2", fill_color=cmap)
    plot.add_glyph(source, glyph)
    
    g1_r = plot.add_glyph(source_or_glyph=source, glyph=glyph)
    g1_hover = HoverTool(renderers=[g1_r],
                             tooltips=[('cluster', '@cluster'), ('text', '@text')])
    plot.add_tools(g1_hover)

    
    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')
    plot.xaxis.axis_label = 'PC1'

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')
    plot.yaxis.axis_label = 'PC2'

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
    
    ## PCA components
    word_glyph = Text(x="C1", y="C2", text="words", text_color="red")
    plot.add_glyph(source2, word_glyph)

    show(plot)

K_means_pca_plot(15, comments_df.comment.values)