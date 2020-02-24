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

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import textwrap

from typing import Iterable, Tuple, Union


stemmer = PorterStemmer()
stopWords = set(stopwords.words('english'))
stopWords.add("article")

class ClusterDimRedPLot(object):

    def __init__(self, k=10, dtm_type:str = 'count', max_cores = 3):
        if not dtm_type in ['count', 'tfidf']:
            raise ValueError(f"Unknown dtm type {dtm_type}") 
            
        if dtm_type == 'tfidf':
            raise NotImplementedError()
        
        self.k = k
        self.dtm_type = dtm_type
        self.max_cores = max_cores # TODO: implement smart max cores
        


    def _remove_stopwords(self, text_list:Iterable, stopWords:Iterable) -> list:
        """Removes all stopwords from a list of words
        
        :param text_list: list of words
        :type text_list: Iterable
        :param stopWords: list of stop words
        :type stopWords: Iterable
        :return: The original list of words except the stopwords
        :rtype: list
        """
        # TODO: more efficient preprocessing
        return [word for word in text_list if not word.lower() in stopWords]



    def _stem_and_lower(self, text_list:Iterable) -> list:
        """lowers and stemms all words in a list
        
        :param text_list: list of words
        :type text_list: Iterable
        :return: cleaned list of words
        :rtype: list
        """
        # TODO: more efficient preprocessing
        return [stemmer.stem(word.lower()) for word in text_list]

    def _preprocess(self, text:str) -> str:
        """Preprocesses a single test
        
        :param text: The text to be processed
        :type text: str
        :return: The processed text.
        :rtype: str
        """

        # TODO more efficient preprocessing
        tokenized = word_tokenize(text)
        tokenized_stopwords_removed = self._remove_stopwords(tokenized, stopWords)
        tokenized_stemmed_lowered = self._stem_and_lower(tokenized_stopwords_removed)
        
        return ' '.join(tokenized_stemmed_lowered)

    def _make_dtm(self, texts:Iterable) -> Tuple[np.ndarray,  CountVectorizer]:
        """Makes a DTM from a list of texts
        
        :param texts: A list of texts
        :type texts: Iterable
        :return: A list of texts as well as the vectorizer model
        :rtype: Tuple[np.ndarray,  CountVectorizer]
        """
        length_before = len(texts)
        
        print('\tText preprocessing')
        with Pool(self.max_cores) as p:
            texts_preprocessed = p.map(self._preprocess, texts)
            
        assert (length_before == len(texts_preprocessed)), 'After preprocessing there should be the same number of items as before'
        
        print('\tApply CountVectorizer')
        
        if self.dtm_type == 'count':
            vectorizer = CountVectorizer()
        elif self.dtm_type == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            # TODO: allow user to enter 
            raise NotImplementedError(print("unknown dtm type"))

        dtm = vectorizer.fit_transform(texts_preprocessed)
        
        assert (length_before == dtm.shape[0]), 'After preprocessing there should be the same number of items as before'
        print(f'\tDTM shape: {dtm.shape}')
        
        return dtm, vectorizer

    def _do_k_means(self, comments_dtm:np.ndarray) -> np.array:

        #TODO: decide whether comments dtm becomes self.
        kmeans = KMeans(n_clusters=self.k, random_state=13, n_jobs = self.max_cores, verbose = 2)
        kmeans.fit(comments_dtm)
        predictions = kmeans.predict(comments_dtm)
        
        return predictions

    def _do_pca(self, comments_dtm) -> Tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components = 2, random_state=13)
        coordinates = pca.fit_transform(comments_dtm.todense())
        
        components = pca.components_
        
        return coordinates, components


    def _analyze(self, texts:list):
        self.dtm, self.vectorizer = self._make_dtm(texts)
        self.predictions = self._do_k_means(self.dtm)
        self.coordinates, self.components = self._do_pca(self.dtm)

    def _make_plot(self, texts):
        print('Make_plot')
        #TODO: make output file an option
        output_file(f'PCA_plot_{self.k}.html')

        xs = self.coordinates[:,0]
        x_scale = 1.0/(xs.max() - xs.min())
    #     print(x_scale)
        ys = self.coordinates[:,1]
        y_scale = 1.0/(ys.max() - ys.min())

        vocab = self.vectorizer.vocabulary_
        inverse_vocab = {value:key for key,value in vocab.items()}

        wordlist = [inverse_vocab[index] for index in sorted(list(inverse_vocab.keys()))]
        
        # TODO: make texts fit for display with textwarp or display x words with the highest tf-idf
        data = {'PC1': x_scale*xs, 'PC2':  y_scale*ys , 'cluster': [str(x) for x in self.predictions], 'text': texts}
        source = ColumnDataSource(data)

        data2 = {'C1': self.components[0,:], 'C2': self.components[1,:], 'words':wordlist}
        source2 = ColumnDataSource(data2)
        
        color_factors = [str(x) for x in sorted(list(set(self.predictions)))]
        cmap = factor_cmap(field_name='cluster', palette =  viridis(self.k), factors = color_factors, nan_color='gray')
        
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


    def make_CDRP(self, texts:list):
        self._analyze(texts)
        self._make_plot(texts)


        
    # def K_means_pca_plot(self, texts):
    #     print('Make DTM')
    #     # TODO: determin whether dtm and vetorizer become attrinutes
    #     dtm, vectorizer = self._make_dtm(texts)
    #     print('Do KMeans')
    #     predictions = self._do_k_means(dtm)
    #     print('Do PCA')
    #     coordinates, components = self._do_pca(dtm)
        
    #     print('Make_plot')
    #     #TODO: make output file an option
    #     output_file(f'PCA_plot_{self.k}.html')
        
    #     xs = coordinates[:,0]
    #     x_scale = 1.0/(xs.max() - xs.min())
    # #     print(x_scale)
    #     ys = coordinates[:,1]
    #     y_scale = 1.0/(ys.max() - ys.min())
        
    #     vocab = vectorizer.vocabulary_
    #     inverse_vocab = {value:key for key,value in vocab.items()}

    #     wordlist = [inverse_vocab[index] for index in sorted(list(inverse_vocab.keys()))]
        
    #     data = {'PC1': x_scale*xs, 'PC2':  y_scale*ys , 'cluster': [str(x) for x in predictions], 'text': texts}
    #     source = ColumnDataSource(data)

    #     data2 = {'C1': components[0,:], 'C2': components[1,:], 'words':wordlist}
    #     source2 = ColumnDataSource(data2)
        
    #     color_factors = [str(x) for x in sorted(list(set(predictions)))]
    #     cmap = factor_cmap(field_name='cluster', palette =  viridis(self.k), factors = color_factors, nan_color='gray')
        
    #     plot = Plot(plot_width=600, plot_height=600, tools = [PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool()])

    #     glyph = Circle(x="PC1", y="PC2", fill_color=cmap)
    #     plot.add_glyph(source, glyph)
        
    #     g1_r = plot.add_glyph(source_or_glyph=source, glyph=glyph)
    #     g1_hover = HoverTool(renderers=[g1_r],
    #                             tooltips=[('cluster', '@cluster'), ('text', '@text')])
    #     plot.add_tools(g1_hover)

        
    #     xaxis = LinearAxis()
    #     plot.add_layout(xaxis, 'below')
    #     plot.xaxis.axis_label = 'PC1'

    #     yaxis = LinearAxis()
    #     plot.add_layout(yaxis, 'left')
    #     plot.yaxis.axis_label = 'PC2'

    #     plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    #     plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
        
    #     ## PCA components
    #     word_glyph = Text(x="C1", y="C2", text="words", text_color="red")
    #     plot.add_glyph(source2, word_glyph)

    #     show(plot)
