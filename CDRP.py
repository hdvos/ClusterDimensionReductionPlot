from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Text
from bokeh.models import BoxZoomTool, HoverTool, PanTool, ResetTool, WheelZoomTool
from bokeh.models.markers import Circle
from bokeh.palettes import viridis
from bokeh.transform import factor_cmap

from collections.abc import Iterable

from functools import partial

from multiprocessing import cpu_count, Pool

from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import textwrap

import time

from typing import Iterable, Tuple, Union

def parse_max_cores(n):
    max_available = cpu_count()

    if 0 < n <= max_available:
        return n
    elif n == 0:
        return 1
    elif -max_available < n < 0:
        return max_available + (n+1)
    else:
        raise ValueError("Invalid nr. Cores")
    

class ClusterDimRedPLot(object):

    def __init__(self, k=10, dtm_type:str = 'count', max_cores = 3, min_df = 3, stopwords = 'english', stemmer = None, 
    output_file = None, overwrite = False, text_display_type = 'snippet', 
    max_snippet_length = 200, snippet_width = 50, max_top_words = 5):

        if not dtm_type in ['count', 'tfidf']:
            raise ValueError(f"Unknown dtm type {dtm_type}") 
        
        self.k = k
        self.dtm_type = dtm_type
        self.max_cores = parse_max_cores(max_cores)
        # input(self.max_cores)
        self.min_df = 3

        if type(stopwords) is str:
            self.stopWords = sw.words('english')
        elif isinstance(stopwords, Iterable):
            self.stopWords = stopwords

        if stemmer:
            self.stemmer = stemmer
        else:
            self.stemmer = PorterStemmer()

        if output_file:
            self.output_file = output_file
        else:
            # TODO: make such that it does noet overwrite
            if overwrite:
                self.output_file = f'CDRP_plot_{self.k}.html'
            else:
                self.output_file = f'CDRP_plot_{self.k}_{str(int(time.time()))}.html'
            
        if text_display_type in ['snippet', 'top_words']:
            self.text_display_type = text_display_type
            self.max_snippet_length = max_snippet_length
            self.snippet_width = snippet_width
            self.max_top_words = max_top_words
        else:
            raise ValueError(f"{text_display_type} is no valid text display type.")


    def _remove_stopwords(self, text_list:Iterable) -> list:
        """Removes all stopwords from a list of words
        
        :param text_list: list of words
        :type text_list: Iterable
        :param stopWords: list of stop words
        :type stopWords: Iterable
        :return: The original list of words except the stopwords
        :rtype: list
        """
        # TODO: more efficient preprocessing
        return [word for word in text_list if not word.lower() in self.stopWords]



    def _stem_and_lower(self, text_list:Iterable) -> list:
        """lowers and stemms all words in a list
        
        :param text_list: list of words
        :type text_list: Iterable
        :return: cleaned list of words
        :rtype: list
        """
        # TODO: more efficient preprocessing
        return [self.stemmer.stem(word.lower()) for word in text_list]

    def _preprocess(self, text:str, stem:bool = True, lower:bool = True,  rm_stopwords:bool = True) -> str:
        """Preprocesses a single test
        
        :param text: The text to be processed
        :type text: str
        :return: The processed text.
        :rtype: str
        """
        # TODO more efficient preprocessing
        newtext = []
        newtext_appender = newtext.append

        tokenized = word_tokenize(text)

        for word in tokenized:
            if rm_stopwords and word in self.stopWords:
                continue

            if lower:
                word = word.lower()
            if stem:
                word = self.stemmer.stem(word)

            newtext_appender(word)
        # tokenized_stopwords_removed = self._remove_stopwords(tokenized)
        # tokenized_stemmed_lowered = self._stem_and_lower(tokenized_stopwords_removed)
        
        return ' '.join(newtext)

    def _make_dtm(self, texts:Iterable, dtm_type, stem:bool = True, lower:bool = True,  rm_stopwords:bool = True) -> Tuple[np.ndarray,  CountVectorizer]:
        #TODO: make preprocessing steps optional
        """Makes a DTM from a list of texts
        
        :param texts: A list of texts
        :type texts: Iterable
        :return: A list of texts as well as the vectorizer model
        :rtype: Tuple[np.ndarray,  CountVectorizer]
        """
        length_before = len(texts)
        
        preprocess_partial = partial(self._preprocess, stem = stem, lower = lower,  rm_stopwords = rm_stopwords)

        print('\tText preprocessing')
        with Pool(self.max_cores) as p:
            texts_preprocessed = p.map(preprocess_partial, texts)
            
        assert (length_before == len(texts_preprocessed)), 'After preprocessing there should be the same number of items as before'
        
        print('\tApply CountVectorizer')
        
        if dtm_type == 'count':
            vectorizer = CountVectorizer(min_df=self.min_df)
        elif dtm_type == 'tfidf':
            vectorizer = TfidfVectorizer(min_df=self.min_df)
        else:
            raise ValueError(print("unknown dtm type"))

        dtm = vectorizer.fit_transform(texts_preprocessed)
        
        assert (length_before == dtm.shape[0]), 'After preprocessing there should be the same number of items as before'
        print(f'\tDTM shape: {dtm.shape}')
        
        return dtm, vectorizer

    def _do_k_means(self) -> np.array:

        kmeans = KMeans(n_clusters=self.k, random_state=13, n_jobs = self.max_cores, verbose = 2)
        kmeans.fit(self.dtm)
        predictions = kmeans.predict(self.dtm)
        
        return predictions

    def _do_pca(self) -> Tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components = 2, random_state=13)
        coordinates = pca.fit_transform(self.dtm.todense())
        
        components = pca.components_
        
        return coordinates, components


    def _analyze(self, texts:list):
        self.dtm, self.vectorizer = self._make_dtm(texts, dtm_type = self.dtm_type)
        self.predictions = self._do_k_means()
        self.coordinates, self.components = self._do_pca()

    def _make_snippets(self, texts):
        snippets = ['\n'.join(textwrap.wrap(textwrap.shorten(text, width = self.max_snippet_length) ,width=self.snippet_width)) for text in texts]
        return snippets

    def _extract_top_words(self, dtm, vectorizer):
        result = []

        vocab = vectorizer.vocabulary_
        inverse_vocab = {value:key for key,value in vocab.items()}

        vocab = np.array([inverse_vocab[index] for index in sorted(list(inverse_vocab.keys()))])


        print(dtm.shape)
        for row in dtm.todense():
            row = np.array(row)[0]

            best_words_selector = np.argsort(row)[::-1]
            # input(best_words_selector)

            best_words = vocab[best_words_selector]
            best_words = best_words[0:self.max_top_words]
            best_words = ', '.join(list(best_words))
            best_words = '\n'.join(textwrap.wrap(best_words, self.snippet_width))
            result.append(best_words)

        return result



    def _make_top_words(self, texts):
        # TODO: make stemming optional. then do not stem
        tfidf_dtm, tf_idf_vectorizer = self._make_dtm(texts, dtm_type = 'tfidf', stem = True, lower = True,  rm_stopwords = True)
        return self._extract_top_words(tfidf_dtm, tf_idf_vectorizer)
        

    def _prepare_texts_for_display(self, texts):
        if self.text_display_type == 'snippet':
            return self._make_snippets(texts)
        elif self.text_display_type == 'top_words':
            return self._make_top_words(texts)

    def _make_plot(self, texts):
        print('Make_plot')
        #TODO: make output file an option
        output_file(self.output_file)

        xs = self.coordinates[:,0]
        x_scale = 1.0/(xs.max() - xs.min())
    #     print(x_scale)
        ys = self.coordinates[:,1]
        y_scale = 1.0/(ys.max() - ys.min())

        vocab = self.vectorizer.vocabulary_
        inverse_vocab = {value:key for key,value in vocab.items()}

        wordlist = [inverse_vocab[index] for index in sorted(list(inverse_vocab.keys()))]
        
        # TODO: make texts fit for display with textwarp or display x words with the highest tf-idf
        texts = self._prepare_texts_for_display(texts)
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

