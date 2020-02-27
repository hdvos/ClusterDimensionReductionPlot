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

from typing import Callable, Iterable, Tuple, Union

def parse_max_cores(n:int) -> int:
    """A parser to parse the n_cores argument.
    
    :param n: number given by the user
    :type n: int
    :raises ValueError: if invalid number is given
    :return: The actual number of cores that will be used.
    :rtype: int
    """
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
    

    def __init__(self, k:int=10, dtm_type:str = 'count', n_cores:int = 3, 
        min_df:int = 3, stopwords:Union[str,Iterable[str]] = 'english', stemmer:Union[Callable, None] = None, 
        output_file:Union[str, None] = None, overwrite:bool = False, text_display_type:str = 'snippet', 
        max_snippet_length:int = 200, snippet_width:int = 50, max_top_words:int = 5):
        """Init
        
        :param k: Number of clusters for the K-means algorithm, defaults to 10
        :type k: int, optional
        :param dtm_type: Will k-means use countvectors ('count') or tf-idf ('tfidf') vectors, defaults to 'count'
        :type dtm_type: str, optional
        :param n_cores: The number of cores to be used during the different processes that can make use of multiprocessing. Use -1 to use all cores, use -2 to use all cores but one. defaults to 3
        :type n_cores: int, optional
        :param min_df: The minimal document frequency of a word. Words with a lower document frequency will be ignored in the document-term matrix. See website of sklearn for further documentation about min_df (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) defaults to 3
        :type min_df: int, optional
        :param stopwords: Either a list of stopwords or a string. The string will be directly passed to the stopwords function of nltk, defaults to 'english'
        :type stopwords: Union[str,Iterable[str]], optional
        :param stemmer: Either a stemming function or None. If none, the Porterstemmer will be used, defaults to None #TODO: make more intuitive, provide option to opt-out for stemming
        :type stemmer: Union[Callable, None], optional
        :param output_file: Either the name of the output (html-file), defaults to None. If none: an output file is generated according to the format: "CDRP_plot_<k>.html".
        :type output_file: Union[str, None], optional
        :param overwrite: Overwrite a previous output, defaults to False. If False: a Unix Time stamp is added to the filename.
        :type overwrite: bool, optional
        :param text_display_type: What should appear when hovering over a point in the scatter plot: a "snippet": the first n characters of a text, or the top n words ("top_words"): the n words with the highest tf-idf. defaults to 'snippet'
        :type text_display_type: str, optional
        :param max_snippet_length: If text display type is "snippet" how many characters are shown, defaults to 200
        :type max_snippet_length: int, optional
        :param snippet_width: What is the width in characters of the text snippet. defaults to 50
        :type snippet_width: int, optional
        :param max_top_words: If text_display_type is top_words: how long is the list of displayed words., defaults to 5
        :type max_top_words: int, optional
        :raises ValueError: if non existent dtm_type is given
        :raises ValueError: if non existent text_display_type is given
        """
        

        if not dtm_type in ['count', 'tfidf']:
            raise ValueError(f"Unknown dtm type {dtm_type}") 
        
        self.k = k
        self.dtm_type = dtm_type
        self.max_cores = parse_max_cores(n_cores)
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


    def _preprocess(self, text:str, stem:bool = True, lower:bool = True,  rm_stopwords:bool = True) -> str:
        """Preprocess a text

        :param text: The text that needs to be pre-processed
        :type text: str
        :param stem: Opt-in/out for stemming, defaults to True
        :type stem: bool, optional
        :param lower: opt-in/-out for lowering, defaults to True
        :type lower: bool, optional
        :param rm_stopwords: Will stopwords be removed, defaults to True
        :type rm_stopwords: bool, optional

        :return: The preprocessed text
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
        
        return ' '.join(newtext)

    def _make_dtm(self, texts:Iterable, dtm_type:str, stem:bool = True, lower:bool = True,  rm_stopwords:bool = True) -> Tuple[np.ndarray,  CountVectorizer]:
        """Make a document-term matrix (a.k.a. vector space model)
        
        :param texts: A list of texts
        :type texts: Iterable
        :param dtm_type: tf-idf vectors ('tfidf') pr count vectors ('count')
        :type dtm_type: str
        :param stem: whether to apply stemming, defaults to True
        :type stem: bool, optional
        :param lower: whether to lowercase, defaults to True
        :type lower: bool, optional
        :param rm_stopwords: whether to remove stopwords, defaults to True
        :type rm_stopwords: bool, optional
        :raises ValueError: if invalid dtm type is given
        :return: both the document term matrix as the vectorizer object.
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
        """Apply k-means to the dtm
        
        :return: the cluster predictions to all texts.
        :rtype: np.array
        """
        kmeans = KMeans(n_clusters=self.k, random_state=13, n_jobs = self.max_cores, verbose = 2)
        kmeans.fit(self.dtm)
        predictions = kmeans.predict(self.dtm)
        
        return predictions

    def _do_pca(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply pca on the dtm
        
        :return: the coordinates (loadings for the texts) as well as the components (loadings for the words.)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        pca = PCA(n_components = 2, random_state=13)
        coordinates = pca.fit_transform(self.dtm.todense())
        
        components = pca.components_
        
        return coordinates, components


    def _analyze(self, texts:list):
        """Applies the whole pipeline: preprocessing, document term matrix, k-means and pca.
        
        :param texts: [description]
        :type texts: list
        """
        self.dtm, self.vectorizer = self._make_dtm(texts, dtm_type = self.dtm_type)
        self.predictions = self._do_k_means()
        self.coordinates, self.components = self._do_pca()

    # All functions below apply to visualisation.
    def _make_snippets(self, texts:list) -> list:
        """Make snippets of the texts to display in the figure.
        
        :param texts: [description]
        :type texts: list
        :return: [description]
        :rtype: list
        """
        snippets = ['\n'.join(textwrap.wrap(textwrap.shorten(text, width = self.max_snippet_length) ,width=self.snippet_width)) for text in texts]
        return snippets

    def _extract_top_words(self, dtm:np.ndarray, vectorizer) -> list:
        """Extract the words with the highest tf-idf for each text.
        
        :param dtm: a document-term matrix
        :type dtm: np.ndarray
        :param vectorizer: the vectorizer object used to make the dtm
        :type vectorizer: Either a sklearn countvectorizer or tf-idf vectorizer.
        :return: List of strings top words. Every string contains the top n words concatenate with ', '
        :rtype: list
        """
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



    def _make_top_words(self, texts:list) -> list:
        """Make a list of top words to display in the figure.
        
        :param texts: list of texts
        :type texts: list[str]
        :return: A list of top n words.
        :rtype: list of strings
        """
        tfidf_dtm, tf_idf_vectorizer = self._make_dtm(texts, dtm_type = 'tfidf', stem = True, lower = True,  rm_stopwords = True)
        return self._extract_top_words(tfidf_dtm, tf_idf_vectorizer)
        

    def _prepare_texts_for_display(self, texts:list) -> list:
        """Manages all steps needed to prepare texts for how they will be presented in the figure.
        
        :param texts: List of tests
        :type texts: list
        :return: a list of the prepared texts.
        :rtype: list
        """
        if self.text_display_type == 'snippet':
            return self._make_snippets(texts)
        elif self.text_display_type == 'top_words':
            return self._make_top_words(texts)

    def _make_plot(self, texts:list):
        """Compile everything in a single plot
        
        :param texts: list of texts
        :type texts: list
        """
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
        """One callable to run all methods above in the correct order.
        
        :param texts: List of texts.
        :type texts: list
        """
        self._analyze(texts)
        self._make_plot(texts)

