from CDRP import ClusterDimRedPLot

texts = []

with open('1000_ab_scientometrics.csv', 'rt') as f:
    texts = f.readlines()
    texts = texts[1:]
    assert(len(texts) == 1000)


analyzer = ClusterDimRedPLot(k=10, dtm_type='tfidf', stopwords = 'english', n_cores=-2, text_display_type='top_words')

analyzer.make_CDRP(texts)