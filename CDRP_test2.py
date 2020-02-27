# Load Comments table
import pandas as pd
from ast import literal_eval
from nltk.corpus import stopwords
from CDRP import ClusterDimRedPLot

delegations = set([ "AT",    # Austria
                    "BE",    # Belgium
                    "BG",    # Bulgaria
                    "HR",    # Croatia
                    "CY",    # Cyprus
                    "CZ",    # Czech Republic
                    "DK",    # Denmark
                    "IE",    # Ireland
                    "EE",    # Estonia
                    "FI",    # Finland
                    "FR",    # France
                    "DE",    # Germany
                    "GR", "HE", "EL",  # Greece
                    "HU",    # Hungary
                    "IT",    # Italy
                    "LV",    # Latvia
                    "LU",    # Luxembourg
                    "LT",    # Lithuania
                    "MT",    # Malta
                    "NL",    # Netherlands
                    "NO",    # Norway
                    "PL",    # Poland
                    "PT",    # Portugal
                    "RO",    # Romania
                    "SK",    # Slovakia
                    "SI",    # Slovenia
                    "ES",    # Spain
                    "SE",    # Sweden
                    "UK",    # United Kingdom
                    "Cion", "COM"    # Commission
                  ])  


# stemmer = PorterStemmer()
stopWords = set(stopwords.words('english'))
stopWords.add("article")

for delegation in delegations:
    stopWords.add(delegation.lower())


def read_comments_df(filename):
    converters = {'initiators':literal_eval,
                  'supporters':literal_eval,
                  'opposers':literal_eval,
                  'other_role':literal_eval}

    comment_df = pd.read_csv(filename, sep='\t', index_col = 0, converters=converters)
    return comment_df
    
comments_df = read_comments_df("comments_heavy_with_delegations_extracted.csv")
print(comments_df.shape)
comments_df = comments_df.drop_duplicates(subset = 'comment')
print(comments_df.shape)

analyzer = ClusterDimRedPLot(dtm_type='count', stopwords = stopWords, max_cores=-2, text_display_type='top_words')

analyzer.make_CDRP(comments_df.comment.values)