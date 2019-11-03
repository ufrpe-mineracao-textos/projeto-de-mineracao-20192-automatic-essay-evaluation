"""
    This module is dedicated to loading and treating the data.
"""

import numpy as np
import uol_redacoes_xml as uol
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_raw_uol_essays():
    """
    Returns the list of raw essay objects of the uol library
    """
    return uol.load()

def get_essays_texts_and_scores():
    """
    Iterates over the essays objects and returns a list of tuples, with only the texts of the essays and the scores
    of each competence being evaluated.
    :return: list of tuples with the texts of e essays
    """
    texts = []
    scores = []
    essays = get_raw_uol_essays()

    for e in essays:
        competences = list(e.criteria_scores.keys())
        texts.append(e.text)
        scores.append([e.criteria_scores[competences[0]],
                       e.criteria_scores[competences[1]],
                       e.criteria_scores[competences[2]],
                       e.criteria_scores[competences[3]],
                       e.criteria_scores[competences[4]]])

    return texts, np.array(scores)


def get_tfidf_of_essays():

    count_vectorizer = CountVectorizer(encoding='latin-1')
    tfidf_transformer = TfidfTransformer(use_idf=True)

    texts, scores = get_essays_texts_and_scores()

    count_vect = count_vectorizer.fit_transform(texts)
    tfidf = tfidf_transformer.fit_transform(count_vect).toarray()

    return tfidf





