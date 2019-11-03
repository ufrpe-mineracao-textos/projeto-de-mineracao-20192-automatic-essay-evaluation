"""
    This module is dedicated to loading and treating the data.
"""

import numpy as np
import uol_redacoes_xml as uol
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_rules_id_dictionary(n_rules=129):
    """
    Creates a dictionary for the rules returned by the CoGroo API.
    This dictionary is going to be used for when the rule is retuned for the API,
    its id will come in the format: "xml:<number>", So based on the number stored in the corresponding
    position of the id in format "xml:<number>" the value in the index of the rule in the feature vector
    will be increased.

    :param n_rules: number of rules
    :return:
    """
    prefix = "xml:"
    rules = {}
    rules_vector = np.zeros((n_rules,))
    for i in range(n_rules):
        rules[prefix+str(i+1)] = i
    return rules, rules_vector


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


def get_tfidf_of_essays(texts):

    count_vectorizer = CountVectorizer(encoding='latin-1')
    tfidf_transformer = TfidfTransformer(use_idf=True)
    count_vect = count_vectorizer.fit_transform(texts)
    tfidf = tfidf_transformer.fit_transform(count_vect).toarray()

    return tfidf

