"""
    This module is dedicated to loading and treating the data.
"""

import numpy as np
import uol_redacoes_xml as uol
import spacy
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nlp = spacy.load("pt")


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
    # adding number of extra rules without a specific numerical ID
    n_rules += 5

    rules = {}

    rules['punctuation:BEFORE_SENTENCES'] = 0
    rules['punctuation:EXTRA_PUNCTUATION'] = 0
    rules['repetition:DUPLICATED_TOKEN'] = 0
    rules['government:GOVERNMENT'] = 0
    rules['probs:paronyms'] = 0

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


# Dummy function necessary to use the count vectorizer with inputs already tokenized and lemmatized
# with stop words removed
def dummy_fun(doc):
    return doc


def get_tfidf_of_essays_with_traintest_split(texts, preprocess=False, verbose=False):
    """
    Performs the TF-IDF processing of the texts .
    And takes the option preprocess to perform the preprocessing on the texts like tokenization, lemmatization,
    and punctuation and stopword removal
    :param texts:
    :param preprocess:
    :return: tf-idf vectors of the texts passed as input
    """
    data = []
    if preprocess:
        if verbose:
            print("[INFO] Preprocessing texts of each essay, performing tokenization," +
                  " lemmatization and stop word removal")
        if type(texts) == list:
            for t in texts:
                tokens = nlp(t)
                non_stop_tokens = [t for t in tokens if not (t.is_stop or t.is_punct)]
                lemmas = [nst.lemma_.lower() for nst in non_stop_tokens]
                data.append(lemmas)
    else:
        if verbose:
            print("[INFO] Using raw texts as input to TF-IDF vectorizers")
        data = texts

    if verbose:
        print("[INFO] Transforming input texts into vectors")
    count_vectorizer = CountVectorizer(tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None, encoding='latin-1')
    tfidf_transformer = TfidfTransformer(use_idf=True)
    count_vect = count_vectorizer.fit_transform(data)
    tfidf = tfidf_transformer.fit_transform(count_vect).toarray()

    return tfidf


def concatenate_tfidf_errors_arrays(tfidf, errors, verbose=False):
    """
    Concatenate the TF-IDF arrays to the errors arrays after the grammar checking is performed using cogroo.

    :param tfidf: Tf-IDF arrays of essays
    :param errors: Error Array of essays
    :return: A numpy array of the new features which are the Tf-IDF arrays concatenated to the Error arrays
    """
    new_features = []
    for f, e in zip(tfidf, errors):
        new_features.append(np.concatenate([f, e]))
    return np.array(new_features)


def save_csv(array, file_path="", file_name="detected_features.csv"):

    with open(file_path+file_name) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(array)
        f.close()
