"""
    This module is dedicated to perform the essay evaluation
    Before running this file, make sure the server to use the cogroo api is running in your computer.
    The instructions to run the CoGroo server can be found in the website https://github.com/gpassero/cogroo4py
"""
import numpy as np
from cogroo_interface import Cogroo
from sklearn.model_selection import train_test_split
from data_procedures import create_rules_id_dictionary, get_essays_texts_and_scores
from data_procedures import get_tfidf_of_essays, concatenate_tfidf_errors_arrays

cogroo = Cogroo.Instance()


def grammar_check_essays(essays):
    """
    Use the cogroo tool to perform the grammar checking of each essay.
    Then an rule vector is created and for the i-th error found in the essay, the i-th cell of the rule vector
    is incremented by one.
    :param essays:
    :return: rule vectors of the errors detected in the essays
    """
    rules_dict, rule_vect = create_rules_id_dictionary()
    errors = []
    for e in essays:
        doc = cogroo.grammar_check(e)
        mistakes = doc.mistakes
        new_rule_vect = rule_vect.copy()
        for m in mistakes:
            # Excluding errors related to spacing in the texts
            if 'space:' not in m.rule_id:
                new_rule_vect[rules_dict[m.rule_id]] += 1
        errors.append(new_rule_vect)

    return np.array(errors)


def get_train_test_features(essays, scores, test_size=.3):
    """
    Calculates the train and test set features for the essays loaded.
    The features are the concatenation of the TF-IDF of the essays, with their corresponding error vectors
    :param essays: List of texts of the essays
    :param scores: Competences scores of each essay
    :param test_size: Size of the test set
    :return: Features corresponding to the train and test set
    """
    X_train_texts, X_test_texts, Y_train_scores, Y_test_scores = train_test_split(essays, scores, test_size=test_size)

    # Computing TF/IDF of Train and validation sets
    X_train_tfidf = get_tfidf_of_essays(X_train_texts, preprocess=True)
    X_test_tfidf = get_tfidf_of_essays(X_test_texts, preprocess=True)

    # Grammar checking essays
    train_errors = grammar_check_essays(X_train_texts)
    test_errors  = grammar_check_essays(X_train_texts)

    train_features = concatenate_tfidf_errors_arrays(X_train_tfidf, train_errors)
    test_features = concatenate_tfidf_errors_arrays(X_test_tfidf, test_errors)

    return train_features, test_features


def classification():
    essays, scores = get_essays_texts_and_scores()

    train_features, test_features = get_train_test_features(essays[0:11], scores[0:11])

    print(train_features.shape)
    print(test_features.shape)


if __name__ == "__main__":
    classification()
