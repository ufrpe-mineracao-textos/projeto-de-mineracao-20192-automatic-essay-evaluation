"""
    This module is dedicated to perform the essay evaluation
    Before running this file, make sure the server to use the cogroo api is running in your computer.
    The instructions to run the CoGroo server can be found in the website https://github.com/gpassero/cogroo4py
"""
import numpy as np
from cogroo_interface import Cogroo
from sklearn.model_selection import train_test_split
from data_procedures import create_rules_id_dictionary, get_essays_texts_and_scores, get_tfidf_of_essays

cogroo = Cogroo.Instance()

def grammar_check_essays(essays):
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

def classification():
    essays, scores = get_essays_texts_and_scores()
    X_train_texts, X_test_texts, Y_train_scores, Y_test_scores = train_test_split(essays, scores, test_size=0.3)

    # Computing TF/IDF of Train and validation sets
    X_train_tfidf = get_tfidf_of_essays(X_train_texts)
    X_test_tfidf = get_tfidf_of_essays(X_test_texts)

    # Grammar checking essays
    train_errors = grammar_check_essays(X_train_texts)
    test_errors  = grammar_check_essays(X_train_texts)
    print(train_errors)




if __name__ == "__main__":
    classification()
