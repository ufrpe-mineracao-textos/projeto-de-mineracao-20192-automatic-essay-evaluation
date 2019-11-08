"""
    This module is dedicated to perform the essay evaluation
    Before running this file, make sure the server to use the cogroo api is running in your computer.
    The instructions to run the CoGroo server can be found in the website https://github.com/gpassero/cogroo4py
"""
import numpy as np
import keras
from cogroo_interface import Cogroo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from models import res_model, evaluate_model
from keras.utils import to_categorical
from data_procedures import create_rules_id_dictionary, get_essays_texts_and_scores
from data_procedures import get_tfidf_of_essays_with_traintest_split, concatenate_tfidf_errors_arrays

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


def get_train_test_features(essays, scores, test_size=.3, verbose=False):
    """
    Calculates the train and test set features for the essays loaded.
    The features are the concatenation of the TF-IDF of the essays, with their corresponding error vectors
    :param essays: List of texts of the essays
    :param scores: Competences scores of each essay
    :param test_size: Size of the test set
    :return: Features corresponding to the train and test set
    """
    if verbose:
        print("[INFO] Grammar checking essays")
    detected_errors = grammar_check_essays(essays)

    if verbose:
        print("Computing TF/IDF of Train and validation sets")

    x_tfidf = get_tfidf_of_essays_with_traintest_split(essays, preprocess=True, verbose=verbose)

    if verbose:
        print("Concatenating detected errors with the TF-IDF of texts")

    detected_features = concatenate_tfidf_errors_arrays(x_tfidf, detected_errors)

    if verbose:
        print("Spliting train and test dataset, with test set size percentage of "+str(test_size))

    x_train_tfidf, x_test_tfidf, y_train_scores, y_test_scores = train_test_split(detected_features, scores,
                                                                                  test_size=test_size)

    return (x_train_tfidf, y_train_scores), (x_test_tfidf, y_test_scores)


def get_tfidf_of_essays_without_data_split(essays, verbose=True):
    """
    Calculates the train and test set features for the essays loaded.
    The features are the concatenation of the TF-IDF of the essays, with their corresponding error vectors.
    All that without performing data spliting
    :param texts:
    :param preprocess:
    :return: tf-idf vectors of the texts passed as input
    """
    if verbose:
        print("[INFO] Grammar checking essays")
    detected_errors = grammar_check_essays(essays)

    if verbose:
        print("Computing TF/IDF of Train and validation sets")

    x_tfidf = get_tfidf_of_essays_with_traintest_split(essays, preprocess=True, verbose=verbose)

    if verbose:
        print("Concatenating detected errors with the TF-IDF of texts")

    detected_features = concatenate_tfidf_errors_arrays(x_tfidf, detected_errors)

    return detected_features


def regression(verbose=False):
    essays_number = 100
    essays, scores = get_essays_texts_and_scores()

    (x_train, y_train), (x_test, y_test) = get_train_test_features(essays[0:essays_number], scores[0:essays_number],
                                                                   verbose=verbose)
    if verbose:
        print("[INFO] Setitng scores from competence 1 appart from the others")
    y_train_c1 = y_train[:, 1]
    y_test_c1 = y_test[:, 1]

    if verbose:
        print("[INFO] Performing linear regression over the data")
    reg = LinearRegression()
    reg.fit(x_train, y_train_c1)

    if verbose:
        print("[INFO] Computing the R2 score of the predictions")

    predictions = reg.predict(x_test)

    mean_scores = y_test_c1.sum()/y_test_c1.shape[0]
    squared_sum_desired = ((y_test_c1 - mean_scores)**2).sum()
    squared_dum_regression = ((y_test_c1 - predictions)**2).sum()

    error = predictions - y_test_c1
    mean_error = error.sum()/predictions.shape[0]
    # standard deviation
    stdd = np.sqrt(((error - mean_error)**2).sum()/error.shape[0])

    R2_SCORE = 1 - squared_dum_regression/squared_sum_desired

    print("R2 for a linear model: ", R2_SCORE)
    print("Mean error: ", mean_error)
    print("Error standard deviation: ", stdd)


def classification(verbose=False):
    essays_number = 100
    essays, scores = get_essays_texts_and_scores()

    essays, scores = shuffle(essays, scores)

    (x_train, y_train), (x_test, y_test) = get_train_test_features(essays[0:essays_number], scores[0:essays_number],
                                                                   verbose=verbose)
    if verbose:
        print("[INFO] Setitng scores from competence 1 appart from the others")
    y_train_c1 = y_train[:, 1]
    y_test_c1 = y_test[:, 1]

    if verbose:
        print("[INFO] using categorized classes")

    # classes 0.0 ,0.5, 1.0, 1.5, 2.0
    n_classes = 5
    y_cat_train_c1 = to_categorical(y_train_c1, n_classes)
    y_cat_test_c1 = to_categorical(y_test_c1, n_classes)

    if verbose:
        print("[INFO] Creating the machine learning model")

    model = res_model(x_train.shape[1:], n_classes)

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=.1, momentum=.3),
                  metrics=['accuracy'])

    batch_size = 10
    n_epochs = 10
    h = model.fit(x_train, y_cat_train_c1, batch_size=batch_size, epochs=n_epochs,
                  validation_data=(x_test, y_cat_test_c1), shuffle=True, verbose=1)

    evaluate_model(x_test, y_cat_test_c1, batch_size, model, n_epochs, h, n_classes, save_results=True)


if __name__ == "__main__":
    # regression(verbose=True)
    classification(verbose=True)
