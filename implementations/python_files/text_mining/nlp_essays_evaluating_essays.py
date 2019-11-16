"""
    This module is dedicated to perform the essay evaluation
    Before running this file, make sure the server to use the cogroo api is running in your computer.
    The instructions to run the CoGroo server can be found in the website https://github.com/gpassero/cogroo4py
"""
import os
import sys
sys.path.append("../")
import numpy as np
import keras
from cogroo_interface import Cogroo
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from text_mining.models import res_model, simple_model, evaluate_model
from keras.utils import to_categorical
from text_mining.data_procedures import create_rules_id_dictionary, get_essays_texts_and_scores, save_csv
from text_mining.data_procedures import get_tfidf_of_essays, concatenate_tfidf_errors_arrays

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
        # mistakes = doc.mistakes
        new_rule_vect = rule_vect.copy()

        if doc is not None:
            mistakes = doc.mistakes
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

    x_tfidf = get_tfidf_of_essays(essays, preprocess=True, verbose=verbose)

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

    x_tfidf = get_tfidf_of_essays(essays, preprocess=True, verbose=verbose)

    if verbose:
        print("Concatenating detected errors with the TF-IDF of texts")

    detected_features = concatenate_tfidf_errors_arrays(x_tfidf, detected_errors)

    return detected_features


def grammacheck_and_run_models(option_reg_class=True, verbose=False):
    essays, scores = get_essays_texts_and_scores()

    essays, scores = shuffle(essays, scores)

    features = get_tfidf_of_essays_without_data_split(essays, verbose=verbose)

    if option_reg_class:
        classification(features, scores, 5, verbose)
    else:
        regression(features, scores, verbose=verbose)


def classification(features, scores, n_classes, model_type=0, save_path='results/',
                   lr=.01, batch_size=10, n_epochs=20, test_size=.3,
                   verbose=False, save_results=False, normalize=True):
    """
    Performs the data spliting and the training of a classification model.
    :param features: array of feature vectors
    :param scores: array of scores per competenec
    :param n_classes: classes number
    :param model_type: model choice. The options are a simple Deep MLP model and a Residual model
    :param save_path: Path to which the results should be saved
    :param lr: learning rate
    :param batch_size:  Size of the batch
    :param n_epochs: Number of training epochs
    :param test_size: Percentage of the test set
    :param verbose: option to indicate the desire of verbosity
    :param save_results: options to indicate  the user desire to save the results in the path
    :param normalize: option to the user to normalize the data or not
    :return: the trained model
    """
    # features, scores = read_data_from_csv()
    verbose_opc = 0
    if verbose:
        print("[INFO] Shuffle Data")
        verbose_opc = 1

    features, scores = shuffle(features, scores, random_state=0)

    if normalize:
        if verbose:
            print("[INFO] Normalizing Data")
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    if verbose:
        print("[INFO] Splitting data into train and test sets")
    x_train, x_test, y_train, y_test = train_test_split(features, scores[:, 1], test_size=test_size)

    # classes 0.0 ,0.5, 1.0, 1.5, 2.0
    y_cat_train = to_categorical(y_train, n_classes)
    y_cat_test = to_categorical(y_test,n_classes)

    if verbose:
        print("[INFO] Creating the machine learning model")

    model = None
    if model_type == 0:
        model = res_model(x_train.shape[1:], n_classes)
    elif model_type == 1:
        model = simple_model(x_train.shape[1:], n_classes)
    elif model_type == 2:
        model = sklearn.svm.SVC(gamma='auto')
    elif model_type == 3:
        model = sklearn.ensemble.forest.RandomForestClassifier()
    elif model_type == 4:
        model = sklearn.ensemble.weight_boosting.AdaBoostClassifier()

    h = None
    if model_type >= 0 and model_type <= 1:
        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(lr=lr, momentum=.3),
                      metrics=['accuracy'])

        h = model.fit(x_train, y_cat_train,
                      batch_size=batch_size,
                      epochs=n_epochs,
                      validation_data=(x_test, y_cat_test),
                      verbose=verbose_opc)
    else:
         model.fit(x_train, y_cat_train)

    evaluate_model(x_test, y_cat_test, batch_size, model, n_epochs, h, n_classes, folder_name=save_path,
                   save_results=save_results)

    return model


def regression(features, scores, test_size=.3, save_path='results/',
               verbose=False, normalize=True, save_results=True):
    """
    Performs linear regression to the features

    :param features: array of data features
    :param scores:  array of scores per competence
    :param test_size: percentage indicating the size of the test set
    :param verbose: indicate the user's desire of verbosity
    :param normalize: indicate the user's desire for normalizing data
    :return: regression model
    """

    features, scores = shuffle(features, scores, random_state=0)

    if normalize:
        if verbose:
            print("[INFO] Normalizing Data")
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    if verbose:
        print("[INFO] Spliting data into test and train")

    x_train, x_test, y_train, y_test = train_test_split(features, scores, test_size=test_size)

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

    mean_scores = y_test_c1.sum() / y_test_c1.shape[0]
    squared_sum_desired = ((y_test_c1 - mean_scores) ** 2).sum()
    squared_sum_regression = ((y_test_c1 - predictions) ** 2).sum()

    error = predictions - y_test_c1
    mean_error = error.sum() / predictions.shape[0]
    # standard deviation
    stdd = np.sqrt(((error - mean_error) ** 2).sum() / error.shape[0])

    R2_SCORE = 1 - squared_sum_regression / squared_sum_desired

    if verbose:
        print("[RESULT] R2 for a linear model: ", R2_SCORE)
        print("[RESULT] Desired squared sum: ", squared_sum_desired)
        print("[RESULT] Desired sum regression: ", squared_sum_regression)
        print("[RESULT] Mean error: ", mean_error)
        print("[RESULT] Error standard deviation: ", stdd)

    if save_results:
        if verbose:
            print("[INFO] Saving Results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(save_path + "eval_regression.txt", 'w') as f:
            string_output = "R2 for a linear model: " + str(R2_SCORE) + " \n"
            string_output += "Desired squared sum: " + str(squared_sum_desired) + " \n"
            string_output += "Desired sum regression: " + str(squared_sum_regression) + " \n"
            string_output += "Mean error: " + str(mean_error) + " \n"
            string_output += "Error standard deviation: " + str(stdd) + " \n"
            f.write(string_output)
            f.close()

    return reg


def save_data_into_a_csv(verbose=False):
    essays, scores = get_essays_texts_and_scores()
    if verbose:
        print("[INFO] GETTING TF-IDF of texts without data spliting")
    data_features = get_tfidf_of_essays_without_data_split(essays,verbose=True)

    if verbose:
        print("[INFO] Saving data into csv files")
        print("[INFO] Saving FEATURES")

    save_csv(data_features)

    if verbose:
        print("[INFO] Saving scores")
    save_csv(scores, file_name="scores_of_features.csv")


def read_data_from_csv(features_file="detected_features.csv", scores_file="scores_of_features.csv"):

    features = None
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            features = np.genfromtxt(f, delimiter=',')
            f.close()
    else:
        raise Exception("File "+features_file+" does not exist")

    scores_of_features = None
    if os.path.exists(scores_file):
        with open(scores_file, 'r') as f:
            scores_of_features = np.genfromtxt(f, delimiter=',')
            f.close()
    else:
        raise Exception("File "+scores_file+" does not exist")

    return features, scores_of_features


if __name__ == "__main__":

    models_types = {'res_net': 0,
                    'mlp': 1,
                    'svm': 2,
                    'random_forest': 3,
                    'ada_boost': 4,
                    'xgboost': 5}

    models_names = list(models_types.keys())
    features, scores = read_data_from_csv()
    classification(features[0:10], scores[0:10], 5, model_type= models_types[models_names[1]], save_results=True)
