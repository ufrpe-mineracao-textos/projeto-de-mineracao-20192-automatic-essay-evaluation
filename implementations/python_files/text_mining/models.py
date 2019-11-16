"""
    This module is dedicated to the machine learning modules dedicated to evaluate the loaded essays
"""
import os
import numpy as np
from keras.layers import Dense, Dropout, Add
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def calculate_mean_and_stdd(list_of_values):
    """
    Calculates the mean and standard deviation of a set
    :param list_of_values:
    :return: mean and standard deviation
    """
    mean = 0
    stdd = 0
    set_size = len(list_of_values)
    for acc in list_of_values:
        mean += acc
    mean = mean/set_size
    for acc in list_of_values:
        stdd += np.power(acc - mean, 2)
    stdd = np.sqrt(stdd/set_size)

    return mean, stdd


def res_model(input_shape, n_classses):

    x_input = Input(input_shape)
    x_skip = Dense(input_shape[0], use_bias=True, activation='relu', name="input_layer")(x_input)
    x = Dense(100, use_bias=True, activation='relu', name="second_layer")(x_skip)
    x = Dense(100, use_bias=True, activation='relu', name="third_layer")(x)
    x = concatenate([x, x_skip])
    x_skip = Dense(150, use_bias=True, activation='relu')(x)
    x = Dense(100, use_bias=True, activation='relu')(x_skip)
    x = Dense(100, use_bias=True, activation='relu')(x)
    x = concatenate([x, x_skip])
    x = Dense(n_classses, use_bias=True, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=x, name="res_model")

    return model


def simple_model(input_shape, n_classes):
    
    x_input = Input(input_shape)
    x = Dense(input_shape[0], use_bias=True, activation='tanh', name='input_layer')(x_input)
    x = Dense(200, use_bias=True, activation='tanh', name="dense_1")(x)
    x = Dense(200, use_bias=True, activation='tanh', name="dense_2")(x)
    x = Dense(200, use_bias=True, activation='tanh', name="dense_3")(x)
    x = Dense(n_classes, use_bias=True, activation='softmax', name="output_layer")(x)

    model = Model(inputs=x_input, outputs=x, name='simple_model')

    return model


def evaluate_model(test_data, test_labels, batch_size, model, n_epochs, H, n_classes,
                   folder_name='results/', save_results=False):
    ## Evaluating model
    print("[INFO] Evaluating Network")

    if save_results and not os.path.exists(folder_name):
        os.mkdir(folder_name)

    if save_results:
        if H is not None:
            train_mean_acc, train_stdd_acc = calculate_mean_and_stdd(H.history["acc"])
            val_mean_acc, val_stdd_acc = calculate_mean_and_stdd(H.history["val_acc"])

        with open(folder_name+"eval.txt", 'w') as f:
            predictions = model.predict(test_data, batch_size=batch_size)
            value = classification_report(test_labels.argmax(axis=1),
                                          predictions.argmax(axis=1))
            value += "\nTrain acc mean: "+str(train_mean_acc)+"\t ,Train acc stdd: "+str(train_stdd_acc)
            value += "\nValidation acc mean: " + str(val_mean_acc) + "\t ,Validation acc stdd: " + str(val_stdd_acc)+ "\n\n"
            print(value)
            f.write(value)
            f.close()

    if H is not None:
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, n_epochs), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, n_epochs), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy ")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        if save_results:
            plt.savefig(folder_name + "LossAccComparison.png")
            plt.close('all')
