import matplotlib.pyplot as plt
import numpy as np
import ctypes
from typing import *
import os
import time
from tkinter import *
from tkinter.messagebox import *
from PIL import Image

# rust_machine_learning_library = ctypes.CDLL(r"C:\Users\Moi\Desktop\Projets\3IABD2 - Projet annuel\rust_lib\target\debug\rust_lib.dll")
rust_machine_learning_library = ctypes.CDLL(r".\rust_lib\target\release\rust_lib.dll")
POINTER_TO_FLOAT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_float)
POINTER_TO_INT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_int32)


rust_machine_learning_library.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]
rust_machine_learning_library.delete_float_array.restype = None

rust_machine_learning_library.generate_linear_model.argtypes = [ctypes.c_int32, ctypes.c_int32]
rust_machine_learning_library.generate_linear_model.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.train_linear_model_classification.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_int32,
]
rust_machine_learning_library.train_linear_model_classification.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.train_linear_model_regression.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]
rust_machine_learning_library.train_linear_model_regression.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.predict_with_linear_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_bool,
]
rust_machine_learning_library.predict_with_linear_model.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.generate_multi_layer_perceptron_model.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
rust_machine_learning_library.generate_multi_layer_perceptron_model.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.train_multi_layer_perceptron_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_bool,
]
rust_machine_learning_library.train_multi_layer_perceptron_model.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.predict_with_multi_layer_perceptron_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_bool,
]
rust_machine_learning_library.predict_with_multi_layer_perceptron_model.restype = ctypes.POINTER(ctypes.c_float)


def get_number_of_weights(layers_list):
    number_of_weights = 0
    for index in range(len(layers_list) - 1):
        number_of_weights += (layers_list[index] + 1) * layers_list[index + 1]
    return number_of_weights


def read_only_train_dataset(dataset_folders: Union[str, list[str]]):
    inputs = []
    labels = []
    number_of_inputs = 0
    for dataset_class in range(len(dataset_folders)) :
        try :
            dataset_folder = os.listdir(dataset_folders[dataset_class])
            for filename in dataset_folder:
                if ".jpg" in filename or ".jpeg" in filename or ".png" in filename:
                    with Image.open(os.path.join(dataset_folders[dataset_class], filename)) as filename :
                        # print(filename)
                        inputs.append(np.asarray(filename))
                        number_of_inputs += 1
                        labels.append([1. if index == dataset_class else -1 for index in range(len(dataset_folders))])
        except FileNotFoundError :
            showerror(f"Wrong path", f"{dataset_folders[dataset_class]} could not be found.")
            raise
    inputs = np.array(inputs, dtype=ctypes.c_float).flatten()  / 255 * 2 - 1
    # inputs = np.array([input / 255 * 2 - 1 for input in inputs])
    labels = np.array(labels, dtype=ctypes.c_float).flatten()
    return (inputs, number_of_inputs, labels)
    

def read_both_datasets(dataset_folders: list[list[str]]):
    train_inputs = []
    tests_inputs = []
    train_labels = []
    tests_labels = []
    number_of_train_inputs = 0
    number_of_tests_inputs = 0
    for dataset_class in range(len(dataset_folders)):
        try :
            dataset_folder = os.listdir(dataset_folders[dataset_class][0])
            for filename in dataset_folder:
                if ".jpg" in filename or ".jpeg" in filename or ".png" in filename:
                    with Image.open(os.path.join(dataset_folders[dataset_class][0], filename)) as filename :
                        # print(filename)
                        train_inputs.append(np.asarray(filename))
                        number_of_train_inputs += 1
                        train_labels.append([1. if index == dataset_class else -1 for index in range(len(dataset_folders))])
        except FileNotFoundError :
            showerror(f"Wrong path", f"{dataset_folders[dataset_class][0]} could not be found.")
            raise
        try :
            dataset_folder = os.listdir(dataset_folders[dataset_class][1])
            for filename in dataset_folder:
                if ".jpg" in filename or ".jpeg" in filename or ".png" in filename:
                    with Image.open(os.path.join(dataset_folders[dataset_class][1], filename)) as filename :
                        # print(filename)
                        tests_inputs.append(np.asarray(filename))
                        number_of_tests_inputs += 1
                        tests_labels.append([1. if index == dataset_class else -1 for index in range(len(dataset_folders))])
        except FileNotFoundError :
            showerror(f"Wrong path", f"{dataset_folders[dataset_class][1]} could not be found.")
            raise
    train_inputs = np.array(train_inputs, dtype=ctypes.c_float).flatten()  / 255 * 2 - 1
    tests_inputs = np.array(tests_inputs, dtype=ctypes.c_float).flatten()  / 255 * 2 - 1
    # inputs = np.array([input / 255 * 2 - 1 for input in inputs])
    train_labels = np.array(train_labels, dtype=ctypes.c_float).flatten()
    tests_labels = np.array(tests_labels, dtype=ctypes.c_float).flatten()
    return (train_inputs, train_labels, number_of_train_inputs, tests_inputs, tests_labels, number_of_tests_inputs)
    

def read_dataset(dataset_folders: Union[str, list[str], list[list[str]]]):
    if not dataset_folders :
        return
    if isinstance(dataset_folders, str) :
        dataset_folders = [dataset_folders]
    if isinstance(dataset_folders[0], str) :
        return read_only_train_dataset(dataset_folders)
    return read_both_datasets(dataset_folders)


def generate_linear_model(dimensions_of_inputs: int, number_of_classes: int):
    number_of_weights = (dimensions_of_inputs + 1) * number_of_classes
    dimensions_of_inputs = ctypes.c_int32(dimensions_of_inputs)
    number_of_classes = ctypes.c_int32(number_of_classes)
    pointer_to_untrained_model = rust_machine_learning_library.generate_linear_model(dimensions_of_inputs, number_of_classes)
    return np.ctypeslib.as_array(pointer_to_untrained_model, (number_of_weights,))


def train_linear_model(is_classification: bool,
                       training_inputs: Union[list[float], np.ndarray],
                       tests_inputs: Union[list[float], np.ndarray] = [],
                       training_labels: Union[list[float], np.ndarray] = [],
                       tests_labels: Union[list[float], np.ndarray] = [],
                       model: Union[list[float], np.ndarray] = [],
                       learning_rate: float = 0.01,
                       number_of_epochs: float = 100,
                       number_of_training_inputs: int = 0,
                       number_of_tests_inputs: int = 0,
                       dimensions_of_inputs: int = 0,
                       number_of_classes: int = 0,
                       batch_size: int = 1):
    
    if not isinstance(is_classification, bool):
        raise TypeError
    
    if not isinstance(number_of_training_inputs, int):
        raise TypeError
    if number_of_training_inputs < 1 :
        raise ValueError
    # if not isinstance(number_of_tests_inputs, int):
    #     raise TypeError
    # if number_of_tests_inputs < 1 :
    #     raise ValueError
    
    if not isinstance(dimensions_of_inputs, int):
        raise TypeError
    if dimensions_of_inputs < 1 :
        raise ValueError
    
    if not isinstance(number_of_classes, int):
        raise TypeError
    if number_of_classes < 1 :
        raise ValueError
    
    if not isinstance(training_inputs, np.ndarray):
        raise TypeError
    if number_of_training_inputs * dimensions_of_inputs != len(training_inputs):
        raise ValueError
    training_inputs_as_c_float_array = (ctypes.c_float * (number_of_training_inputs * dimensions_of_inputs))(*training_inputs)
    pointer_to_training_inputs = ctypes.cast(training_inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    # if not isinstance(tests_inputs, np.ndarray):
    #     raise TypeError
    # if number_of_tests_inputs * dimensions_of_inputs != len(tests_inputs):
    #     raise ValueError
    # tests_inputs_as_c_float_array = (ctypes.c_float * (number_of_tests_inputs * dimensions_of_inputs))(*tests_inputs)
    # pointer_to_tests_inputs = ctypes.cast(tests_inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if not isinstance(training_labels, np.ndarray):
        raise TypeError
    if number_of_training_inputs * number_of_classes != len(training_labels):
        raise ValueError
    training_labels_as_c_float_array = (ctypes.c_float * (number_of_training_inputs * number_of_classes))(*training_labels)
    pointer_to_training_labels = ctypes.cast(training_labels_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    # if not isinstance(tests_labels, np.ndarray):
    #     raise TypeError
    # if number_of_tests_inputs * number_of_classes != len(tests_labels):
    #     raise ValueError
    # tests_labels_as_c_float_array = (ctypes.c_float * (number_of_tests_inputs * number_of_classes))(*tests_labels)
    # pointer_to_tests_labels = ctypes.cast(tests_labels_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(model, list):
        model = np.array(model, dtype=ctypes.c_float)
    if not isinstance(model, np.ndarray):
        raise TypeError
    number_of_weights = (dimensions_of_inputs + 1) * number_of_classes
    model_as_c_float_array = (ctypes.c_float * number_of_weights)(*model)
    pointer_to_model = ctypes.cast(model_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if is_classification :
        if isinstance(learning_rate, float):
            learning_rate = ctypes.c_float(learning_rate)
        if not isinstance(learning_rate, ctypes.c_float):
            raise TypeError
        
        if isinstance(number_of_epochs, int):
            number_of_epochs = ctypes.c_int32(number_of_epochs)
        if not isinstance(number_of_epochs, ctypes.c_int32):
            raise TypeError

    number_of_training_inputs = ctypes.c_int32(number_of_training_inputs)
    # number_of_tests_inputs = ctypes.c_int32(number_of_tests_inputs)
    dimensions_of_inputs = ctypes.c_int32(dimensions_of_inputs)
    number_of_classes = ctypes.c_int32(number_of_classes)
    print("Training...")
    if is_classification :
        pointer_to_trained_model = rust_machine_learning_library.train_linear_model_classification(
                                                                                                   pointer_to_model,
                                                                                                   pointer_to_training_inputs,
                                                                                                   number_of_training_inputs,
                                                                                                   # pointer_to_tests_inputs,
                                                                                                   # number_of_tests_inputs,
                                                                                                   dimensions_of_inputs,
                                                                                                   pointer_to_training_labels,
                                                                                                   # pointer_to_tests_labels,
                                                                                                   number_of_classes,
                                                                                                   learning_rate,
                                                                                                   number_of_epochs,
                                                                                                   # batch_size
                                                                                                  )
    else :
        pointer_to_trained_model = rust_machine_learning_library.train_linear_model_regression(
                                                                                               pointer_to_training_inputs,
                                                                                               number_of_training_inputs,
                                                                                               dimensions_of_inputs,
                                                                                               pointer_to_training_labels,
                                                                                               number_of_classes,
                                                                                              )
    print("The model has been trained.")
    trained_model = np.ctypeslib.as_array(pointer_to_trained_model, (number_of_weights,))
    return trained_model
    

def predict_with_linear_model(is_classification: bool,
                         inputs: Union[list[float], np.ndarray],
                         model: Union[list[float], np.ndarray],
                         number_of_inputs: int = 0,
                         dimensions_of_inputs: int = 0,
                         number_of_classes: int = 0):

    if not isinstance(is_classification, bool):
        raise TypeError
    
    if not isinstance(number_of_inputs, int):
        raise TypeError
    if number_of_inputs < 1 :
        raise ValueError
    
    if not isinstance(dimensions_of_inputs, int):
        raise TypeError
    if dimensions_of_inputs < 1 :
        raise ValueError
    
    if not isinstance(number_of_classes, int):
        raise TypeError
    if number_of_classes < 1 :
        raise ValueError

    if not isinstance(inputs, np.ndarray):
        raise TypeError
    inputs_as_c_float_array = (ctypes.c_float * (number_of_inputs * dimensions_of_inputs))(*inputs)
    pointer_to_inputs = ctypes.cast(inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(model, list) :
        model = np.array(model)
    if not isinstance(model, np.ndarray):
        raise TypeError
    number_of_weights = len(model)
    if number_of_weights != (dimensions_of_inputs + 1) * number_of_classes :
        raise ValueError
    model_as_c_float_array = (ctypes.c_float * number_of_weights)(*model)
    pointer_to_model = ctypes.cast(model_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    pointer_to_predicted_classes = rust_machine_learning_library.predict_with_linear_model(
                                                                                           pointer_to_model,
                                                                                           pointer_to_inputs,
                                                                                           number_of_inputs,
                                                                                           dimensions_of_inputs,
                                                                                           number_of_classes,
                                                                                           is_classification
                                                                                          )
    # rust_machine_learning_library.delete_float_array(pointer_to_model, number_of_weights)
    # rust_machine_learning_library.delete_float_array(pointer_to_inputs, number_of_inputs)

    predicted_classes = np.ctypeslib.as_array(pointer_to_predicted_classes, (number_of_inputs * number_of_classes, ))
    return predicted_classes


def generate_multi_layer_perceptron_model(layers_list: list[int]):
    layers = np.array(layers_list, dtype=ctypes.c_float)
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    number_of_layers = len(layers_as_c_float_array)
    pointer_to_untrained_mlp = rust_machine_learning_library.generate_multi_layer_perceptron_model(pointer_to_layers, number_of_layers)
    untrained_model = np.ctypeslib.as_array(pointer_to_untrained_mlp, (get_number_of_weights(layers_list),))
    return untrained_model


def train_multi_layer_perceptron_model(is_classification: bool,
                                       layers: Union[list[float], np.ndarray],
                                       training_inputs: Union[list[float], np.ndarray],
                                       tests_inputs: Union[list[float], np.ndarray],
                                       training_labels: Union[list[float], np.ndarray],
                                       tests_labels: Union[list[float], np.ndarray],
                                       model: Union[list[float], np.ndarray],
                                       learning_rate: float,
                                       number_of_epochs: float,
                                       number_of_training_inputs: int = 0,
                                       number_of_tests_inputs: int = 0,
                                       dimensions_of_inputs: int = 0,
                                       number_of_classes: int = 0,
                                       batch_size: int = 1):
    timer = time.time()
    
    if not isinstance(is_classification, bool):
        raise TypeError
    
    if not isinstance(number_of_training_inputs, int):
        raise TypeError
    if number_of_training_inputs < 1 :
        raise ValueError
    if not isinstance(number_of_tests_inputs, int):
        raise TypeError
    if number_of_tests_inputs < 1 :
        raise ValueError
    
    if not isinstance(dimensions_of_inputs, int):
        raise TypeError
    if dimensions_of_inputs < 1 :
        raise ValueError
    
    if not isinstance(number_of_classes, int):
        raise TypeError
    if number_of_classes < 1 :
        raise ValueError
    
    if isinstance(layers, list):
        layers = np.array(layers, dtype=ctypes.c_float)
    if not isinstance(layers, np.ndarray):
        raise TypeError
    number_of_layers = len(layers)
    if number_of_layers < 2 :
        raise ValueError
    if dimensions_of_inputs != layers[0] :
        raise ValueError
    if number_of_classes != layers[-1] :
        raise ValueError
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if not isinstance(training_inputs, np.ndarray):
        raise TypeError
    if number_of_training_inputs * dimensions_of_inputs != len(training_inputs):
        raise ValueError
    training_inputs_as_c_float_array = (ctypes.c_float * (number_of_training_inputs * dimensions_of_inputs))(*training_inputs)
    pointer_to_training_inputs = ctypes.cast(training_inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    if not isinstance(tests_inputs, np.ndarray):
        raise TypeError
    if number_of_tests_inputs * dimensions_of_inputs != len(tests_inputs):
        raise ValueError
    tests_inputs_as_c_float_array = (ctypes.c_float * (number_of_tests_inputs * dimensions_of_inputs))(*tests_inputs)
    pointer_to_tests_inputs = ctypes.cast(tests_inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if not isinstance(training_labels, np.ndarray):
        raise TypeError
    if number_of_training_inputs * number_of_classes != len(training_labels):
        raise ValueError
    training_labels_as_c_float_array = (ctypes.c_float * (number_of_training_inputs * number_of_classes))(*training_labels)
    pointer_to_training_labels = ctypes.cast(training_labels_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    if not isinstance(tests_labels, np.ndarray):
        raise TypeError
    if number_of_tests_inputs * number_of_classes != len(tests_labels):
        raise ValueError
    tests_labels_as_c_float_array = (ctypes.c_float * (number_of_tests_inputs * number_of_classes))(*tests_labels)
    pointer_to_tests_labels = ctypes.cast(tests_labels_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(model, list):
        model = np.array(model, dtype=ctypes.c_float)
    if not isinstance(model, np.ndarray):
        raise TypeError
    number_of_weights = len(model)
    if number_of_weights != get_number_of_weights(layers):
        raise ValueError
    model_as_c_float_array = (ctypes.c_float * number_of_weights)(*model)
    pointer_to_model = ctypes.cast(model_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(learning_rate, float):
        learning_rate = ctypes.c_float(learning_rate)
    if not isinstance(learning_rate, ctypes.c_float):
        raise TypeError
    
    if isinstance(number_of_epochs, int):
        number_of_epochs = ctypes.c_int32(number_of_epochs)
    if not isinstance(number_of_epochs, ctypes.c_int32):
        raise TypeError

    number_of_training_inputs = ctypes.c_int32(number_of_training_inputs)
    number_of_tests_inputs = ctypes.c_int32(number_of_tests_inputs)
    dimensions_of_inputs = ctypes.c_int32(dimensions_of_inputs)
    number_of_classes = ctypes.c_int32(number_of_classes)
    print("Training...")
    print(time.time() - timer)
    pointer_to_trained_model = rust_machine_learning_library.train_multi_layer_perceptron_model(
                                                                                                pointer_to_model,
                                                                                                pointer_to_layers,
                                                                                                number_of_layers,
                                                                                                pointer_to_training_inputs,
                                                                                                number_of_training_inputs,
                                                                                                pointer_to_tests_inputs,
                                                                                                number_of_tests_inputs,
                                                                                                dimensions_of_inputs,
                                                                                                pointer_to_training_labels,
                                                                                                pointer_to_tests_labels,
                                                                                                number_of_classes,
                                                                                                learning_rate,
                                                                                                number_of_epochs,
                                                                                                batch_size,
                                                                                                is_classification
                                                                                               )
    # rust_machine_learning_library.delete_float_array(pointer_to_model, number_of_weights)
    # rust_machine_learning_library.delete_float_array(pointer_to_layers, number_of_layers)
    # rust_machine_learning_library.delete_float_array(pointer_to_training_inputs, number_of_training_inputs)
    # rust_machine_learning_library.delete_float_array(pointer_to_tests_inputs, number_of_tests_inputs)
    # rust_machine_learning_library.delete_float_array(pointer_to_training_labels, number_of_training_inputs)
    # rust_machine_learning_library.delete_float_array(pointer_to_tests_labels, number_of_tests_inputs)
    print("The model has been trained.")
    print(time.time() - timer)
    trained_model = np.ctypeslib.as_array(pointer_to_trained_model, (number_of_weights,))
    return trained_model
    

def predict_with_multi_layer_perceptron_model(is_classification: bool,
                                              layers: Union[list[float], np.ndarray],
                                              inputs: Union[list[float], np.ndarray],
                                              model: Union[list[float], np.ndarray],
                                              number_of_inputs: int = 0,
                                              dimensions_of_inputs: int = 0,
                                              number_of_classes: int = 0):
    
    if not isinstance(is_classification, bool):
        raise TypeError
    
    if not isinstance(number_of_inputs, int):
        raise TypeError
    if number_of_inputs < 1 :
        raise ValueError
    
    if not isinstance(dimensions_of_inputs, int):
        raise TypeError
    if dimensions_of_inputs < 1 :
        raise ValueError
    
    if not isinstance(number_of_classes, int):
        raise TypeError
    if number_of_classes < 1 :
        raise ValueError
    
    
    if isinstance(layers, list):
        layers = np.array(layers, dtype=ctypes.c_float)
    if not isinstance(layers, np.ndarray):
        raise TypeError
    number_of_layers = len(layers)
    if number_of_layers < 2 :
        raise ValueError
    if dimensions_of_inputs != layers[0] :
        raise ValueError
    number_of_classes = int(layers[-1])
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if not isinstance(inputs, np.ndarray):
        raise TypeError
    inputs_as_c_float_array = (ctypes.c_float * (number_of_inputs * dimensions_of_inputs))(*inputs)
    pointer_to_inputs = ctypes.cast(inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(model, list) :
        model = np.array(model)
    if not isinstance(model, np.ndarray):
        raise TypeError
    number_of_weights = len(model)
    if number_of_weights != get_number_of_weights(layers):
        raise ValueError
    model_as_c_float_array = (ctypes.c_float * number_of_weights)(*model)
    pointer_to_model = ctypes.cast(model_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    pointer_to_predicted_classes = rust_machine_learning_library.predict_with_multi_layer_perceptron_model(
                                                                                                           pointer_to_model,
                                                                                                           pointer_to_layers,
                                                                                                           number_of_layers,
                                                                                                           pointer_to_inputs,
                                                                                                           number_of_inputs,
                                                                                                           dimensions_of_inputs,
                                                                                                           number_of_classes,
                                                                                                           is_classification
                                                                                                          )
    # rust_machine_learning_library.delete_float_array(pointer_to_model, number_of_weights)
    # rust_machine_learning_library.delete_float_array(pointer_to_layers, number_of_layers)
    # rust_machine_learning_library.delete_float_array(pointer_to_inputs, number_of_inputs)

    predicted_classes = np.ctypeslib.as_array(pointer_to_predicted_classes, (number_of_inputs * number_of_classes, ))
    return predicted_classes

