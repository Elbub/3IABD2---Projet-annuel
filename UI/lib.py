import matplotlib.pyplot as plt
import numpy as np
import ctypes
from typing import *
import random
from PIL import Image
import os
import time

# rust_machine_learning_library = ctypes.CDLL(r"C:\Users\Moi\Desktop\Projets\3IABD2 - Projet annuel \rust_lib\target\debug\rust_lib.dll")
rust_machine_learning_library = ctypes.CDLL(r"C:\Users\Moi\Desktop\Projets\3IABD2 - Projet annuel\rust_lib\target\release\rust_lib.dll")
POINTER_TO_FLOAT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_float)
POINTER_TO_INT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_int32)


rust_machine_learning_library.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]
rust_machine_learning_library.delete_float_array.restype = None

rust_machine_learning_library.generate_multi_layer_perceptron_model.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
rust_machine_learning_library.generate_multi_layer_perceptron_model.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.train_multi_layer_perceptron_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_float,
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


def read_dataset(dataset_folders: Union[str, List[str]]):
    if isinstance(dataset_folders, str) :
        dataset_folders = [dataset_folders]
    inputs = []
    labels = []
    number_of_inputs = 0
    for dataset_class in range(len(dataset_folders)) :
        try :
            dataset_folder = os.listdir(dataset_folders[dataset_class])
            for filename in dataset_folder:
                if 'image' in filename:
                    with Image.open(os.path.join(dataset_folders[dataset_class], filename)) as filename :
                        # print(filename)
                        inputs.append(np.asarray(filename))
                        number_of_inputs += 1
                        labels.append([1. if index == dataset_class else -1 for index in range(len(dataset_folders))])
        except FileNotFoundError :
            print("Wrong path")
            raise
    inputs = np.array(inputs, dtype=ctypes.c_float).flatten()  / 255 * 2 - 1
    # inputs = np.array([input / 255 * 2 - 1 for input in inputs])
    labels = np.array(labels, dtype=ctypes.c_float).flatten()
    return (inputs, number_of_inputs, labels)


def resize_images():
    pass


def get_number_of_weights(layers_list):
    number_of_weights = 0
    for index in range(len(layers_list) - 1):
        number_of_weights += (layers_list[index] + 1) * layers_list[index + 1]
    return number_of_weights


def generate_multi_layer_perceptron_model(layers_list: List[int]):
    layers = np.array(layers_list, dtype=ctypes.c_float)
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    number_of_layers = len(layers_as_c_float_array)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    pointer_to_untrained_mlp = rust_machine_learning_library.generate_multi_layer_perceptron_model(pointer_to_layers, number_of_layers)
    untrained_mlp = np.ctypeslib.as_array(pointer_to_untrained_mlp, (get_number_of_weights(layers_list),))
    return untrained_mlp


def train_multi_layer_perceptron_model(is_classification: bool,
                                       layers: Union[List[float], np.ndarray],
                                       inputs: Union[List[float], np.ndarray],
                                       labels: Union[List[float], np.ndarray],
                                       model: Union[List[float], np.ndarray],
                                       learning_rate: float,
                                       number_of_epochs: float,
                                       number_of_inputs: int = 0,
                                       dimensions_of_inputs: int = 0,
                                       number_of_classes: int = 0):
    timer = time.time()
    
    if not isinstance(is_classification, bool):
        raise TypeError
    
    if isinstance(layers, List):
        layers = np.array(layers, dtype=ctypes.c_float)
    if not isinstance(layers, np.ndarray):
        raise TypeError
    number_of_layers = len(layers)
    if number_of_layers < 2 :
        raise ValueError
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(inputs, List):
        inputs = np.array(inputs, dtype=ctypes.c_float)
    if not isinstance(inputs, np.ndarray):
        raise TypeError
    if len(inputs.shape) > 1 :
        number_of_inputs = len(inputs)
        if number_of_inputs < 1 :
            raise ValueError
        dimensions_of_inputs = len(inputs[0].flatten())
        if dimensions_of_inputs < 1 :
            raise ValueError
        if dimensions_of_inputs != layers[0] :
            raise ValueError
        inputs = inputs.flatten()
    else :
        print(number_of_inputs, '*', dimensions_of_inputs)
        print(len(inputs))
        if number_of_inputs * dimensions_of_inputs != len(inputs):
            raise ValueError
        number_of_inputs = ctypes.c_int32(number_of_inputs)
        dimensions_of_inputs = ctypes.c_int32(dimensions_of_inputs)
    # print(inputs)
    # print(len(inputs))
    # print(number_of_inputs,  dimensions_of_inputs, number_of_inputs * dimensions_of_inputs)
    inputs_as_c_float_array = (ctypes.c_float * (number_of_inputs * dimensions_of_inputs))(*inputs)
    pointer_to_inputs = ctypes.cast(inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(labels, List):
        labels = np.array(labels, dtype=ctypes.c_float)
    if not isinstance(labels, np.ndarray):
        raise TypeError
    if len(labels.shape) > 1 :
        number_of_labels = len(labels)
        if number_of_labels < 1 :
            raise ValueError
        if number_of_labels != number_of_inputs :
            raise ValueError
        number_of_classes = len(labels[0])
        if number_of_classes < 1 :
            raise ValueError
        if number_of_classes != layers[-1] :
            raise ValueError
        labels = labels.flatten()
    else :
        number_of_classes = ctypes.c_int32(number_of_classes)
        if number_of_inputs * number_of_classes != len(labels):
            raise ValueError
    labels_as_c_float_array = (ctypes.c_float * (number_of_labels * number_of_classes))(*labels)
    pointer_to_labels = ctypes.cast(labels_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
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
    print(time.time() - timer)
    print("Training...")
    pointer_to_trained_model = rust_machine_learning_library.train_multi_layer_perceptron_model(
                                                                                                pointer_to_model,
                                                                                                pointer_to_layers,
                                                                                                number_of_layers,
                                                                                                pointer_to_inputs,
                                                                                                number_of_inputs,
                                                                                                dimensions_of_inputs,
                                                                                                pointer_to_labels,
                                                                                                number_of_classes,
                                                                                                learning_rate,
                                                                                                number_of_epochs,
                                                                                                is_classification
                                                                                               )
    print("The model has been trained.")
    print(time.time() - timer)
    trained_model = np.ctypeslib.as_array(pointer_to_trained_model, (number_of_weights,))
    return trained_model
    

def predict_with_multi_layer_perceptron_model(is_classification: bool,
                                              layers: Union[List[float], np.ndarray],
                                              inputs: Union[List[float], np.ndarray],
                                              model: Union[List[float], np.ndarray]):
    
    if not isinstance(is_classification, bool):
        raise TypeError
    
    if isinstance(layers, List):
        layers = np.array(layers, dtype=ctypes.c_float)
    if not isinstance(layers, np.ndarray):
        raise TypeError
    number_of_layers = len(layers)
    if number_of_layers < 2 :
        raise ValueError
    number_of_classes = int(layers[-1])
    if number_of_classes < 1 :
        raise ValueError
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(inputs, List):
        inputs = np.array(inputs, dtype=ctypes.c_float)
    if not isinstance(inputs, np.ndarray):
        raise TypeError
    number_of_inputs = len(inputs)
    if number_of_inputs < 1 :
        raise ValueError
    dimensions_of_inputs = len(inputs[0].flatten())
    if dimensions_of_inputs < 1 :
        raise ValueError
    if dimensions_of_inputs != layers[0] :
        raise ValueError
    inputs = inputs.flatten()
    inputs_as_c_float_array = (ctypes.c_float * (number_of_inputs * dimensions_of_inputs))(*inputs)
    pointer_to_inputs = ctypes.cast(inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
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

    predicted_classes = np.ctypeslib.as_array(pointer_to_predicted_classes, (number_of_inputs * number_of_classes, ))
    return predicted_classes

