import matplotlib.pyplot as plt
import numpy as np
import ctypes
from typing import String, List, Union, Any, Callable, int
import random
from PIL import Image
import os

rust_machine_learning_library = ctypes.CDLL(r".\rust_lib\target\debug\rust_lib.dll")
POINTER_TO_FLOAT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_float)
POINTER_TO_INT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_int)


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
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_bool,
]
rust_machine_learning_library.train_multi_layer_perceptron_model.restype = ctypes.POINTER(ctypes.c_float)

rust_machine_learning_library.predict_multi_layer_perceptron_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_bool,
]
rust_machine_learning_library.predict_multi_layer_perceptron_model.restype = ctypes.POINTER(ctypes.c_float)


def read_dataset(dataset_folders: Union(String, List[String])):
    if isinstance(dataset_folders, String) :
        dataset_folders = [dataset_folders]
    X = []
    Y = []
    for dataset_class in len(range(dataset_folders)) :
        for filename in os.listdir(dataset_folders[dataset_class]):
            if 'image' in filename:
                with Image.open(os.join(dataset_folders[dataset_class], filename)) as filename :
                    # print(filename)
                    X.append(np.asarray(filename))
                    Y.append([1. if index == dataset_class else -1 for index in len(range(dataset_folders))])


def resize_images():
    pass


def get_number_of_weights(layers_list):
    number_of_weights = 0
    for index in range(len(layers_list) - 1):
        number_of_weights += (layers_list[index] + 1) * layers_list[index]
    return number_of_weights


def generate_multi_layer_perceptron_model(layers_list: List[int]):
    layers = np.array(layers_list, dtype=ctypes.c_float)
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    number_of_layers = len(layers_as_c_float_array)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    pointer_to_untrained_mlp = rust_machine_learning_library.generate_multi_layer_perceptron_model(pointer_to_layers, number_of_layers)
    untrained_mlp = np.ctypeslib.as_array(pointer_to_untrained_mlp, (get_number_of_weights(layers_list),))
    return untrained_mlp


def train_multi_layer_perceptron_model(model: Union(List[float], np.array),
                                       layers: Union(List[float], np.array),
                                       inputs: Union(List[float], np.array),
                                       labels: Union(List[float], np.array),
                                       learning_rate: float,
                                       number_of_epochs: float,
                                       is_classification: bool):
    if isinstance(layers, List):
        layers = np.array(layers, dtype=ctypes.c_float)
    if not isinstance(layers, np.array):
        raise ValueError
    layers_as_c_float_array = (ctypes.c_float * len(layers))(*layers)
    number_of_layers = len(layers_as_c_float_array)
    pointer_to_layers = ctypes.cast(layers_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(inputs, List):
        inputs = np.array(inputs, dtype=ctypes.c_float)
    if not isinstance(inputs, np.array):
        raise ValueError
    inputs_as_c_float_array = (ctypes.c_float * len(inputs))(*inputs)
    number_of_inputs = len(inputs_as_c_float_array)
    pointer_to_inputs = ctypes.cast(inputs_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if isinstance(labels, List):
        labels = np.array(labels, dtype=ctypes.c_float)
    if not isinstance(labels, np.array):
        raise ValueError
    labels_as_c_float_array = (ctypes.c_float * len(labels))(*labels)
    number_of_labels = len(labels_as_c_float_array)
    pointer_to_labels = ctypes.cast(labels_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    
    if not isinstance(model, np.array):
        raise ValueError
    model_as_c_float_array = (ctypes.c_float * len(model))(*model)
    number_of_model = len(model_as_c_float_array)
    pointer_to_model = ctypes.cast(model_as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
    


def predict_with_mlp() :
    pass




if __name__ == "__main__" :
    mlp = generate_multi_layer_perceptron_model([2, 2, 1])
    print(mlp)
    
    directory_inputs = ["../database/resized_dataset/asie_sud_est","../database/resized_dataset/rome_grece"]
    # directory_inputs = ["../database/resized_dataset/amerique_sud","../database/resized_dataset/asie_sud_est","../database/resized_dataset/rome_grece"]