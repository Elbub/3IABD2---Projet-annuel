import matplotlib.pyplot as plt
import numpy as np
import ctypes
import random


rust_machine_learning_library = ctypes.CDLL(r".\rust_lib\target\debug\rust_lib.dll")
POINTER_TO_FLOAT_ARRAY_TYPE = ctypes.POINTER(ctypes.c_float)


rust_machine_learning_library.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]

rust_machine_learning_library.delete_float_array.restype = None

def number_of_weights_in_mlp(layers_list):
   number_of_weights = 0
   for index in range(len(layers_list) - 1):
      number_of_weights += (layers_list[index] + 1) * layers_list[index]
   return number_of_weights

def generate_untrained_mlp(layers_list=[]):
   # rust_machine_learning_library.generate_untrained_mlp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
   # rust_machine_learning_library.generate_untrained_mlp.restype = ctypes.POINTER(ctypes.c_float)
   rust_machine_learning_library.generate_random_mpl_w.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
   rust_machine_learning_library.generate_random_mpl_w.restype = ctypes.POINTER(ctypes.c_float)
   
   layers = np.array(layers_list, dtype=ctypes.c_float)
   layers__as_c_float_array = (ctypes.c_float * len(layers))(*layers)
   number_of_layers = len(layers__as_c_float_array)
   pointer_to_layers = ctypes.cast(layers__as_c_float_array, POINTER_TO_FLOAT_ARRAY_TYPE)
   pointer_to_untrained_mlp = rust_machine_learning_library.generate_random_mpl_w(pointer_to_layers, number_of_layers)
   untrained_mlp = np.ctypeslib.as_array(pointer_to_untrained_mlp, (number_of_weights_in_mlp(layers_list),))
   return untrained_mlp
   
   
mlp = generate_untrained_mlp([2, 2, 1])
print(mlp)