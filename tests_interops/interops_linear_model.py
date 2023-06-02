import ctypes
import _ctypes
import numpy as np
import random

my_lib = ctypes.CDLL(r"..\rust_lib\target\debug\rust_lib.dll")

my_lib.points_array.argtypes = [ctypes.c_int32, ctypes.c_int32]

my_lib.points_array.restype = ctypes.POINTER(ctypes.c_float)

my_lib.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]
my_lib.delete_float_array.restype = None

my_lib.generate_random_w.argtypes = [ctypes.c_int32]
my_lib.generate_random_w.restype = ctypes.POINTER(ctypes.c_float)

my_lib.points_label.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]
my_lib.points_label.restype = ctypes.POINTER(ctypes.c_float)

my_lib.linear_model_training.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_int32,
]

my_lib.linear_model_training.restype = ctypes.POINTER(ctypes.c_float)

n = 200
dimension = 3
native_pointer = my_lib.points_array(n, dimension)

print(f"1. native_pointer = {native_pointer}")
arr = np.ctypeslib.as_array(native_pointer, ((n * dimension),))
print("voici as_array: ")
print(arr)
print(f"2. native_pointer = {native_pointer}")
print("Nombre d'éléments de as_array: ")
print(len(arr))
print(f"3. native_pointer = {native_pointer}")


def create_n_dimension_array(arr_dimension, my_array):
    my_points_array = []
    second_array = []
    count = 0
    for value in my_array:
        second_array.append(value)
        count += 1
        if count == arr_dimension:
            count = 0
            my_points_array.append(second_array)
            second_array = []
    return my_points_array


points_array = create_n_dimension_array(dimension, arr)

print("Voici le vecteur applati : ")
print(points_array)
print(f"Nous avons donc maintenant un vecteur contenant {len(points_array)} vecteurs")


print(f"n*dimension = {n * dimension} ; dimension = {dimension}")
print(f"4. native_pointer = {native_pointer}")
native_label_pointer = my_lib.points_label(native_pointer, n, dimension)
print(f"5. native_pointer = {native_pointer}")
print(f"Ceci est native_label_pointer: {native_label_pointer}")

label_arr = np.ctypeslib.as_array(native_label_pointer, (n,))

print(f"6. native_pointer = {native_pointer}")
print(f"label_arr= {label_arr}")

w_array_ptr = my_lib.generate_random_w(dimension)

w_array = np.ctypeslib.as_array(w_array_ptr, ((dimension + 1),))

print(f"this is from generate_random_w :{w_array}")

learning_rate = 0.001
epoch = 100_000
linear_model_training_ptr = my_lib.linear_model_training(
    w_array_ptr,
    native_label_pointer,
    native_pointer,
    n,
    dimension,
    learning_rate,
    epoch,
)

trained_linear_model = np.ctypeslib.as_array(
    linear_model_training_ptr, ((dimension + 1),)
)
print(trained_linear_model)
# print("FFFFFFFFFFFFFFFFFFFFFF")
# print(len(trained_linear_model))

my_lib.predict_linear_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]

my_lib.predict_linear_model.restype = ctypes.POINTER(ctypes.c_float)

LP_c_float = ctypes.POINTER(ctypes.c_float)

arr_to_predict = []

for i in range(3000):
    arr_to_predict.append(random.random())
# print("this is arr_to_predict\n")
# print(len(arr_to_predict))

arr_to_predict_c = (ctypes.c_float * len(arr_to_predict))(*arr_to_predict)
arr_to_predict_c_ptr = ctypes.cast(arr_to_predict_c, LP_c_float)

predict_linear_model_ptr = my_lib.predict_linear_model(
    arr_to_predict_c_ptr,
    # arr_to_predict.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    linear_model_training_ptr,
    1000,
    dimension,
)

predict_linear_model = np.ctypeslib.as_array(predict_linear_model_ptr, (1000,))

print(len(predict_linear_model))

print(predict_linear_model)

my_lib.delete_float_array(native_pointer, (n * dimension))
my_lib.delete_float_array(w_array_ptr, (dimension + 1))
my_lib.delete_float_array(native_label_pointer, n)
# TO DO: Problème lors du delete de linear_model_training_ptr
# my_lib.delete_float_array(linear_model_training_ptr, (dimension + 1))
my_lib.delete_float_array(predict_linear_model_ptr, n)
del native_pointer
del w_array_ptr
del native_label_pointer
del linear_model_training_ptr
del predict_linear_model_ptr
