import ctypes
import _ctypes
import numpy as np

my_lib = ctypes.CDLL(r"..\rust_lib\target\debug\rust_lib.dll")

my_lib.points_array.argtypes = [ctypes.c_int32, ctypes.c_int32]

my_lib.points_array.restype = ctypes.POINTER(ctypes.c_float)

my_lib.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    # ctypes.c_int32,
]
my_lib.delete_float_array.restype = None

n = 5
dimension = 3
native_pointer = my_lib.points_array(n, dimension)

print(f"1. native_pointer = {native_pointer}")
arr = np.ctypeslib.as_array(native_pointer, ((n * dimension),))
print("voici as_array: ")
print(arr)
print(f"2. native_pointer = {native_pointer}")
my_lib.delete_float_array(native_pointer, n * dimension)
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


my_lib.points_label.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]
my_lib.points_label.restype = ctypes.POINTER(ctypes.c_float)
print(f"n*dimension = {n * dimension} ; dimension = {dimension}")
print(f"4. native_pointer = {native_pointer}")
native_label_pointer = my_lib.points_label(native_pointer, (n * dimension), dimension)
print(f"5. native_pointer = {native_pointer}")
# print(f"Ceci est native_label_pointer: {native_label_pointer}")
label_arr = np.ctypeslib.as_array(native_label_pointer, (n,))
print(f"6. native_pointer = {native_pointer}")
print(f"label_arr= {label_arr}")
