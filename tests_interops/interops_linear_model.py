import ctypes
import _ctypes
import numpy as np

my_lib = ctypes.CDLL(r"..\rust_lib\target\debug\rust_lib.dll")

my_lib.points_array.argtypes = [ctypes.c_int32, ctypes.c_int32]

my_lib.points_array.restype = ctypes.POINTER(ctypes.c_float)

my_lib.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]
my_lib.delete_float_array.restype = None

n = 10
dimension = 2
native_pointer = my_lib.points_array(n, dimension)

arr = np.ctypeslib.as_array(native_pointer, ((n * dimension),))
print(arr)

my_lib.delete_float_array(native_pointer, n * dimension)

print(len(arr))


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

print(points_array)
print(len(points_array))

my_lib.points_label.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]
my_lib.points_label.restype = ctypes.POINTER(ctypes.c_float)

native_label_pointer = my_lib.points_label(native_pointer, (n * dimension), dimension)

label_arr = np.ctypeslib.as_array(native_label_pointer, (n,))

# print(label_arr)
