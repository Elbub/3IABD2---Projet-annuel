import ctypes
import numpy as np

my_lib = ctypes.CDLL(r"..\rust_lib\target\debug\rust_lib.dll")
# my_lib.hello_world()
# my_lib.points_array(10)

# my_lib.array_test.argtypes = [ctypes.c_usi]
my_lib.array_test.restype = ctypes.POINTER(ctypes.c_int32)

my_lib.points_array.restype = ctypes.POINTER(ctypes.c_float)
my_lib.another_points_array.restype = ctypes.POINTER(ctypes.c_float)

n = 5
# o = 2

# native_pointer = my_lib.array_test(n, o)
# arr = np.ctypeslib.as_array(native_pointer, (n, o))
# print(arr)

my_lib.points_array.restype = ctypes.c_void_p


ptr = my_lib.points_array(5)

array_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

print(array_ptr[0])

for i in range(5):
    inner_ptr = array_ptr[i]
    point = ctypes.cast(inner_ptr, ctypes.POINTER(ctypes.c_float * 2)).contents
    x = point[0]
    y = point[1]
    print(f"Point {i}: x = {x}, y = {y}")
