import ctypes
import numpy as np

my_lib = ctypes.CDLL(r"..\rust_lib\target\debug\rust_lib.dll")
# my_lib.hello_world()
# my_lib.points_array(10)

# my_lib.array_test.argtypes = [ctypes.c_usi]
my_lib.array_test.restype = ctypes.POINTER(ctypes.c_int32)

n = 5
o = 2

native_pointer = my_lib.array_test(n, o)
arr = np.ctypeslib.as_array(native_pointer, (n, o))
print(arr)
