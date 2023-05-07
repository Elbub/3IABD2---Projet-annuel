from ctypes import CDLL, c_size_t, POINTER
import numpy as np

if __name__ == "__main__":
    my_lib = CDLL(r"rust_lib\target\debug\rust_lib.dll")

    my_lib.hello_world()

    my_lib.points_array.restype = POINTER(POINTER(c_size_t))

    num_points = 10
    points_ptr = my_lib.points_array(num_points)

    # print(type(points_ptr))

    points = np.ctypeslib.as_array(points_ptr.contents, shape=(num_points, 2))
    # print(points_ptr.contents)
    print("testestest")
    print(len(points))

    # my_lib.free_points_array(points_ptr)
