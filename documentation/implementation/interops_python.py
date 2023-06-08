import ctypes
import numpy as np

if __name__ == "__main__":
    # my_lib = ctypes.CDLL(r"..\2023_3A_IABD1_Demo_Interop_Cpp\cmake-build-debug\2023_3A_IABD1_Demo_Interop_Cpp.dll")
    my_lib = ctypes.CDLL(
        r"..\_2023_3A_IABD1_Demo_Interop_Rust\target\debug\_2023_3A_IABD1_Demo_Interop_Rust.dll"
    )

    my_lib.my_add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    my_lib.my_add.restype = ctypes.c_int32

    print(my_lib.my_add(2, 3))

    l = np.array([1, 2, 3])
    l_size = len(l)

    my_lib.my_sum.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_lib.my_sum.restype = ctypes.c_int32

    print(my_lib.my_sum(l.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), l_size))

    my_lib.count_to_n.argtypes = [ctypes.c_int32]
    my_lib.count_to_n.restype = ctypes.POINTER(ctypes.c_int32)

    my_lib.delete_int_array.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_lib.delete_int_array.restype = None

    my_lib.delete_float_array.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    my_lib.delete_float_array.restype = None

    n = 10
    native_pointer = my_lib.count_to_n(n)

    arr = np.ctypeslib.as_array(native_pointer, (n,))
    print(arr)

    my_lib.delete_int_array(native_pointer, n)

    npl = np.array([2, 3, 1])
    npl_size = len(npl)

    my_lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_lib.create_mlp_model.restype = ctypes.c_void_p

    sample_inputs = np.array([1.0, 8.0])
    sample_inputs_size = len(sample_inputs)

    my_lib.predict_mlp_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    my_lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_float)

    all_training_inputs = np.array([[1.0, 8.0], [4.0, 2.0], [5.0, 6.0]]).flatten()
    all_training_inputs_size = len(all_training_inputs)

    all_training_expected_outputs = np.array([[-1.0], [-1.0], [1.0]]).flatten()
    all_training_expected_outputs_size = len(all_training_expected_outputs)

    my_lib.train_mlp_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    my_lib.train_mlp_model.restype = None

    my_lib.delete_mlp_model.argtypes = [ctypes.c_void_p]
    my_lib.delete_mlp_model.restype = None

    # let's loop on methods to ensure no memory leak
    while True:
        model_pointer = my_lib.create_mlp_model(
            npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), npl_size
        )

        native_predict_before_training = my_lib.predict_mlp_model(
            model_pointer,
            sample_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sample_inputs_size,
            True,
        )

        print(
            f"Prediction before training : {np.ctypeslib.as_array(native_predict_before_training, (npl[-1],))}"
        )
        my_lib.delete_float_array(native_predict_before_training, npl[-1])

        my_lib.train_mlp_model(
            model_pointer,
            all_training_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            all_training_inputs_size // npl[0],
            npl[0],
            all_training_expected_outputs.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            ),
            npl[-1],
            0.1,
            100,
            True,
        )

        native_predict_after_training = my_lib.predict_mlp_model(
            model_pointer,
            sample_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sample_inputs_size,
            True,
        )
        print(
            f"Prediction after training : {np.ctypeslib.as_array(native_predict_after_training, (npl[-1],))}"
        )
        my_lib.delete_float_array(native_predict_after_training, npl[-1])

        my_lib.delete_mlp_model(model_pointer)

    # my_lib.train_mlp_model(model_pointer, np.ctypeslib., all_training_expected_outputs_size // npl[0], npl[0],
    #                        all_training_expected_outputs_type(*all_training_expected_outputs), npl[-1], 0.1, 100, True)
