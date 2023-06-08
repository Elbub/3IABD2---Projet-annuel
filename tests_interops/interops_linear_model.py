import ctypes
import numpy as np
import random
import matplotlib.pyplot as plt

# import du dll
my_lib = ctypes.CDLL(r"..\rust_lib\target\debug\rust_lib.dll")

# définition des types d'entrée et de sortie de points array
my_lib.points_array.argtypes = [ctypes.c_int32, ctypes.c_int32]
my_lib.points_array.restype = ctypes.POINTER(ctypes.c_float)

# définition des types d'entrée et de sortie de delete float array
my_lib.delete_float_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
]
my_lib.delete_float_array.restype = None

# définition des types d'entrée et de sortie de generate_random_w
my_lib.generate_random_w.argtypes = [ctypes.c_int32]
my_lib.generate_random_w.restype = ctypes.POINTER(ctypes.c_float)

# définition des types d'entrée et de sortie de label_points
my_lib.points_label.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]
my_lib.points_label.restype = ctypes.POINTER(ctypes.c_float)

# définition des types d'entrée et de sortie de linear-model-training
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

# n est le nombre de points
n = 200
# dimension est le nombre de coordonnées par points, c'est à dire la dimension du vecteur associé au point
dimension = 3

# pointeur vers le dataset créé en rust
native_pointer = my_lib.points_array(n, dimension)

# transformation python des vecteurs créés en rust
arr = np.ctypeslib.as_array(native_pointer, ((n * dimension),))


# fonction pour créer un array de dimension n, pour applatir une liste de vecteurs
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
native_label_pointer = my_lib.points_label(native_pointer, n, dimension)
label_arr = np.ctypeslib.as_array(native_label_pointer, (n,))

# génération d'un vecteur w aléatoire
w_array_ptr = my_lib.generate_random_w(dimension)
w_array = np.ctypeslib.as_array(w_array_ptr, ((dimension + 1),))


learning_rate = 0.001
epoch = 100_000

# création du pointeur pour entrainement du model linéaire
linear_model_training_ptr = my_lib.linear_model_training(
    w_array_ptr,  # pointeur vers le résultat du vecteur W random
    native_label_pointer,  # pointeur vers les résultats labellisés pour entrainer le modele
    native_pointer,  # points qui ont permis de labeliser le vecteur d'au dessus
    n,  # array size
    dimension,  # dimension de chaque vecteur
    learning_rate,  # learning rate
    epoch,  # nombre de fois où l'on passe sur le data set
)

# utilisation du pointeur pour entrainement du modele
trained_linear_model = np.ctypeslib.as_array(
    linear_model_training_ptr, ((dimension + 1),)
)
print("DAYIZDGUAIZDGYUAZGDY")
print(trained_linear_model)

# définition des types d'entrée et de sortie de la prédiction du modele lineaire
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

arr_to_predict_c = (ctypes.c_float * len(arr_to_predict))(*arr_to_predict)
arr_to_predict_c_ptr = ctypes.cast(arr_to_predict_c, LP_c_float)

# creation du pointeur vers le modele prédit
predict_linear_model_ptr = my_lib.predict_linear_model(
    arr_to_predict_c_ptr,  # vecteur sur lequel appliquer l'entrainement pour labéliser ses outputs
    # arr_to_predict.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    linear_model_training_ptr,  # pointeur vers le modele entrainé
    1000,  # nombre de vecteurs sur lesquels on va prédire l'output
    dimension,  # dimension des vecteurs sur lesquels on va prédire l'output
)

# réponse de la prédiction du modele lineaire
predict_linear_model = np.ctypeslib.as_array(predict_linear_model_ptr, (1000,))

print(predict_linear_model)


# suppres
# my_lib.delete_float_array(native_pointer, (n * dimension))
my_lib.delete_float_array(w_array_ptr, (dimension + 1))
# my_lib.delete_float_array(native_label_pointer, n)
# TO DO: Problème lors du delete de linear_model_training_ptr
# my_lib.delete_float_array(linear_model_training_ptr, (dimension + 1))
my_lib.delete_float_array(predict_linear_model_ptr, n)
# del native_pointer
del w_array_ptr
# del native_label_pointer
del linear_model_training_ptr
del predict_linear_model_ptr
