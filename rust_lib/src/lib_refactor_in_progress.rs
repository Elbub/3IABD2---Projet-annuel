// mod multy_layer_perceptron;

use nalgebra::DMatrix;
use std::fs::File;
use std::io::{self, Write};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct AccuraciesAndLosses {
    number_of_training_inputs : usize,
    number_of_tests_inputs : usize,
    number_of_epochs : usize,
    batch_size : usize,
    numbers_of_errors_on_training_dataset : Vec<usize>,
    training_accuracies : Vec<f32>,
    training_losses : Vec<f32>,
    numbers_of_errors_on_tests_dataset : Vec<usize>,
    tests_accuracies : Vec<f32>,
    tests_losses : Vec<f32>,
}

#[no_mangle]
pub extern "C" fn points_array(number_of_points: usize, dimension: usize) -> *mut f32{
    use rand::Rng;
    let mut vec_of_points = Vec::with_capacity(number_of_points*dimension);
    for _ in 0..(number_of_points*dimension){
        let coordinates : f32 = rand::thread_rng().gen_range(0f32..1f32);
        vec_of_points.push(coordinates);
    }

    let arr_slice = vec_of_points.leak();

    arr_slice.as_mut_ptr()

}

#[no_mangle]
extern "C" fn delete_int_array(arr: *mut i32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}

#[no_mangle]
extern "C" fn delete_float_array(arr: *mut f32 , arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}

#[no_mangle]
extern "C" fn points_label(pointer_to_vec_of_points: *mut f32, arr_size: usize, arr_dimension: usize) -> *mut f32{

    unsafe {
        let mut vec_of_labels: Vec<f32> = Vec::with_capacity(arr_size);
        let vec_of_points = std::slice::from_raw_parts(pointer_to_vec_of_points,
                                                arr_size*arr_dimension);



        let mut to_compare: Vec<f32> = Vec::with_capacity(arr_dimension);
        let mut count = 0;

        for point in vec_of_points {

            to_compare.push(*point);
            count += 1;
            if count == arr_dimension{
                if -to_compare[1] + to_compare[0] + 0.25 >= 0. {
                    vec_of_labels.push(1f32);
                } else { vec_of_labels.push(0f32) }
                count = 0;
                to_compare.clear();
            }
        }


        let arr_slice = vec_of_labels.leak();

        arr_slice.as_mut_ptr()
    }
}

#[no_mangle]
extern "C" fn generate_random_w(dimension: usize) -> *mut f32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let number_of_parameters: usize = dimension;
    let mut w: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);   // w : contient les poids associés aux Xi
    for _ in 0..number_of_parameters + 1 {
        w.push(rng.gen_range(0f32..1f32)); // initialisation aléatoire entre 0 et 1
    }
    let arr_slice = w.leak();
    arr_slice.as_mut_ptr()
}

#[no_mangle]
extern "C" fn linear_model_training(pointer_to_model: *mut f32, pointer_to_labels : *mut f32, pointer_to_vec_of_points: *mut f32, arr_size: usize, arr_dimension: usize, learning_rate: f32, number_of_epochs: usize) -> *mut f32 {
    unsafe {
        use rand::Rng;
        let vec_of_points = std::slice::from_raw_parts(pointer_to_vec_of_points,
                                                       arr_size * arr_dimension);
        let labels = std::slice::from_raw_parts(pointer_to_labels,
                                                arr_size);
        let mut w = Vec::from_raw_parts(
            pointer_to_model, arr_dimension + 1, arr_dimension + 1);
        let mut rng = rand::thread_rng();
        for _ in 0..number_of_epochs {
            let k: usize = rng.gen_range(0..arr_size);
            let y_k: f32 = labels[k];
            let mut x_k: Vec<f32> = Vec::with_capacity(arr_dimension + 1);
            x_k.push(1f32);
            for i in 0..arr_dimension {
                x_k.push(vec_of_points[k * arr_dimension + i]);
            }
            let mut signal: f32 = 0f32;
            for i in 0..arr_dimension + 1 {
                signal += w[i] * x_k[i];
            }
            let mut g_x_k: f32 = 0.0; // on avait 0.0 pour des cas 0 ou 1 en output
            if signal >= 0f32 {
                g_x_k = 1f32;
            }
            for i in 0..arr_dimension + 1 {
                w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
            }
        }
        // else {
        //     for _ in 0..number_of_epochs {
        //         let k: usize = rng.gen_range(0..(arr_size/number_of_classes)) * number_of_classes;
        //
        //         let mut y_k: Vec<f32> = Vec::with_capacity(number_of_classes);
        //         for i in 0..number_of_classes {
        //             y_k.push(labels[k+i]);
        //         }
        //
        //         let mut x_k: Vec<f32> = Vec::with_capacity(arr_dimension + 1);
        //         x_k.push(1f32);
        //         for i in 0..arr_dimension {
        //             x_k.push(vec_of_points[k * arr_dimension + i]);
        //         }
        //         let mut signal: f32 = 0f32;
        //         for i in 0..arr_dimension + 1 {
        //             signal += w[i] * x_k[i];
        //         }
        //         let mut g_x_k: f32 = -1.0; // on avait 0.0 pour des cas 0 ou 1 en output
        //         if signal >= 0f32 {
        //             g_x_k = 1f32;
        //         }
        //         for i in 0..arr_dimension + 1 {
        //             w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
        //         }
        //     }
        // }

        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}
 
#[no_mangle]
extern "C" fn find_w_linear_regression(pointer_to_x: *mut f32, pointer_to_y: *mut f32, nombre_lignes_x: usize, nombre_colonnes_x: usize, nombre_lignes_y:usize, nombre_colonnes_y: usize) -> *mut f32 {
    unsafe {
        // let mut w = Vec::with_capacity(nombre_lignes_x_et_y);
        use nalgebra::*;
        use rand::Rng;

        let x_vect = std::slice::from_raw_parts(pointer_to_x, nombre_lignes_x * nombre_colonnes_x);
        let y_vect = std::slice::from_raw_parts(pointer_to_y, nombre_lignes_y * nombre_colonnes_y);
        let mut x_mat:DMatrix<f32> = DMatrix::zeros(nombre_lignes_x, nombre_colonnes_x + 1);
        let mut y_mat:DMatrix<f32> = DMatrix::zeros(nombre_lignes_y, nombre_colonnes_y);

        for i in 0..nombre_lignes_x {
            for j in 0..(nombre_colonnes_x+1) {
                if j == 0 {
                    x_mat[(i,j)] = 1.0;
                } else {
                    x_mat[(i, j)] = x_vect[i * nombre_colonnes_x + j - 1];
                }
            }
        }

        for i in 0..nombre_lignes_y {
            for j in 0..nombre_colonnes_y {
                y_mat[(i, j)] = y_vect[i * nombre_colonnes_y + j];
            }
        }


        let x_transpose:DMatrix<f32> = x_mat.clone().transpose();
        let mut x_t_mult_x = x_transpose.clone() * x_mat.clone();
        let det = x_t_mult_x.clone().determinant();

        if det == 0.0{
            let mut rng = rand::thread_rng();
            for i in 0..(nombre_colonnes_x+1) {
                for j in 0..(nombre_colonnes_x+1) {
                    x_t_mult_x[(i,j)] = x_t_mult_x[(i,j)] + rng.gen_range(-0.005..0.005);
                }
            }
        }



        let inv_x_t_x = x_t_mult_x.try_inverse();

        let inv_times_x_t = match inv_x_t_x {
            Some(inv) => inv * x_transpose.clone(),
            None => panic!("Non inversible"),
        };

        let result_matrix = inv_times_x_t * y_mat;

        let mut w: Vec<f32> = Vec::with_capacity(nombre_colonnes_x * nombre_colonnes_y + 1);

        for i in 0..nombre_colonnes_x+1 {
            for j in 0..nombre_colonnes_y {
                w.push(result_matrix[(i,j)]);
            }
        }

        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn predict_linear_model(pointer_to_vec_to_predict: *const f32, pointer_to_trained_model: *mut f32, arr_size: usize, arr_dimension: usize) -> *mut f32{
    unsafe{
        let vec_to_predict = std::slice::from_raw_parts(pointer_to_vec_to_predict,
                                                       arr_size*arr_dimension);


        let trained_model = std::slice::from_raw_parts(pointer_to_trained_model, arr_dimension + 1);

        let mut predicted_labels = Vec::with_capacity(arr_size);

        for i in 0..arr_size {
            let mut vec_of_coordinates = Vec::with_capacity(arr_dimension);
            for j in 0..arr_dimension{
                vec_of_coordinates.push(vec_to_predict[i*arr_dimension+j]);
            }
            if vec_of_coordinates[0] * trained_model[1] + vec_of_coordinates[1] * trained_model[2] + trained_model[0] >= 0.0 {
            // if vec_of_coordinates[0] * trained_model[1] + vec_of_coordinates[1] * trained_model[2] + vec_of_coordinates[2] * trained_model[3] + trained_model[0] >= 0.0 {
                    predicted_labels.push(1.0);
            } else {
                predicted_labels.push(0.0);
            }
        }
        predicted_labels.leak().as_mut_ptr()
    }
}

#[no_mangle]
extern "C" fn x_transpose_times_x(pointer_to_x: *mut f32, longueur_x: usize, colonnes_x: usize) -> *mut f32 {
    unsafe {
        use nalgebra::*;
        let x_vect = std::slice::from_raw_parts(pointer_to_x, longueur_x * colonnes_x);
        let mut x_mat:DMatrix<f32> = DMatrix::zeros(longueur_x, colonnes_x);

        for i in 0..(longueur_x) {
            for j in 0..(colonnes_x) {
                x_mat[(i, j)] = x_vect[i * colonnes_x + j];
            }
        }

        let x_transpose:DMatrix<f32> = x_mat.clone().transpose();
        let x_t_mult_x = x_transpose.clone() * x_mat.clone();

        let mut w: Vec<f32> = Vec::with_capacity(colonnes_x * colonnes_x);

        for i in 0..colonnes_x {
            for j in 0..colonnes_x {
                w.push(x_t_mult_x[(i,j)]);
            }
        }

        // let w: Vec<_> = x_mat.iter().cloned().collect();
        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}

#[no_mangle]
extern "C" fn get_number_of_w(pointer_to_layers: *mut f32, number_of_layers: usize) -> usize {
    unsafe {
        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);
        let mut total_number_of_weights = 0.0;

        for l in 0..(number_of_layers - 1) {
            total_number_of_weights += (layers[l] + 1.0) * layers[l + 1];
        }
        total_number_of_weights as usize
    }
}

#[no_mangle]
extern "C" fn generate_multi_layer_perceptron_model(pointer_to_layers: *mut f32, number_of_layers: usize) -> *mut f32 {
    unsafe {
        // À vérifier, mais une seed a l'air de généralement être du u64 en Rust.
        use rand::Rng;
        let mut rng = rand::thread_rng();   // À changer pour prendre en compte la seed fournie.

        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);


        let total_number_of_weights = get_number_of_w(pointer_to_layers, number_of_layers);


        let mut w: Vec<f32> = Vec::with_capacity(total_number_of_weights);

        for _ in 0..total_number_of_weights {
            w.push(rng.gen_range(-1f32..1f32));
        }
        // for l in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
        //     // 3 -> 0 à 2
        //     for _ in 0..layers[l] as i32 + 1{
        //         // layers[0] = 2 -> 0 à 2
        //         // layers[1] = 2 -> 0 à 2
        //         w.push(0f32);
        //         // w = [0]
        //         for _ in 1..layers[l + 1] as i32 + 1{
        //             // layers[1] = 2 -> 1 à 2
        //             // layers[1] = 2 -> 1 à
        //             w.push(rng.gen_range(-1f32..1f32));
        //             // w = [0, 1]
        //         }
        //         // w = [0, 1, 0,
        //     }
        // }


        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn train_multi_layer_perceptron_model_old(pointer_to_model: *mut f32,
                                                 pointer_to_layers: *mut f32,
                                                 number_of_layers: usize,
                                                 pointer_to_inputs: *mut f32,
                                                 number_of_inputs: usize,
                                                 dimension_of_inputs: usize,
                                                 pointer_to_labels : *mut f32,
                                                 number_of_classes: usize,
                                                 learning_rate: f32,
                                                 number_of_epochs: usize,
                                                 is_classification: bool) -> *mut f32 {
    unsafe {
        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);
        if layers[0] as usize != dimension_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }


        let mut total_number_of_weights = get_number_of_w(pointer_to_layers, number_of_layers); // = 9 pour XOR

        let w_param = std::slice::from_raw_parts(pointer_to_model, total_number_of_weights);

        let mut w_index:usize = 0;
        let mut w: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        w.push(Vec::from(Vec::new())); // W[0] n'existe pas.

        for l /*layer*/ in 0..(number_of_layers - 1) { // On calcule d'une couche à la suivante, donc on ne prend pas la première.
            let size_of_w_l: usize = layers[l] as usize + 1; // On rajoute 1 pour le neurone de biais
            let mut w_l: Vec<Vec<f32>> = Vec::with_capacity(size_of_w_l);
            for i in 0..size_of_w_l {
                let size_of_w_l_i: usize = layers[l + 1] as usize;
                let mut w_l_i: Vec<f32> = Vec::with_capacity(size_of_w_l_i);
                for j in 0..size_of_w_l_i {
                    w_l_i.push(w_param[w_index]);
                    w_index += 1;
                }
                w_l.push(w_l_i);
            }
            w.push(w_l);
        }


        let inputs_data = std::slice::from_raw_parts(pointer_to_inputs,
                                                       number_of_inputs * dimension_of_inputs);
        let labels = std::slice::from_raw_parts(pointer_to_labels,
                                                number_of_inputs * number_of_classes);
        use rand::thread_rng;
        use rand::seq::SliceRandom;

        for numero_epoch in 0..number_of_epochs {
            let mut randomly_ordered_dataset: Vec<usize> = (0..number_of_inputs).collect();
            randomly_ordered_dataset.shuffle(&mut thread_rng());


            for k in randomly_ordered_dataset {
                let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                let size_of_x_0: usize = layers[0] as usize + 1;
                let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
                x_0.push(1f32);
                for j in 0..(size_of_x_0-1) {
                    x_0.push(inputs_data[k * dimension_of_inputs + j]);
                }
                x.push(x_0);

                let mut y_k:Vec<f32> = Vec::with_capacity(number_of_classes + 1);
                y_k.push(1f32);
                for class_number in 0..number_of_classes {
                    y_k.push(labels[k * number_of_classes + class_number]);
                }

                let mut delta : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                delta.push(vec![0f32; size_of_x_0]);

                for l in 1..number_of_layers {
                    let size_of_x_l: usize = layers[l] as usize +1;

                    let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                    x_l.push(1f32);
                    for j in 1..size_of_x_l {
                        let mut x_l_i = 0f32;
                        for i in 0..layers[l-1] as usize + 1{
                            x_l_i += w[l][i][j-1] * x[l-1][i];
                        }
                        // si on est en régression et sur la derniere couche, on fait un truc spécial, sinon comme d'hab
                        if !is_classification && l==number_of_layers-1 {
                            x_l.push(x_l_i);
                        } else {
                            x_l.push(x_l_i.tanh());
                        }
                    }
                    x.push(x_l);
                    delta.push(vec![0f32; size_of_x_l]);
                }
                let L = number_of_layers - 1;
                let size_of_delta_L = layers[L] as usize + 1;
                for j in 1..size_of_delta_L{
                    delta[L][j] = x[L][j] - y_k[j];
                    if is_classification {
                        delta[L][j] *= 1f32 - x[L][j] * x[L][j]
                    }
                }

                for l in (1..number_of_layers).rev() {
                    for i in 0..layers[l - 1] as usize + 1{
                        let mut weighed_sum_of_errors = 0f32;
                        for j in 1..layers[l] as usize + 1{
                            weighed_sum_of_errors += w[l][i][j-1] * delta[l][j];
                        }
                        delta[l-1][i] = (1f32 - x[l - 1][i] * x[l - 1][i]) * weighed_sum_of_errors;
                    }
                }

                for l in 1..number_of_layers {
                    for i in 0..layers[l - 1] as usize + 1{
                        for j in 1..layers[l] as usize + 1{
                            w[l][i][j-1] -= learning_rate * x[l - 1][i] * delta[l][j];
                        }
                    }
                }
            }
        }
        let mut w_return = Vec::with_capacity(total_number_of_weights);
        for l in 1..number_of_layers {
            for i in 0..layers[l-1] as usize + 1 {
                for j in 1..layers[l] as usize + 1{
                    w_return.push(w[l][i][j-1]);
                }
            }
        }
        let arr_slice = w_return.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn predict_with_multi_layer_perceptron_model(pointer_to_model: *mut f32,
                                                        pointer_to_layers: *mut f32,
                                                        number_of_layers: usize,
                                                        pointer_to_inputs: *mut f32,
                                                        number_of_inputs: usize,
                                                        dimension_of_inputs: usize,
                                                        number_of_classes: usize,
                                                        is_classification: bool) -> *mut f32 {
    unsafe {

        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);
        if layers[0] as usize != dimension_of_inputs {
            println!("{:?}", layers[0]);
            println!("{:?}", dimension_of_inputs);
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }


        let mut total_number_of_weights = get_number_of_w(pointer_to_layers, number_of_layers);

        let w_param = std::slice::from_raw_parts(pointer_to_model, total_number_of_weights);

        let mut w_index:usize = 0;
        let mut w: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        w.push(Vec::from(Vec::new())); // à chaque fois on veut avoir un w[l][i][j] avec rien dans notre couche l

        for l /*layer*/ in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
            let size_of_w_l: usize = layers[l] as usize + 1; // on a rajouté 1 pour le biais
            let mut w_l: Vec<Vec<f32>> = Vec::with_capacity(size_of_w_l);
            for i in 0..size_of_w_l {
                let size_of_w_l_i: usize = layers[l + 1] as usize; // on met pas +1 parce qu'on veut pas le biais
                let mut w_l_i: Vec<f32> = Vec::with_capacity(size_of_w_l_i);
                for j in 0..size_of_w_l_i {
                    w_l_i.push(w_param[w_index]);
                    w_index += 1;
                }
                w_l.push(w_l_i);
            }
            w.push(w_l);
        }

        let inputs_data = std::slice::from_raw_parts(pointer_to_inputs,
                                                     number_of_inputs * dimension_of_inputs);
        let mut Y : Vec<f32> = Vec::with_capacity(number_of_inputs*number_of_classes);

        for k in 0..number_of_inputs {

            let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);

            let size_of_x_0: usize = layers[0] as usize + 1; // = 2 + 1 = 3
            let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
            x_0.push(1f32);
            for j in 0..(size_of_x_0-1) { // 0..2
                // x_0 = [1, 1, 0]
                x_0.push(inputs_data[k * dimension_of_inputs + j]); // inputs = [0, 0, 0, 1, 1, 0, 1, 1]
            }
            x.push(x_0);



            for l in 1..number_of_layers { // nb layers = 2
                let size_of_x_l: usize = layers[l] as usize; // size of layer[1] = 1
                let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                x_l.push(1f32);
                for j in 0..size_of_x_l {
                    let mut x_l_i = 0f32;
                    for i in 0..layers[l-1] as usize + 1{ // layers[0] = 2 + 1 = 3
                        x_l_i += w[l][i][j] * x[l-1][i];
                    }
                    if !is_classification && l==number_of_layers-1 {
                        x_l.push(x_l_i);
                    } else {
                        x_l.push(x_l_i.tanh());
                    }
                }
                x.push(x_l);
            }

            for i in 1..number_of_classes + 1{
                Y.push(x[number_of_layers-1][i]);
            }

        }
        let arr_slice = Y.leak();
        arr_slice.as_mut_ptr()
    }
}



// new

fn multi_layer_perceptron_predict_test( w:  Vec<Vec<Vec<f32>>>, // c'est le w entrainé
                                              inputs: &[f32], // les données qu'on veut prédire
                                              number_of_inputs: usize, // le nombre de données dans le pointeur d'au dessus
                                              dimension_of_inputs: usize, // dimension des inputs
                                              number_of_classes_to_predict: usize,
                                              layers: &[f32], // forme du perceptron, ex: (2, 2, 1)
                                              number_of_layers: usize, // nombre de couches
                                              is_classification: bool) -> Vec<f32> {


        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        if layers[0] as usize != dimension_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes_to_predict {
            panic!("Wrong number of neurons in the last layer.");
        }

        let mut Y : Vec<f32> = Vec::with_capacity(number_of_inputs * number_of_classes_to_predict);

        for k in 0..number_of_inputs {
            let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
            let size_of_x_0: usize = layers[0] as usize + 1;
            let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
            x_0.push(1f32);
            for j in 0..(size_of_x_0-1) {
                x_0.push(inputs[k * dimension_of_inputs + j]);
            }
            x.push(x_0);

            for l in 1..number_of_layers {
                let size_of_x_l: usize = layers[l] as usize + 1;
                let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                x_l.push(1f32);
                for j in 1..size_of_x_l {
                    let mut x_l_i = 0f32;
                    for i in 0..layers[l-1] as usize + 1{
                        x_l_i += w[l][i][j - 1] * x[l-1][i];
                    }
                    if !is_classification && l==number_of_layers-1 {
                        x_l.push(x_l_i);
                    } else {
                        x_l.push(x_l_i.tanh());
                    }
                }
                x.push(x_l);
            }

            for i in 1..number_of_classes_to_predict + 1{
                Y.push(x[number_of_layers-1][i]);
            }
        }
    Y
}


fn save_accuracy_and_losses_as_file(accuracies_and_losses: AccuraciesAndLosses) -> io::Result<()> {
    let mut file = File::create("accuracies_and_losses.json")?;
    let accuracies_and_losses_to_string = serde_json::to_string(&accuracies_and_losses);
    write!(file, "{}", accuracies_and_losses_to_string.unwrap())?;
    Ok(())
}

#[no_mangle]
extern "C" fn train_multi_layer_perceptron_model(pointer_to_model: *mut f32,
                                                 pointer_to_layers: *mut f32,
                                                 number_of_layers: usize,
                                                 pointer_to_training_inputs: *mut f32,
                                                 number_of_training_inputs: usize,
                                                 pointer_to_tests_inputs: *mut f32,
                                                 number_of_tests_inputs: usize,
                                                 dimension_of_inputs: usize,
                                                 pointer_to_training_labels : *mut f32,
                                                 pointer_to_tests_labels : *mut f32,
                                                 number_of_classes: usize,
                                                 learning_rate: f32,
                                                 number_of_epochs: usize,
                                                 batch_size: usize,
                                                 is_classification: bool) -> *mut f32 {
    unsafe {
        if batch_size == 0 {
            panic!("Batch size must be at least 1.");
        }
        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);
        if layers[0] as usize != dimension_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }

        let mut total_number_of_weights = get_number_of_w(pointer_to_layers, number_of_layers);
        let w_param = std::slice::from_raw_parts(pointer_to_model, total_number_of_weights);
        let mut w_index:usize = 0;
        let mut w: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        w.push(Vec::from(Vec::new()));
        for l in 0..(number_of_layers - 1) {
            let size_of_w_l: usize = layers[l] as usize + 1;
            let mut w_l: Vec<Vec<f32>> = Vec::with_capacity(size_of_w_l);
            for i in 0..size_of_w_l {
                let size_of_w_l_i: usize = layers[l + 1] as usize;
                let mut w_l_i: Vec<f32> = Vec::with_capacity(size_of_w_l_i);
                for j in 0..size_of_w_l_i {
                    w_l_i.push(w_param[w_index]);
                    w_index += 1;
                }
                w_l.push(w_l_i);
            }
            w.push(w_l);
        }

        let training_inputs = std::slice::from_raw_parts(pointer_to_training_inputs,
                                                         number_of_training_inputs * dimension_of_inputs);
        let training_labels = std::slice::from_raw_parts(pointer_to_training_labels,
                                                number_of_training_inputs * number_of_classes);

        let tests_inputs = std::slice::from_raw_parts(pointer_to_tests_inputs,
                                                     number_of_tests_inputs * dimension_of_inputs);
        let tests_labels = std::slice::from_raw_parts(pointer_to_tests_labels,
                                                     number_of_tests_inputs * number_of_classes);

        use rand::thread_rng;
        use rand::seq::SliceRandom;

        let mut numbers_of_errors_on_training_dataset: Vec<usize> = Vec::with_capacity(number_of_epochs / batch_size);
        let mut accuracies_on_training_dataset: Vec<f32> = Vec::with_capacity(number_of_epochs / batch_size);
        let mut losses_on_training_dataset: Vec<f32> = Vec::with_capacity(number_of_epochs / batch_size);
        let mut numbers_of_errors_on_tests_dataset: Vec<usize> = Vec::with_capacity(number_of_epochs / batch_size);
        let mut accuracies_on_tests_dataset: Vec<f32> = Vec::with_capacity(number_of_epochs / batch_size);
        let mut losses_on_tests_dataset: Vec<f32> = Vec::with_capacity(number_of_epochs / batch_size);

        for epoch_number in 0..number_of_epochs {
            println!("Epochs {:?}",epoch_number + 1);
            let check_accuracy_and_loss : bool = (epoch_number + 1) % batch_size == 0;
            let mut number_of_mispredicted_training_outputs:usize = 0;
            let mut training_squarred_errors_sum:f32 = 0.;

            let mut randomly_ordered_dataset: Vec<usize> = (0..number_of_training_inputs).collect();
            randomly_ordered_dataset.shuffle(&mut thread_rng());

            for k in randomly_ordered_dataset {
                let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                let size_of_x_0: usize = layers[0] as usize + 1;
                let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
                x_0.push(1f32);
                for j in 0..(size_of_x_0 - 1) {
                    x_0.push(training_inputs[k * dimension_of_inputs + j]);
                }
                x.push(x_0);

                let mut labels:Vec<f32> = Vec::with_capacity(number_of_classes + 1);
                labels.push(1f32);
                for class_number in 0..number_of_classes {
                    labels.push(training_labels[k * number_of_classes + class_number]);
                }

                let mut delta : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                delta.push(vec![0f32; size_of_x_0]);

                for l in 1..number_of_layers {
                    let size_of_x_l: usize = layers[l] as usize + 1;

                    let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                    x_l.push(1f32);
                    for j in 1..size_of_x_l {
                        let mut x_l_i = 0f32;
                        for i in 0..layers[l-1] as usize + 1{
                            x_l_i += w[l][i][j-1] * x[l-1][i];
                        }
                        if !is_classification && l==number_of_layers-1 {
                            x_l.push(x_l_i);
                        } else {
                            x_l.push(x_l_i.tanh());
                        }
                    }
                    x.push(x_l);
                    delta.push(vec![0f32; size_of_x_l]);
                }
                let L = number_of_layers - 1;
                let size_of_delta_L = layers[L] as usize + 1;
                // if check_accuracy_and_loss { // It won't compile with that uncommented, even if the uses of is_mispredicted have the same condition.
                    let mut is_mispredicted = false;
                // }
                for j in 1..size_of_delta_L{
                    delta[L][j] = x[L][j] - labels[j];
                    if check_accuracy_and_loss {
                        // Accuracy
                        if (delta[L][j] >= 1. || delta[L][j] <= -1.) && !is_mispredicted {
                            number_of_mispredicted_training_outputs += 1;
                            is_mispredicted = true;
                        }
                        // Loss
                        training_squarred_errors_sum += delta[L][j] * delta[L][j];

                        if is_classification {
                            delta[L][j] *= 1f32 - x[L][j] * x[L][j];
                        }
                    }
                }

                for l in (1..number_of_layers).rev() {
                    for i in 0..layers[l - 1] as usize + 1{
                        let mut weighted_sum_of_errors = 0f32;
                        for j in 1..layers[l] as usize + 1{
                            weighted_sum_of_errors += w[l][i][j-1] * delta[l][j];
                        }
                        delta[l-1][i] = (1f32 - x[l - 1][i] * x[l - 1][i]) * weighted_sum_of_errors;
                    }
                }

                for l in 1..number_of_layers {
                    for i in 0..layers[l - 1] as usize + 1{
                        for j in 1..layers[l] as usize + 1{
                            w[l][i][j-1] -= learning_rate * x[l - 1][i] * delta[l][j];
                        }
                    }
                }
            }

            if check_accuracy_and_loss {
                let training_accuracy:f32 = number_of_mispredicted_training_outputs as f32 / number_of_training_inputs as f32;
                let training_loss:f32 = training_squarred_errors_sum / number_of_training_inputs as f32;
                println!("Number of training inputs mispredicted : {:?}", number_of_mispredicted_training_outputs);
                println!("Training inputs accuracy : {:?}", training_accuracy);
                println!("Training inputs loss : {:?}", training_loss);
                numbers_of_errors_on_training_dataset.push(number_of_mispredicted_training_outputs);
                accuracies_on_training_dataset.push(training_accuracy);
                losses_on_training_dataset.push(training_loss);

                let precicted_labels = multi_layer_perceptron_predict_test(w.clone(),
                                                                        tests_inputs.clone(),
                                                                        number_of_tests_inputs,
                                                                        dimension_of_inputs,
                                                                        number_of_classes,
                                                                        layers,
                                                                        number_of_layers,
                                                                        is_classification);
                
                let mut number_of_mispredicted_tests_outputs:usize = 0;
                let mut tests_squarred_errors_sum:f32 = 0.;
                
                for i in (0..(number_of_tests_inputs * number_of_classes)).step_by(number_of_classes) {
                    let mut is_mispredicted = false;
                    for j in 0..number_of_classes {
                        let delta_test = tests_labels[i+j] - precicted_labels.clone()[i+j];
                        if !is_mispredicted && (delta_test >= 1. || delta_test <= -1.) {
                            number_of_mispredicted_tests_outputs += 1;
                            is_mispredicted = true;
                        }
                        tests_squarred_errors_sum += delta_test * delta_test;
                    }
                }
                let tests_accuracy:f32 = number_of_mispredicted_tests_outputs as f32 / number_of_tests_inputs as f32;
                let tests_loss:f32 = tests_squarred_errors_sum / number_of_tests_inputs as f32;
                println!("Number of tests inputs mispredicted : {:?}", number_of_mispredicted_tests_outputs);
                println!("Tests inputs accuracy: {:?}", tests_accuracy);
                println!("Tests inputs loss : {:?}", tests_loss);
                numbers_of_errors_on_tests_dataset.push(number_of_mispredicted_tests_outputs);
                accuracies_on_tests_dataset.push(tests_accuracy);
                losses_on_tests_dataset.push(tests_loss);
            }
        }

        let mut accuracies_and_losses = AccuraciesAndLosses {
            number_of_training_inputs : number_of_training_inputs,
            number_of_tests_inputs : number_of_tests_inputs,
            number_of_epochs : number_of_epochs,
            batch_size : batch_size,
            numbers_of_errors_on_training_dataset : numbers_of_errors_on_training_dataset,
            training_accuracies : accuracies_on_training_dataset,
            training_losses : losses_on_training_dataset,
            numbers_of_errors_on_tests_dataset : numbers_of_errors_on_tests_dataset,
            tests_accuracies : accuracies_on_tests_dataset,
            tests_losses : losses_on_tests_dataset,
        };
        save_accuracy_and_losses_as_file(accuracies_and_losses).expect("Doesn't work");
        
        let mut w_return = Vec::with_capacity(total_number_of_weights);
        for l in 1..number_of_layers { 
            for i in 0..layers[l-1] as usize + 1 {
                for j in 1..layers[l] as usize + 1{
                    w_return.push(w[l][i][j-1]);
                }
            }
        }
        let arr_slice = w_return.leak();
        arr_slice.as_mut_ptr()
    }
}
