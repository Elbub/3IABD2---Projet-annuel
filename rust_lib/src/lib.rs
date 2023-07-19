use nalgebra::{DMatrix, OMatrix};
use std::fs::File;
use std::io::{self, Write};
use serde::{Deserialize, Serialize};
use fast_math::exp;
use rand::Rng;
use rand::thread_rng;
use rand::seq::SliceRandom;


#[derive(Serialize, Deserialize)]
struct AccuraciesAndLosses {
    number_of_training_inputs : usize,
    number_of_tests_inputs : usize,
    number_of_epochs : usize,
    batch_size : usize,
    learning_rate : f32,
    numbers_of_errors_on_training_dataset : Vec<usize>,
    training_accuracies : Vec<f32>,
    training_losses : Vec<f32>,
    numbers_of_errors_on_tests_dataset : Vec<usize>,
    tests_accuracies : Vec<f32>,
    tests_losses : Vec<f32>,
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
extern "C" fn generate_linear_model(dimensions_of_inputs: usize, number_of_classes: usize) -> *mut f32 {
    let mut rng = rand::thread_rng();
    let number_of_parameters: usize = dimensions_of_inputs;
    let mut weights: Vec<f32> = Vec::with_capacity((dimensions_of_inputs + 1) * number_of_classes);   // weights : contient les poids associés aux Xi
    for _ in 0..dimensions_of_inputs + 1 {
        for _ in 0..number_of_classes {
            weights.push(rng.gen_range(-1f32..1f32)); // initialisation aléatoire entre -1 et 1
        }
    }
    let arr_slice = weights.leak();
    arr_slice.as_mut_ptr()
}


#[no_mangle]
extern "C" fn train_linear_model_classification(pointer_to_model: *mut f32,
                                                pointer_to_inputs: *mut f32,
                                                number_of_inputs: usize,
                                                dimensions_of_inputs: usize,
                                                pointer_to_labels : *mut f32,
                                                number_of_classes: usize,
                                                learning_rate: f32,
                                                number_of_epochs: usize
) -> *mut f32 {
    unsafe {
        let inputs = std::slice::from_raw_parts(pointer_to_inputs, number_of_inputs * dimensions_of_inputs);
        let labels = std::slice::from_raw_parts(pointer_to_labels, number_of_inputs * number_of_classes);
        let mut weights = Vec::from_raw_parts(pointer_to_model, (dimensions_of_inputs + 1) * number_of_classes, (dimensions_of_inputs + 1) * number_of_classes);
        let mut rng = rand::thread_rng();


        for epoch_number in 0..number_of_epochs {
            let mut randomly_ordered_dataset: Vec<usize> = (0..number_of_inputs).collect();
            randomly_ordered_dataset.shuffle(&mut thread_rng());
            for k in randomly_ordered_dataset {
                let mut label: Vec<f32> = Vec::with_capacity(number_of_classes);
                for i in 0..number_of_classes {
                    label.push(labels[k * number_of_classes +  i]);
                }
                let mut input: Vec<f32> = Vec::with_capacity(dimensions_of_inputs + 1);
                input.push(1f32);
                for i in 0..dimensions_of_inputs {
                    input.push(inputs[k * dimensions_of_inputs + i]);
                }
                for j in 0..number_of_classes{
                    let mut signal: f32 = 0f32;
                    for i in 0..dimensions_of_inputs + 1 {
                        signal += weights[i * number_of_classes + j] * input[i];
                    }
                    let mut predicted_output: f32 = -1.0; // on avait 0.0 pour des cas 0 ou 1 en output
                    if signal >= 0f32 {
                        predicted_output = 1f32;
                    }
                    for i in 0..dimensions_of_inputs + 1 {
                        weights[i * number_of_classes + j] += learning_rate * (label[j] - predicted_output) * input[i];
                    }
                }
            }
        }
        weights.leak().as_mut_ptr()
    }
}

 
#[no_mangle]
extern "C" fn train_linear_model_regression(pointer_to_inputs: *mut f32,
                                            number_of_inputs: usize,
                                            dimensions_of_inputs: usize,
                                            pointer_to_outputs: *mut f32,
                                            number_of_classes: usize
) -> *mut f32 {
    unsafe {
        use nalgebra::*;

        let flattened_inputs = std::slice::from_raw_parts(pointer_to_inputs, number_of_inputs * dimensions_of_inputs);
        let flattened_labels = std::slice::from_raw_parts(pointer_to_outputs, number_of_inputs * number_of_classes);
        let mut inputs:DMatrix<f32> = DMatrix::zeros(number_of_inputs, dimensions_of_inputs + 1);
        let mut outputs:DMatrix<f32> = DMatrix::zeros(number_of_inputs, number_of_classes);

        for i in 0..number_of_inputs {
            inputs[(i, 0)] = 1.0;
            for j in 0..dimensions_of_inputs {
                inputs[(i, j + 1)] = flattened_inputs[i * dimensions_of_inputs + j];
            }
        }

        for i in 0..number_of_inputs {
            for j in 0..number_of_classes {
                outputs[(i, j)] = flattened_labels[i * number_of_classes + j];
            }
        }

        
        let inputs_pseudo_inverse = matrix_pseudo_inverse(inputs, dimensions_of_inputs + 1);

        let result_matrix = inputs_pseudo_inverse * outputs;

        let mut weights: Vec<f32> = Vec::with_capacity((dimensions_of_inputs + 1) * number_of_classes);

        for i in 0..dimensions_of_inputs + 1 {
            for j in 0..number_of_classes {
                weights.push(result_matrix[(i,j)]);
            }
        }

        let arr_slice = weights.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn predict_with_linear_model(pointer_to_trained_model: *mut f32,
                                        pointer_to_inputs: *const f32,
                                        number_of_inputs: usize,
                                        dimensions_of_inputs: usize,
                                        number_of_classes: usize,
                                        is_classification: bool
) -> *mut f32{
    unsafe{
        let inputs = std::slice::from_raw_parts(pointer_to_inputs, number_of_inputs * dimensions_of_inputs);
        let trained_model = std::slice::from_raw_parts(pointer_to_trained_model, (dimensions_of_inputs + 1) * number_of_classes);
        let mut predicted_labels : Vec<f32> = Vec::with_capacity(number_of_inputs * number_of_classes);

        for input_number in 0..number_of_inputs {
            for class in 0..number_of_classes{
                let mut weighted_sum : f32 = 1f32 * trained_model[0 * number_of_classes + class];
                for input_dimension in 1..dimensions_of_inputs + 1 {
                    weighted_sum += inputs[input_number * dimensions_of_inputs + input_dimension - 1] * trained_model[input_dimension * number_of_classes + class]
                }
                if is_classification {
                    if weighted_sum >= 0f32 {
                        predicted_labels.push(1f32);
                    } else {
                        predicted_labels.push(-1f32);
                    }
                } else {
                    predicted_labels.push(weighted_sum);
                }
            }
        }
        predicted_labels.leak().as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn get_number_of_weights(pointer_to_layers: *mut f32, number_of_layers: usize) -> usize {
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
        let mut rng = rand::thread_rng();   // À changer pour prendre en compte la seed fournie.

        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);


        let total_number_of_weights = get_number_of_weights(pointer_to_layers, number_of_layers);


        let mut weights: Vec<f32> = Vec::with_capacity(total_number_of_weights);

        for _ in 0..total_number_of_weights {
            weights.push(rng.gen_range(-1f32..1f32));
        }

        let arr_slice = weights.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn train_multi_layer_perceptron_model_old(pointer_to_model: *mut f32,
                                                 pointer_to_layers: *mut f32,
                                                 number_of_layers: usize,
                                                 pointer_to_inputs: *mut f32,
                                                 number_of_inputs: usize,
                                                 dimensions_of_inputs: usize,
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
        if layers[0] as usize != dimensions_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }


        let mut total_number_of_weights = get_number_of_weights(pointer_to_layers, number_of_layers);

        let w_param = std::slice::from_raw_parts(pointer_to_model, total_number_of_weights);

        let mut w_index:usize = 0;
        let mut weights: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        weights.push(Vec::from(Vec::new()));

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
            weights.push(w_l);
        }


        let inputs_data = std::slice::from_raw_parts(pointer_to_inputs,
                                                       number_of_inputs * dimensions_of_inputs);
        let labels = std::slice::from_raw_parts(pointer_to_labels,
                                                number_of_inputs * number_of_classes);

        for epoch_number in 0..number_of_epochs {
            let mut randomly_ordered_dataset: Vec<usize> = (0..number_of_inputs).collect();
            randomly_ordered_dataset.shuffle(&mut thread_rng());


            for k in randomly_ordered_dataset {
                let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                let size_of_x_0: usize = layers[0] as usize + 1;
                let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
                x_0.push(1f32);
                for j in 0..(size_of_x_0-1) {
                    x_0.push(inputs_data[k * dimensions_of_inputs + j]);
                }
                x.push(x_0);

                let mut label:Vec<f32> = Vec::with_capacity(number_of_classes + 1);
                label.push(1f32);
                for class_number in 0..number_of_classes {
                    label.push(labels[k * number_of_classes + class_number]);
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
                            x_l_i += weights[l][i][j-1] * x[l-1][i];
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
                for j in 1..size_of_delta_L{
                    delta[L][j] = x[L][j] - label[j];
                    if is_classification {
                        delta[L][j] *= 1f32 - x[L][j] * x[L][j]
                    }
                }

                for l in (1..number_of_layers).rev() {
                    for i in 0..layers[l - 1] as usize + 1{
                        let mut weighted_sum_of_errors = 0f32;
                        for j in 1..layers[l] as usize + 1{
                            weighted_sum_of_errors += weights[l][i][j-1] * delta[l][j];
                        }
                        delta[l-1][i] = (1f32 - x[l - 1][i] * x[l - 1][i]) * weighted_sum_of_errors;
                    }
                }

                for l in 1..number_of_layers {
                    for i in 0..layers[l - 1] as usize + 1{
                        for j in 1..layers[l] as usize + 1{
                            weights[l][i][j-1] -= learning_rate * x[l - 1][i] * delta[l][j];
                        }
                    }
                }
            }
        }
        let mut w_return = Vec::with_capacity(total_number_of_weights);
        for l in 1..number_of_layers {
            for i in 0..layers[l-1] as usize + 1 {
                for j in 1..layers[l] as usize + 1{
                    w_return.push(weights[l][i][j-1]);
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
                                                        dimensions_of_inputs: usize,
                                                        number_of_classes: usize,
                                                        is_classification: bool) -> *mut f32 {
    unsafe {

        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(pointer_to_layers, number_of_layers);
        if layers[0] as usize != dimensions_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }


        let mut total_number_of_weights = get_number_of_weights(pointer_to_layers, number_of_layers);

        let w_param = std::slice::from_raw_parts(pointer_to_model, total_number_of_weights);

        let mut w_index:usize = 0;
        let mut weights: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        weights.push(Vec::from(Vec::new())); // à chaque fois on veut avoir un weights[l][i][j] avec rien dans notre couche l

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
            weights.push(w_l);
        }

        let inputs_data = std::slice::from_raw_parts(pointer_to_inputs,
                                                     number_of_inputs * dimensions_of_inputs);
        let mut Y : Vec<f32> = Vec::with_capacity(number_of_inputs*number_of_classes);

        for k in 0..number_of_inputs {

            let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);

            let size_of_x_0: usize = layers[0] as usize + 1;
            let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
            x_0.push(1f32);
            for j in 0..(size_of_x_0-1) {
                x_0.push(inputs_data[k * dimensions_of_inputs + j]);
            }
            x.push(x_0);



            for l in 1..number_of_layers {
                let size_of_x_l: usize = layers[l] as usize;
                let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                x_l.push(1f32);
                for j in 0..size_of_x_l {
                    let mut x_l_i = 0f32;
                    for i in 0..layers[l-1] as usize + 1{
                        x_l_i += weights[l][i][j] * x[l-1][i];
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


fn multi_layer_perceptron_predict_test(weights: &Vec<Vec<Vec<f32>>>, // c'est le weights entrainé
                                              inputs: &[f32], // les données qu'on veut prédire
                                              number_of_inputs: usize, // le nombre de données dans le pointeur d'au dessus
                                              dimensions_of_inputs: usize, // dimension des inputs
                                              number_of_classes: usize,
                                       layers: &[f32], // forme du perceptron, ex: (2, 2, 1)
                                              number_of_layers: usize, // nombre de couches
                                              is_classification: bool) -> Vec<f32> {


        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        if layers[0] as usize != dimensions_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }

        let mut Y : Vec<f32> = Vec::with_capacity(number_of_inputs * number_of_classes);

        for k in 0..number_of_inputs {
            let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
            let size_of_x_0: usize = layers[0] as usize + 1;
            let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
            x_0.push(1f32);
            for j in 0..(size_of_x_0-1) {
                x_0.push(inputs[k * dimensions_of_inputs + j]);
            }
            x.push(x_0);

            for l in 1..number_of_layers {
                let size_of_x_l: usize = layers[l] as usize + 1;
                let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                x_l.push(1f32);
                for j in 1..size_of_x_l {
                    let mut x_l_i = 0f32;
                    for i in 0..layers[l-1] as usize + 1{
                        x_l_i += weights[l][i][j - 1] * x[l-1][i];
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
                                                 dimensions_of_inputs: usize,
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
        if layers[0] as usize != dimensions_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes {
            panic!("Wrong number of neurons in the last layer.");
        }

        let mut total_number_of_weights = get_number_of_weights(pointer_to_layers, number_of_layers);
        let w_param = std::slice::from_raw_parts(pointer_to_model, total_number_of_weights);
        let mut w_index:usize = 0;
        let mut weights: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        weights.push(Vec::from(Vec::new()));
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
            weights.push(w_l);
        }

        let training_inputs = std::slice::from_raw_parts(pointer_to_training_inputs,
                                                         number_of_training_inputs * dimensions_of_inputs);
        let training_labels = std::slice::from_raw_parts(pointer_to_training_labels,
                                                number_of_training_inputs * number_of_classes);

        let tests_inputs = std::slice::from_raw_parts(pointer_to_tests_inputs,
                                                     number_of_tests_inputs * dimensions_of_inputs);
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
                    x_0.push(training_inputs[k * dimensions_of_inputs + j]);
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
                            x_l_i += &weights[l][i][j-1] * x[l-1][i];
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
                            weighted_sum_of_errors += &weights[l][i][j-1] * delta[l][j];
                        }
                        delta[l-1][i] = (1f32 - x[l - 1][i] * x[l - 1][i]) * weighted_sum_of_errors;
                    }
                }

                for l in 1..number_of_layers {
                    for i in 0..layers[l - 1] as usize + 1{
                        for j in 1..layers[l] as usize + 1{
                            weights[l][i][j-1] -= learning_rate * x[l - 1][i] * delta[l][j];
                        }
                    }
                }
            }

            if check_accuracy_and_loss {
                let training_accuracy:f32 = 1f32 - number_of_mispredicted_training_outputs as f32 / number_of_training_inputs as f32;
                let training_loss:f32 = training_squarred_errors_sum / number_of_training_inputs as f32;
                println!("Number of training inputs mispredicted : {:?}", number_of_mispredicted_training_outputs);
                println!("Training inputs accuracy : {:?}", training_accuracy);
                println!("Training inputs loss : {:?}", training_loss);
                numbers_of_errors_on_training_dataset.push(number_of_mispredicted_training_outputs);
                accuracies_on_training_dataset.push(training_accuracy);
                losses_on_training_dataset.push(training_loss);

                let precicted_labels = multi_layer_perceptron_predict_test(&weights,
                                                                        tests_inputs,
                                                                        number_of_tests_inputs,
                                                                        dimensions_of_inputs,
                                                                        number_of_classes,
                                                                        layers,
                                                                        number_of_layers,
                                                                        is_classification);
                
                let mut number_of_mispredicted_tests_outputs:usize = 0;
                let mut tests_squarred_errors_sum:f32 = 0.;
                
                for i in (0..(number_of_tests_inputs * number_of_classes)).step_by(number_of_classes) {
                    let mut is_mispredicted = false;
                    for j in 0..number_of_classes {
                        let delta_test = tests_labels[i+j] - precicted_labels[i+j];
                        if !is_mispredicted && (delta_test >= 1. || delta_test <= -1.) {
                            number_of_mispredicted_tests_outputs += 1;
                            is_mispredicted = true;
                        }
                        tests_squarred_errors_sum += delta_test * delta_test;
                    }
                }
                let tests_accuracy:f32 = 1f32 - number_of_mispredicted_tests_outputs as f32 / number_of_tests_inputs as f32;
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
            learning_rate : learning_rate,
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
                    w_return.push(weights[l][i][j-1]);
                }
            }
        }
        let arr_slice = w_return.leak();
        arr_slice.as_mut_ptr()
    }
}


fn matrix_pseudo_inverse(input_matrix: DMatrix<f32>, dimensions_of_inputs: usize) -> DMatrix<f32>{
    let x_transpose = input_matrix.transpose();
    let mut x_t_mult_x = &x_transpose * input_matrix;
    let mut det = x_t_mult_x.determinant();

    while det >= -0.00005 && det <= 0.00005{
        let mut rng = rand::thread_rng();
        for i in 0..(dimensions_of_inputs) {
            for j in 0..(dimensions_of_inputs) {
                x_t_mult_x[(i,j)] = x_t_mult_x[(i,j)] + rng.gen_range(-0.005..0.005);
            }
        }
        det = x_t_mult_x.determinant();
    }


    let inv_x_t_x = x_t_mult_x.try_inverse();
    let inv_times_x_t = match inv_x_t_x {
        Some(inv) => inv * x_transpose,
        None => panic!("Non inversible"),
    };
    inv_times_x_t
}


fn k_means(
    inputs_train: &[f32],
    number_of_clusters: usize,
    dimensions_of_inputs: usize,
    number_of_points: usize,
) -> Vec<Vec<f32>> {
    // initialiser des centres randoms
    let mut vec_of_mu_k: Vec<Vec<f32>> = Vec::with_capacity(number_of_clusters);
    // vec_of_mu_k= [mu_0, mu_1 ..., mu_k]
    // mu_x = le centre du cluster x
    // mu_x a le meme nombre d'éléments que une image
    for k in 0..number_of_clusters{
        let mut mu_k : Vec<f32> = Vec::with_capacity(dimensions_of_inputs);
        for j in 0..dimensions_of_inputs{
            mu_k.push(rand::thread_rng().gen_range(-1f32..1f32));
        }
        vec_of_mu_k.push(mu_k);
    }

    // création ensemble Sk
    // Pour chaque points on vérifie à quel centre il appartient
    // X -> pour tout point -> on vérifie s'il est plus proche d'un centre ou d'un autre
    let mut old_vec_of_mu_k : Vec<Vec<f32>> = Vec::with_capacity(number_of_clusters);
    for k in 0..number_of_clusters{
        let temp_vec: Vec<f32> = vec![0.0; dimensions_of_inputs];
        old_vec_of_mu_k.push(temp_vec);
    }
    let mut count = 0;
    while old_vec_of_mu_k != vec_of_mu_k && count <= 100 {
        let mut vec_of_Sk: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_clusters);

        for k in 0..number_of_clusters {
            let mut S_k: Vec<Vec<f32>> = Vec::new();
            for n in 0..number_of_points {
                let mut distance_k: f32 = 0.0;
                for j in 0..dimensions_of_inputs {
                    distance_k += (inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[k][j])*(inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[k][j]);
                }
                distance_k = distance_k.sqrt();
                for l in 0..number_of_clusters {
                    if l != k {
                        let mut distance_l: f32 = 0.0;
                        for j in 0..dimensions_of_inputs {
                            distance_l += (inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[l][j])*(inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[l][j]);
                        }
                        distance_l = distance_l.sqrt();
                        if distance_k <= distance_l {
                            let mut vec_to_push = Vec::with_capacity(dimensions_of_inputs);
                            for i in 0..dimensions_of_inputs {
                                vec_to_push.push(inputs_train[n * dimensions_of_inputs + i]);
                            }
                            S_k.push(vec_to_push);
                        }
                    }
                }
            }
            vec_of_Sk.push(S_k);
        }

        //update mu_k
        old_vec_of_mu_k = vec_of_mu_k;
        vec_of_mu_k = Vec::with_capacity(number_of_clusters);
        for k in 0..number_of_clusters {
            let mut mu_k: Vec<f32> = vec![0.0; dimensions_of_inputs];
            for n in &vec_of_Sk[k] {
                for i in 0..dimensions_of_inputs {
                    mu_k[i] += n[i] / vec_of_Sk[k].len() as f32;
                }
            }
            vec_of_mu_k.push(mu_k);
        }
        count += 1;
    }
    vec_of_mu_k

}


#[no_mangle]
extern "C" fn radial_basis_function_model(      pointer_to_inputs_train : *mut f32,
                                                number_of_training_inputs : usize,
                                                pointer_to_inputs_to_predict : *mut f32,
                                                number_of_inputs_to_predict : usize,
                                                dimensions_of_inputs : usize,
                                                pointer_to_labels : *mut f32,
                                                number_of_classes : usize,
                                                gamma : f32,
                                                is_classification : bool,
                                                number_of_clusters : usize,
                                                is_naive: bool
) -> *mut f32 {

    unsafe{
        let inputs_train = std::slice::from_raw_parts(pointer_to_inputs_train,
                                                      number_of_training_inputs * dimensions_of_inputs);
        let labels = std::slice::from_raw_parts(pointer_to_labels,
                                                number_of_training_inputs * number_of_classes);

        let inputs_to_predict = std::slice::from_raw_parts(pointer_to_inputs_to_predict,
                                                           number_of_inputs_to_predict * dimensions_of_inputs);

        let mut phi: DMatrix<f32> = DMatrix::zeros(number_of_training_inputs, number_of_clusters + 1);
        if is_naive && number_of_clusters == number_of_training_inputs {

            for i in 0..number_of_training_inputs {
                phi[(i, 0)] = 1f32;
                for j in 0..number_of_training_inputs {
                    let mut squarred_distance: f32 = 0.0;
                    for dimension in 0..dimensions_of_inputs {
                        squarred_distance += (inputs_train[i * dimensions_of_inputs + dimension] - inputs_train[j * dimensions_of_inputs + dimension]) * (inputs_train[i * dimensions_of_inputs + dimension] - inputs_train[j * dimensions_of_inputs + dimension]);
                    }
                    phi[(i, j + 1)] = exp(-gamma * squarred_distance);
                }
            }
            let phi_pseudo_inverse:DMatrix<f32> = matrix_pseudo_inverse(phi,number_of_clusters+1);
            let mut labels_as_matrix:DMatrix<f32> = DMatrix::zeros(number_of_training_inputs, number_of_classes);
            for i in 0..number_of_training_inputs {
                for j in 0..number_of_classes{
                    labels_as_matrix[(i, j)] = labels[i*number_of_classes+j];
                }
            }

            let weights_as_matrix:DMatrix<f32> = phi_pseudo_inverse * labels_as_matrix;

            let mut outputs:Vec<Vec<f32>> = Vec::with_capacity(number_of_inputs_to_predict);
            for i in 0..number_of_inputs_to_predict {
                let mut output:Vec<f32> = Vec::with_capacity(number_of_classes);
                for j in 0..number_of_classes {
                    let mut weighted_sum:f32 = weights_as_matrix[(0,j)];
                    for k in 1..(number_of_clusters+1) {
                        let mut squarred_distance: f32 = 0.0;
                        for dimension in 0..dimensions_of_inputs {
                            squarred_distance += (inputs_to_predict[i*dimensions_of_inputs+dimension] - inputs_train[(k-1)*dimensions_of_inputs+dimension]) * (inputs_to_predict[i*dimensions_of_inputs+dimension] - inputs_train[(k-1)*dimensions_of_inputs+dimension]);
                        }
                        weighted_sum += weights_as_matrix[(k,j)]*exp(-gamma * squarred_distance);
                    }
                    if is_classification {
                        output.push(weighted_sum.tanh());
                    } else {
                        output.push(weighted_sum)
                    }
                }
                outputs.push(output);
            }
            let mut output_return = Vec::with_capacity(number_of_inputs_to_predict*number_of_classes);
            for l in 0..number_of_inputs_to_predict {
                for i in 0..number_of_classes {
                    output_return.push(outputs[l][i]);
                }
            }
            let arr_slice = output_return.leak();
            arr_slice.as_mut_ptr()

        } else {
            let mut vec_of_mu_k = k_means(inputs_train, number_of_clusters, dimensions_of_inputs, number_of_training_inputs);
            for i in 0..number_of_training_inputs {
                phi[(i, 0)] = 1f32;
                for k in 0..number_of_clusters {
                    let mut squarred_distance: f32 = 0.0;
                    for dimension in 0..dimensions_of_inputs {
                        squarred_distance += (inputs_train[i * dimensions_of_inputs + dimension] - vec_of_mu_k[k][dimension])*(inputs_train[i * dimensions_of_inputs + dimension] - vec_of_mu_k[k][dimension]);
                    }
                    phi[(i,k+1)] = exp(-gamma * squarred_distance);
                }
            }
            let phi_pseudo_inverse:DMatrix<f32> = matrix_pseudo_inverse(phi,number_of_clusters+1);
            let mut labels_as_matrix:DMatrix<f32> = DMatrix::zeros(number_of_training_inputs, number_of_classes);
            for i in 0..number_of_training_inputs {
                for j in 0..number_of_classes{
                    labels_as_matrix[(i, j)] = labels[i*number_of_classes+j];
                }
            }

            let weights_as_matrix:DMatrix<f32> = phi_pseudo_inverse * labels_as_matrix;

            let mut outputs:Vec<Vec<f32>> = Vec::with_capacity(number_of_inputs_to_predict);
            for i in 0..number_of_inputs_to_predict {
                let mut output:Vec<f32> = Vec::with_capacity(number_of_classes);
                for j in 0..number_of_classes {
                    let mut weighted_sum:f32 = weights_as_matrix[(0,j)];
                    for k in 1..(number_of_clusters+1) {
                        let mut squarred_distance: f32 = 0.0;
                        for dimension in 0..dimensions_of_inputs {
                            squarred_distance += (inputs_to_predict[i*dimensions_of_inputs+dimension] - vec_of_mu_k[k-1][dimension])
                                               * (inputs_to_predict[i*dimensions_of_inputs+dimension] - vec_of_mu_k[k-1][dimension]);
                        }
                        weighted_sum += weights_as_matrix[(k,j)]*exp(-gamma * squarred_distance);
                    }
                    if is_classification {
                        output.push(weighted_sum.tanh());
                    } else {
                        output.push(weighted_sum);
                    }
                }
                outputs.push(output);
            }
            let mut output_return = Vec::with_capacity(number_of_inputs_to_predict*number_of_classes);
            for l in 0..number_of_inputs_to_predict {
                for i in 0..number_of_classes {
                    output_return.push(outputs[l][i]);
                }
            }

            let arr_slice = output_return.leak();
            arr_slice.as_mut_ptr()

        }
    }
}


#[no_mangle]
extern "C" fn train_radial_basis_function_model(      pointer_to_inputs_train : *mut f32,
                                                number_of_training_inputs : usize,
                                                pointer_to_inputs_to_predict : *mut f32,
                                                number_of_inputs_to_predict : usize,
                                                dimensions_of_inputs : usize,
                                                pointer_to_labels : *mut f32,
                                                number_of_classes : usize,
                                                gamma : f32,
                                                is_classification : bool,
                                                number_of_clusters : usize,
                                                is_naive: bool
) -> *mut f32 {

    unsafe{
        let inputs_train = std::slice::from_raw_parts(pointer_to_inputs_train,
                                                      number_of_training_inputs * dimensions_of_inputs);
        let labels = std::slice::from_raw_parts(pointer_to_labels,
                                                number_of_training_inputs * number_of_classes);

        let inputs_to_predict = std::slice::from_raw_parts(pointer_to_inputs_to_predict,
                                                           number_of_inputs_to_predict * dimensions_of_inputs);

        let mut phi: DMatrix<f32> = DMatrix::zeros(number_of_training_inputs, number_of_clusters + 1);
        if is_naive && number_of_clusters == number_of_training_inputs {

            for i in 0..number_of_training_inputs {
                phi[(i, 0)] = 1f32;
                for j in 0..number_of_training_inputs {
                    let mut squarred_distance: f32 = 0.0;
                    for dimension in 0..dimensions_of_inputs {
                        squarred_distance += (inputs_train[i * dimensions_of_inputs + dimension] - inputs_train[j * dimensions_of_inputs + dimension]) * (inputs_train[i * dimensions_of_inputs + dimension] - inputs_train[j * dimensions_of_inputs + dimension]);
                    }
                    phi[(i, j + 1)] = exp(-gamma * squarred_distance);
                }
            }
            let phi_pseudo_inverse:DMatrix<f32> = matrix_pseudo_inverse(phi,number_of_clusters+1);
            let mut labels_as_matrix:DMatrix<f32> = DMatrix::zeros(number_of_training_inputs, number_of_classes);
            for i in 0..number_of_training_inputs {
                for j in 0..number_of_classes{
                    labels_as_matrix[(i, j)] = labels[i*number_of_classes+j];
                }
            }

            let weights_as_matrix:DMatrix<f32> = phi_pseudo_inverse * labels_as_matrix;

            let mut outputs:Vec<Vec<f32>> = Vec::with_capacity(number_of_inputs_to_predict);
            for i in 0..number_of_inputs_to_predict {
                let mut output:Vec<f32> = Vec::with_capacity(number_of_classes);
                for j in 0..number_of_classes {
                    let mut weighted_sum:f32 = weights_as_matrix[(0,j)];
                    for k in 1..(number_of_clusters+1) {
                        let mut squarred_distance: f32 = 0.0;
                        for dimension in 0..dimensions_of_inputs {
                            squarred_distance += (inputs_to_predict[i*dimensions_of_inputs+dimension] - inputs_train[(k-1)*dimensions_of_inputs+dimension]) * (inputs_to_predict[i*dimensions_of_inputs+dimension] - inputs_train[(k-1)*dimensions_of_inputs+dimension]);
                        }
                        weighted_sum += weights_as_matrix[(k,j)]*exp(-gamma * squarred_distance);
                    }
                    if is_classification {
                        output.push(weighted_sum.tanh());
                    } else {
                        output.push(weighted_sum)
                    }
                }
                outputs.push(output);
            }
            let mut output_return = Vec::with_capacity(number_of_inputs_to_predict*number_of_classes);
            for l in 0..number_of_inputs_to_predict {
                for i in 0..number_of_classes {
                    output_return.push(outputs[l][i]);
                }
            }
            let arr_slice = output_return.leak();
            arr_slice.as_mut_ptr()

        } else {
            let mut vec_of_mu_k = k_means(inputs_train, number_of_clusters, dimensions_of_inputs, number_of_training_inputs);
            for i in 0..number_of_training_inputs {
                phi[(i, 0)] = 1f32;
                for k in 0..number_of_clusters {
                    let mut squarred_distance: f32 = 0.0;
                    for dimension in 0..dimensions_of_inputs {
                        squarred_distance += (inputs_train[i * dimensions_of_inputs + dimension] - vec_of_mu_k[k][dimension])*(inputs_train[i * dimensions_of_inputs + dimension] - vec_of_mu_k[k][dimension]);
                    }
                    phi[(i,k+1)] = exp(-gamma * squarred_distance);
                }
            }
            let phi_pseudo_inverse:DMatrix<f32> = matrix_pseudo_inverse(phi,number_of_clusters+1);
            let mut labels_as_matrix:DMatrix<f32> = DMatrix::zeros(number_of_training_inputs, number_of_classes);
            for i in 0..number_of_training_inputs {
                for j in 0..number_of_classes{
                    labels_as_matrix[(i, j)] = labels[i*number_of_classes+j];
                }
            }

            let weights_as_matrix:DMatrix<f32> = phi_pseudo_inverse * labels_as_matrix;

            let mut outputs:Vec<Vec<f32>> = Vec::with_capacity(number_of_inputs_to_predict);
            for i in 0..number_of_inputs_to_predict {
                let mut output:Vec<f32> = Vec::with_capacity(number_of_classes);
                for j in 0..number_of_classes {
                    let mut weighted_sum:f32 = weights_as_matrix[(0,j)];
                    for k in 1..(number_of_clusters+1) {
                        let mut squarred_distance: f32 = 0.0;
                        for dimension in 0..dimensions_of_inputs {
                            squarred_distance += (inputs_to_predict[i*dimensions_of_inputs+dimension] - vec_of_mu_k[k-1][dimension])
                                               * (inputs_to_predict[i*dimensions_of_inputs+dimension] - vec_of_mu_k[k-1][dimension]);
                        }
                        weighted_sum += weights_as_matrix[(k,j)]*exp(-gamma * squarred_distance);
                    }
                    if is_classification {
                        output.push(weighted_sum.tanh());
                    } else {
                        output.push(weighted_sum);
                    }
                }
                outputs.push(output);
            }
            let mut output_return = Vec::with_capacity(number_of_inputs_to_predict*number_of_classes);
            for l in 0..number_of_inputs_to_predict {
                for i in 0..number_of_classes {
                    output_return.push(outputs[l][i]);
                }
            }

            let arr_slice = output_return.leak();
            arr_slice.as_mut_ptr()

        }
    }
}