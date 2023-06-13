use nalgebra::*;
pub fn patate() {
    //let uce = SMatrix::new(f32, 3, 2);
    let m = Matrix2::new(2.0, 3.0, 1.0, 5.0);
    println!("{}", m.pseudo_inverse(0.0).unwrap()); //hardcoding 10^-7 is probably not what you want here
    println!("{}", ().try_inverse().unwrap());
}



#[no_mangle]
extern "C" fn generate_random_w(layers_ptr: *mut usize, number_of_layers: usize, seed: u64) -> *mut f32 {
    unsafe {
        // À vérifier, mais une seed a l'air de généralement être du u64 en Rust.
        use rand::Rng;
        let mut rng = rand::thread_rng();   // À changer pour prendre en compte la seed fournie.

        let layers = Vec::from_raw_parts(layers_ptr, number_of_layers, number_of_layers);
        let mut total_number_of_weights = 0;

        for l in 0..(number_of_layers - 1) {
            total_number_of_weights += (layers[l] + 1) * (layers[l + 1] + 1);
        }

        let mut w: Vec<f32> = Vec::with_capacity(total_number_of_weights);
        for l in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
            w.push(rng.gen_range(0f32..1f32));
        }

        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}

#[no_mangle]
extern "C" fn multi_layer_perceptron_training(w_ptr: *mut f32,
                                              labels_ptr : *mut f32,
                                              vec_of_points_ptr: *mut f32,
                                              number_of_elements_and_labels: usize,
                                              dimension_of_elements: usize,
                                              number_of_classes_to_predict: usize,
                                              learning_rate: f32,
                                              epoch: usize,
                                              layers_ptr: *mut usize,
                                              number_of_layers: usize,
                                              is_classification: bool) -> *mut f32 {
    unsafe {
        // needed : number of layers and number of neurons per layer

        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(layers_ptr, number_of_layers);
        if layers[0] != dimension_of_elements + 1{
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] != number_of_classes_to_predict {
            panic!("Wrong number of neurons in the last layer.");
        }

        let mut total_number_of_weights = 0;
        for l in 0..(number_of_layers - 1) {
            total_number_of_weights += (layers[l] + 1) * (layers[l + 1] + 1);
        }
        let w_param = Vec::from_raw_parts(w_ptr, total_number_of_weights, total_number_of_weights);

        let mut w_index:usize = 0;
        let mut w: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        for l /*layer*/ in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
            let size_of_w_l = layers[l] + 1;
            let mut w_l: Vec<Vec<f32>> = Vec::with_capacity(size_of_w_l);
            for i in 0..size_of_w_l {
                let size_of_w_l_i = layers[l + 1] + 1;
                let mut w_l_i: Vec<f32> = Vec::with_capacity(size_of_w_l_i);
                for j in 0..size_of_w_l_i {
                    w_index += 1;
                    w_l_i.push(w_param[w_index]);
                }
                w_l.push(w_l_i);
            }
            w.push(w_l);
        }


        let vec_of_points = std::slice::from_raw_parts(vec_of_points_ptr,
                                                       number_of_elements_and_labels * dimension_of_elements);
        let labels = std::slice::from_raw_parts(labels_ptr,
                                                number_of_elements_and_labels * number_of_classes_to_predict);

        use rand::thread_rng;
        use rand::seq::SliceRandom;


        // REFACTOR : TO BE CONTINUED...

        for _ in 0..epoch {
            let mut randomly_ordered_dataset: Vec<usize> = (0..number_of_elements_and_labels).collect();
            randomly_ordered_dataset.shuffle(&mut thread_rng());

            for element in 0..number_of_elements_and_labels {
                let k: usize = randomly_ordered_dataset[element];

                let mut y_k:Vec<f32> = Vec::with_capacity(number_of_classes_to_predict);
                for class_number in 0..number_of_classes_to_predict {
                    y_k.push(labels[k * number_of_classes_to_predict + class_number]);
                }

                let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                let mut delta : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers);
                let size_of_x_0 = layers[0];
                let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
                x_0.push(1f32);
                for j in 1..size_of_x_0 {
                    x_0.push(vec_of_points[k * dimension_of_elements + j]);
                }
                x.push(x_0);
                delta.push(vec![0f32; size_of_x_0]);
                for l in 1..number_of_layers {
                    let size_of_x_l = layers[l];
                    let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                    x_l.push(1f32);
                    for j in 1..size_of_x_l {
                        let mut x_l_i = 0f32;
                        for i in 0..layers[l-1] {
                            x_l_i += w[l][i][j] * x[l-1][i];
                        }
                        x_l.push(x_l_i.tanh());
                    }
                    x.push(x_l);
                    delta.push(vec![0f32; size_of_x_l]);
                }


                let L = number_of_layers - 1;
                let size_of_delta_L = layers[L];
                for j in 1..size_of_delta_L {
                    if is_classification {
                        delta[L][j] = x[L][j] - y_k[j];
                    } else {
                        delta[L][j] = (1f32 - x[L][j] * x[L][j]) * (x[L][j] - y_k[j]);
                    }
                }
                for l in (number_of_layers - 1)..0 {
                    let size_of_delta_l = layers[l];
                    let mut delta_l: Vec<f32> = Vec::with_capacity(size_of_delta_l);
                    delta_l.push(1f32);
                    for j in 1..size_of_delta_l {
                        let mut delta_l_i = 0f32;
                        for i in 0..layers[l-1] {
                            delta_l_i += w[l][i][j] * delta[l-1][i];
                        }
                        delta_l.push(delta_l_i.tanh());
                    }
                    delta.push(delta_l);
                }


                for l_invert in 0..(number_of_layers - 1) {
                    let l = number_of_layers - 1 - l_invert;
                    
                }


                let mut signal: f32 = 0f32;
                for i in 0..dimension_of_elements + 1 {
                    signal += w[i] * x_k[i];
                }
                let mut g_x_k: f32 = 0.0;
                if signal >= 0f32 {
                    g_x_k = 1f32;
                }
                for i in 0..dimension_of_elements + 1 {
                    w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
                }
            }
        }
        let mut w_return = Vec::with_capacity(total_number_of_weights);
        for l /*layer*/ in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
            for i in 0..layers[l] + 1 {
                for j in 0..layers[l + 1] + 1 {
                    w_return.push(w[l][i][j]);
                }
            }
        }
        let arr_slice = w_return.leak();
        arr_slice.as_mut_ptr()
    }
}
