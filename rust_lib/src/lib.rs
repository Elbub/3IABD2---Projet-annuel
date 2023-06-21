// mod multy_layer_perceptron;

use nalgebra::DMatrix;

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
extern "C" fn points_label(vec_of_points_ptr: *mut f32, arr_size: usize, arr_dimension: usize) -> *mut f32{

    unsafe {
        let mut vec_of_labels: Vec<f32> = Vec::with_capacity(arr_size);
        let vec_of_points = std::slice::from_raw_parts(vec_of_points_ptr,
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
// extern "C" fn linear_model_training(w_ptr: *mut f32, labels_ptr : *mut f32, vec_of_points_ptr: *mut f32, arr_size: usize, arr_dimension: usize, learning_rate: f32, epoch: usize, is_label_list: bool, number_of_classes: usize) -> *mut f32 {
extern "C" fn linear_model_training(w_ptr: *mut f32, labels_ptr : *mut f32, vec_of_points_ptr: *mut f32, arr_size: usize, arr_dimension: usize, learning_rate: f32, epoch: usize) -> *mut f32 {
    unsafe {
        use rand::Rng;
        let vec_of_points = std::slice::from_raw_parts(vec_of_points_ptr,
                                                       arr_size * arr_dimension);
        let labels = std::slice::from_raw_parts(labels_ptr,
                                                arr_size);
        let mut w = Vec::from_raw_parts(
            w_ptr, arr_dimension + 1, arr_dimension + 1);
        let mut rng = rand::thread_rng();
        for _ in 0..epoch {
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
        //     for _ in 0..epoch {
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
extern "C" fn find_w_linear_regression(x_ptr: *mut f32, y_ptr: *mut f32, nombre_lignes_x: usize, nombre_colonnes_x: usize, nombre_lignes_y:usize, nombre_colonnes_y: usize) -> *mut f32 {
    unsafe {
        // let mut w = Vec::with_capacity(nombre_lignes_x_et_y);
        use nalgebra::*;
        use rand::Rng;

        let x_vect = std::slice::from_raw_parts(x_ptr, nombre_lignes_x * nombre_colonnes_x);


        let y_vect = std::slice::from_raw_parts(y_ptr, nombre_lignes_y * nombre_colonnes_y);


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

        println!("matrice x = {x_mat}");
        println!("matrice y = {y_mat}");

        let x_transpose:DMatrix<f32> = x_mat.clone().transpose();

        println!("transposée de x = {x_transpose}");

        // let x_t_mult_x = x_mat.clone() * x_transpose;
        let mut x_t_mult_x = x_transpose.clone() * x_mat.clone();

        println!("x transposée fois x = {x_t_mult_x}");

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

        // let new_x_trans = x_mat.transpose();


        let inv_times_x_t = match inv_x_t_x {
            Some(inv) => inv * x_transpose.clone(),
            None => panic!("Non inversible"),
        };


        let result_matrix = inv_times_x_t * y_mat;

        let mut w: Vec<f32> = Vec::with_capacity(nombre_colonnes_x * nombre_colonnes_y + 1);

        for i in 0..nombre_colonnes_x+1 {
            for j in 0..nombre_colonnes_y {
                w.push(result_matrix[(i,j)]);
                println!("{:?}",result_matrix[(i,j)]);
            }
        }

        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn predict_linear_model(vec_to_predict_ptr: *const f32, trained_model_ptr: *mut f32, arr_size: usize, arr_dimension: usize) -> *mut f32{
    unsafe{
        let vec_to_predict = std::slice::from_raw_parts(vec_to_predict_ptr,
                                                       arr_size*arr_dimension);

        println!("{}",vec_to_predict.len());

        let trained_model = std::slice::from_raw_parts(trained_model_ptr, arr_dimension + 1);

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
extern "C" fn x_transpose_times_x(x_ptr: *mut f32, longueur_x: usize, colonnes_x: usize) -> *mut f32 {
    unsafe {
        use nalgebra::*;
        let x_vect = std::slice::from_raw_parts(x_ptr, longueur_x * colonnes_x);
        let mut x_mat:DMatrix<f32> = DMatrix::zeros(longueur_x, colonnes_x);

        for i in 0..(longueur_x) {
            for j in 0..(colonnes_x) {
                x_mat[(i, j)] = x_vect[i * colonnes_x + j];
            }
        }

        println!("{:?}",x_mat);
        let x_transpose:DMatrix<f32> = x_mat.clone().transpose();
        println!("{:?}",x_transpose);
        let x_t_mult_x = x_transpose.clone() * x_mat.clone();
        println!("{:?}",x_t_mult_x);

        let mut w: Vec<f32> = Vec::with_capacity(colonnes_x * colonnes_x);

        for i in 0..colonnes_x {
            for j in 0..colonnes_x {
                w.push(x_t_mult_x[(i,j)]);
                println!("{:?}",x_t_mult_x[(i,j)]);
            }
        }

        // let w: Vec<_> = x_mat.iter().cloned().collect();
        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}

#[no_mangle]
extern "C" fn get_number_of_w(layers_ptr: *mut f32, number_of_layers: usize) -> usize {
    unsafe {
        let layers = std::slice::from_raw_parts(layers_ptr, number_of_layers);
        let mut total_number_of_weights = 0.0;

        for l in 0..(number_of_layers - 1) {
            total_number_of_weights += (layers[l] + 1.0) * layers[l + 1];
        }
        // println!("{:?}", total_number_of_weights);
        total_number_of_weights as usize
    }
}

#[no_mangle]
extern "C" fn generate_random_mpl_w(layers_ptr: *mut f32, number_of_layers: usize) -> *mut f32 {
    unsafe {
        // À vérifier, mais une seed a l'air de généralement être du u64 en Rust.
        use rand::Rng;
        println!("hello there");
        let mut rng = rand::thread_rng();   // À changer pour prendre en compte la seed fournie.

        let layers = std::slice::from_raw_parts(layers_ptr, number_of_layers);
        println!("{:?}", layers);


        let total_number_of_weights = get_number_of_w(layers_ptr, number_of_layers);


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
        println!("{:?}", w);


        let arr_slice = w.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn multi_layer_perceptron_training(w_ptr: *mut f32,
                                              labels_ptr : *mut f32,
                                              inputs_ptr: *mut f32,
                                              number_of_inputs: usize,
                                              dimension_of_inputs: usize,
                                              number_of_classes_to_predict: usize,
                                              learning_rate: f32,
                                              epoch: usize,
                                              layers_ptr: *mut f32, // forme du perceptron, ex: (2, 2, 1)
                                              number_of_layers: usize, // nombre de couches
                                              is_classification: bool) -> *mut f32 {
    unsafe {

        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(layers_ptr, number_of_layers);
        if layers[0] as usize != dimension_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes_to_predict {
            panic!("Wrong number of neurons in the last layer.");
        }

        // println!("On est rentré dans la fonction");

        let mut total_number_of_weights = get_number_of_w(layers_ptr, number_of_layers); // = 9 pour XOR

        let w_param = std::slice::from_raw_parts(w_ptr, total_number_of_weights);

        let mut w_index:usize = 0;
        let mut w: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
        w.push(Vec::from(Vec::new())); // à chaque fois on veut avoir un w[l][i][j] avec rien dans notre couche l

        for l /*layer*/ in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première. XOR -> nb_layers = 3 donc l in 0..2
            let size_of_w_l: usize = layers[l] as usize + 1; // on a rajouté 1 pour le biais
            //l = 0 -> size_of_w_l = 3 , l=1 -> size_of_w_l = 3
            let mut w_l: Vec<Vec<f32>> = Vec::with_capacity(size_of_w_l);
            for i in 0..size_of_w_l { // l = 0 -> i in 0..3 , l=1 -> i in 0..3
                let size_of_w_l_i: usize = layers[l + 1] as usize; // = 2 / 1
                let mut w_l_i: Vec<f32> = Vec::with_capacity(size_of_w_l_i);
                for j in 0..size_of_w_l_i { // j in 0..2 / j in 0..1
                    w_l_i.push(w_param[w_index]); // w_param[0], w_param[1], w_param[2]
                    w_index += 1;
                }
                w_l.push(w_l_i); // w = [[w_param[0],w_param[1]],[w_param[2]]]
            }
            w.push(w_l);
        }


        let inputs_data = std::slice::from_raw_parts(inputs_ptr,
                                                       number_of_inputs * dimension_of_inputs);
        // inputs_ptr = [0, 0]
        //              [0, 1] dimension of inputs = 2
        //              [1, 0] number of inputs = 4
        //              [1, 1] donc on a bien un vecteur de taille  8
        let labels = std::slice::from_raw_parts(labels_ptr,
                                                number_of_inputs * number_of_classes_to_predict);
        // labels = [-1, 1, 1, -1]
        // len = 4 * 1
        use rand::thread_rng;
        use rand::seq::SliceRandom;


        for numero_epoch in 0..epoch {
            let mut randomly_ordered_dataset: Vec<usize> = (0..number_of_inputs).collect();
            randomly_ordered_dataset.shuffle(&mut thread_rng());

            // println!("randomly ordered dataset : {:?}",randomly_ordered_dataset);

            for k in randomly_ordered_dataset {

                let mut x : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers); // nb_layers = 3
                // x est la totalité de nos x

                let size_of_x_0: usize = layers[0] as usize + 1; // size_of_x_0 = 2 + 1 = 3
                let mut x_0: Vec<f32> = Vec::with_capacity(size_of_x_0);
                x_0.push(1f32);
                for j in 0..(size_of_x_0-1) { // 0..2
                    // x_0 = [1, 1, 0]
                    x_0.push(inputs_data[k * dimension_of_inputs + j]); // inputs = [0, 0, 0, 1, 1, 0, 1, 1]
                }
                x.push(x_0); // x = [[1, 1, 0]]



                let mut y_k:Vec<f32> = Vec::with_capacity(number_of_classes_to_predict + 1); // 2
                y_k.push(1f32);
                for class_number in 0..number_of_classes_to_predict { // 0..1
                    y_k.push(labels[k * number_of_classes_to_predict + class_number]); // push
                    // y_k = [1, 1]
                }

                // println!("y_k : {:?}",y_k);


                let mut delta : Vec<Vec<f32>> = Vec::with_capacity(number_of_layers); // 3
                delta.push(vec![0f32; size_of_x_0]);

                for l in 1..number_of_layers { // nb layers = 3 -> l=1, l=2
                    let size_of_x_l: usize = layers[l] as usize +1; // size of layer[2] = 2

                    let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l); // 2
                    x_l.push(1f32);
                    // x_1=[1, ]
                    for j in 1..size_of_x_l { // 1..3
                        let mut x_l_i = 0f32;
                        for i in 0..layers[l-1] as usize + 1{ // layers[1] = 2 + 1 = 3
                            x_l_i += w[l][i][j-1] * x[l-1][i];
                            // x_1_i = w[1][0][0] * x[0][0]   x_2_i = w[2][0][0] * x[1][0]
                            //       + w[1][1][0] * x[0][1]         + w[2][1][0] * x[1][1]
                            //       + w[1][2][0] * x[0][2]         + w[2][2][0] * x[1][2]
                        }
                        // println!("x_l_i = {:?}", x_l_i);
                        // println!("x_l_i.tanh = {:?}", x_l_i.clone().tanh());
                        // si on est en régression et sur la derniere couche, on fait un truc spécial, sinon comme d'hab
                        if !is_classification && l==number_of_layers-1 {
                            x_l.push(x_l_i);
                        } else {
                            x_l.push(x_l_i.tanh());
                            // x_1 = [1, 0.8, 1, -0.7]
                        }
                        // println!("x_l : {:?}", x_l);
                    }
                    x.push(x_l);
                    // x = [x_1]
                    delta.push(vec![0f32; size_of_x_l]);
                }
                // println!("x : {:?}",x);
                let L = number_of_layers - 1; // L = 2
                let size_of_delta_L = layers[L] as usize + 1; // == 2
                for j in 1..size_of_delta_L{ // j in 1..2
                    if is_classification {
                        // delta[L-j+1][j-1] = (1f32 - x[L-j+1][j] * x[L-j+1][j]) * (x[L-j+1][j] - y_k[j]);
                        delta[L][j] = (1f32 - x[L][j] * x[L][j]) * (x[L][j] - y_k[j]);
                    } else {
                        // delta[L-j+1][j-1] = x[L-j+1][j] - y_k[j];
                        delta[L][j] = x[L][j] - y_k[j];
                    }
                }

                for l in (1..number_of_layers).rev() { // l in 1..2 -> l = 1
                    // println!("l value {:?}", l);
                    for i in 0..layers[l - 1] as usize + 1{ // i in 0..3
                        let mut weighed_sum_of_errors = 0f32;
                        for j in 1..layers[l] as usize + 1{ // j in 1..3
                            weighed_sum_of_errors += w[l][i][j-1] * delta[l][j];
                        }
                        delta[l-1][i] = (1f32 - x[l - 1][i] * x[l - 1][i]) * weighed_sum_of_errors;
                    }
                }

                for l in 1..number_of_layers { // 3 layers
                    for i in 0..layers[l - 1] as usize + 1{ // i in 0..3
                        for j in 1..layers[l] as usize + 1{ // j in 1..2
                            w[l][i][j-1] -= learning_rate * x[l - 1][i] * delta[l][j];

                            // l = 1, i = 0 ; j = 1
                            // w[1][0][1] -= lr * x[0][0] * delta[1][1]
                        }
                    }
                }
            }
        }
        let mut w_return = Vec::with_capacity(total_number_of_weights);
        // total_number_of_weights = 3
        for l /*layer*/ in 1..number_of_layers { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
            // number of layers = 3
            // l = 1
            for i in 0..layers[l-1] as usize + 1 { // i in 0..3
                for j in 1..layers[l] as usize + 1{ // j in 0..2
                    w_return.push(w[l][i][j-1]);
                }
            }
        }
        let arr_slice = w_return.leak();
        arr_slice.as_mut_ptr()
    }
}


#[no_mangle]
extern "C" fn multi_layer_perceptron_predict( w_ptr: *mut f32, // c'est le w entrainé
                                              inputs_ptr: *mut f32, // les données qu'on veut prédire
                                              number_of_inputs: usize, // le nombre de données dans le pointeur d'au dessus
                                              dimension_of_inputs: usize, // dimension des inputs
                                              number_of_classes_to_predict: usize,
                                              layers_ptr: *mut f32, // forme du perceptron, ex: (2, 2, 1)
                                              number_of_layers: usize, // nombre de couches
                                              is_classification: bool) -> *mut f32 {
    unsafe {

        if number_of_layers < 2 {
            panic!("Not enough layers.");
        }
        let layers = std::slice::from_raw_parts(layers_ptr, number_of_layers);
        if layers[0] as usize != dimension_of_inputs {
            panic!("Wrong number of neurons in the first layer.");
        }
        if layers[number_of_layers - 1] as usize != number_of_classes_to_predict {
            panic!("Wrong number of neurons in the last layer.");
        }


        let mut total_number_of_weights = get_number_of_w(layers_ptr, number_of_layers);

        let w_param = std::slice::from_raw_parts(w_ptr, total_number_of_weights);

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

        let inputs_data = std::slice::from_raw_parts(inputs_ptr,
                                                     number_of_inputs * dimension_of_inputs);
        // inputs_ptr = [0, 0]
        //              [0, 1] dimension of inputs = 2
        //              [1, 0] number of inputs = 4
        //              [1, 1] donc on a bien un vecteur de taille  8

        // labels = [-1, 1, 1, -1]
        // len = 4 * 1

        let mut Y : Vec<f32> = Vec::with_capacity(number_of_inputs*number_of_classes_to_predict);

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

            // println!("x : {:?}",x);


            for l in 1..number_of_layers { // nb layers = 2
                let size_of_x_l: usize = layers[l] as usize; // size of layer[1] = 1
                // println!("size of x_l: {:?}", size_of_x_l);
                let mut x_l: Vec<f32> = Vec::with_capacity(size_of_x_l);
                x_l.push(1f32);
                for j in 0..size_of_x_l {
                    let mut x_l_i = 0f32;
                    for i in 0..layers[l-1] as usize + 1{ // layers[0] = 2 + 1 = 3
                        x_l_i += w[l][i][j] * x[l-1][i];
                    }
                    // println!("x_l_i = {:?}", x_l_i);
                    // println!("x_l_i.tanh = {:?}", x_l_i.clone().tanh());
                    if !is_classification && l==number_of_layers-1 {
                        x_l.push(x_l_i);
                    } else {
                        x_l.push(x_l_i.tanh());
                    }
                    // println!("x_l : {:?}", x_l);
                }
                x.push(x_l);
            }
            // println!("x : {:?}",x);

            for i in 1..number_of_classes_to_predict + 1{
                Y.push(x[number_of_layers-1][i]);
            }

        }
        // println!("Y de la fin: {:?}",Y);
        let arr_slice = Y.leak();
        arr_slice.as_mut_ptr()
    }
}


// #[no_mangle]
// // extern "C" fn mlp_training(layers_ptr: *mut usize, number_of_layers: usize, seed: u64) -> *mut f32 {
// extern "C" fn mlp_training(layers_ptr: *mut usize, number_of_layers: usize) -> *mut f32 {
//     let w = multy_layer_perceptron::generate_random_w(layers_ptr, number_of_layers);
//     w
//
// }
//
// #[no_mangle]
// extern "C" fn trained_model(number_of_points: usize) -> Vec<f32>{
//     let points = points_array(number_of_points);
//     let label_points = points_label(&points);
//
//     linear_model_training(&label_points, &points)
// }
