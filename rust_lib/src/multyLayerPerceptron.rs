//
//
// #[no_mangle]
// extern "C" fn generate_random_w(layers_ptr: *mut usize, number_of_layers: usize, seed: u64) -> *mut f32 {
//     unsafe {
//         // À vérifier, mais une seed a l'air de généralement être du u64 en Rust.
//         use rand::Rng;
//         let mut rng = rand::thread_rng();   // À changer pour prendre en compte la seed fournie.
//
//         let layers = Vec::from_raw_parts(layers_ptr, number_of_layers, number_of_layers);
//         let mut total_number_of_weights = 0;
//
//         for l in 0..(number_of_layers - 1) {
//             total_number_of_weights += (layers[l] + 1) * (layers[l + 1] + 1);
//         }
//
//         let mut w: Vec<f32> = Vec::with_capacity(total_number_of_weights);
//         for l in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
//             w.push(rng.gen_range(0f32..1f32));
//         }
//
//         let arr_slice = w.leak();
//         arr_slice.as_mut_ptr()
//     }
// }
//
// #[no_mangle]
// extern "C" fn multi_layer_perceptron_training(w_ptr: *mut f32,
//                                     labels_ptr : *mut f32,
//                                     vec_of_points_ptr: *mut f32,
//                                     arr_size: usize,
//                                     arr_dimension: usize,
//                                     learning_rate: f32,
//                                     epoch: usize,
//                                     layers_ptr: *mut usize,
//                                     number_of_layers: usize) -> *mut f32 {
//     unsafe {
//         // needed : number of layers and number of neurons per layer
//
//         let layers = Vec::from_raw_parts(layers_ptr, number_of_layers, number_of_layers);
//         let mut total_number_of_weights = 0;
//         for l in 0..(number_of_layers - 1) {
//             total_number_of_weights += (layers[l] + 1) * (layers[l + 1] + 1);
//         }
//         let w_param = Vec::from_raw_parts(w_ptr, total_number_of_weights, total_number_of_weights);
//
//         let mut w_index:usize = 0;
//         let mut w: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_layers);
//         for l /*layer*/ in 0..(number_of_layers - 1) { // on calcule d'une couche à la suivante, donc on ne prend pas la première.
//             let size_of_w_l = (layers[l] + 1);
//             let mut w_l: Vec<Vec<f32>> = Vec::with_capacity(size_of_w_l);
//             for i in 0..size_of_w_l {
//                 let size_of_w_l_i = (layers[l + 1] + 1);
//                 let mut w_l_i: Vec<f32> = Vec::with_capacity(size_of_w_l_i);
//                 for j in 0..size_of_w_l_i {
//                     w_index += 1;
//                     w_l_i.push(w_param[w_index]);
//                 }
//                 w_l.push(w_l_i);
//             }
//             w.push(w_l);
//         }
//
//
//         // REFACTOR : TO BE CONTINUED...
//         use rand::Rng;
//         let vec_of_points = std::slice::from_raw_parts(vec_of_points_ptr,
//                                                        arr_size*arr_dimension);
//         let labels = std::slice::from_raw_parts(labels_ptr,
//                                                 arr_size);
//
//         let mut w = Vec::from_raw_parts(
//             w_ptr, arr_dimension + 1, arr_dimension + 1);
//
//         let mut rng = rand::thread_rng();
//
//         for _ in 0..epoch {
//             let k: usize = rng.gen_range(0..arr_size);
//
//             let y_k: f32 = labels[k];
//
//             let mut x_k: Vec<f32> = Vec::with_capacity(arr_dimension + 1);
//             x_k.push(1f32);
//             for i in 0..arr_dimension {
//                 x_k.push(vec_of_points[k*arr_dimension+i]);
//             }
//             let mut signal: f32 = 0f32;
//             for i in 0..arr_dimension + 1 {
//                 signal += w[i] * x_k[i];
//             }
//             let mut g_x_k: f32 = 0.0;
//             if signal >= 0f32 {
//                 g_x_k = 1f32;
//             }
//             for i in 0..arr_dimension + 1 {
//                 w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
//             }
//         }
//         let arr_slice = w.leak();
//         arr_slice.as_mut_ptr()
//     }
// }
