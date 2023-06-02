mod multyLayerPerceptron;

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
extern "C" fn linear_model_training(w_ptr: *mut f32, labels_ptr : *mut f32, vec_of_points_ptr: *mut f32, arr_size: usize, arr_dimension: usize, learning_rate: f32, epoch: usize) -> *mut f32 {
    unsafe {
        use rand::Rng;
        let vec_of_points = std::slice::from_raw_parts(vec_of_points_ptr,
                                                arr_size*arr_dimension);
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
                x_k.push(vec_of_points[k*arr_dimension+i]);
            }
            let mut signal: f32 = 0f32;
            for i in 0..arr_dimension + 1 {
                signal += w[i] * x_k[i];
            }
            let mut g_x_k: f32 = 0.0;
            if signal >= 0f32 {
                g_x_k = 1f32;
            }
            for i in 0..arr_dimension + 1 {
                w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
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
            if vec_of_coordinates[0] * trained_model[1] + vec_of_coordinates[1] * trained_model[2] + vec_of_coordinates[2] * trained_model[3] + trained_model[0] >= 0.0 {
                predicted_labels.push(1.0);
            } else {
                predicted_labels.push(0.0);
            }
        }
        predicted_labels.leak().as_mut_ptr()
    }
}

//
// #[no_mangle]
// extern "C" fn trained_model(number_of_points: usize) -> Vec<f32>{
//     let points = points_array(number_of_points);
//     let label_points = points_label(&points);
//
//     linear_model_training(&label_points, &points)
// }
