use rand::Rng;

pub struct MyLinearModel{
    w:  *mut f32,
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
extern "C" fn points_label(vec_of_points_ptr: *mut f32, arr_size: usize, arr_dimension: usize) -> *mut f32{
    // for i in 0..vec_of_points.len() {
    //     if -vec_of_points[i][1] + vec_of_points[i][0] + 0.25 >= 0. {     //-y + x + 0.25 = 0
    //         vec_of_labels.push(1f32);
    //     } else { vec_of_labels.push(0f32) }
    // }
    unsafe {
        let mut vec_of_labels: Vec<f32> = Vec::with_capacity(arr_size);
        let vec_of_points = Vec::from_raw_parts(vec_of_points_ptr,
                                                arr_size*arr_dimension,
                                                arr_size*arr_dimension);

        //println!("{:?}", vec_of_points);

        let mut to_compare: Vec<f32> = Vec::with_capacity(arr_dimension);
        let mut count = 0;

        for point in vec_of_points {

            to_compare.push(point);
            count += 1;
            if count == arr_dimension{
                if -to_compare[1] + to_compare[0] + 0.25 >= 0. {
                    vec_of_labels.push(1f32);
                } else { vec_of_labels.push(0f32) }
                count = 0;
                to_compare.clear();
            }
        }

        // println!("label from points_label {:?}\n", vec_of_labels);
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
pub extern "C" fn create_linear_model(arr: *mut i32, arr_len: i32, dimension: usize) -> *mut MyLinearModel {
    let model = Box::new(MyLinearModel {
        w: generate_random_w(dimension),
    });

    let leaked = Box::leak(model);
    leaked

}

#[no_mangle]
extern "C" fn train_linear_model(model: *mut MyLinearModel, labels_ptr : *mut f32, vec_of_points_ptr: *mut f32, arr_size: usize, arr_dimension: usize, learning_rate: f32) -> *mut MyLinearModel {
    unsafe {
        let vec_of_points = Vec::from_raw_parts(vec_of_points_ptr,
                                                arr_size*arr_dimension,
                                                arr_size*arr_dimension);

        let labels = Vec::from_raw_parts(labels_ptr,
                                                arr_size,
                                                arr_size);

        let mut w = Vec::from_raw_parts(
            model.w, arr_dimension + 1, arr_dimension +1
        );

        let mut rng = rand::thread_rng();

        for _ in 0..100 {
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
        model.w = arr_slice.as_mut_ptr();
        model
    }
}

#[no_mangle]
extern "C" fn delete_linear_model(model: *mut MyLinearModel) {
    unsafe {
        Box::from_raw(model);
    }
}

