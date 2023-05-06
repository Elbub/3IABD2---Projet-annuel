#[no_mangle]
extern "C" fn my_add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
extern "C" fn my_sum(arr: *const i32, nb_elems: i32) -> i32 {
    let safe_arr =
        unsafe {
            std::slice::from_raw_parts(arr, nb_elems as usize)
        };

    safe_arr.iter().sum()
}


#[no_mangle]
extern "C" fn count_to_n(n: i32) -> *mut i32 {
    let mut v: Vec<i32> = (0..n).collect();

    let arr_slice = v.leak();

    arr_slice.as_mut_ptr()
}

#[no_mangle]
extern "C" fn delete_int_array(arr: *mut i32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}

struct MyMLP {
    nb_layer: usize,
    nb_neurons_per_layer: Vec<usize>,
    W: Vec<Vec<Vec<f32>>>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}


#[no_mangle]
extern "C" fn create_mlp_model(arr: *mut i32, arr_len: i32) -> *mut MyMLP {
    let model = Box::new(MyMLP {
        nb_layer: 0,   //TODO
        nb_neurons_per_layer: vec![], //TODO
        W: vec![], //TODO
        X: vec![], //TODO
        deltas: vec![], //TODO
    });
    //TODO : Initialization

    let leaked = Box::leak(model);
    leaked
}

// We could return losses and metrics too here if needed
// dataset inputs contains all the features of all training samples concatenated
// dataset outputs contains all the expected outputs of all training samples concatenated
#[no_mangle]
extern "C" fn train_mlp_model(model: *mut MyMLP, dataset_inputs: *const f32, lines: i32, columns: i32,
                              dataset_outputs: *const f32, output_columns: i32, alpha: f32, nb_iter: i32,
                              is_classification: bool) {
    //TODO : training
}

#[no_mangle]
extern "C" fn predict_mlp_model(model: *mut MyMLP, sample_inputs: *const f32, columns: i32,
                                is_classification: bool) -> *mut f32 {
    //TODO : Predict
    let fake_output = vec![1.0f32];

    fake_output.leak().as_mut_ptr()
}

#[no_mangle]
extern "C" fn delete_mlp_model(model: *mut MyMLP) {
    unsafe {
        Box::from_raw(model);
    }
}

#[no_mangle]
extern "C" fn delete_float_array(arr: *mut f32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}