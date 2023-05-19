//pub mod linearModel;
#[no_mangle]
extern "C" fn hell_world() {
    println!("Hello World");
}

#[no_mangle]
extern "C" fn array_test(size: usize, inner_size: usize) -> *mut *mut i32 {
    let mut outer_vec = Vec::with_capacity(size);

    for _ in 0..size {
        let mut inner_vec = Vec::with_capacity(inner_size);

        for j in 0..inner_size {
            inner_vec.push(j as i32);
        }
        outer_vec.push(inner_vec);
    }

    println!("{:?}", outer_vec);

    let mut outer_ptr_vec = Vec::with_capacity(size);
    for inner_vec in outer_vec {
        let mut inner_slice = inner_vec.into_boxed_slice();
        let inner_ptr = inner_slice.as_mut_ptr();
        std::mem::forget(inner_slice);
        outer_ptr_vec.push(inner_ptr);
    }

    let mut arr_slice = outer_ptr_vec.into_boxed_slice();
    let ptr = arr_slice.as_mut_ptr();

    std::mem::forget(arr_slice);

    ptr
}


#[no_mangle]
pub extern "C" fn another_points_array(number_of_points: usize) -> *mut f32{
    use rand::Rng;
    let mut vec_of_points = Vec::with_capacity(number_of_points);
    for _ in 0..number_of_points{
        let coordinates : f32 = rand::thread_rng().gen_range(0f32..10f32);
        vec_of_points.push(coordinates);
    }
    println!("{:?}", vec_of_points);

    let arr_slice = vec_of_points.leak();

    arr_slice.as_mut_ptr()

}

    #[no_mangle]
    pub extern "C" fn points_array(number_of_points: usize) -> *mut *mut f32 {
        use rand::Rng;
        unsafe {
            let mut vec_of_points = Vec::with_capacity(number_of_points);
            for _ in 0..number_of_points {
                let coordinates: Vec<f32> = vec![
                    rand::thread_rng().gen_range(0f32..10f32),
                    rand::thread_rng().gen_range(0f32..10f32),
                ];
                println!("{:?}", coordinates);
                let arr_ptr = Box::into_raw(coordinates.into_boxed_slice()) as *mut f32;
                vec_of_points.push(arr_ptr);
            }

            let arr_slice = vec_of_points.leak();
            let ptr = arr_slice.as_mut_ptr();

            ptr
        }
    }

/*
#[no_mangle]
pub extern "C" fn free_points_array(points_ptr: *mut Vec<Vec<f32>>) {
    unsafe {
        Box::from_raw(points_ptr);
    }
}


#[no_mangle]
extern "C" fn points_label(vec_of_points: &Vec<Vec<f32>>) -> Vec<f32>{
    let mut vec_of_labels:Vec<f32> = Vec::with_capacity(vec_of_points.len());
    // for i in 0..vec_of_points.len() {
    //     if -vec_of_points[i][1] + vec_of_points[i][0] + 0.25 >= 0. {     //-y + x + 0.25 = 0
    //         vec_of_labels.push(1f32);
    //     } else { vec_of_labels.push(0f32) }
    // }
    for point in vec_of_points {
        if -point[1] + point[0] + 0.25 >= 0. {
            vec_of_labels.push(1f32);
        } else { vec_of_labels.push(0f32) }
    }
    vec_of_labels
}

#[no_mangle]
extern "C" fn linear_model_training(labels : &Vec<f32>, points: &Vec<Vec<f32>>) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let number_of_parameters:usize = points[0].len();
    let mut w: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);   // w : contient les poids associés aux Xi
    for _ in 0..number_of_parameters + 1 {
        w.push(rng.gen_range(0f32..1f32)); // initialisation aléatoire entre 0 et 1
    }
    //println!("w: {:?}",w);
    let learning_rate: f32 = 0.001;
    for i in 0..10_000 {
        let k: usize = rng.gen_range(0..labels.len());
        //println!("k :{:?}",k);
        let y_k: f32 = labels[k];
        // println!("y_k: {:?}",y_k);
        let mut x_k: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);
        x_k.push(1f32);
        for i in 0..number_of_parameters {
            x_k.push(points[k][i]);
        }
        // println!("x_k: {:?}",x_k);
        let mut signal: f32 = 0f32;
        for i in 0..number_of_parameters+1 {
            signal += w[i] * x_k[i];
        }
        // println!("signal: {:?}",signal);
        let mut g_x_k: f32 = 0.0;
        if signal >= 0f32 {
            g_x_k = 1f32;
        }
        // println!("g_x_k: {:?}",g_x_k);
        for i in 0..number_of_parameters + 1 {
            w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
        }
        // println!("w: {:?}",w);
    }
    w
}


#[no_mangle]
extern "C" fn trained_model(number_of_points: usize) -> Vec<f32>{
    let points = points_array(number_of_points);
    let label_points = points_label(&points);

    linear_model_training(&label_points, &points)
}

*/