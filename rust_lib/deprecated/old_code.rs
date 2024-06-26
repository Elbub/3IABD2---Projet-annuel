// #[no_mangle]
// pub extern "C" fn free_points_array(points_ptr: *mut Vec<Vec<f32>>) {
//     unsafe {
//         Box::from_raw(points_ptr);
//     }
// }

/*
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
                                                arr_size as usize,
                                                arr_size as usize);

        //println!("{:?}", vec_of_points);

        let mut to_compare: Vec<f32> = Vec::with_capacity(arr_dimension);
        let mut count = 0;

        for point in vec_of_points {

            to_compare.push(point);
            count += 1;
            if count == arr_dimension{
                if -to_compare[1] + to_compare[0] + 0.25 >= 0. {
                    vec_of_labels.push(1f32);
                    count = 0;
                } else { vec_of_labels.push(0f32) }
                to_compare.clear();
            }
        }

        let arr_slice = vec_of_labels.leak();

        arr_slice.as_mut_ptr()
    }
}


 */

// #[no_mangle]
// extern "C" fn linear_model_training(labels : &Vec<f32>, points: &Vec<Vec<f32>>) -> Vec<f32> {
//     use rand::Rng;
//     let mut rng = rand::thread_rng();
//     let number_of_parameters:usize = points[0].len();
//     let mut w: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);   // w : contient les poids associés aux Xi
//     for _ in 0..number_of_parameters + 1 {
//         w.push(rng.gen_range(0f32..1f32)); // initialisation aléatoire entre 0 et 1
//     }
//     //println!("w: {:?}",w);
//     let learning_rate: f32 = 0.001;
//     for i in 0..10_000 {
//         let k: usize = rng.gen_range(0..labels.len());
//         //println!("k :{:?}",k);
//         let y_k: f32 = labels[k];
//         // println!("y_k: {:?}",y_k);
//         let mut x_k: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);
//         x_k.push(1f32);
//         for i in 0..number_of_parameters {
//             x_k.push(points[k][i]);
//         }
//         // println!("x_k: {:?}",x_k);
//         let mut signal: f32 = 0f32;
//         for i in 0..number_of_parameters+1 {
//             signal += w[i] * x_k[i];
//         }
//         // println!("signal: {:?}",signal);
//         let mut g_x_k: f32 = 0.0;
//         if signal >= 0f32 {
//             g_x_k = 1f32;
//         }
//         // println!("g_x_k: {:?}",g_x_k);
//         for i in 0..number_of_parameters + 1 {
//             w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
//         }
//         // println!("w: {:?}",w);
//     }
//     w
// }
//
//
// #[no_mangle]
// extern "C" fn trained_model(number_of_points: usize) -> Vec<f32>{
//     let points = points_array(number_of_points);
//     let label_points = points_label(&points);
//
//     linear_model_training(&label_points, &points)
// }
