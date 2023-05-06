use rust_lib::linearModel;

fn main() {
    // println!("Hello, world!");
    // //crate::linearModel;
    // let i_hope_this_works = linearModel::linear_model::points_array(1000);
    // let test_points_label = linearModel::linear_model::points_label(&i_hope_this_works);
    //
    // // println!("{:?}",i_hope_this_works);
    // // println!("{:?}",test_points_label);
    //
    // //linearModel::linear_model::create_chart(&i_hope_this_works);
    // let w:Vec<f32> = linearModel::linear_model::linear_model_training(&test_points_label, &i_hope_this_works);
    //
    // println!("{:?}",w);

    let trained_weights = linearModel::linear_model::trained_model(100);

    println!("{:?}",trained_weights);
}
