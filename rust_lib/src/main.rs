use rust_lib::linearModel;

fn main() {
    println!("Hello, world!");
    //crate::linearModel;
    let i_hope_this_works = linearModel::linear_model::points_array(10);
    let test_points_label = linearModel::linear_model::points_label(&i_hope_this_works);

    println!("{:?}",i_hope_this_works);
    println!("{:?}",test_points_label);

    linearModel::linear_model::create_chart(&i_hope_this_works);
}
