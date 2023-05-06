use rust_lib::linearModel;

#[no_mangle]
extern "C" fn main() {
    let trained_weights = linearModel::linear_model::trained_model(100);
    println!("{:?}",trained_weights);
}
