pub mod linear_model{

    pub fn points_array(number_of_points: usize) -> Vec<Vec<f32>>{
        use rand::Rng;
        let mut vec_of_points = Vec::with_capacity(number_of_points);
        for _ in 0..number_of_points {
            // That...
            // let x: f32 = rand::thread_rng().gen_range(0f32..10f32);
            // let y: f32 = rand::thread_rng().gen_range(0f32..10f32);
            // let mut coordinates : Vec<f32> = Vec::new();
            // coordinates.push(x);
            // coordinates.push(y);
            // ... is equivalent to that.
            let coordinates : Vec<f32> = Vec::from([rand::thread_rng().gen_range(0f32..10f32), rand::thread_rng().gen_range(0f32..10f32)]);
            vec_of_points.push(coordinates);

            // let tuple_of_points = (x,y);
            // vec_of_points.push(tuple_of_points);
        }
        vec_of_points
    }

    pub fn points_label(vec_of_points: &Vec<Vec<f32>>) -> Vec<f32>{
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

    pub fn create_chart(vec_of_points: &Vec<(f32, f32)>){
        use plotters::prelude::*;
        use plotters::coord::types::RangedCoordf32;

        let root = BitMapBackend::new("/Users/bouai/Downloads/test.png", (600,400)).into_drawing_area();
        root.fill(&RGBColor(240, 200, 200));
        let root = root.apply_coord_spec(Cartesian2d::<RangedCoordf32, RangedCoordf32>::new(
            0f32..1f32,
            0f32..1f32,
            (0..640, 0..480),
        ));

        let dot_and_label = |x: f32, y: f32| {
            return EmptyElement::at((x, y))
                + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
                + Text::new(
                format!("({:.2},{:.2})", x, y),
                (10, 0),
                ("sans-serif", 15.0).into_font(),
            );
        };

        for i in 0..vec_of_points.len() {
            root.draw(&dot_and_label(vec_of_points[i].0, vec_of_points[i].1));
        }
        root.present();
    }

    pub fn linear_model_training(labels : &Vec<f32>, points: &Vec<Vec<f32>>) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let number_of_parameters:usize = points[0].len();
        let mut w: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);   // w : contient les poids associés aux Xi
        for _ in 0..number_of_parameters + 1 {
            w.push(rng.gen_range(0f32..1f32)); // initialisation aléatoire entre 0 et 1
        }
        let learning_rate: f32 = 0.001;
        for i in 0..10_000 {
            let k: usize = rng.gen_range(0..labels.len());
            let y_k: f32 = labels[k];
            let mut x_k: Vec<f32> = Vec::with_capacity(number_of_parameters + 1);
            x_k.push(1f32);
            for i in 0..number_of_parameters {
                x_k.push(points[k][i]);
            }
            let mut signal: f32 = 0f32;
            for i in 0..number_of_parameters {
                signal += w[i] * x_k[i];
            }
            let mut g_x_k: f32 = 0.0;
            if signal >= 0f32 {
                g_x_k = 1f32;
            }
            for i in 0..number_of_parameters + 1 {
                w[i] += learning_rate * (y_k - g_x_k) * x_k[i];
            }
        }
        w
    }
}
