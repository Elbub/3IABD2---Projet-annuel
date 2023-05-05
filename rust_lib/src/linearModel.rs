pub mod linear_model{

    pub fn points_array(number_of_points: usize) -> Vec<(f32, f32)>{
        use rand::Rng;
        let mut vec_of_points = Vec::with_capacity(number_of_points);
        for _ in 0..number_of_points {
            let x: f32 = rand::thread_rng().gen_range(0f32..10f32);
            let y: f32 = rand::thread_rng().gen_range(0f32..10f32);
            let tuple_of_points = (x,y);
            vec_of_points.push(tuple_of_points);
        }

        vec_of_points
    }

    pub fn points_label(vec_of_points: &Vec<(f32, f32)>) -> Vec<(u32)>{
        let mut vec_of_labels = Vec::with_capacity(vec_of_points.len());
        for i in 0..vec_of_points.len() {
            if -vec_of_points[i].1 + vec_of_points[i].0 + 0.25 >= 0. {
                vec_of_labels.push(1);
            } else { vec_of_labels.push(0) }
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
}
