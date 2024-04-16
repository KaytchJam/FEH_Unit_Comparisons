use crate::stats_problems::stats_functions;
use crate::stats_structs::vec_f;

pub mod ml_learn {
    use crate::{stats_problems::stats_functions::*, stats_structs::vec_f::VecF};

    pub fn hello_world() {
        print!("Hello world.\n");
    }

    pub fn example_vector() {
        // the data
        let d1: Vec<i32> = vec![1, 2, 4, 5, 6, 7, 8, 1, 1, 4, 4, 5, 7, 8, 2, 4, 5, 6, 5, 5, 3, 9, 9, 2];
        let d2: Vec<i32> = vec![2, 3, 3, 3, 1, 7, 8, 1, 1, 3, 3, 5, 7, 8, 2, 4, 5, 6, 5, 5, 3, 9, 9, 2];
        let d3: Vec<i32> = vec![2, 1, 4, 1, 6, 7, 5, 1, 1, 3, 3, 5, 7, 8, 2, 4, 5, 6, 5, 5, 3, 9, 9, 2];
        let d4: Vec<i32> = vec![10, 11, 21, 10, 61, 74, 51, 13, 10, 11, 21, 88, 72, 72, 74, 88, 88, 10, 11, 61, 74, 21, 21, 21];
        let all_d: Vec<i32> = arbitrary_union(&[&d1, &d2, &d3, &d4]);

        // Vectorize our distributions
        let v1: VecF = VecF::from(gen_event_prob_map(&d1).iter().map(|t| *t.1).collect());
        let v2: VecF = VecF::from(gen_event_prob_map(&d2).iter().map(|t| *t.1).collect());
        let v3: VecF = VecF::from(gen_event_prob_map(&d3).iter().map(|t| *t.1).collect());
        let v4: VecF = VecF::from(gen_event_prob_map(&d4).iter().map(|t| *t.1).collect());

        // Printing out the sizes of our vectors
        print!("v1 size: {}, v2 size: {}\n", v1.size(), v2.size());
        print!("v1: {:?}\nv2: {:?}\nv3: {:?}\nv4: {:?}\n", v1, v2, v3, v4);
        println!();

        // Checking the distances of our distributions converted to vectors
        print!("v1 & v2 distance: {:?}\n", v1.euclid_distance(&v2));
        print!("v1 & v3 distance: {:?}\n", v1.euclid_distance(&v3));
        print!("v1 & v4 distance: {:?}\n", v1.euclid_distance(&v4));
        print!("v2 & v3 distance: {:?}\n", v2.euclid_distance(&v3));
        print!("v2 & v4 distance: {:?}\n", v2.euclid_distance(&v4));
        print!("v3 & v4 distance: {:?}\n", v3.euclid_distance(&v4));
        println!();

        // Checking mutual information of all our data sets
        print!("d1 & d2 mutual information: {:?}\n", mutual_information(&d1, &d2));
        print!("d1 & d3 mutual information: {:?}\n", mutual_information(&d1, &d3));
        print!("d1 & d4 mutual information: {:?}\n", mutual_information(&d1, &d4));
        print!("d2 & d3 mutual information: {:?}\n", mutual_information(&d2, &d3));
        print!("d2 & d4 mutual information: {:?}\n", mutual_information(&d2, &d4));
        print!("d3 & d4 mutual information: {:?}\n", mutual_information(&d3, &d4));
        println!();

        // Mutual information with all_d
        print!("d1 & all_d mutual information: {:?}\n", mutual_information(&d1, &all_d));
        print!("d2 & all_d mutual information: {:?}\n", mutual_information(&d2, &all_d));
        print!("d3 & all_d mutual information: {:?}\n", mutual_information(&d3, &all_d));
        print!("d4 & all_d mutual information: {:?}\n", mutual_information(&d4, &all_d));
        println!();
    }

    pub fn vector_fun() {
        const SIZE: usize = 4;
        let one = VecF::ones(SIZE);
        let two = VecF::ns(2f32, SIZE);
        let three = VecF::ns(3f32, SIZE);

        let items = [one, two, three];
        let mut sum = VecF::zeroes(SIZE);
        for vec in items.into_iter() {
            sum = sum + vec;
        }
        
        print!("{:?}\n", sum);
        print!("normalized: {:?}\n", sum.normalize());
        print!("back to normal: {:?}\n", sum.magnitude() * sum.normalize());
    }

    pub fn example_projection() {
        let foo = VecF::from(vec![3., 2., 1.]);
        let foo2 = VecF::from(vec![35., 13., 22.]);

        let p = VecF::project(&foo2, &foo);
        print!("{:?}\n", p);
        print!("{:?}\n", foo2 - p);
    }
}