
mod mnist;

extern crate rusty_bike;
use rusty_bike::network::Network;


fn main() {
    println!("Loading MNIST database...");
    let mnist_db = mnist::Database::new( "training_data/train-labels.mnist", "training_data/train-images.mnist");
    println!("Done!");

    let mut net = Network::with_shape(&[28*28, 128, 10]);
    net.initialize_random();

    {

        for i in 0..1_000_000 {
            // Training
            let (target, input) = mnist_db.random_batch(100);

            // println!("Traning gen {}...", i);
            net.train(&input, &target, 0.005);

            // Test accuracy
            if i % 100 == 0 {
                let (target, input) = mnist_db.random_batch(1000);
                
                let accuracy = net.test_accuracy(&input, &target, |res, des|{
                    argmax(res) == argmax(des)
                });

                println!("[{}] Accuracy: {:.2}%", i, accuracy * 100.0);
            }
        }

    }

    println!("Training complete!"); 

    let (target, input) = mnist_db.random_batch(1000);

    let accuracy = net.test_accuracy(&input, &target, |res, des|{
        argmax(res) == argmax(des)
    });
    println!("Accuracy: {:.1}%", accuracy * 100.0);
}


fn argmax(s: &[f64]) -> usize {
    let mut index = 0;

    for i in 1..s.len() {
        if s[i] > s[index] {
            index = i;
        }
    }

    index
}
