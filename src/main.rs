
extern crate rusty_bike;
use rusty_bike::network::Network;


fn main() {
    let mut net = Network::with_shape(&[3, 16, 16, 2]);
    net.initialize_random();

    let result = net.feed_forward(&[1.0, 2.0, 3.0]);
    println!("Result: {:?}", result);

    { // Training
        let input = &[vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let target = &[vec![1.0, 0.0], vec![0.0, 1.0]];

        for _ in 0..1_000_000 {
            net.train(input, target);
        }
    }
    
    println!("Increasing: {:?}", net.feed_forward(&[1.0, 2.0, 3.0]));

    println!("Decreasing: {:?}", net.feed_forward(&[3.0, 2.0, 1.0]));
}

