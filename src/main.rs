

extern crate rusty_bike;
use rusty_bike::network::Network;


fn main() {
    println!("Hello World!");

    let net = Network::with_shape(&[3, 3, 2]);
    let result = net.feed_forward(&[1.0, 2.0, 3.0]);
    
    println!("Result: {:?}", result);
}

