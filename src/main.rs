

extern crate rusty_bike;
use rusty_bike::network::Network;


fn main() {
    println!("Hello World!");

    let net = Network::with_shape(&[3, 128, 412, 10]);
    let result = net.feed_forward(&[1.0, 0.5, 1.25]);
    
    println!("Result: {:?}", result);
}

