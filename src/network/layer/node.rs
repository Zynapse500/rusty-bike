
use rand::{Rng};


/// The basic neural building block
#[derive(Clone)]
pub struct Node {
    weights: Vec<f64>,
    bias: f64,
}

impl Node {
    /// Create a new node with specified number of weights
    pub fn with_weights(count: usize) -> Node {
        Node {
            weights: vec![1.0; count],
            bias: 0.0,
        }
    }


    /// Return the number of weights
    pub fn input_size(&self) -> usize {
        self.weights.len()
    }


    /// Calculates this node's activation from inputs
    pub fn feed(&self, input: &[f64]) -> f64 {
        if self.weights.len() != input.len() {
            panic!("Input data size does not match node input size!");
        }

        let mut sum = self.bias;

        // calculate the weighted sum
        for i in 0..input.len() {
            sum += self.weights[i] * input[i];
        }

        // Apply the activation function
        re_lu(sum)
    }


    /// Randomizes this node's weights
    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        let input_size = self.weights.len() as f64;

        for weight in self.weights.iter_mut() {
            *weight = rng.gen_range(0.01, 1.0) * (2.0 / input_size).sqrt();
        }
    }


    /// Determines how this node's weights should be adjusted based on the inputs and the error
    pub fn get_adjustment(& self, input: &[f64], error: f64) -> Node {
        let n = self.weights.len();
        let mut adjustment = Node::with_weights(n);

        for (i, _) in self.weights.iter().enumerate() {
            let gradient = input[i] * error;
            adjustment.weights[i] = gradient;
            adjustment.bias = error;
        }

        adjustment
    }

    /// Adjusts this layer's nodes based on the values in another layer
    pub fn adjust(&mut self, adjustment: &Node, learning_rate: f64) {

        for (n, weight) in self.weights.iter_mut().enumerate() {
            *weight -= adjustment.weights[n] * learning_rate;
        }
        self.bias -= adjustment.bias * learning_rate;
    }


    /// Returns how much this node 'wants' each input to change in order to produce the desired output
    pub fn get_input_delta(&self, error: f64) -> Vec<f64> {
        let mut delta = vec![0.0; self.input_size()];

        for (n, weight) in self.weights.iter().enumerate() {
            delta[n] = weight * error;
        }

        delta
    }
}


/// The activation function of the nodes
fn re_lu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}
