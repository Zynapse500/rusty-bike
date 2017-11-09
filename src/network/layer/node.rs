

/// The basic neural building block
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
        ReLU(sum)
    }
}



/// The activation function of the nodes
fn ReLU(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}
