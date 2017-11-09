

/// The basic neural building block
pub struct Node {
    weights: Vec<f64>,
    bias: f64
}

impl Node {
    /// Create a new node with specified number of weights
    pub fn with_weights(count: usize) -> Node {
        Node {
            weights: vec![0.0; count],
            bias: 0.0
        }
    }
}