
mod node;
use self::node::Node;

/// Contains all the nodes
pub struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    /// Creates a new hidden layer
    pub fn new(previous_size: usize, size: usize) -> Layer {
        let mut nodes = Vec::with_capacity(size);

        for _ in 0..size {
            nodes.push(Node::with_weights(previous_size));
        }

        Layer {
            nodes
        }
    }


    /// Feeds data into the layer
    pub fn feed(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.nodes.len()];

        for (index, node) in self.nodes.iter().enumerate() {
            output[index] = node.feed(input);
        }

        output
    }
}
