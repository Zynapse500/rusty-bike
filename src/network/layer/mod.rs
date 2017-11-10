
mod node;
use self::node::Node;

use rand::{Rng};

/// Contains all the nodes
#[derive(Clone)]
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


    /// Initalizes this layer's weights to random values
    pub fn initialize_random<R: Rng>(&mut self, rng: &mut R) {
        for node in self.nodes.iter_mut() {
            node.randomize(rng);
        }
    }


    /// Determines how this layer's nodes should be adjusted based on the inputs, current activation and desired output
    pub fn get_adjustment(& self, input: &[f64], activation: &[f64], desired: &[f64]) -> Layer {
        let mut adjustment = Layer::new(input.len(), activation.len());

        for (n, node) in self.nodes.iter().enumerate() {
            let error = activation[n] - desired[n];

            adjustment.nodes[n] = node.get_adjustment(input, error);
        }

        adjustment
    }


    /// Adjusts this layer's nodes based on the values in another layer
    pub fn adjust(&mut self, adjustment: &Layer, multiplier: f64) {
        for (n, node) in self.nodes.iter_mut().enumerate() {
            node.adjust(&adjustment.nodes[n], multiplier);
        }
    }

    
    /// Returns how much this layer 'wants' each input to change in order to produce the desired output
    pub fn get_desired_input(&self, activation: &[f64], desired: &[f64]) -> Vec<f64> {
        let mut delta = vec![0.0; self.nodes[0].input_size()];

        for (n, node) in self.nodes.iter().enumerate() {
            let error = activation[n] - desired[n];

            let node_delta = node.get_input_delta(error);
            for i in 0..node_delta.len() {
                delta[i] += node_delta[i];
            }
        }

        delta
    }
}
