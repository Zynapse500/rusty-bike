//! An abstraction over a multilayer neural network

mod layer;
use self::layer::Layer;

use rand::thread_rng;

/// The 'Network'
pub struct Network {
    layers: Vec<Layer>,
}


impl Network {
    /// Creates a new network with a specified shape
    pub fn with_shape(shape: &[usize]) -> Network {
        if shape.len() < 2 {
            panic!("Network must consist of atleast 2 layers!");
        }

        let mut layers = Vec::new();

        // Create hidden layers
        for i in 1..shape.len() {
            layers.push(Layer::new(shape[i - 1], shape[i]));
        }

        Network { layers }
    }


    /// Feeds data into the network and collects it's outputs
    pub fn feed_forward(&self, input: &[f64]) -> Vec<f64> {
        let mut data: Vec<f64> = Vec::from(input);

        for layer in self.layers.iter() {
            data = layer.feed(&data);
        }

        data
    }


    /// Initalizes this network's weights to random values
    pub fn initialize_random(&mut self) {
        let mut rng = thread_rng();

        for layer in self.layers.iter_mut() {
            layer.initialize_random(&mut rng);
        }
    }


    /// Trains this network on some input and target data
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], learning_rate: f64) {
        if inputs.len() != targets.len() {
            panic!("Input size different from target size!");
        }

        let mut adjusted_layers = self.layers.clone();


        for batch in 0..inputs.len() {

            // Compute layer activations
            let mut activations: Vec<Vec<f64>> = vec![inputs[batch].clone()];

            for layer in self.layers.iter() {
                let activation = {
                    let data = &activations[activations.len() - 1];
                    layer.feed(data)
                };
                activations.push(activation);
            }

            // Back propagate
            let mut desired = targets[batch].clone();
            for layer_index in (1..activations.len()).rev() {
                let input = &activations[layer_index - 1];
                let activation = &activations[layer_index];
                let layer = &self.layers[layer_index - 1];

                let adjustment = layer.get_adjustment(input, activation, &desired);

                adjusted_layers[layer_index - 1].adjust(&adjustment, learning_rate / input.len() as f64);

                let desired_delta = layer.get_desired_input(activation, &desired);
                desired.clear();
                for i in 0..desired_delta.len() {
                    desired.push(input[i] - desired_delta[i] / inputs.len() as f64);
                }
            }
        }

        self.layers = adjusted_layers;
    }


    /// Test the network's accuracy on some tests
    pub fn test_accuracy<F: FnMut(&[f64], &[f64]) -> bool>(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], mut predicate: F) -> f64 {
        if inputs.len() != targets.len() {
            panic!("Input size different from target size!");
        }

        let mut correct_guesses = 0;

        for batch in 0..inputs.len() {

            // Compute layer activations
            let activation = self.feed_forward(&inputs[batch]);

            let correct = predicate(&activation, &targets[batch]);

            if correct {
                correct_guesses += 1;
            }
        }

        correct_guesses as f64 / inputs.len() as f64
    }
}
