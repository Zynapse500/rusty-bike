//! An abstraction over a multilayer neural network


mod layer;
use self::layer::Layer;

/// The 'Network'
pub struct Network {
    layers: Vec<Layer>
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

        Network {
            layers
        }
    }
}
