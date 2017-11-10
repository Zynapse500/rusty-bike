//! Contains helper functions for loading the MNIST database

use std::fs::File;
use std::io::{Read, Result, Error, ErrorKind};

extern crate byteorder;
use self::byteorder::{BigEndian, ReadBytesExt};

extern crate rand;
use self::rand::thread_rng;


pub struct Database {
    labels: Vec<Vec<f64>>,
    images: Vec<Vec<f64>>
}


impl Database {
    pub fn new(label_path: &str, image_path: &str) -> Self {
        // Load MNIST
        let labels: Vec<Vec<f64>> = get_labels(label_path).unwrap()
            .into_iter().map(|u|{let mut vec = vec![0.0; 10]; vec[u as usize] = 1.0; vec}).collect();

        let images: Vec<Vec<f64>> = (&get_images(image_path).unwrap()[..])
            .chunks(784).map(|slc|{slc.iter().map(|b|{*b as f64 / 255.0}).collect()}).collect();

        Database { labels, images }
    }


    pub fn random_batch(&self, batch_size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = thread_rng();

        let sample = rand::sample(&mut rng, 0..self.images.len(), batch_size);

        let mut labels = Vec::with_capacity(batch_size);
        let mut images = Vec::with_capacity(batch_size);

        for index in sample {
            labels.push(self.labels[index].clone());
            images.push(self.images[index].clone());
        }

        (labels, images)
    }
}


/// Load labels from file
fn get_labels(path: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let mut buf = &bytes[..];
    let magic_number = buf.read_u32::<BigEndian>().unwrap();
    if magic_number != 2049 {
        return Err(Error::new(ErrorKind::InvalidData, "Magic number not 2049"));
    }

    let label_count = buf.read_u32::<BigEndian>().unwrap();

    Ok((&buf[..label_count as usize]).to_vec())
}


/// Load images from file
fn get_images(path: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let mut buf = &bytes[..];
    let magic_number = buf.read_u32::<BigEndian>().unwrap();
    if magic_number != 2051 {
        return Err(Error::new(ErrorKind::InvalidData, "Magic number not 2049"));
    }

    let image_count = buf.read_u32::<BigEndian>().unwrap();

    let rows = buf.read_u32::<BigEndian>().unwrap();
    let columns = buf.read_u32::<BigEndian>().unwrap();

    if rows != 28 || columns != 28 {
        return Err(Error::new(ErrorKind::InvalidData, "Image size is not 28x28"));
    }

    Ok((&buf[..(image_count as usize * 784)]).to_vec())
}