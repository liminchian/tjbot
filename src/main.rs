use crate::docling::DoclingModel;

mod docling;

mod activations;
mod embeddings;
mod llama;
mod nn;
mod transformers;

fn main() {
    let model = DoclingModel::build();
    println!("{model:?}");
}
