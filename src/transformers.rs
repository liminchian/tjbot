use crate::{
    activations::Activation,
    nn::{Attention, Conv2d, Embedding, Linear, Padding},
};

#[derive(Debug)]
pub struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: VisionEncoder,
    post_layernorm: LayerNorm,
}

impl VisionTransformer {
    pub fn build() -> Self {
        let embeddings = VisionEmbeddings::build();
        let encoder = VisionEncoder::build();
        let post_layernorm = LayerNorm {
            in_features: 768,
            eps: 1e-6,
            elementwise_affine: true,
        };

        Self {
            embeddings,
            encoder,
            post_layernorm,
        }
    }
}

#[derive(Debug)]
pub struct VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
}

impl VisionEmbeddings {
    pub fn build() -> Self {
        Self {
            patch_embedding: Conv2d {
                in_features: 3,
                out_features: 768,
                kernel_size: [16, 16],
                stride: [16, 16],
                padding: Padding::Valid,
            },
            position_embedding: Embedding {
                in_features: 1024,
                out_features: 768,
                padding_idx: None,
            },
        }
    }
}

#[derive(Debug)]
pub struct VisionEncoder(Vec<VisionEncoderLayer>);

impl VisionEncoder {
    pub fn build() -> Self {
        Self(
            [..12]
                .map(|_| VisionEncoderLayer::build())
                .into_iter()
                .collect(),
        )
    }
}

#[derive(Debug)]
pub struct VisionEncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: VisionMLP,
    layer_norm2: LayerNorm,
}

impl VisionEncoderLayer {
    pub fn build() -> Self {
        let self_attn = Attention::build(768, 768, true);
        let layer_norm = LayerNorm {
            in_features: 768,
            eps: 1e-6,
            elementwise_affine: true,
        };
        let mlp = VisionMLP {
            activation_fn: Activation::GELUTanh,
            fc1: Linear {
                in_features: 768,
                out_features: 3072,
                bias: true,
            },
            fc2: Linear {
                in_features: 3072,
                out_features: 768,
                bias: true,
            },
        };
        Self {
            self_attn,
            layer_norm1: layer_norm.clone(),
            mlp,
            layer_norm2: layer_norm,
        }
    }
}

#[derive(Debug)]
pub struct VisionMLP {
    activation_fn: Activation,
    fc1: Linear,
    fc2: Linear,
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub in_features: u32,
    pub eps: f32,
    pub elementwise_affine: bool,
}
