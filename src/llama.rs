use crate::{
    activations::Activation,
    nn::{Attention, Embedding, Linear},
};

#[derive(Debug)]
pub struct LlamaModel {
    embed_tokens: Embedding,
    decoder: LlamaDecoder,
}

impl LlamaModel {
    pub fn build() -> Self {
        let embed_tokens = Embedding {
            in_features: 100352,
            out_features: 576,
            padding_idx: Some(100257),
        };
        let decoder = LlamaDecoder::build();

        Self {
            embed_tokens,
            decoder,
        }
    }
}

#[derive(Debug)]
pub struct LlamaDecoder(Vec<LlamaDecoderLayer>);

impl LlamaDecoder {
    pub fn build() -> Self {
        Self(
            [..12]
                .map(|_| LlamaDecoderLayer {
                    self_attn: Attention::build(576, 576, false),
                    mlp: LlamaMLP::build(),
                    input_layernorm: LlamaRMSNorm {
                        in_features: 576,
                        eps: 1e-5,
                    },
                    post_attention_layernorm: LlamaRMSNorm {
                        in_features: 576,
                        eps: 1e-5,
                    },
                })
                .into_iter()
                .collect(),
        )
    }
}

#[derive(Debug)]
pub struct LlamaDecoderLayer {
    self_attn: Attention,
    mlp: LlamaMLP,
    input_layernorm: LlamaRMSNorm,
    post_attention_layernorm: LlamaRMSNorm,
}

#[derive(Debug)]
pub struct LlamaRMSNorm {
    pub in_features: u32,
    pub eps: f32,
}

#[derive(Debug)]
pub struct LlamaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation_fn: Activation,
}

impl LlamaMLP {
    pub fn build() -> Self {
        Self {
            gate_proj: Linear {
                in_features: 576,
                out_features: 1536,
                bias: false,
            },
            up_proj: Linear {
                in_features: 576,
                out_features: 1536,
                bias: false,
            },
            down_proj: Linear {
                in_features: 1536,
                out_features: 576,
                bias: false,
            },
            activation_fn: Activation::SiLU,
        }
    }
}
