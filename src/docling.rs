use crate::{
    nn::Linear,
    embeddings::LlamaRotaryEmbedding,
    llama::{LlamaModel, LlamaRMSNorm},
    transformers::VisionTransformer,
};

#[derive(Debug)]
pub struct DoclingGeneration {
    model: DoclingModel,
    lm_head: Linear,
}

impl DoclingGeneration {
    pub fn build() -> Self {
        let lm_head = Linear {
            in_features: 576,
            out_features: 100352,
            bias: false,
        };
        let model = DoclingModel::build();

        Self { model, lm_head }
    }
}

#[derive(Debug)]
pub struct DoclingModel {
    vision_model: VisionTransformer,
    connector: Connector,
    text_model: LlamaModel,
    norm: LlamaRMSNorm,
    rotary_emb: LlamaRotaryEmbedding,
}

impl DoclingModel {
    pub fn build() -> Self {
        let vision_model = VisionTransformer::build();
        let connector = Connector::build();
        let text_model = LlamaModel::build();
        let norm = LlamaRMSNorm {
            in_features: 576,
            eps: 1e-05,
        };
        let rotary_emb = LlamaRotaryEmbedding;

        Self {
            vision_model,
            connector,
            text_model,
            norm,
            rotary_emb,
        }
    }
}

#[derive(Debug)]
pub struct Connector(Linear);

impl Connector {
    pub fn build() -> Self {
        Self(Linear {
            in_features: 12288,
            out_features: 576,
            bias: false,
        })
    }
}
