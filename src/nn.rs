#[derive(Debug)]
pub struct Conv2d {
    pub in_features: u32,
    pub out_features: u32,
    pub kernel_size: [u32; 2],
    pub stride: [u32; 2],
    pub padding: Padding,
}

#[derive(Debug)]
pub struct Embedding {
    pub in_features: u32,
    pub out_features: u32,
    pub padding_idx: Option<u32>,
}

#[derive(Debug)]
pub struct Linear {
    pub in_features: u32,
    pub out_features: u32,
    pub bias: bool,
}

#[derive(Debug)]
pub enum Padding {
    Valid,
}

#[derive(Debug)]
pub struct Attention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
}

impl Attention {
    pub fn build(in_features: u32, out_features: u32, bias: bool) -> Self {
        Self {
            k_proj: Linear {
                in_features,
                out_features,
                bias,
            },
            q_proj: Linear {
                in_features,
                out_features,
                bias,
            },
            v_proj: Linear {
                in_features,
                out_features,
                bias,
            },
            out_proj: Linear {
                in_features,
                out_features,
                bias,
            },
        }
    }
}
