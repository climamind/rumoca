use super::*;

/// Pure affine-stride metadata failure shared by Solve-IR admission and
/// evaluator consumers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffineMapMetadataError {
    InvalidLoadStrideOp {
        op_position: usize,
        op_count: usize,
        actual: Option<&'static str>,
    },
    InvalidConstStrideOp {
        op_position: usize,
        op_count: usize,
        actual: Option<&'static str>,
    },
    InvalidLoadStrideDimension {
        dimension: usize,
        dimension_count: usize,
    },
    InvalidConstStrideDimension {
        dimension: usize,
        dimension_count: usize,
    },
    NonFiniteConstStride {
        op_position: usize,
        dimension: usize,
    },
}

impl AffineMapMetadataError {
    pub(crate) fn direct_map_message(self) -> &'static str {
        match self {
            Self::InvalidLoadStrideOp { actual: None, .. } => {
                "direct Map affine load stride op_position is outside base_ops"
            }
            Self::InvalidLoadStrideOp {
                actual: Some(_), ..
            } => "direct Map affine load stride does not point at LoadY or LoadP",
            Self::InvalidConstStrideOp { actual: None, .. } => {
                "direct Map affine constant stride op_position is outside base_ops"
            }
            Self::InvalidConstStrideOp {
                actual: Some(_), ..
            } => "direct Map affine constant stride does not point at Const",
            Self::InvalidLoadStrideDimension { .. } | Self::InvalidConstStrideDimension { .. } => {
                "direct Map affine stride dimension is outside domain"
            }
            Self::NonFiniteConstStride { .. } => "direct Map affine constant stride is non-finite",
        }
    }
}

impl std::fmt::Display for AffineMapMetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLoadStrideOp {
                op_position,
                actual: Some(actual),
                ..
            } => write!(
                f,
                "affine load stride op position {op_position} points at {actual}, \
                 expected LoadY or LoadP"
            ),
            Self::InvalidLoadStrideOp {
                op_position,
                op_count,
                actual: None,
            } => write!(
                f,
                "affine load stride op position {op_position} is out of bounds for \
                 {op_count} ops"
            ),
            Self::InvalidConstStrideOp {
                op_position,
                actual: Some(actual),
                ..
            } => write!(
                f,
                "affine const stride op position {op_position} points at {actual}, expected Const"
            ),
            Self::InvalidConstStrideOp {
                op_position,
                op_count,
                actual: None,
            } => write!(
                f,
                "affine const stride op position {op_position} is out of bounds for \
                 {op_count} ops"
            ),
            Self::InvalidLoadStrideDimension {
                dimension,
                dimension_count,
            } => write!(
                f,
                "affine load stride dimension {dimension} is out of bounds for \
                 {dimension_count} dimensions"
            ),
            Self::InvalidConstStrideDimension {
                dimension,
                dimension_count,
            } => write!(
                f,
                "affine const stride dimension {dimension} is out of bounds for \
                 {dimension_count} dimensions"
            ),
            Self::NonFiniteConstStride {
                op_position,
                dimension,
            } => write!(
                f,
                "affine const stride at op position {op_position}, dimension {dimension} \
                 is non-finite"
            ),
        }
    }
}

impl std::error::Error for AffineMapMetadataError {}

/// Validate the data-only affine metadata contract for a Map or
/// AffineStencil base program.
pub fn validate_affine_map_metadata(
    domain: &StructuredIndexDomain,
    base_ops: &[LinearOp],
    load_strides: &[AffineStencilLoadStride],
    const_strides: &[AffineStencilConstStride],
) -> Result<(), AffineMapMetadataError> {
    for stride in load_strides {
        validate_index_stride_dimensions(domain, &stride.terms)?;
        match base_ops.get(stride.op_position) {
            Some(LinearOp::LoadY { .. } | LinearOp::LoadP { .. }) => {}
            actual => {
                return Err(AffineMapMetadataError::InvalidLoadStrideOp {
                    op_position: stride.op_position,
                    op_count: base_ops.len(),
                    actual: actual.map(LinearOp::kind_name),
                });
            }
        }
    }
    for stride in const_strides {
        validate_const_stride_terms(domain, stride)?;
        match base_ops.get(stride.op_position) {
            Some(LinearOp::Const { .. }) => {}
            actual => {
                return Err(AffineMapMetadataError::InvalidConstStrideOp {
                    op_position: stride.op_position,
                    op_count: base_ops.len(),
                    actual: actual.map(LinearOp::kind_name),
                });
            }
        }
    }
    Ok(())
}

fn validate_index_stride_dimensions(
    domain: &StructuredIndexDomain,
    terms: &[AffineStencilIndexStrideTerm],
) -> Result<(), AffineMapMetadataError> {
    for term in terms {
        if term.dimension >= domain.binders.len() {
            return Err(AffineMapMetadataError::InvalidLoadStrideDimension {
                dimension: term.dimension,
                dimension_count: domain.binders.len(),
            });
        }
    }
    Ok(())
}

fn validate_const_stride_terms(
    domain: &StructuredIndexDomain,
    stride: &AffineStencilConstStride,
) -> Result<(), AffineMapMetadataError> {
    for term in &stride.terms {
        if term.dimension >= domain.binders.len() {
            return Err(AffineMapMetadataError::InvalidConstStrideDimension {
                dimension: term.dimension,
                dimension_count: domain.binders.len(),
            });
        }
        if !term.stride.is_finite() {
            return Err(AffineMapMetadataError::NonFiniteConstStride {
                op_position: stride.op_position,
                dimension: term.dimension,
            });
        }
    }
    Ok(())
}
