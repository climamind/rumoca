use std::collections::HashSet;

use rumoca_core::Span;

use crate::{
    ComputeNode, SolveProblemShapeContractError, StructuredIndexDomain, TensorOutputMap,
    TensorOutputMapError,
};

pub(crate) fn linsolve_output_cursor(
    context: &'static str,
    node_index: usize,
    cursor: usize,
    components: usize,
    output_indices: &[usize],
    span: Span,
) -> Result<usize, SolveProblemShapeContractError> {
    validate_linsolve_output_indices(context, node_index, components, output_indices, span)?;
    if output_indices.is_empty() {
        return cursor
            .checked_add(components)
            .ok_or_else(|| output_index_overflow(context, node_index, Some(span)));
    }
    output_indices.iter().try_fold(cursor, |next, &index| {
        index
            .checked_add(1)
            .map(|end| next.max(end))
            .ok_or_else(|| output_index_overflow(context, node_index, Some(span)))
    })
}

pub(crate) fn validate_linsolve_output_indices(
    context: impl Into<String>,
    node_index: usize,
    components: usize,
    output_indices: &[usize],
    span: Span,
) -> Result<(), SolveProblemShapeContractError> {
    if output_indices.is_empty() {
        return Ok(());
    }
    let context = context.into();
    if output_indices.len() != components {
        return Err(
            SolveProblemShapeContractError::LinSolveOutputIndexMismatch {
                context,
                node_index,
                components,
                output_indices: output_indices.len(),
                span,
            },
        );
    }
    let mut seen = HashSet::with_capacity(output_indices.len());
    for &output_index in output_indices {
        if !seen.insert(output_index) {
            return Err(
                SolveProblemShapeContractError::LinSolveDuplicateOutputIndex {
                    context,
                    node_index,
                    output_index,
                    span,
                },
            );
        }
    }
    Ok(())
}

pub(crate) fn tensor_output_count_for_node(
    context: &'static str,
    node_index: usize,
    node: &ComputeNode,
    domain: &StructuredIndexDomain,
    output_map: &TensorOutputMap,
) -> Result<usize, SolveProblemShapeContractError> {
    let (dimension, span) = match node {
        ComputeNode::Map { span, .. } => ("Map", *span),
        ComputeNode::AffineStencil { span, .. } => ("AffineStencil", *span),
        ComputeNode::ScalarPrograms(_)
        | ComputeNode::MatMul { .. }
        | ComputeNode::LinSolve { .. } => unreachable!("tensor output count requires tensor node"),
    };
    output_map
        .output_count(domain)
        .map_err(|error| tensor_output_map_error(context, node_index, dimension, error, span))
}

pub(crate) fn tensor_output_map_error(
    context: &'static str,
    node_index: usize,
    dimension: &'static str,
    error: TensorOutputMapError,
    span: Span,
) -> SolveProblemShapeContractError {
    match error {
        TensorOutputMapError::Dimension {
            output_dimension,
            domain_rank,
        } => SolveProblemShapeContractError::TensorOutputMapDimension {
            context: context.to_string(),
            node_index,
            dimension,
            output_dimension,
            domain_rank,
            span,
        },
        TensorOutputMapError::StructuredIndexDomain { error } => {
            SolveProblemShapeContractError::StructuredIndexDomain {
                context: context.to_string(),
                node_index,
                dimension,
                error,
                span,
            }
        }
        TensorOutputMapError::NegativeIndex { value } => {
            SolveProblemShapeContractError::TensorOutputMapNegativeIndex {
                context: context.to_string(),
                node_index,
                dimension,
                value,
                span,
            }
        }
        TensorOutputMapError::OutputIndexOverflow => {
            output_index_overflow(context, node_index, Some(span))
        }
    }
}

pub(crate) fn output_index_overflow(
    context: impl Into<String>,
    node_index: usize,
    span: Option<Span>,
) -> SolveProblemShapeContractError {
    SolveProblemShapeContractError::OutputIndexOverflow {
        context: context.into(),
        node_index,
        span,
    }
}
