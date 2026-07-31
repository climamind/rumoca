//! Native, allocation-bounded evaluation of structured Solve-IR Maps.

use super::*;

/// Execute a compact Solve-IR `Map` directly over its structured domain.
///
/// This is intentionally distinct from scalarization: it keeps the map's
/// single `base_ops` owner and applies affine load/constant offsets while each
/// element is evaluated.  The callback sees a borrowed ordinal that is reused
/// for every element, so this path never creates a scalar-row vector or a
/// per-cell `LinearOp` clone.
pub fn eval_map_elements_with_context(
    node: &ComputeNode,
    y: &mut [f64],
    p: &[f64],
    t: f64,
    context: RowEvalContext<'_>,
    mut visit: impl FnMut(&[usize], f64, &mut [f64]) -> Result<(), EvalSolveError>,
) -> Result<MapEvaluationMetrics, EvalSolveError> {
    let ComputeNode::Map {
        domain,
        base_ops,
        load_strides,
        const_strides,
        span,
        ..
    } = node
    else {
        return Err(EvalSolveError::InvalidRow {
            message: "native map evaluation requires a ComputeNode::Map".to_string(),
            span: None,
        });
    };
    let local_runtime_state;
    let context = match context.runtime_state {
        Some(_) => context,
        None => {
            local_runtime_state = SimulationRuntimeState::new();
            context.with_runtime_state(&local_runtime_state)
        }
    };
    rumoca_ir_solve::validate_affine_map_metadata(domain, base_ops, load_strides, const_strides)
        .map_err(|error| affine_map_error(error.to_string(), *span))?;
    let counts = map_domain_counts(domain, *span)?;
    if counts.contains(&0) {
        return Ok(MapEvaluationMetrics::default());
    }
    let register_count = required_registers(base_ops)?.max(1);
    let mut scratch = RowEvalScratch::default();
    let mut ordinal = vec![0usize; counts.len()];
    let mut metrics = MapEvaluationMetrics {
        temporary_values: counts
            .len()
            .saturating_mul(2)
            .saturating_add(register_count),
        ..Default::default()
    };
    loop {
        let mut output = [0.0f64];
        let mut sink = OutputCursor::new(&mut output);
        let input = PreparedRowEval::new(base_ops, register_count, y, p, t, context)
            .with_source_span(Some(*span));
        eval_affine_map_row(
            input,
            &mut scratch,
            &mut sink,
            AffineMapOffsets {
                ordinal: &ordinal,
                load_strides,
                const_strides,
                span: *span,
            },
        )?;
        visit(&ordinal, output[0], y)?;
        metrics.elements = metrics.elements.saturating_add(1);
        if increment_map_ordinal(&mut ordinal, &counts) {
            break;
        }
    }
    Ok(metrics)
}

/// Deterministic resource counters for native map evaluation.  They are used
/// by compact-runtime callers to assert linear traversal rather than relying
/// on host timing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MapEvaluationMetrics {
    pub elements: usize,
    pub temporary_values: usize,
}

pub(super) fn affine_load_offset(
    position: usize,
    offsets: AffineMapOffsets<'_>,
) -> Result<isize, EvalSolveError> {
    offsets
        .load_strides
        .iter()
        .filter(|stride| stride.op_position == position)
        .try_fold(0isize, |total, terms| {
            checked_affine_index_offset(total, &terms.terms, offsets.ordinal, offsets.span)
        })
}

pub(super) fn affine_const_offset(
    position: usize,
    offsets: AffineMapOffsets<'_>,
) -> Result<f64, EvalSolveError> {
    offsets
        .const_strides
        .iter()
        .filter(|stride| stride.op_position == position)
        .try_fold(0.0f64, |total, terms| {
            checked_affine_const_offset(total, &terms.terms, offsets.ordinal, offsets.span)
        })
}

#[derive(Clone, Copy)]
pub(super) struct AffineMapOffsets<'a> {
    pub(super) ordinal: &'a [usize],
    pub(super) load_strides: &'a [rumoca_ir_solve::AffineStencilLoadStride],
    pub(super) const_strides: &'a [rumoca_ir_solve::AffineStencilConstStride],
    pub(super) span: rumoca_core::Span,
}

fn eval_affine_map_row(
    input: PreparedRowEval<'_, '_>,
    scratch: &mut RowEvalScratch,
    sink: &mut OutputCursor<'_>,
    offsets: AffineMapOffsets<'_>,
) -> Result<(), EvalSolveError> {
    scratch.regs.resize(input.register_count, 0.0);
    scratch.initialized.resize(input.register_count, false);
    scratch.regs.fill(0.0);
    scratch.initialized.fill(false);
    let mut evaluator = CheckedRowEvaluator {
        regs: &mut scratch.regs,
        initialized: &mut scratch.initialized,
        input,
        sink,
    };
    for (position, op) in evaluator.input.row.iter().copied().enumerate() {
        evaluator.eval_affine_op(position, op, offsets)?;
    }
    Ok(())
}

fn checked_affine_index_offset(
    total: isize,
    terms: &[rumoca_ir_solve::AffineStencilIndexStrideTerm],
    ordinal: &[usize],
    span: rumoca_core::Span,
) -> Result<isize, EvalSolveError> {
    terms.iter().try_fold(total, |sum, term| {
        let coordinate = *ordinal.get(term.dimension).ok_or_else(|| {
            affine_map_error(
                "affine load stride references a missing domain dimension",
                span,
            )
        })?;
        let coordinate = isize::try_from(coordinate)
            .map_err(|_| affine_map_error("affine load ordinal overflows isize", span))?;
        let offset = term
            .stride
            .checked_mul(coordinate)
            .ok_or_else(|| affine_map_error("affine load stride overflows isize", span))?;
        sum.checked_add(offset)
            .ok_or_else(|| affine_map_error("affine load offset overflows isize", span))
    })
}

fn checked_affine_const_offset(
    total: f64,
    terms: &[rumoca_ir_solve::AffineStencilConstStrideTerm],
    ordinal: &[usize],
    span: rumoca_core::Span,
) -> Result<f64, EvalSolveError> {
    terms.iter().try_fold(total, |sum, term| {
        let coordinate = *ordinal.get(term.dimension).ok_or_else(|| {
            affine_map_error(
                "affine constant stride references a missing domain dimension",
                span,
            )
        })?;
        let offset = term.stride * coordinate as f64;
        if !offset.is_finite() {
            return Err(affine_map_error(
                "affine constant offset is non-finite",
                span,
            ));
        }
        let next = sum + offset;
        next.is_finite()
            .then_some(next)
            .ok_or_else(|| affine_map_error("affine constant offset is non-finite", span))
    })
}

fn map_domain_counts(
    domain: &rumoca_core::StructuredIndexDomain,
    span: rumoca_core::Span,
) -> Result<Vec<usize>, EvalSolveError> {
    domain
        .binders
        .iter()
        .map(|binder| {
            if binder.step == 0 {
                return Err(affine_map_error("structured map binder step is zero", span));
            }
            let count = if binder.step > 0 {
                if binder.lower > binder.upper {
                    return Ok(0);
                }
                let distance = (binder.upper as i128 - binder.lower as i128) as u128;
                distance / binder.step as u128 + 1
            } else {
                if binder.lower < binder.upper {
                    return Ok(0);
                }
                let distance = (binder.lower as i128 - binder.upper as i128) as u128;
                let step = -(binder.step as i128) as u128;
                distance / step + 1
            };
            usize::try_from(count).map_err(|_| {
                affine_map_error("structured map binder count exceeds host range", span)
            })
        })
        .collect()
}

fn increment_map_ordinal(ordinal: &mut [usize], counts: &[usize]) -> bool {
    for dimension in (0..ordinal.len()).rev() {
        ordinal[dimension] += 1;
        if ordinal[dimension] < counts[dimension] {
            return false;
        }
        ordinal[dimension] = 0;
    }
    true
}

pub(super) fn affine_map_error(
    message: impl Into<String>,
    span: rumoca_core::Span,
) -> EvalSolveError {
    EvalSolveError::InvalidRow {
        message: message.into(),
        span: Some(span),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_map_evaluation_accepts_empty_structured_domain_as_zero_rows() {
        let span = rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("empty_native_map.mo"),
            1,
            2,
        );
        let domain = rumoca_core::StructuredIndexDomain {
            binders: vec![rumoca_core::StructuredIndexBinder {
                id: 0,
                display_name: "i".to_string(),
                lower: 3,
                upper: 1,
                step: 1,
            }],
        };
        let node = ComputeNode::Map {
            output_map: rumoca_ir_solve::TensorOutputMap::dense_contiguous(0, &domain)
                .expect("empty output map"),
            domain,
            base_ops: vec![
                LinearOp::Const { dst: 0, value: 1.0 },
                LinearOp::StoreOutput { src: 0 },
            ],
            load_strides: Vec::new(),
            const_strides: Vec::new(),
            metadata: rumoca_ir_solve::TensorNodeMetadata::default(),
            span,
        };
        let mut visits = 0usize;
        let metrics = eval_map_elements_with_context(
            &node,
            &mut [],
            &[],
            0.0,
            RowEvalContext::default(),
            |_, _, _| {
                visits += 1;
                Ok(())
            },
        )
        .expect("canonical empty domain evaluates successfully");
        assert_eq!(visits, 0);
        assert_eq!(metrics, MapEvaluationMetrics::default());
    }
}
