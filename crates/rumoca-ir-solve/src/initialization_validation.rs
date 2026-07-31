use super::*;
use crate::direct_map_semantics::validate_direct_map_semantics;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct InitializationTargetRange {
    pub start: usize,
    pub end: usize,
    /// Mandatory owning source span; dummy/source-free ranges fail admission.
    pub span: Span,
}

/// Count stored initialization rows without expanding tensor output maps.
/// Shape validation independently owns tensor-map placement validity.
pub(super) fn initialization_stored_row_count(
    residual: &ComputeBlock,
    context: &'static str,
) -> Result<usize, SolveProblemShapeContractError> {
    let mut rows = 0usize;
    for (node_index, node) in residual.nodes.iter().enumerate() {
        let (count, span) = match node {
            ComputeNode::ScalarPrograms(block) => {
                (block.stored_output_count(), block.first_source_span())
            }
            ComputeNode::Map { domain, span, .. }
            | ComputeNode::AffineStencil { domain, span, .. } => (
                domain.scalar_count().map_err(|error| {
                    SolveProblemShapeContractError::StructuredIndexDomain {
                        context: context.to_string(),
                        node_index,
                        dimension: "stored-row",
                        error,
                        span: *span,
                    }
                })?,
                Some(*span),
            ),
            ComputeNode::MatMul { m, n, span, .. } => (
                m.checked_mul(*n)
                    .ok_or_else(|| output_index_overflow(context, node_index, Some(*span)))?,
                Some(*span),
            ),
            ComputeNode::LinSolve { n, span, .. } => (*n, Some(*span)),
        };
        rows = rows
            .checked_add(count)
            .ok_or_else(|| output_index_overflow(context, node_index, span))?;
    }
    Ok(rows)
}

/// Validate the complete compact GPU-initialization contract.
///
/// This is the single semantic admission gate shared by wire deserialization
/// and the simulation runtime. It deliberately proves the direct assignment
/// meaning, not only the surrounding vector shapes.
pub fn validate_compact_gpu_initialization(
    initialization: &InitializationSolveSystem,
    y_upper_bound: usize,
) -> Result<(), SolveProblemShapeContractError> {
    initialization
        .residual
        .validate_shape_contract("initialization.residual")?;
    let residual_row_count =
        initialization_stored_row_count(&initialization.residual, "initialization.residual rows")?;
    validate_initialization_direct_families(initialization, y_upper_bound, residual_row_count)
}

pub(super) fn validate_initialization_direct_families(
    initialization: &InitializationSolveSystem,
    y_upper_bound: usize,
    residual_row_count: usize,
) -> Result<(), SolveProblemShapeContractError> {
    if initialization.direct_families.is_empty() {
        return validate_initialization_without_direct_families(
            initialization,
            y_upper_bound,
            residual_row_count,
        );
    }
    validate_count(
        "initialization.row_targets.compact",
        0,
        initialization.row_targets.len(),
    )?;
    if initialization.residual.nodes.len() != initialization.direct_families.len() {
        let family = initialization.direct_families.first().ok_or_else(|| {
            initialization_range_error("direct families must own every residual node exactly once")
        })?;
        return Err(direct_semantic_error(
            family,
            "direct families must own every residual node exactly once",
        ));
    }
    let mut covered_nodes = vec![false; initialization.residual.nodes.len()];
    let mut target_ranges = Vec::with_capacity(initialization.direct_families.len());
    for family in &initialization.direct_families {
        let Some(covered_node) = covered_nodes.get_mut(family.node_index) else {
            return Err(SolveProblemShapeContractError::ZeroTensorDimension {
                context: "initialization.direct_families".to_string(),
                node_index: family.node_index,
                dimension: "direct-family node index outside residual block",
                span: family.span,
            });
        };
        if std::mem::replace(covered_node, true) {
            return Err(SolveProblemShapeContractError::ZeroTensorDimension {
                context: "initialization.direct_families".to_string(),
                node_index: family.node_index,
                dimension: "duplicate direct-family node index",
                span: family.span,
            });
        }
        let target_range = validate_initialization_direct_family(initialization, family)?;
        target_ranges.push((target_range, family.node_index, family.span));
    }
    target_ranges.sort_unstable_by_key(|(range, _, _)| range.start);
    for adjacent in target_ranges.windows(2) {
        let [(left, _, _), (right, node_index, span)] = adjacent else {
            unreachable!("windows(2) always has two entries")
        };
        if right.start < left.end {
            return Err(SolveProblemShapeContractError::ZeroTensorDimension {
                context: "initialization.direct_families".to_string(),
                node_index: *node_index,
                dimension: "overlapping direct-family target map",
                span: *span,
            });
        }
    }
    validate_compact_direct_projection_plan(initialization, &target_ranges)?;
    let direct_ranges = target_ranges
        .into_iter()
        .map(|(range, _, span)| InitializationTargetRange {
            start: range.start,
            end: range.end,
            span,
        })
        .collect::<Vec<_>>();
    validate_compact_target_coverage(initialization, y_upper_bound, direct_ranges)
}

fn validate_compact_target_coverage(
    initialization: &InitializationSolveSystem,
    y_upper_bound: usize,
    direct_ranges: Vec<InitializationTargetRange>,
) -> Result<(), SolveProblemShapeContractError> {
    let required = normalized_ranges(
        &initialization.required_target_ranges,
        y_upper_bound,
        "invalid required target range",
    )?;
    if !covers_complete_target_range(&required, y_upper_bound) {
        return Err(initialization_range_error_at(
            "incomplete required target coverage of the solver Y vector",
            required.first().map(|range| range.span),
        ));
    }
    let fixed = normalized_ranges(
        &initialization.fixed_target_ranges,
        y_upper_bound,
        "invalid fixed-start target range",
    )?;
    let mut actual = direct_ranges;
    actual.extend(fixed);
    let actual = normalized_ranges(&actual, y_upper_bound, "invalid target union range")?;
    if !same_target_coverage(&actual, &required) {
        return Err(initialization_range_error_at(
            "incomplete direct plus fixed-start target union",
            actual
                .first()
                .map(|range| range.span)
                .or_else(|| required.first().map(|range| range.span)),
        ));
    }
    Ok(())
}

fn validate_compact_direct_projection_plan(
    initialization: &InitializationSolveSystem,
    target_ranges: &[(std::ops::Range<usize>, usize, Span)],
) -> Result<(), SolveProblemShapeContractError> {
    if !initialization.projection_indices.is_empty() {
        return Err(compact_projection_error(
            initialization,
            0,
            "compact direct projection must not recover scalar projection indices",
        ));
    }
    if initialization.projection_plan.blocks.len() != initialization.direct_families.len() {
        return Err(compact_projection_error(
            initialization,
            0,
            "compact direct projection must own every direct family exactly once",
        ));
    }
    let ranges_by_family = initialization
        .direct_families
        .iter()
        .map(|family| {
            target_ranges
                .iter()
                .find(|(_, node_index, _)| *node_index == family.node_index)
                .map(|(range, _, _)| range.clone())
                .ok_or_else(|| {
                    compact_projection_error(
                        initialization,
                        family.node_index,
                        "compact direct projection is missing a target owner",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dependencies = initialization
        .direct_families
        .iter()
        .map(|family| direct_family_dependencies(initialization, family, &ranges_by_family))
        .collect::<Result<Vec<_>, _>>()?;
    let mut seen = vec![false; initialization.direct_families.len()];
    for (block_index, block) in initialization.projection_plan.blocks.iter().enumerate() {
        let [family_index] = block.rows.as_slice() else {
            return Err(compact_projection_error(
                initialization,
                block_index,
                "compact direct projection block must name exactly one family",
            ));
        };
        let Some(family) = initialization.direct_families.get(*family_index) else {
            return Err(compact_projection_error(
                initialization,
                block_index,
                "compact direct projection family index is out of bounds",
            ));
        };
        if seen[*family_index] {
            return Err(compact_projection_error(
                initialization,
                block_index,
                "compact direct projection family ownership is duplicated",
            ));
        }
        if block.y_indices.as_slice() != [family.targets.start] || !block.causal_steps.is_empty() {
            return Err(compact_projection_error(
                initialization,
                block_index,
                "compact direct projection block has an invalid target anchor",
            ));
        }
        if dependencies[*family_index]
            .iter()
            .copied()
            .any(|owner| !seen[owner])
        {
            return Err(SolveProblemShapeContractError::ZeroTensorDimension {
                context: "initialization.projection_plan".to_string(),
                node_index: block_index,
                dimension: "compact direct projection violates dependency order",
                span: family.span,
            });
        }
        seen[*family_index] = true;
    }
    Ok(())
}

fn compact_projection_error(
    initialization: &InitializationSolveSystem,
    block_index: usize,
    dimension: &'static str,
) -> SolveProblemShapeContractError {
    let span = initialization
        .projection_plan
        .blocks
        .get(block_index)
        .and_then(|block| block.rows.first())
        .and_then(|family_index| initialization.direct_families.get(*family_index))
        .or_else(|| initialization.direct_families.first())
        .map(|family| family.span);
    let Some(span) = span else {
        return initialization_range_error(dimension);
    };
    SolveProblemShapeContractError::ZeroTensorDimension {
        context: "initialization.projection_plan".to_string(),
        node_index: block_index,
        dimension,
        span,
    }
}

fn validate_initialization_without_direct_families(
    initialization: &InitializationSolveSystem,
    y_upper_bound: usize,
    residual_row_count: usize,
) -> Result<(), SolveProblemShapeContractError> {
    validate_count(
        "initialization.row_targets",
        residual_row_count,
        initialization.row_targets.len(),
    )?;
    if initialization.residual.is_empty() && initialization.row_targets.is_empty() {
        let required = normalized_ranges(
            &initialization.required_target_ranges,
            y_upper_bound,
            "invalid required target range",
        )?;
        let fixed = normalized_ranges(
            &initialization.fixed_target_ranges,
            y_upper_bound,
            "invalid fixed-start target range",
        )?;
        if !same_target_coverage(&required, &fixed) {
            return Err(initialization_range_error_at(
                "incomplete fixed-start target union",
                fixed
                    .first()
                    .map(|range| range.span)
                    .or_else(|| required.first().map(|range| range.span)),
            ));
        }
    } else if !initialization.required_target_ranges.is_empty()
        || !initialization.fixed_target_ranges.is_empty()
    {
        return Err(initialization_range_error(
            "target coverage metadata without compact direct families",
        ));
    }
    Ok(())
}

fn normalized_ranges(
    ranges: &[InitializationTargetRange],
    upper_bound: usize,
    error: &'static str,
) -> Result<Vec<InitializationTargetRange>, SolveProblemShapeContractError> {
    let mut ranges = ranges.to_vec();
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut normalized: Vec<InitializationTargetRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if range.span.is_dummy() {
            return Err(initialization_range_error_at(
                "initialization target range requires a non-dummy source span",
                None,
            ));
        }
        if range.start >= range.end || range.end > upper_bound {
            return Err(initialization_range_error_at(error, Some(range.span)));
        }
        if let Some(last) = normalized.last_mut() {
            if range.start < last.end {
                return Err(initialization_range_error_at(
                    "overlapping initialization target ranges",
                    Some(range.span),
                ));
            }
            if range.start == last.end {
                last.end = range.end;
                continue;
            }
        }
        normalized.push(range);
    }
    Ok(normalized)
}

fn initialization_range_error(reason: &'static str) -> SolveProblemShapeContractError {
    initialization_range_error_at(reason, None)
}

fn initialization_range_error_at(
    reason: &'static str,
    span: Option<Span>,
) -> SolveProblemShapeContractError {
    SolveProblemShapeContractError::InitializationTargetCoverage { reason, span }
}

fn same_target_coverage(
    left: &[InitializationTargetRange],
    right: &[InitializationTargetRange],
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.start == right.start && left.end == right.end)
}

fn covers_complete_target_range(ranges: &[InitializationTargetRange], upper_bound: usize) -> bool {
    match (upper_bound, ranges) {
        (0, []) => true,
        (_, [range]) => range.start == 0 && range.end == upper_bound,
        _ => false,
    }
}

fn validate_initialization_direct_family(
    initialization: &InitializationSolveSystem,
    family: &InitializationDirectFamily,
) -> Result<std::ops::Range<usize>, SolveProblemShapeContractError> {
    if !matches!(family.residual_sign, -1 | 1) {
        return Err(SolveProblemShapeContractError::ZeroTensorDimension {
            context: "initialization.direct_families".to_string(),
            node_index: family.node_index,
            dimension: "direct-family residual sign must be -1 or +1",
            span: family.span,
        });
    }
    let Some(node) = initialization.residual.nodes.get(family.node_index) else {
        return Err(SolveProblemShapeContractError::ZeroTensorDimension {
            context: "initialization.direct_families".to_string(),
            node_index: family.node_index,
            dimension: "direct-family node index outside residual block",
            span: family.span,
        });
    };
    let ComputeNode::Map {
        domain,
        base_ops,
        load_strides,
        const_strides,
        span,
        ..
    } = node
    else {
        return Err(SolveProblemShapeContractError::ZeroTensorDimension {
            context: "initialization.direct_families".to_string(),
            node_index: family.node_index,
            dimension: "non-Map direct family",
            span: family.span,
        });
    };
    let dense =
        TensorOutputMap::dense_contiguous(family.targets.start, domain).map_err(|error| {
            tensor_output_map_error(
                "initialization.direct_families.targets",
                family.node_index,
                "Map",
                error,
                *span,
            )
        })?;
    if family.targets.strides != dense.strides {
        return Err(SolveProblemShapeContractError::ZeroTensorDimension {
            context: "initialization.direct_families.targets".to_string(),
            node_index: family.node_index,
            dimension: "non-contiguous direct-family target map",
            span: *span,
        });
    }
    let count = domain.scalar_count().map_err(|error| {
        SolveProblemShapeContractError::StructuredIndexDomain {
            context: "initialization.direct_families.targets".to_string(),
            node_index: family.node_index,
            dimension: "Map",
            error,
            span: *span,
        }
    })?;
    let end = family.targets.start.checked_add(count).ok_or_else(|| {
        output_index_overflow(
            "initialization.direct_families.targets",
            family.node_index,
            Some(*span),
        )
    })?;
    validate_affine_map_metadata(domain, base_ops, load_strides, const_strides)
        .map_err(|error| direct_semantic_error(family, error.direct_map_message()))?;
    validate_direct_map_semantics(family, base_ops, load_strides)?;
    Ok(family.targets.start..end)
}

fn direct_family_dependencies(
    initialization: &InitializationSolveSystem,
    family: &InitializationDirectFamily,
    target_ranges: &[std::ops::Range<usize>],
) -> Result<std::collections::BTreeSet<usize>, SolveProblemShapeContractError> {
    let Some(ComputeNode::Map {
        domain,
        base_ops,
        load_strides,
        ..
    }) = initialization.residual.nodes.get(family.node_index)
    else {
        return Err(direct_semantic_error(family, "non-Map direct family"));
    };
    let target_position = base_ops
        .iter()
        .position(
            |op| matches!(op, LinearOp::LoadY { index, .. } if *index == family.targets.start),
        )
        .ok_or_else(|| direct_semantic_error(family, "missing target LoadY"))?;
    let mut dependencies = std::collections::BTreeSet::new();
    for (op_position, op) in base_ops.iter().enumerate() {
        let LinearOp::LoadY { index, .. } = op else {
            continue;
        };
        if op_position == target_position {
            continue;
        }
        let load_range = affine_load_range(*index, op_position, domain, load_strides, family)?;
        dependencies.extend(
            target_ranges
                .iter()
                .enumerate()
                .filter_map(|(owner, target)| ranges_overlap(&load_range, target).then_some(owner)),
        );
    }
    Ok(dependencies)
}

fn affine_load_range(
    base: usize,
    op_position: usize,
    domain: &StructuredIndexDomain,
    load_strides: &[AffineStencilLoadStride],
    family: &InitializationDirectFamily,
) -> Result<std::ops::Range<usize>, SolveProblemShapeContractError> {
    let mut minimum = base as i128;
    let mut maximum = base as i128;
    for term in load_strides
        .iter()
        .filter(|stride| stride.op_position == op_position)
        .flat_map(|stride| &stride.terms)
    {
        let Some(binder) = domain.binders.get(term.dimension) else {
            return Err(direct_semantic_error(
                family,
                "direct LoadY stride dimension is outside its domain",
            ));
        };
        let count = StructuredIndexDomain {
            binders: vec![binder.clone()],
        }
        .scalar_count()
        .map_err(|_| direct_semantic_error(family, "invalid direct LoadY domain"))?;
        let extent = (term.stride as i128)
            .checked_mul(count.saturating_sub(1) as i128)
            .ok_or_else(|| direct_semantic_error(family, "direct LoadY extent overflow"))?;
        if extent < 0 {
            minimum = minimum
                .checked_add(extent)
                .ok_or_else(|| direct_semantic_error(family, "direct LoadY range overflow"))?;
        } else {
            maximum = maximum
                .checked_add(extent)
                .ok_or_else(|| direct_semantic_error(family, "direct LoadY range overflow"))?;
        }
    }
    let start = usize::try_from(minimum)
        .map_err(|_| direct_semantic_error(family, "direct LoadY starts outside solver Y"))?;
    let end = usize::try_from(maximum)
        .ok()
        .and_then(|maximum| maximum.checked_add(1))
        .ok_or_else(|| direct_semantic_error(family, "direct LoadY range overflow"))?;
    Ok(start..end)
}

fn ranges_overlap(left: &std::ops::Range<usize>, right: &std::ops::Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

pub(super) fn direct_semantic_error(
    family: &InitializationDirectFamily,
    dimension: &'static str,
) -> SolveProblemShapeContractError {
    SolveProblemShapeContractError::ZeroTensorDimension {
        context: "initialization.direct_families".to_string(),
        node_index: family.node_index,
        dimension,
        span: family.span,
    }
}
