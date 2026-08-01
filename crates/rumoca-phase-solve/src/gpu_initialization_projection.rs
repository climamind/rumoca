use super::*;

/// Build a compact causal projection order from the source-owned direct Maps.
///
/// For compact GPU initialization, each projection block owns one direct-family
/// index in `rows` and its target-range anchor in `y_indices`. This keeps the
/// existing initialization projection envelope without recovering scalar rows.
pub(super) fn lower_gpu_initialization_projection_plan(
    residual: &solve::ComputeBlock,
    families: &[solve::InitializationDirectFamily],
) -> Result<solve::AlgebraicProjectionPlan, LowerError> {
    if families.is_empty() {
        return Ok(solve::AlgebraicProjectionPlan::default());
    }
    let ranges = families
        .iter()
        .map(|family| direct_family_target_range(residual, family))
        .collect::<Result<Vec<_>, _>>()?;
    let dependencies = families
        .iter()
        .map(|family| direct_family_dependencies(residual, family, &ranges))
        .collect::<Result<Vec<_>, _>>()?;
    let order = topological_direct_family_order(&dependencies, families)?;
    let blocks = order
        .into_iter()
        .map(|family_index| solve::AlgebraicProjectionBlock {
            rows: vec![family_index],
            y_indices: vec![families[family_index].targets.start],
            causal_steps: Vec::new(),
        })
        .collect();
    Ok(solve::AlgebraicProjectionPlan { blocks })
}

fn direct_family_dependencies(
    residual: &solve::ComputeBlock,
    family: &solve::InitializationDirectFamily,
    target_ranges: &[std::ops::Range<usize>],
) -> Result<std::collections::BTreeSet<usize>, LowerError> {
    let node = residual.nodes.get(family.node_index).ok_or_else(|| {
        gpu_projection_error(
            "GPU initialization projection references a missing direct Map",
            family.span,
        )
    })?;
    let solve::ComputeNode::Map {
        domain,
        base_ops,
        load_strides,
        ..
    } = node
    else {
        return Err(gpu_projection_error(
            "GPU initialization projection requires direct Map owners",
            family.span,
        ));
    };
    let target_load_position =
        direct_target_load_position(base_ops, family.targets.start, family.span)?;
    let mut dependencies = std::collections::BTreeSet::new();
    for (op_position, op) in base_ops.iter().enumerate() {
        let solve::LinearOp::LoadY { index, .. } = op else {
            continue;
        };
        if op_position == target_load_position {
            continue;
        }
        let load_range = affine_load_range(*index, op_position, domain, load_strides, family.span)?;
        dependencies.extend(
            target_ranges
                .iter()
                .enumerate()
                .filter_map(|(owner, target)| ranges_overlap(&load_range, target).then_some(owner)),
        );
    }
    Ok(dependencies)
}

fn direct_family_target_range(
    residual: &solve::ComputeBlock,
    family: &solve::InitializationDirectFamily,
) -> Result<std::ops::Range<usize>, LowerError> {
    let Some(solve::ComputeNode::Map { domain, .. }) = residual.nodes.get(family.node_index) else {
        return Err(gpu_projection_error(
            "GPU initialization projection requires direct Map owners",
            family.span,
        ));
    };
    let count = domain
        .scalar_count()
        .map_err(|error| LowerError::contract_violation(error.to_string(), family.span))?;
    let end = family.targets.start.checked_add(count).ok_or_else(|| {
        LowerError::contract_violation(
            "GPU initialization projection target range overflow",
            family.span,
        )
    })?;
    Ok(family.targets.start..end)
}

fn direct_target_load_position(
    ops: &[solve::LinearOp],
    target_start: usize,
    span: rumoca_core::Span,
) -> Result<usize, LowerError> {
    let mut positions = ops.iter().enumerate().filter_map(|(position, op)| {
        matches!(op, solve::LinearOp::LoadY { index, .. } if *index == target_start)
            .then_some(position)
    });
    let Some(position) = positions.next() else {
        return Err(gpu_projection_error(
            "GPU initialization projection is missing its direct target load",
            span,
        ));
    };
    if positions.next().is_some() {
        return Err(gpu_projection_error(
            "GPU initialization projection has ambiguous direct target loads",
            span,
        ));
    }
    Ok(position)
}

fn affine_load_range(
    base: usize,
    op_position: usize,
    domain: &rumoca_core::StructuredIndexDomain,
    strides: &[solve::AffineStencilLoadStride],
    span: rumoca_core::Span,
) -> Result<std::ops::Range<usize>, LowerError> {
    let mut minimum = base as i128;
    let mut maximum = base as i128;
    for stride in strides
        .iter()
        .filter(|stride| stride.op_position == op_position)
        .flat_map(|stride| &stride.terms)
    {
        let binder = domain.binders.get(stride.dimension).ok_or_else(|| {
            LowerError::contract_violation(
                "GPU initialization projection stride dimension is missing",
                span,
            )
        })?;
        let count = rumoca_core::StructuredIndexDomain {
            binders: vec![binder.clone()],
        }
        .scalar_count()
        .map_err(|error| LowerError::contract_violation(error.to_string(), span))?;
        let extent = (stride.stride as i128)
            .checked_mul(count.saturating_sub(1) as i128)
            .ok_or_else(|| {
                LowerError::contract_violation(
                    "GPU initialization projection load extent overflow",
                    span,
                )
            })?;
        if extent < 0 {
            minimum = minimum.checked_add(extent).ok_or_else(|| {
                LowerError::contract_violation(
                    "GPU initialization projection load range overflow",
                    span,
                )
            })?;
        } else {
            maximum = maximum.checked_add(extent).ok_or_else(|| {
                LowerError::contract_violation(
                    "GPU initialization projection load range overflow",
                    span,
                )
            })?;
        }
    }
    let start = usize::try_from(minimum).map_err(|_| {
        LowerError::contract_violation(
            "GPU initialization projection load starts outside solver Y",
            span,
        )
    })?;
    let end = usize::try_from(maximum)
        .ok()
        .and_then(|maximum| maximum.checked_add(1))
        .ok_or_else(|| {
            LowerError::contract_violation(
                "GPU initialization projection load range overflow",
                span,
            )
        })?;
    Ok(start..end)
}

fn topological_direct_family_order(
    dependencies: &[std::collections::BTreeSet<usize>],
    families: &[solve::InitializationDirectFamily],
) -> Result<Vec<usize>, LowerError> {
    let mut emitted = vec![false; families.len()];
    let mut order = Vec::with_capacity(families.len());
    while order.len() < families.len() {
        let next = dependencies.iter().enumerate().find_map(|(index, owners)| {
            (!emitted[index] && owners.iter().all(|owner| emitted[*owner])).then_some(index)
        });
        let Some(next) = next else {
            let blocked = emitted.iter().position(|emitted| !*emitted).unwrap_or(0);
            return Err(gpu_projection_error(
                "GPU initialization projection contains a cyclic direct-family dependency",
                families[blocked].span,
            ));
        };
        emitted[next] = true;
        order.push(next);
    }
    Ok(order)
}

fn ranges_overlap(left: &std::ops::Range<usize>, right: &std::ops::Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn gpu_projection_error(reason: &'static str, span: rumoca_core::Span) -> LowerError {
    LowerError::UnsupportedAt {
        reason: reason.to_string(),
        contexts: Vec::new(),
        span,
    }
}
