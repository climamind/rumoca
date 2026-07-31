use super::*;
use crate::initialization_validation::direct_semantic_error;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate_direct_map_semantics(
    family: &InitializationDirectFamily,
    base_ops: &[LinearOp],
    load_strides: &[AffineStencilLoadStride],
) -> Result<(), SolveProblemShapeContractError> {
    let definitions = validate_ssa_register_flow(family, base_ops)?;
    let mut target_loads = base_ops
        .iter()
        .enumerate()
        .filter_map(|(position, op)| match op {
            LinearOp::LoadY { dst, index } if *index == family.targets.start => {
                Some((position, *dst))
            }
            _ => None,
        });
    let Some((target_position, target_register)) = target_loads.next() else {
        return Err(direct_semantic_error(
            family,
            "direct Map is missing its target LoadY",
        ));
    };
    if target_loads.next().is_some() {
        return Err(direct_semantic_error(
            family,
            "direct Map has more than one target LoadY",
        ));
    }
    if definitions.get(&target_register) != Some(&target_position) {
        return Err(direct_semantic_error(
            family,
            "direct target register does not reach from its target LoadY",
        ));
    }
    validate_target_affine_map(family, target_position, load_strides)?;
    let Some(LinearOp::StoreOutput { src }) = base_ops.last() else {
        return Err(direct_semantic_error(
            family,
            "direct Map is missing terminal StoreOutput",
        ));
    };
    let Some(producer_position) = definitions.get(src).copied() else {
        return Err(direct_semantic_error(
            family,
            "direct Map terminal output reads an undefined register",
        ));
    };
    let LinearOp::Binary {
        op: BinaryOp::Sub,
        lhs,
        rhs,
        ..
    } = base_ops[producer_position]
    else {
        return Err(direct_semantic_error(
            family,
            "direct Map terminal residual reaching definition is not a subtraction",
        ));
    };
    let actual_sign = if lhs == target_register {
        1
    } else if rhs == target_register {
        -1
    } else {
        return Err(direct_semantic_error(
            family,
            "direct Map terminal residual does not contain its target load",
        ));
    };
    if actual_sign != family.residual_sign {
        return Err(direct_semantic_error(
            family,
            "direct Map residual direction disagrees with residual_sign",
        ));
    }
    Ok(())
}

fn validate_target_affine_map(
    family: &InitializationDirectFamily,
    target_position: usize,
    load_strides: &[AffineStencilLoadStride],
) -> Result<(), SolveProblemShapeContractError> {
    let mut target_terms = load_strides
        .iter()
        .filter(|stride| stride.op_position == target_position)
        .flat_map(|stride| stride.terms.iter().cloned())
        .collect::<Vec<_>>();
    target_terms.sort_unstable_by_key(|term| term.dimension);
    let mut expected_terms = family.targets.strides.clone();
    expected_terms.sort_unstable_by_key(|term| term.dimension);
    if target_terms != expected_terms {
        return Err(direct_semantic_error(
            family,
            "direct target LoadY affine map does not match family.targets",
        ));
    }
    Ok(())
}

fn validate_ssa_register_flow(
    family: &InitializationDirectFamily,
    base_ops: &[LinearOp],
) -> Result<BTreeMap<Reg, usize>, SolveProblemShapeContractError> {
    let mut definitions = BTreeMap::new();
    let mut defined = BTreeSet::new();
    for (position, op) in base_ops.iter().enumerate() {
        if !sources_are_defined(op, &defined) {
            return Err(direct_semantic_error(
                family,
                "direct Map reads a register before definition",
            ));
        }
        if let Some(dst) = op.dst_register() {
            if !defined.insert(dst) {
                return Err(direct_semantic_error(
                    family,
                    "direct Map register is defined more than once",
                ));
            }
            definitions.insert(dst, position);
        }
    }
    Ok(definitions)
}

fn sources_are_defined(op: &LinearOp, defined: &BTreeSet<Reg>) -> bool {
    match *op {
        LinearOp::Const { .. }
        | LinearOp::LoadTime { .. }
        | LinearOp::LoadY { .. }
        | LinearOp::LoadP { .. }
        | LinearOp::LoadSeed { .. } => true,
        LinearOp::Move { src, .. }
        | LinearOp::Unary { arg: src, .. }
        | LinearOp::LoadIndexedP { index: src, .. }
        | LinearOp::LoadIndexedSeed { index: src, .. }
        | LinearOp::StoreOutput { src } => defined.contains(&src),
        LinearOp::Binary { lhs, rhs, .. } | LinearOp::Compare { lhs, rhs, .. } => {
            defined.contains(&lhs) && defined.contains(&rhs)
        }
        LinearOp::Select {
            cond,
            if_true,
            if_false,
            ..
        } => defined.contains(&cond) && defined.contains(&if_true) && defined.contains(&if_false),
        LinearOp::LinearSolveComponent {
            matrix_start,
            rhs_start,
            n,
            component,
            ..
        } => {
            let Some(matrix_len) = n.checked_mul(n) else {
                return false;
            };
            component < n
                && register_range_is_defined(defined, matrix_start, matrix_len)
                && register_range_is_defined(defined, rhs_start, n)
        }
        LinearOp::TableBounds { table_id, .. } => defined.contains(&table_id),
        LinearOp::TableLookup {
            table_id,
            column,
            input,
            ..
        }
        | LinearOp::TableLookupSlope {
            table_id,
            column,
            input,
            ..
        } => defined.contains(&table_id) && defined.contains(&column) && defined.contains(&input),
        LinearOp::TableNextEvent { table_id, time, .. } => {
            defined.contains(&table_id) && defined.contains(&time)
        }
        LinearOp::RandomInitialState {
            local_seed,
            global_seed,
            ..
        } => defined.contains(&local_seed) && defined.contains(&global_seed),
        LinearOp::RandomResult {
            state_start,
            state_len,
            ..
        }
        | LinearOp::RandomState {
            state_start,
            state_len,
            ..
        } => register_range_is_defined(defined, state_start, state_len),
        LinearOp::ImpureRandomInit { seed, .. } => defined.contains(&seed),
        LinearOp::ImpureRandom { id, .. } => defined.contains(&id),
        LinearOp::ImpureRandomInteger { id, imin, imax, .. } => {
            defined.contains(&id) && defined.contains(&imin) && defined.contains(&imax)
        }
        LinearOp::ExternalCall {
            args, arg_count, ..
        } => {
            arg_count <= args.len()
                && args
                    .iter()
                    .take(arg_count)
                    .all(|argument| defined.contains(argument))
        }
    }
}

fn register_range_is_defined(defined: &BTreeSet<Reg>, start: Reg, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    let Some(last) = Reg::try_from(len - 1)
        .ok()
        .and_then(|offset| start.checked_add(offset))
    else {
        return false;
    };
    defined.range(start..=last).count() == len
}
