//! Lean, backend-neutral initial-condition settlement for GPU preparation.
//!
//! This path intentionally accepts only lowering-proven direct assignments.
//! General nonlinear or coupled initialization remains unsupported here rather
//! than silently using a finite-difference CPU projection.

use rumoca_ir_solve as solve;

const INITIAL_RESIDUAL_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuInitializationMetrics {
    pub residual_evaluations: usize,
    pub passes: usize,
    pub temporary_values: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum GpuInitializationError {
    #[error("GPU initial projection does not support {feature} (row={row}, span={span:?})")]
    Unsupported {
        feature: &'static str,
        row: usize,
        span: Option<rumoca_core::Span>,
    },
    #[error("GPU initial projection is malformed: {message} (row={row}, span={span:?})")]
    Malformed {
        message: String,
        row: usize,
        span: Option<rumoca_core::Span>,
    },
    #[error(
        "GPU initial projection {kind} did not settle (row={row}, value={value:.6e}, span={span:?})"
    )]
    NonConverged {
        kind: &'static str,
        row: usize,
        value: f64,
        span: Option<rumoca_core::Span>,
    },
    #[error("GPU initial projection evaluation failed: {message} (span={span:?})")]
    Evaluation {
        message: String,
        span: Option<rumoca_core::Span>,
    },
}

#[derive(Debug)]
pub struct GpuInitializationResult {
    pub y0: Vec<f64>,
    pub p0: Vec<f64>,
    pub metrics: GpuInitializationMetrics,
}

/// Settle a GPU-prepared model without introducing a continuous solver/JVP
/// payload.  The artifact is complete-or-error: input vectors are cloned and
/// never exposed after a failed evaluation or residual check.
pub fn settle_gpu_initial_conditions(
    model: &solve::SolveModel,
    t_start: f64,
) -> Result<GpuInitializationResult, GpuInitializationError> {
    let initialization = &model.problem.initialization;
    reject_unsupported_runtime_features(model)?;
    let mut y0 = model.initial_y.clone();
    let p0 = model.parameters.clone();
    ensure_finite(&y0, "initial y", None)?;
    ensure_finite(&p0, "initial p", None)?;
    validate_assignment_shape(initialization, model.initial_y.len())?;
    if initialization.residual.is_empty() {
        return Ok(GpuInitializationResult {
            y0,
            p0,
            metrics: GpuInitializationMetrics::default(),
        });
    }
    let runtime_state = rumoca_eval_solve::SimulationRuntimeState::new();
    let eval_context = rumoca_eval_solve::RowEvalContext {
        external_tables: Some(model.external_tables.as_slice()),
        runtime_state: Some(&runtime_state),
        ..Default::default()
    };
    let mut worst = (0usize, 0.0f64, None);
    let mut native_metrics = rumoca_eval_solve::MapEvaluationMetrics::default();
    for block in &initialization.projection_plan.blocks {
        let family = compact_projection_family(initialization, block)?;
        execute_direct_family(
            family,
            DirectFamilyExecution {
                initialization,
                y: &mut y0,
                p: &p0,
                t: t_start,
                context: eval_context,
                apply: true,
                worst: &mut worst,
                metrics: &mut native_metrics,
            },
        )?;
    }
    ensure_finite(&y0, "settled y", None)?;
    worst = (0usize, 0.0f64, None);
    for family in &initialization.direct_families {
        execute_direct_family(
            family,
            DirectFamilyExecution {
                initialization,
                y: &mut y0,
                p: &p0,
                t: t_start,
                context: eval_context,
                apply: false,
                worst: &mut worst,
                metrics: &mut native_metrics,
            },
        )?;
    }
    if !worst.1.is_finite() || worst.1.abs() > INITIAL_RESIDUAL_TOLERANCE {
        return Err(GpuInitializationError::NonConverged {
            kind: "residual",
            row: worst.0,
            value: worst.1,
            span: worst.2,
        });
    }
    Ok(GpuInitializationResult {
        y0,
        p0,
        metrics: GpuInitializationMetrics {
            residual_evaluations: 2,
            passes: 1,
            temporary_values: native_metrics
                .temporary_values
                .saturating_add(initialization.direct_families.len()),
        },
    })
}

fn validate_assignment_shape(
    initialization: &solve::InitializationSolveSystem,
    y_len: usize,
) -> Result<(), GpuInitializationError> {
    solve::validate_compact_gpu_initialization(initialization, y_len).map_err(|error| {
        GpuInitializationError::Malformed {
            message: error.to_string(),
            row: 0,
            span: error.source_span(),
        }
    })?;
    if !initialization.residual.is_empty() && initialization.direct_families.is_empty() {
        return Err(GpuInitializationError::Unsupported {
            feature: "non-direct or incomplete initial residual system",
            row: 0,
            span: None,
        });
    }
    Ok(())
}

fn compact_projection_family<'a>(
    initialization: &'a solve::InitializationSolveSystem,
    block: &solve::AlgebraicProjectionBlock,
) -> Result<&'a solve::InitializationDirectFamily, GpuInitializationError> {
    let [family_index] = block.rows.as_slice() else {
        return Err(GpuInitializationError::Malformed {
            message: "compact projection block must name exactly one direct family".to_string(),
            row: 0,
            span: initialization
                .direct_families
                .first()
                .map(|family| family.span),
        });
    };
    let family = initialization
        .direct_families
        .get(*family_index)
        .ok_or_else(|| GpuInitializationError::Malformed {
            message: "compact projection direct family index is out of bounds".to_string(),
            row: *family_index,
            span: initialization
                .direct_families
                .first()
                .map(|family| family.span),
        })?;
    if block.y_indices.as_slice() != [family.targets.start] || !block.causal_steps.is_empty() {
        return Err(GpuInitializationError::Malformed {
            message: "compact projection block has an invalid target anchor".to_string(),
            row: *family_index,
            span: Some(family.span),
        });
    }
    Ok(family)
}

struct DirectFamilyExecution<'a> {
    initialization: &'a solve::InitializationSolveSystem,
    y: &'a mut [f64],
    p: &'a [f64],
    t: f64,
    context: rumoca_eval_solve::RowEvalContext<'a>,
    apply: bool,
    worst: &'a mut (usize, f64, Option<rumoca_core::Span>),
    metrics: &'a mut rumoca_eval_solve::MapEvaluationMetrics,
}

fn execute_direct_family(
    family: &solve::InitializationDirectFamily,
    execution: DirectFamilyExecution<'_>,
) -> Result<(), GpuInitializationError> {
    let Some(node @ solve::ComputeNode::Map { .. }) = execution
        .initialization
        .residual
        .nodes
        .get(family.node_index)
    else {
        return Err(GpuInitializationError::Unsupported {
            feature: "non-Map direct initial family",
            row: 0,
            span: Some(family.span),
        });
    };
    let evaluation = rumoca_eval_solve::eval_map_elements_with_context(
        node,
        execution.y,
        execution.p,
        execution.t,
        execution.context,
        |ordinal, value, y| {
            let row = direct_map_index(&family.targets, ordinal, family.span).map_err(|error| {
                rumoca_eval_solve::EvalSolveError::InvalidRow {
                    message: error.to_string(),
                    span: Some(family.span),
                }
            })?;
            if !value.is_finite() {
                return Err(rumoca_eval_solve::EvalSolveError::InvalidRow {
                    message: format!("non-finite direct initial residual at y[{row}]"),
                    span: Some(family.span),
                });
            }
            if execution.apply {
                *y.get_mut(row).ok_or_else(|| {
                    rumoca_eval_solve::EvalSolveError::InvalidRow {
                        message: format!("direct target y[{row}] is outside the state vector"),
                        span: Some(family.span),
                    }
                })? -= f64::from(family.residual_sign) * value;
            }
            if value.abs() > execution.worst.1.abs() {
                *execution.worst = (row, value, Some(family.span));
            }
            Ok(())
        },
    )
    .map_err(|error| GpuInitializationError::Evaluation {
        message: error.to_string(),
        span: error.source_span().or(Some(family.span)),
    })?;
    execution.metrics.elements = execution
        .metrics
        .elements
        .saturating_add(evaluation.elements);
    execution.metrics.temporary_values = execution
        .metrics
        .temporary_values
        .max(evaluation.temporary_values);
    Ok(())
}

fn direct_map_index(
    map: &solve::TensorOutputMap,
    ordinal: &[usize],
    span: rumoca_core::Span,
) -> Result<usize, GpuInitializationError> {
    let offset = map
        .strides
        .iter()
        .try_fold(0isize, |total, term| {
            total.checked_add(
                term.stride
                    .checked_mul(isize::try_from(*ordinal.get(term.dimension)?).ok()?)?,
            )
        })
        .ok_or_else(|| GpuInitializationError::Malformed {
            message: "direct target map overflow".to_string(),
            row: 0,
            span: Some(span),
        })?;
    map.start
        .checked_add_signed(offset)
        .ok_or_else(|| GpuInitializationError::Malformed {
            message: "direct target map overflow".to_string(),
            row: 0,
            span: Some(span),
        })
}

fn reject_unsupported_runtime_features(
    model: &solve::SolveModel,
) -> Result<(), GpuInitializationError> {
    let problem = &model.problem;
    let has_events = !problem.events.root_conditions.is_empty()
        || !problem.events.root_relation_memory_targets.is_empty()
        || !problem.events.scheduled_root_conditions.is_empty()
        || !problem.events.scheduled_time_events.is_empty()
        || !problem.events.dynamic_time_event_names.is_empty()
        || !problem.events.dynamic_time_event_rhs.is_empty()
        || !problem.events.action_conditions.is_empty()
        || !problem.events.actions.is_empty();
    let has_discrete = !problem.discrete.runtime_assignment_rhs.is_empty()
        || !problem.discrete.rhs.is_empty()
        || !problem.discrete.update_targets.is_empty()
        || !problem.discrete.pre_modes.is_empty();
    let has_memory = !problem
        .solve_layout
        .relation_memory_parameter_indices
        .is_empty()
        || !problem.solve_layout.pre_param_bindings.is_empty();
    if has_events
        || has_discrete
        || has_memory
        || !problem.clocks.periodic_event_schedules.is_empty()
    {
        return Err(GpuInitializationError::Unsupported {
            feature: "event, discrete, pre, relation-memory, or clock initialization",
            row: 0,
            span: None,
        });
    }
    Ok(())
}

fn ensure_finite(
    values: &[f64],
    kind: &'static str,
    span: Option<rumoca_core::Span>,
) -> Result<(), GpuInitializationError> {
    if let Some((row, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(GpuInitializationError::NonConverged {
            kind,
            row,
            value,
            span,
        });
    }
    let _ = kind;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumoca_ir_solve::{
        AffineStencilIndexStrideTerm, BinaryOp, ComputeBlock, ComputeNode, LinearOp,
        TensorNodeMetadata, TensorOutputMap,
    };

    fn span() -> rumoca_core::Span {
        rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("gpu_initialization_test.mo"),
            1,
            2,
        )
    }

    fn direct_model() -> solve::SolveModel {
        let span = span();
        let mut rows = vec![
            vec![
                LinearOp::LoadY { dst: 0, index: 0 },
                LinearOp::Const { dst: 1, value: 2.0 },
                LinearOp::Binary {
                    dst: 2,
                    op: BinaryOp::Sub,
                    lhs: 0,
                    rhs: 1,
                },
                LinearOp::StoreOutput { src: 2 },
            ],
            vec![
                LinearOp::LoadY { dst: 0, index: 1 },
                LinearOp::Const { dst: 1, value: 0.0 },
                LinearOp::Binary {
                    dst: 2,
                    op: BinaryOp::Sub,
                    lhs: 0,
                    rhs: 1,
                },
                LinearOp::StoreOutput { src: 2 },
            ],
        ];
        let domain = rumoca_core::StructuredIndexDomain {
            binders: vec![rumoca_core::StructuredIndexBinder {
                id: 0,
                display_name: "i".to_string(),
                lower: 1,
                upper: 2,
                step: 1,
            }],
        };
        let residual = ComputeNode::Map {
            domain: domain.clone(),
            output_map: TensorOutputMap::dense_contiguous(0, &domain).unwrap(),
            base_ops: rows.remove(0),
            load_strides: vec![rumoca_ir_solve::AffineStencilLoadStride {
                op_position: 0,
                terms: vec![AffineStencilIndexStrideTerm {
                    dimension: 0,
                    stride: 1,
                }],
            }],
            const_strides: vec![rumoca_ir_solve::AffineStencilConstStride {
                op_position: 1,
                terms: vec![rumoca_ir_solve::AffineStencilConstStrideTerm {
                    dimension: 0,
                    stride: -2.0,
                }],
            }],
            metadata: TensorNodeMetadata::default(),
            span,
        };
        let initialization = solve::InitializationSolveSystem {
            residual: ComputeBlock {
                nodes: vec![residual.clone()],
            },
            direct_families: vec![solve::InitializationDirectFamily {
                node_index: 0,
                targets: TensorOutputMap::dense_contiguous(0, &domain).unwrap(),
                residual_sign: 1,
                span,
            }],
            required_target_ranges: vec![solve::InitializationTargetRange {
                start: 0,
                end: 2,
                span,
            }],
            projection_plan: solve::AlgebraicProjectionPlan {
                blocks: vec![solve::AlgebraicProjectionBlock {
                    rows: vec![0],
                    y_indices: vec![0],
                    causal_steps: Vec::new(),
                }],
            },
            ..Default::default()
        };
        solve::SolveModel {
            problem: solve::SolveProblem {
                initialization,
                ..Default::default()
            },
            initial_y: vec![0.0, 0.0],
            ..Default::default()
        }
    }

    fn singleton_domain(active_upper: i64) -> rumoca_core::StructuredIndexDomain {
        rumoca_core::StructuredIndexDomain {
            binders: vec![
                rumoca_core::StructuredIndexBinder {
                    id: 0,
                    display_name: "i".to_string(),
                    lower: 1,
                    upper: 1,
                    step: 1,
                },
                rumoca_core::StructuredIndexBinder {
                    id: 1,
                    display_name: "j".to_string(),
                    lower: 1,
                    upper: active_upper,
                    step: 1,
                },
            ],
        }
    }

    fn compact_projection_reverse_dependency_nodes(
        domain: &rumoca_core::StructuredIndexDomain,
        span: rumoca_core::Span,
    ) -> [ComputeNode; 2] {
        let dependent = ComputeNode::Map {
            domain: domain.clone(),
            output_map: TensorOutputMap::dense_contiguous(0, domain).unwrap(),
            base_ops: vec![
                LinearOp::LoadY { dst: 0, index: 0 },
                LinearOp::LoadY { dst: 1, index: 2 },
                LinearOp::Binary {
                    dst: 2,
                    op: BinaryOp::Sub,
                    lhs: 0,
                    rhs: 1,
                },
                LinearOp::StoreOutput { src: 2 },
            ],
            load_strides: vec![
                rumoca_ir_solve::AffineStencilLoadStride {
                    op_position: 0,
                    terms: vec![AffineStencilIndexStrideTerm {
                        dimension: 0,
                        stride: 1,
                    }],
                },
                rumoca_ir_solve::AffineStencilLoadStride {
                    op_position: 1,
                    terms: vec![AffineStencilIndexStrideTerm {
                        dimension: 0,
                        stride: 1,
                    }],
                },
            ],
            const_strides: Vec::new(),
            metadata: TensorNodeMetadata::default(),
            span,
        };
        let source = ComputeNode::Map {
            domain: domain.clone(),
            output_map: TensorOutputMap::dense_contiguous(2, domain).unwrap(),
            base_ops: vec![
                LinearOp::LoadY { dst: 0, index: 2 },
                LinearOp::Const { dst: 1, value: 1.0 },
                LinearOp::Binary {
                    dst: 2,
                    op: BinaryOp::Sub,
                    lhs: 0,
                    rhs: 1,
                },
                LinearOp::StoreOutput { src: 2 },
            ],
            load_strides: vec![rumoca_ir_solve::AffineStencilLoadStride {
                op_position: 0,
                terms: vec![AffineStencilIndexStrideTerm {
                    dimension: 0,
                    stride: 1,
                }],
            }],
            const_strides: vec![rumoca_ir_solve::AffineStencilConstStride {
                op_position: 1,
                terms: vec![rumoca_ir_solve::AffineStencilConstStrideTerm {
                    dimension: 0,
                    stride: 1.0,
                }],
            }],
            metadata: TensorNodeMetadata::default(),
            span,
        };
        [dependent, source]
    }

    #[test]
    fn direct_initial_assignment_is_one_pass_with_linear_temporary_storage() {
        let result = settle_gpu_initial_conditions(&direct_model(), 0.0)
            .expect("proven direct rows should settle");
        assert_eq!(result.y0, vec![2.0, 0.0]);
        assert_eq!(result.metrics.residual_evaluations, 2);
        assert_eq!(result.metrics.passes, 1);
        assert!(result.metrics.temporary_values <= result.y0.len() * 3);
    }

    #[test]
    fn settlement_replays_mixed_singleton_domain_without_scalar_rows() {
        let mut model = direct_model();
        let domain = singleton_domain(2);
        let solve::ComputeNode::Map {
            domain: residual_domain,
            output_map,
            load_strides,
            const_strides,
            ..
        } = &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *residual_domain = domain.clone();
        *output_map = TensorOutputMap::dense_contiguous(0, &domain).unwrap();
        load_strides[0].terms[0].dimension = 1;
        const_strides[0].terms[0].dimension = 1;
        model.problem.initialization.direct_families[0].targets =
            TensorOutputMap::dense_contiguous(0, &domain).unwrap();

        let result = settle_gpu_initial_conditions(&model, 0.0)
            .expect("mixed-singleton direct family must replay natively");
        assert_eq!(result.y0, vec![2.0, 0.0]);
    }

    #[test]
    fn settlement_replays_all_singleton_domain_with_empty_strides() {
        let mut model = direct_model();
        let domain = singleton_domain(1);
        let solve::ComputeNode::Map {
            domain: residual_domain,
            output_map,
            base_ops,
            load_strides,
            const_strides,
            ..
        } = &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *residual_domain = domain.clone();
        *output_map = TensorOutputMap::dense_contiguous(0, &domain).unwrap();
        let LinearOp::Const { value, .. } = &mut base_ops[1] else {
            unreachable!()
        };
        *value = 7.0;
        load_strides.clear();
        const_strides.clear();
        model.problem.initialization.direct_families[0].targets =
            TensorOutputMap::dense_contiguous(0, &domain).unwrap();
        model.problem.initialization.required_target_ranges[0].end = 1;
        model.initial_y.truncate(1);

        let result = settle_gpu_initial_conditions(&model, 0.0)
            .expect("all-singleton direct family must replay natively");
        assert_eq!(result.y0, vec![7.0]);
    }

    #[test]
    fn settlement_executes_compact_projection_order_before_final_verification() {
        let mut model = direct_model();
        let span = span();
        let domain = match &model.problem.initialization.residual.nodes[0] {
            ComputeNode::Map { domain, .. } => domain.clone(),
            _ => unreachable!(),
        };
        model.problem.initialization.residual.nodes =
            compact_projection_reverse_dependency_nodes(&domain, span).into();
        model.problem.initialization.direct_families = vec![
            solve::InitializationDirectFamily {
                node_index: 0,
                targets: TensorOutputMap::dense_contiguous(0, &domain).unwrap(),
                residual_sign: 1,
                span,
            },
            solve::InitializationDirectFamily {
                node_index: 1,
                targets: TensorOutputMap::dense_contiguous(2, &domain).unwrap(),
                residual_sign: 1,
                span,
            },
        ];
        model.problem.initialization.required_target_ranges[0].end = 4;
        model.problem.initialization.projection_plan = solve::AlgebraicProjectionPlan {
            blocks: vec![
                solve::AlgebraicProjectionBlock {
                    rows: vec![1],
                    y_indices: vec![2],
                    causal_steps: Vec::new(),
                },
                solve::AlgebraicProjectionBlock {
                    rows: vec![0],
                    y_indices: vec![0],
                    causal_steps: Vec::new(),
                },
            ],
        };
        model.initial_y.resize(4, 0.0);

        let result = settle_gpu_initial_conditions(&model, 0.0)
            .expect("compact projection order should settle reverse source dependencies");
        assert_eq!(result.y0, vec![1.0, 2.0, 1.0, 2.0]);
        assert_eq!(result.metrics.residual_evaluations, 2);
        assert_eq!(result.metrics.passes, 1);
    }

    #[test]
    fn settlement_semantic_gate_rejects_constant_zero_and_wrong_target_maps() {
        let mut constant = direct_model();
        let solve::ComputeNode::Map { base_ops, .. } =
            &mut constant.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *base_ops = vec![
            LinearOp::Const { dst: 0, value: 0.0 },
            LinearOp::StoreOutput { src: 0 },
        ];
        let error = settle_gpu_initial_conditions(&constant, 0.0)
            .expect_err("constant-zero direct Map cannot return an unsettled y");
        assert!(error.to_string().contains("target LoadY"), "{error}");

        let mut wrong_target = direct_model();
        let solve::ComputeNode::Map { base_ops, .. } =
            &mut wrong_target.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        let LinearOp::LoadY { index, .. } = &mut base_ops[0] else {
            unreachable!()
        };
        *index = 1;
        let error = settle_gpu_initial_conditions(&wrong_target, 0.0)
            .expect_err("wrong direct target cannot return an unsettled y");
        assert!(error.to_string().contains("target LoadY"), "{error}");
    }

    #[test]
    fn settlement_semantic_gate_rejects_output_register_overwrite() {
        let mut model = direct_model();
        let solve::ComputeNode::Map { base_ops, .. } =
            &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        base_ops.insert(3, LinearOp::Const { dst: 2, value: 0.0 });

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("overwritten residual cannot return an unsettled y");
        assert!(
            error.to_string().contains("defined more than once"),
            "{error}"
        );
    }

    #[test]
    fn settlement_semantic_gate_rejects_target_register_overwrite() {
        let mut model = direct_model();
        let solve::ComputeNode::Map { base_ops, .. } =
            &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        base_ops.insert(2, LinearOp::Move { dst: 0, src: 1 });

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("overwritten target load cannot return an unsettled y");
        assert!(
            error.to_string().contains("defined more than once"),
            "{error}"
        );
    }

    #[test]
    fn settlement_semantic_gate_rejects_target_minus_target_before_returning_y0() {
        let mut model = direct_model();
        let original_y0 = model.initial_y.clone();
        let solve::ComputeNode::Map {
            base_ops,
            const_strides,
            ..
        } = &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *base_ops = vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::Binary {
                dst: 1,
                op: BinaryOp::Sub,
                lhs: 0,
                rhs: 0,
            },
            LinearOp::StoreOutput { src: 1 },
        ];
        const_strides.clear();

        let result = settle_gpu_initial_conditions(&model, 0.0);
        if let Ok(success) = &result {
            assert_ne!(
                success.y0, original_y0,
                "target - target must not return the original unsettled y0 as success"
            );
        }
        let error = result.expect_err("target - target must fail semantic admission");
        assert!(
            error.to_string().contains("depends on target LoadY"),
            "{error}"
        );
    }

    #[test]
    fn settlement_semantic_gate_rejects_transitive_target_dependency_before_returning_y0() {
        let mut model = direct_model();
        let original_y0 = model.initial_y.clone();
        let solve::ComputeNode::Map {
            base_ops,
            const_strides,
            ..
        } = &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *base_ops = vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::Move { dst: 1, src: 0 },
            LinearOp::Unary {
                dst: 2,
                op: solve::UnaryOp::Neg,
                arg: 1,
            },
            LinearOp::Const { dst: 3, value: 0.0 },
            LinearOp::Compare {
                dst: 4,
                op: solve::CompareOp::Eq,
                lhs: 3,
                rhs: 3,
            },
            LinearOp::Select {
                dst: 5,
                cond: 4,
                if_true: 2,
                if_false: 3,
            },
            LinearOp::Binary {
                dst: 6,
                op: BinaryOp::Sub,
                lhs: 0,
                rhs: 5,
            },
            LinearOp::StoreOutput { src: 6 },
        ];
        const_strides.clear();

        let result = settle_gpu_initial_conditions(&model, 0.0);
        if let Ok(success) = &result {
            assert_ne!(
                success.y0, original_y0,
                "transitive target dependency must not return the original unsettled y0 as success"
            );
        }
        let error = result.expect_err("transitive target dependency must fail semantic admission");
        assert!(
            error.to_string().contains("depends on target LoadY"),
            "{error}"
        );
    }

    #[test]
    fn settlement_semantic_gate_rejects_dependency_reorder_and_cycle() {
        let mut reordered = direct_model();
        let span = span();
        let domain = match &reordered.problem.initialization.residual.nodes[0] {
            ComputeNode::Map { domain, .. } => domain.clone(),
            _ => unreachable!(),
        };
        reordered.problem.initialization.residual.nodes =
            compact_projection_reverse_dependency_nodes(&domain, span).into();
        reordered.problem.initialization.direct_families = vec![
            solve::InitializationDirectFamily {
                node_index: 0,
                targets: TensorOutputMap::dense_contiguous(0, &domain).unwrap(),
                residual_sign: 1,
                span,
            },
            solve::InitializationDirectFamily {
                node_index: 1,
                targets: TensorOutputMap::dense_contiguous(2, &domain).unwrap(),
                residual_sign: 1,
                span,
            },
        ];
        reordered.problem.initialization.required_target_ranges[0].end = 4;
        reordered.problem.initialization.projection_plan = solve::AlgebraicProjectionPlan {
            blocks: vec![
                solve::AlgebraicProjectionBlock {
                    rows: vec![0],
                    y_indices: vec![0],
                    causal_steps: Vec::new(),
                },
                solve::AlgebraicProjectionBlock {
                    rows: vec![1],
                    y_indices: vec![2],
                    causal_steps: Vec::new(),
                },
            ],
        };
        reordered.initial_y.resize(4, 0.0);
        let error = settle_gpu_initial_conditions(&reordered, 0.0)
            .expect_err("dependency reorder cannot return an unsettled y");
        assert!(error.to_string().contains("dependency order"), "{error}");

        let mut cycle = reordered;
        let ComputeNode::Map {
            base_ops,
            load_strides,
            ..
        } = &mut cycle.problem.initialization.residual.nodes[1]
        else {
            unreachable!()
        };
        base_ops[1] = LinearOp::LoadY { dst: 1, index: 0 };
        load_strides.push(rumoca_ir_solve::AffineStencilLoadStride {
            op_position: 1,
            terms: vec![AffineStencilIndexStrideTerm {
                dimension: 0,
                stride: 1,
            }],
        });
        let error = settle_gpu_initial_conditions(&cycle, 0.0)
            .expect_err("dependency cycle cannot return an unsettled y");
        assert!(error.to_string().contains("dependency order"), "{error}");
    }

    #[test]
    fn direct_initial_assignment_uses_model_external_tables_for_apply_and_verify() {
        let mut model = direct_model();
        let table_id = 515_151_u64;
        model.external_tables = solve::ExternalTables::new(vec![rumoca_core::ExternalTableData {
            id: table_id,
            data: vec![vec![1.0, 10.0], vec![3.0, 30.0]],
            columns: vec![2],
            smoothness: 3,
            extrapolation: 1,
        }]);
        let solve::ComputeNode::Map {
            base_ops,
            const_strides,
            ..
        } = &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *base_ops = vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::Const {
                dst: 1,
                value: table_id as f64,
            },
            LinearOp::Const { dst: 2, value: 1.0 },
            LinearOp::Const { dst: 3, value: 2.0 },
            LinearOp::TableLookup {
                dst: 4,
                table_id: 1,
                column: 2,
                input: 3,
            },
            LinearOp::Binary {
                dst: 5,
                op: BinaryOp::Sub,
                lhs: 0,
                rhs: 4,
            },
            LinearOp::StoreOutput { src: 5 },
        ];
        const_strides.clear();

        let settled = settle_gpu_initial_conditions(&model, 0.0)
            .expect("table-backed direct initialization should settle and verify");
        assert_eq!(settled.y0, vec![10.0, 10.0]);
    }

    #[test]
    fn settlement_semantic_gate_rejects_malformed_random_with_owner_span() {
        let mut model = direct_model();
        let solve::ComputeNode::Map {
            base_ops,
            const_strides,
            ..
        } = &mut model.problem.initialization.residual.nodes[0]
        else {
            unreachable!()
        };
        *base_ops = vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::Const { dst: 1, value: 1.0 },
            LinearOp::Const { dst: 2, value: 2.0 },
            LinearOp::RandomInitialState {
                dst: 3,
                generator: solve::RandomGenerator::Xorshift64Star,
                local_seed: 1,
                global_seed: 2,
                state_len: 0,
                state_index: 0,
            },
            LinearOp::Binary {
                dst: 4,
                op: BinaryOp::Sub,
                lhs: 0,
                rhs: 3,
            },
            LinearOp::StoreOutput { src: 4 },
        ];
        const_strides.clear();

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("malformed random initialization must fail shared semantic admission");
        assert!(matches!(
            error,
            GpuInitializationError::Malformed {
                ref message,
                span: Some(actual),
                ..
            } if message.contains("random or impure direct Map operation") && actual == span()
        ));
    }

    #[test]
    fn settlement_rejects_partial_required_target_union_independently() {
        let mut model = direct_model();
        model.problem.initialization.required_target_ranges[0].end = 1;
        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("settlement must reject a partial hand-built artifact");
        assert!(error.to_string().contains("incomplete"));
    }

    #[test]
    fn settlement_rejects_partial_fixed_only_metadata_before_empty_residual_return() {
        let span = rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("fixed_only_partial.mo"),
            20,
            30,
        );
        let mut model = solve::SolveModel {
            initial_y: vec![1.0, 2.0],
            ..Default::default()
        };
        model.problem.initialization.required_target_ranges =
            vec![solve::InitializationTargetRange {
                start: 0,
                end: 2,
                span,
            }];
        model.problem.initialization.fixed_target_ranges = vec![solve::InitializationTargetRange {
            start: 0,
            end: 1,
            span,
        }];

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("partial fixed-only metadata must be validated before early return");
        assert!(error.to_string().contains("incomplete"));
        assert!(matches!(
            error,
            GpuInitializationError::Malformed { span: Some(actual), .. } if actual == span
        ));
    }

    #[test]
    fn settlement_rejects_duplicate_direct_node_ownership() {
        let mut model = direct_model();
        let node = model.problem.initialization.residual.nodes[0].clone();
        model.problem.initialization.residual.nodes.push(node);
        let mut duplicate = model.problem.initialization.direct_families[0].clone();
        duplicate.targets.start = 2;
        model.problem.initialization.direct_families.push(duplicate);
        model.problem.initialization.required_target_ranges[0].end = 4;
        model.initial_y.resize(4, 0.0);

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("every residual node must have one unique direct owner");
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn settlement_rejects_unowned_residual_node() {
        let mut model = direct_model();
        model
            .problem
            .initialization
            .residual
            .nodes
            .push(model.problem.initialization.residual.nodes[0].clone());

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("every residual node must have a direct owner");
        assert!(error.to_string().contains("own every residual node"));
    }

    #[test]
    fn settlement_rejects_direct_fixed_overlap_at_fixed_span() {
        let span = rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("direct_fixed_overlap.mo"),
            40,
            50,
        );
        let mut model = direct_model();
        model.problem.initialization.fixed_target_ranges = vec![solve::InitializationTargetRange {
            start: 1,
            end: 2,
            span,
        }];

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("simulation must reject direct/fixed ownership overlap");
        assert!(error.to_string().contains("overlap"));
        assert!(matches!(
            error,
            GpuInitializationError::Malformed { span: Some(actual), .. } if actual == span
        ));
    }

    #[test]
    fn settlement_accepts_adjacent_complete_fixed_only_ranges() {
        let span = rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("adjacent_fixed.mo"),
            10,
            20,
        );
        let mut model = solve::SolveModel {
            initial_y: vec![-0.0, 2.0],
            ..Default::default()
        };
        model.problem.initialization.required_target_ranges =
            vec![solve::InitializationTargetRange {
                start: 0,
                end: 2,
                span,
            }];
        model.problem.initialization.fixed_target_ranges = vec![
            solve::InitializationTargetRange {
                start: 0,
                end: 1,
                span,
            },
            solve::InitializationTargetRange {
                start: 1,
                end: 2,
                span,
            },
        ];

        let settled = settle_gpu_initial_conditions(&model, 0.0)
            .expect("adjacent fixed ranges form exact complete coverage");
        assert_eq!(settled.y0[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(settled.y0[1], 2.0);
    }

    #[test]
    fn settlement_invalid_range_reports_span_after_json() {
        let span = rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("invalid_range_roundtrip.mo"),
            70,
            80,
        );
        let range = solve::InitializationTargetRange {
            start: 0,
            end: 3,
            span,
        };
        let json = serde_json::to_string(&range).expect("serialize invalid range JSON");
        let from_json: solve::InitializationTargetRange =
            serde_json::from_str(&json).expect("deserialize invalid range JSON");
        let mut model = solve::SolveModel {
            initial_y: vec![0.0, 0.0],
            ..Default::default()
        };
        model.problem.initialization.required_target_ranges = vec![from_json];
        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("out-of-bounds range must fail closed");
        assert!(matches!(
            error,
            GpuInitializationError::Malformed { span: Some(actual), .. } if actual == span
        ));
    }

    #[test]
    fn event_system_is_rejected_before_returning_initial_vectors() {
        let mut model = solve::SolveModel::default();
        model.problem.events.scheduled_time_events.push(0.0);
        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("GPU preparation must reject event systems");
        assert!(matches!(error, GpuInitializationError::Unsupported { .. }));
    }

    #[test]
    fn nonfinite_initial_vector_is_rejected_without_partial_settlement() {
        let mut model = direct_model();
        model.initial_y[0] = f64::NAN;

        let error = settle_gpu_initial_conditions(&model, 0.0)
            .expect_err("non-finite GPU initialization input must fail closed");
        assert!(matches!(error, GpuInitializationError::NonConverged { .. }));
        assert!(error.to_string().contains("initial y"));
    }

    #[test]
    fn no_initial_equations_preserve_finite_vectors_exactly() {
        let model = solve::SolveModel {
            initial_y: vec![-0.0, 3.25],
            parameters: vec![7.5, -2.0],
            ..Default::default()
        };
        let result = settle_gpu_initial_conditions(&model, 0.0).expect("finite vectors pass");
        assert_eq!(
            result.y0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            model
                .initial_y
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            result.p0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            model
                .parameters
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn no_initial_equations_still_reject_nonfinite_vectors() {
        let model = solve::SolveModel {
            initial_y: vec![f64::INFINITY],
            ..Default::default()
        };
        assert!(settle_gpu_initial_conditions(&model, 0.0).is_err());
    }
}
