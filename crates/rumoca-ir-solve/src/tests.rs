use super::*;
use indexmap::IndexMap;
use rumoca_core::{SourceId, StructuredIndexBinder};

const REPRESENTATIVE_SOLVE_PROBLEM_GOLDEN: &str =
    include_str!("../tests/golden/representative_solve_problem.solve.json");

fn test_tensor_domain(count: usize) -> StructuredIndexDomain {
    StructuredIndexDomain {
        binders: vec![StructuredIndexBinder {
            id: 0,
            display_name: "i".to_string(),
            lower: 1,
            upper: count as i64,
            step: 1,
        }],
    }
}

fn direct_initialization_family(
    node_index: usize,
    target_start: usize,
    target_strides: Vec<AffineStencilIndexStrideTerm>,
) -> (ComputeNode, InitializationDirectFamily) {
    let domain = test_tensor_domain(3);
    let target_load_strides = if target_strides.is_empty() {
        Vec::new()
    } else {
        vec![AffineStencilLoadStride {
            op_position: 0,
            terms: target_strides.clone(),
        }]
    };
    let node = ComputeNode::Map {
        output_map: TensorOutputMap::dense_contiguous(node_index * 3, &domain)
            .expect("dense residual map"),
        domain,
        base_ops: vec![
            LinearOp::LoadY {
                dst: 0,
                index: target_start,
            },
            LinearOp::Const { dst: 1, value: 0.0 },
            LinearOp::Binary {
                op: BinaryOp::Sub,
                lhs: 0,
                rhs: 1,
                dst: 2,
            },
            LinearOp::StoreOutput { src: 2 },
        ],
        load_strides: target_load_strides,
        const_strides: Vec::new(),
        metadata: TensorNodeMetadata::default(),
        span: fixture_span(),
    };
    let family = InitializationDirectFamily {
        node_index,
        targets: TensorOutputMap {
            start: target_start,
            strides: target_strides,
        },
        residual_sign: 1,
        span: fixture_span(),
    };
    (node, family)
}

#[test]
fn compact_initialization_validation_rejects_noncontiguous_target_strides() {
    let (node, family) = direct_initialization_family(
        0,
        0,
        vec![AffineStencilIndexStrideTerm {
            dimension: 0,
            stride: 2,
        }],
    );
    let initialization = InitializationSolveSystem {
        residual: ComputeBlock { nodes: vec![node] },
        direct_families: vec![family],
        ..Default::default()
    };

    let error = validate_initialization_direct_families(&initialization, 6, 3)
        .expect_err("sparse compact target maps must fail closed");
    assert!(error.to_string().contains("non-contiguous"));
}

#[test]
fn compact_initialization_validation_rejects_negative_target_strides() {
    let (node, family) = direct_initialization_family(
        0,
        2,
        vec![AffineStencilIndexStrideTerm {
            dimension: 0,
            stride: -1,
        }],
    );
    let initialization = InitializationSolveSystem {
        residual: ComputeBlock { nodes: vec![node] },
        direct_families: vec![family],
        required_target_ranges: vec![InitializationTargetRange {
            start: 0,
            end: 3,
            span: fixture_span(),
        }],
        ..Default::default()
    };

    let error = validate_initialization_direct_families(&initialization, 3, 3)
        .expect_err("descending target maps must fail closed");
    assert!(error.to_string().contains("non-contiguous"));
}

#[test]
fn compact_initialization_validation_rejects_overlapping_affine_ranges() {
    let dense = vec![AffineStencilIndexStrideTerm {
        dimension: 0,
        stride: 1,
    }];
    let (first_node, first_family) = direct_initialization_family(0, 0, dense.clone());
    let (second_node, second_family) = direct_initialization_family(1, 2, dense);
    let initialization = InitializationSolveSystem {
        residual: ComputeBlock {
            nodes: vec![first_node, second_node],
        },
        direct_families: vec![first_family, second_family],
        ..Default::default()
    };

    let error = validate_initialization_direct_families(&initialization, 6, 6)
        .expect_err("overlapping compact target ranges must fail closed");
    assert!(error.to_string().contains("overlapping"));
}

#[test]
fn compact_initialization_validation_rejects_direct_fixed_overlap() {
    let mut initialization = complete_compact_initialization();
    initialization.fixed_target_ranges = vec![InitializationTargetRange {
        start: 1,
        end: 2,
        span: fixture_span(),
    }];

    let error = validate_initialization_direct_families(&initialization, 3, 3)
        .expect_err("direct and fixed-start target ownership must not overlap");
    assert!(error.to_string().contains("overlap"));
    assert_eq!(error.source_span(), Some(fixture_span()));
}

#[test]
fn compact_initialization_validation_rejects_fixed_fixed_overlap() {
    let initialization = InitializationSolveSystem {
        required_target_ranges: vec![InitializationTargetRange {
            start: 0,
            end: 3,
            span: fixture_span(),
        }],
        fixed_target_ranges: vec![
            InitializationTargetRange {
                start: 0,
                end: 2,
                span: fixture_span(),
            },
            InitializationTargetRange {
                start: 1,
                end: 3,
                span: fixture_span(),
            },
        ],
        ..Default::default()
    };

    let error = validate_initialization_direct_families(&initialization, 3, 0)
        .expect_err("fixed-start target ownership must not overlap");
    assert!(error.to_string().contains("overlap"));
    assert_eq!(error.source_span(), Some(fixture_span()));
}

#[test]
fn compact_initialization_validation_merges_adjacent_fixed_ranges() {
    let initialization = InitializationSolveSystem {
        required_target_ranges: vec![InitializationTargetRange {
            start: 0,
            end: 3,
            span: fixture_span(),
        }],
        fixed_target_ranges: vec![
            InitializationTargetRange {
                start: 0,
                end: 1,
                span: fixture_span(),
            },
            InitializationTargetRange {
                start: 1,
                end: 3,
                span: fixture_span(),
            },
        ],
        ..Default::default()
    };

    validate_initialization_direct_families(&initialization, 3, 0)
        .expect("adjacent target ranges are one exact partition");
}

fn complete_compact_initialization() -> InitializationSolveSystem {
    let (node, family) = direct_initialization_family(
        0,
        0,
        vec![AffineStencilIndexStrideTerm {
            dimension: 0,
            stride: 1,
        }],
    );
    InitializationSolveSystem {
        residual: ComputeBlock { nodes: vec![node] },
        direct_families: vec![family],
        required_target_ranges: vec![InitializationTargetRange {
            start: 0,
            end: 3,
            span: fixture_span(),
        }],
        projection_plan: AlgebraicProjectionPlan {
            blocks: vec![AlgebraicProjectionBlock {
                rows: vec![0],
                y_indices: vec![0],
                causal_steps: Vec::new(),
            }],
        },
        ..Default::default()
    }
}

fn assert_compact_wire_rejects(initialization: InitializationSolveSystem, expected: &str) {
    let problem = compact_problem(initialization);
    let json = serde_json::to_string(&problem).expect("serialize invalid compact JSON");
    let json_error = serde_json::from_str::<SolveProblem>(&json)
        .expect_err("invalid compact JSON must fail semantic admission");
    assert!(json_error.to_string().contains(expected), "{json_error}");
    let bytes = bincode::serialize(&problem).expect("serialize invalid compact bincode");
    let bincode_error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("invalid compact bincode must fail semantic admission");
    assert!(
        bincode_error.to_string().contains(expected),
        "{bincode_error}"
    );
}

fn compact_problem(initialization: InitializationSolveSystem) -> SolveProblem {
    SolveProblem {
        layout: make_layout(&[("x", vec![3])], &[]),
        initialization,
        ..Default::default()
    }
}

fn compact_initialization_with_base_ops(base_ops: Vec<LinearOp>) -> InitializationSolveSystem {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map {
        base_ops: actual, ..
    } = &mut initialization.residual.nodes[0]
    else {
        unreachable!()
    };
    *actual = base_ops;
    initialization
}

fn malformed_random_initialization() -> InitializationSolveSystem {
    compact_initialization_with_base_ops(vec![
        LinearOp::LoadY { dst: 0, index: 0 },
        LinearOp::Const { dst: 1, value: 1.0 },
        LinearOp::Const { dst: 2, value: 2.0 },
        LinearOp::RandomInitialState {
            dst: 3,
            generator: RandomGenerator::Xorshift64Star,
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
    ])
}

fn compact_initialization_with_const_stride(
    op_position: usize,
    dimension: usize,
    stride: f64,
) -> InitializationSolveSystem {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map { const_strides, .. } = &mut initialization.residual.nodes[0] else {
        unreachable!()
    };
    const_strides.push(AffineStencilConstStride {
        op_position,
        terms: vec![AffineStencilConstStrideTerm { dimension, stride }],
    });
    initialization
}

fn compact_initialization_with_non_target_load_stride() -> InitializationSolveSystem {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map { load_strides, .. } = &mut initialization.residual.nodes[0] else {
        unreachable!()
    };
    load_strides.push(AffineStencilLoadStride {
        op_position: 1,
        terms: vec![AffineStencilIndexStrideTerm {
            dimension: 0,
            stride: 1,
        }],
    });
    initialization
}

#[test]
fn compact_initialization_json_rejects_const_stride_targeting_load_y() {
    let problem = compact_problem(compact_initialization_with_const_stride(0, 0, 1.0));
    let json = serde_json::to_string(&problem).expect("serialize malformed affine JSON");

    let error = serde_json::from_str::<SolveProblem>(&json)
        .expect_err("const stride targeting LoadY must fail JSON admission");
    assert!(
        error
            .to_string()
            .contains("affine constant stride does not point at Const"),
        "{error}"
    );
}

#[test]
fn compact_initialization_bincode_rejects_const_stride_targeting_load_y() {
    let problem = compact_problem(compact_initialization_with_const_stride(0, 0, 1.0));
    let bytes = bincode::serialize(&problem).expect("serialize malformed affine bincode");

    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("const stride targeting LoadY must fail bincode admission");
    assert!(
        error
            .to_string()
            .contains("affine constant stride does not point at Const"),
        "{error}"
    );
}

#[test]
fn compact_initialization_json_rejects_non_target_load_stride_targeting_const() {
    let problem = compact_problem(compact_initialization_with_non_target_load_stride());
    let json = serde_json::to_string(&problem).expect("serialize malformed affine JSON");

    let error = serde_json::from_str::<SolveProblem>(&json)
        .expect_err("non-target load stride targeting Const must fail JSON admission");
    assert!(
        error
            .to_string()
            .contains("affine load stride does not point at LoadY or LoadP"),
        "{error}"
    );
}

#[test]
fn compact_initialization_bincode_rejects_non_target_load_stride_targeting_const() {
    let problem = compact_problem(compact_initialization_with_non_target_load_stride());
    let bytes = bincode::serialize(&problem).expect("serialize malformed affine bincode");

    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("non-target load stride targeting Const must fail bincode admission");
    assert!(
        error
            .to_string()
            .contains("affine load stride does not point at LoadY or LoadP"),
        "{error}"
    );
}

#[test]
fn compact_initialization_json_rejects_const_stride_dimension_out_of_bounds() {
    let problem = compact_problem(compact_initialization_with_const_stride(1, 1, 1.0));
    let json = serde_json::to_string(&problem).expect("serialize malformed affine JSON");

    let error = serde_json::from_str::<SolveProblem>(&json)
        .expect_err("out-of-bounds const stride dimension must fail JSON admission");
    assert!(
        error
            .to_string()
            .contains("affine stride dimension is outside domain"),
        "{error}"
    );
}

#[test]
fn compact_initialization_bincode_rejects_const_stride_dimension_out_of_bounds() {
    let problem = compact_problem(compact_initialization_with_const_stride(1, 1, 1.0));
    let bytes = bincode::serialize(&problem).expect("serialize malformed affine bincode");

    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("out-of-bounds const stride dimension must fail bincode admission");
    assert!(
        error
            .to_string()
            .contains("affine stride dimension is outside domain"),
        "{error}"
    );
}

#[test]
fn compact_initialization_bincode_rejects_nonfinite_const_stride() {
    let problem = compact_problem(compact_initialization_with_const_stride(
        1,
        0,
        f64::INFINITY,
    ));
    let bytes = bincode::serialize(&problem).expect("serialize non-finite affine bincode");

    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("non-finite const stride must fail bincode admission");
    assert!(
        error
            .to_string()
            .contains("affine constant stride is non-finite"),
        "{error}"
    );
}

#[test]
fn compact_initialization_json_rejects_reachable_malformed_random_initial_state() {
    let problem = compact_problem(malformed_random_initialization());
    let json = serde_json::to_string(&problem).expect("serialize malformed random JSON");

    let error = serde_json::from_str::<SolveProblem>(&json)
        .expect_err("malformed random JSON must fail semantic admission");
    assert!(
        error
            .to_string()
            .contains("random or impure direct Map operation"),
        "{error}"
    );
}

#[test]
fn compact_initialization_bincode_rejects_reachable_malformed_random_initial_state() {
    let problem = compact_problem(malformed_random_initialization());
    let bytes = bincode::serialize(&problem).expect("serialize malformed random bincode");

    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("malformed random bincode must fail semantic admission");
    assert!(
        error
            .to_string()
            .contains("random or impure direct Map operation"),
        "{error}"
    );
}

#[test]
fn compact_initialization_rejects_every_random_and_impure_variant_with_owner_span() {
    let variants = [
        LinearOp::RandomInitialState {
            dst: 4,
            generator: RandomGenerator::Xorshift64Star,
            local_seed: 1,
            global_seed: 2,
            state_len: 2,
            state_index: 0,
        },
        LinearOp::RandomResult {
            dst: 4,
            generator: RandomGenerator::Xorshift128Plus,
            state_start: 1,
            state_len: 3,
        },
        LinearOp::RandomState {
            dst: 4,
            generator: RandomGenerator::Xorshift1024Star,
            state_start: 1,
            state_len: 3,
            state_index: 2,
        },
        LinearOp::ImpureRandomInit { dst: 4, seed: 1 },
        LinearOp::ImpureRandom {
            dst: 4,
            id: 1,
            call_site: 7,
        },
        LinearOp::ImpureRandomInteger {
            dst: 4,
            id: 1,
            imin: 2,
            imax: 3,
            call_site: 11,
        },
    ];

    for random_op in variants {
        let initialization = compact_initialization_with_base_ops(vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::Const { dst: 1, value: 1.0 },
            LinearOp::Const { dst: 2, value: 2.0 },
            LinearOp::Const { dst: 3, value: 3.0 },
            random_op,
            LinearOp::Binary {
                dst: 5,
                op: BinaryOp::Sub,
                lhs: 0,
                rhs: 4,
            },
            LinearOp::StoreOutput { src: 5 },
        ]);

        let error = validate_compact_gpu_initialization(&initialization, 3)
            .expect_err("random and impure direct Maps must fail semantic admission");
        assert!(
            error
                .to_string()
                .contains("random or impure direct Map operation"),
            "{error}"
        );
        assert_eq!(error.source_span(), Some(fixture_span()));
    }
}

#[test]
fn compact_initialization_json_and_bincode_reject_target_minus_target() {
    let initialization = compact_initialization_with_base_ops(vec![
        LinearOp::LoadY { dst: 0, index: 0 },
        LinearOp::Binary {
            dst: 1,
            op: BinaryOp::Sub,
            lhs: 0,
            rhs: 0,
        },
        LinearOp::StoreOutput { src: 1 },
    ]);

    assert_compact_wire_rejects(initialization, "depends on target LoadY");
}

#[test]
fn compact_initialization_json_and_bincode_reject_target_dependency_through_move() {
    let initialization = compact_initialization_with_base_ops(vec![
        LinearOp::LoadY { dst: 0, index: 0 },
        LinearOp::Move { dst: 1, src: 0 },
        LinearOp::Binary {
            dst: 2,
            op: BinaryOp::Sub,
            lhs: 0,
            rhs: 1,
        },
        LinearOp::StoreOutput { src: 2 },
    ]);

    assert_compact_wire_rejects(initialization, "depends on target LoadY");
}

#[test]
fn compact_initialization_json_and_bincode_reject_deep_target_dependency() {
    let initialization = compact_initialization_with_base_ops(vec![
        LinearOp::LoadY { dst: 0, index: 0 },
        LinearOp::Const { dst: 1, value: 0.0 },
        LinearOp::Unary {
            dst: 2,
            op: UnaryOp::Neg,
            arg: 0,
        },
        LinearOp::Compare {
            dst: 3,
            op: CompareOp::Eq,
            lhs: 1,
            rhs: 1,
        },
        LinearOp::Select {
            dst: 4,
            cond: 3,
            if_true: 2,
            if_false: 1,
        },
        LinearOp::Binary {
            dst: 5,
            op: BinaryOp::Add,
            lhs: 4,
            rhs: 1,
        },
        LinearOp::Binary {
            dst: 6,
            op: BinaryOp::Sub,
            lhs: 0,
            rhs: 5,
        },
        LinearOp::StoreOutput { src: 6 },
    ]);

    assert_compact_wire_rejects(initialization, "depends on target LoadY");
}

#[test]
fn compact_initialization_json_and_bincode_reject_multiple_store_outputs() {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map { base_ops, .. } = &mut initialization.residual.nodes[0] else {
        unreachable!()
    };
    base_ops.insert(3, LinearOp::StoreOutput { src: 2 });

    assert_compact_wire_rejects(initialization, "exactly one StoreOutput");
}

#[test]
fn compact_initialization_json_and_bincode_reject_constant_zero_map() {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map {
        base_ops,
        load_strides,
        ..
    } = &mut initialization.residual.nodes[0]
    else {
        unreachable!()
    };
    *base_ops = vec![
        LinearOp::Const { dst: 0, value: 0.0 },
        LinearOp::StoreOutput { src: 0 },
    ];
    load_strides.clear();
    assert_compact_wire_rejects(initialization, "target LoadY");
}

#[test]
fn compact_initialization_json_and_bincode_reject_output_register_overwrite() {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map { base_ops, .. } = &mut initialization.residual.nodes[0] else {
        unreachable!()
    };
    base_ops.insert(3, LinearOp::Const { dst: 2, value: 0.0 });
    assert_compact_wire_rejects(initialization, "defined more than once");
}

#[test]
fn compact_initialization_json_and_bincode_reject_target_register_overwrite() {
    let mut initialization = complete_compact_initialization();
    let ComputeNode::Map { base_ops, .. } = &mut initialization.residual.nodes[0] else {
        unreachable!()
    };
    base_ops.insert(1, LinearOp::Const { dst: 0, value: 0.0 });
    assert_compact_wire_rejects(initialization, "defined more than once");
}

#[test]
fn compact_initialization_json_and_bincode_reject_wrong_target_load_and_sign() {
    let mut wrong_load = complete_compact_initialization();
    let ComputeNode::Map { base_ops, .. } = &mut wrong_load.residual.nodes[0] else {
        unreachable!()
    };
    let LinearOp::LoadY { index, .. } = &mut base_ops[0] else {
        unreachable!()
    };
    *index = 1;
    assert_compact_wire_rejects(wrong_load, "target LoadY");

    let mut wrong_map = complete_compact_initialization();
    let ComputeNode::Map { load_strides, .. } = &mut wrong_map.residual.nodes[0] else {
        unreachable!()
    };
    load_strides.clear();
    assert_compact_wire_rejects(wrong_map, "affine map");

    let mut wrong_sign = complete_compact_initialization();
    wrong_sign.direct_families[0].residual_sign = -1;
    assert_compact_wire_rejects(wrong_sign, "residual direction");
}

fn dependent_compact_initialization(cycle: bool) -> InitializationSolveSystem {
    let dense = vec![AffineStencilIndexStrideTerm {
        dimension: 0,
        stride: 1,
    }];
    let (mut first_node, mut first_family) = direct_initialization_family(0, 0, dense.clone());
    let (mut second_node, mut second_family) = direct_initialization_family(1, 3, dense);
    let second_span = Span::from_offsets(
        SourceId::from_source_name("ir_solve_second_family.mo"),
        10,
        20,
    );
    second_family.span = second_span;
    if let ComputeNode::Map { span, base_ops, .. } = &mut second_node {
        *span = second_span;
        base_ops[1] = LinearOp::LoadY { dst: 1, index: 0 };
    }
    if cycle {
        let ComputeNode::Map { base_ops, .. } = &mut first_node else {
            unreachable!()
        };
        base_ops[1] = LinearOp::LoadY { dst: 1, index: 3 };
    }
    first_family.span = fixture_span();
    InitializationSolveSystem {
        residual: ComputeBlock {
            nodes: vec![first_node, second_node],
        },
        direct_families: vec![first_family, second_family],
        required_target_ranges: vec![InitializationTargetRange {
            start: 0,
            end: 6,
            span: fixture_span(),
        }],
        projection_plan: AlgebraicProjectionPlan {
            blocks: vec![
                AlgebraicProjectionBlock {
                    rows: vec![0],
                    y_indices: vec![0],
                    causal_steps: Vec::new(),
                },
                AlgebraicProjectionBlock {
                    rows: vec![1],
                    y_indices: vec![3],
                    causal_steps: Vec::new(),
                },
            ],
        },
        ..Default::default()
    }
}

#[test]
fn compact_initialization_json_and_bincode_reject_dependency_reorder() {
    let mut initialization = dependent_compact_initialization(false);
    initialization.projection_plan.blocks.swap(0, 1);
    let problem = SolveProblem {
        layout: make_layout(&[("x", vec![6])], &[]),
        initialization,
        ..Default::default()
    };
    let json = serde_json::to_string(&problem).expect("serialize reordered compact JSON");
    let error = serde_json::from_str::<SolveProblem>(&json)
        .expect_err("dependency reorder must fail JSON admission");
    assert!(error.to_string().contains("dependency order"), "{error}");
    let bytes = bincode::serialize(&problem).expect("serialize reordered compact bincode");
    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("dependency reorder must fail bincode admission");
    assert!(error.to_string().contains("dependency order"), "{error}");
}

#[test]
fn compact_initialization_cycle_reports_first_blocked_owner_span() {
    let initialization = dependent_compact_initialization(true);
    let error = validate_compact_gpu_initialization(&initialization, 6)
        .expect_err("direct dependency cycle must fail admission");
    assert!(error.to_string().contains("dependency order"), "{error}");
    assert_eq!(
        error.source_span(),
        Some(initialization.direct_families[0].span)
    );
}

#[test]
fn compact_initialization_validation_rejects_partial_required_union() {
    let mut initialization = complete_compact_initialization();
    initialization.required_target_ranges[0].end = 4;
    let error = validate_initialization_direct_families(&initialization, 4, 3)
        .expect_err("hand-built partial target union must fail closed");
    assert!(error.to_string().contains("incomplete"));
}

#[test]
fn compact_initialization_json_rejects_partial_required_union() {
    let problem = SolveProblem {
        layout: make_layout(&[("x", vec![3])], &[]),
        initialization: complete_compact_initialization(),
        ..Default::default()
    };
    let mut value = serde_json::to_value(problem).expect("serialize compact Solve artifact");
    value["layout"]["y_scalars"] = serde_json::json!(4);
    let error = serde_json::from_value::<SolveProblem>(value)
        .expect_err("JSON with a partial target union must fail closed");
    assert!(error.to_string().contains("incomplete"));
}

#[test]
fn compact_initialization_range_span_survives_json_and_bincode() {
    let mut initialization = complete_compact_initialization();
    initialization.required_target_ranges[0].span = fixture_span();
    let problem = SolveProblem {
        layout: make_layout(&[("x", vec![3])], &[]),
        initialization,
        ..Default::default()
    };
    let json = serde_json::to_string(&problem).expect("serialize compact Solve artifact");
    let from_json: SolveProblem =
        serde_json::from_str(&json).expect("deserialize compact Solve JSON");
    assert_eq!(
        from_json.initialization.required_target_ranges[0].span,
        fixture_span()
    );
    let bytes = bincode::serialize(&problem).expect("serialize compact Solve bincode");
    let from_bincode: SolveProblem =
        bincode::deserialize(&bytes).expect("deserialize compact Solve bincode");
    assert_eq!(
        from_bincode.initialization.required_target_ranges[0].span,
        fixture_span()
    );
    assert_eq!(
        from_bincode.initialization.direct_families[0].span,
        fixture_span()
    );
}

#[test]
fn compact_initialization_json_rejects_missing_and_dummy_range_spans() {
    let mut initialization = complete_compact_initialization();
    initialization.required_target_ranges[0].span = fixture_span();
    let problem = SolveProblem {
        layout: make_layout(&[("x", vec![3])], &[]),
        initialization,
        ..Default::default()
    };
    let mut missing = serde_json::to_value(&problem).expect("serialize compact Solve artifact");
    missing["initialization"]["required_target_ranges"][0]
        .as_object_mut()
        .expect("target range object")
        .remove("span");
    let missing_error = serde_json::from_value::<SolveProblem>(missing)
        .expect_err("compact target range span is a mandatory wire field");
    assert!(
        missing_error.to_string().contains("span"),
        "{missing_error}"
    );

    let mut dummy = serde_json::to_value(problem).expect("serialize compact Solve artifact");
    dummy["initialization"]["required_target_ranges"][0]["span"] =
        serde_json::to_value(Span::DUMMY).expect("serialize dummy span");
    let dummy_error = serde_json::from_value::<SolveProblem>(dummy)
        .expect_err("dummy compact target range spans must fail admission");
    assert!(dummy_error.to_string().contains("span"), "{dummy_error}");
}

#[test]
fn compact_initialization_bincode_rejects_missing_and_dummy_range_spans() {
    #[derive(Serialize)]
    struct SourceLessRange {
        start: usize,
        end: usize,
    }
    let source_less = bincode::serialize(&SourceLessRange { start: 0, end: 3 })
        .expect("serialize source-less range");
    let missing_error = bincode::deserialize::<InitializationTargetRange>(&source_less)
        .expect_err("range bincode without a span must fail decoding");
    assert!(!missing_error.to_string().is_empty());

    let mut initialization = complete_compact_initialization();
    initialization.required_target_ranges[0].span = Span::DUMMY;
    let problem = SolveProblem {
        layout: make_layout(&[("x", vec![3])], &[]),
        initialization,
        ..Default::default()
    };
    let bytes = bincode::serialize(&problem).expect("serialize invalid compact Solve bincode");
    let error = bincode::deserialize::<SolveProblem>(&bytes)
        .expect_err("dummy compact target range spans must fail bincode admission");
    assert!(error.to_string().contains("span"), "{error}");
}

#[test]
fn compact_initialization_json_and_bincode_reject_nonunit_residual_signs() {
    for residual_sign in [0, -2] {
        let mut initialization = complete_compact_initialization();
        initialization.required_target_ranges[0].span = fixture_span();
        initialization.direct_families[0].residual_sign = residual_sign;
        let problem = SolveProblem {
            layout: make_layout(&[("x", vec![3])], &[]),
            initialization,
            ..Default::default()
        };
        let json = serde_json::to_string(&problem).expect("serialize invalid compact Solve JSON");
        let json_error = serde_json::from_str::<SolveProblem>(&json)
            .expect_err("non-unit direct residual sign must fail JSON admission");
        assert!(json_error.to_string().contains("sign"), "{json_error}");

        let bytes = bincode::serialize(&problem).expect("serialize invalid compact Solve bincode");
        let bincode_error = bincode::deserialize::<SolveProblem>(&bytes)
            .expect_err("non-unit direct residual sign must fail bincode admission");
        assert!(
            bincode_error.to_string().contains("sign"),
            "{bincode_error}"
        );
    }
}

#[test]
fn invalid_initialization_range_reports_span_after_json_and_bincode() {
    let range = InitializationTargetRange {
        start: 2,
        end: 2,
        span: fixture_span(),
    };
    let json = serde_json::to_string(&range).expect("serialize invalid range JSON");
    let from_json: InitializationTargetRange =
        serde_json::from_str(&json).expect("deserialize invalid range JSON");
    let bytes = bincode::serialize(&range).expect("serialize invalid range bincode");
    let from_bincode: InitializationTargetRange =
        bincode::deserialize(&bytes).expect("deserialize invalid range bincode");

    for decoded in [from_json, from_bincode] {
        let initialization = InitializationSolveSystem {
            required_target_ranges: vec![decoded],
            ..Default::default()
        };
        let error = validate_initialization_direct_families(&initialization, 2, 0)
            .expect_err("empty invalid range must fail closed");
        assert_eq!(error.source_span(), Some(fixture_span()));
    }
}

fn fixture_span() -> Span {
    Span::from_offsets(
        SourceId::from_source_name("ir_solve_tests_source_44.mo"),
        0,
        1,
    )
}

#[test]
fn scalar_program_block_with_source_span_preserves_explicit_fixture_span() {
    let block = ScalarProgramBlock::with_source_span(vec![vec![]], fixture_span());
    assert_eq!(block.program_spans, vec![fixture_span()]);
}

fn make_layout(y_shapes: &[(&str, Vec<usize>)], p_shapes: &[(&str, Vec<usize>)]) -> VarLayout {
    let mut bindings = IndexMap::new();
    let mut shapes = IndexMap::new();
    let mut y_offset = 0usize;
    let mut p_offset = 0usize;
    for (name, shape) in y_shapes {
        let size: usize = shape.iter().product();
        bindings.insert(name.to_string(), scalar_slot_y(y_offset));
        shapes.insert(name.to_string(), shape.clone());
        y_offset += size;
    }
    for (name, shape) in p_shapes {
        let size: usize = shape.iter().product();
        bindings.insert(name.to_string(), scalar_slot_p(p_offset));
        shapes.insert(name.to_string(), shape.clone());
        p_offset += size;
    }
    VarLayout::from_parts_with_shapes(bindings, shapes, y_offset, p_offset)
        .expect("representative Solve fixture layout should satisfy shape contract")
}

fn representative_solve_problem_fixture() -> SolveProblem {
    SolveProblem {
        schema_version: SOLVE_SCHEMA_VERSION,
        layout: make_layout(
            &[("x", vec![1]), ("y", vec![1]), ("hold.y", vec![1])],
            &[("p", vec![1]), ("__pre__.hold.y", vec![1])],
        ),
        solve_layout: representative_solve_layout(),
        continuous: representative_continuous_system(),
        initialization: representative_initialization_system(),
        discrete: representative_discrete_system(),
        events: representative_event_partition(),
        clocks: representative_clock_partition(),
    }
}

fn representative_solver_maps() -> SolverNameIndexMaps {
    let mut name_to_idx = IndexMap::new();
    name_to_idx.insert("x".to_string(), 0);
    name_to_idx.insert("y".to_string(), 1);
    name_to_idx.insert("hold.y".to_string(), 2);

    let mut base_to_indices = IndexMap::new();
    base_to_indices.insert("x".to_string(), vec![0]);
    base_to_indices.insert("y".to_string(), vec![1]);
    base_to_indices.insert("hold.y".to_string(), vec![2]);

    SolverNameIndexMaps {
        names: vec!["x".to_string(), "y".to_string(), "hold.y".to_string()],
        name_to_idx,
        base_to_indices,
    }
}

fn representative_solve_layout() -> SolveLayout {
    SolveLayout {
        solver_maps: representative_solver_maps(),
        state_scalar_count: 1,
        algebraic_scalar_count: 1,
        output_scalar_count: 1,
        parameter_count: 1,
        compiled_parameter_len: 2,
        discrete_real_scalar_names: vec!["hold.y".to_string()],
        relation_memory_parameter_indices: vec![1],
        initial_event_parameter_index: Some(1),
        pre_param_bindings: vec![PreParamBinding {
            dest_p_index: 1,
            source: PreParamSource::Y { index: 2 },
        }],
        ..SolveLayout::default()
    }
}

fn representative_continuous_system() -> ContinuousSolveSystem {
    ContinuousSolveSystem {
        implicit_rhs: ComputeBlock {
            nodes: vec![ComputeNode::ScalarPrograms(
                ScalarProgramBlock::with_source_span(
                    vec![vec![
                        LinearOp::LoadY { dst: 0, index: 0 },
                        LinearOp::LoadP { dst: 1, index: 0 },
                        LinearOp::Binary {
                            dst: 2,
                            op: BinaryOp::Sub,
                            lhs: 0,
                            rhs: 1,
                        },
                        LinearOp::StoreOutput { src: 2 },
                    ]],
                    fixture_span(),
                ),
            )],
        },
        implicit_row_targets: vec![Some(scalar_slot_y(1))],
        algebraic_projection_plan: AlgebraicProjectionPlan {
            blocks: vec![AlgebraicProjectionBlock {
                rows: vec![1],
                y_indices: vec![1],
                causal_steps: Vec::new(),
            }],
        },
        residual: ComputeBlock::from_scalar_program_block(ScalarProgramBlock::with_source_span(
            vec![vec![
                LinearOp::LoadY { dst: 0, index: 1 },
                LinearOp::StoreOutput { src: 0 },
            ]],
            fixture_span(),
        )),
        derivative_rhs: representative_derivative_rhs(),
    }
}

fn representative_derivative_rhs() -> ComputeBlock {
    ComputeBlock {
        nodes: vec![ComputeNode::MatMul {
            lhs_ops: vec![LinearOp::LoadP { dst: 0, index: 0 }],
            lhs_start: 0,
            rhs_ops: vec![LinearOp::LoadY { dst: 1, index: 0 }],
            rhs_start: 1,
            m: 1,
            k: 1,
            n: 1,
            lhs_sparsity: SparsityPattern::Diagonal,
            rhs_sparsity: SparsityPattern::Dense,
            metadata: TensorNodeMetadata::default(),
            span: Span::DUMMY,
        }],
    }
}

fn representative_initialization_system() -> InitializationSolveSystem {
    InitializationSolveSystem {
        row_targets: vec![Some(scalar_slot_y(1))],
        direct_families: Vec::new(),
        required_target_ranges: Vec::new(),
        fixed_target_ranges: Vec::new(),
        residual: ComputeBlock::from_scalar_program_block(ScalarProgramBlock::with_source_span(
            vec![vec![
                LinearOp::Const { dst: 0, value: 0.0 },
                LinearOp::StoreOutput { src: 0 },
            ]],
            fixture_span(),
        )),
        projection_indices: Vec::new(),
        projection_plan: AlgebraicProjectionPlan::default(),
        update_rhs: ScalarProgramBlock::default(),
        update_targets: Vec::new(),
    }
}

fn representative_discrete_system() -> DiscreteSolveSystem {
    DiscreteSolveSystem {
        runtime_assignment_rhs: ScalarProgramBlock::with_source_span(
            vec![vec![
                LinearOp::LoadY { dst: 0, index: 2 },
                LinearOp::StoreOutput { src: 0 },
            ]],
            fixture_span(),
        ),
        runtime_assignment_targets: vec![scalar_slot_p(1)],
        rhs: ScalarProgramBlock::with_source_span(
            vec![vec![
                LinearOp::LoadY { dst: 0, index: 1 },
                LinearOp::Const { dst: 1, value: 0.0 },
                LinearOp::Compare {
                    dst: 2,
                    op: CompareOp::Gt,
                    lhs: 0,
                    rhs: 1,
                },
                LinearOp::StoreOutput { src: 2 },
            ]],
            fixture_span(),
        ),
        update_targets: vec![scalar_slot_y(2)],
        pre_modes: vec![DiscreteEventPreMode::Fixed],
        observation_refresh: vec![true],
    }
}

fn representative_event_partition() -> SolveEventPartition {
    SolveEventPartition {
        root_conditions: ScalarProgramBlock::with_source_span(
            vec![vec![
                LinearOp::LoadTime { dst: 0 },
                LinearOp::LoadP { dst: 1, index: 0 },
                LinearOp::Compare {
                    dst: 2,
                    op: CompareOp::Ge,
                    lhs: 0,
                    rhs: 1,
                },
                LinearOp::StoreOutput { src: 2 },
            ]],
            fixture_span(),
        ),
        scheduled_time_events: vec![0.1],
        ..SolveEventPartition::default()
    }
}

fn representative_clock_partition() -> SolveClockPartition {
    SolveClockPartition {
        periodic_event_schedules: vec![PeriodicEventSchedule {
            period_seconds: 0.1,
            phase_seconds: 0.0,
        }],
    }
}

fn assert_same_json_shape<T: serde::Serialize>(actual: &T, expected: &T) {
    assert_eq!(
        serde_json::to_value(actual).expect("serialize actual"),
        serde_json::to_value(expected).expect("serialize expected")
    );
}

#[test]
fn y_slice_returns_some_for_y_array_variable() {
    let layout = make_layout(&[("x", vec![3, 3])], &[]);
    let src = layout
        .y_slice("x")
        .expect("3×3 Y-slot variable should yield YSlice");
    assert!(matches!(src, TensorSource::YSlice { start: 0, shape } if shape == [3, 3]));
}

#[test]
fn p_slice_returns_some_for_p_array_variable() {
    let layout = make_layout(&[], &[("A", vec![2, 4])]);
    let src = layout
        .p_slice("A")
        .expect("2×4 P-slot variable should yield PSlice");
    assert!(matches!(src, TensorSource::PSlice { start: 0, shape } if shape == [2, 4]));
}

#[test]
fn indexed_bindings_are_derived_from_shape_metadata() {
    let layout = make_layout(&[("body.frame.R.T", vec![3, 3])], &[]);
    let entries = layout
        .indexed_bindings()
        .get(&ComponentReferenceKey::generated("body.frame.R.T"))
        .expect("array layout should expose structured scalar slots");

    assert_eq!(entries.len(), 9);
    assert_eq!(entries[0].indices, vec![1, 1]);
    assert!(matches!(entries[0].slot, ScalarSlot::Y { index: 0, .. }));
    assert_eq!(entries[8].indices, vec![3, 3]);
    assert!(matches!(entries[8].slot, ScalarSlot::Y { index: 8, .. }));
}

#[test]
fn scalar_program_block_rejects_span_count_mismatch_with_span() {
    let span = Span::from_offsets(SourceId::from_source_name("bad_scalar_spans.mo"), 2, 5);

    let err = ScalarProgramBlock::with_program_spans(
        vec![vec![LinearOp::StoreOutput { src: 0 }]],
        vec![span, span],
    )
    .expect_err("explicit scalar row spans must match row count");

    assert!(matches!(
        err,
        SolveProblemShapeContractError::ScalarProgramSpanMismatch {
            programs: 1,
            spans: 2,
            span: actual,
            ..
        } if actual == Some(span)
    ));
}

#[test]
fn scalar_program_block_rejects_output_index_count_mismatch_with_span() {
    let span = Span::from_offsets(SourceId::from_source_name("bad_scalar_outputs.mo"), 7, 11);

    let err = ScalarProgramBlock::with_output_indices(
        vec![vec![LinearOp::StoreOutput { src: 0 }]],
        vec![span],
        vec![0, 1],
    )
    .expect_err("explicit scalar output indices must match row count");

    assert!(matches!(
        err,
        SolveProblemShapeContractError::ScalarProgramOutputIndexMismatch {
            programs: 1,
            output_indices: 2,
            span: actual,
            ..
        } if actual == Some(span)
    ));
}

#[test]
fn scalar_program_block_first_source_span_skips_dummy_rows() {
    let span = Span::from_offsets(SourceId::from_source_name("scalar_source.mo"), 13, 21);
    let block = ScalarProgramBlock::with_program_spans(
        vec![
            vec![LinearOp::StoreOutput { src: 0 }],
            vec![LinearOp::StoreOutput { src: 1 }],
        ],
        vec![Span::DUMMY, span],
    )
    .expect("scalar span fixture metadata should match row count");

    assert_eq!(block.program_span(0), None);
    assert_eq!(block.program_span(1), Some(span));
    assert_eq!(block.first_source_span(), Some(span));
}

#[test]
fn y_slice_returns_none_for_p_slot_variable() {
    let layout = make_layout(&[], &[("p", vec![2])]);
    assert!(
        layout.y_slice("p").is_none(),
        "P-slot variable must not yield YSlice"
    );
}

#[test]
fn p_slice_returns_none_for_y_slot_variable() {
    let layout = make_layout(&[("x", vec![2])], &[]);
    assert!(
        layout.p_slice("x").is_none(),
        "Y-slot variable must not yield PSlice"
    );
}

#[test]
fn y_slice_returns_none_for_scalar_variable_without_shape() {
    let mut bindings = IndexMap::new();
    bindings.insert("s".to_string(), scalar_slot_y(0));
    let layout = VarLayout::from_parts_with_shapes(bindings, IndexMap::new(), 1, 0)
        .expect("scalar variable fixture layout should satisfy shape contract");
    assert!(
        layout.y_slice("s").is_none(),
        "scalar variable with no recorded shape must not yield YSlice"
    );
}

#[test]
fn y_slice_returns_none_for_unknown_variable() {
    let layout = make_layout(&[("x", vec![2])], &[]);
    assert!(layout.y_slice("unknown").is_none());
}

fn serde_roundtrip_tensor_block_fixture() -> ComputeBlock {
    ComputeBlock {
        nodes: vec![
            serde_roundtrip_scalar_node(),
            serde_roundtrip_matmul_node(),
            serde_roundtrip_linsolve_node(),
            serde_roundtrip_map_node(),
            serde_roundtrip_affine_stencil_node(),
        ],
    }
}

fn serde_roundtrip_scalar_node() -> ComputeNode {
    ComputeNode::ScalarPrograms(ScalarProgramBlock::with_source_span(
        vec![vec![
            LinearOp::Const { dst: 0, value: 1.0 },
            LinearOp::StoreOutput { src: 0 },
        ]],
        fixture_span(),
    ))
}

fn serde_roundtrip_matmul_node() -> ComputeNode {
    ComputeNode::MatMul {
        lhs_ops: vec![
            LinearOp::Const { dst: 0, value: 2.0 },
            LinearOp::Move { dst: 1, src: 0 },
        ],
        lhs_start: 1,
        rhs_ops: vec![
            LinearOp::LoadSeed { dst: 2, index: 0 },
            LinearOp::Move { dst: 3, src: 2 },
        ],
        rhs_start: 3,
        m: 1,
        k: 1,
        n: 1,
        lhs_sparsity: SparsityPattern::Diagonal,
        rhs_sparsity: SparsityPattern::Dense,
        metadata: TensorNodeMetadata::default(),
        span: Span::DUMMY,
    }
}

fn serde_roundtrip_linsolve_node() -> ComputeNode {
    ComputeNode::LinSolve {
        setup_ops: vec![
            LinearOp::LoadP { dst: 0, index: 0 },
            LinearOp::LoadP { dst: 1, index: 1 },
            LinearOp::LoadP { dst: 2, index: 2 },
            LinearOp::LoadY { dst: 3, index: 0 },
        ],
        matrix_start: 0,
        rhs_start: 3,
        n: 2,
        next_reg: 4,
        metadata: TensorNodeMetadata::default(),
        span: Span::DUMMY,
    }
}

fn serde_roundtrip_map_node() -> ComputeNode {
    ComputeNode::Map {
        domain: test_tensor_domain(3),
        output_map: TensorOutputMap::dense_contiguous(0, &test_tensor_domain(3))
            .expect("valid dense output map"),
        base_ops: vec![
            LinearOp::Const { dst: 0, value: 1.0 },
            LinearOp::StoreOutput { src: 0 },
        ],
        load_strides: Vec::new(),
        const_strides: vec![AffineStencilConstStride {
            op_position: 0,
            terms: vec![AffineStencilConstStrideTerm {
                dimension: 0,
                stride: 1.0,
            }],
        }],
        metadata: TensorNodeMetadata::default(),
        span: Span::DUMMY,
    }
}

fn serde_roundtrip_affine_stencil_node() -> ComputeNode {
    ComputeNode::AffineStencil {
        domain: test_tensor_domain(8),
        output_map: TensorOutputMap::dense_contiguous(0, &test_tensor_domain(8))
            .expect("valid dense output map"),
        base_ops: vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::StoreOutput { src: 0 },
        ],
        load_strides: vec![AffineStencilLoadStride {
            op_position: 0,
            terms: vec![AffineStencilIndexStrideTerm {
                dimension: 0,
                stride: 1,
            }],
        }],
        const_strides: Vec::new(),
        metadata: TensorNodeMetadata::default(),
        span: Span::DUMMY,
    }
}

fn assert_tensor_node_tags_survive_json(json: &str) {
    for tag in [
        "MatMul",
        "LinSolve",
        "Map",
        "AffineStencil",
        "lhs_sparsity",
        "metadata",
    ] {
        assert!(json.contains(tag), "{tag} must appear in JSON: {json}");
    }
}

fn assert_tensor_nodes_survive_roundtrip(back: &ComputeBlock) {
    assert_eq!(
        back.nodes.len(),
        5,
        "all five compute nodes must survive round-trip"
    );
    assert!(matches!(&back.nodes[0], ComputeNode::ScalarPrograms(_)));
    assert!(matches!(&back.nodes[2], ComputeNode::LinSolve { n: 2, .. }));
    assert!(matches!(&back.nodes[3], ComputeNode::Map { .. }));
    assert!(matches!(
        &back.nodes[4],
        ComputeNode::AffineStencil { domain, .. }
            if domain
                .scalar_count()
                .expect("fixture domain should have a valid scalar count")
                == 8
    ));
    assert_roundtrip_matmul_shape(&back.nodes[1]);
}

fn assert_roundtrip_matmul_shape(node: &ComputeNode) {
    assert!(matches!(
        node,
        ComputeNode::MatMul {
            m: 1,
            k: 1,
            n: 1,
            lhs_sparsity: SparsityPattern::Diagonal,
            metadata: TensorNodeMetadata {
                element_type: TensorElementType::Real64,
                layout: TensorLayout::RowMajorDense,
                scalar_fallback: ScalarFallback::Exact,
            },
            ..
        }
    ));
}

#[test]
fn compute_block_tensor_nodes_survive_serde_roundtrip() {
    let block = serde_roundtrip_tensor_block_fixture();
    let json = serde_json::to_string(&block).expect("serialize ComputeBlock");
    assert_tensor_node_tags_survive_json(&json);

    let back: ComputeBlock = serde_json::from_str(&json).expect("deserialize ComputeBlock");
    assert_tensor_nodes_survive_roundtrip(&back);
}

#[test]
fn solve_problem_json_has_supported_schema_version() {
    let value = serde_json::to_value(SolveProblem::default()).expect("serialize SolveProblem");
    assert_eq!(
        value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64),
        Some(u64::from(SOLVE_SCHEMA_VERSION))
    );

    let mut missing = value.clone();
    missing
        .as_object_mut()
        .expect("SolveProblem JSON should be object")
        .remove("schema_version");
    assert!(
        serde_json::from_value::<SolveProblem>(missing).is_err(),
        "SolveProblem JSON must carry an explicit schema_version"
    );

    let mut previous = value.clone();
    previous["schema_version"] = serde_json::json!(SOLVE_SCHEMA_VERSION - 1);
    let err = serde_json::from_value::<SolveProblem>(previous)
        .expect_err("previous Solve schema version must fail after initialization IR replacement");
    assert!(err.to_string().contains("unsupported Solve schema_version"));

    let mut unsupported = value;
    unsupported["schema_version"] = serde_json::json!(SOLVE_SCHEMA_VERSION + 1);
    let err = serde_json::from_value::<SolveProblem>(unsupported)
        .expect_err("unsupported SolveProblem schema version must fail");
    assert!(err.to_string().contains("unsupported Solve schema_version"));
}

#[test]
fn representative_solve_problem_json_roundtrip_preserves_schema_shape() {
    let problem = representative_solve_problem_fixture();
    let json = serde_json::to_string_pretty(&problem).expect("serialize SolveProblem");
    let decoded: SolveProblem = serde_json::from_str(&json).expect("deserialize SolveProblem");
    assert_same_json_shape(&decoded, &problem);
}

#[test]
fn representative_solve_problem_json_matches_committed_golden() {
    let problem = representative_solve_problem_fixture();
    let actual = serde_json::to_value(&problem).expect("serialize representative SolveProblem");
    let expected: serde_json::Value = serde_json::from_str(REPRESENTATIVE_SOLVE_PROBLEM_GOLDEN)
        .expect("valid SolveProblem golden JSON");

    serde_json::from_value::<SolveProblem>(expected.clone())
        .expect("golden uses supported Solve schema");
    assert_eq!(actual, expected);
}

#[test]
fn representative_solve_problem_bincode_roundtrip_preserves_schema_shape() {
    let problem = representative_solve_problem_fixture();
    let bytes = bincode::serialize(&problem).expect("serialize SolveProblem as bincode");
    let decoded: SolveProblem =
        bincode::deserialize(&bytes).expect("deserialize SolveProblem from bincode");
    assert_same_json_shape(&decoded, &problem);
}

#[test]
fn solve_problem_shape_contract_rejects_bad_schema_version() {
    let mut problem = representative_solve_problem_fixture();
    problem.schema_version = SOLVE_SCHEMA_VERSION + 1;

    assert_eq!(
        problem.validate_shape_contract(),
        Err(SolveProblemShapeContractError::SchemaVersion {
            actual: SOLVE_SCHEMA_VERSION + 1,
            expected: SOLVE_SCHEMA_VERSION,
        })
    );
}

#[test]
fn solve_problem_shape_contract_rejects_zero_tensor_dimension() {
    let mut problem = representative_solve_problem_fixture();
    problem.continuous.derivative_rhs = ComputeBlock {
        nodes: vec![ComputeNode::LinSolve {
            setup_ops: Vec::new(),
            matrix_start: 0,
            rhs_start: 0,
            n: 0,
            next_reg: 0,
            metadata: TensorNodeMetadata::default(),
            span: Span::DUMMY,
        }],
    };

    assert_eq!(
        problem.validate_shape_contract(),
        Err(SolveProblemShapeContractError::ZeroTensorDimension {
            context: "continuous.derivative_rhs".to_string(),
            node_index: 0,
            dimension: "LinSolve",
            span: Span::DUMMY,
        })
    );
}

#[test]
fn solve_problem_shape_contract_rejects_zero_step_tensor_domain() {
    let mut problem = representative_solve_problem_fixture();
    problem.continuous.derivative_rhs = ComputeBlock {
        nodes: vec![ComputeNode::Map {
            domain: StructuredIndexDomain {
                binders: vec![StructuredIndexBinder {
                    id: 0,
                    display_name: "i".to_string(),
                    lower: 1,
                    upper: 3,
                    step: 0,
                }],
            },
            output_map: TensorOutputMap {
                start: 0,
                strides: Vec::new(),
            },
            base_ops: vec![
                LinearOp::Const { dst: 0, value: 1.0 },
                LinearOp::StoreOutput { src: 0 },
            ],
            load_strides: Vec::new(),
            const_strides: Vec::new(),
            metadata: TensorNodeMetadata::default(),
            span: Span::DUMMY,
        }],
    };

    assert_eq!(
        problem.validate_shape_contract(),
        Err(SolveProblemShapeContractError::StructuredIndexDomain {
            context: "continuous.derivative_rhs".to_string(),
            node_index: 0,
            dimension: "Map",
            error: StructuredIndexDomainError::ZeroStep {
                binder_id: 0,
                display_name: "i".to_string(),
            },
            span: Span::DUMMY,
        })
    );
}
