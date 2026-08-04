use super::matrix_product_projection::{
    add_equation, assert_projection_error, binary, builtin, colon_array, colon_vector,
    dae_scaling_with_missing_base, declare_array, declare_dae_array, expression_row_slice,
    flatten_dot_terms, flatten_literal_dot_terms, literal_subscripts, multiply, real, residual_rhs,
};
use super::*;

#[test]
fn test_todae_projects_proven_scalar_scaling_forms() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "gain", &[]);
    for name in ["literal", "declared", "compound", "function", "right"] {
        declare_array(&mut flat, name, &[3]);
    }
    let mut function =
        rumoca_core::Function::new("scalarFunction", crate::test_support::test_span());
    function.add_input(rumoca_core::FunctionParam::new(
        "u",
        "Real",
        crate::test_support::test_span(),
    ));
    function.add_output(rumoca_core::FunctionParam::new(
        "y",
        "Real",
        crate::test_support::test_span(),
    ));
    function.external = Some(rumoca_core::ExternalFunction {
        language: "C".to_string(),
        function_name: Some("scalar_function".to_string()),
        output_name: Some("y".to_string()),
        ..Default::default()
    });
    flat.add_function(function);

    let gain = make_structured_var_ref("gain");
    let cases = [
        ("literal", multiply(real(2.0), colon_vector("x"))),
        ("declared", multiply(gain.clone(), colon_vector("x"))),
        (
            "compound",
            multiply(
                binary(rumoca_core::OpBinary::Add, gain.clone(), real(1.0)),
                colon_vector("x"),
            ),
        ),
        (
            "function",
            multiply(
                Expression::FunctionCall {
                    name: VarName::new("scalarFunction").into(),
                    args: vec![gain.clone()],
                    is_constructor: false,
                    span: crate::test_support::test_span(),
                },
                colon_vector("x"),
            ),
        ),
        ("right", multiply(colon_vector("x"), gain)),
    ];
    for (name, rhs) in cases {
        add_equation(&mut flat, colon_vector(name), rhs, 3);
    }

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("proven scalar factors should remain scalar");

    assert_eq!(dae.continuous.equations.len(), 15);
    for (index, equation) in dae.continuous.equations.iter().enumerate() {
        let lane = i64::try_from(index % 3 + 1).expect("lane fits i64");
        let Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected scaling Mul");
        };
        let case = index / 3;
        let (factor, vector) = if case == 4 { (rhs, lhs) } else { (lhs, rhs) };
        assert_eq!(literal_subscripts(vector), Some(("x", vec![lane])));
        match case {
            0 => assert!(matches!(
                factor.as_ref(),
                Expression::Literal {
                    value: Literal::Real(2.0),
                    ..
                }
            )),
            1 | 4 => assert_eq!(literal_subscripts(factor), Some(("gain", vec![]))),
            2 => assert!(matches!(
                factor.as_ref(),
                Expression::Binary {
                    op: rumoca_core::OpBinary::Add,
                    ..
                }
            )),
            3 => assert!(
                matches!(factor.as_ref(), Expression::FunctionCall { name, .. } if name.as_str() == "scalarFunction")
            ),
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_todae_keeps_same_shape_mulelem_on_the_same_lane() {
    let mut flat = Model::new();
    for name in ["a", "b", "y"] {
        declare_array(&mut flat, name, &[3]);
    }
    add_equation(
        &mut flat,
        colon_vector("y"),
        binary(
            rumoca_core::OpBinary::MulElem,
            colon_vector("a"),
            colon_vector("b"),
        ),
        3,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("same-shape MulElem should lower elementwise");
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary {
            op: rumoca_core::OpBinary::MulElem,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected MulElem, got {:?}", equation.rhs);
        };
        let lane = i64::try_from(lane + 1).expect("lane fits i64");
        assert_eq!(literal_subscripts(lhs), Some(("a", vec![lane])));
        assert_eq!(literal_subscripts(rhs), Some(("b", vec![lane])));
    }
}

#[test]
fn test_todae_lowers_vector_mul_to_dot_only_for_scalar_targets() {
    let mut flat = Model::new();
    declare_array(&mut flat, "a", &[3]);
    declare_array(&mut flat, "b", &[3]);
    declare_array(&mut flat, "colonDot", &[]);
    declare_array(&mut flat, "bareDot", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("colonDot"),
        multiply(colon_vector("a"), colon_vector("b")),
        1,
    );
    add_equation(
        &mut flat,
        make_structured_var_ref("bareDot"),
        multiply(make_structured_var_ref("a"), make_structured_var_ref("b")),
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("scalar-target vector products should become dots");
    for equation in &dae.continuous.equations {
        let mut terms = Vec::new();
        assert!(flatten_dot_terms(residual_rhs(equation), &mut terms));
        assert_eq!(terms.len(), 3);
    }
}

#[test]
fn test_todae_projects_nested_matrix_vector_scaling_in_both_orders() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 2]);
    declare_array(&mut flat, "x", &[2]);
    declare_array(&mut flat, "left", &[2]);
    declare_array(&mut flat, "right", &[2]);
    let product = || multiply(make_structured_var_ref("A"), colon_vector("x"));
    add_equation(
        &mut flat,
        colon_vector("left"),
        multiply(real(2.0), product()),
        2,
    );
    add_equation(
        &mut flat,
        colon_vector("right"),
        multiply(product(), real(2.0)),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("nested scaling should project only the array side");
    for (index, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary { lhs, rhs, .. } = residual_rhs(equation) else {
            panic!("expected outer multiplication");
        };
        let (literal, product) = if index < 2 { (lhs, rhs) } else { (rhs, lhs) };
        assert!(matches!(
            literal.as_ref(),
            Expression::Literal {
                value: Literal::Real(2.0),
                ..
            }
        ));
        let mut terms = Vec::new();
        assert!(flatten_dot_terms(product, &mut terms), "got {product:?}");
        let row = i64::try_from(index % 2 + 1).expect("row fits i64");
        assert_eq!(
            terms,
            (1_i64..=2)
                .map(|inner| (("A", vec![row, inner]), ("x", vec![inner])))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_rejects_inner_and_target_shape_mismatches() {
    let mut inner = Model::new();
    declare_array(&mut inner, "A", &[2, 3]);
    declare_array(&mut inner, "x", &[2]);
    declare_array(&mut inner, "y", &[2]);
    add_equation(
        &mut inner,
        colon_vector("y"),
        multiply(make_structured_var_ref("A"), colon_vector("x")),
        2,
    );
    assert_projection_error(&inner, "inner dimension mismatch");

    let mut target = Model::new();
    declare_array(&mut target, "A", &[3, 3]);
    declare_array(&mut target, "x", &[3]);
    declare_array(&mut target, "y", &[2]);
    add_equation(
        &mut target,
        colon_vector("y"),
        multiply(make_structured_var_ref("A"), colon_vector("x")),
        2,
    );
    assert_projection_error(&target, "result shape mismatch");
}

#[test]
fn test_todae_rejects_matrix_result_in_scalar_context_and_rank_three() {
    let mut scalar = Model::new();
    declare_array(&mut scalar, "A", &[2, 2]);
    declare_array(&mut scalar, "x", &[2]);
    declare_array(&mut scalar, "s", &[]);
    add_equation(
        &mut scalar,
        make_structured_var_ref("s"),
        multiply(make_structured_var_ref("A"), make_structured_var_ref("x")),
        1,
    );
    assert_projection_error(&scalar, "non-scalar result in scalar context");

    let mut rank_three = Model::new();
    declare_array(&mut rank_three, "T", &[2, 2, 2]);
    declare_array(&mut rank_three, "x", &[2]);
    declare_array(&mut rank_three, "y", &[2]);
    add_equation(
        &mut rank_three,
        colon_vector("y"),
        multiply(make_structured_var_ref("T"), colon_vector("x")),
        2,
    );
    assert_projection_error(&rank_three, "unsupported rank");
}

#[test]
fn test_todae_rejects_mulelem_shape_mismatch() {
    let mut flat = Model::new();
    declare_array(&mut flat, "a", &[3]);
    declare_array(&mut flat, "b", &[2]);
    declare_array(&mut flat, "y", &[3]);
    add_equation(
        &mut flat,
        colon_vector("y"),
        binary(
            rumoca_core::OpBinary::MulElem,
            colon_vector("a"),
            colon_vector("b"),
        ),
        3,
    );
    assert_projection_error(&flat, "elementwise shape mismatch");
}

#[test]
fn test_todae_rejects_dynamic_range_and_unknown_product_operands() {
    let range = Expression::Range {
        start: Box::new(Expression::Literal {
            value: Literal::Integer(1),
            span: crate::test_support::test_span(),
        }),
        step: None,
        end: Box::new(make_structured_var_ref("i")),
        span: crate::test_support::test_span(),
    };
    let operands = [
        expression_row_slice("A", make_structured_var_ref("i")),
        expression_row_slice("A", range),
        Expression::VarRef {
            name: VarName::new("unknown").into(),
            subscripts: vec![rumoca_core::Subscript::Index {
                value: 1,
                span: crate::test_support::test_span(),
            }],
            span: crate::test_support::test_span(),
        },
    ];
    for operand in operands {
        let mut flat = Model::new();
        declare_array(&mut flat, "A", &[2, 3]);
        declare_array(&mut flat, "x", &[3]);
        declare_array(&mut flat, "z", &[]);
        declare_array(&mut flat, "i", &[]);
        add_equation(
            &mut flat,
            make_structured_var_ref("z"),
            multiply(operand, colon_vector("x")),
            1,
        );
        assert_projection_error(&flat, "unknown operand shape");
    }
}

#[test]
fn test_todae_rejects_scalar_dot_with_two_dynamic_row_slices() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[2, 3]);
    declare_array(&mut flat, "i", &[]);
    declare_array(&mut flat, "j", &[]);
    declare_array(&mut flat, "z", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("z"),
        multiply(
            expression_row_slice("A", make_structured_var_ref("i")),
            expression_row_slice("B", make_structured_var_ref("j")),
        ),
        1,
    );
    assert_projection_error(&flat, "unknown operand shape");
}

#[test]
fn test_todae_rejects_scalar_dot_with_unary_wrapped_dynamic_row_slices() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[2, 3]);
    declare_array(&mut flat, "i", &[]);
    declare_array(&mut flat, "j", &[]);
    declare_array(&mut flat, "z", &[]);
    let negate = |expr| Expression::Unary {
        op: rumoca_core::OpUnary::Minus,
        rhs: Box::new(expr),
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        make_structured_var_ref("z"),
        multiply(
            negate(expression_row_slice("A", make_structured_var_ref("i"))),
            negate(expression_row_slice("B", make_structured_var_ref("j"))),
        ),
        1,
    );
    assert_projection_error(&flat, "unknown operand shape");
}

#[test]
fn test_todae_rejects_scalar_dot_with_scaled_dynamic_row_slices() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[2, 3]);
    declare_array(&mut flat, "i", &[]);
    declare_array(&mut flat, "j", &[]);
    declare_array(&mut flat, "z", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("z"),
        multiply(
            multiply(
                real(2.0),
                expression_row_slice("A", make_structured_var_ref("i")),
            ),
            multiply(
                real(3.0),
                expression_row_slice("B", make_structured_var_ref("j")),
            ),
        ),
        1,
    );
    assert_projection_error(&flat, "unknown operand shape");
}

#[test]
fn test_todae_preserves_dynamic_row_slice_reductions_in_scalar_products() {
    for reduction in [
        rumoca_core::BuiltinFunction::Sum,
        rumoca_core::BuiltinFunction::Product,
        rumoca_core::BuiltinFunction::Min,
        rumoca_core::BuiltinFunction::Max,
    ] {
        let mut flat = Model::new();
        declare_array(&mut flat, "A", &[2, 3]);
        declare_array(&mut flat, "i", &[]);
        declare_array(&mut flat, "z", &[]);
        add_equation(
            &mut flat,
            make_structured_var_ref("z"),
            multiply(
                real(2.0),
                builtin(
                    reduction,
                    vec![expression_row_slice("A", make_structured_var_ref("i"))],
                ),
            ),
            1,
        );
        to_dae_with_options(
            &flat,
            ToDaeOptions {
                error_on_unbalanced: false,
            },
        )
        .expect("scalar reductions must stay outside matrix-product projection");
    }
}

#[test]
fn test_todae_preserves_scalar_scaled_sum_of_vector_slices() {
    let mut flat = Model::new();
    declare_array(&mut flat, "diameters", &[1]);
    declare_array(&mut flat, "dimensions", &[2]);
    let range_slice = |start, end| Expression::VarRef {
        name: VarName::new("dimensions").into(),
        subscripts: vec![rumoca_core::Subscript::Expr {
            expr: Box::new(Expression::Range {
                start: Box::new(Expression::Literal {
                    value: Literal::Integer(start),
                    span: crate::test_support::test_span(),
                }),
                step: None,
                end: Box::new(Expression::Literal {
                    value: Literal::Integer(end),
                    span: crate::test_support::test_span(),
                }),
                span: crate::test_support::test_span(),
            }),
            span: crate::test_support::test_span(),
        }],
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        make_structured_var_ref("diameters"),
        multiply(
            real(0.5),
            binary(
                rumoca_core::OpBinary::Add,
                range_slice(1, 1),
                range_slice(2, 2),
            ),
        ),
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("scalar scaling of a vector sum is not matrix-product projection");

    assert_eq!(dae.continuous.equations.len(), 1);
    let Expression::Binary {
        op: rumoca_core::OpBinary::Mul,
        rhs,
        ..
    } = residual_rhs(&dae.continuous.equations[0])
    else {
        panic!("expected preserved scalar scaling");
    };
    assert!(matches!(
        rhs.as_ref(),
        Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            ..
        }
    ));
}

#[test]
fn test_todae_projects_literal_range_slice_dot_product() {
    let mut flat = Model::new();
    declare_array(&mut flat, "a", &[2]);
    declare_array(&mut flat, "x", &[1]);
    declare_array(&mut flat, "y", &[]);
    let last_a = Expression::VarRef {
        name: VarName::new("a").into(),
        subscripts: vec![rumoca_core::Subscript::Expr {
            expr: Box::new(Expression::Range {
                start: Box::new(Expression::Literal {
                    value: Literal::Integer(2),
                    span: crate::test_support::test_span(),
                }),
                step: None,
                end: Box::new(Expression::Literal {
                    value: Literal::Integer(2),
                    span: crate::test_support::test_span(),
                }),
                span: crate::test_support::test_span(),
            }),
            span: crate::test_support::test_span(),
        }],
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        make_structured_var_ref("y"),
        multiply(last_a, make_structured_var_ref("x")),
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("a compile-time literal range has a provable dot-product shape");
    let Expression::Binary {
        op: rumoca_core::OpBinary::Mul,
        lhs,
        rhs,
        ..
    } = residual_rhs(&dae.continuous.equations[0])
    else {
        panic!("expected one projected dot term");
    };
    assert_eq!(literal_subscripts(lhs), Some(("a", vec![2])));
    assert_eq!(literal_subscripts(rhs), Some(("x", vec![1])));
}

#[test]
fn test_scalarizer_preserves_unknown_sum_with_scalar_only_descendant_product() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "gain", &[]);
    declare_dae_array(&mut dae, "u", &[2]);
    declare_dae_array(&mut dae, "y", &[2]);
    let dynamic_range = Expression::Range {
        start: Box::new(make_structured_var_ref("i")),
        step: None,
        end: Box::new(make_structured_var_ref("j")),
        span: crate::test_support::test_span(),
    };
    let dynamic_slice = Expression::VarRef {
        name: VarName::new("u").into(),
        subscripts: vec![rumoca_core::Subscript::Expr {
            expr: Box::new(dynamic_range),
            span: crate::test_support::test_span(),
        }],
        span: crate::test_support::test_span(),
    };
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                colon_vector("y"),
                multiply(
                    real(2.0),
                    binary(
                        rumoca_core::OpBinary::Add,
                        multiply(
                            make_structured_var_ref("gain"),
                            make_structured_var_ref("unknown_scalar"),
                        ),
                        dynamic_slice,
                    ),
                ),
            ),
            crate::test_support::test_span(),
            "scalar-only descendant product",
            2,
        ));

    scalarize_phantom_vector_equations(&mut dae)
        .expect("a scalar-only descendant multiplication is not a matrix-product candidate");
    assert_eq!(dae.continuous.equations.len(), 2);
}

#[test]
fn test_scalarizer_rejects_scalar_scaled_unknown_sum_containing_matrix_product() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "A", &[2, 2]);
    declare_dae_array(&mut dae, "x", &[2]);
    declare_dae_array(&mut dae, "u", &[2]);
    declare_dae_array(&mut dae, "y", &[2]);
    let dynamic_range = Expression::Range {
        start: Box::new(make_structured_var_ref("i")),
        step: None,
        end: Box::new(make_structured_var_ref("j")),
        span: crate::test_support::test_span(),
    };
    let dynamic_slice = Expression::VarRef {
        name: VarName::new("u").into(),
        subscripts: vec![rumoca_core::Subscript::Expr {
            expr: Box::new(dynamic_range),
            span: crate::test_support::test_span(),
        }],
        span: crate::test_support::test_span(),
    };
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                colon_vector("y"),
                multiply(
                    real(2.0),
                    binary(
                        rumoca_core::OpBinary::Add,
                        multiply(colon_array("A", 2), colon_vector("x")),
                        dynamic_slice,
                    ),
                ),
            ),
            crate::test_support::test_span(),
            "scalar-scaled unknown sum containing matrix product",
            2,
        ));

    let error = scalarize_phantom_vector_equations(&mut dae)
        .expect_err("unknown sum containing a matrix product must fail closed");
    assert!(error.to_string().contains("unknown operand shape"));
    assert_eq!(error.source_span(), Some(crate::test_support::test_span()));
}

#[test]
fn test_todae_projects_fill_vector_dot_inside_vector_scaling() {
    let mut flat = Model::new();
    declare_array(&mut flat, "velocity", &[2]);
    declare_array(&mut flat, "work", &[2]);
    let fill = || {
        builtin(
            rumoca_core::BuiltinFunction::Fill,
            vec![
                real(0.5),
                Expression::Literal {
                    value: Literal::Integer(2),
                    span: crate::test_support::test_span(),
                },
            ],
        )
    };
    add_equation(
        &mut flat,
        colon_vector("work"),
        multiply(fill(), multiply(colon_vector("velocity"), fill())),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("literal fill dimensions must prove the nested vector dot shape");

    assert_eq!(dae.continuous.equations.len(), 2);
    for equation in &dae.continuous.equations {
        let Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!(
                "expected scaled dot product, got {:?}",
                residual_rhs(equation)
            );
        };
        assert!(matches!(
            lhs.as_ref(),
            Expression::Literal {
                value: Literal::Real(0.5),
                ..
            }
        ));
        let mut terms = Vec::new();
        assert!(
            flatten_literal_dot_terms(rhs, &mut terms),
            "expected a complete fill-vector dot product, got {rhs:?}"
        );
        assert_eq!(
            terms,
            vec![(("velocity", vec![1]), 0.5), (("velocity", vec![2]), 0.5)]
        );
    }
}

#[test]
fn test_todae_rejects_dynamic_row_slices_hidden_by_division_or_unknown_factor() {
    for op in [rumoca_core::OpBinary::Div, rumoca_core::OpBinary::DivElem] {
        let mut flat = Model::new();
        declare_array(&mut flat, "A", &[2, 3]);
        declare_array(&mut flat, "B", &[2, 3]);
        declare_array(&mut flat, "i", &[]);
        declare_array(&mut flat, "j", &[]);
        declare_array(&mut flat, "z", &[]);
        add_equation(
            &mut flat,
            make_structured_var_ref("z"),
            multiply(
                binary(
                    op.clone(),
                    expression_row_slice("A", make_structured_var_ref("i")),
                    real(2.0),
                ),
                binary(
                    op,
                    expression_row_slice("B", make_structured_var_ref("j")),
                    real(3.0),
                ),
            ),
            1,
        );
        assert_projection_error(&flat, "unknown operand shape");
    }

    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "i", &[]);
    declare_array(&mut flat, "z", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("z"),
        multiply(
            multiply(
                make_structured_var_ref("unknown"),
                expression_row_slice("A", make_structured_var_ref("i")),
            ),
            real(2.0),
        ),
        1,
    );
    assert_projection_error(&flat, "unknown operand shape");
}

#[test]
fn test_todae_rejects_vectorized_builtins_hiding_dynamic_row_slices() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[2, 3]);
    declare_array(&mut flat, "i", &[]);
    declare_array(&mut flat, "j", &[]);
    declare_array(&mut flat, "z", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("z"),
        multiply(
            builtin(
                rumoca_core::BuiltinFunction::Sin,
                vec![expression_row_slice("A", make_structured_var_ref("i"))],
            ),
            builtin(
                rumoca_core::BuiltinFunction::Sin,
                vec![expression_row_slice("B", make_structured_var_ref("j"))],
            ),
        ),
        1,
    );
    assert_projection_error(&flat, "unknown operand shape");
}

#[test]
fn test_todae_projects_array_valued_function_product_operand() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "z", &[]);
    let mut function =
        rumoca_core::Function::new("arrayFunction", crate::test_support::test_span());
    function.add_output(
        rumoca_core::FunctionParam::new("y", "Real", crate::test_support::test_span())
            .with_dims(vec![3]),
    );
    function.external = Some(rumoca_core::ExternalFunction {
        language: "C".to_string(),
        function_name: Some("array_function".to_string()),
        output_name: Some("y".to_string()),
        ..Default::default()
    });
    flat.add_function(function);
    add_equation(
        &mut flat,
        make_structured_var_ref("z"),
        multiply(
            Expression::FunctionCall {
                name: VarName::new("arrayFunction").into(),
                args: Vec::new(),
                is_constructor: false,
                span: crate::test_support::test_span(),
            },
            colon_vector("x"),
        ),
        1,
    );
    to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("an array-valued function with a proven output shape is projectable");
}

#[test]
fn test_todae_rejects_bare_unknown_scaling_in_array_projection() {
    let mut dae = dae_scaling_with_missing_base(&[]);
    let error = scalarize_phantom_vector_equations(&mut dae)
        .expect_err("missing declared scalar must fail during array projection");
    assert!(
        error.to_string().contains("unknown operand shape"),
        "{error}"
    );
    assert_eq!(error.source_span(), Some(crate::test_support::test_span()));
}

#[test]
fn test_scalarizer_rejects_phantom_base_as_scaling_scalar() {
    let mut dae = dae_scaling_with_missing_base(&["gain[1]", "gain[2]"]);
    let error = scalarize_phantom_vector_equations(&mut dae)
        .expect_err("phantom base must not be guessed scalar during array projection");
    assert!(
        error.to_string().contains("unknown operand shape"),
        "{error}"
    );
    assert_eq!(error.source_span(), Some(crate::test_support::test_span()));
}

#[test]
fn test_todae_projects_zero_inner_matrix_product_to_zero() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 0]);
    declare_array(&mut flat, "B", &[0, 2]);
    declare_array(&mut flat, "C", &[2, 2]);
    add_equation(
        &mut flat,
        colon_array("C", 2),
        multiply(colon_array("A", 2), colon_array("B", 2)),
        4,
    );
    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("zero inner dimensions have the empty-sum value zero");
    assert_eq!(dae.continuous.equations.len(), 4);
    for equation in &dae.continuous.equations {
        assert!(matches!(
            residual_rhs(equation),
            Expression::Literal {
                value: Literal::Real(0.0),
                ..
            }
        ));
    }
}

#[test]
fn test_todae_rejects_negative_matrix_dimensions() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, -1]);
    declare_array(&mut flat, "x", &[-1]);
    declare_array(&mut flat, "y", &[2]);
    add_equation(
        &mut flat,
        colon_vector("y"),
        multiply(make_structured_var_ref("A"), colon_vector("x")),
        2,
    );
    assert_projection_error(&flat, "NegativeDimension");
}
