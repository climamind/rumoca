use super::matrix_product_projection::{
    add_equation, assert_projection_error, binary, builtin, colon_array, colon_vector,
    declare_array, literal_subscripts, multiply, residual_rhs,
};
use super::*;

fn wrapped_matrix_vector_model(rhs: Expression) -> Model {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[2, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "y", &[2]);
    declare_array(&mut flat, "c", &[]);
    declare_array(&mut flat, "s", &[]);
    add_equation(&mut flat, colon_vector("y"), rhs, 2);
    flat
}

fn matrix_vector_product(name: &str) -> Expression {
    multiply(colon_array(name, 2), colon_vector("x"))
}

fn matrix_row_vector_product(name: &str, row: i64) -> Expression {
    multiply(
        Expression::Index {
            base: Box::new(make_structured_var_ref(name)),
            subscripts: vec![
                rumoca_core::Subscript::Index {
                    value: row,
                    span: crate::test_support::test_span(),
                },
                rumoca_core::Subscript::Colon {
                    span: crate::test_support::test_span(),
                },
            ],
            span: crate::test_support::test_span(),
        },
        colon_vector("x"),
    )
}

fn wrapped_elementwise_vector_model(rhs: Expression) -> Model {
    let mut flat = Model::new();
    declare_array(&mut flat, "a", &[2]);
    declare_array(&mut flat, "b", &[2]);
    declare_array(&mut flat, "y", &[2]);
    add_equation(&mut flat, colon_vector("y"), rhs, 2);
    flat
}

fn elementwise_vector_product() -> Expression {
    binary(
        rumoca_core::OpBinary::MulElem,
        colon_vector("a"),
        colon_vector("b"),
    )
}

fn assert_elementwise_lane_product(expr: &Expression, lane: i64) {
    let Expression::Binary {
        op: rumoca_core::OpBinary::MulElem,
        lhs,
        rhs,
        ..
    } = expr
    else {
        panic!("expected lane-projected MulElem, got {expr:?}");
    };
    assert_eq!(literal_subscripts(lhs), Some(("a", vec![lane])));
    assert_eq!(literal_subscripts(rhs), Some(("b", vec![lane])));
}

fn matrix_vector_terms(expr: &Expression, terms: &mut Vec<(Vec<i64>, Vec<i64>)>) -> bool {
    match expr {
        Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } => matrix_vector_terms(lhs, terms) && matrix_vector_terms(rhs, terms),
        Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } => {
            let (Some(("A", lhs)), Some(("x", rhs))) =
                (literal_subscripts(lhs), literal_subscripts(rhs))
            else {
                return false;
            };
            terms.push((lhs, rhs));
            true
        }
        _ => false,
    }
}

fn assert_complete_matrix_vector_lane(expr: &Expression, row: i64) {
    let mut terms = Vec::new();
    assert!(
        matrix_vector_terms(expr, &mut terms),
        "expected complete matrix-vector dot sum, got {expr:?}"
    );
    assert_eq!(
        terms,
        (1_i64..=3)
            .map(|inner| (vec![row, inner], vec![inner]))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_todae_preserves_plain_scalar_wrapper_around_non_matrix_expression() {
    for op in [
        rumoca_core::OpBinary::Div,
        rumoca_core::OpBinary::DivElem,
        rumoca_core::OpBinary::MulElem,
    ] {
        let mut flat = Model::new();
        declare_array(&mut flat, "x", &[]);
        declare_array(&mut flat, "s", &[]);
        declare_array(&mut flat, "y", &[]);
        add_equation(
            &mut flat,
            make_structured_var_ref("y"),
            binary(
                op.clone(),
                builtin(
                    rumoca_core::BuiltinFunction::Sin,
                    vec![make_structured_var_ref("x")],
                ),
                make_structured_var_ref("s"),
            ),
            1,
        );

        let dae = to_dae_with_options(
            &flat,
            ToDaeOptions {
                error_on_unbalanced: false,
            },
        )
        .expect("ordinary scalar wrappers must bypass matrix-product projection");
        let [equation] = dae.continuous.equations.as_slice() else {
            panic!(
                "expected one scalar equation, got {:?}",
                dae.continuous.equations
            );
        };
        let Expression::Binary {
            op: actual,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected preserved scalar wrapper, got {:?}", equation.rhs);
        };
        assert_eq!(actual, &op);
        assert!(matches!(
            lhs.as_ref(),
            Expression::BuiltinCall {
                function: rumoca_core::BuiltinFunction::Sin,
                args,
                ..
            } if literal_subscripts(&args[0]) == Some(("x", vec![]))
        ));
        assert_eq!(literal_subscripts(rhs), Some(("s", vec![])));
    }
}

#[test]
fn test_todae_rejects_plain_elementwise_result_shape_mismatch() {
    let mut flat = Model::new();
    declare_array(&mut flat, "a", &[3]);
    declare_array(&mut flat, "b", &[3]);
    declare_array(&mut flat, "y", &[2]);
    add_equation(
        &mut flat,
        colon_vector("y"),
        binary(
            rumoca_core::OpBinary::MulElem,
            colon_vector("a"),
            colon_vector("b"),
        ),
        2,
    );

    assert_projection_error(&flat, "result shape mismatch");
}

#[test]
fn test_todae_rejects_plain_elementwise_vector_in_scalar_context() {
    let mut flat = Model::new();
    declare_array(&mut flat, "a", &[3]);
    declare_array(&mut flat, "b", &[3]);
    declare_array(&mut flat, "y", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("y"),
        binary(
            rumoca_core::OpBinary::MulElem,
            colon_vector("a"),
            colon_vector("b"),
        ),
        1,
    );

    assert_projection_error(&flat, "result shape mismatch");
}

#[test]
fn test_todae_projects_div_wrapper_around_matrix_product() {
    let flat = wrapped_matrix_vector_model(binary(
        rumoca_core::OpBinary::Div,
        matrix_vector_product("A"),
        make_structured_var_ref("s"),
    ));

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("array/scalar division around a matrix product must remain projectable");
    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary {
            op: rumoca_core::OpBinary::Div,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected projected Div wrapper, got {:?}", equation.rhs);
        };
        assert_complete_matrix_vector_lane(lhs, i64::try_from(lane + 1).expect("lane fits i64"));
        assert_eq!(literal_subscripts(rhs), Some(("s", vec![])));
    }
}

#[test]
fn test_todae_projects_divelem_wrapper_around_matrix_product() {
    let flat = wrapped_matrix_vector_model(binary(
        rumoca_core::OpBinary::DivElem,
        make_structured_var_ref("s"),
        matrix_vector_product("A"),
    ));

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("scalar/array elementwise division around a matrix product must remain projectable");
    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary {
            op: rumoca_core::OpBinary::DivElem,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected projected DivElem wrapper, got {:?}", equation.rhs);
        };
        assert_eq!(literal_subscripts(lhs), Some(("s", vec![])));
        assert_complete_matrix_vector_lane(rhs, i64::try_from(lane + 1).expect("lane fits i64"));
    }
}

#[test]
fn test_todae_projects_scalar_mulelem_wrapper_around_matrix_product_in_both_orders() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "s", &[]);
    declare_array(&mut flat, "left", &[2]);
    declare_array(&mut flat, "right", &[2]);
    add_equation(
        &mut flat,
        colon_vector("left"),
        binary(
            rumoca_core::OpBinary::MulElem,
            make_structured_var_ref("s"),
            matrix_vector_product("A"),
        ),
        2,
    );
    add_equation(
        &mut flat,
        colon_vector("right"),
        binary(
            rumoca_core::OpBinary::MulElem,
            matrix_vector_product("A"),
            make_structured_var_ref("s"),
        ),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("MulElem must broadcast a scalar on either side of a matrix product");
    assert_eq!(dae.continuous.equations.len(), 4);
    for (index, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary {
            op: rumoca_core::OpBinary::MulElem,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected projected MulElem wrapper, got {:?}", equation.rhs);
        };
        let (scalar, product) = if index < 2 { (lhs, rhs) } else { (rhs, lhs) };
        assert_eq!(literal_subscripts(scalar), Some(("s", vec![])));
        assert_complete_matrix_vector_lane(
            product,
            i64::try_from(index % 2 + 1).expect("lane fits i64"),
        );
    }
}

#[test]
fn test_todae_rejects_mismatched_array_mulelem_wrapper_around_matrix_product() {
    let flat = wrapped_matrix_vector_model(binary(
        rumoca_core::OpBinary::MulElem,
        matrix_vector_product("A"),
        colon_vector("x"),
    ));

    assert_projection_error(&flat, "elementwise shape mismatch");
}

#[test]
fn test_todae_rejects_unary_wrapper_around_matrix_product() {
    let flat = wrapped_matrix_vector_model(Expression::Unary {
        op: rumoca_core::OpUnary::Minus,
        rhs: Box::new(matrix_vector_product("A")),
        span: crate::test_support::test_span(),
    });

    assert_projection_error(&flat, "unsupported matrix-product wrapper");
}

#[test]
fn test_todae_rejects_nonscalar_builtin_wrapper_around_matrix_product() {
    let flat = wrapped_matrix_vector_model(builtin(
        rumoca_core::BuiltinFunction::Sin,
        vec![matrix_vector_product("A")],
    ));

    assert_projection_error(&flat, "unsupported matrix-product wrapper");
}

#[test]
fn test_todae_projects_homotopy_wrapper_around_matrix_products() {
    let flat = wrapped_matrix_vector_model(builtin(
        rumoca_core::BuiltinFunction::Homotopy,
        vec![matrix_vector_product("A"), matrix_vector_product("A")],
    ));

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("homotopy preserves its operand shape and must remain lane-projectable");
    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Homotopy,
            args,
            ..
        } = residual_rhs(equation)
        else {
            panic!(
                "expected projected homotopy wrapper, got {:?}",
                equation.rhs
            );
        };
        let [actual, simplified] = args.as_slice() else {
            panic!("expected homotopy actual and simplified operands, got {args:?}");
        };
        let row = i64::try_from(lane + 1).expect("lane fits i64");
        assert_complete_matrix_vector_lane(actual, row);
        assert_complete_matrix_vector_lane(simplified, row);
    }
}

#[test]
fn test_todae_projects_homotopy_with_scalar_simplified_operand() {
    let flat = wrapped_matrix_vector_model(builtin(
        rumoca_core::BuiltinFunction::Homotopy,
        vec![
            matrix_vector_product("A"),
            multiply(make_structured_var_ref("s"), make_structured_var_ref("s")),
        ],
    ));

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("homotopy must scalar-expand a simplified operand across the actual result shape");
    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Homotopy,
            args,
            ..
        } = residual_rhs(equation)
        else {
            panic!(
                "expected projected homotopy wrapper, got {:?}",
                equation.rhs
            );
        };
        let [actual, simplified] = args.as_slice() else {
            panic!("expected homotopy actual and simplified operands, got {args:?}");
        };
        assert_complete_matrix_vector_lane(actual, i64::try_from(lane + 1).expect("lane fits i64"));
        let Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } = simplified
        else {
            panic!("expected preserved scalar product, got {simplified:?}");
        };
        assert_eq!(literal_subscripts(lhs), Some(("s", vec![])));
        assert_eq!(literal_subscripts(rhs), Some(("s", vec![])));
    }
}

#[test]
fn test_todae_broadcasts_scalar_dot_in_each_homotopy_operand_position() {
    for scalar_actual in [true, false] {
        let scalar_dot = matrix_row_vector_product("A", 1);
        let vector_product = matrix_vector_product("A");
        let args = if scalar_actual {
            vec![scalar_dot, vector_product]
        } else {
            vec![vector_product, scalar_dot]
        };
        let flat =
            wrapped_matrix_vector_model(builtin(rumoca_core::BuiltinFunction::Homotopy, args));

        let dae = to_dae_with_options(
            &flat,
            ToDaeOptions {
                error_on_unbalanced: false,
            },
        )
        .expect("a scalar dot must broadcast from either homotopy operand");
        assert_eq!(dae.continuous.equations.len(), 2);
        for (lane, equation) in dae.continuous.equations.iter().enumerate() {
            let Expression::BuiltinCall { args, .. } = residual_rhs(equation) else {
                panic!("expected projected homotopy");
            };
            let [actual, simplified] = args.as_slice() else {
                panic!("expected two homotopy arguments");
            };
            let (scalar, vector) = if scalar_actual {
                (actual, simplified)
            } else {
                (simplified, actual)
            };
            assert_complete_matrix_vector_lane(scalar, 1);
            assert_complete_matrix_vector_lane(
                vector,
                i64::try_from(lane + 1).expect("lane fits i64"),
            );
        }
    }
}

#[test]
fn test_todae_accepts_if_wrapped_scalar_scaled_dot_product() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "c", &[]);
    declare_array(&mut flat, "s", &[]);
    declare_array(&mut flat, "y", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("y"),
        Expression::If {
            branches: vec![(
                make_structured_var_ref("c"),
                Expression::Unary {
                    op: rumoca_core::OpUnary::Minus,
                    rhs: Box::new(multiply(
                        make_structured_var_ref("s"),
                        matrix_row_vector_product("A", 1),
                    )),
                    span: crate::test_support::test_span(),
                },
            )],
            else_branch: Box::new(Expression::Literal {
                value: rumoca_core::Literal::Integer(0),
                span: crate::test_support::test_span(),
            }),
            span: crate::test_support::test_span(),
        },
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("a scalar if wrapper around a scalar-scaled dot product must lower");
    let [equation] = dae.continuous.equations.as_slice() else {
        panic!(
            "expected one scalar equation, got {:?}",
            dae.continuous.equations
        );
    };
    let Expression::If { branches, .. } = residual_rhs(equation) else {
        panic!(
            "expected preserved scalar if wrapper, got {:?}",
            equation.rhs
        );
    };
    let [(_, Expression::Unary { rhs, .. })] = branches.as_slice() else {
        panic!("expected scalar-scaled dot-product branch, got {branches:?}");
    };
    let Expression::Binary {
        op: rumoca_core::OpBinary::Mul,
        lhs,
        rhs,
        ..
    } = rhs.as_ref()
    else {
        panic!("expected scalar scale around dot product, got {rhs:?}");
    };
    assert_eq!(literal_subscripts(lhs), Some(("s", vec![])));
    assert_complete_matrix_vector_lane(rhs, 1);
}

#[test]
fn test_todae_projects_scalar_division_with_unary_and_no_event_if_operands() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "c", &[]);
    declare_array(&mut flat, "s", &[]);
    declare_array(&mut flat, "y", &[]);
    let numerator = binary(
        rumoca_core::OpBinary::Sub,
        Expression::Unary {
            op: rumoca_core::OpUnary::Minus,
            rhs: Box::new(make_structured_var_ref("s")),
            span: crate::test_support::test_span(),
        },
        matrix_row_vector_product("A", 1),
    );
    let denominator = builtin(
        rumoca_core::BuiltinFunction::NoEvent,
        vec![Expression::If {
            branches: vec![(make_structured_var_ref("c"), make_structured_var_ref("s"))],
            else_branch: Box::new(make_structured_var_ref("s")),
            span: crate::test_support::test_span(),
        }],
    );
    add_equation(
        &mut flat,
        make_structured_var_ref("y"),
        binary(rumoca_core::OpBinary::Div, numerator, denominator),
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("scalar shape must propagate through unary, noEvent, and if wrappers");
    let [equation] = dae.continuous.equations.as_slice() else {
        panic!("expected one scalar equation");
    };
    assert!(matches!(
        residual_rhs(equation),
        Expression::Binary {
            op: rumoca_core::OpBinary::Div,
            ..
        }
    ));
}

#[test]
fn test_todae_rejects_array_or_unknown_if_condition_shape() {
    let conditions = [
        colon_vector("condition"),
        make_structured_var_ref("unknown"),
    ];
    for condition in conditions {
        let mut flat = Model::new();
        declare_array(&mut flat, "A", &[2, 3]);
        declare_array(&mut flat, "x", &[3]);
        declare_array(&mut flat, "condition", &[2]);
        declare_array(&mut flat, "s", &[]);
        declare_array(&mut flat, "y", &[]);
        let guarded_denominator = builtin(
            rumoca_core::BuiltinFunction::NoEvent,
            vec![Expression::If {
                branches: vec![(condition, make_structured_var_ref("s"))],
                else_branch: Box::new(make_structured_var_ref("s")),
                span: crate::test_support::test_span(),
            }],
        );
        add_equation(
            &mut flat,
            make_structured_var_ref("y"),
            binary(
                rumoca_core::OpBinary::Div,
                matrix_row_vector_product("A", 1),
                guarded_denominator,
            ),
            1,
        );

        assert_projection_error(&flat, "if condition must have proven scalar shape");
    }
}

#[test]
fn test_todae_accepts_scalar_relational_and_boolean_constructor_if_conditions() {
    let boolean_constructor = Expression::FunctionCall {
        name: VarName::new("Boolean").into(),
        args: vec![Expression::FunctionCall {
            name: VarName::new("__rumoca_named_arg__.fixed").into(),
            args: vec![Expression::Literal {
                value: rumoca_core::Literal::Boolean(false),
                span: crate::test_support::test_span(),
            }],
            is_constructor: true,
            span: crate::test_support::test_span(),
        }],
        is_constructor: true,
        span: crate::test_support::test_span(),
    };
    let relational = binary(
        rumoca_core::OpBinary::Gt,
        builtin(
            rumoca_core::BuiltinFunction::Abs,
            vec![Expression::FunctionCall {
                name: VarName::new("scalarSource").into(),
                args: Vec::new(),
                is_constructor: false,
                span: crate::test_support::test_span(),
            }],
        ),
        Expression::Literal {
            value: rumoca_core::Literal::Real(0.0),
            span: crate::test_support::test_span(),
        },
    );
    for condition in [boolean_constructor, relational] {
        let mut flat = Model::new();
        declare_array(&mut flat, "A", &[2, 3]);
        declare_array(&mut flat, "x", &[3]);
        declare_array(&mut flat, "s", &[]);
        declare_array(&mut flat, "y", &[]);
        let mut scalar_source =
            rumoca_core::Function::new("scalarSource", crate::test_support::test_span());
        scalar_source.add_output(rumoca_core::FunctionParam::new(
            "result",
            "Real",
            crate::test_support::test_span(),
        ));
        scalar_source.external = Some(rumoca_core::ExternalFunction {
            language: "C".to_string(),
            function_name: Some("scalar_source".to_string()),
            output_name: Some("result".to_string()),
            ..Default::default()
        });
        flat.add_function(scalar_source);
        add_equation(
            &mut flat,
            make_structured_var_ref("y"),
            binary(
                rumoca_core::OpBinary::Div,
                matrix_row_vector_product("A", 1),
                builtin(
                    rumoca_core::BuiltinFunction::NoEvent,
                    vec![Expression::If {
                        branches: vec![(condition, make_structured_var_ref("s"))],
                        else_branch: Box::new(make_structured_var_ref("s")),
                        span: crate::test_support::test_span(),
                    }],
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
        .expect("typed scalar Boolean constructors and relations must prove condition shape");
    }
}

#[test]
fn test_todae_rejects_array_matrix_product_in_if_condition_with_scalar_dot_value() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[2, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "y", &[]);
    add_equation(
        &mut flat,
        make_structured_var_ref("y"),
        Expression::If {
            branches: vec![(
                matrix_vector_product("A"),
                matrix_row_vector_product("B", 1),
            )],
            else_branch: Box::new(Expression::Literal {
                value: rumoca_core::Literal::Integer(0),
                span: crate::test_support::test_span(),
            }),
            span: crate::test_support::test_span(),
        },
        1,
    );

    assert_projection_error(&flat, "unsupported matrix-product wrapper");
}

#[test]
fn test_todae_rejects_if_wrapper_around_matrix_products() {
    let flat = wrapped_matrix_vector_model(Expression::If {
        branches: vec![(make_structured_var_ref("c"), matrix_vector_product("A"))],
        else_branch: Box::new(matrix_vector_product("B")),
        span: crate::test_support::test_span(),
    });

    assert_projection_error(&flat, "unsupported matrix-product wrapper");
}

#[test]
fn test_todae_rejects_unspanned_wrapper_around_matrix_product_with_typed_error() {
    let flat = wrapped_matrix_vector_model(Expression::Unary {
        op: rumoca_core::OpUnary::Minus,
        rhs: Box::new(matrix_vector_product("A")),
        span: rumoca_core::Span::source_free_serde_default(),
    });

    let error = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect_err("unspanned matrix-product wrapper must fail closed");
    assert!(matches!(
        error,
        ToDaeError::UnspannedRuntimeContractViolation { ref detail }
            if detail == "DAE matrix-product projection: unsupported matrix-product wrapper"
    ));
}

#[test]
fn test_todae_accepts_unary_wrapper_around_elementwise_product() {
    let flat = wrapped_elementwise_vector_model(Expression::Unary {
        op: rumoca_core::OpUnary::Minus,
        rhs: Box::new(elementwise_vector_product()),
        span: crate::test_support::test_span(),
    });

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("wrapped elementwise multiplication must remain lane-projectable");
    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Unary {
            op: rumoca_core::OpUnary::Minus,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected unary wrapper, got {:?}", equation.rhs);
        };
        assert_elementwise_lane_product(rhs, i64::try_from(lane + 1).expect("lane fits i64"));
    }
}

#[test]
fn test_todae_accepts_builtin_wrapper_around_elementwise_product() {
    let flat = wrapped_elementwise_vector_model(builtin(
        rumoca_core::BuiltinFunction::Sin,
        vec![elementwise_vector_product()],
    ));

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("builtin-wrapped elementwise multiplication must remain lane-projectable");
    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Sin,
            args,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected sin wrapper, got {:?}", equation.rhs);
        };
        let [product] = args.as_slice() else {
            panic!("expected one projected sin argument, got {args:?}");
        };
        assert_elementwise_lane_product(product, i64::try_from(lane + 1).expect("lane fits i64"));
    }
}
