use super::*;

pub(super) fn declare_array(flat: &mut Model, name: &str, dims: &[i64]) {
    flat.add_variable(
        VarName::new(name),
        crate::test_support::with_component_ref(flat::Variable {
            name: VarName::new(name),
            dims: dims.to_vec(),
            is_primitive: true,
            ..flat::Variable::empty_with_span(crate::test_support::test_span())
        }),
    );
}

pub(super) fn colon_vector(name: &str) -> Expression {
    colon_array(name, 1)
}

pub(super) fn colon_array(name: &str, rank: usize) -> Expression {
    Expression::Index {
        base: Box::new(make_structured_var_ref(name)),
        subscripts: (0..rank)
            .map(|_| rumoca_core::Subscript::Colon {
                span: crate::test_support::test_span(),
            })
            .collect(),
        span: crate::test_support::test_span(),
    }
}

fn row_slice(name: &str, row: i64) -> Expression {
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
    }
}

pub(super) fn multiply(lhs: Expression, rhs: Expression) -> Expression {
    binary(rumoca_core::OpBinary::Mul, lhs, rhs)
}

pub(super) fn binary(op: rumoca_core::OpBinary, lhs: Expression, rhs: Expression) -> Expression {
    Expression::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        span: crate::test_support::test_span(),
    }
}

pub(super) fn real(value: f64) -> Expression {
    Expression::Literal {
        value: Literal::Real(value),
        span: crate::test_support::test_span(),
    }
}

pub(super) fn builtin(function: rumoca_core::BuiltinFunction, args: Vec<Expression>) -> Expression {
    Expression::BuiltinCall {
        function,
        args,
        span: crate::test_support::test_span(),
    }
}

pub(super) fn add_equation(
    flat: &mut Model,
    lhs: Expression,
    rhs: Expression,
    scalar_count: usize,
) {
    flat.add_equation(flat::Equation {
        residual: Expression::Binary {
            op: rumoca_core::OpBinary::Sub,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span: crate::test_support::test_span(),
        },
        span: crate::test_support::test_span(),
        origin: flat::EquationOrigin::ComponentEquation {
            component: "MatrixProductProjection".to_string(),
        },
        scalar_count,
    });
}

pub(super) fn literal_subscripts(expr: &Expression) -> Option<(&str, Vec<i64>)> {
    let Expression::VarRef {
        name, subscripts, ..
    } = expr
    else {
        return None;
    };
    let indices = subscripts
        .iter()
        .map(|subscript| match subscript {
            rumoca_core::Subscript::Index { value, .. } => Some(*value),
            rumoca_core::Subscript::Expr { expr, .. } => match expr.as_ref() {
                Expression::Literal {
                    value: Literal::Integer(value),
                    ..
                } => Some(*value),
                _ => None,
            },
            rumoca_core::Subscript::Colon { .. } => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((name.as_str(), indices))
}

pub(super) type ProductTerm<'a> = ((&'a str, Vec<i64>), (&'a str, Vec<i64>));

pub(super) fn flatten_dot_terms<'a>(
    expr: &'a Expression,
    terms: &mut Vec<ProductTerm<'a>>,
) -> bool {
    match expr {
        Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } => flatten_dot_terms(lhs, terms) && flatten_dot_terms(rhs, terms),
        Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } => {
            let (Some(lhs), Some(rhs)) = (literal_subscripts(lhs), literal_subscripts(rhs)) else {
                return false;
            };
            terms.push((lhs, rhs));
            true
        }
        _ => false,
    }
}

fn indexed_function_output(expr: &Expression) -> Option<(&str, Vec<i64>)> {
    let Expression::Index {
        base, subscripts, ..
    } = expr
    else {
        return None;
    };
    let Expression::FunctionCall { name, .. } = base.as_ref() else {
        return None;
    };
    let indices = subscripts
        .iter()
        .map(|subscript| match subscript {
            rumoca_core::Subscript::Index { value, .. } => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((name.as_str(), indices))
}

type FunctionProductTerm<'a> = ((&'a str, Vec<i64>), (&'a str, Vec<i64>));

fn flatten_function_dot_terms<'a>(
    expr: &'a Expression,
    terms: &mut Vec<FunctionProductTerm<'a>>,
) -> bool {
    match expr {
        Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } => flatten_function_dot_terms(lhs, terms) && flatten_function_dot_terms(rhs, terms),
        Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } => {
            let (Some(lhs), Some(rhs)) = (indexed_function_output(lhs), literal_subscripts(rhs))
            else {
                return false;
            };
            terms.push((lhs, rhs));
            true
        }
        _ => false,
    }
}

pub(super) type LiteralProductTerm<'a> = ((&'a str, Vec<i64>), f64);

pub(super) fn flatten_literal_dot_terms<'a>(
    expr: &'a Expression,
    terms: &mut Vec<LiteralProductTerm<'a>>,
) -> bool {
    match expr {
        Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } => flatten_literal_dot_terms(lhs, terms) && flatten_literal_dot_terms(rhs, terms),
        Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } => {
            let Some(lhs) = literal_subscripts(lhs) else {
                return false;
            };
            let Expression::Literal {
                value: Literal::Real(rhs),
                ..
            } = rhs.as_ref()
            else {
                return false;
            };
            terms.push((lhs, *rhs));
            true
        }
        _ => false,
    }
}

fn flatten_builtin_dot_terms(expr: &Expression, terms: &mut Vec<(Vec<i64>, i64)>) -> bool {
    match expr {
        Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } => flatten_builtin_dot_terms(lhs, terms) && flatten_builtin_dot_terms(rhs, terms),
        Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } => {
            let Some(("rotation", lhs_indices)) = literal_subscripts(lhs) else {
                return false;
            };
            let Expression::Index {
                base, subscripts, ..
            } = rhs.as_ref()
            else {
                return false;
            };
            if !matches!(
                base.as_ref(),
                Expression::BuiltinCall {
                    function: rumoca_core::BuiltinFunction::Cross,
                    ..
                }
            ) {
                return false;
            }
            let [rumoca_core::Subscript::Index { value, .. }] = subscripts.as_slice() else {
                return false;
            };
            terms.push((lhs_indices, *value));
            true
        }
        _ => false,
    }
}

pub(super) fn residual_rhs(equation: &rumoca_ir_dae::Equation) -> &Expression {
    let Expression::Binary {
        op: rumoca_core::OpBinary::Sub,
        rhs,
        ..
    } = &equation.rhs
    else {
        panic!("expected scalar residual, got {:?}", equation.rhs);
    };
    rhs
}

pub(super) fn assert_projection_error(flat: &Model, expected: &str) {
    let error = to_dae_with_options(
        flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect_err("invalid matrix-product projection must fail closed");
    assert!(
        error.to_string().contains(expected),
        "expected `{expected}` in {error}"
    );
    assert_eq!(error.source_span(), Some(crate::test_support::test_span()));
}

fn add_size_shaped_function(flat: &mut Model, name: &str, rank: usize) {
    let mut function = rumoca_core::Function::new(name, crate::test_support::test_span());
    let mut input = rumoca_core::FunctionParam::new("v", "Real", crate::test_support::test_span())
        .with_dims(vec![0; rank]);
    input.shape_expr = (0..rank)
        .map(|_| rumoca_core::Subscript::Colon {
            span: crate::test_support::test_span(),
        })
        .collect();
    function.add_input(input);
    let mut output =
        rumoca_core::FunctionParam::new("result", "Real", crate::test_support::test_span())
            .with_dims(vec![0; rank]);
    output.shape_expr = (1..=rank)
        .map(|dimension| rumoca_core::Subscript::Expr {
            expr: Box::new(builtin(
                rumoca_core::BuiltinFunction::Size,
                vec![
                    make_structured_var_ref("v"),
                    Expression::Literal {
                        value: Literal::Integer(
                            i64::try_from(dimension).expect("test rank fits i64"),
                        ),
                        span: crate::test_support::test_span(),
                    },
                ],
            )),
            span: crate::test_support::test_span(),
        })
        .collect();
    function.add_output(output);
    function.external = Some(rumoca_core::ExternalFunction {
        language: "C".to_string(),
        function_name: Some(name.to_string()),
        output_name: Some("result".to_string()),
        ..Default::default()
    });
    flat.add_function(function);
}

pub(super) fn expression_row_slice(name: &str, selector: Expression) -> Expression {
    Expression::Index {
        base: Box::new(make_structured_var_ref(name)),
        subscripts: vec![
            rumoca_core::Subscript::Expr {
                expr: Box::new(selector),
                span: crate::test_support::test_span(),
            },
            rumoca_core::Subscript::Colon {
                span: crate::test_support::test_span(),
            },
        ],
        span: crate::test_support::test_span(),
    }
}

pub(super) fn declare_dae_array(dae: &mut rumoca_ir_dae::Dae, name: &str, dims: &[i64]) {
    let mut variable =
        rumoca_ir_dae::Variable::new(VarName::new(name), crate::test_support::test_span());
    variable.dims = dims.to_vec();
    dae.variables
        .algebraics
        .insert(VarName::new(name), variable);
}

pub(super) fn dae_scaling_with_missing_base(variants: &[&str]) -> rumoca_ir_dae::Dae {
    let mut dae = rumoca_ir_dae::Dae::new();
    for (name, dims) in [("x", vec![2]), ("y", vec![2])]
        .into_iter()
        .chain(variants.iter().map(|name| (*name, vec![])))
    {
        declare_dae_array(&mut dae, name, &dims);
    }
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                colon_vector("y"),
                multiply(make_structured_var_ref("gain"), colon_vector("x")),
            ),
            crate::test_support::test_span(),
            "missing-base scaling",
            2,
        ));
    dae
}

#[test]
fn test_todae_preserves_ordinary_scalar_product_operands() {
    let cases = [
        builtin(
            rumoca_core::BuiltinFunction::Sin,
            vec![make_structured_var_ref("x")],
        ),
        Expression::Unary {
            op: rumoca_core::OpUnary::Minus,
            rhs: Box::new(make_structured_var_ref("x")),
            span: crate::test_support::test_span(),
        },
        Expression::If {
            branches: vec![(
                Expression::Literal {
                    value: Literal::Boolean(true),
                    span: crate::test_support::test_span(),
                },
                make_structured_var_ref("x"),
            )],
            else_branch: Box::new(real(1.0)),
            span: crate::test_support::test_span(),
        },
    ];
    for operand in cases {
        let mut flat = Model::new();
        declare_array(&mut flat, "x", &[]);
        declare_array(&mut flat, "y", &[]);
        add_equation(
            &mut flat,
            make_structured_var_ref("y"),
            binary(
                rumoca_core::OpBinary::Add,
                real(1.0),
                multiply(operand, make_structured_var_ref("x")),
            ),
            1,
        );
        let dae = to_dae_with_options(
            &flat,
            ToDaeOptions {
                error_on_unbalanced: false,
            },
        )
        .expect("ordinary scalar products must retain the existing lowering path");
        assert!(matches!(
            residual_rhs(&dae.continuous.equations[0]),
            Expression::Binary {
                op: rumoca_core::OpBinary::Add,
                ..
            }
        ));
    }
}

#[test]
fn test_todae_preserves_scalar_product_for_selected_array_element_target() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[]);
    declare_array(&mut flat, "y", &[2]);
    let lhs = Expression::VarRef {
        name: VarName::new("y").into(),
        subscripts: vec![rumoca_core::Subscript::Index {
            value: 1,
            span: crate::test_support::test_span(),
        }],
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        lhs,
        multiply(
            builtin(
                rumoca_core::BuiltinFunction::Sin,
                vec![make_structured_var_ref("x")],
            ),
            make_structured_var_ref("x"),
        ),
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("a selected array element is a scalar projection target");
    let Expression::Binary { lhs, rhs, .. } = residual_rhs(&dae.continuous.equations[0]) else {
        panic!("expected scalar multiplication");
    };
    let Expression::BuiltinCall { args, .. } = lhs.as_ref() else {
        panic!("expected scalar sin operand");
    };
    assert_eq!(literal_subscripts(&args[0]), Some(("x", vec![])));
    assert_eq!(literal_subscripts(rhs), Some(("x", vec![])));
}

#[test]
fn test_todae_uses_row_major_lane_for_multidimensional_selected_target() {
    let mut flat = Model::new();
    declare_array(&mut flat, "C", &[2, 2]);
    declare_array(&mut flat, "x", &[4]);
    let lhs = Expression::VarRef {
        name: VarName::new("C").into(),
        subscripts: vec![
            rumoca_core::Subscript::Index {
                value: 2,
                span: crate::test_support::test_span(),
            },
            rumoca_core::Subscript::Index {
                value: 1,
                span: crate::test_support::test_span(),
            },
        ],
        span: crate::test_support::test_span(),
    };
    let rhs = Expression::ArrayComprehension {
        expr: Box::new(Expression::VarRef {
            name: VarName::new("x").into(),
            subscripts: vec![rumoca_core::Subscript::Expr {
                expr: Box::new(make_structured_var_ref("i")),
                span: crate::test_support::test_span(),
            }],
            span: crate::test_support::test_span(),
        }),
        indices: vec![rumoca_core::ComprehensionIndex {
            name: "i".to_string(),
            range: Expression::Range {
                start: Box::new(Expression::Literal {
                    value: Literal::Integer(1),
                    span: crate::test_support::test_span(),
                }),
                step: None,
                end: Box::new(Expression::Literal {
                    value: Literal::Integer(4),
                    span: crate::test_support::test_span(),
                }),
                span: crate::test_support::test_span(),
            },
        }],
        filter: None,
        span: crate::test_support::test_span(),
    };
    add_equation(&mut flat, lhs, rhs, 1);

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("a multidimensional selected target must retain its row-major lane");
    assert_eq!(
        literal_subscripts(residual_rhs(&dae.continuous.equations[0])),
        Some(("x", vec![3]))
    );
}

#[test]
fn test_todae_preserves_derivative_vector_scalar_scaling() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[2]);
    add_equation(
        &mut flat,
        builtin(rumoca_core::BuiltinFunction::Der, vec![colon_vector("x")]),
        multiply(real(2.0), colon_vector("x")),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("derivative vector scaling must retain lane scalarization");

    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary { lhs, .. } = &equation.rhs else {
            panic!("expected residual");
        };
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Der,
            args,
            ..
        } = lhs.as_ref()
        else {
            panic!("expected derivative lhs, got {lhs:?}");
        };
        let index = i64::try_from(lane + 1).expect("two lanes fit i64");
        assert_eq!(literal_subscripts(&args[0]), Some(("x", vec![index])));
        let Expression::Binary {
            op: rumoca_core::OpBinary::Mul,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected scalar scaling rhs");
        };
        assert!(matches!(
            lhs.as_ref(),
            Expression::Literal {
                value: Literal::Real(2.0),
                ..
            }
        ));
        assert_eq!(literal_subscripts(rhs), Some(("x", vec![index])));
    }
}

#[test]
fn test_todae_preserves_compound_derivative_vector_target() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[2]);
    add_equation(
        &mut flat,
        binary(
            rumoca_core::OpBinary::Add,
            builtin(rumoca_core::BuiltinFunction::Der, vec![colon_vector("x")]),
            colon_vector("x"),
        ),
        multiply(real(2.0), colon_vector("x")),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("shape-preserving compound derivative lhs must scalarize by lane");

    assert_eq!(dae.continuous.equations.len(), 2);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let index = i64::try_from(lane + 1).expect("two lanes fit i64");
        let Expression::Binary { lhs, .. } = &equation.rhs else {
            panic!("expected residual");
        };
        let Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs: derivative,
            rhs: current,
            ..
        } = lhs.as_ref()
        else {
            panic!("expected compound derivative lhs, got {lhs:?}");
        };
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Der,
            args,
            ..
        } = derivative.as_ref()
        else {
            panic!("expected derivative term");
        };
        assert_eq!(literal_subscripts(&args[0]), Some(("x", vec![index])));
        assert_eq!(literal_subscripts(current), Some(("x", vec![index])));
    }
}

#[test]
fn test_todae_projects_matrix_product_for_derivative_vector_target() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "A", &[3, 3]);
    declare_dae_array(&mut dae, "x", &[3]);
    declare_dae_array(&mut dae, "y", &[3]);
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                builtin(rumoca_core::BuiltinFunction::Der, vec![colon_vector("y")]),
                multiply(
                    builtin(
                        rumoca_core::BuiltinFunction::Transpose,
                        vec![make_structured_var_ref("A")],
                    ),
                    colon_vector("x"),
                ),
            ),
            crate::test_support::test_span(),
            "derivative matrix product",
            3,
        ));

    scalarize_phantom_vector_equations(&mut dae)
        .expect("derivative target shape must support matrix-product projection");

    assert_eq!(dae.continuous.equations.len(), 3);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let output_index = i64::try_from(lane + 1).expect("three lanes fit i64");
        let Expression::Binary { lhs, rhs, .. } = &equation.rhs else {
            panic!("expected scalar residual");
        };
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Der,
            args,
            ..
        } = lhs.as_ref()
        else {
            panic!("expected derivative lhs, got {lhs:?}");
        };
        assert_eq!(
            literal_subscripts(&args[0]),
            Some(("y", vec![output_index]))
        );
        let mut terms = Vec::new();
        assert!(
            flatten_dot_terms(rhs, &mut terms),
            "expected complete dot: {rhs:?}"
        );
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|row| (("A", vec![row, output_index]), ("x", vec![row])))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_projects_matrix_product_nested_in_vector_addition() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "position", &[3]);
    declare_dae_array(&mut dae, "rotation", &[3, 3]);
    declare_dae_array(&mut dae, "offset", &[3]);
    declare_dae_array(&mut dae, "target", &[3]);
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                colon_vector("target"),
                binary(
                    rumoca_core::OpBinary::Add,
                    colon_vector("position"),
                    multiply(make_structured_var_ref("rotation"), colon_vector("offset")),
                ),
            ),
            crate::test_support::test_span(),
            "compound matrix product",
            3,
        ));

    scalarize_phantom_vector_equations(&mut dae)
        .expect("shape-preserving addition must project its nested matrix product");

    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let index = i64::try_from(lane + 1).expect("three lanes fit i64");
        let Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected scalar addition");
        };
        assert_eq!(literal_subscripts(lhs), Some(("position", vec![index])));
        let mut terms = Vec::new();
        assert!(
            flatten_dot_terms(rhs, &mut terms),
            "expected complete dot: {rhs:?}"
        );
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|inner| { (("rotation", vec![index, inner]), ("offset", vec![inner]),) })
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_projects_fixed_vector_builtin_inside_matrix_product() {
    let mut dae = rumoca_ir_dae::Dae::new();
    for (name, dims) in [
        ("velocity", &[3][..]),
        ("rotation", &[3, 3][..]),
        ("omega", &[3][..]),
        ("offset", &[3][..]),
        ("target", &[3][..]),
    ] {
        declare_dae_array(&mut dae, name, dims);
    }
    let cross = builtin(
        rumoca_core::BuiltinFunction::Cross,
        vec![colon_vector("omega"), colon_vector("offset")],
    );
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                colon_vector("target"),
                binary(
                    rumoca_core::OpBinary::Add,
                    colon_vector("velocity"),
                    multiply(make_structured_var_ref("rotation"), cross),
                ),
            ),
            crate::test_support::test_span(),
            "builtin matrix product",
            3,
        ));

    scalarize_phantom_vector_equations(&mut dae)
        .expect("fixed-vector builtins have sufficient shape evidence for matrix projection");

    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let row = i64::try_from(lane + 1).expect("three lanes fit i64");
        let Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected scalar addition");
        };
        assert_eq!(literal_subscripts(lhs), Some(("velocity", vec![row])));
        let mut terms = Vec::new();
        assert!(flatten_builtin_dot_terms(rhs, &mut terms));
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|inner| (vec![row, inner], inner))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_projects_function_sibling_and_nested_matrix_product() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[3, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "y", &[3]);
    let mut function =
        rumoca_core::Function::new("arrayFunction", crate::test_support::test_span());
    function.add_output(
        rumoca_core::FunctionParam::new("result", "Real", crate::test_support::test_span())
            .with_dims(vec![3]),
    );
    function.external = Some(rumoca_core::ExternalFunction {
        language: "C".to_string(),
        function_name: Some("array_function".to_string()),
        output_name: Some("result".to_string()),
        ..Default::default()
    });
    flat.add_function(function);
    let call = Expression::FunctionCall {
        name: VarName::new("arrayFunction").into(),
        args: Vec::new(),
        is_constructor: false,
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        colon_vector("y"),
        binary(
            rumoca_core::OpBinary::Add,
            call,
            multiply(make_structured_var_ref("A"), colon_vector("x")),
        ),
        3,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("function sibling must not disable nested matrix projection");
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let index = i64::try_from(lane + 1).expect("three lanes fit i64");
        let Expression::Binary {
            op: rumoca_core::OpBinary::Add,
            lhs,
            rhs,
            ..
        } = residual_rhs(equation)
        else {
            panic!("expected projected addition");
        };
        let Expression::Index { subscripts, .. } = lhs.as_ref() else {
            panic!("expected indexed function output");
        };
        assert!(
            matches!(subscripts.as_slice(), [rumoca_core::Subscript::Index { value, .. }] if *value == index)
        );
        let mut terms = Vec::new();
        assert!(flatten_dot_terms(rhs, &mut terms));
        assert_eq!(terms.len(), 3);
    }
}

#[test]
fn test_todae_rejects_compound_vector_rhs_for_scalar_target() {
    for explicit_slice in [true, false] {
        let mut flat = Model::new();
        declare_array(&mut flat, "A", &[3, 3]);
        declare_array(&mut flat, "x", &[3]);
        declare_array(&mut flat, "position", &[3]);
        declare_array(&mut flat, "y", &[2]);
        let lhs = Expression::VarRef {
            name: VarName::new("y").into(),
            subscripts: vec![rumoca_core::Subscript::Index {
                value: 2,
                span: crate::test_support::test_span(),
            }],
            span: crate::test_support::test_span(),
        };
        let vector = |name| {
            if explicit_slice {
                colon_vector(name)
            } else {
                make_structured_var_ref(name)
            }
        };
        add_equation(
            &mut flat,
            lhs,
            binary(
                rumoca_core::OpBinary::Add,
                vector("position"),
                multiply(make_structured_var_ref("A"), vector("x")),
            ),
            1,
        );
        assert_projection_error(&flat, "result shape mismatch");
    }
}

#[test]
fn test_todae_projects_matrix_product_for_derivative_matrix_target() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "A", &[2, 3]);
    declare_dae_array(&mut dae, "B", &[3, 2]);
    declare_dae_array(&mut dae, "C", &[2, 2]);
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                builtin(rumoca_core::BuiltinFunction::Der, vec![colon_array("C", 2)]),
                multiply(make_structured_var_ref("A"), make_structured_var_ref("B")),
            ),
            crate::test_support::test_span(),
            "derivative matrix product",
            4,
        ));

    scalarize_phantom_vector_equations(&mut dae)
        .expect("derivative matrix target must project every result cell");

    assert_eq!(dae.continuous.equations.len(), 4);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let row = i64::try_from(lane / 2 + 1).expect("row fits i64");
        let column = i64::try_from(lane % 2 + 1).expect("column fits i64");
        let Expression::Binary { lhs, rhs, .. } = &equation.rhs else {
            panic!("expected scalar residual");
        };
        let Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Der,
            args,
            ..
        } = lhs.as_ref()
        else {
            panic!("expected derivative lhs, got {lhs:?}");
        };
        assert_eq!(literal_subscripts(&args[0]), Some(("C", vec![row, column])));
        let mut terms = Vec::new();
        assert!(
            flatten_dot_terms(rhs, &mut terms),
            "expected complete dot: {rhs:?}"
        );
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|inner| (("A", vec![row, inner]), ("B", vec![inner, column])))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_scalarizer_rejects_unknown_target_for_proven_array_product() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "A", &[2, 2]);
    declare_dae_array(&mut dae, "x", &[2]);
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                make_structured_var_ref("missing_target"),
                multiply(colon_array("A", 2), colon_vector("x")),
            ),
            crate::test_support::test_span(),
            "unknown matrix-product target",
            2,
        ));

    let error = scalarize_phantom_vector_equations(&mut dae)
        .expect_err("proven array product must not use a same-lane unknown-target fallback");
    assert!(error.to_string().contains("unknown target shape"));
    assert_eq!(error.source_span(), Some(crate::test_support::test_span()));
}

#[test]
fn test_scalarizer_rejects_unknown_target_for_compound_rhs_with_matrix_product() {
    let mut dae = rumoca_ir_dae::Dae::new();
    declare_dae_array(&mut dae, "position", &[2]);
    declare_dae_array(&mut dae, "A", &[2, 2]);
    declare_dae_array(&mut dae, "x", &[2]);
    dae.continuous
        .equations
        .push(rumoca_ir_dae::Equation::residual_array(
            binary(
                rumoca_core::OpBinary::Sub,
                make_structured_var_ref("missing_target"),
                binary(
                    rumoca_core::OpBinary::Add,
                    colon_vector("position"),
                    multiply(colon_array("A", 2), colon_vector("x")),
                ),
            ),
            crate::test_support::test_span(),
            "unknown compound matrix-product target",
            2,
        ));

    let error = scalarize_phantom_vector_equations(&mut dae)
        .expect_err("compound matrix-product RHS must reject an unknown target shape");
    assert!(error.to_string().contains("unknown target shape"));
    assert_eq!(error.source_span(), Some(crate::test_support::test_span()));
}

#[test]
fn test_todae_projects_transposed_matrix_vector_rows_as_three_term_dots() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[3, 3]);
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "y", &[3]);

    let transpose_a = Expression::BuiltinCall {
        function: rumoca_core::BuiltinFunction::Transpose,
        args: vec![make_structured_var_ref("A")],
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        colon_vector("y"),
        multiply(transpose_a, colon_vector("x")),
        3,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("shape-correct matrix-vector product should reach finalized DAE");

    assert_eq!(dae.continuous.equations.len(), 3);
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let Expression::Binary {
            op: rumoca_core::OpBinary::Sub,
            lhs,
            rhs,
            ..
        } = &equation.rhs
        else {
            panic!("expected scalar residual, got {:?}", equation.rhs);
        };
        let output_index = i64::try_from(lane + 1).expect("three lanes fit i64");
        assert_eq!(literal_subscripts(lhs), Some(("y", vec![output_index])));

        let mut terms = Vec::new();
        assert!(
            flatten_dot_terms(rhs, &mut terms),
            "DAE lane {} must be a complete dot product, got {rhs:?}",
            lane + 1
        );
        let expected = (1_i64..=3)
            .map(|row| (("A", vec![row, output_index]), ("x", vec![row])))
            .collect::<Vec<_>>();
        assert_eq!(
            terms,
            expected,
            "DAE lane {} must contain every inner-dimension term",
            lane + 1
        );
    }
}

#[test]
fn test_todae_projects_indexed_vector_matrix_columns_as_three_term_dots() {
    let mut flat = Model::new();
    declare_array(&mut flat, "source", &[2, 3]);
    declare_array(&mut flat, "B", &[3, 2]);
    declare_array(&mut flat, "y", &[2]);
    add_equation(
        &mut flat,
        colon_vector("y"),
        multiply(row_slice("source", 1), make_structured_var_ref("B")),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("indexed vector-matrix product should lower");

    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let column = i64::try_from(lane + 1).expect("two lanes fit i64");
        let Expression::Binary { lhs, rhs, .. } = &equation.rhs else {
            panic!("expected residual");
        };
        assert_eq!(literal_subscripts(lhs), Some(("y", vec![column])));
        let mut terms = Vec::new();
        assert!(flatten_dot_terms(rhs, &mut terms), "got {rhs:?}");
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|inner| (("source", vec![1, inner]), ("B", vec![inner, column])))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_projects_matrix_matrix_cells_as_three_term_dots() {
    let mut flat = Model::new();
    declare_array(&mut flat, "A", &[2, 3]);
    declare_array(&mut flat, "B", &[3, 2]);
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
    .expect("matrix-matrix product should lower");

    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let row = i64::try_from(lane / 2 + 1).expect("row fits i64");
        let column = i64::try_from(lane % 2 + 1).expect("column fits i64");
        let Expression::Binary { lhs, rhs, .. } = &equation.rhs else {
            panic!("expected residual");
        };
        assert_eq!(literal_subscripts(lhs), Some(("C", vec![row, column])));
        let mut terms = Vec::new();
        assert!(flatten_dot_terms(rhs, &mut terms), "got {rhs:?}");
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|inner| (("A", vec![row, inner]), ("B", vec![inner, column])))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_projects_matrix_valued_function_times_vector_with_multidimensional_indices() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "y", &[2]);
    let mut function = rumoca_core::Function::new("matrixSource", crate::test_support::test_span());
    function.add_output(
        rumoca_core::FunctionParam::new("result", "Real", crate::test_support::test_span())
            .with_dims(vec![2, 3]),
    );
    function.external = Some(rumoca_core::ExternalFunction {
        language: "C".to_string(),
        function_name: Some("matrix_source".to_string()),
        output_name: Some("result".to_string()),
        ..Default::default()
    });
    flat.add_function(function);
    let call = Expression::FunctionCall {
        name: VarName::new("matrixSource").into(),
        args: Vec::new(),
        is_constructor: false,
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        colon_vector("y"),
        multiply(call, make_structured_var_ref("x")),
        2,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("matrix-valued function output must retain row and inner indices");
    for (lane, equation) in dae.continuous.equations.iter().enumerate() {
        let row = i64::try_from(lane + 1).expect("lane fits i64");
        let mut terms = Vec::new();
        assert!(flatten_function_dot_terms(
            residual_rhs(equation),
            &mut terms
        ));
        assert_eq!(
            terms,
            (1_i64..=3)
                .map(|inner| { (("matrixSource", vec![row, inner]), ("x", vec![inner]),) })
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_todae_specializes_function_output_size_from_array_actual() {
    let mut flat = Model::new();
    declare_array(&mut flat, "x", &[3]);
    declare_array(&mut flat, "y", &[]);
    let mut function = rumoca_core::Function::new("normalize", crate::test_support::test_span());
    let mut input = rumoca_core::FunctionParam::new("v", "Real", crate::test_support::test_span())
        .with_dims(vec![0]);
    input.shape_expr = vec![rumoca_core::Subscript::Colon {
        span: crate::test_support::test_span(),
    }];
    function.add_input(input);
    let mut output =
        rumoca_core::FunctionParam::new("result", "Real", crate::test_support::test_span())
            .with_dims(vec![0]);
    output.shape_expr = vec![rumoca_core::Subscript::Expr {
        expr: Box::new(builtin(
            rumoca_core::BuiltinFunction::Size,
            vec![
                make_structured_var_ref("v"),
                Expression::Literal {
                    value: Literal::Integer(1),
                    span: crate::test_support::test_span(),
                },
            ],
        )),
        span: crate::test_support::test_span(),
    }];
    function.add_output(output);
    function.external = Some(rumoca_core::ExternalFunction {
        language: "C".to_string(),
        function_name: Some("normalize".to_string()),
        output_name: Some("result".to_string()),
        ..Default::default()
    });
    flat.add_function(function);
    let call = Expression::FunctionCall {
        name: VarName::new("normalize").into(),
        args: vec![Expression::Array {
            elements: vec![real(1.0), real(0.0), real(0.0)],
            is_matrix: false,
            span: crate::test_support::test_span(),
        }],
        is_constructor: false,
        span: crate::test_support::test_span(),
    };
    add_equation(
        &mut flat,
        make_structured_var_ref("y"),
        multiply(call, make_structured_var_ref("x")),
        1,
    );

    let dae = to_dae_with_options(
        &flat,
        ToDaeOptions {
            error_on_unbalanced: false,
        },
    )
    .expect("size(v, 1) must specialize from the array actual before dot projection");
    assert_eq!(dae.continuous.equations.len(), 1);
}

#[test]
fn test_todae_specializes_size_dimensions_from_single_row_and_nested_matrix_literals() {
    let single_row = Expression::Array {
        elements: vec![real(1.0), real(2.0), real(3.0)],
        is_matrix: true,
        span: crate::test_support::test_span(),
    };
    let nested_rows = Expression::Array {
        elements: vec![
            Expression::Array {
                elements: vec![real(1.0), real(2.0), real(3.0)],
                is_matrix: true,
                span: crate::test_support::test_span(),
            },
            Expression::Array {
                elements: vec![real(4.0), real(5.0), real(6.0)],
                is_matrix: true,
                span: crate::test_support::test_span(),
            },
        ],
        is_matrix: true,
        span: crate::test_support::test_span(),
    };
    for (actual, rows) in [(single_row, 1_i64), (nested_rows, 2_i64)] {
        let mut flat = Model::new();
        declare_array(&mut flat, "B", &[3, 2]);
        declare_array(&mut flat, "Y", &[rows, 2]);
        add_size_shaped_function(&mut flat, "matrixIdentity", 2);
        let call = Expression::FunctionCall {
            name: VarName::new("matrixIdentity").into(),
            args: vec![actual],
            is_constructor: false,
            span: crate::test_support::test_span(),
        };
        add_equation(
            &mut flat,
            colon_array("Y", 2),
            multiply(call, make_structured_var_ref("B")),
            usize::try_from(rows * 2).expect("positive matrix size"),
        );

        let dae = to_dae_with_options(
            &flat,
            ToDaeOptions {
                error_on_unbalanced: false,
            },
        )
        .expect("size(actual, 1/2) must preserve matrix literal row and column counts");
        assert_eq!(
            dae.continuous.equations.len(),
            usize::try_from(rows * 2).expect("positive matrix size")
        );
        for (lane, equation) in dae.continuous.equations.iter().enumerate() {
            let row = i64::try_from(lane / 2 + 1).expect("row fits i64");
            let column = i64::try_from(lane % 2 + 1).expect("column fits i64");
            let mut terms = Vec::new();
            assert!(
                flatten_function_dot_terms(residual_rhs(equation), &mut terms),
                "unexpected projected matrix literal product: {:?}",
                residual_rhs(equation)
            );
            assert_eq!(
                terms,
                (1_i64..=3)
                    .map(|inner| {
                        (
                            ("matrixIdentity", vec![row, inner]),
                            ("B", vec![inner, column]),
                        )
                    })
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn test_todae_rejects_unprovable_literal_array_actual_shapes() {
    let expression_element = Expression::Array {
        elements: vec![real(1.0), make_structured_var_ref("element")],
        is_matrix: false,
        span: crate::test_support::test_span(),
    };
    let mut expression_flat = Model::new();
    declare_array(&mut expression_flat, "element", &[]);
    declare_array(&mut expression_flat, "x", &[2]);
    declare_array(&mut expression_flat, "y", &[]);
    add_size_shaped_function(&mut expression_flat, "vectorIdentity", 1);
    add_equation(
        &mut expression_flat,
        make_structured_var_ref("y"),
        multiply(
            Expression::FunctionCall {
                name: VarName::new("vectorIdentity").into(),
                args: vec![expression_element],
                is_constructor: false,
                span: crate::test_support::test_span(),
            },
            make_structured_var_ref("x"),
        ),
        1,
    );
    assert_projection_error(&expression_flat, "unknown operand shape");

    let ragged_matrix = Expression::Array {
        elements: vec![
            Expression::Array {
                elements: vec![real(1.0), real(2.0)],
                is_matrix: true,
                span: crate::test_support::test_span(),
            },
            Expression::Array {
                elements: vec![real(3.0)],
                is_matrix: true,
                span: crate::test_support::test_span(),
            },
        ],
        is_matrix: true,
        span: crate::test_support::test_span(),
    };
    let mut ragged_flat = Model::new();
    declare_array(&mut ragged_flat, "x", &[2]);
    declare_array(&mut ragged_flat, "y", &[2]);
    add_size_shaped_function(&mut ragged_flat, "matrixIdentity", 2);
    add_equation(
        &mut ragged_flat,
        colon_vector("y"),
        multiply(
            Expression::FunctionCall {
                name: VarName::new("matrixIdentity").into(),
                args: vec![ragged_matrix],
                is_constructor: false,
                span: crate::test_support::test_span(),
            },
            make_structured_var_ref("x"),
        ),
        2,
    );
    assert_projection_error(&ragged_flat, "unknown operand shape");
}
