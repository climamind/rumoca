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
    add_equation(&mut flat, colon_vector("y"), rhs, 2);
    flat
}

fn matrix_vector_product(name: &str) -> Expression {
    multiply(colon_array(name, 2), colon_vector("x"))
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
