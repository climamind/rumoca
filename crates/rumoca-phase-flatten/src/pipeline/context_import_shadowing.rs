use super::*;

pub(super) fn qualify_expression_with_effective_imports(
    expr: &ast::Expression,
    prefix: &QualifiedName,
    imports: &qualify::ImportMap,
    def_map: Option<&crate::ResolveDefMap>,
    opts: qualify::QualifyOptions,
    instance_name: Option<&str>,
    locals: Option<&std::collections::HashSet<String>>,
) -> Result<rumoca_core::Expression, FlattenError> {
    let qualified = locals.map_or_else(
        || qualify::qualify_expression_with_imports(expr, prefix, opts, imports),
        |locals| {
            qualify::qualify_expression_with_imports_and_locals(expr, prefix, opts, locals, imports)
        },
    );
    crate::ast_lower::expression_from_ast_with_context(
        &qualified,
        crate::ast_lower::LoweringContext {
            def_map,
            class_tree: None,
            instance_name,
        },
    )
}
