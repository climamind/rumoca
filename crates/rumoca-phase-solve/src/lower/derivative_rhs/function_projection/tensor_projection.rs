use super::*;

impl<'a> FunctionProjectionAnalysis<'a> {
    pub(super) fn project_scoped_selection_value(
        &self,
        name: &rumoca_core::Reference,
        subscripts: &[rumoca_core::Subscript],
        ctx: ScopedSelectionValueCtx<'_>,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let ScopedSelectionValueCtx {
            result_dims,
            flat_index,
            scope,
            depth,
            span,
        } = ctx;
        let Some(base_dims) = scope.dims.get(name.as_str()) else {
            return Ok(None);
        };
        let Some(values) = scope.scalars.get(name.as_str()) else {
            return Ok(None);
        };
        let result_indices = required_flat_index_to_subscripts(result_dims, flat_index, span)?;
        let mut result_axis = 0usize;
        let mut scalar_subscripts = projection_vec_with_capacity(
            base_dims.len(),
            "projected selected expression subscript count",
            span,
        )?;
        for axis in 0..base_dims.len() {
            let projected = self.project_scoped_selection_subscript(
                subscripts.get(axis),
                ScopedSubscriptProjectionCtx {
                    result_indices: &result_indices,
                    result_axis: &mut result_axis,
                    scope,
                    depth,
                    span,
                },
            )?;
            let Some(projected) = projected else {
                return Ok(None);
            };
            scalar_subscripts.push(projected);
        }
        if result_axis != result_indices.len() {
            return Err(LowerError::contract_violation(
                "projected selected expression did not consume its result dimensions",
                span,
            ));
        }
        projected_scalar_selection(
            ScalarSelectionCtx {
                name: name.as_str(),
                subscripts: &scalar_subscripts,
                dims: base_dims,
                values,
                span,
                depth,
            },
            self,
            scope,
        )
        .map(Some)
    }

    fn project_scoped_selection_subscript(
        &self,
        subscript: Option<&rumoca_core::Subscript>,
        ctx: ScopedSubscriptProjectionCtx<'_>,
    ) -> Result<Option<rumoca_core::Subscript>, LowerError> {
        match subscript {
            None | Some(rumoca_core::Subscript::Colon { .. }) => {
                result_axis_subscript(ctx.result_indices, ctx.result_axis, ctx.span).map(Some)
            }
            Some(subscript @ rumoca_core::Subscript::Index { .. }) => Ok(Some(subscript.clone())),
            Some(rumoca_core::Subscript::Expr { expr, span }) => {
                self.project_scoped_expression_subscript(expr, *span, ctx)
            }
        }
    }

    fn project_scoped_expression_subscript(
        &self,
        expr: &rumoca_core::Expression,
        source_span: rumoca_core::Span,
        ctx: ScopedSubscriptProjectionCtx<'_>,
    ) -> Result<Option<rumoca_core::Subscript>, LowerError> {
        let span = inherited_projection_span(source_span, ctx.span);
        let Some(index_dims) = self.expr_dims_with_owner(expr, ctx.scope, ctx.depth + 1, span)?
        else {
            return Ok(None);
        };
        if index_dims.is_empty() {
            return Ok(Some(rumoca_core::Subscript::Expr {
                expr: Box::new(expr.clone()),
                span,
            }));
        }
        let [index_width] = index_dims.as_slice() else {
            return Err(unsupported_at(
                "array subscript expression must be scalar or one-dimensional",
                span,
            ));
        };
        let coordinate = result_axis_coordinate(ctx.result_indices, ctx.result_axis, span)?;
        validate_projected_subscript_coordinate(coordinate, *index_width, span)?;
        let projected = self
            .project_value(
                expr,
                &index_dims,
                coordinate - 1,
                ctx.scope,
                ctx.depth + 1,
                span,
            )?
            .ok_or_else(|| {
                unsupported_at("array subscript expression could not be projected", span)
            })?;
        Ok(Some(rumoca_core::Subscript::Expr {
            expr: Box::new(projected),
            span,
        }))
    }

    pub(super) fn project_function_call_value(
        &self,
        expr: &rumoca_core::Expression,
        dims: &[i64],
        flat_index: usize,
        scope: &FunctionProjectionScope,
        depth: usize,
        owner_span: rumoca_core::Span,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        if let Some(indexed_call) =
            self.indexed_selected_output_call(expr, dims, flat_index, owner_span)?
        {
            return self.project_function_call_value(
                &indexed_call,
                &[],
                0,
                scope,
                depth + 1,
                owner_span,
            );
        }
        let (outputs, first_probe_declined) = match self
            .function_call_outputs_with_projection_scope(expr, depth + 1, owner_span, Some(scope))
        {
            Ok(Some(outputs)) => (Some(outputs), false),
            Ok(None) => (None, true),
            Err(err) if err.is_projection_budget_exceeded() => (None, false),
            Err(err) => return Err(err),
        };
        if let Some(outputs) = outputs {
            let span = inherited_projection_source_span(expr.span(), owner_span);
            let ctx = projection_value_ctx(dims, flat_index, scope, depth, span);
            if let [output] = outputs.as_slice() {
                return self
                    .project_lane_or_substitute(&output.expr, &ctx)
                    .map(Some);
            }
            if function_call_declared_output_count(expr, self.dae_model)
                .is_some_and(|count| count > 1)
                && let Some(output) = outputs.first()
            {
                return self
                    .project_lane_or_substitute(&output.expr, &ctx)
                    .map(Some);
            }
            return outputs
                .get(flat_index)
                .map(|output| {
                    self.project_lane_or_substitute(&output.expr, &ctx)
                        .map(Some)
                })
                .unwrap_or(Ok(None));
        }
        let span = inherited_projection_source_span(expr.span(), owner_span);
        let mut call =
            self.project_function_call_with_lane_args(expr, dims, flat_index, scope, depth, span)?;
        if call.span().is_none() {
            call = call.with_span(owner_span);
        }
        let outputs = self.function_call_outputs_with_owner(&call, depth + 1, owner_span)?;
        let Some(outputs) = outputs else {
            if first_probe_declined && is_direct_single_array_output_call(expr, self.dae_model) {
                return Ok(None);
            }
            return Ok(Some(call));
        };
        if let [output] = outputs.as_slice() {
            return Ok(Some(output.expr.clone()));
        }
        if function_call_declared_output_count(&call, self.dae_model).is_some_and(|count| count > 1)
        {
            return Ok(outputs.first().map(|output| output.expr.clone()));
        }
        Ok(outputs.get(flat_index).map(|output| output.expr.clone()))
    }

    pub(super) fn project_indexed_value(
        &self,
        expr: &rumoca_core::Expression,
        dims: &[i64],
        flat_index: usize,
        scope: &FunctionProjectionScope,
        owner_span: rumoca_core::Span,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let span = inherited_projection_source_span(expr.span(), owner_span);
        let indices = required_flat_index_to_subscripts(dims, flat_index, span)?;
        let mut subscripts = projection_vec_with_capacity(
            indices.len(),
            "projected expression subscript count",
            span,
        )?;
        for idx in indices {
            subscripts.push(checked_generated_subscript_from_usize(
                idx,
                span,
                "projected expression index subscript",
            )?);
        }
        Ok(Some(rumoca_core::Expression::Index {
            base: Box::new(self.substitute(expr, scope)?),
            subscripts,
            span,
        }))
    }

    pub(super) fn project_binary_elementwise(
        &self,
        op: OpBinary,
        lhs: &rumoca_core::Expression,
        rhs: &rumoca_core::Expression,
        ctx: &ProjectionValueCtx<'_>,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let lhs_dims = self.known_expr_dims(lhs, ctx.scope, ctx.depth, "binary lhs", ctx.span)?;
        let rhs_dims = self.known_expr_dims(rhs, ctx.scope, ctx.depth, "binary rhs", ctx.span)?;
        let lhs_expr = if lhs_dims.is_empty() {
            self.project_lane_or_substitute(lhs, ctx)?
        } else {
            let flat_index = projected_child_flat_index(&lhs_dims, ctx.flat_index);
            self.project_value(lhs, &lhs_dims, flat_index, ctx.scope, ctx.depth, ctx.span)?
                .ok_or_else(|| unsupported_at("binary lhs could not be projected", ctx.span))?
        };
        let rhs_expr = if rhs_dims.is_empty() {
            self.project_lane_or_substitute(rhs, ctx)?
        } else {
            let flat_index = projected_child_flat_index(&rhs_dims, ctx.flat_index);
            self.project_value(rhs, &rhs_dims, flat_index, ctx.scope, ctx.depth, ctx.span)?
                .ok_or_else(|| unsupported_at("binary rhs could not be projected", ctx.span))?
        };
        Ok(Some(rumoca_core::Expression::Binary {
            op,
            lhs: Box::new(lhs_expr),
            rhs: Box::new(rhs_expr),
            span: ctx.span,
        }))
    }

    pub(super) fn project_tensor_product(
        &self,
        lhs: &rumoca_core::Expression,
        rhs: &rumoca_core::Expression,
        ctx: &ProjectionValueCtx<'_>,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let Some(lhs_dims) = self.expr_dims_with_owner(lhs, ctx.scope, ctx.depth, ctx.span)? else {
            return Ok(None);
        };
        let Some(rhs_dims) = self.expr_dims_with_owner(rhs, ctx.scope, ctx.depth, ctx.span)? else {
            return Ok(None);
        };
        match (lhs_dims.as_slice(), rhs_dims.as_slice(), ctx.dims) {
            ([rows, cols], [n], [_]) if cols == n => self.project_matrix_vector_product(
                lhs,
                rhs,
                MatrixVectorProductDims {
                    lhs_dims: &lhs_dims,
                    rhs_dims: &rhs_dims,
                    rows: *rows,
                    cols: *cols,
                },
                ctx,
            ),
            ([n], [rows, cols], [_]) if n == rows => {
                self.project_vector_matrix_product(lhs, rhs, &rhs_dims, ctx, *rows, *cols)
            }
            ([rows, inner_lhs], [inner_rhs, cols], [out_rows, out_cols])
                if inner_lhs == inner_rhs && rows == out_rows && cols == out_cols =>
            {
                self.project_matrix_matrix_product(lhs, rhs, &lhs_dims, &rhs_dims, ctx, *cols)
            }
            _ => Ok(None),
        }
    }

    pub(super) fn project_matrix_vector_product(
        &self,
        lhs: &rumoca_core::Expression,
        rhs: &rumoca_core::Expression,
        product_dims: MatrixVectorProductDims<'_>,
        ctx: &ProjectionValueCtx<'_>,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let rows = valid_product_dim(product_dims.rows, ctx.span, "matrix-vector rows")?;
        let cols = valid_product_dim(product_dims.cols, ctx.span, "matrix-vector columns")?;
        if ctx.flat_index >= rows {
            return Ok(None);
        }
        let row = ctx.flat_index;
        let mut terms =
            projection_vec_with_capacity(cols, "matrix-vector product term count", ctx.span)?;
        for col in 0..cols {
            let lhs_idx = checked_projection_offset(
                row,
                cols,
                col,
                "matrix-vector lhs flat index",
                ctx.span,
            )?;
            let lhs_term = self
                .project_value(
                    lhs,
                    product_dims.lhs_dims,
                    lhs_idx,
                    ctx.scope,
                    ctx.depth,
                    ctx.span,
                )?
                .ok_or_else(|| {
                    unsupported_at("matrix-vector lhs could not be projected", ctx.span)
                })?;
            let rhs_term = self
                .project_value(
                    rhs,
                    product_dims.rhs_dims,
                    col,
                    ctx.scope,
                    ctx.depth,
                    ctx.span,
                )?
                .ok_or_else(|| {
                    unsupported_at("matrix-vector rhs could not be projected", ctx.span)
                })?;
            terms.push(rumoca_core::Expression::Binary {
                op: OpBinary::Mul,
                lhs: Box::new(lhs_term),
                rhs: Box::new(rhs_term),
                span: ctx.span,
            });
        }
        Ok(Some(sum_expressions(terms, ctx.span)))
    }

    pub(super) fn project_vector_matrix_product(
        &self,
        lhs: &rumoca_core::Expression,
        rhs: &rumoca_core::Expression,
        rhs_dims: &[i64],
        ctx: &ProjectionValueCtx<'_>,
        rows: i64,
        cols: i64,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let rows = valid_product_dim(rows, ctx.span, "vector-matrix rows")?;
        let cols = valid_product_dim(cols, ctx.span, "vector-matrix columns")?;
        if ctx.flat_index >= cols {
            return Ok(None);
        }
        let col = ctx.flat_index;
        let lhs_dims = [checked_usize_to_i64(rows, "vector-matrix rows", ctx.span)?];
        let mut terms =
            projection_vec_with_capacity(rows, "vector-matrix product term count", ctx.span)?;
        for row in 0..rows {
            let lhs_term = self
                .project_value(lhs, &lhs_dims, row, ctx.scope, ctx.depth, ctx.span)?
                .ok_or_else(|| {
                    unsupported_at("vector-matrix lhs could not be projected", ctx.span)
                })?;
            let rhs_idx = checked_projection_offset(
                row,
                cols,
                col,
                "vector-matrix rhs flat index",
                ctx.span,
            )?;
            let rhs_term = self
                .project_value(rhs, rhs_dims, rhs_idx, ctx.scope, ctx.depth, ctx.span)?
                .ok_or_else(|| {
                    unsupported_at("vector-matrix rhs could not be projected", ctx.span)
                })?;
            terms.push(rumoca_core::Expression::Binary {
                op: OpBinary::Mul,
                lhs: Box::new(lhs_term),
                rhs: Box::new(rhs_term),
                span: ctx.span,
            });
        }
        Ok(Some(sum_expressions(terms, ctx.span)))
    }

    pub(super) fn project_matrix_matrix_product(
        &self,
        lhs: &rumoca_core::Expression,
        rhs: &rumoca_core::Expression,
        lhs_dims: &[i64],
        rhs_dims: &[i64],
        ctx: &ProjectionValueCtx<'_>,
        cols: i64,
    ) -> Result<Option<rumoca_core::Expression>, LowerError> {
        let inner = valid_product_dim(lhs_dims[1], ctx.span, "matrix-matrix inner dimension")?;
        let cols = valid_product_dim(cols, ctx.span, "matrix-matrix columns")?;
        if cols == 0 {
            return Ok(None);
        }
        let row = ctx.flat_index / cols;
        let col = ctx.flat_index % cols;
        let mut terms =
            projection_vec_with_capacity(inner, "matrix-matrix product term count", ctx.span)?;
        for inner_idx in 0..inner {
            let lhs_idx = checked_projection_offset(
                row,
                inner,
                inner_idx,
                "matrix-matrix lhs flat index",
                ctx.span,
            )?;
            let rhs_idx = checked_projection_offset(
                inner_idx,
                cols,
                col,
                "matrix-matrix rhs flat index",
                ctx.span,
            )?;
            let lhs_term = self
                .project_value(lhs, lhs_dims, lhs_idx, ctx.scope, ctx.depth, ctx.span)?
                .ok_or_else(|| {
                    unsupported_at("matrix-matrix lhs could not be projected", ctx.span)
                })?;
            let rhs_term = self
                .project_value(rhs, rhs_dims, rhs_idx, ctx.scope, ctx.depth, ctx.span)?
                .ok_or_else(|| {
                    unsupported_at("matrix-matrix rhs could not be projected", ctx.span)
                })?;
            terms.push(rumoca_core::Expression::Binary {
                op: OpBinary::Mul,
                lhs: Box::new(lhs_term),
                rhs: Box::new(rhs_term),
                span: ctx.span,
            });
        }
        Ok(Some(sum_expressions(terms, ctx.span)))
    }
}

fn split_flattened_projection_input_name(name: &str) -> Option<(&str, &str)> {
    let (prefix, field) = name.split_once('_')?;
    (!prefix.is_empty() && !field.is_empty()).then_some((prefix, field))
}

fn flattened_projection_input_has_prefix(name: &str, prefix: &str) -> bool {
    split_flattened_projection_input_name(name).is_some_and(|(candidate, _)| candidate == prefix)
}

fn flattened_projection_group_has_prefix(
    inputs: &[rumoca_core::FunctionParam],
    prefix: &str,
) -> bool {
    inputs
        .iter()
        .filter(|input| flattened_projection_input_has_prefix(&input.name, prefix))
        .take(2)
        .count()
        >= 2
}

fn flattened_projection_input_is_group_start(
    inputs: &[rumoca_core::FunctionParam],
    input_idx: usize,
    prefix: &str,
) -> bool {
    !inputs
        .iter()
        .take(input_idx)
        .any(|input| flattened_projection_input_has_prefix(&input.name, prefix))
}

fn only_projected_scalar_assignment_output(
    mut outputs: Vec<ProjectedFunctionOutput>,
    span: rumoca_core::Span,
) -> Result<ProjectedFunctionOutput, LowerError> {
    outputs.pop().ok_or_else(|| {
        LowerError::contract_violation(
            "projected scalar function assignment produced no output",
            span,
        )
    })
}
