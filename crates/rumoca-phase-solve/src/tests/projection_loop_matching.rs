use super::*;

#[test]
fn algebraic_projection_loop_does_not_promote_structural_pairs_to_causal_steps()
-> Result<(), LowerError> {
    let projection_incidence = ProjectionIncidence {
        incidence: Incidence::new(
            vec![
                BTreeSet::from([0, 1]).into_iter().collect(),
                BTreeSet::from([0, 1]).into_iter().collect(),
            ],
            vec![EquationRef(3), EquationRef(4)],
            vec![UnknownId::SolverY(20), UnknownId::SolverY(21)],
        ),
        unknown_y_indices: vec![20, 21],
        preferred_unknowns: vec![Some(0), Some(1)],
    };

    let block = super::super::lower_algebraic_loop_projection_block(
        &[EquationRef(3), EquationRef(4)],
        &[UnknownId::SolverY(21), UnknownId::SolverY(20)],
        &projection_incidence,
        solve_test_span(),
    )?;

    assert_eq!(block.rows, vec![3, 4]);
    assert_eq!(block.y_indices, vec![20, 21]);
    assert!(block.causal_steps.is_empty());
    Ok(())
}

#[test]
fn algebraic_projection_loop_keeps_only_matched_assignment_targets() -> Result<(), LowerError> {
    let projection_incidence = ProjectionIncidence {
        incidence: Incidence::new(
            vec![BTreeSet::from([0, 1]).into_iter().collect()],
            vec![EquationRef(3)],
            vec![UnknownId::SolverY(20), UnknownId::SolverY(21)],
        ),
        unknown_y_indices: vec![20, 21],
        preferred_unknowns: vec![Some(0)],
    };
    let regular = rumoca_phase_structural::maximum_regular_subsystem(
        &projection_incidence.incidence,
        &projection_incidence.preferred_unknowns,
    )
    .map_err(|error| LowerError::contract_violation(error.to_string(), solve_test_span()))?;
    let [rumoca_phase_structural::BltBlock::Scalar { equation, unknown }] =
        regular.blocks.as_slice()
    else {
        panic!("single row must match its preferred assignment target");
    };
    assert_eq!(*unknown, UnknownId::SolverY(20));
    let block = super::super::lower_algebraic_loop_projection_block(
        std::slice::from_ref(equation),
        std::slice::from_ref(unknown),
        &projection_incidence,
        solve_test_span(),
    )?;
    assert_eq!(block.y_indices, vec![20]);
    assert!(block.causal_steps.is_empty());
    Ok(())
}
