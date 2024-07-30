//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_ABSTRACT_FIELD_H
#define CRACKS_ABSTRACT_FIELD_H

#include "controller.h"
#include "dealii_includes.h"

template <int dim> class AbstractField {
public:
  explicit AbstractField(Controller<dim> &ctl);

  virtual void setup_boundary_condition(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void assemble_system(bool residual_only, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual unsigned int solve(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };

  void setup_system(Controller<dim> &ctl);
  void record_old_solution(Controller<dim> &ctl);
  void return_old_solution(Controller<dim> &ctl);
  void distribute_hanging_node_constraints(LA::MPI::Vector &vector,
                                                   Controller<dim> &ctl);
  void distribute_all_constraints(LA::MPI::Vector &vector,
                                          Controller<dim> &ctl);

  double newton_iteration(Controller<dim> &ctl);

  /*
   * Preconditioners
   */
  LA::MPI::PreconditionAMG preconditioner;
  std::vector<std::vector<bool>> constant_modes;

  /*
   * FE system, constraints, and dof handler
   */
  FESystem<dim> fe;
  const QGauss<dim> quadrature_formula;
  DoFHandler<dim> dof_handler;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  AffineConstraints<double> constraints_hanging_nodes;
  AffineConstraints<double> constraints_all;

  /*
   * Solutions
   */
  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector solution, newton_update, old_solution;
  //  LA::MPI::Vector system_total_residual;
  LA::MPI::Vector system_rhs;
  //  LA::MPI::Vector diag_mass, diag_mass_relevant;
  LA::MPI::Vector increment;
};

template <int dim>
AbstractField<dim>::AbstractField(Controller<dim> &ctl)
    : fe(FE_Q<dim>(ctl.params.poly_degree), dim),
      quadrature_formula(fe.degree + 1), dof_handler(ctl.triangulation) {}

template <int dim>
void AbstractField<dim>::setup_system(Controller<dim> &ctl) {
    system_matrix.clear();
    /**
   * DOF
   **/
    {
      dof_handler.distribute_dofs(fe);
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      constant_modes.clear();
      DoFTools::extract_constant_modes(dof_handler, ComponentMask(),
                                       constant_modes);
    }
    /**
   * Hanging node and boundary value constraints
     */
    {
      constraints_hanging_nodes.clear();
      constraints_hanging_nodes.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              constraints_hanging_nodes);
      constraints_hanging_nodes.close();
      
      constraints_all.clear();
      constraints_all.reinit(locally_relevant_dofs);
      setup_boundary_condition(ctl);
      constraints_all.close();
    }

    /**
   * Sparsity pattern
     */
    {
      DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern,
                                      constraints_all,
                                      /*keep constrained dofs*/ false);
      SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                                 locally_owned_dofs, ctl.mpi_com,
                                                 locally_relevant_dofs);
      sparsity_pattern.compress();
      system_matrix.reinit(locally_owned_dofs, locally_owned_dofs,
                                   sparsity_pattern, ctl.mpi_com);
    }

    /**
   * Initialize solution
     */
    {
      // solution has ghost elements.
      solution.reinit(locally_owned_dofs, locally_relevant_dofs, ctl.mpi_com);
      // system_rhs, system_matrix, and the solution vector increment do not have
      // ghost elements
      increment.reinit(locally_owned_dofs, ctl.mpi_com);
      system_rhs.reinit(locally_owned_dofs, ctl.mpi_com);
      solution = 0;
      old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, ctl.mpi_com);
      old_solution = solution;
      // Initialize fields. Trilino does not allow writing into its parallel
      // vector.
      //    VectorTools::interpolate(dof_handler, ZeroFunction<dim>(dim),
      //                             solution);
    }

}

template <int dim>
double AbstractField<dim>::newton_iteration(Controller<dim> &ctl) {
  ctl.pcout << "It.\tResidual\tReduction\tLSrch\t\t#LinIts" << std::endl;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1;

  // Line search parameters
  unsigned int line_search_step = 0;
  double new_newton_residual = 0.0;

  // Cannot distribute constraints to parallel vectors with ghost dofs.
  LA::MPI::Vector distributed_solution(locally_owned_dofs, ctl.mpi_com);
  distributed_solution = solution;

  // Application of the initial boundary conditions to the
  // variational equations:
  setup_boundary_condition(ctl);
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;
  assemble_system(true, ctl);

  double newton_residual = system_rhs.linfty_norm();
  double old_newton_residual = newton_residual;
  unsigned int newton_step = 1;
  unsigned int no_linear_iterations = 0;

  ctl.pcout << "0\t" << std::scientific << newton_residual << std::endl;

  while (newton_residual > ctl.params.lower_bound_newton_residual &&
         newton_step < ctl.params.max_no_newton_steps) {
    old_newton_residual = newton_residual;

    assemble_system(true, ctl);
    newton_residual = system_rhs.linfty_norm();

    if (newton_residual < ctl.params.lower_bound_newton_residual) {
      ctl.pcout << '\t' << std::scientific << newton_residual << std::endl;
      break;
    }

    if (newton_step == 1 ||
        newton_residual / old_newton_residual > nonlinear_rho)
      assemble_system(false, ctl);

    // Solve Ax = b
    no_linear_iterations = solve(ctl);

    line_search_step = 0;
    // Relaxation
    for (; line_search_step < ctl.params.max_no_line_search_steps;
         ++line_search_step) {
      distributed_solution -= increment;
      distribute_all_constraints(distributed_solution, ctl);
      solution = distributed_solution;
      assemble_system(true, ctl);
      new_newton_residual = system_rhs.linfty_norm();

      if (new_newton_residual < newton_residual)
        break;
      else {
        distributed_solution += increment;
        solution = distributed_solution;
      }

      increment *= ctl.params.line_search_damping;
    }
    old_newton_residual = newton_residual;
    newton_residual = new_newton_residual;

    ctl.pcout << std::setprecision(5) << newton_step << '\t' << std::scientific
              << newton_residual;

    ctl.pcout << '\t' << std::scientific
              << newton_residual / old_newton_residual << '\t';

    if (newton_step == 1 ||
        newton_residual / old_newton_residual > nonlinear_rho)
      ctl.pcout << "rebuild" << '\t';
    else
      ctl.pcout << " " << '\t';
    ctl.pcout << line_search_step << '\t' << std::scientific
              << no_linear_iterations << '\t' << std::scientific << std::endl;

    // Terminate if nothing is solved anymore. After this,
    // we cut the time step.
    if ((newton_residual / old_newton_residual >
         ctl.params.upper_newton_rho) &&
        (newton_step > 1)) {
      break;
    }

    // Updates
    newton_step++;
  }

  if ((newton_residual > ctl.params.lower_bound_newton_residual) &&
      (newton_step == ctl.params.max_no_newton_steps)) {
    ctl.pcout << "Newton iteration did not converge in " << newton_step
              << " steps :-(" << std::endl;
    throw SolverControl::NoConvergence(0, 0);
  }

  solution = distributed_solution;
  return newton_residual / old_newton_residual;
}

template <int dim>
void AbstractField<dim>::return_old_solution(Controller<dim> &ctl) {
  solution = old_solution;
}

template <int dim>
void AbstractField<dim>::record_old_solution(Controller<dim> &ctl) {
  old_solution = solution;
}

template <int dim>
void AbstractField<dim>::distribute_hanging_node_constraints(
    LA::MPI::Vector &vector, Controller<dim> &ctl) {
  constraints_hanging_nodes.distribute(vector);
}

template <int dim>
void AbstractField<dim>::distribute_all_constraints(LA::MPI::Vector &vector,
                                                 Controller<dim> &ctl) {
  constraints_all.distribute(vector);
}

#endif // CRACKS_ABSTRACT_FIELD_H
