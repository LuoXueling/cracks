//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_ELASTICITY_H
#define CRACKS_ELASTICITY_H

#include "abaqus_grid_in.h"
#include "abstract_field.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "dirichlet_boundary.h"
#include "parameters.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class Elasticity : public AbstractField<dim> {
public:
  Elasticity(Controller<dim> &ctl);

  void setup_system(Controller<dim> &ctl) override;
  void setup_boundary_condition(Controller<dim> &ctl) override;
  void assemble_system(bool residual_only, Controller<dim> &ctl) override;
  double newton_iteration(Controller<dim> &ctl) override;
  unsigned int solve(Controller<dim> &ctl) override;
  void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) override;
  void return_old_solution(Controller<dim> &ctl) override;
  void compute_load(Controller<dim> &ctl);
  void distribute_hanging_node_constraints(LA::MPI::Vector &vector,
                                           Controller<dim> &ctl);
  void distribute_all_constraints(LA::MPI::Vector &vector,
                                  Controller<dim> &ctl);

  ConstitutiveLaw<dim> constitutive_law;

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
Elasticity<dim>::Elasticity(Controller<dim> &ctl)
    : AbstractField<dim>(ctl),
      constitutive_law(ctl.params.E, ctl.params.v, ctl.params.plane_state),
      fe(FE_Q<dim>(ctl.params.poly_degree), dim),
      quadrature_formula(fe.degree + 1), dof_handler(ctl.triangulation) {}

template <int dim> void Elasticity<dim>::setup_system(Controller<dim> &ctl) {
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
    // system_rhs, system_matrix, and the solution vector increment do not have ghost elements
    increment.reinit(locally_owned_dofs, ctl.mpi_com);
    system_rhs.reinit(locally_owned_dofs, ctl.mpi_com);
    solution = 0;

    // Initialize fields. Trilino does not allow writing into its parallel
    // vector.
    //    VectorTools::interpolate(dof_handler, ZeroFunction<dim>(dim),
    //                             solution);
  }
}

template <int dim>
void Elasticity<dim>::setup_boundary_condition(Controller<dim> &ctl) {
  for (auto &face : ctl.triangulation.active_face_iterators())
    if (face->at_boundary()) {
      if (face->center()[1] <= 0.0) {
        face->set_boundary_id(1);
      } else if (face->center()[1] >= 1.0) {
        face->set_boundary_id(2);
      }
    }
  constraints_all.clear();
  constraints_all.reinit(locally_relevant_dofs);
  constraints_all.merge(constraints_hanging_nodes,
                        ConstraintMatrix::right_object_wins);
  VectorTools::interpolate_boundary_values(
      dof_handler, 1, ZeroFunction<dim>(dim), constraints_all, ComponentMask());
  VectorTools::interpolate_boundary_values(
      dof_handler, 2, IncrementalBoundaryValues<dim>(ctl.time), constraints_all,
      ComponentMask());
  constraints_all.close();
}

template <int dim>
void Elasticity<dim>::assemble_system(bool residual_only,
                                      Controller<dim> &ctl) {
  system_rhs = 0;
  system_matrix = 0;

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  const FEValuesExtractors::Vector displacement(0);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);

  // Old Newton grads
  std::vector<Tensor<2, dim>> old_displacement_grads(n_q_points);

  std::vector<Tensor<1, dim>> Nu_kq(dofs_per_cell);
  std::vector<Tensor<2, dim>> Bu_kq(dofs_per_cell);

  Tensor<2, dim> zero_matrix;
  zero_matrix.clear();

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values[displacement].get_function_values(solution,
                                                  old_displacement_values);
      fe_values[displacement].get_function_gradients(solution,
                                                     old_displacement_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nu_kq[k] = fe_values[displacement].value(k, q);
          Bu_kq[k] = fe_values[displacement].gradient(k, q);
        }
        const Tensor<2, dim> grad_u = old_displacement_grads[q];
        const double divergence_u = Tensors::get_divergence_u<dim>(grad_u);
        const Tensor<2, dim> Identity = Tensors ::get_Identity<dim>();
        const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);
        Tensor<2, dim> stress = constitutive_law.lambda * tr_E * Identity +
                                2 * constitutive_law.mu * E;

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          const Tensor<2, dim> Bu_iq_symmetric =
              0.5 * (Bu_kq[i] + transpose(Bu_kq[i]));
          const double Bu_iq_symmetric_tr = trace(Bu_iq_symmetric);

          if (!residual_only) {

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              const unsigned int comp_j = fe.system_to_component_index(j).first;
              {
                cell_matrix(i, j) +=
                    (scalar_product(constitutive_law.lambda *
                                            Bu_iq_symmetric_tr * Identity +
                                        2 * constitutive_law.mu *
                                            Bu_iq_symmetric,
                                    Bu_kq[j])) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) += (scalar_product(Bu_kq[i], stress) * fe_values.JxW(q));
        }
      }

      cell->get_dof_indices(local_dof_indices);
      if (residual_only) {
        constraints_all.distribute_local_to_global(cell_rhs, local_dof_indices,
                                                   system_rhs);
      } else {
        constraints_all.distribute_local_to_global(
            cell_matrix, local_dof_indices, system_matrix);
        constraints_all.distribute_local_to_global(cell_rhs, local_dof_indices,
                                                   system_rhs);
      }
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim> unsigned int Elasticity<dim>::solve(Controller<dim> &ctl) {

  SolverControl solver_control(dof_handler.n_dofs(),
                               1e-8 * system_rhs.l2_norm());
  SolverGMRES<LA::MPI::Vector> solver(solver_control);
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.constant_modes = constant_modes;
    data.elliptic = true;
    data.higher_order_elements = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
    preconditioner.initialize(system_matrix, data);
  }

  solver.solve(system_matrix, increment, system_rhs, preconditioner);

  //   ctl.pcout << "   Solved in " << solver_control.last_step() << "
  //   iterations."
  //        << std::endl;
}

template <int dim>
void Elasticity<dim>::output_results(DataOut<dim> &data_out,
                                     Controller<dim> &ctl) {
  std::vector<std::string> solution_names(dim, "displacement");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(dof_handler, solution, solution_names,
                           data_component_interpretation);

  // Record statistics
  compute_load(ctl);
}

template <int dim> void Elasticity<dim>::compute_load(Controller<dim> &ctl) {
  const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<2, dim>> face_solution_grads(n_face_q_points);

  Tensor<1, dim> load_value;

  const Tensor<2, dim> Identity = Tensors::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  const FEValuesExtractors::Vector displacement(0);

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 2) {
          fe_face_values.reinit(cell, face);
          fe_face_values[displacement].get_function_gradients(
              solution, face_solution_grads);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
            const Tensor<2, dim> grad_u = face_solution_grads[q_point];

            const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
            const double tr_E = trace(E);

            Tensor<2, dim> stress_term;
            stress_term = constitutive_law.lambda * tr_E * Identity +
                          2 * constitutive_law.mu * E;

            load_value += stress_term * fe_face_values.normal_vector(q_point) *
                          fe_face_values.JxW(q_point);
          }
        } // end boundary 3 for structure
    }

  load_value[0] *= -1.0;

  double load_y = Utilities::MPI::sum(load_value[1], ctl.mpi_com);
  ctl.pcout << "  Load y: " << load_y;
  (ctl.statistics).add_value("Load y", load_y);
  (ctl.statistics).set_precision("Load y", 8);
  (ctl.statistics).set_scientific("Load y", true);
}

template <int dim>
void Elasticity<dim>::return_old_solution(Controller<dim> &ctl) {
  solution = old_solution;
}

template <int dim>
void Elasticity<dim>::distribute_hanging_node_constraints(
    LA::MPI::Vector &vector, Controller<dim> &ctl) {
  constraints_hanging_nodes.distribute(vector);
}

template <int dim>
void Elasticity<dim>::distribute_all_constraints(LA::MPI::Vector &vector,
                                                 Controller<dim> &ctl) {
  constraints_all.distribute(vector);
}

template <int dim>
double Elasticity<dim>::newton_iteration(Controller<dim> &ctl) {
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

  while (newton_residual > (ctl.params).lower_bound_newton_residual &&
         newton_step < (ctl.params).max_no_newton_steps) {
    old_newton_residual = newton_residual;

    assemble_system(true, ctl);
    newton_residual = system_rhs.linfty_norm();

    if (newton_residual < (ctl.params).lower_bound_newton_residual) {
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
    for (; line_search_step < (ctl.params).max_no_line_search_steps;
         ++line_search_step) {
      distributed_solution -= increment;
      distribute_all_constraints(distributed_solution, ctl);
      solution = distributed_solution;
      assemble_system(true, ctl);
      new_newton_residual = system_rhs.linfty_norm();

      if (new_newton_residual < newton_residual)
        break;
      else
      {distributed_solution += increment;
        solution = distributed_solution;}

      increment *= (ctl.params).line_search_damping;
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
         (ctl.params).upper_newton_rho) &&
        (newton_step > 1)) {
      break;
    }

    // Updates
    newton_step++;
  }

  if ((newton_residual > (ctl.params).lower_bound_newton_residual) &&
      (newton_step == (ctl.params).max_no_newton_steps)) {
    ctl.pcout << "Newton iteration did not converge in " << newton_step
              << " steps :-(" << std::endl;
    throw SolverControl::NoConvergence(0, 0);
  }

  solution = distributed_solution;
  return newton_residual / old_newton_residual;
}

#endif // CRACKS_ELASTICITY_H
