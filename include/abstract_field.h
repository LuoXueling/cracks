//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_ABSTRACT_FIELD_H
#define CRACKS_ABSTRACT_FIELD_H

#include "boundary.h"
#include "controller.h"
#include "dealii_includes.h"
#include "newton_variations.h"
#include <typeinfo>

template <int dim> class AbstractField {
public:
  AbstractField(unsigned int n_components, std::string boundary_from,
                std::string update_scheme, Controller<dim> &ctl);

  virtual void assemble_newton_system(bool residual_only,
                                      LA::MPI::Vector &neumann_rhs,
                                      Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void assemble_linear_system(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual unsigned int solve(NewtonInformation<dim> &info,
                             Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void define_boundary_condition(std::string boundary_from,
                                         Controller<dim> &ctl);
  virtual void setup_dirichlet_boundary_condition(Controller<dim> &ctl);
  virtual void setup_neumann_boundary_condition(
      std::tuple<unsigned int, std::string, std::vector<double>,
                 std::vector<double>>
          neumann_info,
      LA::MPI::Vector &neumann_rhs, Controller<dim> &ctl);
  virtual void setup_system(Controller<dim> &ctl);
  virtual void record_old_solution(Controller<dim> &ctl);
  virtual void return_old_solution(Controller<dim> &ctl);
  virtual void distribute_hanging_node_constraints(LA::MPI::Vector &vector,
                                                   Controller<dim> &ctl);
  virtual void distribute_all_constraints(LA::MPI::Vector &vector,
                                          Controller<dim> &ctl);

  virtual double update(Controller<dim> &ctl);
  virtual double update_linear_system(Controller<dim> &ctl);
  virtual double update_newton_system(Controller<dim> &ctl);
  parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
  prepare_refine();
  void post_refine(
      parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> &soltrans,
      Controller<dim> &ctl);
  /*
   * Solver
   */
  std::string update_scheme_timestep;
  LA::MPI::PreconditionAMG preconditioner;
  std::vector<std::vector<bool>> constant_modes;

  /*
   * FE system, constraints, and dof handler
   */
  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  AffineConstraints<double> constraints_hanging_nodes;
  AffineConstraints<double> constraints_all;

  std::vector<ComponentMask> component_masks;
  std::vector<std::tuple<unsigned int, std::string, unsigned int, double,
                         std::vector<double>>>
      dirichlet_boundary_info;
  std::vector<std::tuple<unsigned int, std::string, std::vector<double>,
                         std::vector<double>>>
      neumann_boundary_info;
  /*
   * Solutions
   */
  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector solution, newton_update, old_solution;
  //  LA::MPI::Vector system_total_residual;
  LA::MPI::Vector system_rhs;
  //  LA::MPI::Vector diag_mass, diag_mass_relevant;
  LA::MPI::Vector system_solution;

  SolverControl direct_solver_control;
  TrilinosWrappers::SolverDirect direct_solver;

  std::unique_ptr<NewtonVariation<dim>> newton_ctl;
};

template <int dim>
AbstractField<dim>::AbstractField(const unsigned int n_components,
                                  std::string boundary_from,
                                  std::string update_scheme,
                                  Controller<dim> &ctl)
    : fe(FE_Q<dim>(ctl.params.poly_degree), n_components),
      dof_handler(ctl.triangulation), update_scheme_timestep(update_scheme),
      direct_solver(direct_solver_control) {
  for (unsigned int d = 0; d < n_components; ++d) {
    component_masks.push_back(ComponentMask(n_components, false));
    component_masks[d].set(d, true);
  }
  newton_ctl = select_newton_variation<dim>(ctl.params.adjustment_method, ctl);
  define_boundary_condition(boundary_from, ctl);
}

template <int dim> void AbstractField<dim>::setup_system(Controller<dim> &ctl) {
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
    setup_dirichlet_boundary_condition(ctl);
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
    // system_rhs, system_matrix, and the solution vector system_solution do not
    // have ghost elements
    system_solution.reinit(locally_owned_dofs, ctl.mpi_com);
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

template <int dim> double AbstractField<dim>::update(Controller<dim> &ctl) {
  if (update_scheme_timestep == "linear") {
    update_linear_system(ctl);
    return 0.0;
  } else if (update_scheme_timestep == "newton") {
    return update_newton_system(ctl);
  } else {
    AssertThrow(false, ExcNotImplemented());
  }
}

template <int dim>
void AbstractField<dim>::define_boundary_condition(
    const std::string boundary_from, Controller<dim> &ctl) {
  if (boundary_from == "none") {
    return;
  }
  std::filebuf fb;
  if (fb.open(boundary_from, std::ios::in)) {
    std::istream is(&fb);
    std::string line;
    unsigned int boundary_id, constrained_dof;
    std::string constraint_type;
    double constraint_value;
    while (std::getline(is, line)) {
      if (line[0] == '#') {
        continue;
      }
      std::istringstream iss(line);
      iss >> boundary_id >> constraint_type;
      if (constraint_type == "velocity" || constraint_type == "dirichlet" ||
          constraint_type == "triangulardirichlet" ||
          constraint_type == "sinedirichlet") {
        std::vector<double> additional_info;
        if (constraint_type == "velocity" || constraint_type == "dirichlet") {
          iss >> constrained_dof >> constraint_value;
        } else {
          iss >> constrained_dof;
          constraint_value = 0.0;
          double temp_value;
          do {
            iss >> temp_value;
            additional_info.push_back(temp_value);
          } while (!iss.eof());
        }
        std::tuple<unsigned int, std::string, unsigned int, double,
                   std::vector<double>>
            info(boundary_id, constraint_type, constrained_dof,
                 constraint_value, additional_info);
        dirichlet_boundary_info.push_back(info);
      } else if (constraint_type == "neumann" ||
                 constraint_type == "neumannrate" ||
                 constraint_type == "sineneumann" ||
                 constraint_type == "triangularneumann") {
        std::vector<double> constraint_vector;
        std::vector<double> additional_info;
        double temp_value;
        do {
          iss >> temp_value;
          if (constraint_vector.size() < fe.n_components()) {
            constraint_vector.push_back(temp_value);
          } else {
            additional_info.push_back(temp_value);
          }
        } while (!iss.eof());
        std::tuple<unsigned int, std::string, std::vector<double>,
                   std::vector<double>>
            info(boundary_id, constraint_type, constraint_vector,
                 additional_info);
        neumann_boundary_info.push_back(info);
      } else {
        AssertThrow(false, ExcNotImplemented(constraint_type));
      }
    }
    fb.close();
  }
}

template <int dim>
void AbstractField<dim>::setup_dirichlet_boundary_condition(
    Controller<dim> &ctl) {
  // Dealing with dirichlet boundary conditions
  constraints_all.clear();
  constraints_all.reinit(locally_relevant_dofs);
  constraints_all.merge(constraints_hanging_nodes,
                        ConstraintMatrix::right_object_wins);
  for (const std::tuple<unsigned int, std::string, unsigned int, double,
                        std::vector<double>> &info : dirichlet_boundary_info) {
    ctl.debug_dcout << "Setting dirichlet boundary" << std::endl;
    std::unique_ptr<Function<dim>> dirichlet_boundary =
        select_dirichlet_boundary<dim>(info, fe.n_components(), ctl.time);
    VectorTools::interpolate_boundary_values(
        dof_handler, std::get<0>(info), *dirichlet_boundary, constraints_all,
        component_masks[std::get<2>(info)]);
  }
  constraints_all.close();
}

template <int dim>
void AbstractField<dim>::setup_neumann_boundary_condition(
    std::tuple<unsigned int, std::string, std::vector<double>,
               std::vector<double>>
        neumann_info,
    LA::MPI::Vector &neumann_rhs, Controller<dim> &ctl) {
  const QGauss<dim - 1> face_quadrature_formula(ctl.params.poly_degree + 1);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                       update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>> neumann_values(n_face_q_points);
  // Vector<double> cannot be automatically initialized like Tensor
  for (unsigned int j = 0; j < n_face_q_points; ++j) {
    neumann_values[j].reinit(fe.n_components());
  }

  unsigned int boundary_id = std::get<0>(neumann_info);

  std::unique_ptr<GeneralNeumannBoundary<dim>> neumann_boundary =
      select_neumann_boundary<dim>(neumann_info, fe.n_components(), ctl.time);

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      for (const auto &face : cell->face_iterators()) {
        if (face->at_boundary() && face->boundary_id() == boundary_id) {
          cell_rhs = 0;
          fe_face_values.reinit(cell, face);
          neumann_boundary->vector_value_list(
              fe_face_values.get_quadrature_points(), neumann_values);
          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const unsigned int comp_i = fe.system_to_component_index(i).first;
              cell_rhs(i) +=
                  (fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
                   neumann_values[q_point][comp_i] *        // g(x_q)
                   fe_face_values.JxW(q_point));            // dx
            }
          }
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            neumann_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
        }
      }
    }
}

template <int dim>
double AbstractField<dim>::update_linear_system(Controller<dim> &ctl) {
  // Cannot distribute constraints to parallel vectors with ghost dofs.
  LA::MPI::Vector distributed_solution(locally_owned_dofs, ctl.mpi_com);
  distributed_solution = solution;

  ctl.debug_dcout << "Solve linear system - initialize" << std::endl;
  setup_dirichlet_boundary_condition(ctl);
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;
  ctl.debug_dcout << "Solve linear system - assemble" << std::endl;
  assemble_linear_system(ctl);

  ctl.debug_dcout << "Solve linear system - solve" << std::endl;
  solve(ctl);
  ctl.debug_dcout << "Solve linear system - constraints" << std::endl;
  distributed_solution = system_solution;
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;

  return 0.0;
}

template <int dim>
double AbstractField<dim>::update_newton_system(Controller<dim> &ctl) {
  ctl.dcout << "It.\tResidual\tReduction\t#LinIts" << std::endl;

  // Cannot distribute constraints to parallel vectors with ghost dofs.
  LA::MPI::Vector distributed_solution(locally_owned_dofs, ctl.mpi_com);
  distributed_solution = solution;

  // Application of the initial boundary conditions to the
  // variational equations:
  ctl.debug_dcout << "Solve Newton system - Newton iteration - initialize"
                  << std::endl;
  setup_dirichlet_boundary_condition(ctl);
  distribute_all_constraints(distributed_solution, ctl);
  solution = distributed_solution;
  ctl.debug_dcout
      << "Solve Newton system - Newton iteration - first residual assemble"
      << std::endl;
  LA::MPI::Vector neumann_rhs = system_rhs;
  neumann_rhs = 0;
  assemble_newton_system(true, neumann_rhs, ctl);

  NewtonInformation<dim> newton_info;
  newton_info.residual = system_rhs.linfty_norm();
  newton_info.old_residual = newton_info.residual * 1e8;
  newton_info.i_step = 1;
  newton_info.iterative_solver_nonlinear_step = 0;
  newton_info.adjustment_step = 0;
  newton_info.new_residual = 0.0;
  newton_info.system_matrix_rebuilt = false;

  ctl.dcout << "0\t" << std::scientific << newton_info.residual << std::endl;

  while (newton_info.residual > ctl.params.lower_bound_newton_residual &&
         newton_info.i_step < ctl.params.max_no_newton_steps) {
    if (newton_ctl->quit_newton(newton_info, ctl)) {
      ctl.dcout << '\t' << std::scientific << newton_info.residual << std::endl;
      break;
    }

    if (newton_ctl->rebuild_jacobian(newton_info, ctl)) {
      ctl.debug_dcout
          << "Solve Newton system - Newton iteration - system assemble"
          << std::endl;
      assemble_newton_system(false, neumann_rhs, ctl);
      newton_info.system_matrix_rebuilt = true;
    }

    // Solve Ax = b
    ctl.debug_dcout
        << "Solve Newton system - Newton iteration - solve linear system"
        << std::endl;
    newton_info.iterative_solver_nonlinear_step = solve(ctl);
    newton_info.system_matrix_rebuilt = false;
    ctl.debug_dcout
        << "Solve Newton system - Newton iteration - solve linear system exit"
        << std::endl;
    newton_info.adjustment_step = 0;
    // Relaxation
    for (; newton_info.adjustment_step < ctl.params.max_adjustment_steps;) {
      ctl.debug_dcout
          << "Solve Newton system - Newton iteration - dealing solution"
          << std::endl;
      newton_info.new_residual = system_rhs.linfty_norm();
      newton_ctl->apply_increment(system_solution, distributed_solution,
                                  this->system_matrix, this->system_rhs,
                                  neumann_rhs, newton_info, ctl);
      ctl.debug_dcout << "Solve Newton system - Newton iteration - distribute"
                      << std::endl;
      distribute_all_constraints(distributed_solution, ctl);
      solution = distributed_solution;
      ctl.debug_dcout << "Solve Newton system - Newton iteration - "
                         "residual assemble"
                      << std::endl;
      if (newton_ctl->re_solve(newton_info, ctl)) {
        if (newton_ctl->rebuild_jacobian(newton_info, ctl)) {
          ctl.debug_dcout << "Solve Newton system - Newton iteration - resolve "
                             "- system assemble"
                          << std::endl;
          assemble_newton_system(false, neumann_rhs, ctl);
          newton_info.system_matrix_rebuilt = true;
        }
        ctl.debug_dcout << "Solve Newton system - Newton iteration - resolve"
                        << std::endl;
        newton_info.iterative_solver_nonlinear_step = solve(ctl);
        newton_info.system_matrix_rebuilt = false;
      } else {
        ctl.debug_dcout
            << "Solve Newton system - Newton iteration - residual assemble"
            << std::endl;
        assemble_newton_system(true, neumann_rhs, ctl);
      }

      newton_info.new_residual = system_rhs.linfty_norm();

      if (newton_ctl->quit_adjustment(newton_info, ctl)) {
        ctl.debug_dcout
            << "Solve Newton system - Newton iteration - stop adjustment"
            << std::endl;
        break;
      } else {
        newton_info.adjustment_step += 1;
        ctl.debug_dcout
            << "Solve Newton system - Newton iteration - next adjustment"
            << std::endl;
        newton_ctl->prepare_next_adjustment(
            system_solution, distributed_solution, this->system_matrix,
            this->system_rhs, neumann_rhs, newton_info, ctl);
        distribute_all_constraints(distributed_solution, ctl);
        solution = distributed_solution;
      }
    }
    newton_info.old_residual = newton_info.residual;
    newton_info.residual = newton_info.new_residual;

    ctl.dcout << std::setprecision(-1) << std::defaultfloat
              << newton_info.i_step << '\t' << std::setprecision(5)
              << std::scientific << newton_info.residual;

    ctl.dcout << '\t' << std::scientific
              << newton_info.residual / newton_info.old_residual << '\t';

    ctl.dcout << newton_info.adjustment_step << '\t' << std::scientific
              << newton_info.iterative_solver_nonlinear_step << '\t'
              << std::scientific << std::endl;

    // Terminate if nothing is solved anymore. After this,
    // we cut the time step.
    if (newton_ctl->give_up(newton_info, ctl)) {
      ctl.dcout << "Newton iteration did not converge in " << newton_info.i_step
                << " steps. Go to adaptive time stepping" << std::endl;
      throw SolverControl::NoConvergence(0, 0);
    }

    // Updates
    newton_info.i_step++;
  }

  solution = distributed_solution;
  return newton_info.residual / newton_info.old_residual;
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

template <int dim>
parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
AbstractField<dim>::prepare_refine() {
  parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> soltrans(
      dof_handler);
  soltrans.prepare_for_coarsening_and_refinement(solution);
  return soltrans;
}

template <int dim>
void AbstractField<dim>::post_refine(
    parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> &soltrans,
    Controller<dim> &ctl) {
  LA::MPI::Vector interpolated_solution;
  interpolated_solution.reinit(locally_owned_dofs, ctl.mpi_com);
  soltrans.interpolate(interpolated_solution);
  solution = interpolated_solution;
  record_old_solution(ctl);
}

#endif // CRACKS_ABSTRACT_FIELD_H
