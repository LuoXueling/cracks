/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PHASE_FIELD_FRACTURE_H
#define PHASE_FIELD_FRACTURE_H

#include "abaqus_grid_in.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "dirichlet_boundary.h"
#include "multi_field.h"
#include "parameters.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseFieldFracture {
public:
  PhaseFieldFracture(Parameters::AllParameters &);

  void run();

private:
  void setup_mesh();
  void setup_system();
  void setup_boundary_condition(double time);
  void assemble_system(bool residual_only = false);
  void enforce_phase_field_limitation();
  unsigned int solve();
  void refine_grid();
  double newton_iteration(double time);
  void output_results(const unsigned int cycle);
  void compute_load();

  MPI_Comm mpi_com;

  parallel::distributed::Triangulation<dim> triangulation;
  Parameters::AllParameters params;

  /*
   * Preconditioners
   */
  LA::MPI::PreconditionAMG preconditioner_u;
  std::vector<std::vector<bool>> constant_modes;


  /*
   * FE system, constraints, and dof handler
   */
  FESystem<dim> fe_u;
  const QGauss<dim> quadrature_formula_u;
  //  FE_Q<dim> fe_phi;
  DoFHandler<dim> dof_handler_u;
  IndexSet locally_owned_dofs_u;
  IndexSet locally_relevant_dofs_u;
  //  DoFHandler<dim> dof_handler_phi;
  AffineConstraints<double> constraints_hanging_nodes_u;
  AffineConstraints<double> constraints_all_u;

  /*
   * Solutions
   */
  LA::MPI::SparseMatrix system_matrix_u;
  LA::MPI::Vector solution_u, newton_update_u, old_solution_u,
      old_old_solution_u;
  //  LA::MPI::Vector system_total_residual;
  LA::MPI::Vector system_rhs_u;
  //  LA::MPI::Vector diag_mass, diag_mass_relevant;
  LA::MPI::Vector incremental_u;

  FE_Q<dim> fe_phi;
  const QGauss<dim> quadrature_formula_phi;
  //  FE_Q<dim> fe_phi;
  DoFHandler<dim> dof_handler_phi;
  //  DoFHandler<dim> dof_handler_phi;
  AffineConstraints<double> constraints_hanging_nodes_phi;

  LA::MPI::PreconditionAMG preconditioner_phase_field;
  LA::MPI::SparseMatrix system_matrix_phi;
  LA::MPI::Vector solution_phi, newton_update_phi, old_solution_phi,
      old_old_solution_phi;
  //  LA::MPI::Vector system_total_residual;
  LA::MPI::Vector system_rhs_phi;
  //  LA::MPI::Vector diag_mass, diag_mass_relevant;
  LA::MPI::Vector incremental_phi;

  IndexSet locally_owned_dofs_phi;
  IndexSet locally_relevant_dofs_phi;


  ConstitutiveLaw<dim> constitutive_law;

  ConditionalOStream pcout;
  DualOStream dcout;
  DualTimerOutput timer;
  TimerOutput computing_timer;

  double time;
  unsigned int timestep_number;
  double current_timestep;
  double old_timestep, old_old_timestep;

  TableHandler statistics;
};

template <int dim>
PhaseFieldFracture<dim>::PhaseFieldFracture(Parameters::AllParameters &prms)
    : mpi_com(MPI_COMM_WORLD), params(prms),
      triangulation(mpi_com, typename Triangulation<dim>::MeshSmoothing(
                                 Triangulation<dim>::smoothing_on_refinement |
                                 Triangulation<dim>::smoothing_on_coarsening)),
      fe_u(FE_Q<dim>(params.poly_degree), dim), fe_phi(params.poly_degree),
      quadrature_formula_u(fe_u.degree + 1), dof_handler_u(triangulation),
      quadrature_formula_phi(fe_phi.degree + 1), dof_handler_phi(triangulation),
      constitutive_law(params.E, params.v, params.plane_state),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      dcout(pcout, params.log_file), timer(mpi_com, pcout, TimerOutput::never,
                                           TimerOutput::cpu_and_wall_times),
      computing_timer(mpi_com, pcout, TimerOutput::never,
                      TimerOutput::wall_times),
      time(0), timestep_number(0) {
  statistics.set_auto_fill_mode(true);
}

template <int dim> void PhaseFieldFracture<dim>::run() {
  dcout << "Project: " << params.project_name << std::endl;
  dcout << "Mesh from: " << params.mesh_from << std::endl;
  dcout << "Load sequence from: " << params.load_sequence_from << std::endl;
  dcout << "Output directory: " << params.output_dir << std::endl;
  dcout << "Solving " << params.dim << " dimensional PFM problem" << std::endl;
  dcout << "Running on " << Utilities::MPI::n_mpi_processes(mpi_com)
        << " MPI rank(s)" << std::endl;
  dcout << "Number of threads " << MultithreadInfo::n_threads() << std::endl;
  dcout << "Number of cores " << MultithreadInfo::n_cores() << std::endl;

  setup_mesh();
  setup_system();

  // Initialize fields
  VectorTools::interpolate(dof_handler_u, InitialValues<dim>(dim), solution_u);

  //  if (params.enable_phase_field) {
  //    enforce_phase_field_limitation();
  //  }

  unsigned int refinement_cycle = 0;
  double finishing_timestep_loop = 0;
  double tmp_timestep = 0.0;

  // Initialize old and old_old_solutions
  // old_old is needed for extrapolation for pf_extra to avoid pf^2 in
  // block(0,0)
  old_old_solution_u = solution_u;
  old_solution_u = solution_u;

  current_timestep = params.timestep;
  // Initialize old and old_old timestep sizes
  old_timestep = current_timestep;
  old_old_timestep = current_timestep;

  do {
    double newton_reduction = 1.0;
    if (timestep_number > params.switch_timestep && params.switch_timestep > 0)
      current_timestep = params.timestep_size_2;

    double tmp_current_timestep = current_timestep;
    old_old_timestep = old_timestep;
    old_timestep = current_timestep;
    // Initialize previous solutions
    old_old_solution_u = old_solution_u;
    old_solution_u = solution_u;

  mesh_refine_checkpoint:
    pcout << std::endl;
    pcout << "\n=============================="
          << "=========================================" << std::endl;
    pcout << "Time " << timestep_number << ": " << time << " ("
          << current_timestep << ")" << "   "
          << "Cells: " << triangulation.n_global_active_cells() << "   "
          << "Displacement DoFs: " << dof_handler_u.n_dofs();
    pcout << "\n--------------------------------"
          << "---------------------------------------" << std::endl;
    pcout << std::endl;

    time += current_timestep;

    do {
      // The Newton method can either stagnate or the linear solver
      // might not converge. To not abort the program we catch the
      // exception and retry with a smaller step.
      //          use_old_timestep_pf = false;
      try {
        // Normalize phase-field function between 0 and 1
        if (params.enable_phase_field) {
          enforce_phase_field_limitation();
        }
        newton_reduction = newton_iteration(time);

        while (newton_reduction > params.upper_newton_rho) {
          //              use_old_timestep_pf = true;
          time -= current_timestep;
          current_timestep = current_timestep / 10.0;
          time += current_timestep;
          solution_u = old_solution_u;
          newton_reduction = newton_iteration(time);

          if (current_timestep < 1.0e-9) {
            pcout << "Step size too small - keeping the step size" << std::endl;
            break;
          }
        }

        break;

      } catch (SolverControl::NoConvergence &e) {
        pcout << "Solver did not converge! Adjusting time step." << std::endl;
      }

      time -= current_timestep;
      solution_u = old_solution_u;
      current_timestep = current_timestep / 10.0;
      time += current_timestep;
    } while (true);
    //    if (params.enable_phase_field) {
    //      enforce_phase_field_limitation();
    //    }
    constraints_hanging_nodes_u.distribute(solution_u);

    // Refine mesh and return to the beginning if mesh is changed.
    //    if (params.refine) {
    //      bool changed = refine_grid();
    //      if (changed) {
    //        // redo the current time step
    //        pcout << "Mesh changed! Re-do the current time step" << std::endl;
    //        time -= current_timestep;
    //        solution = old_solution;
    //        goto mesh_refine_checkpoint;
    //        continue;
    //      }
    //    }

    // Recover time step
    current_timestep = tmp_current_timestep;

    output_results(timestep_number);

    ++timestep_number;

    computing_timer.print_summary();
    computing_timer.reset();
    pcout << std::endl;
  } while (timestep_number <= params.max_no_timesteps);
  timer.manual_print_summary(dcout.fout);
}

template <int dim> void PhaseFieldFracture<dim>::setup_mesh() {
  timer.enter_subsection("Setup mesh");
  //  AbaqusGridIn<dim> grid_in;
  //  /**
  //   * similar to normal use of GridIn.
  //   */
  //  grid_in.attach_triangulation(triangulation);
  //  if (!checkFileExsit(params.mesh_from)) {
  //    throw std::runtime_error("Mesh file does not exist");
  //  }
  //  grid_in.read_abaqus_inp(params.mesh_from);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(8);

  //  if (dim == 2) {
  //    std::ofstream out(params.output_dir + "initial_grid.svg");
  //    GridOut grid_out;
  //    grid_out.write_svg(triangulation, out);
  //  }
  //  dcout << "Find " << triangulation.n_global_active_cells() << " elements"
  //        << std::endl;

  timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::setup_system() {
  timer.enter_subsection("Setup system");
  TimerOutput::Scope t(computing_timer, "Setup system");

  system_matrix_u.clear();
  system_matrix_phi.clear();
  /**
   * DOF
   **/
  {
    dof_handler_u.distribute_dofs(fe_u);
    locally_owned_dofs_u = dof_handler_u.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_u,
                                            locally_relevant_dofs_u);
    constant_modes.clear();
    DoFTools::extract_constant_modes(dof_handler_u, ComponentMask(),
                                     constant_modes);
  }
  {
    dof_handler_phi.distribute_dofs(fe_phi);
    locally_owned_dofs_phi = dof_handler_phi.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_phi,
                                            locally_relevant_dofs_phi);
  }
  /**
   * Hanging node and boundary value constraints
   */
  {
    constraints_hanging_nodes_u.clear();
    constraints_hanging_nodes_u.reinit(locally_relevant_dofs_u);
    DoFTools::make_hanging_node_constraints(dof_handler_u,
                                            constraints_hanging_nodes_u);
    constraints_hanging_nodes_u.close();

    constraints_all_u.clear();
    constraints_all_u.reinit(locally_relevant_dofs_u);
    setup_boundary_condition(0.0);
    constraints_all_u.close();
  }
  {
    constraints_hanging_nodes_phi.clear();
    constraints_hanging_nodes_phi.reinit(locally_relevant_dofs_phi);
    DoFTools::make_hanging_node_constraints(dof_handler_phi,
                                            constraints_hanging_nodes_phi);
    constraints_hanging_nodes_phi.close();
  }

  /**
   * Sparsity pattern
   */
  {
    DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs_u);
    DoFTools::make_sparsity_pattern(dof_handler_u, sparsity_pattern,
                                    constraints_all_u,
                                    /*keep constrained dofs*/ false);
    SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                               locally_owned_dofs_u, mpi_com,
                                               locally_relevant_dofs_u);
    sparsity_pattern.compress();
    system_matrix_u.reinit(locally_owned_dofs_u, locally_owned_dofs_u,
                           sparsity_pattern, mpi_com);
  }

  {
    DynamicSparsityPattern dsp(locally_relevant_dofs_phi);

    DoFTools::make_sparsity_pattern(dof_handler_phi, dsp,
                                    constraints_hanging_nodes_phi, false);
    SparsityTools::distribute_sparsity_pattern(
        dsp, dof_handler_phi.locally_owned_dofs(), mpi_com,
        locally_relevant_dofs_phi);

    system_matrix_phi.reinit(locally_owned_dofs_phi, locally_owned_dofs_phi,
                             dsp, mpi_com);
  }

  /**
   * Initialize solution
   */
  {
    solution_u.reinit(locally_owned_dofs_u, locally_relevant_dofs_u, mpi_com);
    system_rhs_u.reinit(locally_owned_dofs_u, mpi_com);
    incremental_u = solution_u;
  }

  {
    solution_phi.reinit(locally_owned_dofs_phi, locally_relevant_dofs_phi,
                        mpi_com);
    system_rhs_phi.reinit(locally_owned_dofs_phi, mpi_com);
    //    incremental_phi.reinit(dof_handler_phi.n_dofs());
  }

  timer.leave_subsection();
}

template <int dim>
void PhaseFieldFracture<dim>::setup_boundary_condition(double time) {
  for (auto &face : triangulation.active_face_iterators())
    if (face->at_boundary()) {
      if (face->center()[1] <= 0.0) {
        face->set_boundary_id(1);
      } else if (face->center()[1] >= 1.0) {
        face->set_boundary_id(2);
      }
    }
  constraints_all_u.clear();
  constraints_all_u.reinit(locally_relevant_dofs_u);
  constraints_all_u.merge(constraints_hanging_nodes_u,
                          ConstraintMatrix::right_object_wins);
  VectorTools::interpolate_boundary_values(dof_handler_u, 1,
                                           ZeroFunction<dim>(dim),
                                           constraints_all_u, ComponentMask());
  VectorTools::interpolate_boundary_values(dof_handler_u, 2,
                                           IncrementalBoundaryValues<dim>(time),
                                           constraints_all_u, ComponentMask());
  constraints_all_u.close();
}

template <int dim>
void PhaseFieldFracture<dim>::assemble_system(bool residual_only) {
  timer.enter_subsection("Assemble system");
  TimerOutput::Scope t(computing_timer, "Assemble system");

  system_rhs_u = 0;
  system_matrix_u = 0;

  FEValues<dim> fe_values_u(fe_u, quadrature_formula_u,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell_u = fe_u.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula_u.size();

  FullMatrix<double> cell_matrix(dofs_per_cell_u, dofs_per_cell_u);
  Vector<double> cell_rhs(dofs_per_cell_u);

  const FEValuesExtractors::Vector displacement(0);
  //  std::vector<Tensor<1, dim>> old_displacement_grads(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_u);

  LA::MPI::Vector rel_solution = solution_u;

  // Old Newton values
  std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
  //  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
  //  std::vector<double> old_phase_field_values(n_q_points);

  // Old Newton grads
  std::vector<Tensor<2, dim>> old_displacement_grads(n_q_points);
  //  std::vector<Tensor<1, dim>> old_phase_field_grads(n_q_points);

  // Old timestep values
  //  std::vector<Tensor<1, dim>> old_timestep_displacement_values(n_q_points);
  //  std::vector<double> old_timestep_phase_field_values(n_q_points);
  //  std::vector<Tensor<1, dim>> old_timestep_velocity_values(n_q_points);

  //  std::vector<Tensor<1, dim>>
  //  old_old_timestep_displacement_values(n_q_points); std::vector<double>
  //  old_old_timestep_phase_field_values(n_q_points);

  // Declaring test functions:
  std::vector<Tensor<1, dim>> Nu_kq(dofs_per_cell_u);
  std::vector<Tensor<2, dim>> Bu_kq(dofs_per_cell_u);
  //  std::vector<double> phi_i_pf(dofs_per_cell);
  //  std::vector<Tensor<1, dim>> phi_i_grads_pf(dofs_per_cell);

  Tensor<2, dim> zero_matrix;
  zero_matrix.clear();

  for (const auto &cell : dof_handler_u.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values_u.reinit(cell);

      fe_values_u[displacement].get_function_values(rel_solution,
                                                    old_displacement_values);
      fe_values_u[displacement].get_function_gradients(rel_solution,
                                                       old_displacement_grads);

      //      dcout << strain_tensor << std::endl;
      // Old Newton iteration values
      //      fe_values_u.get_function_values(rel_solution,
      //      old_displacement_values); fe_values_phi.get_function_values(
      //          rel_solution, old_phase_field_values);
      //      fe_values_u.get_function_gradients(rel_solution,
      //      old_displacement_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell_u; ++k) {
          Nu_kq[k] = fe_values_u[displacement].value(k, q);
          Bu_kq[k] = fe_values_u[displacement].gradient(k, q);
        }
        const Tensor<2, dim> grad_u = old_displacement_grads[q];
        const double divergence_u = Tensors::get_divergence_u<dim>(grad_u);
        const Tensor<2, dim> Identity = Tensors ::get_Identity<dim>();
        const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);
        Tensor<2, dim> stress = constitutive_law.lambda * tr_E * Identity +
                                2 * constitutive_law.mu * E;
        // First, we prepare things coming from the previous Newton
        // iteration...
        //        double pf = old_phase_field_values[q];
        //        double old_timestep_pf = old_timestep_phase_field_values[q];
        //        double old_old_timestep_pf =
        //        old_old_timestep_phase_field_values[q]; pf = std::max(0.0,
        //        old_phase_field_values[q]); old_timestep_pf = std::max(0.0,
        //        old_timestep_phase_field_values[q]); old_old_timestep_pf =
        //              std::max(0.0, old_old_timestep_phase_field_values[q]);

        //        const Tensor<2, dim> grad_u = old_displacement_grads[q];
        ////        const Tensor<1, dim> grad_pf = old_phase_field_grads[q];
        //        // \nabla \cdot u
        //        const double divergence_u =
        //        Tensors::get_divergence_u<dim>(grad_u); const Tensor<2, dim>
        //        Identity = Tensors::get_Identity<dim>();
        //        // total strain
        //        const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
        //        const double tr_E = trace(E);

        //        Tensor<2, dim> stress_term_plus;
        //        Tensor<2, dim> stress_term_minus;
        //        if (params.decompose_stress_rhs_u == "spectral" &&
        //            timestep_number > 0) {
        //          decompose_stress(stress_term_plus, stress_term_minus, E,
        //          tr_E,
        //                           zero_matrix, 0.0, lame_coefficient_lambda,
        //                           lame_coefficient_mu, false);
        //        } else {
        //          stress_term_plus = lame_coefficient_lambda * tr_E * Identity
        //          +
        //                             2 * lame_coefficient_mu * E;
        //          stress_term_minus = 0;
        //        }
        for (unsigned int i = 0; i < dofs_per_cell_u; ++i) {
          const unsigned int comp_i = fe_u.system_to_component_index(i).first;
          const Tensor<2, dim> Bu_iq_symmetric =
              0.5 * (Bu_kq[i] + transpose(Bu_kq[i]));
          const double Bu_iq_symmetric_tr = trace(Bu_iq_symmetric);

          if (!residual_only) {

            for (unsigned int j = 0; j < dofs_per_cell_u; ++j) {
              const unsigned int comp_j =
                  fe_u.system_to_component_index(j).first;
              {
                cell_matrix(i, j) +=
                    (scalar_product(constitutive_law.lambda *
                                            Bu_iq_symmetric_tr * Identity +
                                        2 * constitutive_law.mu *
                                            Bu_iq_symmetric,
                                    Bu_kq[j])) *
                    fe_values_u.JxW(q);
              }

              //              if (comp_j < introspection.disp_dim) {
              // Solid
              //                cell_matrix(j, i) +=
              //                    (scalar_product(
              ////                       ((1 - params.constant_k) *
              /// old_timestep_pf * / old_timestep_pf + / params.constant_k) *
              //                                        stress_term_plus_LinU,
              //                                    current_grads_u[j])
              //                     // stress term minus
              //                     + scalar_product(stress_term_minus_LinU,
              //                                      current_grads_u[j])) *
              //                    fe_values_u.JxW(q);

              //              } else if (comp_j ==
              //                         introspection.component_indices.phase_field)
              //                         {
              //                // Phase-field
              //                local_matrix(j, i) +=
              //                    // off-diagonal term
              //                    ((1 - params.constant_k) *
              //                         // two components mean two times
              //                         (scalar_product(stress_term_plus_LinU,
              //                         E) +
              //                          scalar_product(stress_term_plus,
              //                          E_LinU)) *
              //                         pf * current_pf[j] +
              //                     // diagonal term
              //                     (1 - params.constant_k) *
              //                         scalar_product(stress_term_plus, E) *
              //                         current_pf[i] * current_pf[j] +
              //                     params.Gc / params.l_phi * current_pf[i] *
              //                     current_pf[j] + params.Gc * params.l_phi *
              //                     current_grads_pf[i] *
              //                         current_grads_pf[j]) *
              //                    fe_values.JxW(q);
              //              }

              // end j dofs
            }
          }

          //          if (comp_i < dim) {
          //          const Tensor<2, dim> current_grads_u =
          //          fe_values_u.shape_grad(i, q); const double
          //          divergence_u_LinU =
          //              Tensors ::get_divergence_u<dim>(current_grads_u);

          // Solid
          cell_rhs(i) +=
              (scalar_product(Bu_kq[i], stress) * fe_values_u.JxW(q));

          //          cell_rhs(i) -=
          //              (scalar_product(
          //                   //                     ((1.0 - params.constant_k)
          //                   *
          //                   //                     old_timestep_pf *
          //                   // old_timestep_pf +
          //                   // params.constant_k) * stress_term_plus,
          //                   current_grads_u) +
          //               scalar_product(stress_term_minus, current_grads_u)) *
          //              fe_values_u.JxW(q);

          //          } else if (comp_i ==
          //          introspection.component_indices.phase_field) {
          //            const double phi_i_pf =
          //                fe_values[introspection.extractors.phase_field].value(i,
          //                q);
          //            const Tensor<1, dim> phi_i_grads_pf =
          //                fe_values[introspection.extractors.phase_field].gradient(i,
          //                q);
          //
          //            // Phase field
          //            local_rhs(i) -=
          //                ((1.0 - params.constant_k) *
          //                     scalar_product(stress_term_plus, E) * pf *
          //                     phi_i_pf -
          //                 params.Gc / params.l_phi * (1.0 - pf) * phi_i_pf +
          //                 params.Gc * params.l_phi * grad_pf *
          //                 phi_i_grads_pf) *
          //                fe_values.JxW(q);
          //          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      if (residual_only) {
        constraints_all_u.distribute_local_to_global(
            cell_rhs, local_dof_indices, system_rhs_u);
        //        constraints_update.distribute_local_to_global(
        //            local_rhs, local_dof_indices, system_total_residual);
      } else {
        constraints_all_u.distribute_local_to_global(
            cell_matrix, local_dof_indices, system_matrix_u);
        constraints_all_u.distribute_local_to_global(
            cell_rhs, local_dof_indices, system_rhs_u);
        //        constraints_update.distribute_local_to_global(
        //            local_matrix, local_rhs, local_dof_indices, system_matrix,
        //            system_rhs);
      }
    }
  //  if (residual_only) {
  //  }
  //  //    system_rhs_u.compress(VectorOperation::add);
  //  //    system_total_residual.compress(VectorOperation::add);
  //  else
  //
  system_matrix_u.compress(VectorOperation::add);
  system_rhs_u.compress(VectorOperation::add);

  //  if (params.enable_phase_field) {
  //    LA::MPI::PreconditionAMG::AdditionalData data;
  //    // data.constant_modes = constant_modes;
  //    data.elliptic = true;
  //    data.higher_order_elements = true;
  //    data.smoother_sweeps = 2;
  //    data.aggregation_threshold = 0.02;
  //    preconditioner_phase_field.initialize(system_matrix.block(1, 1), data);
  //  }

  timer.leave_subsection();
}

template <int dim>
void PhaseFieldFracture<dim>::enforce_phase_field_limitation() {
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_phi
                                                            .begin_active(),
                                                 endc = dof_handler_phi.end();

  std::vector<types::global_dof_index> local_dof_indices(fe_phi.dofs_per_cell);
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < fe_phi.dofs_per_cell; ++i) {
        const unsigned int comp_i = fe_phi.system_to_component_index(i).first;
        const types::global_dof_index idx = local_dof_indices[i];
        if (!dof_handler_phi.locally_owned_dofs().is_element(idx))
          continue;

        solution_phi(idx) = std::max(
            0.0, std::min(static_cast<double>(solution_phi(idx)), 1.0));
      }
    }

  solution_phi.compress(VectorOperation::insert);
}

template <int dim> unsigned int PhaseFieldFracture<dim>::solve() {
  timer.enter_subsection("Solve");
  TimerOutput::Scope t(computing_timer, "Solve");

  SolverControl solver_control(dof_handler_u.n_dofs(),
                               1e-8 * system_rhs_u.l2_norm());
  SolverGMRES<LA::MPI::Vector> solver(solver_control);
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.constant_modes = constant_modes;
    data.elliptic = true;
    data.higher_order_elements = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
    preconditioner_u.initialize(system_matrix_u, data);
  }

  solver.solve(system_matrix_u, incremental_u, system_rhs_u, preconditioner_u);

  //  pcout << "   Solved in " << solver_control.last_step() << " iterations."
  //        << std::endl;

  timer.leave_subsection();
  return solver_control.last_step();
}

template <int dim>
double PhaseFieldFracture<dim>::newton_iteration(double time) {
  pcout << "It.\tResidual\tReduction\tLSrch\t\t#LinIts" << std::endl;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1;

  // Line search parameters
  unsigned int line_search_step = 0;
  double new_newton_residual = 0.0;

  // Application of the initial boundary conditions to the
  // variational equations:
  setup_boundary_condition(time);
  constraints_all_u.distribute(solution_u);
  assemble_system(true);

  double newton_residual = system_rhs_u.linfty_norm();
  double old_newton_residual = newton_residual;
  unsigned int newton_step = 1;
  unsigned int no_linear_iterations = 0;

  pcout << "0\t" << std::scientific << newton_residual << std::endl;

  while (newton_residual > params.lower_bound_newton_residual &&
         newton_step < params.max_no_newton_steps) {
    old_newton_residual = newton_residual;

    assemble_system(true);
    newton_residual = system_rhs_u.linfty_norm();

    if (newton_residual < params.lower_bound_newton_residual) {
      pcout << '\t' << std::scientific << newton_residual << std::endl;
      break;
    }

    if (newton_step == 1 ||
        newton_residual / old_newton_residual > nonlinear_rho)
      assemble_system();

    // Solve Ax = b
    no_linear_iterations = solve();

    line_search_step = 0;
    // Relaxation
    for (; line_search_step < params.max_no_line_search_steps;
         ++line_search_step) {
      solution_u -= incremental_u;
      assemble_system(true);
      new_newton_residual = system_rhs_u.linfty_norm();

      if (new_newton_residual < newton_residual)
        break;
      else
        solution_u += incremental_u;

      incremental_u *= params.line_search_damping;
    }
    old_newton_residual = newton_residual;
    newton_residual = new_newton_residual;

    pcout << std::setprecision(5) << newton_step << '\t' << std::scientific
          << newton_residual;

    pcout << '\t' << std::scientific << newton_residual / old_newton_residual
          << '\t';

    if (newton_step == 1 ||
        newton_residual / old_newton_residual > nonlinear_rho)
      pcout << "rebuild" << '\t';
    else
      pcout << " " << '\t';
    pcout << line_search_step << '\t' << std::scientific << no_linear_iterations
          << '\t' << std::scientific << std::endl;

    // Terminate if nothing is solved anymore. After this,
    // we cut the time step.
    if ((newton_residual / old_newton_residual > params.upper_newton_rho) &&
        (newton_step > 1)) {
      break;
    }

    // Updates
    newton_step++;
  }

  if ((newton_residual > params.lower_bound_newton_residual) &&
      (newton_step == params.max_no_newton_steps)) {
    pcout << "Newton iteration did not converge in " << newton_step
          << " steps :-(" << std::endl;
    throw SolverControl::NoConvergence(0, 0);
  }

  return newton_residual / old_newton_residual;
}

template <int dim> void PhaseFieldFracture<dim>::refine_grid() {
  //  timer.enter_subsection("Refine grid");
  //  TimerOutput::Scope t(computing_timer, "Refine grid");
  //
  //  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  //  KellyErrorEstimator<dim>::estimate(
  //      dof_handler, QGauss<dim - 1>(fe.degree + 1),
  //      std::map<types::boundary_id, const Function<dim> *>(),
  //      solution.block(0), estimated_error_per_cell);
  //  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
  //      triangulation, estimated_error_per_cell, 0.3, 0.03);
  //  triangulation.execute_coarsening_and_refinement();
  //
  //  timer.leave_subsection();
}

template <int dim>
void PhaseFieldFracture<dim>::output_results(const unsigned int cycle) {
  timer.enter_subsection("Output results");
  TimerOutput::Scope t(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_u);

  std::vector<std::string> solution_names(dim, "displacement");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  //  if (introspection.disp_dim == 1) {
  //    data_component_interpretation[0] =
  //        DataComponentInterpretation::component_is_scalar;
  //  }

  //  if (params.enable_phase_field) {
  //    solution_names.push_back("phasefield");
  //    data_component_interpretation.push_back(
  //        DataComponentInterpretation::component_is_scalar);
  //  }
  data_out.add_data_vector(dof_handler_u, solution_u, solution_names,
                           data_component_interpretation);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(params.output_dir, "solution", cycle,
                                      mpi_com, 2, 8);

  // Record statistics
  statistics.add_value("Step", cycle);
  statistics.set_precision("Step", 1);
  statistics.set_scientific("Step", false);
  statistics.add_value("Time", time);
  statistics.set_precision("Time", 8);
  statistics.set_scientific("Time", true);
  compute_load();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::ofstream stat_file((params.output_dir + "/statistics.txt").c_str());
    statistics.write_text(stat_file);
    stat_file.close();
  }

  timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::compute_load() {
  const QGauss<dim - 1> face_quadrature_formula(fe_u.degree + 1);
  FEFaceValues<dim> fe_face_values(fe_u, face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe_u.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<2, dim>> face_solution_grads(n_face_q_points);

  Tensor<1, dim> load_value;

  LA::MPI::Vector rel_solution = solution_u;

  const Tensor<2, dim> Identity = Tensors::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_u
                                                            .begin_active(),
                                                 endc = dof_handler_u.end();

  const FEValuesExtractors::Vector displacement(0);

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 2) {
          fe_face_values.reinit(cell, face);
          fe_face_values[displacement].get_function_gradients(
              rel_solution, face_solution_grads);

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

  double load_y = Utilities::MPI::sum(load_value[1], mpi_com);
  pcout << "  Load y: " << load_y;
  statistics.add_value("Load y", load_y);
  statistics.set_precision("Load y", 8);
  statistics.set_scientific("Load y", true);
}

#endif