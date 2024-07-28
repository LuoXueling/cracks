/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PHASE_FIELD_FRACTURE_H
#define PHASE_FIELD_FRACTURE_H

#include "abaqus_grid_in.h"
#include "dealii_includes.h"
#include "multi_field.h"
#include "parameters.h"
#include "preconditioner.h"
#include "support_functions.h"
#include "dirichlet_boundary.h"
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
  void setup_boundary_condition(const double time, const bool initial_step,
                                AffineConstraints<double> &constraints);
  void assemble_diag_mass_matrix();
  void assemble_system();
  unsigned int solve();
  void refine_grid();
  void output_results(const unsigned int cycle);

  MPI_Comm mpi_com;

  parallel::distributed::Triangulation<dim> triangulation;
  Parameters::AllParameters params;

  MultiFieldCfg<dim> introspection;
  PreconditionerCfg PC;
  LA::MPI::PreconditionAMG preconditioner_solid;
  LA::MPI::PreconditionAMG preconditioner_phase_field;

  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  IndexSet active_set;
  std::vector<IndexSet> partition;
  std::vector<IndexSet> partition_relevant;

  AffineConstraints<double> constraints_update;
  AffineConstraints<double> constraints_hanging_nodes;

  std::vector<std::vector<bool>> constant_modes;

  LA::MPI::BlockSparseMatrix system_matrix;
  LA::MPI::BlockVector solution, newton_update, old_solution, old_old_solution;
  LA::MPI::BlockVector system_total_residual;
  LA::MPI::BlockVector system_rhs;
  LA::MPI::BlockVector diag_mass, diag_mass_relevant;

  ConditionalOStream pcout;
  DualOStream dcout;
  DualTimerOutput timer;
  TimerOutput computing_timer;

  double time;
  unsigned int load_step;
};

template <int dim>
PhaseFieldFracture<dim>::PhaseFieldFracture(Parameters::AllParameters &prms)
    : mpi_com(MPI_COMM_WORLD), params(prms), introspection(params),
      triangulation(mpi_com, typename Triangulation<dim>::MeshSmoothing(
                                 Triangulation<dim>::smoothing_on_refinement |
                                 Triangulation<dim>::smoothing_on_coarsening)),
      fe(introspection.FE_Q_sequence, introspection.FE_Q_dim_sequence),
      dof_handler(triangulation),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      dcout(pcout, params.log_file), timer(mpi_com, pcout, TimerOutput::never,
                                           TimerOutput::cpu_and_wall_times),
      computing_timer(mpi_com, pcout, TimerOutput::never,
                      TimerOutput::wall_times), time(0) {}

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

  const unsigned int n_cycles = 8;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
    pcout << "Cycle " << cycle << ':' << std::endl;

    if (cycle > 0) {
      refine_grid();
    }

    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;
    assemble_system();
    solve();

    output_results(cycle);

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }
  timer.manual_print_summary(dcout.fout);
}

template <int dim> void PhaseFieldFracture<dim>::setup_mesh() {
  timer.enter_subsection("Setup mesh");
  //  AbaqusGridIn<dim> grid_in;
  /**
   * similar to normal use of GridIn.
   */
  //  grid_in.attach_triangulation(triangulation);
  //  if (!checkFileExsit(params.mesh_from)) {
  //    throw std::runtime_error("Mesh file does not exist");
  //  }
  //  grid_in.read_abaqus_inp(params.mesh_from);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(5);

  if (dim == 2) {
    std::ofstream out(params.output_dir + "initial_grid.svg");
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);
  }
  dcout << "Find " << triangulation.n_global_active_cells() << " elements"
        << std::endl;
  timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::setup_system() {
  timer.enter_subsection("Setup system");
  TimerOutput::Scope t(computing_timer, "Setup system");

  system_matrix.clear();

  /**
   * DOF
   **/
  dof_handler.distribute_dofs(fe);

  std::vector<unsigned int> sub_blocks(introspection.n_components, 0);
  if (params.enable_phase_field) {
    sub_blocks[introspection.component_indices.phase_field] = 1;
  }
  DoFRenumbering::component_wise(dof_handler, introspection.block_component);

  constant_modes.clear();
  DoFTools::extract_constant_modes(
      dof_handler, introspection.component_masks.displacements, constant_modes);

  #if DEAL_II_VERSION_GTE(9, 2, 0)
    std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler,
                                          introspection.components_to_blocks);
  #else
    std::vector<types::global_dof_index> dofs_per_block(introspection.n_blocks);
    DoFTools::count_dofs_per_block(dof_handler, dofs_per_block,
                                   introspection.components_to_blocks);
  #endif
  partition.clear();
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  compatibility::split_by_block(dofs_per_block, locally_owned_dofs, partition);

  partition_relevant.clear();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  compatibility::split_by_block(dofs_per_block, locally_relevant_dofs,
                                partition_relevant);

  /**
   * Hanging node and boundary value constraints
   */
  {
    constraints_hanging_nodes.clear();
    constraints_hanging_nodes.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_hanging_nodes);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(introspection.n_components),
                                             constraints_hanging_nodes, introspection.component_masks.displacement[0]);

    constraints_hanging_nodes.close();
  }
  {
    constraints_update.clear();
    constraints_update.reinit(locally_relevant_dofs);

//    setup_boundary_condition(time, false, constraints_update);
    constraints_update.merge(constraints_hanging_nodes,
                             AffineConstraints<double>::right_object_wins);
    constraints_update.close();
  }

  /**
   * Print DOF status
   */
    {
   #if DEAL_II_VERSION_GTE(9, 2, 0)
      std::vector<types::global_dof_index> dofs_per_var =
          DoFTools::count_dofs_per_fe_block(dof_handler, introspection.block_component);
   #else
      std::vector<types::global_dof_index> dofs_per_var(2);
      DoFTools::count_dofs_per_block(dof_handler, dofs_per_var, sub_blocks);
   #endif

      const unsigned int n_solid = dofs_per_var[0];
      const unsigned int n_phase = dofs_per_var[1];
      dcout << std::endl;
      dcout << "DoFs: " << n_solid << " disp + " << n_phase << " phase"
            << std::endl;
    }

  /**
   * Sparsity pattern
   */
  BlockDynamicSparsityPattern dsp(partition);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_update, false, Utilities::MPI::this_mpi_process(mpi_com));
  dsp.compress();
  SparsityTools::distribute_sparsity_pattern(
      dsp, dof_handler.locally_owned_dofs(), mpi_com, locally_relevant_dofs);

  system_matrix.reinit(dsp);

  /**
   * Initialize solution
   */
  // Actual solution at time step n
  solution.reinit(partition);

  // Old timestep solution at time step n-1
  old_solution.reinit(partition_relevant);

  // Old timestep solution at time step n-2
  old_old_solution.reinit(partition_relevant);

  // Updates for Newton's method
  newton_update.reinit(partition);

  // Residual for  Newton's method
  system_rhs.reinit(partition);

  system_total_residual.reinit(partition);

  diag_mass.reinit(partition);
  diag_mass_relevant.reinit(partition_relevant);
  assemble_diag_mass_matrix();

  active_set.clear();
  active_set.set_size(dof_handler.n_dofs());

  timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::setup_boundary_condition(const double time, const bool initial_step, AffineConstraints<double> &constraints) {
  compatibility::ZeroFunction<dim> f_zero(introspection.n_components);
  VectorTools::interpolate_boundary_values(
      dof_handler, 2, f_zero, constraints,
      introspection.component_masks.displacement[1]);

  if (initial_step)
    VectorTools::interpolate_boundary_values(
        dof_handler, 3,
        BoundaryTensionTest<dim>(introspection.n_components, time),
        constraints, introspection.component_masks.displacements);
  else
    VectorTools::interpolate_boundary_values(
        dof_handler, 3, f_zero, constraints,
        introspection.component_masks.displacements);
}

template <int dim> void PhaseFieldFracture<dim>::assemble_diag_mass_matrix() {
  diag_mass = 0;

  QGaussLobatto<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  const unsigned int n_q_points = quadrature_formula.size();

  Vector<double> local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      local_rhs = 0;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i != introspection.component_indices.phase_field)
            continue; // only look at phase field

          local_rhs(i) += fe_values.shape_value(i, q_point) *
                          fe_values.shape_value(i, q_point) *
                          fe_values.JxW(q_point);
        }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        diag_mass(local_dof_indices[i]) += local_rhs(i);
    }

  diag_mass.compress(VectorOperation::add);
  diag_mass_relevant = diag_mass;
}

template <int dim> void PhaseFieldFracture<dim>::assemble_system() {
  timer.enter_subsection("Assemble system");
  TimerOutput::Scope t(computing_timer, "Assemble system");
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0.;
      cell_rhs = 0.;

      fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const double rhs_value =
            (fe_values.quadrature_point(q_point)[1] >
                     0.5 +
                         0.25 * std::sin(4.0 * numbers::PI *
                                         fe_values.quadrature_point(q_point)[0])
                 ? 1.
                 : -1.);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                 fe_values.shape_grad(j, q_point) *
                                 fe_values.JxW(q_point);

          cell_rhs(i) += rhs_value * fe_values.shape_value(i, q_point) *
                         fe_values.JxW(q_point);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints_update.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  {
    LA::MPI::PreconditionAMG::AdditionalData data;
//    data.constant_modes = constant_modes;
//    data.elliptic = true;
//    data.higher_order_elements = true;
//    data.smoother_sweeps = 2;
//    data.aggregation_threshold = 0.02;
    preconditioner_solid.initialize(system_matrix.block(0, 0), data);
  }
  if (params.enable_phase_field)
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    // data.constant_modes = constant_modes;
    data.elliptic = true;
    data.higher_order_elements = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
    preconditioner_phase_field.initialize(system_matrix.block(1, 1),
                                          data);
  }

  timer.leave_subsection();
}

template <int dim> unsigned int PhaseFieldFracture<dim>::solve() {
  timer.enter_subsection("Solve");
  TimerOutput::Scope t(computing_timer, "Solve");

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
  SolverCG<LA::MPI::BlockVector> solver(solver_control);

  BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
                              LA::MPI::PreconditionAMG>
      preconditioner(system_matrix, preconditioner_solid,
                     preconditioner_phase_field);

  solver.solve(system_matrix, newton_update, system_rhs,
               preconditioner);

  pcout << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  constraints_update.distribute(newton_update);

  solution = newton_update; // Change this
  timer.leave_subsection();
  return solver_control.last_step();
}

template <int dim> void PhaseFieldFracture<dim>::refine_grid() {
  timer.enter_subsection("Refine grid");
  TimerOutput::Scope t(computing_timer, "Refine grid");

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution.block(0), estimated_error_per_cell);
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
  triangulation.execute_coarsening_and_refinement();

  timer.leave_subsection();
}

template <int dim>
void PhaseFieldFracture<dim>::output_results(const unsigned int cycle) {
  timer.enter_subsection("Output results");
  TimerOutput::Scope t(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(introspection.disp_dim, "displacement");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            introspection.disp_dim, DataComponentInterpretation::component_is_part_of_vector);
  if (introspection.disp_dim == 1){
    data_component_interpretation[0] = DataComponentInterpretation::component_is_scalar;
  }

  if (params.enable_phase_field) {
    solution_names.push_back("phasefield");
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
  }
  data_out.add_data_vector(dof_handler, solution, solution_names,
                           data_component_interpretation);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(params.output_dir, "solution", cycle,
                                      mpi_com, 2, 8);

  timer.leave_subsection();
}
#endif