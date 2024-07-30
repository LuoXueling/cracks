/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PHASE_FIELD_FRACTURE_H
#define PHASE_FIELD_FRACTURE_H

#include "abaqus_grid_in.h"
#include "abstract_field.h"
#include "controller.h"
#include "dealii_includes.h"
#include "elasticity.h"
#include "parameters.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseFieldFracture {
public:
  explicit PhaseFieldFracture(Parameters::AllParameters &prms);

  void run();

private:
  void setup_mesh();
  void setup_system();
  void refine_grid();
  void output_results();

  Controller<dim> ctl;

  Elasticity<dim> elasticity;
};

template <int dim>
PhaseFieldFracture<dim>::PhaseFieldFracture(Parameters::AllParameters &prms)
    : ctl(prms), elasticity(ctl) {}

template <int dim> void PhaseFieldFracture<dim>::run() {
  ctl.dcout << "Project: " << ctl.params.project_name << std::endl;
  ctl.dcout << "Mesh from: " << ctl.params.mesh_from << std::endl;
  ctl.dcout << "Load sequence from: " << ctl.params.load_sequence_from
            << std::endl;
  ctl.dcout << "Output directory: " << ctl.params.output_dir << std::endl;
  ctl.dcout << "Solving " << ctl.params.dim << " dimensional PFM problem"
            << std::endl;
  ctl.dcout << "Running on " << Utilities::MPI::n_mpi_processes(ctl.mpi_com)
            << " MPI rank(s)" << std::endl;
  ctl.dcout << "Number of threads " << MultithreadInfo::n_threads()
            << std::endl;
  ctl.dcout << "Number of cores " << MultithreadInfo::n_cores() << std::endl;

  setup_mesh();
  setup_system();

  //  if (ctl.params.enable_phase_field) {
  //    enforce_phase_field_limitation();
  //  }

  unsigned int refinement_cycle = 0;
  double finishing_timestep_loop = 0;
  double tmp_timestep = 0.0;

  ctl.current_timestep = ctl.params.timestep;
  // Initialize old and old_old timestep sizes
  ctl.old_timestep = ctl.current_timestep;

  do {
    double newton_reduction = 1.0;
    if (ctl.timestep_number > ctl.params.switch_timestep &&
        ctl.params.switch_timestep > 0)
      ctl.current_timestep = ctl.params.timestep_size_2;

    double tmp_current_timestep = ctl.current_timestep;
    ctl.old_timestep = ctl.current_timestep;

  mesh_refine_checkpoint:
    ctl.pcout << std::endl;
    ctl.pcout << "\n=============================="
              << "=========================================" << std::endl;
    ctl.pcout << "Time " << ctl.timestep_number << ": " << ctl.time << " ("
              << ctl.current_timestep << ")" << "   "
              << "Cells: " << ctl.triangulation.n_global_active_cells() << "   "
              << "Displacement DoFs: " << elasticity.dof_handler.n_dofs();
    ctl.pcout << "\n--------------------------------"
              << "---------------------------------------" << std::endl;
    ctl.pcout << std::endl;

    ctl.time += ctl.current_timestep;

    do {
      // The Newton method can either stagnate or the linear solver
      // might not converge. To not abort the program we catch the
      // exception and retry with a smaller step.
      //          use_old_timestep_pf = false;
      elasticity.record_old_solution(ctl);
      try {
        newton_reduction = elasticity.newton_iteration(ctl);

        while (newton_reduction > ctl.params.upper_newton_rho) {
          //              use_old_timestep_pf = true;
          ctl.time -= ctl.current_timestep;
          ctl.current_timestep = ctl.current_timestep / 10.0;
          ctl.time += ctl.current_timestep;
          elasticity.return_old_solution(ctl);
          newton_reduction = elasticity.newton_iteration(ctl);

          if (ctl.current_timestep < 1.0e-9) {
            ctl.pcout << "Step size too small - keeping the step size"
                      << std::endl;
            break;
          }
        }

        break;

      } catch (SolverControl::NoConvergence &e) {
        ctl.pcout << "Solver did not converge! Adjusting time step."
                  << std::endl;
      }

      ctl.time -= ctl.current_timestep;
      elasticity.return_old_solution(ctl);
      ctl.current_timestep = ctl.current_timestep / 10.0;
      ctl.time += ctl.current_timestep;
    } while (true);

//    LA::MPI::Vector distributed_solution(elasticity.locally_owned_dofs, ctl.mpi_com);
//    distributed_solution = elasolution;
//    elasticity.distribute_hanging_node_constraints(distributed_solution, ctl);

    // Refine mesh and return to the beginning if mesh is changed.
    //    if (ctl.params.refine) {
    //      bool changed = refine_grid();
    //      if (changed) {
    //        // redo the current time step
    //        ctl.pcout << "Mesh changed! Re-do the current time step" <<
    //        std::endl; ctl.time -= ctl.current_timestep; solution =
    //        old_solution; goto mesh_refine_checkpoint; continue;
    //      }
    //    }

    // Recover time step
    ctl.current_timestep = tmp_current_timestep;

    output_results();

    ++ctl.timestep_number;

    ctl.computing_timer.print_summary();
    ctl.computing_timer.reset();
    ctl.pcout << std::endl;
  } while (ctl.timestep_number <= ctl.params.max_no_timesteps);
  ctl.timer.manual_print_summary(ctl.dcout.fout);
}

template <int dim> void PhaseFieldFracture<dim>::setup_mesh() {
  ctl.timer.enter_subsection("Setup mesh");
  AbaqusGridIn<dim> grid_in;
  /**
   * similar to normal use of GridIn.
   */
  grid_in.attach_triangulation(ctl.triangulation);
  if (!checkFileExsit(ctl.params.mesh_from)) {
    throw std::runtime_error("Mesh file does not exist");
  }
  grid_in.read_abaqus_inp(ctl.params.mesh_from);
  //  GridGenerator::hyper_cube(ctl.triangulation);
  //  ctl.triangulation.refine_global(5);

  if (dim == 2) {
    std::ofstream out(ctl.params.output_dir + "initial_grid.svg");
    GridOut grid_out;
    grid_out.write_svg(ctl.triangulation, out);
  }
  ctl.dcout << "Find " << ctl.triangulation.n_global_active_cells()
            << " elements" << std::endl;

  ctl.timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::setup_system() {
  ctl.timer.enter_subsection("Setup system");
  TimerOutput::Scope t(ctl.computing_timer, "Setup system");

  elasticity.setup_system(ctl);

  ctl.timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::refine_grid() {
  //  ctl.timer.enter_subsection("Refine grid");
  //  TimerOutput::Scope t(ctl.computing_timer, "Refine grid");
  //
  //  Vector<float>
  //  estimated_error_per_cell(ctl.triangulation.n_active_cells());
  //  KellyErrorEstimator<dim>::estimate(
  //      dof_handler, QGauss<dim - 1>(fe.degree + 1),
  //      std::map<types::boundary_id, const Function<dim> *>(),
  //      solution.block(0), estimated_error_per_cell);
  //  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
  //      ctl.triangulation, estimated_error_per_cell, 0.3, 0.03);
  //  ctl.triangulation.execute_coarsening_and_refinement();
  //
  //  ctl.timer.leave_subsection();
}

template <int dim> void PhaseFieldFracture<dim>::output_results() {
  ctl.timer.enter_subsection("Output results");
  TimerOutput::Scope t(ctl.computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_triangulation(ctl.triangulation);

  Vector<float> subdomain(ctl.triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = ctl.triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  // Record statistics
  ctl.statistics.add_value("Step", ctl.timestep_number);
  ctl.statistics.set_precision("Step", 1);
  ctl.statistics.set_scientific("Step", false);
  ctl.statistics.add_value("Time", ctl.time);
  ctl.statistics.set_precision("Time", 8);
  ctl.statistics.set_scientific("Time", true);

  elasticity.output_results(data_out, ctl);

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(ctl.params.output_dir, "solution",
                                      ctl.timestep_number, ctl.mpi_com, 2, 8);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::ofstream stat_file(
        (ctl.params.output_dir + "/statistics.txt").c_str());
    ctl.statistics.write_text(stat_file);
    stat_file.close();
  }

  ctl.timer.leave_subsection();
}

#endif