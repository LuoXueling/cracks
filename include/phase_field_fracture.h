/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PHASE_FIELD_FRACTURE_H
#define PHASE_FIELD_FRACTURE_H

#include "abstract_field.h"
#include "abstract_multiphysics.h"
#include "controller.h"
#include "dealii_includes.h"
#include "elasticity.h"
#include "parameters.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseFieldFracture : public AbstractMultiphysics<dim> {
public:
  explicit PhaseFieldFracture(Parameters::AllParameters &prms);

private:
  void setup_system() override;
  void refine_grid() override;
  void record_old_solution() override;
  void return_old_solution() override;
  double staggered_scheme() override;
  void respective_output_results(DataOut<dim> &data_out) override;

  Elasticity<dim> elasticity;
  PhaseField<dim> phasefield;
};

template <int dim>
PhaseFieldFracture<dim>::PhaseFieldFracture(Parameters::AllParameters &prms)
    : AbstractMultiphysics<dim>(prms),
      elasticity(dim, (this->ctl).params.boundary_from, "newton", this->ctl),
      phasefield(prms.phase_field_scheme, this->ctl) {}

template <int dim> void PhaseFieldFracture<dim>::setup_system() {
  this->ctl.debug_dcout << "Initialize system - elasticity" << std::endl;
  elasticity.setup_system(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    this->ctl.debug_dcout << "Initialize system - phase field" << std::endl;
    phasefield.setup_system(this->ctl);
  }
  (this->ctl).quadrature_point_history.initialize(
      (this->ctl).triangulation.begin_active(), (this->ctl).triangulation.end(),
      (this->ctl).quadrature_formula.size());
}

template <int dim> void PhaseFieldFracture<dim>::record_old_solution() {
  elasticity.record_old_solution(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    phasefield.record_old_solution(this->ctl);
  }
}

template <int dim> void PhaseFieldFracture<dim>::return_old_solution() {
  elasticity.return_old_solution(this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    phasefield.return_old_solution(this->ctl);
  }
}

template <int dim> double PhaseFieldFracture<dim>::staggered_scheme() {
  if ((this->ctl).params.enable_phase_field) {
    (this->ctl).dcout << "Staggered scheme - Solving phase field" << std::endl;
    (this->ctl).computing_timer.enter_subsection("Solve phase field");
    double newton_reduction_phasefield = phasefield.update(this->ctl);
    (this->ctl).finalize();
    phasefield.enforce_phase_field_limitation(this->ctl);
    (this->ctl).computing_timer.leave_subsection("Solve phase field");

    (this->ctl).dcout << "Staggered scheme - Solving elasticity" << std::endl;
    (this->ctl).computing_timer.enter_subsection("Solve elasticity");
    double newton_reduction_elasticity = elasticity.update(this->ctl);
    (this->ctl).finalize();
    (this->ctl).computing_timer.leave_subsection("Solve elasticity");

    return std::max(newton_reduction_elasticity, newton_reduction_phasefield);
  } else {
    (this->ctl).dcout
        << "Solve Newton system - staggered scheme - Solving elasticity"
        << std::endl;
    (this->ctl).computing_timer.enter_subsection("Solve elasticity");
    double newton_reduction_elasticity = elasticity.update(this->ctl);
    (this->ctl).computing_timer.leave_subsection("Solve elasticity");
    return newton_reduction_elasticity;
  }
}

template <int dim>
void PhaseFieldFracture<dim>::respective_output_results(
    DataOut<dim> &data_out) {
  (this->ctl).dcout << "Computing output - elasticity" << std::endl;
  elasticity.output_results(data_out, this->ctl);
  if ((this->ctl).params.enable_phase_field) {
    (this->ctl).dcout << "Computing output - phase field" << std::endl;
    phasefield.output_results(data_out, this->ctl);
  }
}

template <int dim> void PhaseFieldFracture<dim>::refine_grid() {
  (this->ctl).timer.enter_subsection("Refine grid");
  //  TimerOutput::Scope t((this->ctl).computing_timer, "Refine grid");
  //
  //  Vector<float>
  //  estimated_error_per_cell((this->ctl).triangulation.n_active_cells());
  //  KellyErrorEstimator<dim>::estimate(
  //      elasticity.dof_handler, QGauss<dim - 1>(elasticity.fe.degree + 1),
  //      std::map<types::boundary_id, const Function<dim> *>(),
  //      elasticity.solution, estimated_error_per_cell);
  //  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
  //      (this->ctl).triangulation, estimated_error_per_cell, 0.3, 0.03);
  //  typename DoFHandler<dim>::active_cell_iterator
  //      cell = elasticity.dof_handler.begin_active(),
  //      endc = elasticity.dof_handler.end();
  //  for (; cell != endc; ++cell)
  //    if (cell->is_locally_owned())
  //      cell->set_refine_flag();
  //
  //  parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> soltrans =
  //      elasticity.prepare_refine();
  //
  //  (this->ctl).triangulation.execute_coarsening_and_refinement();
  //  setup_system();
  //
  //  elasticity.post_refine(soltrans, this->ctl);

  (this->ctl).timer.leave_subsection();
}

#endif