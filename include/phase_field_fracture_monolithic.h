//
// Created by xlluo on 24-8-13.
//

#ifndef CRACKS_PHASE_FIELD_FRACTURE_MONOLITHIC_H
#define CRACKS_PHASE_FIELD_FRACTURE_MONOLITHIC_H

#include "abstract_field.h"
#include "abstract_multiphysics.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "degradation.h"
#include "elasticity.h"
#include "fatigue_degradation.h"
#include "parameters.h"
#include "phase_field.h"
#include "post_processors.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseFieldElasticity : public AbstractField<dim> {
public:
  PhaseFieldElasticity(std::string boundary_from, Controller<dim> &ctl);
  void assemble_newton_system(bool residual_only,
                              LA::MPI::BlockVector &neumann_rhs,
                              Controller<dim> &ctl) override;
  void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) override;

  void compute_load(Controller<dim> &ctl);
  void enforce_phase_field_limitation(Controller<dim> &ctl);

  ConstitutiveLaw<dim> constitutive_law;
};

template <int dim>
PhaseFieldElasticity<dim>::PhaseFieldElasticity(std::string boundary_from,
                                                Controller<dim> &ctl)
    : AbstractField<dim>(std::vector<unsigned int>{dim, 1},
                         std::vector<std::string>{"elasticity", "phasefield"},
                         std::vector<std::string>{boundary_from, "none"},
                         "newton", ctl),
      constitutive_law(ctl.params.E, ctl.params.v, ctl.params.plane_state) {}

template <int dim>
void PhaseFieldElasticity<dim>::assemble_newton_system(
    bool residual_only, LA::MPI::BlockVector &neumann_rhs,
    Controller<dim> &ctl) {
  ctl.aggregate_point_history();

  this->system_rhs = 0;
  if (!residual_only) {
    this->system_matrix = 0;
  }
  /*
   * From Elasticity
   */

  FEValues<dim> fe_values((this->fe), ctl.quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  const FEValuesExtractors::Vector displacement =
      (this->fields).extractors_vector["elasticity"];

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);

  // Old Newton grads
  std::vector<Tensor<2, dim>> old_displacement_grads(n_q_points);

  std::vector<Tensor<1, dim>> Nu_kq(dofs_per_cell);
  std::vector<Tensor<2, dim>> Bu_kq(dofs_per_cell);
  std::vector<SymmetricTensor<2, dim>> Bu_kq_symmetric(dofs_per_cell);

  Tensor<2, dim> zero_matrix;
  zero_matrix.clear();

  // Integrate face load
  // If integrated (or modified by arc length control), skip the process.
  if (neumann_rhs.all_zero()) {
    this->setup_neumann_boundary_condition(neumann_rhs, ctl);
  }
  this->system_rhs -= neumann_rhs;

  // Determine decomposition
  std::unique_ptr<Decomposition<dim>> decomposition =
      select_decomposition<dim>(ctl.params.decomposition);
  // Determine degradation
  std::unique_ptr<Degradation<dim>> degradation =
      select_degradation<dim>(ctl.params.degradation);

  /*
   * From phase field
   */

  const FEValuesExtractors::Scalar phasefield =
      (this->fields).extractors_scalar["phasefield"];

  // Old Newton values
  std::vector<double> old_phasefield_values(n_q_points);
  // Old Newton grads
  std::vector<Tensor<1, dim>> old_phasefield_grads(n_q_points);

  std::vector<double> Nphi_kq(dofs_per_cell);
  std::vector<Tensor<1, dim>> Bphi_kq(dofs_per_cell);

  std::unique_ptr<FatigueDegradation<dim>> fatigue_degradation =
      select_fatigue_degradation<dim>(ctl.params.fatigue_degradation, ctl);
  std::unique_ptr<FatigueAccumulation<dim>> fatigue_accumulation =
      select_fatigue_accumulation<dim>(ctl.params.fatigue_accumulation, ctl);

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);

      /*
       * From Elasticity
       */

      fe_values[displacement].get_function_values((this->solution),
                                                  old_displacement_values);
      fe_values[displacement].get_function_gradients((this->solution),
                                                     old_displacement_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Get history
        double phasefield = lqph[q]->get("Phase field", 0.0);
        double degrade = degradation->value(phasefield, ctl);

        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nu_kq[k] = fe_values[displacement].value(k, q);
          Bu_kq[k] = fe_values[displacement].gradient(k, q);
          Bu_kq_symmetric[k] = fe_values[displacement].symmetric_gradient(k, q);
          // equivalent to:
          // 0.5 * (Bu_kq[k] + transpose(Bu_kq[k]));
        }
        const Tensor<2, dim> grad_u = old_displacement_grads[q];
        const double divergence_u = Tensors::get_divergence_u<dim>(grad_u);
        const Tensor<2, dim> Identity = Tensors ::get_Identity<dim>();
        const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));

        // Solution A:
        SymmetricTensor<2, dim> strain_symm;
        SymmetricTensor<2, dim> stress_0;
        SymmetricTensor<4, dim> elasticity_tensor;
        constitutive_law.get_stress_strain_tensor(E, strain_symm, stress_0,
                                                  elasticity_tensor);
        double energy_positive;
        double energy_negative;
        SymmetricTensor<2, dim> stress_positive;
        SymmetricTensor<2, dim> stress_negative;
        SymmetricTensor<4, dim> elasticity_tensor_positive;
        SymmetricTensor<4, dim> elasticity_tensor_negative;

        if (!residual_only) {
          decomposition->decompose_elasticity_tensor_stress_and_energy(
              strain_symm, stress_0, elasticity_tensor, energy_positive,
              energy_negative, stress_positive, stress_negative,
              elasticity_tensor_positive, elasticity_tensor_negative,
              constitutive_law);
        } else {
          decomposition->decompose_stress_and_energy(
              strain_symm, stress_0, energy_positive, energy_negative,
              stress_positive, stress_negative, constitutive_law);
        }

        // Solution B (without split):
        // Tensor<2, dim> stress_0 = constitutive_law.lambda * tr_E * Identity +
        //                                  2 * constitutive_law.mu * E;
        // const double tr_E = trace(E);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          if (!this->dof_is_this_field(i, "elasticity")) {
            continue;
          }
          // Solution B (without split):
          // const double Bu_kq_symmetric_tr = trace(Bu_kq_symmetric[i]);
          if (!residual_only) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              if (!this->dof_is_this_field(j, "elasticity")) {
                continue;
              }
              {
                cell_matrix(i, j) +=
                    // Solution A:
                    (Bu_kq_symmetric[i] *
                     (degrade * elasticity_tensor_positive +
                      elasticity_tensor_negative) *
                     Bu_kq_symmetric[j]) *
                    // Solution B (without split):
                    // degradation *
                    // (scalar_product(constitutive_law.lambda *
                    //                         Bu_kq_symmetric_tr * Identity +
                    //                    2 * constitutive_law.mu *
                    //                        Bu_kq_symmetric[i],
                    //                Bu_kq[j])) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) += scalar_product(Bu_kq[i], degrade * stress_positive +
                                                      stress_negative) *
                         fe_values.JxW(q);
        }

        // Update history
        lqph[q]->update("Driving force", energy_positive, "max");
        lqph[q]->update("Positive elastic energy", energy_positive);
      }

      /*
       * From phase field
       */

      fe_values[phasefield].get_function_values((this->solution),
                                                old_phasefield_values);
      fe_values[phasefield].get_function_gradients((this->solution),
                                                   old_phasefield_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nphi_kq[k] = fe_values[phasefield].value(k, q);
          Bphi_kq[k] = fe_values[phasefield].gradient(k, q);
        }

        double H = lqph[q]->get("Driving force", 0.0);
        double degrade = degradation->value(old_phasefield_values[q], ctl);
        double degrade_derivative =
            degradation->derivative(old_phasefield_values[q], ctl);
        double degrade_second_derivative =
            degradation->second_derivative(old_phasefield_values[q], ctl);
        double cw, w, w_derivative;
        if (ctl.params.phasefield_model == "AT1") {
          cw = 2.0 / 3.0;
          w = 1;
          w_derivative = 0;
        } else if (ctl.params.phasefield_model == "AT2") {
          cw = 0.5;
          w = 2 * old_phasefield_values[q];
          w_derivative = 2;
        } else {
          AssertThrow(false,
                      ExcNotImplemented("Phase field model not available."));
        }

        double fatigue_degrade, fatigue_degrade_derivative;
        Tensor<1, dim> fatigue_degrade_grad;
        if (ctl.params.enable_fatigue) {
          fatigue_accumulation->step(lqph[q], old_phasefield_values[q], degrade,
                                     degrade_derivative,
                                     degrade_second_derivative, ctl);
          fatigue_degrade = fatigue_degradation->degradation_value(
              lqph[q], old_phasefield_values[q], degrade, ctl);
        } else {
          fatigue_degrade = 1.0;
          fatigue_degrade_derivative = 0.0;
        }

        if (ctl.params.phasefield_model == "AT1") {
          H = std::max(H, 3.0 * ctl.params.Gc / (16.0 * ctl.params.l_phi) *
                              fatigue_degrade);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          if (!this->dof_is_this_field(i, "phasefield")) {
            continue;
          }
          if (!residual_only) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              if (!this->dof_is_this_field(j, "phasefield")) {
                continue;
              }
              {
                cell_matrix(i, j) +=
                    (ctl.params.Gc / (2 * cw) * fatigue_degrade *
                         ctl.params.l_phi * Bphi_kq[i] * Bphi_kq[j] +
                     Nphi_kq[i] * Nphi_kq[j] *
                         (degrade_second_derivative * H +
                          ctl.params.Gc / (2 * cw) * fatigue_degrade /
                              (2 * ctl.params.l_phi) * w_derivative)) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) += (degrade_derivative * H * Nphi_kq[i] +
                          ctl.params.Gc / (2 * cw) *
                              (fatigue_degrade * ctl.params.l_phi *
                                   old_phasefield_grads[q] * Bphi_kq[i] +
                               fatigue_degrade / (2 * ctl.params.l_phi) * w *
                                   Nphi_kq[i])) *
                         fe_values.JxW(q);

          lqph[q]->update("Phase field", old_phasefield_values[q]);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      if (residual_only) {
        (this->constraints_all)
            .distribute_local_to_global(cell_rhs, local_dof_indices,
                                        (this->system_rhs));
      } else {
        (this->constraints_all)
            .distribute_local_to_global(cell_matrix, local_dof_indices,
                                        (this->system_matrix));
        (this->constraints_all)
            .distribute_local_to_global(cell_rhs, local_dof_indices,
                                        (this->system_rhs));
      }
    }

  (this->system_matrix).compress(VectorOperation::add);
  (this->system_rhs).compress(VectorOperation::add);
}

template <int dim>
void PhaseFieldElasticity<dim>::output_results(DataOut<dim> &data_out,
                                               Controller<dim> &ctl) {
  ctl.debug_dcout << "Computing output - all fields" << std::endl;
  LA::MPI::BlockVector distributed_solution(this->fields_locally_owned_dofs);
  distributed_solution = this->solution;
  std::vector<std::string> solution_names(dim, "Displacement");
  solution_names.emplace_back("Phase_field");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
  data_out.add_data_vector(this->dof_handler, distributed_solution,
                           solution_names, data_component_interpretation);

  /*
   * From Elasticity
   */
  ctl.debug_dcout << "Computing output - elasticity - strain" << std::endl;
  //   data_out.add_data_vector((this->dof_handler), (this->solution), strain);
  StrainProcessor<dim> strain_processor(this->fe, this->fields, ctl);
  strain_processor.add_data_vector(this->solution, this->fields, data_out,
                                   this->dof_handler, ctl);
  ctl.debug_dcout << "Computing output - elasticity - stress" << std::endl;
  //  data_out.add_data_vector((this->dof_handler), (this->solution), stress);
  StressProcessor<dim> stress_processor(constitutive_law, this->fe,
                                        this->fields, ctl);
  stress_processor.add_data_vector(this->solution, this->fields, data_out,
                                   this->dof_handler, ctl);
  // Record statistics
  ctl.debug_dcout << "Computing output - elasticity - load" << std::endl;
  compute_load(ctl);

  /*
   * From Phase field
   */
  PointHistoryProcessor<dim> hist_processor("Driving force", this->fields,
                                            this->fe, ctl);
  hist_processor.add_data_scalar(this->solution, this->fields, data_out,
                                 this->dof_handler, ctl);
  if (ctl.params.enable_fatigue) {
    PointHistoryProcessor<dim> fatigue_processor("Fatigue history",
                                                 this->fields, this->fe, ctl);
    fatigue_processor.add_data_scalar(this->solution, this->fields, data_out,
                                      this->dof_handler, ctl);
  }
}

template <int dim>
void PhaseFieldElasticity<dim>::compute_load(Controller<dim> &ctl) {
  /*
   * Simply copied from elasticity
   */
  const QGauss<dim - 1> face_quadrature_formula(ctl.params.poly_degree + 1);
  FEFaceValues<dim> fe_face_values((this->fe), face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).dofs_per_cell;
  const unsigned int n_q_points = ctl.quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<2, dim>> face_solution_grads(n_face_q_points);

  std::map<int, Tensor<1, dim>> load_value;

  const Tensor<2, dim> Identity = Tensors::get_Identity<dim>();

  std::unique_ptr<Decomposition<dim>> decomposition =
      select_decomposition<dim>(ctl.params.decomposition);

  for (const int id : ctl.boundary_ids)
    load_value[id] = Tensor<1, dim>();
  const FEValuesExtractors::Vector displacement =
      (this->fields).extractors_vector["elasticity"];
  ctl.debug_dcout << "Computing output - elasticity - load - computing"
                  << std::endl;
  // Determine degradation
  std::unique_ptr<Degradation<dim>> degradation =
      select_degradation<dim>(ctl.params.degradation);
  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() != 0) {
          fe_face_values.reinit(cell, face);
          fe_face_values[displacement].get_function_gradients(
              (this->solution), face_solution_grads);

          const std::vector<std::shared_ptr<PointHistory>> lqph =
              ctl.quadrature_point_history.get_data(cell);
          double phasefield = 0;
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            phasefield += lqph[q_point]->get("Phase field", 0.0) / n_q_points;
          }
          double degrade = degradation->value(phasefield, ctl);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
            const Tensor<2, dim> grad_u = face_solution_grads[q_point];

            const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
            const double tr_E = trace(E);

            SymmetricTensor<2, dim> strain_symm;
            SymmetricTensor<2, dim> stress_0;
            SymmetricTensor<4, dim> elasticity_tensor;
            constitutive_law.get_stress_strain_tensor(E, strain_symm, stress_0,
                                                      elasticity_tensor);
            double energy_positive;
            double energy_negative;
            SymmetricTensor<2, dim> stress_positive;
            SymmetricTensor<2, dim> stress_negative;
            SymmetricTensor<4, dim> elasticity_tensor_positive;
            SymmetricTensor<4, dim> elasticity_tensor_negative;

            decomposition->decompose_stress_and_energy(
                strain_symm, stress_0, energy_positive, energy_negative,
                stress_positive, stress_negative, constitutive_law);

            load_value[face->boundary_id()] +=
                degrade * stress_positive *
                fe_face_values.normal_vector(q_point) *
                fe_face_values.JxW(q_point);
          }
        }
    }
  ctl.debug_dcout << "Computing output - elasticity - load - recording"
                  << std::endl;

  for (auto const &it : load_value) {
    ctl.debug_dcout
        << "Computing output - elasticity - load - recording - boundary id=" +
               std::to_string(it.first)
        << std::endl;
    for (int i = 0; i < dim; ++i) {
      ctl.debug_dcout
          << "Computing output - elasticity - load - recording - dim - sum"
          << std::endl;
      double load = Utilities::MPI::sum(it.second[i], ctl.mpi_com);
      ctl.debug_dcout
          << "Computing output - elasticity - load - recording - dim - record"
          << std::endl;
      std::ostringstream stringStream;
      stringStream << "Boundary-" << it.first << "-Dir-" << i;
      (ctl.statistics).add_value(stringStream.str(), load);
      (ctl.statistics).set_precision(stringStream.str(), 8);
      (ctl.statistics).set_scientific(stringStream.str(), true);
    }
  }
}

template <int dim>
void PhaseFieldElasticity<dim>::enforce_phase_field_limitation(
    Controller<dim> &ctl) {
  /*
   * Simply copied from Phase field
   */
  typename DoFHandler<dim>::active_cell_iterator cell = (this->dof_handler)
                                                            .begin_active(),
                                                 endc =
                                                     (this->dof_handler).end();

  LA::MPI::BlockVector distributed_solution(this->fields_locally_owned_dofs);
  distributed_solution = this->solution;

  std::vector<types::global_dof_index> local_dof_indices(
      (this->fe).dofs_per_cell);
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < (this->fe).dofs_per_cell; ++i) {
        if (!this->dof_is_this_field(i, "phasefield")) {
          continue;
        }
        const types::global_dof_index idx = local_dof_indices[i];
        if (!(this->dof_handler).locally_owned_dofs().is_element(idx))
          continue;

        distributed_solution(idx) = std::max(
            0.0, std::min(static_cast<double>((this->solution)(idx)), 1.0));
      }
    }

  distributed_solution.compress(VectorOperation::insert);
  this->solution = distributed_solution;
}

template <int dim>
class PhaseFieldFractureMonolithic : public AbstractMultiphysics<dim> {
public:
  explicit PhaseFieldFractureMonolithic(Parameters::AllParameters &prms);

private:
  void setup_system() override;
  bool refine_grid() override;
  void record_old_solution() override;
  void return_old_solution() override;
  double staggered_scheme() override;
  void respective_output_results(DataOut<dim> &data_out) override;

  PhaseFieldElasticity<dim> fracture;
};

template <int dim>
PhaseFieldFractureMonolithic<dim>::PhaseFieldFractureMonolithic(
    Parameters::AllParameters &prms)
    : AbstractMultiphysics<dim>(prms),
      fracture((this->ctl).params.boundary_from, this->ctl) {}

template <int dim> void PhaseFieldFractureMonolithic<dim>::setup_system() {
  this->ctl.debug_dcout << "Initialize system" << std::endl;
  fracture.setup_system(this->ctl);
}

template <int dim>
void PhaseFieldFractureMonolithic<dim>::record_old_solution() {
  (this->ctl).record_point_history((this->ctl).quadrature_point_history,
                                   (this->ctl).old_quadrature_point_history);
  fracture.record_old_solution(this->ctl);
}

template <int dim>
void PhaseFieldFractureMonolithic<dim>::return_old_solution() {
  (this->ctl).record_point_history((this->ctl).old_quadrature_point_history,
                                   (this->ctl).quadrature_point_history);
  fracture.return_old_solution(this->ctl);
}

template <int dim>
double PhaseFieldFractureMonolithic<dim>::staggered_scheme() {
  (this->ctl).dcout << "Solve Newton system" << std::endl;
  double newton_reduction = fracture.update(this->ctl);
  fracture.enforce_phase_field_limitation(this->ctl);
  (this->ctl).aggregate_point_history();
  return newton_reduction;
}

template <int dim>
void PhaseFieldFractureMonolithic<dim>::respective_output_results(
    DataOut<dim> &data_out) {
  (this->ctl).dcout << "Computing output" << std::endl;
  fracture.output_results(data_out, this->ctl);
}

template <int dim> bool PhaseFieldFractureMonolithic<dim>::refine_grid() {
  typename DoFHandler<dim>::active_cell_iterator cell = fracture.dof_handler
                                                            .begin_active(),
                                                 endc =
                                                     fracture.dof_handler.end();

  FEValues<dim> fe_values(fracture.fe, (this->ctl).quadrature_formula,
                          update_gradients);

  unsigned int n_q_points = (this->ctl).quadrature_formula.size();
  std::vector<Tensor<1, dim>> phasefield_grads(n_q_points);

  // Define refinement criterion and mark cells to refine
  unsigned int will_refine = 0;
  double a1 = (this->ctl).params.refine_influence_initial;
  double a2 = (this->ctl).params.refine_influence_final;
  double phi_ref = std::exp(-a2) / std::exp(-a1);
  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      if (cell->diameter() < (this->ctl).params.l_phi *
                                 (this->ctl).params.refine_minimum_size_ratio) {
        cell->clear_refine_flag();
        continue;
      }
      fe_values.reinit(cell);
      fe_values[fracture.fields.extractors_scalar["phasefield"]]
          .get_function_gradients((fracture.solution), phasefield_grads);
      double max_grad = 0;
      for (unsigned int q = 0; q < n_q_points; ++q) {
        double prod = std::sqrt(phasefield_grads[q] * phasefield_grads[q]);
        max_grad = std::max(max_grad, prod);
      }
      if (max_grad > 1 / (this->ctl).params.l_phi * phi_ref * exp(-a1)) {
        cell->set_refine_flag();
        will_refine = 1;
      }
    }
  }
  (this->ctl).debug_dcout << "Refine - finish marking" << std::endl;
  double will_refine_global =
      Utilities::MPI::sum(will_refine, (this->ctl).mpi_com);
  if (!static_cast<bool>(will_refine_global)) {
    (this->ctl).dcout << "No cell to refine" << std::endl;
    return false;
  } else {
    (this->ctl).debug_dcout << "Refine - prepare" << std::endl;
    // Prepare transferring of point history
    parallel::distributed::ContinuousQuadratureDataTransfer<dim, PointHistory>
        point_history_transfer(FE_Q<dim>((this->ctl).params.poly_degree),
                               QGauss<dim>((this->ctl).params.poly_degree + 1),
                               QGauss<dim>((this->ctl).params.poly_degree + 1));
    point_history_transfer.prepare_for_coarsening_and_refinement(
        (this->ctl).triangulation, (this->ctl).quadrature_point_history);

    // Prepare transferring of fields
    parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
        soltrans = fracture.prepare_refine();

    (this->ctl).debug_dcout << "Refine - start refinement" << std::endl;
    // Execute refinement
    (this->ctl).triangulation.execute_coarsening_and_refinement();
    setup_system();

    (this->ctl).debug_dcout << "Refine - after refinement - point history"
                            << std::endl;
    // Finalize transferring of point history
    (this->ctl).initialize_point_history();
    point_history_transfer.interpolate();
    (this->ctl).debug_dcout << "Refine - after refinement - transfer fields"
                            << std::endl;
    // Finalize transferring of fields
    fracture.post_refine(soltrans, this->ctl);
    (this->ctl).debug_dcout << "Refine - done" << std::endl;
    return true;
  }
}

#endif // CRACKS_PHASE_FIELD_FRACTURE_MONOLITHIC_H
