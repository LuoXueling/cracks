//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_PHASE_FIELD_H
#define CRACKS_PHASE_FIELD_H

#include "abstract_field.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "degradation.h"
#include "fatigue_degradation.h"
#include "parameters.h"
#include "post_processors.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseField : public AbstractField<dim> {
public:
  PhaseField(std::string update_scheme, Controller<dim> &ctl);

  void assemble_newton_system(bool residual_only, LA::MPI::Vector &neumann_rhs,
                              Controller<dim> &ctl) override;
  void assemble_linear_system(Controller<dim> &ctl) override;
  unsigned int solve(NewtonInformation<dim> &info,
                     Controller<dim> &ctl) override;
  void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) override;

  void enforce_phase_field_limitation(Controller<dim> &ctl);
};

template <int dim>
PhaseField<dim>::PhaseField(std::string update_scheme, Controller<dim> &ctl)
    : AbstractField<dim>(1, "none", update_scheme, ctl) {}

template <int dim>
void PhaseField<dim>::assemble_linear_system(Controller<dim> &ctl) {
  (this->system_rhs) = 0;
  (this->system_matrix) = 0;

  FEValues<dim> fe_values((this->fe), ctl.quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<double> old_phasefield_values(n_q_points);
  // Old Newton grads
  std::vector<Tensor<1, dim>> old_phasefield_grads(n_q_points);

  std::vector<double> Nphi_kq(dofs_per_cell);
  std::vector<Tensor<1, dim>> Bphi_kq(dofs_per_cell);

  if (ctl.params.degradation != "quadratic") {
    AssertThrow(false,
                ExcInternalError("Cannot solve linear equations for phase "
                                 "field when degradation is not quadratic."))
  }
  if (ctl.params.enable_fatigue) {
    AssertThrow(false,
                ExcInternalError("Cannot solve linear equations for phase "
                                 "field when fatigue is activated."))
  }

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values.get_function_values((this->solution), old_phasefield_values);
      fe_values.get_function_gradients((this->solution), old_phasefield_grads);

      // Get history
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        double H = lqph[q]->get("Driving force", 0.0);

        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nphi_kq[k] = fe_values.shape_value(k, q);
          Bphi_kq[k] = fe_values.shape_grad(k, q);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            {
              cell_matrix(i, j) +=
                  (Nphi_kq[i] * Nphi_kq[j] * 2 * H +
                   Bphi_kq[i] * Bphi_kq[j] * ctl.params.Gc * ctl.params.l_phi +
                   Nphi_kq[i] * Nphi_kq[j] * ctl.params.Gc / ctl.params.l_phi) *
                  fe_values.JxW(q);
            }
          }

          cell_rhs(i) +=
              (Nphi_kq[i] * (2 * H + std::abs(2 * H)) / 2) * fe_values.JxW(q);

          lqph[q]->update("Phase field", old_phasefield_values[q]);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      (this->constraints_all)
          .distribute_local_to_global(cell_matrix, local_dof_indices,
                                      (this->system_matrix));
      (this->constraints_all)
          .distribute_local_to_global(cell_rhs, local_dof_indices,
                                      (this->system_rhs));
    }

  (this->system_matrix).compress(VectorOperation::add);
  (this->system_rhs).compress(VectorOperation::add);
}

template <int dim>
void PhaseField<dim>::assemble_newton_system(bool residual_only,
                                             LA::MPI::Vector &neumann_rhs,
                                             Controller<dim> &ctl) {
  (this->system_rhs) = 0;
  if (!residual_only) {
    (this->system_matrix) = 0;
  }

  FEValues<dim> fe_values((this->fe), ctl.quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).n_dofs_per_cell();
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<double> old_phasefield_values(n_q_points);
  // Old Newton grads
  std::vector<Tensor<1, dim>> old_phasefield_grads(n_q_points);

  std::vector<double> Nphi_kq(dofs_per_cell);
  std::vector<Tensor<1, dim>> Bphi_kq(dofs_per_cell);

  std::unique_ptr<Degradation<dim>> degradation =
      select_degradation<dim>(ctl.params.degradation);
  std::unique_ptr<FatigueDegradation<dim>> fatigue_degradation =
      select_fatigue_degradation<dim>(ctl.params.fatigue_degradation, ctl);
  std::unique_ptr<FatigueAccumulation<dim>> fatigue_accumulation =
      select_fatigue_accumulation<dim>(ctl.params.fatigue_accumulation, ctl);

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values.get_function_values((this->solution), old_phasefield_values);
      fe_values.get_function_gradients((this->solution), old_phasefield_grads);
      // Get history
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nphi_kq[k] = fe_values.shape_value(k, q);
          Bphi_kq[k] = fe_values.shape_grad(k, q);
        }

        double H = lqph[q]->get("Driving force", 0.0);
        double degrade = degradation->value(old_phasefield_values[q], ctl);
        double degrade_derivative =
            degradation->derivative(old_phasefield_values[q], ctl);
        double degrade_second_derivative =
            degradation->second_derivative(old_phasefield_values[q], ctl);

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

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          if (!residual_only) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              {
                cell_matrix(i, j) +=
                    (ctl.params.Gc * fatigue_degrade * ctl.params.l_phi *
                         Bphi_kq[i] * Bphi_kq[j] +
                     Nphi_kq[i] * Nphi_kq[j] *
                         (degrade_second_derivative * H +
                          ctl.params.Gc * fatigue_degrade / ctl.params.l_phi)) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) +=
              (degrade_derivative * H * Nphi_kq[i] +
               ctl.params.Gc * (fatigue_degrade * ctl.params.l_phi *
                                    old_phasefield_grads[q] * Bphi_kq[i] +
                                fatigue_degrade / ctl.params.l_phi *
                                    old_phasefield_values[q] * Nphi_kq[i])) *
              fe_values.JxW(q);
          //          if (ctl.params.enable_fatigue) {
          //            cell_rhs(i) += -ctl.params.Gc *
          //                           (fatigue_degrade_grad * ctl.params.l_phi
          //                           *
          //                            old_phasefield_grads[q] * Nphi_kq[i]) *
          //                           fe_values.JxW(q);
          //          }

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
unsigned int PhaseField<dim>::solve(NewtonInformation<dim> &info,
                                    Controller<dim> &ctl) {

  if (ctl.params.direct_solver) {
    if (info.system_matrix_rebuilt) {
      ctl.timer.enter_subsection("Factorization");
      this->direct_solver.initialize(this->system_matrix);
      ctl.timer.leave_subsection("Factorization");
    }
    ctl.timer.enter_subsection("Solve LUx=b");
    this->direct_solver.solve(this->system_solution, this->system_rhs);
    ctl.timer.leave_subsection("Solve LUx=b");
    return this->solver_control.last_step();
  } else {
    SolverControl solver_control((this->dof_handler).n_dofs(),
                                 1e-8 * (this->system_rhs).l2_norm());
    ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                       "system - preconditioner"
                    << std::endl;
    SolverGMRES<LA::MPI::Vector> solver(solver_control);
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.elliptic = true;
      data.higher_order_elements = true;
      data.smoother_sweeps = 2;
      data.aggregation_threshold = 0.02;
      (this->preconditioner).initialize((this->system_matrix), data);
    }
    ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                       "system - solve"
                    << std::endl;
    solver.solve((this->system_matrix), (this->system_solution),
                 (this->system_rhs), (this->preconditioner));
    ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                       "system - solve complete"
                    << std::endl;

    return solver_control.last_step();
  }
}

template <int dim>
void PhaseField<dim>::output_results(DataOut<dim> &data_out,
                                     Controller<dim> &ctl) {

  data_out.add_data_vector((this->dof_handler), (this->solution),
                           "Phase_field");
  PointHistoryProcessor<dim> hist_processor("Driving force", this->fe, ctl);
  hist_processor.add_data_scalar(this->solution, data_out, this->dof_handler,
                                 ctl);
  if (ctl.params.enable_fatigue) {
    PointHistoryProcessor<dim> fatigue_processor("Fatigue history", this->fe,
                                                 ctl);
    fatigue_processor.add_data_scalar(this->solution, data_out,
                                      this->dof_handler, ctl);
  }
}

template <int dim>
void PhaseField<dim>::enforce_phase_field_limitation(Controller<dim> &ctl) {
  typename DoFHandler<dim>::active_cell_iterator cell = (this->dof_handler)
                                                            .begin_active(),
                                                 endc =
                                                     (this->dof_handler).end();

  LA::MPI::Vector distributed_solution(this->locally_owned_dofs, ctl.mpi_com);
  distributed_solution = this->solution;

  std::vector<types::global_dof_index> local_dof_indices(
      (this->fe).dofs_per_cell);
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < (this->fe).dofs_per_cell; ++i) {
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

#endif // CRACKS_PHASE_FIELD_H
