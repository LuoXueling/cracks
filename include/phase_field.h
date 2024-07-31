//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_PHASE_FIELD_H
#define CRACKS_PHASE_FIELD_H

#include "abstract_field.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "parameters.h"
#include "post_processors.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class PhaseField : public AbstractField<dim> {
public:
  explicit PhaseField(Controller<dim> &ctl);

  void assemble_system(bool residual_only, Controller<dim> &ctl) override;
  unsigned int solve(Controller<dim> &ctl) override;
  void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) override;

  void enforce_phase_field_limitation(Controller<dim> &ctl);
};

template <int dim>
PhaseField<dim>::PhaseField(Controller<dim> &ctl)
    : AbstractField<dim>(1, "none", ctl) {}

template <int dim>
void PhaseField<dim>::assemble_system(bool residual_only,
                                      Controller<dim> &ctl) {
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

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values.get_function_values((this->solution),
                                                old_phasefield_values);
      fe_values.get_function_gradients((this->solution),
                                                   old_phasefield_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Get history
        const std::vector<std::shared_ptr<PointHistory>> lqph =
            ctl.quadrature_point_history.get_data(cell);

        double H = lqph[q]->get("Driving force", 0.0);

        // Values of fields and their derivatives
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          Nphi_kq[k] = fe_values.shape_value(k, q);
          Bphi_kq[k] = fe_values.shape_grad(k, q);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          if (!residual_only) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              {
                cell_matrix(i, j) +=
                    (ctl.params.Gc * ctl.params.l_phi * Bphi_kq[i] *
                         Bphi_kq[j] +
                     Nphi_kq[i] * Nphi_kq[j] *
                         (2 * H + ctl.params.Gc / ctl.params.l_phi)) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) +=
              (-2 * (1 - old_phasefield_values[q]) * Nphi_kq[i] * H +
               ctl.params.Gc *
                   (ctl.params.l_phi * old_phasefield_grads[q] * Bphi_kq[i] +
                    1 / ctl.params.l_phi * old_phasefield_values[q] *
                        Nphi_kq[i])) *
              fe_values.JxW(q);
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

template <int dim> unsigned int PhaseField<dim>::solve(Controller<dim> &ctl) {

  SolverControl solver_control((this->dof_handler).n_dofs(),
                               1e-8 * (this->system_rhs).l2_norm());
  SolverGMRES<LA::MPI::Vector> solver(solver_control);
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.elliptic = true;
    data.higher_order_elements = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
    (this->preconditioner).initialize((this->system_matrix), data);
  }

  solver.solve((this->system_matrix), (this->increment), (this->system_rhs),
               (this->preconditioner));
}

template <int dim>
void PhaseField<dim>::output_results(DataOut<dim> &data_out,
                                     Controller<dim> &ctl) {

  data_out.add_data_vector((this->dof_handler), (this->solution),
                           "Phase_field");
}

template <int dim>
void PhaseField<dim>::enforce_phase_field_limitation(Controller<dim> &ctl) {
  typename DoFHandler<dim>::active_cell_iterator cell = (this->dof_handler)
                                                            .begin_active(),
                                                 endc = (this->dof_handler).end();

  LA::MPI::Vector distributed_solution(this->locally_owned_dofs, ctl.mpi_com);
  distributed_solution = this->solution;

  std::vector<types::global_dof_index> local_dof_indices((this->fe).dofs_per_cell);
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
