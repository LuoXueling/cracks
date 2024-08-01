//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_ELASTICITY_H
#define CRACKS_ELASTICITY_H

#include "abstract_field.h"
#include "constitutive_law.h"
#include "dealii_includes.h"
#include "parameters.h"
#include "post_processors.h"
#include "utils.h"
#include <fstream>
#include <iostream>
using namespace dealii;

template <int dim> class Elasticity : public AbstractField<dim> {
public:
  Elasticity(unsigned int n_components, std::string boundary_from,
             std::string update_scheme, Controller<dim> &ctl);

  void assemble_newton_system(bool residual_only,
                              Controller<dim> &ctl) override;
  unsigned int solve(Controller<dim> &ctl) override;
  void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) override;

  void compute_load(Controller<dim> &ctl);

  ConstitutiveLaw<dim> constitutive_law;

  StrainPostprocessor<dim> strain;
  StressPostprocessor<dim> stress;
};

template <int dim>
Elasticity<dim>::Elasticity(const unsigned int n_components,
                            std::string boundary_from,
                            std::string update_scheme, Controller<dim> &ctl)
    : AbstractField<dim>(n_components, boundary_from, update_scheme, ctl),
      constitutive_law(ctl.params.E, ctl.params.v, ctl.params.plane_state),
      stress(constitutive_law) {}

template <int dim>
void Elasticity<dim>::assemble_newton_system(bool residual_only,
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

  for (const auto &cell : (this->dof_handler).active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values[displacement].get_function_values((this->solution),
                                                  old_displacement_values);
      fe_values[displacement].get_function_gradients((this->solution),
                                                     old_displacement_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Get history
        const std::vector<std::shared_ptr<PointHistory>> lqph =
            ctl.quadrature_point_history.get_data(cell);

        double phasefield = lqph[q]->get("Phase field", 0.0);
        double degradation = pow(1 - phasefield, 2) + ctl.params.constant_k;

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
        Tensor<2, dim> stress_0 = constitutive_law.lambda * tr_E * Identity +
                                  2 * constitutive_law.mu * E;

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const Tensor<2, dim> Bu_iq_symmetric =
              0.5 * (Bu_kq[i] + transpose(Bu_kq[i]));
          const double Bu_iq_symmetric_tr = trace(Bu_iq_symmetric);

          if (!residual_only) {

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              {
                cell_matrix(i, j) +=
                    degradation *
                    (scalar_product(constitutive_law.lambda *
                                            Bu_iq_symmetric_tr * Identity +
                                        2 * constitutive_law.mu *
                                            Bu_iq_symmetric,
                                    Bu_kq[j])) *
                    fe_values.JxW(q);
              }
            }
          }

          cell_rhs(i) += degradation * (scalar_product(Bu_kq[i], stress_0) *
                                        fe_values.JxW(q));
        }

        // Update history
        lqph[q]->update("Driving force", 0.5 * scalar_product(stress_0, E));
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

template <int dim> unsigned int Elasticity<dim>::solve(Controller<dim> &ctl) {

  SolverControl solver_control((this->dof_handler).n_dofs(),
                               1e-8 * (this->system_rhs).l2_norm());
  ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                     "system - preconditioner"
                  << std::endl;
  SolverGMRES<LA::MPI::Vector> solver(solver_control);
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.constant_modes = (this->constant_modes);
    data.elliptic = true;
    data.higher_order_elements = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
    (this->preconditioner).initialize((this->system_matrix), data);
  }
  ctl.debug_dcout
      << "Solve Newton system - Newton iteration - solve linear system - solve"
      << std::endl;
  solver.solve((this->system_matrix), (this->increment), (this->system_rhs),
               (this->preconditioner));
  ctl.debug_dcout << "Solve Newton system - Newton iteration - solve linear "
                     "system - solve complete"
                  << std::endl;

  return solver_control.last_step();
}

template <int dim>
void Elasticity<dim>::output_results(DataOut<dim> &data_out,
                                     Controller<dim> &ctl) {
  std::vector<std::string> solution_names(dim, "Displacement");
  ctl.debug_dcout << "Computing output - elasticity - displacement"
                  << std::endl;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector((this->dof_handler), (this->solution),
                           solution_names, data_component_interpretation);
  ctl.debug_dcout << "Computing output - elasticity - strain" << std::endl;
  data_out.add_data_vector((this->dof_handler), (this->solution), strain);
  ctl.debug_dcout << "Computing output - elasticity - stress" << std::endl;
  data_out.add_data_vector((this->dof_handler), (this->solution), stress);
  // Record statistics
  ctl.debug_dcout << "Computing output - elasticity - load" << std::endl;
  compute_load(ctl);
}

template <int dim> void Elasticity<dim>::compute_load(Controller<dim> &ctl) {
  const QGauss<dim - 1> face_quadrature_formula(ctl.params.poly_degree + 1);
  FEFaceValues<dim> fe_face_values((this->fe), face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = (this->fe).dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<2, dim>> face_solution_grads(n_face_q_points);

  std::map<int, Tensor<1, dim>> load_value;

  const Tensor<2, dim> Identity = Tensors::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell = (this->dof_handler)
                                                            .begin_active(),
                                                 endc =
                                                     (this->dof_handler).end();
  for (const int id : ctl.boundary_ids)
    load_value[id] = Tensor<1, dim>();
  const FEValuesExtractors::Vector displacement(0);
  ctl.debug_dcout << "Computing output - elasticity - load - computing"
                  << std::endl;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() != 0) {
          fe_face_values.reinit(cell, face);
          fe_face_values[displacement].get_function_gradients(
              (this->solution), face_solution_grads);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
            const Tensor<2, dim> grad_u = face_solution_grads[q_point];

            const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
            const double tr_E = trace(E);

            Tensor<2, dim> stress_term;
            stress_term = constitutive_law.lambda * tr_E * Identity +
                          2 * constitutive_law.mu * E;

            load_value[cell->face(face)->boundary_id()] +=
                stress_term * fe_face_values.normal_vector(q_point) *
                fe_face_values.JxW(q_point);
          }
        }
    }
  ctl.debug_dcout << "Computing output - elasticity - load - recording"
                  << std::endl;
  typename std::map<int, Tensor<1, dim>>::iterator it;
  for (it = load_value.begin(); it != load_value.end(); it++) {
    ctl.debug_dcout << "Computing output - elasticity - load - recording - item"
                    << std::endl;
    for (int i = 0; i < dim; ++i) {
      ctl.debug_dcout
          << "Computing output - elasticity - load - recording - dim - sum"
          << std::endl;
      double load = Utilities::MPI::sum(it->second[i], ctl.mpi_com) * -1;
      ctl.debug_dcout
          << "Computing output - elasticity - load - recording - dim - record"
          << std::endl;
      std::ostringstream stringStream;
      stringStream << "Boundary-" << it->first << "-Dir-" << i;
      (ctl.statistics).add_value(stringStream.str(), load);
      (ctl.statistics).set_precision(stringStream.str(), 8);
      (ctl.statistics).set_scientific(stringStream.str(), true);
    }
  }
}

#endif // CRACKS_ELASTICITY_H
