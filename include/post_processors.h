//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_POST_PROCESSORS_H
#define CRACKS_POST_PROCESSORS_H

#include "constitutive_law.h"
#include "dealii_includes.h"
#include "utils.h"

template <int dim>
class StrainPostprocessor : public DataPostprocessorTensor<dim> {
public:
  StrainPostprocessor()
      : DataPostprocessorTensor<dim>("Strain", update_gradients) {}

  void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override {
    AssertDimension(input_data.solution_gradients.size(),
                    computed_quantities.size());

    for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {
      AssertDimension(computed_quantities[p].size(),
                      (Tensor<2, dim>::n_independent_components));
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
          computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(
              TableIndices<2>(d, e))] =
              (input_data.solution_gradients[p][d][e] +
               input_data.solution_gradients[p][e][d]) /
              2;
    }
  }
};

template <int dim>
class StressPostprocessor : public DataPostprocessorTensor<dim> {
public:
  ConstitutiveLaw<dim> constitutive_law;

  StressPostprocessor(ConstitutiveLaw<dim> &constitutive_law_input)
      : DataPostprocessorTensor<dim>("Stress", update_gradients),
        constitutive_law(constitutive_law_input) {}

  void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override {
    AssertDimension(input_data.solution_gradients.size(),
                    computed_quantities.size());

    for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {
      AssertDimension(computed_quantities[p].size(),
                      (Tensor<2, dim>::n_independent_components));

      Tensor<2, dim> grad_u;
      grad_u[0] = input_data.solution_gradients[p][0];
      grad_u[1] = input_data.solution_gradients[p][1];
      const Tensor<2, dim> Identity = Tensors ::get_Identity<dim>();
      const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
      const double tr_E = trace(E);
      Tensor<2, dim> stress = constitutive_law.lambda * tr_E * Identity +
                              2 * constitutive_law.mu * E;

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int e = 0; e < dim; ++e) {
          computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(
              TableIndices<2>(d, e))] = stress[d][e];
        }
    }
  }
};

#endif // CRACKS_POST_PROCESSORS_H
