//
// Created by xlluo on 2022/3/26.
//

#ifndef CRACKS_SUPPORT_FUNCTIONS_H
#define CRACKS_SUPPORT_FUNCTIONS_H

#include <deal.II/numerics/vector_tools.h>
using namespace dealii;

namespace compatibility {
/**
 * Split the set of DoFs (typically locally owned or relevant) in @p whole_set
 * into blocks given by the @p dofs_per_block structure.
 */
void split_by_block(const std::vector<types::global_dof_index> &dofs_per_block,
                    const IndexSet &whole_set,
                    std::vector<IndexSet> &partitioned) {
  const unsigned int n_blocks = dofs_per_block.size();
  partitioned.clear();

  partitioned.resize(n_blocks);
  types::global_dof_index start = 0;
  for (unsigned int i = 0; i < n_blocks; ++i) {
    partitioned[i] = whole_set.get_view(start, start + dofs_per_block[i]);
    start += dofs_per_block[i];
  }
}
} // namespace compatibility

namespace compatibility {
template <int dim> using ZeroFunction = dealii::Functions::ZeroFunction<dim>;
}

namespace LA {
using namespace dealii::LinearAlgebraTrilinos;
}

// Define some tensors for cleaner notation later.
namespace Tensors {

template <int dim>
inline Tensor<1, dim> get_grad_pf(
    unsigned int q,
    const std::vector<std::vector<Tensor<1, dim>>> &old_solution_grads) {
  Tensor<1, dim> grad_pf;
  grad_pf[0] = old_solution_grads[q][dim][0];
  grad_pf[1] = old_solution_grads[q][dim][1];
  if (dim == 3)
    grad_pf[2] = old_solution_grads[q][dim][2];

  return grad_pf;
}

template <int dim>
inline Tensor<2, dim>
get_grad_u(unsigned int q,
           const std::vector<std::vector<Tensor<1, dim>>> &old_solution_grads) {
  Tensor<2, dim> grad_u;
  grad_u[0][0] = old_solution_grads[q][0][0];
  grad_u[0][1] = old_solution_grads[q][0][1];

  grad_u[1][0] = old_solution_grads[q][1][0];
  grad_u[1][1] = old_solution_grads[q][1][1];
  if (dim == 3) {
    grad_u[0][2] = old_solution_grads[q][0][2];

    grad_u[1][2] = old_solution_grads[q][1][2];

    grad_u[2][0] = old_solution_grads[q][2][0];
    grad_u[2][1] = old_solution_grads[q][2][1];
    grad_u[2][2] = old_solution_grads[q][2][2];
  }

  return grad_u;
}

template <int dim> inline Tensor<2, dim> get_Identity() {
  Tensor<2, dim> identity;
  identity[0][0] = 1.0;
  identity[1][1] = 1.0;
  if (dim == 3)
    identity[2][2] = 1.0;

  return identity;
}

template <int dim>
inline Tensor<1, dim>
get_u(unsigned int q, const std::vector<Vector<double>> &old_solution_values) {
  Tensor<1, dim> u;
  u[0] = old_solution_values[q](0);
  u[1] = old_solution_values[q](1);
  if (dim == 3)
    u[2] = old_solution_values[q](2);

  return u;
}

template <int dim>
inline Tensor<1, dim> get_u_LinU(const Tensor<1, dim> &phi_i_u) {
  Tensor<1, dim> tmp;
  tmp[0] = phi_i_u[0];
  tmp[1] = phi_i_u[1];
  if (dim == 3)
    tmp[2] = phi_i_u[2];
  return tmp;
}

template <int dim> inline double get_divergence_u(const Tensor<2, dim> grad_u) {
  double tmp;
  if (dim == 2) {
    tmp = grad_u[0][0] + grad_u[1][1];
  } else if (dim == 3) {
    tmp = grad_u[0][0] + grad_u[1][1] + grad_u[2][2];
  }

  return tmp;
}

} // namespace Tensors

// Now, there follow several functions to perform
// the spectral decomposition of the stress tensor
// into tension and compression parts
// assumes the matrix is symmetric!
// The explicit calculation does only work
// in 2d. For 3d, we should use other libraries or approximative
// tools to compute eigenvectors and -functions.
// Borden et al. (2012, 2013) suggested some papers to look into.
template <int dim>
void eigen_vectors_and_values(double &E_eigenvalue_1, double &E_eigenvalue_2,
                              Tensor<2, dim> &ev_matrix,
                              const Tensor<2, dim> &matrix) {
  // Compute eigenvectors
  Tensor<1, dim> E_eigenvector_1;
  Tensor<1, dim> E_eigenvector_2;
  if (std::abs(matrix[0][1]) < 1e-10 * std::abs(matrix[0][0]) ||
      std::abs(matrix[0][1]) < 1e-10 * std::abs(matrix[1][1])) {
    // E is close to diagonal
    E_eigenvalue_1 = matrix[0][0];
    E_eigenvector_1[0] = 1;
    E_eigenvector_1[1] = 0;
    E_eigenvalue_2 = matrix[1][1];
    E_eigenvector_2[0] = 0;
    E_eigenvector_2[1] = 1;
  } else {
    double sq = std::sqrt((matrix[0][0] - matrix[1][1]) *
                              (matrix[0][0] - matrix[1][1]) +
                          4.0 * matrix[0][1] * matrix[1][0]);
    E_eigenvalue_1 = 0.5 * ((matrix[0][0] + matrix[1][1]) + sq);
    E_eigenvalue_2 = 0.5 * ((matrix[0][0] + matrix[1][1]) - sq);

    E_eigenvector_1[0] =
        1.0 /
        (std::sqrt(1 + (E_eigenvalue_1 - matrix[0][0]) / matrix[0][1] *
                           (E_eigenvalue_1 - matrix[0][0]) / matrix[0][1]));
    E_eigenvector_1[1] =
        (E_eigenvalue_1 - matrix[0][0]) /
        (matrix[0][1] *
         (std::sqrt(1 + (E_eigenvalue_1 - matrix[0][0]) / matrix[0][1] *
                            (E_eigenvalue_1 - matrix[0][0]) / matrix[0][1])));
    E_eigenvector_2[0] =
        1.0 /
        (std::sqrt(1 + (E_eigenvalue_2 - matrix[0][0]) / matrix[0][1] *
                           (E_eigenvalue_2 - matrix[0][0]) / matrix[0][1]));
    E_eigenvector_2[1] =
        (E_eigenvalue_2 - matrix[0][0]) /
        (matrix[0][1] *
         (std::sqrt(1 + (E_eigenvalue_2 - matrix[0][0]) / matrix[0][1] *
                            (E_eigenvalue_2 - matrix[0][0]) / matrix[0][1])));
  }

  ev_matrix[0][0] = E_eigenvector_1[0];
  ev_matrix[0][1] = E_eigenvector_2[0];
  ev_matrix[1][0] = E_eigenvector_1[1];
  ev_matrix[1][1] = E_eigenvector_2[1];

  // Sanity check if orthogonal
  double scalar_prod = 1.0e+10;
  scalar_prod = E_eigenvector_1[0] * E_eigenvector_2[0] +
                E_eigenvector_1[1] * E_eigenvector_2[1];

  if (scalar_prod > 1.0e-6) {
    std::cout << "Seems not to be orthogonal" << std::endl;
    abort();
  }
}

template <int dim>
void decompose_stress(Tensor<2, dim> &stress_term_plus,
                      Tensor<2, dim> &stress_term_minus,
                      const Tensor<2, dim> &E, const double tr_E,
                      const Tensor<2, dim> &E_LinU, const double tr_E_LinU,
                      const double lame_coefficient_lambda,
                      const double lame_coefficient_mu, const bool derivative) {
  static const Tensor<2, dim> Identity = Tensors::get_Identity<dim>();

  Tensor<2, dim> zero_matrix;
  zero_matrix.clear();

  // Compute first the eigenvalues for u (as in the previous function)
  // and then for \delta u

  // Compute eigenvalues/vectors
  double E_eigenvalue_1, E_eigenvalue_2;
  Tensor<2, dim> P_matrix;
  eigen_vectors_and_values(E_eigenvalue_1, E_eigenvalue_2, P_matrix, E);

  double E_eigenvalue_1_plus = std::max(0.0, E_eigenvalue_1);
  double E_eigenvalue_2_plus = std::max(0.0, E_eigenvalue_2);

  Tensor<2, dim> Lambda_plus;
  Lambda_plus[0][0] = E_eigenvalue_1_plus;
  Lambda_plus[0][1] = 0.0;
  Lambda_plus[1][0] = 0.0;
  Lambda_plus[1][1] = E_eigenvalue_2_plus;

  if (!derivative) {
    Tensor<2, dim> E_plus = P_matrix * Lambda_plus * transpose(P_matrix);

    double tr_E_positive = std::max(0.0, tr_E);

    stress_term_plus = lame_coefficient_lambda * tr_E_positive * Identity +
                       2 * lame_coefficient_mu * E_plus;

    stress_term_minus =
        lame_coefficient_lambda * (tr_E - tr_E_positive) * Identity +
        2 * lame_coefficient_mu * (E - E_plus);
  } else {
    // Derviatives (\delta u)

    // Compute eigenvalues/vectors
    double E_eigenvalue_1_LinU, E_eigenvalue_2_LinU;
    Tensor<1, dim> E_eigenvector_1_LinU;
    Tensor<1, dim> E_eigenvector_2_LinU;
    Tensor<2, dim> P_matrix_LinU;

    // Compute linearized Eigenvalues
    double diskriminante = std::sqrt(
        E[0][1] * E[1][0] + (E[0][0] - E[1][1]) * (E[0][0] - E[1][1]) / 4.0);

    E_eigenvalue_1_LinU =
        0.5 * tr_E_LinU +
        1.0 / (2.0 * diskriminante) *
            (E_LinU[0][1] * E[1][0] + E[0][1] * E_LinU[1][0] +
             (E[0][0] - E[1][1]) * (E_LinU[0][0] - E_LinU[1][1]) / 2.0);

    E_eigenvalue_2_LinU =
        0.5 * tr_E_LinU -
        1.0 / (2.0 * diskriminante) *
            (E_LinU[0][1] * E[1][0] + E[0][1] * E_LinU[1][0] +
             (E[0][0] - E[1][1]) * (E_LinU[0][0] - E_LinU[1][1]) / 2.0);

    // Compute normalized Eigenvectors and P
    double normalization_1 =
        1.0 / (std::sqrt(1 + (E_eigenvalue_1 - E[0][0]) / E[0][1] *
                                 (E_eigenvalue_1 - E[0][0]) / E[0][1]));
    double normalization_2 =
        1.0 / (std::sqrt(1 + (E_eigenvalue_2 - E[0][0]) / E[0][1] *
                                 (E_eigenvalue_2 - E[0][0]) / E[0][1]));

    double normalization_1_LinU = 0.0;
    double normalization_2_LinU = 0.0;

    normalization_1_LinU =
        -1.0 *
        (1.0 /
         (1.0 + (E_eigenvalue_1 - E[0][0]) / E[0][1] *
                    (E_eigenvalue_1 - E[0][0]) / E[0][1]) *
         1.0 /
         (2.0 * std::sqrt(1.0 + (E_eigenvalue_1 - E[0][0]) / E[0][1] *
                                    (E_eigenvalue_1 - E[0][0]) / E[0][1])) *
         (2.0 * (E_eigenvalue_1 - E[0][0]) / E[0][1]) *
         ((E_eigenvalue_1_LinU - E_LinU[0][0]) * E[0][1] -
          (E_eigenvalue_1 - E[0][0]) * E_LinU[0][1]) /
         (E[0][1] * E[0][1]));

    normalization_2_LinU =
        -1.0 *
        (1.0 /
         (1.0 + (E_eigenvalue_2 - E[0][0]) / E[0][1] *
                    (E_eigenvalue_2 - E[0][0]) / E[0][1]) *
         1.0 /
         (2.0 * std::sqrt(1.0 + (E_eigenvalue_2 - E[0][0]) / E[0][1] *
                                    (E_eigenvalue_2 - E[0][0]) / E[0][1])) *
         (2.0 * (E_eigenvalue_2 - E[0][0]) / E[0][1]) *
         ((E_eigenvalue_2_LinU - E_LinU[0][0]) * E[0][1] -
          (E_eigenvalue_2 - E[0][0]) * E_LinU[0][1]) /
         (E[0][1] * E[0][1]));

    E_eigenvector_1_LinU[0] = normalization_1 * 1.0;
    E_eigenvector_1_LinU[1] =
        normalization_1 * (E_eigenvalue_1 - E[0][0]) / E[0][1];

    E_eigenvector_2_LinU[0] = normalization_2 * 1.0;
    E_eigenvector_2_LinU[1] =
        normalization_2 * (E_eigenvalue_2 - E[0][0]) / E[0][1];

    // Apply product rule to normalization and vector entries
    double EV_1_part_1_comp_1 = 0.0; // LinU in vector entries, normalization U
    double EV_1_part_1_comp_2 = 0.0; // LinU in vector entries, normalization U
    double EV_1_part_2_comp_1 = 0.0; // vector entries U, normalization LinU
    double EV_1_part_2_comp_2 = 0.0; // vector entries U, normalization LinU

    double EV_2_part_1_comp_1 = 0.0; // LinU in vector entries, normalization U
    double EV_2_part_1_comp_2 = 0.0; // LinU in vector entries, normalization U
    double EV_2_part_2_comp_1 = 0.0; // vector entries U, normalization LinU
    double EV_2_part_2_comp_2 = 0.0; // vector entries U, normalization LinU

    // Effizienter spaeter, aber erst einmal uebersichtlich und verstehen!
    EV_1_part_1_comp_1 = normalization_1 * 0.0;
    EV_1_part_1_comp_2 = normalization_1 *
                         ((E_eigenvalue_1_LinU - E_LinU[0][0]) * E[0][1] -
                          (E_eigenvalue_1 - E[0][0]) * E_LinU[0][1]) /
                         (E[0][1] * E[0][1]);

    EV_1_part_2_comp_1 = normalization_1_LinU * 1.0;
    EV_1_part_2_comp_2 =
        normalization_1_LinU * (E_eigenvalue_1 - E[0][0]) / E[0][1];

    EV_2_part_1_comp_1 = normalization_2 * 0.0;
    EV_2_part_1_comp_2 = normalization_2 *
                         ((E_eigenvalue_2_LinU - E_LinU[0][0]) * E[0][1] -
                          (E_eigenvalue_2 - E[0][0]) * E_LinU[0][1]) /
                         (E[0][1] * E[0][1]);

    EV_2_part_2_comp_1 = normalization_2_LinU * 1.0;
    EV_2_part_2_comp_2 =
        normalization_2_LinU * (E_eigenvalue_2 - E[0][0]) / E[0][1];

    // Build eigenvectors
    E_eigenvector_1_LinU[0] = EV_1_part_1_comp_1 + EV_1_part_2_comp_1;
    E_eigenvector_1_LinU[1] = EV_1_part_1_comp_2 + EV_1_part_2_comp_2;

    E_eigenvector_2_LinU[0] = EV_2_part_1_comp_1 + EV_2_part_2_comp_1;
    E_eigenvector_2_LinU[1] = EV_2_part_1_comp_2 + EV_2_part_2_comp_2;

    // P-Matrix
    P_matrix_LinU[0][0] = E_eigenvector_1_LinU[0];
    P_matrix_LinU[0][1] = E_eigenvector_2_LinU[0];
    P_matrix_LinU[1][0] = E_eigenvector_1_LinU[1];
    P_matrix_LinU[1][1] = E_eigenvector_2_LinU[1];

    double E_eigenvalue_1_plus_LinU = 0.0;
    double E_eigenvalue_2_plus_LinU = 0.0;

    // Very important: Set E_eigenvalue_1_plus_LinU to zero when
    // the corresponding rhs-value is set to zero and NOT when
    // the value itself is negative!!!
    if (E_eigenvalue_1 < 0.0) {
      E_eigenvalue_1_plus_LinU = 0.0;
    } else
      E_eigenvalue_1_plus_LinU = E_eigenvalue_1_LinU;

    if (E_eigenvalue_2 < 0.0) {
      E_eigenvalue_2_plus_LinU = 0.0;
    } else
      E_eigenvalue_2_plus_LinU = E_eigenvalue_2_LinU;

    Tensor<2, dim> Lambda_plus_LinU;
    Lambda_plus_LinU[0][0] = E_eigenvalue_1_plus_LinU;
    Lambda_plus_LinU[0][1] = 0.0;
    Lambda_plus_LinU[1][0] = 0.0;
    Lambda_plus_LinU[1][1] = E_eigenvalue_2_plus_LinU;

    Tensor<2, dim> E_plus_LinU =
        P_matrix_LinU * Lambda_plus * transpose(P_matrix) +
        P_matrix * Lambda_plus_LinU * transpose(P_matrix) +
        P_matrix * Lambda_plus * transpose(P_matrix_LinU);

    double tr_E_positive_LinU = 0.0;
    if (tr_E < 0.0) {
      tr_E_positive_LinU = 0.0;

    } else
      tr_E_positive_LinU = tr_E_LinU;

    stress_term_plus = lame_coefficient_lambda * tr_E_positive_LinU * Identity +
                       2 * lame_coefficient_mu * E_plus_LinU;

    stress_term_minus =
        lame_coefficient_lambda * (tr_E_LinU - tr_E_positive_LinU) * Identity +
        2 * lame_coefficient_mu * (E_LinU - E_plus_LinU);

    // Sanity check
    // Tensor<2,dim> stress_term = lame_coefficient_lambda * tr_E_LinU *
    // Identity
    //  + 2 * lame_coefficient_mu * E_LinU;

    // std::cout << stress_term.norm() << "   " << stress_term_plus.norm() << "
    // " << stress_term_minus.norm() << std::endl;
  }
}

int value_to_bucket(double x, unsigned int n_buckets) {
  const double x1 = -1.5;
  const double x2 = 1.5;
  return static_cast<int>(std::floor((x - x1) / (x2 - x1) * n_buckets + 0.5));
}

double bucket_to_value(unsigned int idx, unsigned int n_buckets) {
  const double x1 = -1.5;
  const double x2 = 1.5;
  return x1 + idx * (x2 - x1) / n_buckets;
}

#endif // CRACKS_SUPPORT_FUNCTIONS_H
