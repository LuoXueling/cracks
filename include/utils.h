/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef SUPPORT_FUNCTIONS_H
#define SUPPORT_FUNCTIONS_H

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

bool contains(const std::string str1, const std::string str2) {
  std::string::size_type idx = str1.find(str2);
  if (idx != std::string::npos) {
    return true;
  } else {
    return false;
  }
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

template <int dim>
SymmetricTensor<4, dim> get_stress_strain_tensor(const double lambda,
                                                 const double mu)
{
  SymmetricTensor<4, dim> tmp;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                             ((i == l) && (j == k) ? mu : 0.0) +
                             ((i == j) && (k == l) ? lambda : 0.0));
  return tmp;
}


template <int dim>
inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                          const unsigned int   shape_func,
                                          const unsigned int   q_point)
{
  SymmetricTensor<2, dim> tmp;

  for (unsigned int i = 0; i < dim; ++i)
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] =
          (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
           fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
          2;

  return tmp;
}


template <int dim>
inline SymmetricTensor<2, dim>
get_strain(const std::vector<Tensor<1, dim>> &grad)
{
  Assert(grad.size() == dim, ExcInternalError());

  SymmetricTensor<2, dim> strain;
  for (unsigned int i = 0; i < dim; ++i)
    strain[i][i] = grad[i][i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

  return strain;
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

/**
 * An extension of std::cout to redirect outputs to a log file for pcout.
 * To be honest, it's not a perfect implementation, but it's enough.
 */
class DualOStream {
public:
  DualOStream(ConditionalOStream &stream, const std::string &log_file)
      : pcout(stream), is_active(stream.is_active()) {
    fout.open(log_file);
  }

  template <typename T> const DualOStream &operator<<(const T &t) const {
    pcout << t;
    if (is_active) {
      std::streambuf *oldcout;
      oldcout = std::cout.rdbuf(fout.rdbuf());
      std::cout << t;
      std::cout.rdbuf(oldcout);
    }
    return *this;
  }

  const DualOStream &operator<<(std::ostream &(*p)(std::ostream &)) const {
    pcout << p;
    if (is_active) {
      std::streambuf *oldcout;
      oldcout = std::cout.rdbuf(fout.rdbuf());
      std::cout << p;
      std::cout.rdbuf(oldcout);
    }
    return *this;
  }

  ConditionalOStream pcout;
  std::ofstream fout;

  bool is_active;
};

class DualTimerOutput : public TimerOutput {
public:
  DualTimerOutput(const MPI_Comm &mpi_communicator, ConditionalOStream &stream,
                  const OutputFrequency output_frequency,
                  const OutputType output_type)
      : TimerOutput(mpi_communicator, stream, output_frequency, output_type){};

  // Use the ofstream object from dcout
  void manual_print_summary(const std::ofstream &fout) {
    TimerOutput::print_summary();
    std::streambuf *oldcout;
    oldcout = std::cout.rdbuf(fout.rdbuf());
    TimerOutput::print_summary();
    std::cout.rdbuf(oldcout);
  }
};

inline bool checkFileExsit(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

#endif