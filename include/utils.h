/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef SUPPORT_FUNCTIONS_H
#define SUPPORT_FUNCTIONS_H

#include "dealii_includes.h"
#include <fstream>
#include <iostream>
#include <streambuf>
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

template <int dim> inline SymmetricTensor<2, dim> get_Identity() {
  SymmetricTensor<2, dim> identity;
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
                                                 const double mu) {
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
                                          const unsigned int shape_func,
                                          const unsigned int q_point) {
  SymmetricTensor<2, dim> tmp;

  for (unsigned int i = 0; i < dim; ++i)
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                   fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                  2;

  return tmp;
}

template <int dim>
inline SymmetricTensor<2, dim>
get_strain(const std::vector<Tensor<1, dim>> &grad) {
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

template <int dim>
void tensor_product(Tensor<2, dim> &kronecker, const Tensor<1, dim> &x,
                    const Tensor<1, dim> &y) {
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      kronecker[i][j] = x[i] * y[j];
    }
  }
};

inline bool checkFileExsit(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

// https://stackoverflow.com/questions/13665090/trying-to-write-stdout-and-file-at-the-same-time
struct teebuf
    : std::streambuf
{
  std::streambuf* sb1_;
  std::streambuf* sb2_;

  teebuf(std::streambuf* sb1, std::streambuf* sb2)
      : sb1_(sb1), sb2_(sb2) {
  }
  int overflow(int c) {
    typedef std::streambuf::traits_type traits;
    bool rc(true);
    if (!traits::eq_int_type(traits::eof(), c)) {
      traits::eq_int_type(this->sb1_->sputc(c), traits::eof())
          && (rc = false);
      traits::eq_int_type(this->sb2_->sputc(c), traits::eof())
          && (rc = false);
    }
    return rc? traits::not_eof(c): traits::eof();
  }
  int sync() {
    bool rc(true);
    this->sb1_->pubsync() != -1 || (rc = false);
    this->sb2_->pubsync() != -1 || (rc = false);
    return rc? 0: -1;
  }
};

#endif