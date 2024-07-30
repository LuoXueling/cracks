//
// Created by xlluo on 24-7-29.
//

#ifndef CRACKS_CONSTITUTIVE_LAW_H
#define CRACKS_CONSTITUTIVE_LAW_H

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

template <int dim> class ConstitutiveLaw {
public:
  ConstitutiveLaw(const double E, const double nu, std::string plane_state);

  void
  get_stress_strain_tensor(const SymmetricTensor<2, dim> &strain_tensor,
                           SymmetricTensor<2, dim> &stress_tensor,
                           SymmetricTensor<4, dim> &stress_strain_tensor) const;

  double E;
  double nu;
  double kappa;
  const double mu;
  double lambda;
  std::string plane_state;

private:
  SymmetricTensor<4, dim> stress_strain_tensor_kappa;
  SymmetricTensor<4, dim> stress_strain_tensor_mu;
};

template <int dim>
ConstitutiveLaw<dim>::ConstitutiveLaw(double E_in, double nu_in,
                                      std::string plane_state_in)
    : E(E_in), nu(nu_in), plane_state(plane_state_in), mu(E / (2 * (1 + nu))) {
  double kappa_3d = E / (3 * (1 - 2 * nu));
  if (dim == 3) {
    lambda = E * nu / (1 + nu) / (1 - 2 * nu);
    kappa = kappa_3d;
    stress_strain_tensor_kappa =
        kappa * outer_product(unit_symmetric_tensor<dim>(),
                              unit_symmetric_tensor<dim>());
    stress_strain_tensor_mu =
        2 * mu *
        (identity_tensor<dim>() - outer_product(unit_symmetric_tensor<dim>(),
                                                unit_symmetric_tensor<dim>()) /
                                      3.0);
  } else if (dim == 2) {
    lambda = E * nu / (1 + nu) /
             (1 - 2 * nu); // This is actually wrong for 2D. But since everyone
                           // is using this formula, I just keep it here.
    if (plane_state == "stress") {
      kappa = 9 * kappa_3d * mu / (3 * kappa_3d + 4 * mu);
    } else {
      kappa = kappa_3d + mu / 3; // kappa = E / (2 * (1 - nu))
    }
    //    kappa = E / (2 * (1 - nu));
    stress_strain_tensor_kappa =
        kappa * outer_product(unit_symmetric_tensor<dim>(),
                              unit_symmetric_tensor<dim>());
    stress_strain_tensor_mu =
        2 * mu *
        (identity_tensor<dim>() - outer_product(unit_symmetric_tensor<dim>(),
                                                unit_symmetric_tensor<dim>()) /
                                      2);
  } else
    AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void ConstitutiveLaw<dim>::get_stress_strain_tensor(
    const SymmetricTensor<2, dim> &strain_tensor,
    SymmetricTensor<2, dim> &stress_tensor,
    SymmetricTensor<4, dim> &stress_strain_tensor) const {

  stress_strain_tensor = stress_strain_tensor_mu + stress_strain_tensor_kappa;
  stress_tensor = stress_strain_tensor * strain_tensor;
}

#endif // CRACKS_CONSTITUTIVE_LAW_H
