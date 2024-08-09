//
// Created by xlluo on 24-8-5.
//

#ifndef CRACKS_NEWTON_VARIATIONS_H
#define CRACKS_NEWTON_VARIATIONS_H

#include "controller.h"
// #include "abstract_field.h"
#include "dealii_includes.h"
// #include "elasticity.h"
#include "utils.h"
using namespace dealii;

template <int dim> class NewtonInformation {
public:
  double residual;
  double old_residual;
  double new_residual;
  double i_step;
  unsigned int adjustment_step;
  unsigned int iterative_solver_nonlinear_step;
  bool system_matrix_rebuilt;
};

template <int dim> class NewtonVariation {
public:
  NewtonVariation(Controller<dim> &ctl) = default;

  virtual bool quit_newton(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return info.residual <= ctl.params.lower_bound_newton_residual;
  };
  virtual bool quit_adjustment(NewtonInformation<dim> &info,
                               Controller<dim> &ctl) {
    return info.new_residual < info.residual;
  }
  virtual void apply_increment(LA::MPI::Vector &negative_increment,
                               LA::MPI::Vector &solution,
                               LA::MPI::SparseMatrix &system_matrix,
                               LA::MPI::Vector &system_rhs,
                               LA::MPI::Vector &neumann_rhs,
                               NewtonInformation<dim> &info,
                               Controller<dim> &ctl) {
    solution -= negative_increment;
  };
  virtual bool re_solve(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return false;
  };
  virtual bool rebuild_jacobian(NewtonInformation<dim> &info,
                                Controller<dim> &ctl) {
    if (ctl.params.direct_solver) {
      return true;
    } else {
      if (info.i_step == 1 || (info.residual / info.old_residual) > 0.1) {
        return true;
      } else {
        return false;
      }
    }
  };
  virtual void prepare_next_adjustment(LA::MPI::Vector &negative_increment,
                                       LA::MPI::Vector &solution,
                                       LA::MPI::SparseMatrix &system_matrix,
                                       LA::MPI::Vector &system_rhs,
                                       LA::MPI::Vector &neumann_rhs,
                                       NewtonInformation<dim> &info,
                                       Controller<dim> &ctl) {
    throw SolverControl::NoConvergence(0, 0);
  };
  virtual bool give_up(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return (info.residual / info.old_residual > ctl.params.upper_newton_rho) &&
           (info.i_step > 1);
  };
};

template <int dim> class LineSearch : public NewtonVariation<dim> {
public:
  LineSearch(Controller<dim> &ctl) : NewtonVariation<dim>(ctl) {
    AssertThrow(ctl.params.linesearch_parameters != "",
                ExcInternalError("No damping factor is assigned."));
    std::istringstream iss(ctl.params.adjustment_method_parameters);
    iss >> damping;
  }

  bool quit_adjustment(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return info.new_residual < info.residual;
  }

  void prepare_next_adjustment(LA::MPI::Vector &negative_increment,
                               LA::MPI::Vector &solution,
                               LA::MPI::SparseMatrix &system_matrix,
                               LA::MPI::Vector &system_rhs,
                               LA::MPI::Vector &neumann_rhs,
                               NewtonInformation<dim> &info,
                               Controller<dim> &ctl) override {
    if (damping >= 1) {
      throw SolverControl::NoConvergence(0, 0);
    }
    solution += negative_increment;
    negative_increment *= damping;
  };
  double damping;
};

template <int dim> class ArcLengthControl : public NewtonVariation<dim> {
public:
  void apply_increment(LA::MPI::Vector &negative_increment,
                       LA::MPI::Vector &solution,
                       LA::MPI::SparseMatrix &system_matrix,
                       LA::MPI::Vector &system_rhs,
                       LA::MPI::Vector &neumann_rhs,
                       NewtonInformation<dim> &info,
                       Controller<dim> &ctl) override {
    if (info.adjustment_step == 0) {
      Delta_u = negative_increment;
      Delta_u = 0;
      Delta_lambda = 0;
      delta_lambda = 0;
      lambda = 1.0;
      delta_u = Delta_u;
      delta_ut = Delta_u;
      delta_ubar = Delta_u;
      delta_ubar -= negative_increment;
      delta_u = delta_ubar;
      full_neumann_rhs = neumann_rhs;
    } else {
      delta_ut = 0;
      delta_ubar = 0;
      //      Elasticity<dim>::solve_system(delta_ut, system_matrix,
      //      neumann_rhs);
      SolverControl solver_control;
      TrilinosWrappers::SolverDirect solver(solver_control);
      solver.solve(system_matrix, delta_ut, neumann_rhs);

      double neum_inner_prod = neumann_rhs * neumann_rhs;
      LA::MPI::Vector bar_add_Du = delta_ubar;
      bar_add_Du += Delta_u;
      LA::MPI::Vector twice_bar_add_Du = bar_add_Du;
      twice_bar_add_Du *= 2;
      double a = delta_u * delta_u + neum_inner_prod * std::pow(psi, 2);
      double b = twice_bar_add_Du * delta_ut +
                 2 * std::pow(psi, 2) * Delta_lambda * neum_inner_prod;

      double c =
          bar_add_Du * bar_add_Du +
          std::pow(psi, 2) * std::pow(Delta_lambda, 2) * neum_inner_prod -
          std::pow(Delta_l, 2);
      double d = b * b - 4 * a * c;
      double x1, x2;
      ctl.debug_dcout << "Quadratic formula: Delta=" << d << std::endl;
      if (d >= 0) {
        double temp = -0.5 * (b + sgn<double>(b) * std::sqrt(d));
        x1 = temp / a;
        x2 = c / temp;
        delta_u1 = 0;
        delta_u1 = delta_ut;
        delta_u1 *= x1;
        delta_u1 += delta_ubar;

        delta_u2 = 0;
        delta_u2 = delta_ut;
        delta_u2 *= x2;
        delta_u2 += delta_ubar;

        // dot 1
        LA::MPI::Vector delta_u_tmp = Delta_u;
        delta_u_tmp += delta_u1;
        double DOT1 = delta_u_tmp * Delta_u + std::pow(psi, 2) * Delta_lambda *
                                                  (Delta_lambda + x1) *
                                                  neum_inner_prod;

        // dot 2
        delta_u_tmp = Delta_u;
        delta_u_tmp += delta_u2;
        double DOT2 = delta_u_tmp * Delta_u + std::pow(psi, 2) * Delta_lambda *
                                                  (Delta_lambda + x2) *
                                                  neum_inner_prod;

        delta_lambda = (DOT1 > DOT2) ? x1 : x2;
        delta_u = (DOT1 > DOT2) ? delta_u1 : delta_u2;
        ctl.debug_dcout << "delta_lambda: " << delta_lambda
                        << " norm of delta_u: " << delta_u.l2_norm();
      } else {
        delta_lambda = 0;
        delta_u = delta_ubar;
      }
    }
    solution += delta_u;
    lambda += delta_lambda;
    lambda = std::min(lambda, 1.0);
    Delta_u = delta_u;
    Delta_lambda = delta_lambda;
    neumann_rhs = full_neumann_rhs;
    neumann_rhs *= lambda;
    ctl.dcout << "Lambda: " << lambda << std::endl;
  };

  virtual bool quit_adjustment(NewtonInformation<dim> &info,
                               Controller<dim> &ctl) {
    return (info.new_residual < info.residual) || lambda >= 1;
  }

  bool re_solve(NewtonInformation<dim> &info, Controller<dim> &ctl) override {
    return (info.adjustment_step == 0 || quit_adjustment(info, ctl)) ? false
                                                                     : true;
  }

  void prepare_next_adjustment(LA::MPI::Vector &negative_increment,
                               LA::MPI::Vector &solution,
                               LA::MPI::SparseMatrix &system_matrix,
                               LA::MPI::Vector &system_rhs,
                               LA::MPI::Vector &neumann_rhs,
                               NewtonInformation<dim> &info,
                               Controller<dim> &ctl) override {};

  virtual bool give_up(NewtonInformation<dim> &info, Controller<dim> &ctl) {
    return (info.residual / info.old_residual > ctl.params.upper_newton_rho) &&
           (info.i_step > 1);
  };

  LA::MPI::Vector full_neumann_rhs;
  LA::MPI::Vector Delta_u, delta_u, delta_u1, delta_u2, delta_ut, delta_ubar;
  double Delta_lambda = 0, delta_lambda = 0;
  double lambda = 1.0;
  double psi = 1;
  double Delta_l = 0.1;
};

template <int dim>
std::unique_ptr<NewtonVariation<dim>>
select_newton_variation(std::string method, Controller<dim> &ctl) {
  if (method == "none")
    return std::make_unique<NewtonVariation<dim>>(ctl);
  else if (method == "linesearch")
    return std::make_unique<LineSearch<dim>>(ctl);
  else if (method == "arclength")
    return std::make_unique<ArcLengthControl<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_NEWTON_VARIATIONS_H
