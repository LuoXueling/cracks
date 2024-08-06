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
};

template <int dim> class NewtonVariation {
public:
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
    if (info.i_step == 1 || (info.residual / info.old_residual) > 0.1) {
      return true;
    } else {
      return false;
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
  void prepare_next_adjustment(LA::MPI::Vector &negative_increment,
                               LA::MPI::Vector &solution,
                               LA::MPI::SparseMatrix &system_matrix,
                               LA::MPI::Vector &system_rhs,
                               LA::MPI::Vector &neumann_rhs,
                               NewtonInformation<dim> &info,
                               Controller<dim> &ctl) override {
    if (ctl.params.line_search_damping >= 1) {
      throw SolverControl::NoConvergence(0, 0);
    }
    solution += negative_increment;
    negative_increment *= ctl.params.line_search_damping;
  };
};


template <int dim>
std::unique_ptr<NewtonVariation<dim>>
select_newton_variation(std::string method) {
  if (method == "none")
    return std::make_unique<NewtonVariation<dim>>();
  else if (method == "linesearch")
    return std::make_unique<LineSearch<dim>>();
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_NEWTON_VARIATIONS_H
