//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_ABSTRACT_FIELD_H
#define CRACKS_ABSTRACT_FIELD_H

#include "controller.h"
#include "dealii_includes.h"

template <int dim> class AbstractField {
public:
  explicit AbstractField(Controller<dim> &ctl){};

  virtual void setup_system(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void setup_boundary_condition(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void assemble_system(bool residual_only, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual double newton_iteration(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual unsigned int solve(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void output_results(DataOut<dim> &data_out, Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void return_old_solution(Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void distribute_hanging_node_constraints(LA::MPI::Vector &vector,
                                                   Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
  virtual void distribute_all_constraints(LA::MPI::Vector &vector,
                                          Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented())
  };
};

#endif // CRACKS_ABSTRACT_FIELD_H
