//
// Created by xlluo on 24-8-4.
//

#ifndef CRACKS_ADAPTIVE_TIMESTEP_H
#define CRACKS_ADAPTIVE_TIMESTEP_H

#include "controller.h"

template <int dim> class AdaptiveTimeStep {
public:
  AdaptiveTimeStep() : last_time(0), count_reduction(0){};

  void execute(Controller<dim> &ctl) {
    ctl.time -= ctl.current_timestep;
    check_time(ctl);
    double new_timestep = get_new_timestep(ctl);
    failure_criteria(new_timestep, ctl);
    ctl.current_timestep = new_timestep;
    record(ctl);
    ctl.time += ctl.current_timestep;
  }

  virtual double get_new_timestep(Controller<dim> &ctl) {
    return ctl.current_timestep * 0.1;
  }

  void check_time(Controller<dim> &ctl) {
    if (ctl.time != last_time) {
      last_time = ctl.time;
      count_reduction = 0;
      historical_timesteps.push_back(ctl.current_timestep);
    }
  }

  void record(Controller<dim> &ctl) { count_reduction++; }

  virtual void failure_criteria(double new_timestep, Controller<dim> &ctl) {
    if (new_timestep < ctl.params.timestep * 1e-8) {
      AssertThrow(false, ExcInternalError("Step size too small"))
    }
  }

  std::vector<double> historical_timesteps;
  double last_time;
  int count_reduction;
};

template <int dim> class ConstantTimeStep : public AdaptiveTimeStep<dim> {
public:
  void failure_criteria(double new_timestep, Controller<dim> &ctl) override {
    AssertThrow(false,
                ExcInternalError(
                    "Staggered scheme does not converge, and ConstantTimeStep "
                    "does not allow adaptive time stepping"));
  }
};

template <int dim>
std::unique_ptr<AdaptiveTimeStep<dim>>
select_adaptive_timestep(std::string method) {
  if (method == "constant")
    return std::make_unique<ConstantTimeStep<dim>>();
  else if (method == "exponential")
    return std::make_unique<AdaptiveTimeStep<dim>>();
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_ADAPTIVE_TIMESTEP_H
