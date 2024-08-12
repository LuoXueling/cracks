//
// Created by xlluo on 24-8-4.
//

#ifndef CRACKS_ADAPTIVE_TIMESTEP_H
#define CRACKS_ADAPTIVE_TIMESTEP_H

#include "controller.h"
#include <fstream>
#include <iostream>

template <int dim> class AdaptiveTimeStep {
public:
  AdaptiveTimeStep(Controller<dim> &ctl) : last_time(0), count_reduction(0){};

  virtual void initialize_timestep(Controller<dim> &ctl) {};

  virtual double current_timestep(Controller<dim> &ctl) {
    return ctl.current_timestep;
  }

  void execute_when_fail(Controller<dim> &ctl) {
    ctl.time -= ctl.current_timestep;
    check_time(ctl);
    double new_timestep = get_new_timestep_when_fail(ctl);
    failure_criteria(new_timestep, ctl);
    ctl.current_timestep = new_timestep;
    record(ctl);
    ctl.time += ctl.current_timestep;
  }

  virtual double get_new_timestep_when_fail(Controller<dim> &ctl) {
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
  virtual bool terminate(Controller<dim> &ctl) { return false; }

  std::vector<double> historical_timesteps;
  double last_time;
  int count_reduction;
};

template <int dim> class ConstantTimeStep : public AdaptiveTimeStep<dim> {
public:
  ConstantTimeStep(Controller<dim> &ctl) : AdaptiveTimeStep<dim>(ctl){};
  void failure_criteria(double new_timestep, Controller<dim> &ctl) override {
    throw std::runtime_error(
        "Staggered scheme does not converge, and ConstantTimeStep "
        "does not allow adaptive time stepping");
  }
};

template <int dim> class KristensenCLATimeStep : public ConstantTimeStep<dim> {
public:
  KristensenCLATimeStep(Controller<dim> &ctl) : ConstantTimeStep<dim>(ctl) {
    if (ctl.params.fatigue_accumulation != "KristensenCLA") {
      ctl.dcout << "KristensenCLATimeStep is expected to used with "
                   "KristensenCLAAccumulation, but it's not. Please make sure "
                   "that the accumulation rule is consistent with constant "
                   "amplitude accumulation."
                << std::endl;
    }
    AssertThrow(ctl.params.adaptive_timestep_parameters != "",
                ExcInternalError(
                    "Parameters of KristensenCLATimeStep is not assigned."));
    std::istringstream iss(ctl.params.adaptive_timestep_parameters);
    iss >> R >> f;
    T = 1 / f;
    AssertThrow(ctl.params.timestep * (ctl.params.switch_timestep + 1) ==
                    0.25 * T,
                ExcInternalError("The initial timestep has to be switched when "
                                 "reaching a quarter of a cycle."));
    n_cycles_per_vtk = static_cast<int>(ctl.params.save_vtk_per_step /
                                        (T / ctl.params.timestep_size_2));
  }
  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "KristensenCLATimeStep using parameter: R=" << R << ", f=" << f
              << "Hz" << std::endl;

    ctl.params.timestep_size_2 = T;
    ctl.params.save_vtk_per_step = n_cycles_per_vtk;
    ctl.dcout << "KristensenCLATimeStep setting timestep to a cycle (" << T
              << "s), setting save_vtk_per_step to " << n_cycles_per_vtk
              << std::endl;
  }
  double R, f, T;
  int n_cycles_per_vtk;
};

template <int dim>
std::unique_ptr<AdaptiveTimeStep<dim>>
select_adaptive_timestep(std::string method, Controller<dim> &ctl) {
  if (method == "constant")
    return std::make_unique<ConstantTimeStep<dim>>(ctl);
  else if (method == "exponential")
    return std::make_unique<AdaptiveTimeStep<dim>>(ctl);
  else if (method == "KristensenCLA")
    return std::make_unique<KristensenCLATimeStep<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_ADAPTIVE_TIMESTEP_H
