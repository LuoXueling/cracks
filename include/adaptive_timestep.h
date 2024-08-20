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

  virtual bool save_checkpoint(Controller<dim> &ctl){
    return false;
  }

  virtual std::string return_solution_or_checkpoint(Controller<dim> &ctl){
    return "solution";
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

template <int dim> class CojocaruCycleJump : public ConstantTimeStep<dim> {
public:
  CojocaruCycleJump(Controller<dim> &ctl)
      : ConstantTimeStep<dim>(ctl), subcycle(0), n_jump(0) {
    if (ctl.params.fatigue_accumulation != "CojocaruCLA") {
      ctl.dcout << "CojocaruCycleJump is expected to used with "
                   "CojocaruCLAAccumulation, but it's not. Please make sure "
                   "that the accumulation rule is consistent with constant "
                   "amplitude accumulation with cycle jumping support."
                << std::endl;
    }
    AssertThrow(
        ctl.params.adaptive_timestep_parameters != "",
        ExcInternalError("Parameters of CojocaruCycleJump is not assigned."));
    std::istringstream iss(ctl.params.adaptive_timestep_parameters);
    iss >> R >> f >> max_jumps;
    T = 1 / f;
    AssertThrow(ctl.params.timestep * (ctl.params.switch_timestep + 1) ==
                    0.25 * T,
                ExcInternalError("The initial timestep has to be switched when "
                                 "reaching a quarter of a cycle."));
    ctl.set_info("Maximum jump", max_jumps);
    expected_cycles = ctl.params.max_no_timesteps;
  };

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "KristensenCLATimeStep using parameter: R=" << R << ", f=" << f
              << "Hz" << std::endl;

    ctl.params.timestep_size_2 = T;
    ctl.params.save_vtk_per_step = 5;
    ctl.dcout << "KristensenCLATimeStep setting timestep to a cycle (" << T
              << "s), setting save_vtk_per_step to 5, which is a period of "
                 "cycle jump."
              << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    subcycle++;
    ctl.set_info("Subcycle", static_cast<double>(subcycle));
    if (subcycle == 1) {
      n_jump = 0;
      ctl.set_info("N jump", n_jump);
      ctl.set_info("N jump local", max_jumps);
      return ctl.current_timestep;
    } else if (subcycle == 5) {
      subcycle = 0;

      double n_jump_temp = ctl.get_info("N jump local", max_jumps);
      n_jump_temp = Utilities::MPI::min(n_jump_temp, ctl.mpi_com);

      n_jump = std::min(max_jumps,
                        static_cast<unsigned int>(std::floor(n_jump_temp)));
      n_jump = std::max(n_jump, static_cast<unsigned int>(1));
      ctl.set_info("N jump", n_jump);
      ctl.dcout << "Doing cycle jumping in this timestep: jumping " << n_jump
                << " cycles" << std::endl;
      ctl.output_timestep_number +=
          n_jump - 1; // the remaining 1 will be added automatically as normal
      return ctl.current_timestep * n_jump;
    } else {
      return ctl.current_timestep;
    }
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time == expected_cycles) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else {
      return false;
    }
  }

  double R, f, T;
  unsigned int max_jumps;
  unsigned int subcycle;
  unsigned int n_jump;
  unsigned int expected_cycles;
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
  else if (method == "CojocaruCycleJump")
    return std::make_unique<CojocaruCycleJump<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_ADAPTIVE_TIMESTEP_H
