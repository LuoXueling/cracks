//
// Created by xlluo on 24-8-4.
//

#ifndef CRACKS_ADAPTIVE_TIMESTEP_H
#define CRACKS_ADAPTIVE_TIMESTEP_H

#include "controller.h"
#include "fatigue_degradation.h"
#include "global_estimator.h"
#include "utils.h"
#include <fstream>
#include <iostream>

template <int dim> class AdaptiveTimeStep {
public:
  AdaptiveTimeStep(Controller<dim> &ctl)
      : last_time(0), count_reduction(0), save_results(false),
        new_timestep(ctl.current_timestep){};

  virtual void initialize_timestep(Controller<dim> &ctl) {};

  double get_timestep(Controller<dim> &ctl) {
    last_time = ctl.time;
    new_timestep = current_timestep(ctl);
    return new_timestep;
  }

  virtual double current_timestep(Controller<dim> &ctl) {
    return ctl.current_timestep;
  }

  void execute_when_fail(Controller<dim> &ctl) {
    ctl.time = last_time;
    check_time(ctl);
    new_timestep = get_new_timestep_when_fail(ctl);
    failure_criteria(ctl);
    record(ctl);
    ctl.time += new_timestep;
  }

  virtual bool fail(double newton_reduction, Controller<dim> &ctl) {
    return newton_reduction > ctl.params.upper_newton_rho;
  }

  virtual void after_step(Controller<dim> &ctl) {}

  virtual double get_new_timestep_when_fail(Controller<dim> &ctl) {
    return new_timestep * 0.1;
  }

  void check_time(Controller<dim> &ctl) {
    if (ctl.time != last_time) {
      last_time = ctl.time;
      count_reduction = 0;
      historical_timesteps.push_back(new_timestep);
    }
  }

  virtual bool save_checkpoint(Controller<dim> &ctl){
    return false;
  }

  virtual std::string return_solution_or_checkpoint(Controller<dim> &ctl){
    return "solution";
  }

  void record(Controller<dim> &ctl) { count_reduction++; }

  virtual void failure_criteria(Controller<dim> &ctl) {
    if (new_timestep < ctl.params.timestep * 1e-8) {
      AssertThrow(false, ExcInternalError("Step size too small"))
    }
  }
  virtual bool terminate(Controller<dim> &ctl) { return false; }

  std::vector<double> historical_timesteps;
  double last_time;
  int count_reduction;
  bool save_results;
  double new_timestep;
};

template <int dim> class ConstantTimeStep : public AdaptiveTimeStep<dim> {
public:
  ConstantTimeStep(Controller<dim> &ctl) : AdaptiveTimeStep<dim>(ctl){};
  void failure_criteria(Controller<dim> &ctl) override {
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
    ctl.dcout << "CojocaruCycleJump using parameter: R=" << R << ", f=" << f
              << "Hz" << std::endl;

    ctl.params.timestep_size_2 = T;
    ctl.params.save_vtk_per_step = 5;
    ctl.dcout << "CojocaruCycleJump setting timestep to a cycle (" << T
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

/*
 * arXiv:2404.07003v1
 */
template <int dim> class JonasCycleJump : public ConstantTimeStep<dim> {
public:
  unsigned int Ns;
  unsigned int stage;
  double Lambda0, Lambda1, Lambda2;
  double lambda2, lambda3;

  double alpha_t;
  double R, f, T;
  double subcycle;
  unsigned int n_jump;
  unsigned int last_jump;
  unsigned int n_jump_initial;
  unsigned int expected_cycles;
  unsigned int n_cracks;
  double monitor, trial_monitor;
  double Delta, trial_Delta;
  std::vector<double> monitor1, monitor2, monitor3;
  std::vector<double> resolved_cycles;

  unsigned int initial_save_period;
  bool doing_cycle_jump;
  bool consecutive_n_jump_0;

  bool trial_cycle;
  double trial_cycle_start, trial_cycle_end;
  double trial_cycle_start_output_time, trial_cycle_start_timestep_number;

  bool refine_state;

  JonasCycleJump(Controller<dim> &ctl)
      : ConstantTimeStep<dim>(ctl), Ns(4), stage(1), Lambda0(0), Lambda1(0),
        Lambda2(0), subcycle(1), n_jump(0), lambda2(1.0), lambda3(1.0),
        Delta(0), last_jump(0), monitor(0), trial_monitor(0),
        initial_save_period(0), doing_cycle_jump(false),
        consecutive_n_jump_0(false), n_jump_initial(0), trial_cycle(false),
        trial_cycle_start(-1), trial_cycle_end(-1),
        trial_cycle_start_output_time(-1),
        trial_cycle_start_timestep_number(-1), trial_Delta(0),
        // The zero step is skipped, so the first element of monitor1 should
        // be initialized
        monitor1(1, 0), resolved_cycles(1, 0) {
    if (ctl.params.fatigue_accumulation != "Jonas") {
      ctl.dcout << "JonasCycleJump is expected to used with "
                   "JonasAccumulation, but it's not. Please make sure "
                   "that the accumulation rule supports cycle jump."
                << std::endl;
    }
    AssertThrow(
        ctl.params.adaptive_timestep_parameters != "",
        ExcInternalError("Parameters of JonasCycleJump is not assigned."));
    std::istringstream iss(ctl.params.adaptive_timestep_parameters);
    iss >> R >> f >> alpha_t >> n_cracks;
    T = 1 / f;
    expected_cycles = std::round(
        (ctl.params.timestep * (ctl.params.switch_timestep + 1) +
         ctl.params.timestep_size_2 *
             (ctl.params.max_no_timesteps - ctl.params.switch_timestep - 1)) /
        T);
    initial_save_period = ctl.params.save_vtk_per_step;
    ctl.set_info("Subcycle", 1.0 + ctl.params.timestep /
                                       T); // PointHistory won't record at the
                                           // first step (it should be zero).
    ctl.set_info("Stage", stage);
    AssertThrow(
        std::fmod(T, ctl.params.timestep) < 1e-8 &&
            std::fmod(T, ctl.params.timestep_size_2) < 1e-8,
        ExcInternalError("The period has to be divisible by the time step"));
  };

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "JonasCycleJump using parameter: R=" << R << ", f=" << f
              << "Hz" << std::endl;
    ctl.params.save_vtk_per_step = 1e10;
    ctl.dcout << "JonasCycleJump disables periodical outputs. Instead, it will "
                 "save after each cycle jump, or after every "
              << initial_save_period << " steps if the system changes rapidly."
              << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    if (!trial_cycle) {
      if (stage == 1 && monitor > alpha_t) {
        stage = 2;
        ctl.dcout << "Entering stage 2" << std::endl;
        subcycle = std::fmod(ctl.time, T) /
                   T; // PointHistory will record from y1 again.
      } else if (stage == 2 && monitor > 0.99) {
        stage = 3;
        ctl.dcout << "Entering stage 3" << std::endl;
        subcycle = std::fmod(ctl.time, T) / T;
      }
    } else {
      n_jump = 0;
      ctl.set_info("N jump", n_jump);
      doing_cycle_jump = false;
      ctl.dcout << "*********************************" << std::endl;
      ctl.dcout << "********** Trial cycle **********" << std::endl;
      ctl.dcout << "*********************************" << std::endl;
    }
    double time_step = ctl.current_timestep;
    if (std::abs(subcycle - 1) < 1e-8) {
      n_jump = 0;
      doing_cycle_jump = false;
      ctl.set_info("N jump", n_jump);
    } else if (subcycle > Ns - 1 && std::fmod(subcycle, Ns) < 1e-8 &&
               !trial_cycle) {
      std::vector<double> monitors;
      if (stage == 1)
        monitors = monitor1;
      else if (stage == 2)
        monitors = monitor2;
      else if (stage == 3)
        monitors = monitor3;

      if (monitors.size() < Ns) {
        ctl.dcout << "No enough resolved cycles. Wait for the next loop."
                  << std::endl;
        subcycle = 0.0; // PointHistory will record from y1 again.
        time_step = ctl.current_timestep;
      } else {
        PolynomialRegression<double> polyfit;
        std::vector<double> coeffs(3, 0);
        std::vector<double> x_last3Ns;
        std::vector<double> y_last3Ns;

        x_last3Ns = std::vector<double>(
            resolved_cycles.end() -
                std::min(3 * Ns, static_cast<unsigned int>(monitors.size())),
            resolved_cycles.end());
        y_last3Ns = std::vector<double>(
            monitors.end() -
                std::min(3 * Ns, static_cast<unsigned int>(monitors.size())),
            monitors.end());

        polyfit.fitIt(x_last3Ns, y_last3Ns, 2, coeffs);
        ctl.dcout << "Estimating the number of jumps" << std::endl;
        ctl.dcout << "Values to fit:" << std::endl;
        for (unsigned int i = 0; i < x_last3Ns.size(); ++i) {
          ctl.dcout << x_last3Ns[i] << " " << y_last3Ns[i] << std::endl;
        }
        ctl.dcout << "Fitted monitor: Lambda = (" << coeffs[0] << ") + ("
                  << coeffs[1] << ")N + (" << coeffs[2] << ") N^2" << std::endl;
        Lambda0 = coeffs[0] - monitor - Delta;
        Lambda1 = coeffs[1];
        Lambda2 = coeffs[2];
        ctl.dcout << "Quadratic equation to be solved: (" << Lambda0 << ") + ("
                  << Lambda1 << ")N + (" << Lambda2 << ") N^2 = 0" << std::endl;
        double d = std::pow(Lambda1, 2) - 4 * Lambda0 * Lambda2;
        if (d > 0) {
          ctl.dcout << "Using quadratic roots." << std::endl;
          n_jump = std::round((-Lambda1 + std::sqrt(d)) / (2 * Lambda2)) -
                   static_cast<int>(ctl.time / T);
          ctl.dcout << "Estimated number of jumps: " << n_jump << std::endl;
          if (n_jump < 0) {
            n_jump = std::floor(last_jump / 2);
            ctl.dcout
                << "Negative number of jumps. Using half of the last jump: "
                << n_jump << std::endl;
          }
        } else {
          ctl.dcout << "Using linear fit." << std::endl;
          std::vector<double> coeffs_lin(2, 0);
          polyfit.fitIt(x_last3Ns, y_last3Ns, 1, coeffs_lin);
          ctl.dcout << "Fitted monitor: Lambda = (" << coeffs_lin[0] << ") + ("
                    << coeffs_lin[1] << ")N" << std::endl;
          n_jump =
              std::round((monitor + Delta - coeffs_lin[0]) / coeffs_lin[1]) -
              static_cast<int>(ctl.time / T);
          ctl.dcout << "Estimated number of jumps: " << n_jump << std::endl;
          subcycle = 0.0; // PointHistory will record from y1 again.
        }

        AssertThrow(n_jump < 1e10, ExcInternalError("Infinite cycle jump."));
        if (n_jump > 1) {
          ctl.set_info("N jump", n_jump);
          ctl.dcout << "Doing cycle jumping in this timestep: jumping "
                    << n_jump << " cycles with 1 trial cycle" << std::endl;
          time_step = T * (n_jump-1);

          doing_cycle_jump = true;
          n_jump_initial = n_jump;
          trial_cycle = true;
          ctl.set_info("Trial cycle", 1.0);
          trial_cycle_start = ctl.time;
          trial_cycle_end = ctl.time + n_jump_initial * T;
          trial_cycle_start_output_time = ctl.output_timestep_number;
          trial_cycle_start_timestep_number = ctl.timestep_number;
          ctl.params.save_vtk_per_step = 1e10;
          refine_state = ctl.params.refine;
          if (refine_state){
            ctl.dcout << "Refinement is disabled temporarily" << std::endl;
          }
          ctl.params.refine = false; // We need the checkpoint work.

          ctl.dcout << "Delta: " << Delta << std::endl;
        } else {
          ctl.dcout << "The system is changing rapidly. No cycle jumping is "
                       "executed in this time step"
                    << std::endl;
          ctl.params.save_vtk_per_step = initial_save_period;
          n_jump = 0;
          ctl.set_info("N jump", n_jump);
          doing_cycle_jump = false;
          subcycle = 0.0; // PointHistory will record from y1 again.
        }
      }
    }
    if (!doing_cycle_jump) {
      subcycle += ctl.current_timestep / T;
    } else {
      subcycle = 0.0;
    }
    ctl.set_info("Subcycle", subcycle);
    ctl.dcout << "Current subcycle: " << subcycle << std::endl;
    return time_step;
  }

  bool fail(double newton_reduction, Controller<dim> &ctl) override {
    if (doing_cycle_jump || trial_cycle) {
      // Doing cycle jump
      if (newton_reduction > ctl.params.upper_newton_rho) {
        n_jump =
            std::max(static_cast<int>(std::round(n_jump_initial / 2)), 0);
        ctl.set_info("N jump", n_jump);
        ctl.dcout << "The system fails to establish equilibrium. Reducing the "
                     "number of jumps to "
                  << n_jump << " cycles with 1 trial cycle" << std::endl;
        return true;
      } else {
        if (stage == 1) {
          trial_monitor = get_stage1_monitor(ctl);
        } else if (stage == 2) {
          trial_monitor = get_stage2_monitor(ctl);
        } else if (stage == 3) {
          trial_monitor = get_stage3_monitor(ctl);
        }
        trial_Delta = trial_monitor - monitor;
        AssertThrow(
            trial_Delta >= 0,
            ExcInternalError("The increment of monitored value is negative."));
        if (trial_Delta > 1.5 * Delta) {
          n_jump = std::max(
              static_cast<int>(
                  std::round(Delta / trial_Delta * n_jump_initial)),
              0);
          ctl.set_info("N jump", n_jump);
          ctl.dcout << "The real increment of the monitored value ("
                    << trial_Delta
                    << ") is much "
                       "higher than expected ("
                    << Delta
                    << "). Reducing the "
                       "number of jumps to "
                    << n_jump << " cycles with 1 trial cycle" << std::endl;
          return true;
        } else {
          ctl.dcout << "Trial Delta: " << trial_Delta
                    << " Expected Delta: " << Delta << std::endl;
          return false;
        }
      }
    } else {
      return newton_reduction > ctl.params.upper_newton_rho;
    }
  }

  double get_stage1_monitor(Controller<dim> &ctl) {
    return GlobalEstimator::max<dim>("Fatigue history", 0.0, ctl);
  }

  double get_stage2_monitor(Controller<dim> &ctl) {
    return GlobalEstimator::max<dim>("Phase field", 0.0, ctl);
  }

  double get_stage3_monitor(Controller<dim> &ctl) {
    double phase_integral =
        GlobalEstimator::sum<dim>("Phase field JxW", 0.0, ctl);
    double res;
    if (ctl.params.phasefield_model == "AT1") {
      res = (phase_integral -
             n_cracks * numbers::PI * std::pow(ctl.params.l_phi, 2) / 3.0) /
            (4.0 / 3.0 * ctl.params.l_phi);
    } else if (ctl.params.phasefield_model == "AT2") {
      res = (phase_integral -
             n_cracks * numbers::PI * std::pow(ctl.params.l_phi, 2)) /
            (2.0 * ctl.params.l_phi);
    } else {
      AssertThrow(false, ExcNotImplemented());
    }
    return res;
  }

  double get_new_timestep_when_fail(Controller<dim> &ctl) override {
    if (trial_cycle) {
      ctl.time = trial_cycle_start;
      ctl.timestep_number = trial_cycle_start_timestep_number;
      ctl.output_timestep_number = trial_cycle_start_output_time;

      doing_cycle_jump = true;
      n_jump_initial = n_jump;
      trial_cycle = true;
      ctl.set_info("Trial cycle", 1.0);
      trial_cycle_start = ctl.time;
      trial_cycle_end = ctl.time + n_jump_initial * T;
      trial_cycle_start_output_time = ctl.output_timestep_number;
      trial_cycle_start_timestep_number = ctl.timestep_number;
      subcycle = 0.0;

      ctl.set_info("Subcycle", subcycle);

      ctl.dcout << "Trial cycle failed. Returning to time ("
                << trial_cycle_start_timestep_number - 1 << ") "
                << trial_cycle_start << " and jump again with " << n_jump
                << " cycles + 1 trial cycle" << std::endl;
    }
    return T * ((n_jump > 1) ? (n_jump-1) : 1);
  }

  void failure_criteria(Controller<dim> &ctl) override {
    if (!trial_cycle) {
      throw std::runtime_error("Staggered scheme does not converge when no "
                               "cycle jump is performed.");
    } else if (doing_cycle_jump && consecutive_n_jump_0) {
      throw std::runtime_error("Staggered scheme does not converge when the "
                               "number of jump is reduced to zero.");
    } else if (doing_cycle_jump && n_jump == 0) {
      consecutive_n_jump_0 = true;
    } else {
      consecutive_n_jump_0 = false;
    }
  }

  void after_step(Controller<dim> &ctl) override {
    if (std::abs(ctl.time - trial_cycle_end) < 1e-8) {
      last_jump = n_jump_initial;
      ctl.output_timestep_number += n_jump_initial - 1;
      if (last_jump > 0) {
        ctl.dcout
            << "Cycle jump is done successfully. The number of jumped cycle: "
            << last_jump << "." << std::endl;
      }
      n_jump = 0;
      doing_cycle_jump = false;
      trial_cycle = false;
      ctl.set_info("Trial cycle", 0.0);
      ctl.set_info("N jump", n_jump);
      this->save_results = true;
      ctl.params.save_vtk_per_step = 1e10;
      ctl.params.refine = refine_state;
    } else {
      ctl.output_timestep_number +=
          (std::fmod(subcycle, 1) < 1e-8 && subcycle > 0) ? 0 : (-1);
    }

    if (!trial_cycle){
      if (stage == 1) {
        monitor = get_stage1_monitor(ctl);
      } else if (stage == 2) {
        monitor = get_stage2_monitor(ctl);
      } else if (stage == 3) {
        monitor = get_stage3_monitor(ctl);
      }
      ctl.dcout << "The monitored value (at stage " << stage << "): " << monitor
                << "." << std::endl;
    }
    if (std::fmod(subcycle, 1) < 1e-8 && !trial_cycle) {
      resolved_cycles.emplace_back(ctl.time / T);
      if (stage == 1) {
        // The jump directly skip the first stage after the first Ns cycles
        // are resolved.
        monitor1.emplace_back(monitor);
        Delta = std::max(alpha_t - monitor,
                         0.0); // may be dealing with a negative increment,
                               // but we have to finish Ns cycles first.
      } else if (stage == 2) {
        monitor2.emplace_back(monitor);
        Delta = lambda2 * 0.02;
      } else if (stage == 3) {
        monitor3.emplace_back(monitor);
        Delta = lambda3 * ctl.params.l_phi / 2;
      }
    }

    ctl.dcout << "The number of resolved cycles (including trial cycles): "
              << resolved_cycles.size() << std::endl;
  }

  bool save_checkpoint(Controller<dim> &ctl) override{
    if (std::abs(subcycle - Ns) < 1e-8){
      return true;
    } else {
      return false;
    }
  }

  std::string return_solution_or_checkpoint(Controller<dim> &ctl) override{
    if (trial_cycle){
      return "checkpoint";
    } else {
      return "solution";
    }
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time / T >= expected_cycles) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else {
      return false;
    }
  }
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
  else if (method == "JonasCycleJump")
    return std::make_unique<JonasCycleJump<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_ADAPTIVE_TIMESTEP_H
