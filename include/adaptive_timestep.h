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

  virtual bool save_checkpoint(Controller<dim> &ctl) { return false; }

  virtual std::string return_solution_or_checkpoint(Controller<dim> &ctl) {
    return "solution";
  }

  void record(Controller<dim> &ctl) { count_reduction++; }

  virtual void failure_criteria(Controller<dim> &ctl) {
    if (new_timestep < ctl.params.timestep * 1e-8) {
      AssertThrow(false, ExcInternalError("Step size too small"))
    }
  }
  virtual bool terminate(Controller<dim> &ctl) {
    double crack_length = GlobalEstimator::sum<dim>("Diffusion JxW", 0.0, ctl);
    ctl.dcout << "Crack length estimated by diffusion: " << crack_length
              << std::endl;
    if (crack_length > ctl.params.max_crack_length) {
      ctl.dcout
          << "Terminating as the crack length exceeds the expected value ("
          << ctl.params.max_crack_length << ")" << std::endl;
      return true;
    } else {
      return false;
    }
  }

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
    if (!iss.eof()) {
      iss >> n_jump;
    } else {
      n_jump = 1;
    }
    ctl.set_info("N jump", 1);
    if (ctl.params.timestep * ctl.params.switch_timestep != 0.25 * T) {
      ctl.dcout << "The initial timestep has to be switched when "
                   "reaching a quarter of a cycle. Otherwise a phase has to be "
                   "assigned in the cyclic boundary condition."
                << std::endl;
    }
    n_cycles_per_vtk = static_cast<int>(std::round(
        ctl.params.save_vtk_per_step / (T / ctl.params.timestep_size_2)));
    expected_cycles =
        std::round((ctl.params.timestep * ctl.params.switch_timestep +
                    ctl.params.timestep_size_2 * (ctl.params.max_no_timesteps -
                                                  ctl.params.switch_timestep)) /
                   T);
  }

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "KristensenCLATimeStep using parameter: R=" << R << ", f=" << f
              << "Hz, n_jump=" << n_jump << std::endl;

    ctl.params.timestep_size_2 = T;
    ctl.params.save_vtk_per_step = static_cast<int>(
        std::ceil(static_cast<double>(n_cycles_per_vtk) / n_jump));
    ctl.dcout << "KristensenCLATimeStep setting timestep to a cycle (" << T
              << "s), setting save_vtk_per_step to "
              << ctl.params.save_vtk_per_step << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    if (ctl.current_timestep != ctl.params.timestep_size_2) {
      ctl.set_info("N jump", 1);
      return ctl.current_timestep;
    } else {
      ctl.set_info("N jump", n_jump);
      return T * n_jump;
    }
  }

  void after_step(Controller<dim> &ctl) override {
    if (ctl.current_timestep == ctl.params.timestep_size_2) {
      ctl.output_timestep_number += n_jump - 1;
    } else {
      ctl.output_timestep_number += (std::fmod(ctl.time, T) < 1e-8) ? 0 : (-1);
    }
    ctl.dcout << "Maximum fatigue variable: "
              << GlobalEstimator::max<dim>("Fatigue history", 0.0, ctl)
              << std::endl;
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time / T >= expected_cycles) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else {
      return AdaptiveTimeStep<dim>::terminate(ctl);
    }
  }

  double R, f, T;
  int n_cycles_per_vtk;
  unsigned int n_jump;
  unsigned int expected_cycles;
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
    AssertThrow(ctl.params.timestep * ctl.params.switch_timestep == 0.25 * T,
                ExcInternalError("The initial timestep has to be switched when "
                                 "reaching a quarter of a cycle."));
    ctl.set_info("N jump", n_jump);
    ctl.set_info("Maximum jump", max_jumps);
    expected_cycles =
        std::round((ctl.params.timestep * (ctl.params.switch_timestep) +
                    ctl.params.timestep_size_2 * (ctl.params.max_no_timesteps -
                                                  ctl.params.switch_timestep)) /
                   T);
  };

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "CojocaruCycleJump using parameter: R=" << R << ", f=" << f
              << "Hz" << std::endl;
    ctl.params.save_vtk_per_step = 1e10;
    ctl.dcout << "Cojocaru disables periodical outputs. Instead, it will "
                 "save after each cycle jump."
              << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    if (ctl.current_timestep != ctl.params.timestep_size_2)
      return ctl.current_timestep;
    double timestep = 0.0;
    if (subcycle < 3) {
      n_jump = 0;
      ctl.set_info("N jump", n_jump);
      ctl.set_info("N jump local", max_jumps);
      timestep = ctl.current_timestep;
    } else if (std::abs(subcycle - 3) < 1e-8) {
      double n_jump_temp = ctl.get_info("N jump local", max_jumps);
      n_jump_temp = Utilities::MPI::min(n_jump_temp, ctl.mpi_com);

      n_jump = std::min(max_jumps,
                        static_cast<unsigned int>(std::floor(n_jump_temp)));
      n_jump = std::max(n_jump, static_cast<unsigned int>(1));
      ctl.set_info("N jump", n_jump);
      ctl.dcout << "Doing cycle jumping in this timestep: jumping " << n_jump
                << " cycles" << std::endl;
      timestep = T * n_jump;
    } else {
      timestep = ctl.current_timestep;
    }
    if (std::abs(subcycle - 3) < 1e-8) {
      subcycle = 1;
    } else {
      subcycle += ctl.current_timestep / T;
    }
    ctl.set_info("Subcycle", subcycle);
    ctl.dcout << "Current subcycle: " << subcycle << std::endl;
    return timestep;
  }

  void after_step(Controller<dim> &ctl) {
    if (n_jump > 0) {
      subcycle = 1;
      ctl.output_timestep_number += n_jump - 1;
      this->save_results = true;
    } else {
      ctl.output_timestep_number += (std::fmod(subcycle, 1) < 1e-8) ? 0 : (-1);
    }
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time / T >= expected_cycles) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else {
      return AdaptiveTimeStep<dim>::terminate(ctl);
    }
  }

  double R, f, T;
  unsigned int max_jumps;
  double subcycle;
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
  double corrected_estimation, f, T;
  double subcycle;
  unsigned int n_jump;
  unsigned int last_jump;
  unsigned int n_jump_initial;
  unsigned int expected_cycles;
  unsigned int n_tips;
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
  bool throw_multipass_state;

  double D_tip, D_ext;
  double initial_length;
  double c_tip, c_ext;

  JonasCycleJump(Controller<dim> &ctl)
      : ConstantTimeStep<dim>(ctl), Ns(4), stage(1), Lambda0(0), Lambda1(0),
        Lambda2(0), subcycle(0), n_jump(0), lambda2(1.0), lambda3(1.0),
        Delta(0), last_jump(0), monitor(0), trial_monitor(0),
        initial_save_period(0), doing_cycle_jump(false),
        consecutive_n_jump_0(false), n_jump_initial(0), trial_cycle(false),
        trial_cycle_start(-1), trial_cycle_end(-1),
        trial_cycle_start_output_time(-1),
        trial_cycle_start_timestep_number(-1), trial_Delta(0),
        initial_length(0.0) {
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
    iss >> corrected_estimation >> f >> alpha_t >> n_tips >> lambda2 >> lambda3;
    T = 1 / f;

    if (ctl.params.phasefield_model == "AT1") {
      D_tip = n_tips * numbers::PI * std::pow(ctl.params.l_phi, 2) / 3.0;
      D_ext = 4.0 / 3.0 * ctl.params.l_phi;
      if (std::abs(corrected_estimation) > 1e-8) {
        c_tip = 7.15261495 * std::exp(-0.70789363 * corrected_estimation) +
                1.4236912;
        c_ext = 1.2628246 * std::exp(-0.55521829 * corrected_estimation) +
                1.16223472;
        D_tip *= c_tip;
        D_ext *= c_ext;
      }
    } else if (ctl.params.phasefield_model == "AT2") {
      D_tip = n_tips * numbers::PI * std::pow(ctl.params.l_phi, 2);
      D_ext = 2.0 * ctl.params.l_phi;
      if (std::abs(corrected_estimation) > 1e-8) {
        c_tip = 7.59315987 * std::exp(-0.64105545 * corrected_estimation) +
                1.80994109;
        c_ext = 1.23659923 * std::exp(-0.53470298 * corrected_estimation) +
                1.1750714;
        D_tip *= c_tip;
        D_ext *= c_ext;
      }
    } else {
      AssertThrow(false, ExcNotImplemented());
    }

    expected_cycles =
        std::round((ctl.params.timestep * (ctl.params.switch_timestep) +
                    ctl.params.timestep_size_2 * (ctl.params.max_no_timesteps -
                                                  ctl.params.switch_timestep)) /
                   T);
    initial_save_period = ctl.params.save_vtk_per_step;
    throw_multipass_state = ctl.params.throw_if_multipass_increase;
    ctl.set_info("N jump", n_jump);
    ctl.set_info("Subcycle", ctl.params.timestep / T);
    ctl.set_info("Stage", stage);
    AssertThrow(
        std::fmod(T, ctl.params.timestep) < 1e-8 &&
            std::fmod(T, ctl.params.timestep_size_2) < 1e-8,
        ExcInternalError("The period has to be divisible by the time step"));
  };

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "JonasCycleJump using parameter: f=" << f
              << "Hz, alpha_t=" << alpha_t << ", n_tips=" << n_tips
              << std::endl;
    ctl.params.save_vtk_per_step = 1e10;
    ctl.dcout << "JonasCycleJump disables periodical outputs. Instead, it will "
                 "save after each cycle jump, or after every "
              << initial_save_period << " steps if the system changes rapidly."
              << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    if (ctl.current_timestep != ctl.params.timestep_size_2)
      return ctl.current_timestep;
    if (!trial_cycle) {
      if (stage == 1 && monitor > alpha_t) {
        stage = 2;
        ctl.dcout << "Entering stage 2" << std::endl;
        subcycle = std::fmod(ctl.time, T) / T -
                   ((ctl.params.timestep != ctl.params.timestep_size_2)
                        ? ctl.params.timestep * ctl.params.switch_timestep / T
                        : 0);
        // PointHistory will record from y1 again.
      } else if (stage == 2 && monitor > 0.99) {
        stage = 3;
        ctl.dcout << "Entering stage 3" << std::endl;
        subcycle = std::fmod(ctl.time, T) / T -
                   ((ctl.params.timestep != ctl.params.timestep_size_2)
                        ? ctl.params.timestep * ctl.params.switch_timestep / T
                        : 0);
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
        if (n_jump > 1e8) {
          ctl.dcout << "Infinite cycle jump. Set to zero" << std::endl;
          n_jump = 0;
        }
        if (n_jump > 1) {
          ctl.set_info("N jump", n_jump);
          ctl.dcout << "Doing cycle jumping in this timestep: jumping "
                    << n_jump << " cycles with 1 trial cycle" << std::endl;
          time_step = T * (n_jump - 1);

          doing_cycle_jump = true;
          n_jump_initial = n_jump;
          trial_cycle = true;
          ctl.set_info("Trial cycle", 1.0);
          trial_cycle_start = ctl.time;
          trial_cycle_end = ctl.time + n_jump_initial * T;
          trial_cycle_start_output_time = ctl.output_timestep_number;
          trial_cycle_start_timestep_number = ctl.timestep_number;
          ctl.params.save_vtk_per_step = 1e10;
          ctl.params.throw_if_multipass_increase =
              false; // This is disabled for now
          refine_state = ctl.params.refine;
          if (refine_state) {
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
        n_jump = std::max(static_cast<int>(std::round(n_jump_initial / 2)), 1);
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
        if (std::abs(ctl.time - trial_cycle_end) < 1e-8) {
          if (trial_Delta < 0) {
            n_jump =
                std::max(static_cast<int>(std::round(n_jump_initial / 2)), 1);
            ctl.dcout << "The increment of monitored value is negative ("
                      << trial_Delta << ")" << std::endl;
            ctl.dcout << "Reducing the "
                         "number of jumps to "
                      << n_jump << " cycles with 1 trial cycle" << std::endl;
            return true;
          } else if (trial_Delta > 1.5 * Delta) {
            n_jump = std::max(static_cast<int>(std::round(Delta / trial_Delta *
                                                          n_jump_initial)),
                              1);
            ctl.set_info("N jump", n_jump);
            ctl.dcout << "The real increment of the monitored value ("
                      << trial_Delta << ") is much higher than expected ("
                      << Delta
                      << "). Adjusting the "
                         "number of jumps to "
                      << n_jump << " cycles with 1 trial cycle" << std::endl;
            return true;
          } else {
            ctl.dcout << "Trial Delta: " << trial_Delta
                      << " Expected Delta: " << Delta << std::endl;
            return false;
          }
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
    double res = (phase_integral - D_tip) / D_ext - initial_length;
    if (stage == 3 && std::abs(initial_length) < 1e-8) {
      initial_length = res;
      res = 0.0;
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
    return T * ((n_jump >= 1) ? (n_jump - 1) : 1);
  }

  void failure_criteria(Controller<dim> &ctl) override {
    if (!trial_cycle) {
      throw std::runtime_error("Staggered scheme does not converge when no "
                               "cycle jump is performed.");
    } else if (doing_cycle_jump && consecutive_n_jump_0) {
      throw std::runtime_error("Staggered scheme does not converge when the "
                               "number of jump is reduced to one.");
    } else if (doing_cycle_jump && n_jump == 1) {
      consecutive_n_jump_0 = true;
      ctl.params.throw_if_multipass_increase = false;
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
        this->save_results = true;
        ctl.params.save_vtk_per_step = 1e10;
      }
      n_jump = 0;
      doing_cycle_jump = false;
      trial_cycle = false;
      ctl.set_info("Trial cycle", 0.0);
      ctl.set_info("N jump", n_jump);
      ctl.params.refine = refine_state;
      ctl.params.throw_if_multipass_increase = throw_multipass_state;
      consecutive_n_jump_0 = false;
    } else {
      ctl.output_timestep_number +=
          (std::fmod(subcycle, 1) < 1e-8 && subcycle > 0) ? 0 : (-1);
    }

    if (!trial_cycle) {
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
    if (std::fmod(subcycle, 1) < 1e-8 &&
        ctl.current_timestep == ctl.params.timestep_size_2 && !trial_cycle) {
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

  bool save_checkpoint(Controller<dim> &ctl) override {
    if (std::abs(subcycle - Ns) < 1e-8) {
      return true;
    } else {
      return false;
    }
  }

  std::string return_solution_or_checkpoint(Controller<dim> &ctl) override {
    if (trial_cycle) {
      return "checkpoint";
    } else {
      return "solution";
    }
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time / T >= expected_cycles && !trial_cycle) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else if (stage == 3 && std::abs(ctl.time - trial_cycle_end) < 1e-8 &&
               trial_Delta / Delta < 1e-2) {
      ctl.dcout << "Terminating as the increase of crack length is too small."
                << std::endl;
      return true;
    } else {
      return AdaptiveTimeStep<dim>::terminate(ctl);
    }
  }
};

template <int dim> class YangCycleJump : public ConstantTimeStep<dim> {
public:
  double f, T;
  double E, epsilon, epsilon_max;
  double subcycle;
  double tol;
  double max_diff;
  unsigned int max_jump;
  unsigned int n_jump;
  unsigned int last_jump, last_last_jump;
  unsigned int expected_cycles;
  unsigned int n_resolved_cycles;

  unsigned int initial_save_period;

  double trial_start;
  double trial_start_output_time, trial_start_timestep_number;

  bool refine_state;

  YangCycleJump(Controller<dim> &ctl)
      : ConstantTimeStep<dim>(ctl), subcycle(0), n_jump(0), last_jump(1),
        last_last_jump(1), initial_save_period(0), n_resolved_cycles(0),
        trial_start(-1), trial_start_output_time(-1), max_diff(0),
        trial_start_timestep_number(-1) {
    if (ctl.params.fatigue_accumulation != "Yang") {
      ctl.dcout << "YangCycleJump is expected to used with "
                   "YangAccumulation, but it's not. Please make sure "
                   "that the accumulation rule supports cycle jump."
                << std::endl;
    }
    AssertThrow(
        ctl.params.adaptive_timestep_parameters != "",
        ExcInternalError("Parameters of YangCycleJump is not assigned."));
    std::istringstream iss(ctl.params.adaptive_timestep_parameters);
    iss >> f >> epsilon >> E >> epsilon_max >> max_jump;
    T = 1 / f;
    expected_cycles =
        std::round((ctl.params.timestep * (ctl.params.switch_timestep) +
                    ctl.params.timestep_size_2 * (ctl.params.max_no_timesteps -
                                                  ctl.params.switch_timestep)) /
                   T);
    initial_save_period = ctl.params.save_vtk_per_step;
    ctl.set_info("N jump", n_jump);
    ctl.set_info("Subcycle", subcycle); // PointHistory won't record at the
                                        // first step (it should be zero).
    AssertThrow(
        std::fmod(T, ctl.params.timestep) < 1e-8 &&
            std::fmod(T, ctl.params.timestep_size_2) < 1e-8,
        ExcInternalError("The period has to be divisible by the time step"));
    tol = epsilon * 1e5 * E * epsilon_max * epsilon_max / 2;
    ctl.set_info("Last jump", last_jump);
  };

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "YangCycleJump using parameter: f=" << f
              << " Hz, epsilon=" << epsilon << ", E=" << E
              << " MPa, varepsilon_max=" << epsilon_max << std::endl;
    ctl.params.save_vtk_per_step = 1e10;
    ctl.dcout << "YangCycleJump disables periodical outputs. Instead, it will "
                 "save after each cycle jump, or after every "
              << initial_save_period << " steps if the system changes rapidly."
              << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    if (ctl.current_timestep != ctl.params.timestep_size_2)
      return ctl.current_timestep;
    double time_step = ctl.current_timestep;
    if (std::abs(subcycle - 1) < 1e-8) {
      subcycle = 0.0;
    }
    if (std::abs(subcycle) < 1e-8 && n_resolved_cycles >= 2 && n_jump == 0) {
      n_jump = new_jump(last_jump, ctl);
      if (n_jump > 1) {
        ctl.set_info("N jump", n_jump);
        ctl.dcout << "Doing cycle jumping in this timestep: jumping " << n_jump
                  << " cycles (including the preceding cycle)" << std::endl;
        time_step = T * (n_jump - 1);

        trial_start = ctl.time;
        trial_start_output_time = ctl.output_timestep_number;
        trial_start_timestep_number = ctl.timestep_number;
        ctl.params.save_vtk_per_step = 1e10;
        refine_state = ctl.params.refine;
        if (refine_state) {
          ctl.dcout << "Refinement is disabled temporarily" << std::endl;
        }
        ctl.params.refine = false; // We need the checkpoint work.
      } else {
        ctl.dcout << "The system is changing rapidly. No cycle jumping is "
                     "executed in this time step"
                  << std::endl;
        ctl.params.save_vtk_per_step = initial_save_period;
        n_jump = 0;
        ctl.set_info("N jump", n_jump);
        last_last_jump = last_jump;
        last_jump = 1;
        ctl.set_info("Last jump", last_jump);
        subcycle = 0.0;
      }
    } else {
      n_jump = 0;
      ctl.set_info("N jump", n_jump);
    }
    if (n_jump == 0) {
      subcycle += ctl.current_timestep / T;
    } else {
      subcycle = 0.0;
    }
    ctl.set_info("Subcycle", subcycle);
    ctl.dcout << "Current subcycle: " << subcycle << std::endl;
    return time_step;
  }

  bool pass(unsigned int m, Controller<dim> &ctl) {
    return (m * max_diff / 2 <= tol || m == 1);
  }

  unsigned int new_jump(unsigned int m, Controller<dim> &ctl) {
    if (n_resolved_cycles < 2) {
      return 1;
    } else {
      return std::min<unsigned int>(
          max_jump, std::max(1, static_cast<int>(std::floor(
                                    std::sqrt(tol * 2 * m / max_diff)))));
    }
  }

  bool fail(double newton_reduction, Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8) {
      max_diff = GlobalEstimator::absmax<dim>("Fast increment diff", 0.0, ctl);
      if (n_resolved_cycles >= 2) {
        return !pass(last_jump, ctl);
      } else {
        return newton_reduction > ctl.params.upper_newton_rho;
      }
    } else {
      return newton_reduction > ctl.params.upper_newton_rho;
    }
  }

  double get_new_timestep_when_fail(Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 2) {
      ctl.time = trial_start;
      ctl.timestep_number = trial_start_timestep_number;
      ctl.output_timestep_number = trial_start_output_time;

      n_jump = new_jump(last_jump, ctl);
      ctl.set_info("N jump", n_jump);
      last_jump = last_last_jump;
      ctl.set_info("Last jump", last_jump);
      subcycle = 0;
      ctl.set_info("Subcycle", subcycle);
      trial_start = ctl.time;
      trial_start_output_time = ctl.output_timestep_number;
      trial_start_timestep_number = ctl.timestep_number;

      ctl.dcout << "Trial jump failed. Returning to time ("
                << trial_start_timestep_number - 1 << ") " << trial_start
                << " and jump again with " << n_jump
                << " cycles (including the preceding cycle)" << std::endl;
    }
    return T * ((n_jump > 1) ? (n_jump - 1) : 1);
  }

  void failure_criteria(Controller<dim> &ctl) override {
    if (!((std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 2) ||
          (std::abs(subcycle) < 1e-8 && n_jump > 0))) {
      throw std::runtime_error("Staggered scheme does not converge when "
                               "cycle jump is performed.");
    }
  }

  void after_step(Controller<dim> &ctl) override {
    if (n_jump > 0) {
      last_last_jump = last_jump;
      last_jump = n_jump;
      ctl.set_info("Last jump", last_jump);
    }
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 2) {
      ctl.output_timestep_number += last_jump - 1;
      if (last_jump > 1) {
        ctl.dcout
            << "Cycle jump is done successfully. The number of jumped cycle: "
            << last_jump << " (including the preceding cycle)." << std::endl;
        this->save_results = true;
        ctl.params.save_vtk_per_step = 1e10;
      }
      n_jump = 0;
      ctl.set_info("N jump", n_jump);
      n_resolved_cycles++;
      ctl.params.refine = refine_state;
    } else {
      if (std::abs(subcycle - 1) < 1e-8) {
        n_resolved_cycles++;
      }
      ctl.output_timestep_number += (std::abs(subcycle - 1) < 1e-8) ? 0 : (-1);
    }
    ctl.dcout << "The number of resolved cycles: " << n_resolved_cycles
              << std::endl;
  }

  bool save_checkpoint(Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 2) {
      return true;
    } else {
      return false;
    }
  }

  std::string return_solution_or_checkpoint(Controller<dim> &ctl) override {
    if (std::abs(subcycle) < 1e-8 && n_jump > 0) {
      return "checkpoint";
    } else {
      return "solution";
    }
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time / T >= expected_cycles && std::abs(subcycle - 1) < 1e-8) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else {
      return AdaptiveTimeStep<dim>::terminate(ctl);
    }
  }
};

template <int dim> class JacconCycleJump : public ConstantTimeStep<dim> {
public:
  double f, T;
  unsigned int n_jump;
  unsigned int expected_cycles;
  unsigned int n_resolved_cycles;

  double trial_start;
  double trial_start_output_time, trial_start_timestep_number;

  double subcycle;
  double initial_max_alpha;
  unsigned int n_trials;
  double last_last_residual, last_residual, residual;

  bool refine_state;

  JacconCycleJump(Controller<dim> &ctl)
      : ConstantTimeStep<dim>(ctl), n_jump(0), n_resolved_cycles(0),
        trial_start(-1), trial_start_output_time(-1),
        trial_start_timestep_number(-1), subcycle(0), initial_max_alpha(1e10),
        n_trials(0), last_last_residual(1e9), last_residual(1e8),
        residual(1e7) {
    if (ctl.params.fatigue_accumulation != "Jaccon") {
      ctl.dcout << "JacconCycleJump is expected to used with "
                   "JacconAccumulation, but it's not. Please make sure "
                   "that the accumulation rule supports cycle jump."
                << std::endl;
    }
    AssertThrow(
        ctl.params.adaptive_timestep_parameters != "",
        ExcInternalError("Parameters of JacconCycleJump is not assigned."));
    std::istringstream iss(ctl.params.adaptive_timestep_parameters);
    iss >> f >> n_jump;
    T = 1 / f;
    ctl.set_info("N jump", n_jump);
    ctl.set_info("N trials", 0);
    expected_cycles =
        std::round((ctl.params.timestep * (ctl.params.switch_timestep) +
                    ctl.params.timestep_size_2 * (ctl.params.max_no_timesteps -
                                                  ctl.params.switch_timestep)) /
                   T);
    AssertThrow(
        std::fmod(T, ctl.params.timestep) < 1e-8 &&
            std::fmod(T, ctl.params.timestep_size_2) < 1e-8,
        ExcInternalError("The period has to be divisible by the time step"));
  };

  void initialize_timestep(Controller<dim> &ctl) {
    ctl.dcout << "JacconCycleJump using parameter: f=" << f
              << " Hz, n_jump=" << n_jump << std::endl;
    ctl.params.save_vtk_per_step = 1e10;
    ctl.dcout << "JacconCycleJump will save outputs at the end of a successful "
                 "cycle jump."
              << std::endl;
  }

  double current_timestep(Controller<dim> &ctl) override {
    if (ctl.current_timestep != ctl.params.timestep_size_2)
      return ctl.current_timestep;
    double time_step = ctl.current_timestep;
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 1) {
      trial_start = ctl.time;
      trial_start_output_time = ctl.output_timestep_number;
      trial_start_timestep_number = ctl.timestep_number;
      time_step = T * (n_jump - 1);
      ctl.dcout << "Doing cycle jump" << std::endl;
      refine_state = ctl.params.refine;
      if (refine_state) {
        ctl.dcout << "Refinement is disabled temporarily" << std::endl;
      }
      ctl.params.refine = false; // We need the checkpoint work.
    } else if (n_resolved_cycles >= 1) {
      ctl.dcout << "*********************************" << std::endl;
      ctl.dcout << "********** Trial cycle **********" << std::endl;
      ctl.dcout << "*********************************" << std::endl;
      ctl.dcout << "The number of trial iterations: " << n_trials + 1
                << std::endl;
    }

    if (std::abs(subcycle - 1) > 1e-8) {
      subcycle += std::fmod(ctl.current_timestep / T, 1);
    } else {
      subcycle = 0.0;
    }
    ctl.set_info("Subcycle", subcycle);
    ctl.dcout << "Current subcycle: " << subcycle << std::endl;
    return time_step;
  }

  bool fail(double newton_reduction, Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 1) {
      residual = GlobalEstimator::absmax<dim>("Residual", 0.0, ctl);
      ctl.dcout << "Current max trapezoidal residual: " << residual
                << " Target: " << initial_max_alpha * 1e-6 << std::endl;
      return residual > initial_max_alpha * 1e-6;
    } else {
      return newton_reduction > ctl.params.upper_newton_rho;
    }
  }

  double get_new_timestep_when_fail(Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 1) {
      ctl.time = trial_start;
      ctl.timestep_number = trial_start_timestep_number;
      ctl.output_timestep_number = trial_start_output_time;

      subcycle = 0;
      ctl.set_info("Subcycle", subcycle);
      n_trials++;
      ctl.set_info("N trials", n_trials);
      n_resolved_cycles++;

      trial_start = ctl.time;
      trial_start_output_time = ctl.output_timestep_number;
      trial_start_timestep_number = ctl.timestep_number;

      ctl.dcout << "Trial jump failed (the global highest residual is higher "
                   "than the tolerance). Returning to time ("
                << trial_start_timestep_number - 1 << ") " << trial_start
                << " and jump again" << std::endl;
    }
    return T * (n_jump - 1);
  }

  void failure_criteria(Controller<dim> &ctl) override {
    if (!(std::abs(subcycle) < 1e-8 && n_resolved_cycles >= 1)) {
      throw std::runtime_error("Staggered scheme does not converge when no"
                               "cycle jump is performed.");
    } else if (n_trials > 100 || (last_residual > last_last_residual &&
                                  residual > last_residual)) {
      throw std::runtime_error(
          "Trapezoidal iterative extrapolation does not converge. Consider "
          "using a smaller cycle jump.");
    } else {
      last_last_residual = last_residual;
      last_residual = residual;
    }
  }

  void after_step(Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 1) {
      ctl.output_timestep_number += n_jump - 1;
      ctl.dcout << "Cycle jump is done successfully. The number of trials: "
                << n_trials + 1 << std::endl;
      n_resolved_cycles++;
      n_trials = 0;
      ctl.set_info("N trials", 0);
      last_last_residual = 1e9;
      last_residual = 1e8;
      residual = 1e7;
      this->save_results = true;
      ctl.params.refine = refine_state;
    } else {
      if (std::abs(subcycle - 1) < 1e-8) {
        n_resolved_cycles++;
      }
      ctl.output_timestep_number += (std::abs(subcycle - 1) < 1e-8) ? 0 : (-1);
    }
    if (std::abs(subcycle) < 1e-8 && n_trials == 0) {
      initial_max_alpha =
          GlobalEstimator::absmax<dim>("Fatigue history", 0.0, ctl);
    }
    if (std::abs(subcycle - 1) < 1e-8) {
      double crack_length =
          GlobalEstimator::sum<dim>("Diffusion JxW", 0.0, ctl);
      ctl.dcout << "Crack length estimated by diffusion: " << crack_length
                << " at cycle " << ctl.output_timestep_number + 1 << std::endl;
    }
    ctl.dcout << "Maximum fatigue variable: "
              << GlobalEstimator::max<dim>("Fatigue history", 0.0, ctl)
              << std::endl;
    ctl.dcout << "The number of resolved cycles: " << n_resolved_cycles
              << std::endl;
  }

  bool save_checkpoint(Controller<dim> &ctl) override {
    if (std::abs(subcycle - 1) < 1e-8 && n_resolved_cycles >= 1) {
      return true;
    } else {
      return false;
    }
  }

  std::string return_solution_or_checkpoint(Controller<dim> &ctl) override {
    if (std::abs(subcycle) < 1e-8 && n_resolved_cycles >= 1) {
      return "checkpoint";
    } else {
      return "solution";
    }
  }

  bool terminate(Controller<dim> &ctl) override {
    if (ctl.time / T >= expected_cycles && std::abs(subcycle - 1) < 1e-8) {
      ctl.dcout << "Terminating as the number of cycles reaches the expected "
                   "number (Max no of timestep in the configuration)."
                << std::endl;
      return true;
    } else {
      return AdaptiveTimeStep<dim>::terminate(ctl);
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
  else if (method == "YangCycleJump")
    return std::make_unique<YangCycleJump<dim>>(ctl);
  else if (method == "JacconCycleJump")
    return std::make_unique<JacconCycleJump<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_ADAPTIVE_TIMESTEP_H
