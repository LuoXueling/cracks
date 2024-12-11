//
// Created by xlluo on 24-8-5.
//

#ifndef CRACKS_FATIGUE_DEGRADATION_H
#define CRACKS_FATIGUE_DEGRADATION_H

#include "controller.h"
#include "dealii_includes.h"
using namespace dealii;

template <int dim> class FatigueAccumulation {
public:
  FatigueAccumulation(Controller<dim> &ctl){};
  void step(const std::shared_ptr<PointHistory> &lqph_q, double phasefield,
            double degrade, double degrade_derivative,
            double degrade_second_derivative, Controller<dim> &ctl) {
    double increm = increment(lqph_q, phasefield, degrade, degrade_derivative,
                              degrade_second_derivative, ctl);
    lqph_q->update("Fatigue history", std::max(increm, 0.0), "accumulate");
    record(lqph_q, phasefield, degrade, degrade_derivative,
           degrade_second_derivative, ctl);
  }

  virtual double increment(const std::shared_ptr<PointHistory> &lqph,
                           double phasefield, double degrade,
                           double degrade_derivative,
                           double degrade_second_derivative,
                           Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
  };

  virtual void record(const std::shared_ptr<PointHistory> &lqph,
                      double phasefield, double degrade,
                      double degrade_derivative,
                      double degrade_second_derivative, Controller<dim> &ctl) {
    // If one is using the latest fatigue history to record something, it's
    // necessary to record it in this function, because if solving the phase
    // field first and then the displacement, and the multistep staggered is not
    // triggered, the record in increment() is still using the old fatigue
    // variable updated by old elastic energy. The issue happens in VHCF.
  };
};

template <int dim>
class CarraraNoMeanEffectAccumulation : public FatigueAccumulation<dim> {
public:
  CarraraNoMeanEffectAccumulation(Controller<dim> &ctl)
      : FatigueAccumulation<dim>(ctl){};

  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    double dpsi =
        lqph->get_increment_latest("Positive elastic energy") * degrade;
    double increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
    return increm;
  };
};

template <int dim>
class KristensenAccumulation : public FatigueAccumulation<dim> {
public:
  KristensenAccumulation(Controller<dim> &ctl)
      : FatigueAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    double dpsi = lqph->get_increment_latest("Positive elastic energy", 0.0);
    double increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
    return increm;
  };
};

template <int dim>
class KristensenCLAAccumulation : public FatigueAccumulation<dim> {
public:
  KristensenCLAAccumulation(Controller<dim> &ctl)
      : FatigueAccumulation<dim>(ctl) {
    AssertThrow(ctl.params.adaptive_timestep == "KristensenCLA",
                ExcInternalError("KristensenCLATimeStep must be used "
                                 "with KristensenCLAAccumulation."));
    AssertThrow(
        ctl.params.fatigue_accumulation_parameters != "",
        ExcInternalError(
            "Parameters of KristensenCLAAccumulation is not assigned."));
    std::istringstream iss(ctl.params.fatigue_accumulation_parameters);
    iss >> R;
    AssertThrow(
        R >= 0 || (R < 0 && ctl.params.degradation == "hybridnotension"),
        ExcInternalError("Cannot use KristensenCLAAccumulation when "
                         "R<0 while hybridnotension split is not used"));
  };
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    if (ctl.current_timestep != ctl.params.timestep_size_2) {
      double dpsi =
          lqph->get_increment_latest("Positive elastic energy", 0.0) * degrade;
      double increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
      return increm;
    } else {
      double psi = lqph->get_latest("Positive elastic energy", 0.0) * degrade;
      double n_jump = ctl.get_info("N jump", 1);
      double increm = psi * (1 - R * R * (R >= 0 ? 1 : 0)) * n_jump;
      return increm;
    }
  };
  double R;
};

template <int dim>
class CojocaruAccumulation : public FatigueAccumulation<dim> {
public:
  CojocaruAccumulation(Controller<dim> &ctl) : FatigueAccumulation<dim>(ctl) {
    AssertThrow(ctl.params.fatigue_accumulation_parameters != "",
                ExcInternalError(
                    "Parameters of CojocaruCLAAccumulation is not assigned."));
    std::istringstream iss(ctl.params.fatigue_accumulation_parameters);
    iss >> R >> q_jump;
  };
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    int n_jumps = static_cast<int>(ctl.get_info("N jump", 0.0));
    double increm;
    if (n_jumps == 0 || ctl.current_timestep != ctl.params.timestep_size_2) {
      // Regular accumulation
      double dpsi =
          lqph->get_increment_latest("Positive elastic energy", 0.0) * degrade;
      increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
    } else {
      double s12 = lqph->get_initial("s12", 0.0);
      double s23 = lqph->get_initial("s23", 0.0);
      increm = s12 * n_jumps + (s12 - s23) * std::pow(n_jumps, 2) / 2.0;
    }
    return increm;
  };

  void record(const std::shared_ptr<PointHistory> &lqph, double phasefield,
              double degrade, double degrade_derivative,
              double degrade_second_derivative, Controller<dim> &ctl) {
    // Determine the number of jumps
    double subcycle = ctl.get_info("Subcycle", 0.0);
    if (std::abs(subcycle - 1) < 1e-8) {
      lqph->update("y3", lqph->get_latest("Fatigue history", 0.0));
    } else if (std::abs(subcycle - 2) < 1e-8) {
      double y3 = lqph->get_latest("y3", 0.0);
      double y2 = lqph->get_latest("Fatigue history", 0.0);
      lqph->update("y2", y2);
      lqph->update("s23", y2 - y3);
    } else if (std::abs(subcycle - 3) < 1e-8) {
      double s23 = lqph->get_latest("s23", 0.0);
      double y1 = lqph->get_latest("Fatigue history", 0.0);
      double y2 = lqph->get_latest("y2", 0.0);
      double s12 = y1 - y2;
      lqph->update("s12", s12);
      double max_jump = ctl.get_info("Maximum jump", 1.0e8);
      double n_jump_local = max_jump;
      if (phasefield < 0.95 && phasefield > 0.001 &&
          std::abs(s12 - s23) / std::abs(s12) > 1e-5) {
        n_jump_local = q_jump * s12 / std::abs(s12 - s23);
      }
      lqph->update("n_jump_local", n_jump_local);
    }
  }
  double R, q_jump;
};

template <int dim>
class CojocaruCLAAccumulation : public CojocaruAccumulation<dim> {
public:
  CojocaruCLAAccumulation(Controller<dim> &ctl)
      : CojocaruAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    int n_jumps = static_cast<int>(ctl.get_info("N jump", 0.0));
    double increm;
    if (n_jumps == 0 || ctl.current_timestep != ctl.params.timestep_size_2) {
      // Regular accumulation
      if (ctl.current_timestep != ctl.params.timestep_size_2) {
        double dpsi =
            lqph->get_increment_latest("Positive elastic energy", 0.0) *
            degrade;
        increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
      } else {
        double psi = lqph->get_latest("Positive elastic energy", 0.0) * degrade;
        increm = psi * (1 - this->R * this->R * (this->R >= 0 ? 1 : 0));
      }
    } else {
      double s12 = lqph->get_initial("s12", 0.0);
      double s23 = lqph->get_initial("s23", 0.0);
      increm = s12 * n_jumps + (s12 - s23) * std::pow(n_jumps, 2) / 2.0;
    }
    return increm;
  };
};

template <int dim> class JonasAccumulation : public FatigueAccumulation<dim> {
public:
  JonasAccumulation(Controller<dim> &ctl) : FatigueAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    int n_jumps = static_cast<int>(ctl.get_info("N jump", 0.0));
    double increm = 0;
    double trial_cycle = ctl.get_info("Trial cycle", 0.0);

    if ((n_jumps == 0 && std::abs(trial_cycle) < 1e-8) ||
        ctl.current_timestep != ctl.params.timestep_size_2) {
      // Regular accumulation
      double dpsi =
          lqph->get_increment_latest("Positive elastic energy") * degrade;
      increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
    } else if (n_jumps > 0) {
      double y1 = lqph->get_initial("y1", 0.0);
      double y2 = lqph->get_initial("y2", 0.0);
      double y3 = lqph->get_initial("y3", 0.0);
      double y4 = lqph->get_initial("y4", 0.0);
      increm =
          1.0 / 6.0 * (-2.0 * y1 + 9.0 * y2 - 18.0 * y3 + 11.0 * y4) * n_jumps +
          0.5 * (-y1 + 4.0 * y2 - 5.0 * y3 + 2.0 * y4) * std::pow(n_jumps, 2.0);
    }
    return increm;
  };

  void record(const std::shared_ptr<PointHistory> &lqph, double phasefield,
              double degrade, double degrade_derivative,
              double degrade_second_derivative, Controller<dim> &ctl) {
    double subcycle = ctl.get_info("Subcycle", 0.0);
    if (std::fmod(subcycle, 1) < 1e-8 && subcycle < 5 && subcycle > 1e-8) {
      // y1, y2, y3, and y4
      lqph->update(
          "y" + std::to_string(static_cast<unsigned int>(std::round(subcycle))),
          lqph->get_latest("Fatigue history", 0.0));
    }
  }
};

template <int dim> class JonasCLAAccumulation : public JonasAccumulation<dim> {
public:
  JonasCLAAccumulation(Controller<dim> &ctl) : JonasAccumulation<dim>(ctl) {
    AssertThrow(ctl.params.fatigue_accumulation_parameters != "",
                ExcInternalError(
                    "Parameters of JonasCLAAccumulation is not assigned."));
    std::istringstream iss(ctl.params.fatigue_accumulation_parameters);
    iss >> R;
  };
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    int n_jumps = static_cast<int>(ctl.get_info("N jump", 0.0));
    double increm = 0;
    double trial_cycle = ctl.get_info("Trial cycle", 0.0);

    if ((n_jumps == 0 && std::abs(trial_cycle) < 1e-8) ||
        ctl.current_timestep != ctl.params.timestep_size_2) {
      // Regular accumulation
      if (ctl.current_timestep != ctl.params.timestep_size_2) {
        double dpsi =
            lqph->get_increment_latest("Positive elastic energy", 0.0) *
            degrade;
        increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
      } else {
        double psi = lqph->get_latest("Positive elastic energy", 0.0) * degrade;
        increm = psi * (1 - R * R * (R >= 0 ? 1 : 0));
      }
    } else if (n_jumps > 0) {
      double y1 = lqph->get_initial("y1", 0.0);
      double y2 = lqph->get_initial("y2", 0.0);
      double y3 = lqph->get_initial("y3", 0.0);
      double y4 = lqph->get_initial("y4", 0.0);
      increm =
          1.0 / 6.0 * (-2.0 * y1 + 9.0 * y2 - 18.0 * y3 + 11.0 * y4) * n_jumps +
          0.5 * (-y1 + 4.0 * y2 - 5.0 * y3 + 2.0 * y4) * std::pow(n_jumps, 2.0);
    }
    return increm;
  };
  double R;
};

template <int dim>
class JonasNodegradeAccumulation : public JonasAccumulation<dim> {
public:
  JonasNodegradeAccumulation(Controller<dim> &ctl)
      : JonasAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    return JonasAccumulation<dim>::increment(
        lqph, phasefield, ctl.params.constant_k + 1, -2.0, 0.0, ctl);
  };
};

template <int dim> class YangAccumulation : public FatigueAccumulation<dim> {
public:
  YangAccumulation(Controller<dim> &ctl) : FatigueAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    double n_jumps = ctl.get_info("N jump", 0.0);
    double increm = 0;
    double subcycle = ctl.get_info("Subcycle", 0.0);

    if (std::abs(subcycle - 0) > 1e-8 || std::abs(n_jumps) < 1e-8 ||
        ctl.current_timestep != ctl.params.timestep_size_2) {
      // Regular accumulation
      double dpsi =
          lqph->get_increment_latest("Positive elastic energy") * degrade;
      increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
    } else {
      double last_jump = ctl.get_info("Last jump", 0.0);
      double new_increment = lqph->get_initial("Fast increment", 0.0);
      double diff = lqph->get_initial("Fast increment diff", 0.0);
      double extra_increm = new_increment + n_jumps / last_jump * diff;
      increm = (n_jumps - 1) * (extra_increm + new_increment) / 2;
    }
    return increm;
  };

  void record(const std::shared_ptr<PointHistory> &lqph, double phasefield,
              double degrade, double degrade_derivative,
              double degrade_second_derivative, Controller<dim> &ctl) {
    double subcycle = ctl.get_info("Subcycle", 0.0);

    if (std::fmod(subcycle, 1) < 1e-8) {
      lqph->update("Initial history", lqph->get_latest("Fatigue history", 0.0));
      if (std::abs(subcycle - 1) < 1e-8) {
        double new_increment = lqph->get_latest("Fatigue history", 0.0) -
                               lqph->get_initial("Initial history", 0.0);
        double old_increment =
            lqph->get_initial("Fast increment", new_increment);
        lqph->update("Fast increment", new_increment);
        lqph->update("Fast increment diff", new_increment - old_increment);
      }
    }
  }
};

template <int dim> class JacconAccumulation : public FatigueAccumulation<dim> {
public:
  JacconAccumulation(Controller<dim> &ctl) : FatigueAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    double n_jumps = ctl.get_info("N jump", 0.0);
    double increm = 0;
    double subcycle = ctl.get_info("Subcycle", 0.0);

    if (std::abs(subcycle - 0) > 1e-8 ||
        ctl.current_timestep != ctl.params.timestep_size_2) {
      // Regular accumulation
      double dpsi =
          lqph->get_increment_latest("Positive elastic energy") * degrade;
      increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi;
    } else {
      double n_trials = ctl.get_info("N trials", 0);
      if (std::abs(n_trials) < 1e-8) {
        increm = n_jumps * lqph->get_initial("Trial increment");
      } else {
        double last_increm = lqph->get_independent_initial("increm", 0.0);
        double residual = lqph->get_independent_latest(
            "Residual", 0.0); // pointhistory is not finalized when it has not
                              // converged, so we need the latest one.
        increm = last_increm - (-1) * residual;
      }
      lqph->update_independent("increm", increm);
      increm -= lqph->get_initial("Trial increment"); // we are at cycle N+1
    }
    return increm;
  };

  void record(const std::shared_ptr<PointHistory> &lqph, double phasefield,
              double degrade, double degrade_derivative,
              double degrade_second_derivative, Controller<dim> &ctl) {
    double n_jumps = ctl.get_info("N jump", 0.0);
    double subcycle = ctl.get_info("Subcycle", 0.0);
    if (std::fmod(subcycle, 1) < 1e-8) {
      double alpha_n1 = lqph->get_latest("Fatigue history", 0.0);
      lqph->update("alpha_n1", alpha_n1);
      if (std::abs(subcycle - 1) < 1e-8) {
        double alpha_n0 = lqph->get_initial("alpha_n1", 0.0);
        double new_increment = alpha_n1 - alpha_n0;
        lqph->update("Trial increment", new_increment);
        lqph->update("alpha_0", alpha_n0);
        lqph->update("alpha_1", alpha_n1);
      }
    }
    if (std::abs(subcycle - 1) < 1e-8) {
      double alpha_n0 = lqph->get_initial("alpha_n1", 0.0);
      double alpha_n1 = lqph->get_latest("alpha_n1", 0.0);
      double alpha_0 = lqph->get_initial("alpha_0", 0.0);
      double alpha_1 = lqph->get_initial("alpha_1", 0.0);
      double residual = alpha_0 * (1 - n_jumps / 2) +
                        (alpha_n1 + alpha_1) * (n_jumps / 2) -
                        alpha_n0 * (1 + n_jumps / 2);
      lqph->update_independent("Residual", residual);
    }
  }
};

template <int dim>
class JacconNodegradeAccumulation : public JacconAccumulation<dim> {
public:
  JacconNodegradeAccumulation(Controller<dim> &ctl)
      : JacconAccumulation<dim>(ctl){};
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    return JacconAccumulation<dim>::increment(
        lqph, phasefield, ctl.params.constant_k + 1, -2.0, 0.0, ctl);
  };
};

template <int dim>
class CarraraMeanEffectAccumulation : public FatigueAccumulation<dim> {
public:
  CarraraMeanEffectAccumulation(Controller<dim> &ctl)
      : FatigueAccumulation<dim>(ctl) {
    // Eq. 47 does not match any of alpha_t claimed in the result section
    // We have to multiply another 0.5 to reproduce the results.
    double epsilon_at2 =
        std::sqrt(ctl.params.Gc / (3 * ctl.params.l_phi * ctl.params.E));
    if (ctl.params.fatigue_accumulation_parameters == "") {
      alpha_n = 0.5 * 0.5 * epsilon_at2 * ctl.params.E * epsilon_at2;
      ctl.dcout << "Using alpha_n: " << alpha_n << std::endl;
    } else {
      std::istringstream iss(ctl.params.fatigue_accumulation_parameters);
      iss >> alpha_n;
      ctl.dcout << "Using alpha_n: " << alpha_n << "from configuration"
                << std::endl;
    }
  };

  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    double dpsi =
        lqph->get_increment_latest("Positive elastic energy", 0.0) * degrade;
    double psi = lqph->get_latest("Positive elastic energy", 0.0) * degrade;
    double increm = (dpsi > 0 ? 1.0 : 0.0) * dpsi * psi / alpha_n;
    return increm;
  };

private:
  double alpha_n;
};

template <int dim>
std::unique_ptr<FatigueAccumulation<dim>>
select_fatigue_accumulation(std::string method, Controller<dim> &ctl) {
  if (method == "CarraraNoMeanEffect")
    return std::make_unique<CarraraNoMeanEffectAccumulation<dim>>(ctl);
  else if (method == "CarraraMeanEffect")
    return std::make_unique<CarraraMeanEffectAccumulation<dim>>(ctl);
  else if (method == "Kristensen")
    return std::make_unique<KristensenAccumulation<dim>>(ctl);
  else if (method == "KristensenCLA")
    return std::make_unique<KristensenCLAAccumulation<dim>>(ctl);
  else if (method == "Cojocaru")
    return std::make_unique<CojocaruAccumulation<dim>>(ctl);
  else if (method == "CojocaruCLA")
    return std::make_unique<CojocaruCLAAccumulation<dim>>(ctl);
  else if (method == "Jonas")
    return std::make_unique<JonasAccumulation<dim>>(ctl);
  else if (method == "JonasCLA")
    return std::make_unique<JonasCLAAccumulation<dim>>(ctl);
  else if (method == "JonasNodegrade")
    return std::make_unique<JonasNodegradeAccumulation<dim>>(ctl);
  else if (method == "Yang")
    return std::make_unique<YangAccumulation<dim>>(ctl);
  else if (method == "Jaccon")
    return std::make_unique<JacconAccumulation<dim>>(ctl);
  else if (method == "JacconNodegrade")
    return std::make_unique<JacconNodegradeAccumulation<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

template <int dim> class FatigueDegradation {
public:
  FatigueDegradation(Controller<dim> &ctl){};
  virtual double degradation_value(const std::shared_ptr<PointHistory> &lqph,
                                   double phasefield, double degrade,
                                   Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
  };
};

// https://www.sciencedirect.com/science/article/pii/S0045782519306218
template <int dim>
class CarraraAsymptoticFatigueDegradation : public FatigueDegradation<dim> {
public:
  CarraraAsymptoticFatigueDegradation(Controller<dim> &ctl)
      : FatigueDegradation<dim>(ctl) {
    if (ctl.params.fatigue_degradation_parameters == "") {
      // Eq. 47 does not match any of alpha_t claimed in the result section
      // We have to multiply another 0.5 to reproduce the results.
      double epsilon_at2 =
          std::sqrt(ctl.params.Gc / (3 * ctl.params.l_phi * ctl.params.E));
      alpha_t = 0.5 * 0.5 * epsilon_at2 * ctl.params.E * epsilon_at2;
      ctl.dcout << "Using alpha_t: " << alpha_t << std::endl;
    } else {
      std::istringstream iss(ctl.params.fatigue_degradation_parameters);
      iss >> alpha_t;
      ctl.dcout << "Using alpha_t: " << alpha_t << " from configuration"
                << std::endl;
    }
  };
  double degradation_value(const std::shared_ptr<PointHistory> &lqph,
                           double phasefield, double phasefield_degrade,
                           Controller<dim> &ctl) override {

    double degrade;
    double alpha = lqph->get_latest("Fatigue history", 0.0);
    if (alpha <= alpha_t) {
      degrade = 1;
    } else {
      degrade = std::pow(2 * alpha_t / (alpha + alpha_t), 2);
    }
    return degrade;
  };

  double alpha_t;
};

// https://www.sciencedirect.com/science/article/pii/S0045782519306218
template <int dim>
class CarraraLogarithmicFatigueDegradation : public FatigueDegradation<dim> {
public:
  CarraraLogarithmicFatigueDegradation(Controller<dim> &ctl)
      : FatigueDegradation<dim>(ctl) {
    AssertThrow(
        ctl.params.fatigue_degradation_parameters != "",
        ExcInternalError("Parameters of CarraraLogarithmicFatigueDegradation "
                         "is not assigned."));
    std::istringstream iss(ctl.params.fatigue_degradation_parameters);
    iss >> alpha_t >> kappa;
    ctl.dcout << "Using alpha_t: " << alpha_t << " and kappa: " << kappa
              << std::endl;
  };
  double degradation_value(const std::shared_ptr<PointHistory> &lqph,
                           double phasefield, double phasefield_degrade,
                           Controller<dim> &ctl) override {
    double degrade;
    double alpha = lqph->get_latest("Fatigue history", 0.0);
    if (alpha <= alpha_t) {
      degrade = 1;
    } else if (alpha <= (alpha_t * std::pow(10, 1 / kappa))) {
      degrade = std::pow(1 - kappa * std::log10(alpha / alpha_t), 2);
    } else {
      degrade = 0;
    }
    return degrade;
  };

  double alpha_t, kappa;
};

template <int dim>
class KristensenAsymptoticFatigueDegradation
    : public CarraraAsymptoticFatigueDegradation<dim> {
public:
  KristensenAsymptoticFatigueDegradation(Controller<dim> &ctl)
      : CarraraAsymptoticFatigueDegradation<dim>(ctl) {
    this->alpha_t = ctl.params.Gc / (12 * ctl.params.l_phi);
  };
};

template <int dim>
class CojocaruAsymptoticFatigueDegradation : public FatigueDegradation<dim> {
public:
  CojocaruAsymptoticFatigueDegradation(Controller<dim> &ctl)
      : FatigueDegradation<dim>(ctl) {
    AssertThrow(ctl.params.fatigue_degradation_parameters != "",
                ExcInternalError(
                    "Parameters of CojocaruCLAAccumulation is not assigned."));
    std::istringstream iss(ctl.params.fatigue_degradation_parameters);
    iss >> alpha_t;
  };
  double degradation_value(const std::shared_ptr<PointHistory> &lqph,
                           double phasefield, double phasefield_degrade,
                           Controller<dim> &ctl) override {
    double alpha = lqph->get_latest("Fatigue history", 0.0);
    double degrade = std::pow(alpha_t / (alpha + alpha_t), 2);
    return degrade;
  };
  double alpha_t;
};

template <int dim>
std::unique_ptr<FatigueDegradation<dim>>
select_fatigue_degradation(std::string method, Controller<dim> &ctl) {
  if (method == "CarraraAsymptotic")
    return std::make_unique<CarraraAsymptoticFatigueDegradation<dim>>(ctl);
  else if (method == "CarraraLogarithmic")
    return std::make_unique<CarraraLogarithmicFatigueDegradation<dim>>(ctl);
  else if (method == "KristensenAsymptotic")
    return std::make_unique<KristensenAsymptoticFatigueDegradation<dim>>(ctl);
  else if (method == "CojocaruAsymptotic")
    return std::make_unique<CojocaruAsymptoticFatigueDegradation<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_FATIGUE_DEGRADATION_H
