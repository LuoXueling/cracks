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
    lqph_q->update("Fatigue history", increm, "accumulate");
  }

  virtual double increment(const std::shared_ptr<PointHistory> &lqph,
                           double phasefield, double degrade,
                           double degrade_derivative,
                           double degrade_second_derivative,
                           Controller<dim> &ctl) {
    AssertThrow(false, ExcNotImplemented());
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
    double dpsi = lqph->get_increment("Positive elastic energy", 0.0) * degrade;
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
    double dpsi = lqph->get_increment("Positive elastic energy", 0.0);
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
    double psi = lqph->get("Positive elastic energy", 0.0);
    double increm = psi * (1 - R * R * (R >= 0 ? 1 : 0));
    return increm;
  };
  double R;
};

template <int dim>
class CojocaruCLAAccumulation : public FatigueAccumulation<dim> {
public:
  CojocaruCLAAccumulation(Controller<dim> &ctl)
      : FatigueAccumulation<dim>(ctl) {
    AssertThrow(ctl.params.adaptive_timestep == "KristensenCLA" ||
                    ctl.params.adaptive_timestep == "CojocaruCycleJump",
                ExcInternalError(
                    "KristensenCLATimeStep or CojocaruCycleJump must be used "
                    "with CojocaruCLAAccumulation."));
    AssertThrow(ctl.params.fatigue_accumulation_parameters != "",
                ExcInternalError(
                    "Parameters of CojocaruCLAAccumulation is not assigned."));
    alpha_c = 3 * ctl.params.Gc / (16 * ctl.params.l_phi);
    std::istringstream iss(ctl.params.fatigue_accumulation_parameters);
    iss >> R >> alpha_t >> Se;
    if (!iss.eof()) {
      iss >> q_jump;
    } else {
      q_jump = 0;
    }
    alpha_e = std::pow(Se, 2) / 2 / ctl.params.E;
  };
  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    int n_jumps = static_cast<int>(ctl.get_info("N jump", 0.0));
    double increm;
    if (n_jumps == 0) {
      // Regular accumulation
      double psi = lqph->get("Positive elastic energy", 0.0) * degrade;
      double H = lqph->get("Driving force", 0.0);
      double alpha_max = std::max(lqph->get("Accumulated increment", 0.0), psi);
      if (H * (1 - R) / 2 > alpha_e) {
        increm = alpha_max * (1 - R) / 2 / alpha_c;
        lqph->update("Accumulated increment", 0.0, "latest");
      } else {
        increm = 0.0;
        lqph->update("Accumulated increment", alpha_max, "latest");
      }
      // Determine the number of jumps
      double subcycle = ctl.get_info("Subcycle", 0.0);
      if (subcycle == 2) {
        lqph->update("y3", lqph->get("Fatigue history", 0.0));
      } else if (subcycle == 3) {
        double y3 = lqph->get("y3", 0.0);
        double y2 = lqph->get("Fatigue history", 0.0);
        lqph->update("y2", y2);
        lqph->update("s23", y2 - y3);
      } else if (subcycle == 4) {
        double s23 = lqph->get("s23", 0.0);
        double y1 = lqph->get("Fatigue history", 0.0);
        double y2 = lqph->get("y2", 0.0);
        double s12 = y1 - y2;
        lqph->update("s12", s12);
        double max_jump = ctl.get_info("Maximum jump", 1.0e8);
        double n_jump_local = ctl.get_info("N jump local", max_jump);
        if (phasefield < 0.95 && std::abs(s12 - s23) / std::abs(s12) > 1e-5) {
          n_jump_local =
              std::min(q_jump * s12 / std::abs(s12 - s23), n_jump_local);
        }
        ctl.set_info("N jump local", n_jump_local);
      }
    } else {
      double s12 = lqph->get("s12", 0.0);
      double s23 = lqph->get("s23", 0.0);
      increm = s12 * n_jumps + (s12 - s23) * std::pow(n_jumps, 2) / 2.0;
    }
    return increm;
  };
  double R, alpha_c, alpha_t, Se, alpha_e, q_jump;
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
    alpha_n = 0.5 * 0.5 * epsilon_at2 * ctl.params.E * epsilon_at2;
  };

  double increment(const std::shared_ptr<PointHistory> &lqph, double phasefield,
                   double degrade, double degrade_derivative,
                   double degrade_second_derivative,
                   Controller<dim> &ctl) override {
    double dpsi = lqph->get_increment("Positive elastic energy", 0.0) * degrade;
    double psi = lqph->get("Positive elastic energy", 0.0) * degrade;
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
  else if (method == "CojocaruCLA")
    return std::make_unique<CojocaruCLAAccumulation<dim>>(ctl);
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
    double epsilon_at2 =
        std::sqrt(ctl.params.Gc / (3 * ctl.params.l_phi * ctl.params.E));
    // Eq. 47 does not match any of alpha_t claimed in the result section
    // We have to multiply another 0.5 to reproduce the results.
    alpha_t = 0.5 * 0.5 * epsilon_at2 * ctl.params.E * epsilon_at2;
  };
  double degradation_value(const std::shared_ptr<PointHistory> &lqph,
                           double phasefield, double phasefield_degrade,
                           Controller<dim> &ctl) override {

    double degrade;
    double alpha = lqph->get("Fatigue history", 0.0);
    if (alpha <= alpha_t) {
      degrade = 1;
    } else {
      degrade = std::pow(2 * alpha_t / (alpha + alpha_t), 2);
    }
    return degrade;
  };

  double alpha_t;
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
    double alpha = lqph->get("Fatigue history", 0.0);
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
  else if (method == "KristensenAsymptotic")
    return std::make_unique<KristensenAsymptoticFatigueDegradation<dim>>(ctl);
  else if (method == "CojocaruAsymptotic")
    return std::make_unique<CojocaruAsymptoticFatigueDegradation<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_FATIGUE_DEGRADATION_H
