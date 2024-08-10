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
std::unique_ptr<FatigueDegradation<dim>>
select_fatigue_degradation(std::string method, Controller<dim> &ctl) {
  if (method == "CarraraAsymptotic")
    return std::make_unique<CarraraAsymptoticFatigueDegradation<dim>>(ctl);
  else if (method == "KristensenAsymptotic")
    return std::make_unique<KristensenAsymptoticFatigueDegradation<dim>>(ctl);
  else
    AssertThrow(false, ExcNotImplemented());
}

#endif // CRACKS_FATIGUE_DEGRADATION_H
