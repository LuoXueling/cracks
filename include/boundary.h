//
// Created by xlluo on 24-7-28.
//

#ifndef CRACKS_BOUNDARY_H
#define CRACKS_BOUNDARY_H

#include "dealii_includes.h"
#include <cmath>

// Can only use apply AbstractBoundary condition on one dof
template <int dim> class AbstractBoundary : public Function<dim> {
public:
  AbstractBoundary(double present_time_inp, unsigned int n_components_inp);

  virtual double value(const Point<dim> &p,
                       unsigned int component) const override {
    return 0.0;
  };

  void vector_value(const Point<dim> &p, Vector<double> &values) const override;

  void
  vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &value_list) const override;

  const unsigned int n_components;
  const double present_time;
};

template <int dim>
AbstractBoundary<dim>::AbstractBoundary(const double present_time_inp,
                                        unsigned int n_components_inp)
    : Function<dim>(n_components_inp), present_time(present_time_inp),
      n_components(n_components_inp) {}

template <int dim>
void AbstractBoundary<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values[c] = value(p, c);
}

template <int dim>
void AbstractBoundary<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &value_list) const {
  const unsigned int n_points = points.size();

  AssertDimension(value_list.size(), n_points);

  for (unsigned int p = 0; p < n_points; ++p)
    vector_value(points[p], value_list[p]);
}

template <int dim>
class GeneralDirichletBoundary : public AbstractBoundary<dim> {
public:
  GeneralDirichletBoundary(double present_time_inp, double val,
                           unsigned int n_components_inp)
      : AbstractBoundary<dim>(present_time_inp, n_components_inp),
        constraint_value(val){};

  double value(const Point<dim> &p, unsigned int component) const override {
    return this->constraint_value;
  };

private:
  const double constraint_value;
};

template <int dim>
class VelocityBoundary : public GeneralDirichletBoundary<dim> {
public:
  VelocityBoundary(double present_time_inp, double velocity_inp,
                   unsigned int n_components_inp)
      : GeneralDirichletBoundary<dim>(present_time_inp,
                                      velocity_inp * present_time_inp,
                                      n_components_inp){};
};

template <int dim>
std::unique_ptr<Function<dim>> select_dirichlet_boundary(
    std::tuple<unsigned int, std::string, unsigned int, double> dirichlet_info,
    unsigned int n_components, double time) {
  std::string boundary_type = std::get<1>(dirichlet_info);
  double constraint_value = std::get<3>(dirichlet_info);
  if (boundary_type == "velocity") {
    return std::make_unique<VelocityBoundary<dim>>(time, constraint_value,
                                            n_components);
  } else if (boundary_type == "dirichlet") {
    return std::make_unique<GeneralDirichletBoundary<dim>>(time, constraint_value,
                                                    n_components);
  } else {
    AssertThrow(false, ExcNotImplemented());
  }
}

  };

private:
  const double velocity;
};

template <int dim>
DisplacementBoundary<dim>::DisplacementBoundary(const double present_time_inp,
                                                double velocity_inp)
    : AbstractBoundary<dim>(present_time_inp), velocity(velocity_inp) {}

#endif // CRACKS_DIRICHLET_BOUNDARY_H
