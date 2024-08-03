//
// Created by xlluo on 24-7-28.
//

#ifndef CRACKS_BOUNDARY_H
#define CRACKS_BOUNDARY_H

#include "dealii_includes.h"

// Can only use apply AbstractBoundary condition on one dof
template <int dim> class AbstractBoundary : public Function<dim> {
public:
  explicit AbstractBoundary(double present_time_inp);

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
AbstractBoundary<dim>::AbstractBoundary(const double present_time_inp)
    : Function<dim>(dim), present_time(present_time_inp), n_components(dim) {}

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
  GeneralDirichletBoundary(double present_time_inp, double val)
      : AbstractBoundary<dim>(present_time_inp), constraint_value(val){};

  double value(const Point<dim> &p, unsigned int component) const override {
    return this->constraint_value;
  };

private:
  const double constraint_value;
};

template <int dim>
class VelocityBoundary : public GeneralDirichletBoundary<dim> {
public:
  VelocityBoundary(double present_time_inp, double velocity_inp)
      : GeneralDirichletBoundary<dim>(present_time_inp,
                                      velocity_inp * present_time_inp){};
};

  };

private:
  const double velocity;
};

template <int dim>
DisplacementBoundary<dim>::DisplacementBoundary(const double present_time_inp,
                                                double velocity_inp)
    : AbstractBoundary<dim>(present_time_inp), velocity(velocity_inp) {}

#endif // CRACKS_DIRICHLET_BOUNDARY_H
