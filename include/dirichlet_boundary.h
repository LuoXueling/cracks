//
// Created by xlluo on 24-7-28.
//

#ifndef CRACKS_DIRICHLET_BOUNDARY_H
#define CRACKS_DIRICHLET_BOUNDARY_H

#endif // CRACKS_DIRICHLET_BOUNDARY_H

#include "dealii_includes.h"
#include "utils.h"

template <int dim> class IncrementalBoundaryValues : public Function<dim> {
public:
  IncrementalBoundaryValues(const double present_time_inp);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;

  virtual void
  vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>> &value_list) const override;

private:
  const unsigned int n_components;
  const double velocity;
  const double present_time;
};

template <int dim>
IncrementalBoundaryValues<dim>::IncrementalBoundaryValues(
    const double present_time_inp)
    : Function<dim>(dim), velocity(1.0),
      present_time(present_time_inp), n_components(dim) {}

template <int dim>
double
IncrementalBoundaryValues<dim>::value(const Point<dim> &p,
                                      const unsigned int component) const {
  Assert (component < this->n_components,
         ExcIndexRange (component, 0, this->n_components));

  Assert(dim==2, ExcNotImplemented());


  if (component == 1)  // u_y
  {
    if ((p(1) == 1.0) && (p(0) <= 1.0) && (p(0) >= 0.0)){
      return present_time *velocity;
    }
    else return 0.0;
  }

  return 0.0;
}

template <int dim>
void IncrementalBoundaryValues<dim>::vector_value(
    const Point<dim> &p, Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values[c] = IncrementalBoundaryValues<dim>::value(p, c);
}

template <int dim>
void IncrementalBoundaryValues<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &value_list) const {
  const unsigned int n_points = points.size();

  AssertDimension(value_list.size(), n_points);

  for (unsigned int p = 0; p < n_points; ++p)
    IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
}
