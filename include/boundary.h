//
// Created by xlluo on 2022/3/26.
//

#ifndef CRACKS_BOUNDARY_H
#define CRACKS_BOUNDARY_H

#include "dealii_includes.h"
using namespace dealii;
// Several classes for Dirichlet boundary conditions
// for displacements for the single-edge notched test (for phase-field see Miehe
// et al. 2010) Example 2a (tension test) Example 2b (shear test; see below)
template <int dim> class BoundaryTensionTest : public Function<dim> {
public:
  BoundaryTensionTest(const unsigned int n_components, const double time)
      : Function<dim>(n_components), n_components(n_components), _time(time) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
  double _time;
};

template <int dim>
double BoundaryTensionTest<dim>::value(const Point<dim> &p,
                                       const unsigned int component) const {
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));

  Assert(dim == 2, ExcNotImplemented());

  double dis_step_per_timestep = 1.0;

  if (component == 1) // u_y
  {
    return (((p(1) == 1.0) && (p(0) <= 1.0) && (p(0) >= 0.0))
                ? (1.0) * _time * dis_step_per_timestep
                : 0);
  }

  return 0;
}

template <int dim>
void BoundaryTensionTest<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryTensionTest<dim>::value(p, c);
}

// Dirichlet boundary conditions for
// Miehe's et al. shear test 2010
// Example 2b
template <int dim> class BoundaryShearTest : public Function<dim> {
public:
  BoundaryShearTest(const unsigned int n_components, const double time)
      : Function<dim>(n_components), _time(time) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  double _time;
};

template <int dim>
double BoundaryShearTest<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const {
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));

  double dis_step_per_timestep = -1.0;

  if (component == 0) {
    return (((p(1) == 1.0)) ? (1.0) * _time * dis_step_per_timestep : 0);
  }

  return 0;
}

template <int dim>
void BoundaryShearTest<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryShearTest<dim>::value(p, c);
}

template <int dim> class BoundaryThreePoint : public Function<dim> {
public:
  BoundaryThreePoint(const unsigned int n_components, const double time)
      : Function<dim>(n_components), _time(time) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  double _time;
};

// The boundary values are given to component
// with number 0.
template <int dim>
double BoundaryThreePoint<dim>::value(const Point<dim> & /*p*/,
                                      const unsigned int component) const {
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));

  double dis_step_per_timestep = -1.0;

  if (component == 1) {
    return 1.0 * _time * dis_step_per_timestep;
  }

  return 0;
}

template <int dim>
void BoundaryThreePoint<dim>::vector_value(const Point<dim> &p,
                                           Vector<double> &values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryThreePoint<dim>::value(p, c);
}

#endif // CRACKS_BOUNDARY_H
