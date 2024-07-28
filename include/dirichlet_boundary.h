//
// Created by xlluo on 24-7-28.
//

#ifndef CRACKS_DIRICHLET_BOUNDARY_H
#define CRACKS_DIRICHLET_BOUNDARY_H

#endif // CRACKS_DIRICHLET_BOUNDARY_H

#include "dealii_includes.h"

template <int dim>
class BoundaryTensionTest : public Function<dim>
{
public:
  BoundaryTensionTest (const unsigned int n_components, const double time)
      : Function<dim>(n_components),
        n_components (n_components),
        _time (time)
  {}

  virtual double value (const Point<dim>   &p,
                       const unsigned int  component = 0) const {
    Assert (component < this->n_components,
           ExcIndexRange (component, 0, this->n_components));

    Assert(dim==2, ExcNotImplemented());

    double dis_step_per_timestep = 1.0;

    if (component == 1)  // u_y
    {
      return ( ((p(1) == 1.0) && (p(0) <= 1.0) && (p(0) >= 0.0))
                  ?
                  (1.0) * _time *dis_step_per_timestep : 0 );

    }
    return 0;
  };

  virtual void vector_value (const Point<dim> &p,
                            Vector<double>   &value) const {
    for (unsigned int c=0; c<this->n_components; ++c)
      value (c) = BoundaryTensionTest<dim>::value (p, c);
  };

private:
  const unsigned int n_components;
  double _time;
};


// Dirichlet boundary conditions for
// Miehe's et al. shear test 2010
// Example 2b
template <int dim>
class BoundaryShearTest : public Function<dim>
{
public:
  BoundaryShearTest (const unsigned int n_components, const double time)
      : Function<dim>(n_components),
        _time (time)
  {}

  virtual double value (const Point<dim>   &p,
                       const unsigned int  component = 0) const {
    Assert (component < this->n_components,
           ExcIndexRange (component, 0, this->n_components));


    double dis_step_per_timestep = -1.0;

    if (component == 0)
    {
      return ( ((p(1) == 1.0) )
                  ?
                  (1.0) * _time *dis_step_per_timestep : 0 );

    }


    return 0;
  };

  virtual void vector_value (const Point<dim> &p,
                            Vector<double>   &value) const{
    for (unsigned int c=0; c<this->n_components; ++c)
      value (c) = BoundaryShearTest<dim>::value (p, c);
  };

private:
  double _time;

};


template <int dim>
class InitialValuesTensionOrShear : public Function<dim>
{
public:
  InitialValuesTensionOrShear (const unsigned int n_components,
                              const double min_cell_diameter)
      :
        Function<dim> (n_components),
        n_components (n_components),
        _min_cell_diameter (min_cell_diameter)
  {}

  virtual double
  value (
      const Point<dim> &p, const unsigned int component = 0) const;

  virtual void
  vector_value (
      const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
  double _min_cell_diameter;
};
