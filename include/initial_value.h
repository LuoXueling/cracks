//
// Created by xlluo on 2022/3/26.
//

#ifndef CRACKS_INITIAL_VALUE_H
#define CRACKS_INITIAL_VALUE_H

#include "dealii_includes.h"
using namespace dealii;

// Several classes for initial (phase-field) values
// Here, we prescribe initial (multiple) cracks
template <int dim> class InitialValuesSneddon : public Function<dim> {
public:
  InitialValuesSneddon(const unsigned int n_components,
                       const double min_cell_diameter)
      : Function<dim>(n_components), n_components(n_components),
        _min_cell_diameter(min_cell_diameter) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
  double _min_cell_diameter;
};

template <int dim>
double InitialValuesSneddon<dim>::value(const Point<dim> &p,
                                        const unsigned int component) const {
  // impose crack [-1,1]x[-h,h]

  double l_0 = 1.0;
  double thickness = 2.0 * _min_cell_diameter;
  double r_squared;
  if (dim == 2)
    r_squared = p(0) * p(0);
  else
    r_squared = p(0) * p(0) + p(2) * p(2);

  if (component == dim) {
    if ((r_squared <= l_0 * l_0) && (abs(2.0 * p(1)) <= thickness))
      return 0.0;
    else
      return 1.0;
  } else
    return 0.0;
}

template <int dim>
void InitialValuesSneddon<dim>::vector_value(const Point<dim> &p,
                                             Vector<double> &values) const {
  for (unsigned int comp = 0; comp < this->n_components; ++comp)
    values(comp) = InitialValuesSneddon<dim>::value(p, comp);
}

template <int dim> class ExactPhiSneddon : public Function<dim> {
public:
  ExactPhiSneddon(const int n_components, const double eps_)
      : Function<dim>(n_components), eps(eps_) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const {
    (void)component;

    double l_0 = 1.0;
    Point<dim> left;
    left(0) = -l_0;
    Point<dim> right;
    right(0) = l_0;

    double dist;

    if (p(0) < left(0))
      dist = left.distance(p);
    else if (p(0) > right(0))
      dist = right.distance(p);
    else
      dist = (dim == 2) ? (std::sqrt(p(1) * p(1)))
                        : (std::sqrt(p(1) * p(1) + p(2) * p(2)));

    return 1.0 - std::exp(-dist / eps);
  }

private:
  double eps;
};

template <int dim>
class SneddonExactPostProc : public DataPostprocessorScalar<dim> {
public:
  SneddonExactPostProc(const unsigned int n_components, const double eps)
      : DataPostprocessorScalar<dim>("exact_phi", update_quadrature_points),
        exact(n_components, eps) {}

  void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const {
    for (unsigned int i = 0; i < computed_quantities.size(); ++i)
      computed_quantities[i][0] = exact.value(input_data.evaluation_points[i]);
  }

private:
  ExactPhiSneddon<dim> exact;
};

// Class for initial values multiple fractures in a heterogeneous material
template <int dim> class InitialValuesMultipleHet : public Function<dim> {
public:
  InitialValuesMultipleHet(const unsigned int n_components,
                           const double min_cell_diameter)
      : Function<dim>(n_components), n_components(n_components),
        _min_cell_diameter(min_cell_diameter) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
  double _min_cell_diameter;
};

template <int dim>
double
InitialValuesMultipleHet<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const {
  double width = _min_cell_diameter;
  double height = _min_cell_diameter;
  // Defining the initial crack(s)
  // 0 = crack
  // 1 = no crack
  bool example_3 = true;
  if (component == n_components - 1) {
    if (dim == 3) {
      if (((p(0) >= 2.6 - width / 2.0) && (p(0) <= 2.6 + width / 2.0)) &&
          ((p(1) >= 3.8 - width / 2.0) && (p(1) <= 5.5 + width / 2.0)) &&
          (p(2) >= 4.0 - width / 2.0) && (p(2) <= 4.0 + width / 2.0))
        return 0.0;
      else if (((p(0) >= 5.5 - width / 2.0) && (p(0) <= 7.0 + width / 2.0)) &&
               ((p(1) >= 4.0 - width / 2.0) && (p(1) <= 4.0 + width / 2.0)) &&
               (p(2) >= 6.0 - width / 2.0) && (p(2) <= 6.0 + width / 2.0))
        return 0.0;
      else
        return 1.0;
    } else if (example_3) {
      // Example 3 of our paper
      if (((p(0) >= 2.5 - width / 2.0) && (p(0) <= 2.5 + width / 2.0)) &&
          ((p(1) >= 0.8) && (p(1) <= 1.5)))
        return 0.0;
      else if (((p(0) >= 0.5) && (p(0) <= 1.5)) &&
               ((p(1) >= 3.0 - height / 2.0) && (p(1) <= 3.0 + height / 2.0)))
        return 0.0;
      else
        return 1.0;
    } else {
      // Two parallel fractures
      if (((p(0) >= 1.6 - width) && (p(0) <= 2.4 + width)) &&
          ((p(1) >= 2.75 - height) && (p(1) <= 2.75 + height)))
        return 0.0;
      else if (((p(0) >= 1.6 - width) && (p(0) <= 2.4 + width)) &&
               ((p(1) >= 1.25 - height) && (p(1) <= 1.25 + height)))
        return 0.0;
      else
        return 1.0;
    }
  }

  return 0.0;
}

template <int dim>
void InitialValuesMultipleHet<dim>::vector_value(const Point<dim> &p,
                                                 Vector<double> &values) const {
  for (unsigned int comp = 0; comp < this->n_components; ++comp)
    values(comp) = InitialValuesMultipleHet<dim>::value(p, comp);
}

template <int dim> class InitialValuesTensionOrShear : public Function<dim> {
public:
  InitialValuesTensionOrShear(const unsigned int n_components,
                              const double min_cell_diameter)
      : Function<dim>(n_components), n_components(n_components),
        _min_cell_diameter(min_cell_diameter) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
  double _min_cell_diameter;
};

template <int dim>
double
InitialValuesTensionOrShear<dim>::value(const Point<dim> & /*p*/,
                                        const unsigned int component) const {
  // Defining the initial crack(s)
  // 0 = crack
  // 1 = no crack
  if (component == n_components - 1) {
    return 1.0;
  }

  return 0.0;
}

template <int dim>
void InitialValuesTensionOrShear<dim>::vector_value(
    const Point<dim> &p, Vector<double> &values) const {
  for (unsigned int comp = 0; comp < this->n_components; ++comp)
    values(comp) = InitialValuesTensionOrShear<dim>::value(p, comp);
}

template <int dim> class InitialValuesNoCrack : public Function<dim> {
public:
  InitialValuesNoCrack(const unsigned int n_components)
      : Function<dim>(n_components), n_components(n_components) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
};

template <int dim>
double InitialValuesNoCrack<dim>::value(const Point<dim> & /*p*/,
                                        const unsigned int component) const {
  if (component == n_components - 1) {
    return 1.0;
  }
  return 0.0;
}

template <int dim>
void InitialValuesNoCrack<dim>::vector_value(const Point<dim> &p,
                                             Vector<double> &values) const {
  for (unsigned int comp = 0; comp < this->n_components; ++comp)
    values(comp) = InitialValuesNoCrack<dim>::value(p, comp);
}

// Class for initial values multiple fractures in a homogeneous material
template <int dim> class InitialValuesMultipleHomo : public Function<dim> {
public:
  InitialValuesMultipleHomo(const unsigned int n_components,
                            const double min_cell_diameter)
      : Function<dim>(n_components), n_components(n_components),
        _min_cell_diameter(min_cell_diameter) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  const unsigned int n_components;
  double _min_cell_diameter;
};

template <int dim>
double
InitialValuesMultipleHomo<dim>::value(const Point<dim> &p,
                                      const unsigned int component) const {
  double width = _min_cell_diameter;
  double height = _min_cell_diameter;
  // Defining the initial crack(s)
  // 0 = crack
  // 1 = no crack
  bool example_3 = true;
  if (component == n_components - 1) {
    if (example_3) {
      // Example 3 of our paper
      if (((p(0) >= 2.5 - width / 2.0) && (p(0) <= 2.5 + width / 2.0)) &&
          ((p(1) >= 0.8) && (p(1) <= 1.5)))
        return 0.0;
      else if (((p(0) >= 0.5) && (p(0) <= 1.5)) &&
               ((p(1) >= 3.0 - height / 2.0) && (p(1) <= 3.0 + height / 2.0)))
        return 0.0;
      else
        return 1.0;
    } else {
      // Two parallel fractures
      if (((p(0) >= 1.6 - width) && (p(0) <= 2.4 + width)) &&
          ((p(1) >= 2.75 - height) && (p(1) <= 2.75 + height)))
        return 0.0;
      else if (((p(0) >= 1.6 - width) && (p(0) <= 2.4 + width)) &&
               ((p(1) >= 1.25 - height) && (p(1) <= 1.25 + height)))
        return 0.0;
      else
        return 1.0;
    }
  }

  return 0.0;
}

template <int dim>
void InitialValuesMultipleHomo<dim>::vector_value(
    const Point<dim> &p, Vector<double> &values) const {
  for (unsigned int comp = 0; comp < this->n_components; ++comp)
    values(comp) = InitialValuesMultipleHomo<dim>::value(p, comp);
}

#endif // CRACKS_INITIAL_VALUE_H
