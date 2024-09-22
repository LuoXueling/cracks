//
// Created by xlluo on 24-8-17.
//

#ifndef CRACKS_GLOBAL_ESTIMATOR_H
#define CRACKS_GLOBAL_ESTIMATOR_H

#include "controller.h"
#include "dealii_includes.h"

namespace GlobalEstimator {
template <int dim>
double sum(std::string name, double default_value, Controller<dim> &ctl) {
  DoFHandler<dim> dof_handler(ctl.triangulation);
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  double local_integration = 0.0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        local_integration += lqph[q]->get_either_latest(name, default_value);
      }
    }
  }
  double global_integration =
      Utilities::MPI::sum(local_integration, ctl.mpi_com);
  return global_integration;
}

template <int dim>
double abssum(std::string name, double default_value, Controller<dim> &ctl) {
  DoFHandler<dim> dof_handler(ctl.triangulation);
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  double local_integration = 0.0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        local_integration +=
            std::abs(lqph[q]->get_either_latest(name, default_value));
      }
    }
  }
  double global_integration =
      Utilities::MPI::sum(local_integration, ctl.mpi_com);
  return global_integration;
}

template <int dim>
double max(std::string name, double default_value, Controller<dim> &ctl) {
  DoFHandler<dim> dof_handler(ctl.triangulation);
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  double local_max = 0.0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        local_max = std::max(lqph[q]->get_either_latest(name, default_value),
                             local_max);
      }
    }
  }
  double global_max = Utilities::MPI::max(local_max, ctl.mpi_com);
  return global_max;
}

template <int dim>
double absmax(std::string name, double default_value, Controller<dim> &ctl) {
  DoFHandler<dim> dof_handler(ctl.triangulation);
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  double local_max = 0.0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        local_max =
            std::max(std::abs(lqph[q]->get_either_latest(name, default_value)),
                     local_max);
      }
    }
  }
  double global_max = Utilities::MPI::max(local_max, ctl.mpi_com);
  return global_max;
}

template <int dim>
double min(std::string name, double default_value, Controller<dim> &ctl) {
  DoFHandler<dim> dof_handler(ctl.triangulation);
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  double local_min = 1e10;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        local_min = std::min(lqph[q]->get_either_latest(name, default_value),
                             local_min);
      }
    }
  }
  double global_min = Utilities::MPI::min(local_min, ctl.mpi_com);
  return global_min;
}

template <int dim>
double absmin(std::string name, double default_value, Controller<dim> &ctl) {
  DoFHandler<dim> dof_handler(ctl.triangulation);
  const unsigned int n_q_points = ctl.quadrature_formula.size();

  double local_min = 1e10;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          ctl.quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        local_min =
            std::min(std::abs(lqph[q]->get_either_latest(name, default_value)),
                     local_min);
      }
    }
  }
  double global_min = Utilities::MPI::min(local_min, ctl.mpi_com);
  return global_min;
}

}; // namespace GlobalEstimator

#endif // CRACKS_GLOBAL_ESTIMATOR_H
