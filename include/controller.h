//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_CONTROLLER_H
#define CRACKS_CONTROLLER_H

#include "dealii_includes.h"
#include "parameters.h"
#include "utils.h"

class PointHistory {
public:
  void update(std::string name, double solution,
              std::string scheme = "latest") {
    solution_dict_temp[name] = solution;
    finalize_scheme[name] = scheme;
  };
  double get(std::string name, double default_value = 0.0) {
    try {
      return solution_dict[name];
    } catch (...) {
      return default_value;
    }
  };
  void finalize() {
    typename std::map<std::string, double>::iterator it;
    for (it = solution_dict_temp.begin(); it != solution_dict_temp.end();
         it++) {
      std::string scheme = finalize_scheme[it->first];
      if (scheme == "latest") {
        solution_dict[it->first] = it->second;
      } else if (scheme == "max") {
        solution_dict[it->first] =
            std::max(it->second, get(it->first, -1.0e10));
      } else if (scheme == "min") {
        solution_dict[it->first] = std::min(it->second, get(it->first, 1.0e10));
      } else if (scheme == "accumulate") {
        solution_dict[it->first] = it->second + get(it->first, 0.0);
      } else if (scheme == "multiplicative") {
        solution_dict[it->first] = it->second * get(it->first, 1.0);
      } else {
        AssertThrow(false, ExcNotImplemented(
                               "Point history update scheme is illegal."));
      }
    }
  }

private:
  std::map<std::string, double> solution_dict_temp;
  std::map<std::string, double> solution_dict;

  std::map<std::string, std::string> finalize_scheme;
};

template <int dim> class Controller {
public:
  explicit Controller(Parameters::AllParameters &prms);

  void finalize();

  MPI_Comm mpi_com;

  parallel::distributed::Triangulation<dim> triangulation;
  QGauss<dim> quadrature_formula;
  Parameters::AllParameters params;

  ConditionalOStream dcout;
  ConditionalOStream debug_dcout;
  TimerOutput timer;
  TimerOutput computing_timer;

  double time;
  unsigned int timestep_number;
  double current_timestep;
  double old_timestep;

  TableHandler statistics;

  std::vector<int> boundary_ids;

  CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
      quadrature_point_history;
};

template <int dim>
Controller<dim>::Controller(Parameters::AllParameters &prms)
    : mpi_com(MPI_COMM_WORLD), params(prms),
      triangulation(mpi_com, typename Triangulation<dim>::MeshSmoothing(
                                 Triangulation<dim>::smoothing_on_refinement |
                                 Triangulation<dim>::smoothing_on_coarsening)),
      quadrature_formula(prms.poly_degree + 1),
      dcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      debug_dcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_com) == 0) &&
                                 prms.debug_output),
      timer(mpi_com, dcout, TimerOutput::never,
            TimerOutput::cpu_and_wall_times),
      computing_timer(mpi_com, dcout, TimerOutput::never,
                      TimerOutput::wall_times),
      time(0), timestep_number(0), current_timestep(0), old_timestep(0) {
  statistics.set_auto_fill_mode(true);
}

template <int dim> void Controller<dim>::finalize() {
  /*
   * Finalize point history
   */
  const unsigned int n_q_points = quadrature_formula.size();
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned()) {
      const std::vector<std::shared_ptr<PointHistory>> lqph =
          quadrature_point_history.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) {
        lqph[q]->finalize();
      }
    }
}

#endif // CRACKS_CONTROLLER_H
