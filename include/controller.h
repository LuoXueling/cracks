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
  void update(std::string name, double solution) {
    solution_dict[name] = solution;
  };
  double get(std::string name) { return solution_dict[name]; };

private:
  std::map<std::string, double> solution_dict;
};

template <int dim> class Controller {
public:
  explicit Controller(Parameters::AllParameters &prms);

  MPI_Comm mpi_com;

  parallel::distributed::Triangulation<dim> triangulation;
  QGauss<dim> quadrature_formula;
  Parameters::AllParameters params;

  ConditionalOStream pcout;
  DualOStream dcout;
  DualTimerOutput timer;
  TimerOutput computing_timer;

  double time;
  unsigned int timestep_number;
  double current_timestep;
  double old_timestep;

  TableHandler statistics;

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
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      dcout(pcout, params.log_file), timer(mpi_com, pcout, TimerOutput::never,
                                           TimerOutput::cpu_and_wall_times),
      computing_timer(mpi_com, pcout, TimerOutput::never,
                      TimerOutput::wall_times),
      time(0), timestep_number(0), current_timestep(0), old_timestep(0) {
  statistics.set_auto_fill_mode(true);
}

#endif // CRACKS_CONTROLLER_H
