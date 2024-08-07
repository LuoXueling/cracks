//
// Created by xlluo on 24-7-30.
//

#ifndef CRACKS_CONTROLLER_H
#define CRACKS_CONTROLLER_H

#include "dealii_includes.h"
#include "parameters.h"
#include "utils.h"

class PointHistory : public TransferableQuadraturePointData {
public:
  void update(std::string name, double solution,
              std::string scheme = "latest") {
    solution_dict_temp[name] = solution;
    finalize_scheme[name] = scheme;
  };
  double get(std::string name, double default_value = 0.0) const {
    // This function has to be const for pack_values so we cannot use
    // solution_dict[name]
    try {
      auto pos = solution_dict.find(name);
      if (pos == solution_dict.end()) {
        return default_value;
      } else
        return pos->second;
    } catch (...) {
      return default_value;
    }
  };
  double get_increment(std::string name, double default_value = 0.0) const {
    // This function has to be const for pack_values so we cannot use
    // solution_dict[name]
    try {
      auto pos = solution_increment.find(name);
      if (pos == solution_increment.end()) {
        return default_value;
      } else
        return pos->second;
    } catch (...) {
      return default_value;
    }
  };
  void finalize() {
    typename std::map<std::string, double>::iterator it;
    for (it = solution_dict_temp.begin(); it != solution_dict_temp.end();
         it++) {
      std::string scheme = finalize_scheme[it->first];
      double res;
      if (scheme == "latest") {
        res = it->second;
      } else if (scheme == "max") {
        res = std::max(it->second, get(it->first, -1.0e10));
        solution_dict_temp[it->first] = -1.0e10;
      } else if (scheme == "min") {
        res = std::min(it->second, get(it->first, 1.0e10));
        solution_dict_temp[it->first] = 1.0e10;
      } else if (scheme == "accumulate") {
        res = it->second + get(it->first, 0.0);
        solution_dict_temp[it->first] = 0.0;
      } else if (scheme == "multiplicative") {
        res = it->second * get(it->first, 1.0);
        solution_dict_temp[it->first] = 1.0;
      } else {
        AssertThrow(false, ExcNotImplemented(
                               "Point history update scheme is illegal."));
      }
      solution_increment[it->first] = res - solution_dict[it->first];
      solution_dict[it->first] = res;
    }
  }

  unsigned int number_of_values() const override {
    return finalize_scheme.size() * 2;
  }

  void pack_values(std::vector<double> &values) const override {
    Assert(values.size() == finalize_scheme.size() * 2, ExcInternalError());
    std::vector<std::string> names = get_names();
    for (unsigned int i = 0; i < finalize_scheme.size() * 2; ++i) {
      values[i] = i < finalize_scheme.size()
                      ? get(names[i], 0.0)
                      : get_increment(names[i - finalize_scheme.size()], 0.0);
    }
  }

  void unpack_values(const std::vector<double> &values) override {
    Assert(values.size() == finalize_scheme.size() * 2, ExcInternalError());
    std::vector<std::string> names = get_names();
    for (unsigned int i = 0; i < finalize_scheme.size() * 2; ++i) {
      if (i < finalize_scheme.size())
        solution_dict[names[i]] = values[i];
      else
        solution_increment[names[i - finalize_scheme.size()]] = values[i];
    }
  }

  std::vector<std::string> get_names() const {
    std::vector<std::string> names;
    for (std::map<std::string, std::string>::iterator it =
             finalize_scheme.begin();
         it != finalize_scheme.end(); ++it) {
      names.push_back(it->first);
    }
    return names;
  }

private:
  std::map<std::string, double> solution_dict_temp;
  std::map<std::string, double> solution_dict;
  std::map<std::string, double> solution_increment;

  inline static std::map<std::string, std::string> finalize_scheme;
};

template <int dim> class Controller {
public:
  explicit Controller(Parameters::AllParameters &prms);

  void finalize_point_history();
  void initialize_point_history();

  MPI_Comm mpi_com;

  parallel::distributed::Triangulation<dim> triangulation;
  QGauss<dim> quadrature_formula;
  Parameters::AllParameters params;

  ConditionalOStream dcout;
  DebugConditionalOStream debug_dcout;
  TimerOutput timer;
  TimerOutput computing_timer;
  std::ofstream fout;
  teebuf sbuf;
  std::ostream pout;

  double time;
  unsigned int timestep_number;
  double current_timestep;
  double old_timestep;

  TableHandler statistics;

  std::vector<int> boundary_ids;

  CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory>
      quadrature_point_history, old_quadrature_point_history;
};

template <int dim>
Controller<dim>::Controller(Parameters::AllParameters &prms)
    : mpi_com(MPI_COMM_WORLD), params(prms),
      triangulation(mpi_com, typename Triangulation<dim>::MeshSmoothing(
                                 Triangulation<dim>::smoothing_on_refinement |
                                 Triangulation<dim>::smoothing_on_coarsening)),
      quadrature_formula(prms.poly_degree + 1),
      fout(prms.output_dir + "log.txt"), sbuf(fout.rdbuf(), std::cout.rdbuf()),
      pout(&sbuf),
      dcout(pout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      debug_dcout(std::cout, &mpi_com, prms.debug_output),
      timer(mpi_com, dcout, TimerOutput::never,
            TimerOutput::cpu_and_wall_times),
      computing_timer(mpi_com, dcout, TimerOutput::never,
                      TimerOutput::wall_times),
      time(0), timestep_number(0), current_timestep(0), old_timestep(0) {
  statistics.set_auto_fill_mode(true);
}

template <int dim> void Controller<dim>::initialize_point_history() {
  // The original CellDataStorage.initialize use tria.begin_active() and
  // tria.end() and does not really loop over locally-owned cells
  // https://github.com/rezarastak/dealii/blob/381a8d3739e10a450b7efeb62fd2f74add7ee19c/tests/base/quadrature_point_data_04.cc#L101
  for (auto cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned()) {
      quadrature_point_history.template initialize<PointHistory>(
          cell, quadrature_formula.size());
      old_quadrature_point_history.template initialize<PointHistory>(
          cell, quadrature_formula.size());
    }
}

template <int dim> void Controller<dim>::finalize_point_history() {
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
