//
// Created by xlluo on 24-7-31.
//

#ifndef CRACKS_SOLUTION_COMMUNICATOR_H
#define CRACKS_SOLUTION_COMMUNICATOR_H

#include "dealii_includes.h"
using namespace dealii;

template <int dim> class SolutionCommunicator {
public:
  SolutionCommunicator();
  void register_solution(std::string name, LA::MPI::Vector *solution);
  LA::MPI::Vector get(std::string name);

private:
  std::map<std::string, LA::MPI::Vector*> solution_dict;
};

template <int dim> SolutionCommunicator<dim>::SolutionCommunicator() {}

template <int dim>
void SolutionCommunicator<dim>::register_solution(std::string name,
                                                  LA::MPI::Vector *solution) {
  solution_dict[name] = solution;
}

template <int dim>
LA::MPI::Vector SolutionCommunicator<dim>::get(std::string name) {
  return *solution_dict[name];
}

#endif // CRACKS_SOLUTION_COMMUNICATOR_H
