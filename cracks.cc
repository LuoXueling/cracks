/**
  This code is licensed under the "GNU GPL version 2 or later". See
  LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html

  Copyright 2013-2020: Thomas Wick and Timo Heister

  Rearranged by Xueling Luo
*/

// Main features of this crack phase-field program
// -----------------------------------------------
// 1. Quasi-monolithic formulation for the displacement-phase-field system
// 2. Primal dual active set strategy to treat crack irreversibility
// 3. Predictor-corrector mesh adaptivity
// 4. Parallel computing using MPI, p4est, and trilinos

// This makes IDEs like QtCreator happy (note that this is defined in cmake):
#ifndef SOURCE_DIR
#define SOURCE_DIR ""
#endif

#define CATCH_CONFIG_RUNNER

#include "dealii_includes.h"
#include "contrib/catch.hpp"
#include "pfm.h"
#include <sstream>
#include <sys/stat.h> // for mkdir

using namespace dealii;

// The main function looks almost the same
// as in all other deal.II tuturial steps.
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  if (argc == 1) // run unit tests
  {
    int ret = Catch::Session().run(argc, argv);
    if (ret != 0)
      return ret;
  }

  try {
    deallog.depth_console(0);

    ParameterHandler prm;
    FracturePhaseFieldProblem<2>::declare_parameters(prm);
    if (argc > 1) {
      prm.parse_input(argv[1]);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        // generate parameters.prm in the output directory:
        prm.enter_subsection("Global parameters");
        const std::string output_folder = prm.get("Output directory");
        prm.leave_subsection();

        // create output folder (only on rank 0) if needed
        {
          const mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
          int mkdir_return_value = mkdir(output_folder.c_str(), mode);

          if (0 != mkdir_return_value && errno != EEXIST) {
            AssertThrow(false, ExcMessage("Can not create output directory"));
          }
        }

        std::ofstream out((output_folder + "/parameters.prm").c_str());
        prm.print_parameters(out, ParameterHandler::Text);
      }

      // make sure the directory is created before anyone continues
      MPI_Barrier(MPI_COMM_WORLD);
    } else {
      std::ofstream out("default.prm");
      prm.print_parameters(out, ParameterHandler::Text);
      std::cout << "usage: ./cracks <parameter_file>" << std::endl
                << " (created default.prm)" << std::endl;
      return 0;
    }

    prm.enter_subsection("Global parameters");
    unsigned int problem_dimension = prm.get_integer("Dimension");
    prm.leave_subsection();

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Problem dimension: " << problem_dimension << std::endl;

    if (problem_dimension == 2) {
      FracturePhaseFieldProblem<2> fracture_problem(prm);
      fracture_problem.run();
    } else if (problem_dimension == 3) {
      FracturePhaseFieldProblem<3> fracture_problem(prm);
      fracture_problem.run();
    } else
      AssertThrow(false, ExcNotImplemented());

  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
