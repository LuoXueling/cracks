/**
 * Xueling Luo @ Shanghai Jiao Tong University, 2022
 * This code is for multiscale phase field fracture.
 **/

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "dealii_includes.h"
#include <ctime>

namespace Parameters {
struct Project {
  std::string mesh_from;
  std::string project_name;
  std::string load_sequence_from;
  bool enable_phase_field;

  static void declare_parameters(ParameterHandler &prm);

  void parse_parameters(ParameterHandler &prm);
};

void Project::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Project");
  {
    prm.declare_entry("Mesh from", "script",
                      Patterns::FileName(Patterns::FileName::FileType::input));
    prm.declare_entry("Project name", "Default project",
                      Patterns::FileName(Patterns::FileName::FileType::output));
    prm.declare_entry("Load sequence from", "script",
                      Patterns::FileName(Patterns::FileName::FileType::input));
    prm.declare_entry("Enable phase field", "true", Patterns::Bool());
  }
  prm.leave_subsection();
}

void Project::parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Project");
  {
    mesh_from = prm.get("Mesh from");
    project_name = prm.get("Project name");
    load_sequence_from = prm.get("Load sequence from");
    enable_phase_field = prm.get_bool("Enable phase field");
  }
  prm.leave_subsection();
}

struct Runtime {
  unsigned int max_load_step;
  unsigned int max_no_timesteps;
  double timestep;
  double timestep_size_2;
  unsigned int switch_timestep;
  bool direct_solver;
  double lower_bound_newton_residual;
  unsigned int max_no_newton_steps;
  double upper_newton_rho;
  unsigned int max_no_line_search_steps;
  double line_search_damping;
  std::string decompose_stress_rhs_u;
  std::string decompose_stress_matrix_u;
  std::string decompose_energy_phi;
  double constant_k;

  static void declare_parameters(ParameterHandler &prm);

  void parse_parameters(ParameterHandler &prm);
};

void Runtime::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Runtime");
  {
    prm.declare_entry("Max load step", "10", Patterns::Integer(1));
    prm.declare_entry("Max No of timesteps", "1", Patterns::Integer(0));

    prm.declare_entry("Timestep size", "1.0", Patterns::Double(0));

    prm.declare_entry("Timestep size to switch to", "1.0", Patterns::Double(0));

    prm.declare_entry("Switch timestep after steps", "0", Patterns::Integer(0));
    prm.declare_entry("Use Direct Inner Solver", "false", Patterns::Bool());

    prm.declare_entry("Newton lower bound", "1.0e-10", Patterns::Double(0));

    prm.declare_entry("Newton maximum steps", "10", Patterns::Integer(0));

    prm.declare_entry("Upper Newton rho", "0.999", Patterns::Double(0));

    prm.declare_entry("Line search maximum steps", "5", Patterns::Integer(0));

    prm.declare_entry("Line search damping", "0.5", Patterns::Double(0));

    prm.declare_entry("Decompose stress in rhs of displacement", "spectral",
                      Patterns::Selection("spectral|none"));

    prm.declare_entry("Decompose stress in matrix of displacement", "spectral",
                      Patterns::Selection("spectral|none"));

    prm.declare_entry("Decompose energy in phase field", "spectral",
                      Patterns::Selection("spectral|none"));

    prm.declare_entry("Constant small quantity k", "1.0e-6",
                      Patterns::Double(0));
  }
  prm.leave_subsection();
}

void Runtime::parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Runtime");
  {
    max_load_step = prm.get_integer("Max load step");
    max_no_timesteps = prm.get_integer("Max No of timesteps");
    timestep = prm.get_double("Timestep size");
    timestep_size_2 = prm.get_double("Timestep size to switch to");
    switch_timestep = prm.get_integer("Switch timestep after steps");

    direct_solver = prm.get_bool("Use Direct Inner Solver");

    // Newton tolerances and maximum steps
    lower_bound_newton_residual = prm.get_double("Newton lower bound");
    max_no_newton_steps = prm.get_integer("Newton maximum steps");

    // Criterion when time step should be cut
    // Higher number means: almost never
    // only used for simple penalization
    upper_newton_rho = prm.get_double("Upper Newton rho");

    // Line search control
    max_no_line_search_steps = prm.get_integer("Line search maximum steps");
    line_search_damping = prm.get_double("Line search damping");

    // Decompose stress in plus (tensile) and minus (compression)
    // 0.0: no decomposition, 1.0: with decomposition
    // Motivation see Miehe et al. (2010)
    decompose_stress_rhs_u = prm.get("Decompose stress in rhs of displacement");
    decompose_stress_matrix_u =
        prm.get("Decompose stress in matrix of displacement");

    decompose_energy_phi = prm.get("Decompose energy in phase field");

    constant_k = prm.get_double("Constant small quantity k");
  }
  prm.leave_subsection();
}

struct Material {
  double E;
  double v;
  double Gc;
  double l_phi;
  double lambda;
  double mu;
  double lame_coefficient_mu;
  double lame_coefficient_lambda;
  std::string plane_state;

  static void declare_parameters(ParameterHandler &prm);

  void parse_parameters(ParameterHandler &prm);
};

void Material::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Material");
  {
    prm.declare_entry("Young's modulus", "1000", Patterns::Double(0));
    prm.declare_entry("Poisson's ratio", "0.3", Patterns::Double(0, 0.5));
    prm.declare_entry("Critical energy release rate", "1", Patterns::Double(0));
    prm.declare_entry("Phase field length scale", "0.01", Patterns::Double(0));
    prm.declare_entry("Plane state", "stress", Patterns::Selection("stress|strain"));
  }
  prm.leave_subsection();
}

void Material::parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Material");
  {
    E = prm.get_double("Young's modulus");
    v = prm.get_double("Poisson's ratio");
    Gc = prm.get_double("Critical energy release rate");
    l_phi = prm.get_double("Phase field length scale");
  }
  prm.leave_subsection();
  lame_coefficient_mu = E / (2.0 * (1 + v));
  lame_coefficient_lambda = (2 * v * lame_coefficient_mu) / (1.0 - 2 * v);
}

struct FESystem {
  unsigned int dim;
  unsigned int poly_degree;
  unsigned int quad_order;
  bool refine;
  unsigned int n_global_pre_refine;
  unsigned int n_local_pre_refine;
  unsigned int n_refinement_cycles;
  double value_phase_field_for_refinement;

  static void declare_parameters(ParameterHandler &prm);

  void parse_parameters(ParameterHandler &prm);
};

void FESystem::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Finite element system");
  {
    prm.declare_entry("Physical dimension", "2", Patterns::Integer(0),
                      "Physical dimension");
    prm.declare_entry("Polynomial degree", "2", Patterns::Integer(0),
                      "Displacement system polynomial order");

    prm.declare_entry("Quadrature order", "3", Patterns::Integer(0),
                      "Gauss quadrature order");
    prm.declare_entry("Refine", "false", Patterns::Bool());
    prm.declare_entry("Global pre-refinement steps", "1", Patterns::Integer(0));

    prm.declare_entry("Local pre-refinement steps", "0", Patterns::Integer(0));

    prm.declare_entry("Adaptive refinement cycles", "0", Patterns::Integer(0));
    prm.declare_entry("Value of phase field for refinement", "0.0",
                      Patterns::Double(0));
  }
  prm.leave_subsection();
}

void FESystem::parse_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Finite element system");
  {
    dim = prm.get_integer("Physical dimension");
    poly_degree = prm.get_integer("Polynomial degree");
    quad_order = prm.get_integer("Quadrature order");
    refine = prm.get_bool("Refine");
    n_global_pre_refine = prm.get_integer("Global pre-refinement steps");
    n_local_pre_refine = prm.get_integer("Local pre-refinement steps");
    n_refinement_cycles = prm.get_integer("Adaptive refinement cycles");
    value_phase_field_for_refinement =
        prm.get_double("Value of phase field for refinement");
  }
  prm.leave_subsection();
}

struct AllParameters : public FESystem,
                       public Project,
                       public Runtime,
                       public Material {
  AllParameters(){};
  AllParameters(const std::string &input_file);

  static void declare_parameters(ParameterHandler &prm);

  void set_parameters(const std::string &input_file);
  void parse_parameters(ParameterHandler &prm);

  std::string output_dir;
  std::string param_dir;
  std::string log_file;
};

AllParameters::AllParameters(const std::string &input_file) {
  set_parameters(input_file);
}

void AllParameters::set_parameters(const std::string &input_file) {
  ParameterHandler prm;
  declare_parameters(prm);
  prm.parse_input(input_file);
  parse_parameters(prm);

  // set output directory

  std::time_t currenttime = std::time(0);
  char tAll[255];
  std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S",
                std::localtime(&currenttime));
  std::string stime;
  std::stringstream strtime;
  strtime << tAll;
  stime = strtime.str();

  output_dir = "./output/" + this->project_name + "-" + stime + "/";

  param_dir = input_file;

  log_file = output_dir + "log.txt";
}

void AllParameters::declare_parameters(ParameterHandler &prm) {
  FESystem::declare_parameters(prm);
  Project::declare_parameters(prm);
  Runtime::declare_parameters(prm);
  Material::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm) {
  FESystem::parse_parameters(prm);
  Project::parse_parameters(prm);
  Runtime::parse_parameters(prm);
  Material::parse_parameters(prm);
}
} // namespace Parameters

#endif