//
// Created by xlluo on 2022/3/26.
//

#ifndef CRACKS_PFM_H
#define CRACKS_PFM_H

#include "bitmap.h"
#include "boundary.h"
#include "dealii_includes.h"
#include "initial_value.h"
#include "introspection.h"
#include "support_functions.h"
using namespace dealii;

template <int dim> class FracturePhaseFieldProblem {
public:
  FracturePhaseFieldProblem(ParameterHandler &);

  void run();

  static void declare_parameters(ParameterHandler &prm);

private:
  void set_runtime_parameters();

  void determine_mesh_dependent_parameters();

  void setup_mesh();

  void setup_system();

  void assemble_system(bool residual_only = false);

  void assemble_nl_residual();

  void assemble_diag_mass_matrix();

  void set_boundary_conditions(const double time, const bool initial_step,
                               ConstraintMatrix &constraints);

  void set_initial_bc(const double time);

  void set_newton_bc();

  unsigned int solve();

  double newton_active_set();

  double newton_iteration(const double time);

  double compute_point_value(const DoFHandler<dim> &dofh,
                             const LA::MPI::BlockVector &vector,
                             const Point<dim> &p,
                             const unsigned int component) const;

  void compute_point_stress();

  void output_results() const;

  void compute_functional_values();

  void compute_load();

  void compute_cod_array();

  void compute_tcv();

  double compute_cod(const double eval_line);

  double compute_energy();

  bool refine_mesh();

  void project_back_phase_field();

  MPI_Comm mpi_com;

  Introspection<dim> introspection;

  ParameterHandler &prm;

  parallel::distributed::Triangulation<dim> triangulation;

  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  ConstraintMatrix constraints_update;
  ConstraintMatrix constraints_hanging_nodes;

  LA::MPI::BlockSparseMatrix system_pde_matrix;
  LA::MPI::BlockVector solution, newton_update, old_solution, old_old_solution,
      system_pde_residual;
  LA::MPI::BlockVector system_total_residual;

  LA::MPI::BlockVector diag_mass, diag_mass_relevant;

  ConditionalOStream pcout;
  TimerOutput timer;

  IndexSet active_set;

  Function<dim> *func_emodulus;

  std::vector<IndexSet> partition;
  std::vector<IndexSet> partition_relevant;

  std::vector<std::vector<bool>> constant_modes;

  LA::MPI::PreconditionAMG preconditioner_solid;
  LA::MPI::PreconditionAMG preconditioner_phase_field;

  // Global variables for timestepping scheme
  unsigned int timestep_number;
  unsigned int max_no_timesteps;
  double timestep, timestep_size_2, time;
  unsigned int switch_timestep;
  struct OuterSolverType {
    enum Enum { active_set, simple_monolithic };
  };
  typename OuterSolverType::Enum outer_solver;

  struct TestCase {
    enum Enum {
      sneddon,
      miehe_tension,
      miehe_shear,
      multiple_homo,
      multiple_het,
      three_point_bending
    };
  };
  typename TestCase::Enum test_case;

  struct RefinementStrategy {
    enum Enum {
      phase_field_ref,
      fixed_preref_sneddon,
      fixed_preref_miehe_tension,
      fixed_preref_miehe_shear,
      fixed_preref_multiple_homo,
      fixed_preref_multiple_het,
      global,
      mix,
      phase_field_ref_three_point_top
    };
  };
  typename RefinementStrategy::Enum refinement_strategy;

  bool direct_solver;

  // Biot parameters
  double c_biot, alpha_biot, lame_coefficient_biot, K_biot, density_biot;

  // Structure parameters
  double lame_coefficient_mu, lame_coefficient_lambda, poisson_ratio_nu;

  // Other parameters to control the fluid mesh motion

  FunctionParser<1> func_pressure;
  double constant_k, alpha_eps, G_c, viscosity_biot, gamma_penal;

  double E_modulus, E_prime;
  double min_cell_diameter, norm_part_iterations,
      value_phase_field_for_refinement;

  unsigned int n_global_pre_refine, n_local_pre_refine, n_refinement_cycles;

  double lower_bound_newton_residual;
  unsigned int max_no_newton_steps;
  double upper_newton_rho;
  unsigned int max_no_line_search_steps;
  double line_search_damping;
  double decompose_stress_rhs, decompose_stress_matrix;
  std::string output_folder;
  std::string filename_basis;
  double old_timestep, old_old_timestep;
  bool use_old_timestep_pf;

  TableHandler statistics;
};

template <int dim>
FracturePhaseFieldProblem<dim>::FracturePhaseFieldProblem(
    ParameterHandler &param)
    : mpi_com(MPI_COMM_WORLD), introspection(param), prm(param),
      triangulation(mpi_com),
      fe(introspection.fes, introspection.multiplicities),
      dof_handler(triangulation),

      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_com) == 0)),
      timer(mpi_com, pcout, TimerOutput::every_call_and_summary,
            TimerOutput::cpu_and_wall_times) {
  statistics.set_auto_fill_mode(true);
}

template <int dim> void FracturePhaseFieldProblem<dim>::setup_mesh() {
  std::string mesh_info = "";

  switch (test_case) {
  case TestCase::miehe_shear:
  case TestCase::miehe_tension:
    mesh_info = "ucd $SRC/meshes/unit_slit.inp";
    break;

  case TestCase::sneddon:
    if (dim == 2)
      mesh_info = "rect -10 -10 10 10";
    else
      mesh_info = "rect -10 -10 -10 10 10 10";
    break;

  case TestCase::multiple_homo:
  case TestCase::multiple_het:
    if (dim == 2)
      mesh_info = "ucd $SRC/meshes/unit_square_4.inp";
    else
      mesh_info = "ucd $SRC/meshes/unit_cube_10.inp";
    break;

  case TestCase::three_point_bending:
    // mesh_info = "msh $SRC/meshes/threepoint-notsym.msh";
    // mesh_info = "msh $SRC/meshes/threepoint-notsym_b.msh";
    mesh_info = "msh $SRC/meshes/threepoint.msh";
    break;
  }

  // TODO: overwrite defaults from parameter file if given
  // if (mesh != "") mesh_info = mesh;

  AssertThrow(mesh_info != "", ExcMessage("Error: no mesh information given."));

  std::istringstream is(mesh_info);
  std::string type;
  std::string grid_name = "";
  typename GridIn<dim>::Format format = GridIn<dim>::ucd;
  is >> type;

  if (type == "rect") {
    Point<dim> p1, p2;
    if (dim == 2)
      is >> p1[0] >> p1[1] >> p2[0] >> p2[1];
    else
      is >> p1[0] >> p1[1] >> p1[2] >> p2[0] >> p2[1] >> p2[2];

    std::vector<unsigned int> repetitions(dim, 10); // 10 in each direction
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1,
                                              p2,
                                              /*colorize*/ true);
  } else if (type == "msh") {
    format = GridIn<dim>::msh;
    is >> grid_name;
  } else if (type == "ucd") {
    format = GridIn<dim>::ucd;
    is >> grid_name;
  }

  if (grid_name != "") {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    grid_name = Utilities::replace_in_string(grid_name, "$SRC", SOURCE_DIR);
    std::ifstream input_file(grid_name.c_str());
    grid_in.read(input_file, format);
  }

  if (test_case == TestCase::three_point_bending) {
    // adjust boundary conditions
    double eps_machine = 1.0e-10;

    typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
    for (; cell != endc; ++cell)
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const Point<dim> face_center = cell->face(f)->center();
        if (cell->face(f)->at_boundary()) {
          if ((face_center[1] < 2.0 + eps_machine) &&
              (face_center[1] > 2.0 - eps_machine))
            cell->face(f)->set_boundary_id(3);
          else if ((face_center[0] < -4.0 + eps_machine) &&
                   (face_center[0] > -4.0 - eps_machine))
            cell->face(f)->set_boundary_id(0);
          else if ((face_center[0] < 4.0 + eps_machine) &&
                   (face_center[0] > 4.0 - eps_machine))
            cell->face(f)->set_boundary_id(1);
        }
      }
  }
}

template <int dim>
void FracturePhaseFieldProblem<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("Global parameters");
  {
    prm.declare_entry("Dimension", "2", Patterns::Integer(0));

    prm.declare_entry("FE degree", "1", Patterns::Integer(1));

    prm.declare_entry("Global pre-refinement steps", "1", Patterns::Integer(0));

    prm.declare_entry("Local pre-refinement steps", "0", Patterns::Integer(0));

    prm.declare_entry("Adaptive refinement cycles", "0", Patterns::Integer(0));

    prm.declare_entry("Max No of timesteps", "1", Patterns::Integer(0));

    prm.declare_entry("Timestep size", "1.0", Patterns::Double(0));

    prm.declare_entry("Timestep size to switch to", "1.0", Patterns::Double(0));

    prm.declare_entry("Switch timestep after steps", "0", Patterns::Integer(0));

    prm.declare_entry("outer solver", "active set",
                      Patterns::Selection("active set|simple monolithic"));

    prm.declare_entry(
        "test case", "sneddon",
        Patterns::Selection("sneddon|miehe tension|miehe shear|multiple "
                            "homo|multiple het|three point bending"));

    prm.declare_entry(
        "ref strategy", "phase field",
        Patterns::Selection(
            "phase field|fixed preref sneddon|fixed preref miehe tension|fixed "
            "preref miehe shear|fixed preref multiple homo|fixed preref "
            "multiple het|global|mix|phase field three point top"));

    prm.declare_entry("value phase field for refinement", "0.0",
                      Patterns::Double(0));

    prm.declare_entry("Output directory", "output", Patterns::Anything());
    prm.declare_entry("Output filename", "solution_", Patterns::Anything());
  }
  prm.leave_subsection();

  prm.enter_subsection("Problem dependent parameters");
  {
    prm.declare_entry("K reg", "1.0 * h", Patterns::Anything());

    prm.declare_entry("Eps reg", "1.0 * h", Patterns::Anything());

    prm.declare_entry("Gamma penalization", "0.0", Patterns::Double(0));

    prm.declare_entry("Pressure", "0.0", Patterns::Anything());

    prm.declare_entry("Fracture toughness G_c", "0.0", Patterns::Double(0));

    prm.declare_entry("Poisson ratio nu", "0.0", Patterns::Double(0));

    prm.declare_entry("E modulus", "0.0", Patterns::Double(0));

    prm.declare_entry("Lame mu", "0.0", Patterns::Double(0));

    prm.declare_entry("Lame lambda", "0.0", Patterns::Double(0));
  }
  prm.leave_subsection();

  prm.enter_subsection("Solver parameters");
  {
    prm.declare_entry("Use Direct Inner Solver", "false", Patterns::Bool());

    prm.declare_entry("Newton lower bound", "1.0e-10", Patterns::Double(0));

    prm.declare_entry("Newton maximum steps", "10", Patterns::Integer(0));

    prm.declare_entry("Upper Newton rho", "0.999", Patterns::Double(0));

    prm.declare_entry("Line search maximum steps", "5", Patterns::Integer(0));

    prm.declare_entry("Line search damping", "0.5", Patterns::Double(0));

    prm.declare_entry("Decompose stress in rhs", "0.0", Patterns::Double(0));

    prm.declare_entry("Decompose stress in matrix", "0.0", Patterns::Double(0));
  }
  prm.leave_subsection();
}

// In this method, we set up runtime parameters that
// could also come from a paramter file.
template <int dim>
void FracturePhaseFieldProblem<dim>::set_runtime_parameters() {
  // Get parameters from file
  prm.enter_subsection("Global parameters");
  n_global_pre_refine = prm.get_integer("Global pre-refinement steps");
  n_local_pre_refine = prm.get_integer("Local pre-refinement steps");
  n_refinement_cycles = prm.get_integer("Adaptive refinement cycles");
  max_no_timesteps = prm.get_integer("Max No of timesteps");
  timestep = prm.get_double("Timestep size");
  timestep_size_2 = prm.get_double("Timestep size to switch to");
  switch_timestep = prm.get_integer("Switch timestep after steps");

  if (prm.get("outer solver") == "active set")
    outer_solver = OuterSolverType::active_set;
  else if (prm.get("outer solver") == "simple monolithic")
    outer_solver = OuterSolverType::simple_monolithic;

  if (prm.get("test case") == "sneddon")
    test_case = TestCase::sneddon;
  else if (prm.get("test case") == "miehe tension")
    test_case = TestCase::miehe_tension; // straight crack
  else if (prm.get("test case") == "miehe shear")
    test_case = TestCase::miehe_shear; // curved crack
  else if (prm.get("test case") == "multiple homo")
    test_case =
        TestCase::multiple_homo; // multiple fractures homogeneous material
  else if (prm.get("test case") == "multiple het")
    test_case =
        TestCase::multiple_het; // multiple fractures heterogeneous material
  else if (prm.get("test case") == "three point bending")
    test_case = TestCase::three_point_bending;
  else
    AssertThrow(false, ExcNotImplemented());

  if (prm.get("ref strategy") == "phase field")
    refinement_strategy = RefinementStrategy::phase_field_ref;
  else if (prm.get("ref strategy") == "fixed preref sneddon")
    refinement_strategy = RefinementStrategy::fixed_preref_sneddon;
  else if (prm.get("ref strategy") == "fixed preref miehe tension")
    refinement_strategy = RefinementStrategy::fixed_preref_miehe_tension;
  else if (prm.get("ref strategy") == "fixed preref miehe shear")
    refinement_strategy = RefinementStrategy::fixed_preref_miehe_shear;
  else if (prm.get("ref strategy") == "fixed preref multiple homo")
    refinement_strategy = RefinementStrategy::fixed_preref_multiple_homo;
  else if (prm.get("ref strategy") == "fixed preref multiple het")
    refinement_strategy = RefinementStrategy::fixed_preref_multiple_het;
  else if (prm.get("ref strategy") == "global")
    refinement_strategy = RefinementStrategy::global;
  else if (prm.get("ref strategy") == "mix")
    refinement_strategy = RefinementStrategy::mix;
  else if (prm.get("ref strategy") == "phase field three point top")
    refinement_strategy = RefinementStrategy::phase_field_ref_three_point_top;
  else
    AssertThrow(false, ExcNotImplemented());

  value_phase_field_for_refinement =
      prm.get_double("value phase field for refinement");

  output_folder = prm.get("Output directory");
  filename_basis = prm.get("Output filename");

  prm.leave_subsection();

  prm.enter_subsection("Problem dependent parameters");

  // Phase-field parameters
  // They are given some values below
  constant_k = 0; // prm.get_double("K reg");
  alpha_eps = 0;  // prm.get_double("Eps reg");

  // Switch between active set strategy
  // and simple penalization
  // in order to enforce crack irreversiblity
  if (outer_solver == OuterSolverType::active_set)
    gamma_penal = 0.0;
  else
    gamma_penal = prm.get_double("Gamma penalization");

  // Material and problem-rhs parameters
  func_pressure.initialize("time", prm.get("Pressure"),
                           FunctionParser<1>::ConstMap());

  G_c = prm.get_double("Fracture toughness G_c");

  // In all examples chosen as 0. Will be non-zero
  // if a Darcy fluid is computed
  alpha_biot = 0.0;

  if (test_case == TestCase::sneddon || test_case == TestCase::multiple_homo ||
      test_case == TestCase::multiple_het) {
    poisson_ratio_nu = prm.get_double("Poisson ratio nu");
    E_modulus = prm.get_double("E modulus");

    lame_coefficient_mu = E_modulus / (2.0 * (1 + poisson_ratio_nu));

    lame_coefficient_lambda = (2 * poisson_ratio_nu * lame_coefficient_mu) /
                              (1.0 - 2 * poisson_ratio_nu);
  } else {
    // Miehe 2010
    lame_coefficient_mu = prm.get_double("Lame mu");
    lame_coefficient_lambda = prm.get_double("Lame lambda");

    // Dummy
    poisson_ratio_nu = prm.get_double("Poisson ratio nu");
    E_modulus = prm.get_double("E modulus");
  }

  prm.leave_subsection();

  E_prime = E_modulus / (1.0 - poisson_ratio_nu * poisson_ratio_nu);

  // A variable to count the number of time steps
  timestep_number = 0;

  // Counts total time
  time = 0;

  setup_mesh();
  triangulation.refine_global(n_global_pre_refine);

  pcout << "Cells:\t" << triangulation.n_active_cells() << std::endl;

  if (test_case == TestCase::multiple_het) {
    const std::string filename =
        Utilities::replace_in_string("$SRC/test.pgm", "$SRC", SOURCE_DIR);
    func_emodulus = new BitmapFunction<dim>(filename, 0, 10, 0, 10, E_modulus,
                                            10.0 * E_modulus);
  }

  prm.enter_subsection("Solver parameters");
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
  decompose_stress_rhs = prm.get_double("Decompose stress in rhs");
  decompose_stress_matrix = prm.get_double("Decompose stress in matrix");

  // For pf_extra
  use_old_timestep_pf = false;

  prm.leave_subsection();
}

template <int dim> void FracturePhaseFieldProblem<dim>::setup_system() {
  system_pde_matrix.clear();

  dof_handler.distribute_dofs(fe);

  std::vector<unsigned int> sub_blocks(introspection.n_components, 0);
  sub_blocks[introspection.component_indices.phase_field] = 1;
  DoFRenumbering::component_wise(dof_handler, sub_blocks);

  constant_modes.clear();
  DoFTools::extract_constant_modes(
      dof_handler, introspection.component_masks.displacements, constant_modes);

  {
    // extract DoF counts for printing statistics:
#if DEAL_II_VERSION_GTE(9, 2, 0)
    std::vector<types::global_dof_index> dofs_per_var =
        DoFTools::count_dofs_per_fe_block(dof_handler, sub_blocks);
#else
    std::vector<types::global_dof_index> dofs_per_var(2);
    DoFTools::count_dofs_per_block(dof_handler, dofs_per_var, sub_blocks);
#endif

    const unsigned int n_solid = dofs_per_var[0];
    const unsigned int n_phase = dofs_per_var[1];
    pcout << std::endl;
    pcout << "DoFs: " << n_solid << " solid + " << n_phase << " phase"
          << " = " << dof_handler.n_dofs() << std::endl;
  }

#if DEAL_II_VERSION_GTE(9, 2, 0)
  std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler,
                                        introspection.components_to_blocks);
#else
  std::vector<types::global_dof_index> dofs_per_block(introspection.n_blocks);
  DoFTools::count_dofs_per_block(dof_handler, dofs_per_block,
                                 introspection.components_to_blocks);
#endif

  partition.clear();
  compatibility::split_by_block(dofs_per_block,
                                dof_handler.locally_owned_dofs(), partition);

  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_set);

  partition_relevant.clear();
  compatibility::split_by_block(dofs_per_block, relevant_set,
                                partition_relevant);

  {
    constraints_hanging_nodes.clear();
    constraints_hanging_nodes.reinit(relevant_set);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_hanging_nodes);
    constraints_hanging_nodes.close();
  }
  {
    constraints_update.clear();
    constraints_update.reinit(relevant_set);

    set_newton_bc();
    constraints_update.merge(constraints_hanging_nodes,
                             ConstraintMatrix::right_object_wins);
    constraints_update.close();
  }

  {
    TrilinosWrappers::BlockSparsityPattern csp(partition, mpi_com);

    DoFTools::make_sparsity_pattern(dof_handler, csp, constraints_update, false,
                                    Utilities::MPI::this_mpi_process(mpi_com));

    csp.compress();
    system_pde_matrix.reinit(csp);
  }

  // Actual solution at time step n
  solution.reinit(partition);

  // Old timestep solution at time step n-1
  old_solution.reinit(partition_relevant);

  // Old timestep solution at time step n-2
  old_old_solution.reinit(partition_relevant);

  // Updates for Newton's method
  newton_update.reinit(partition);

  // Residual for  Newton's method
  system_pde_residual.reinit(partition);

  system_total_residual.reinit(partition);

  diag_mass.reinit(partition);
  diag_mass_relevant.reinit(partition_relevant);
  assemble_diag_mass_matrix();

  active_set.clear();
  active_set.set_size(dof_handler.n_dofs());
}

// In this function, we assemble the Jacobian matrix
// for the Newton iteration.
template <int dim>
void FracturePhaseFieldProblem<dim>::assemble_system(bool residual_only) {
  if (residual_only)
    system_total_residual = 0;
  else
    system_pde_matrix = 0;
  system_pde_residual = 0;

  // This function is only necessary
  // when working with simple penalization
  if ((outer_solver == OuterSolverType::simple_monolithic) &&
      (timestep_number < 1)) {
    gamma_penal = 0.0;
  }
  const double current_pressure = func_pressure.value(Point<1>(time), 0);

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  LA::MPI::BlockVector rel_old_solution(partition_relevant);
  rel_old_solution = old_solution;

  LA::MPI::BlockVector rel_old_old_solution(partition_relevant);
  rel_old_old_solution = old_old_solution;

  QGauss<dim> quadrature_formula(fe.degree + 2);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Old Newton values
  std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
  std::vector<double> old_phase_field_values(n_q_points);

  // Old Newton grads
  std::vector<Tensor<2, dim>> old_displacement_grads(n_q_points);
  std::vector<Tensor<1, dim>> old_phase_field_grads(n_q_points);

  // Old timestep values
  std::vector<Tensor<1, dim>> old_timestep_displacement_values(n_q_points);
  std::vector<double> old_timestep_phase_field_values(n_q_points);
  std::vector<Tensor<1, dim>> old_timestep_velocity_values(n_q_points);

  std::vector<Tensor<1, dim>> old_old_timestep_displacement_values(n_q_points);
  std::vector<double> old_old_timestep_phase_field_values(n_q_points);

  // Declaring test functions:
  std::vector<Tensor<1, dim>> phi_i_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> phi_i_grads_u(dofs_per_cell);
  std::vector<double> phi_i_pf(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_i_grads_pf(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  Tensor<2, dim> zero_matrix;
  zero_matrix.clear();

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);

      // update lame coefficients based on current cell
      // when working with heterogeneous materials
      if (test_case == TestCase::multiple_het) {
        E_modulus = func_emodulus->value(cell->center(), 0);
        E_modulus += 1.0;

        lame_coefficient_mu = E_modulus / (2.0 * (1 + poisson_ratio_nu));

        lame_coefficient_lambda = (2 * poisson_ratio_nu * lame_coefficient_mu) /
                                  (1.0 - 2 * poisson_ratio_nu);
      }

      local_matrix = 0;
      local_rhs = 0;

      // Old Newton iteration values
      fe_values[introspection.extractors.displacement].get_function_values(
          rel_solution, old_displacement_values);
      fe_values[introspection.extractors.phase_field].get_function_values(
          rel_solution, old_phase_field_values);

      fe_values[introspection.extractors.displacement].get_function_gradients(
          rel_solution, old_displacement_grads);
      fe_values[introspection.extractors.phase_field].get_function_gradients(
          rel_solution, old_phase_field_grads);

      // Old_timestep_solution values
      fe_values[introspection.extractors.phase_field].get_function_values(
          rel_old_solution, old_timestep_phase_field_values);

      // Old Old_timestep_solution values
      fe_values[introspection.extractors.phase_field].get_function_values(
          rel_old_old_solution, old_old_timestep_phase_field_values);

      {
        for (unsigned int q = 0; q < n_q_points; ++q) {
          for (unsigned int k = 0; k < dofs_per_cell; ++k) {
            phi_i_u[k] =
                fe_values[introspection.extractors.displacement].value(k, q);
            phi_i_grads_u[k] =
                fe_values[introspection.extractors.displacement].gradient(k, q);
            phi_i_pf[k] =
                fe_values[introspection.extractors.phase_field].value(k, q);
            phi_i_grads_pf[k] =
                fe_values[introspection.extractors.phase_field].gradient(k, q);
          }

          // First, we prepare things coming from the previous Newton
          // iteration...
          double pf = old_phase_field_values[q];
          double old_timestep_pf = old_timestep_phase_field_values[q];
          double old_old_timestep_pf = old_old_timestep_phase_field_values[q];
          if (outer_solver == OuterSolverType::simple_monolithic) {
            pf = std::max(0.0, old_phase_field_values[q]);
            old_timestep_pf = std::max(0.0, old_timestep_phase_field_values[q]);
            old_old_timestep_pf =
                std::max(0.0, old_old_timestep_phase_field_values[q]);
          }

          double pf_minus_old_timestep_pf_plus =
              std::max(0.0, pf - old_timestep_pf);

          double pf_extra = pf;
          // Linearization by extrapolation to cope with non-convexity of the
          // underlying energy functional. This idea might be refined in a
          // future work (be also careful because theoretically, we do not have
          // time regularity; therefore extrapolation in time might be
          // questionable. But for the time being, this is numerically robust.
          pf_extra = old_old_timestep_pf +
                     (time - (time - old_timestep - old_old_timestep)) /
                         (time - old_timestep -
                          (time - old_timestep - old_old_timestep)) *
                         (old_timestep_pf - old_old_timestep_pf);
          if (pf_extra <= 0.0)
            pf_extra = 0.0;
          if (pf_extra >= 1.0)
            pf_extra = 1.0;

          if (use_old_timestep_pf)
            pf_extra = old_timestep_pf;

          const Tensor<2, dim> grad_u = old_displacement_grads[q];
          const Tensor<1, dim> grad_pf = old_phase_field_grads[q];

          const double divergence_u = Tensors ::get_divergence_u<dim>(grad_u);

          const Tensor<2, dim> Identity = Tensors ::get_Identity<dim>();

          const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
          const double tr_E = trace(E);

          Tensor<2, dim> stress_term_plus;
          Tensor<2, dim> stress_term_minus;
          if (decompose_stress_matrix > 0 && timestep_number > 0) {
            decompose_stress(stress_term_plus, stress_term_minus, E, tr_E,
                             zero_matrix, 0.0, lame_coefficient_lambda,
                             lame_coefficient_mu, false);
          } else {
            stress_term_plus = lame_coefficient_lambda * tr_E * Identity +
                               2 * lame_coefficient_mu * E;
            stress_term_minus = 0;
          }

          if (!residual_only)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              double pf_minus_old_timestep_pf_plus = 0.0;
              if ((pf - old_timestep_pf) < 0.0)
                pf_minus_old_timestep_pf_plus = 0.0;
              else
                pf_minus_old_timestep_pf_plus = phi_i_pf[i];

              const Tensor<2, dim> E_LinU =
                  0.5 * (phi_i_grads_u[i] + transpose(phi_i_grads_u[i]));
              const double tr_E_LinU = trace(E_LinU);

              const double divergence_u_LinU =
                  Tensors ::get_divergence_u<dim>(phi_i_grads_u[i]);

              Tensor<2, dim> stress_term_LinU;
              stress_term_LinU =
                  lame_coefficient_lambda * tr_E_LinU * Identity +
                  2 * lame_coefficient_mu * E_LinU;

              Tensor<2, dim> stress_term_plus_LinU;
              Tensor<2, dim> stress_term_minus_LinU;

              const unsigned int comp_i = fe.system_to_component_index(i).first;
              if (comp_i == introspection.component_indices.phase_field) {
                stress_term_plus_LinU = 0;
                stress_term_minus_LinU = 0;
              } else if (decompose_stress_matrix > 0.0 && timestep_number > 0) {
                decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                                 E, tr_E, E_LinU, tr_E_LinU,
                                 lame_coefficient_lambda, lame_coefficient_mu,
                                 true);
              } else {
                stress_term_plus_LinU =
                    lame_coefficient_lambda * tr_E_LinU * Identity +
                    2 * lame_coefficient_mu * E_LinU;
                stress_term_minus = 0;
              }

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const unsigned int comp_j =
                    fe.system_to_component_index(j).first;
                if (comp_j < dim) {
                  // Solid
                  local_matrix(j, i) +=
                      1.0 *
                      (scalar_product(((1 - constant_k) * pf_extra * pf_extra +
                                       constant_k) *
                                          stress_term_plus_LinU,
                                      phi_i_grads_u[j])
                       // stress term minus
                       + decompose_stress_matrix *
                             scalar_product(stress_term_minus_LinU,
                                            phi_i_grads_u[j])) *
                      fe_values.JxW(q);

                } else if (comp_j ==
                           introspection.component_indices.phase_field) {
                  // Simple penalization for simple monolithic
                  local_matrix(j, i) += gamma_penal / timestep * 1.0 /
                                        (cell->diameter() * cell->diameter()) *
                                        pf_minus_old_timestep_pf_plus *
                                        phi_i_pf[j] * fe_values.JxW(q);

                  // Phase-field
                  local_matrix(j, i) +=
                      ((1 - constant_k) *
                           (scalar_product(stress_term_plus_LinU, E) +
                            scalar_product(stress_term_plus, E_LinU)) *
                           pf * phi_i_pf[j] +
                       (1 - constant_k) * scalar_product(stress_term_plus, E) *
                           phi_i_pf[i] * phi_i_pf[j] +
                       G_c / alpha_eps * phi_i_pf[i] * phi_i_pf[j] +
                       G_c * alpha_eps * phi_i_grads_pf[i] * phi_i_grads_pf[j]
                       // Pressure terms
                       - 2.0 * (alpha_biot - 1.0) * current_pressure *
                             (pf * divergence_u_LinU +
                              phi_i_pf[i] * divergence_u) *
                             phi_i_pf[j]) *
                      fe_values.JxW(q);
                }

                // end j dofs
              }
              // end i dofs
            }

          // RHS:
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const unsigned int comp_i = fe.system_to_component_index(i).first;
            if (comp_i < dim) {
              const Tensor<2, dim> phi_i_grads_u =
                  fe_values[introspection.extractors.displacement].gradient(i,
                                                                            q);
              const double divergence_u_LinU =
                  Tensors ::get_divergence_u<dim>(phi_i_grads_u);

              // Solid
              local_rhs(i) -=
                  (scalar_product(
                       ((1.0 - constant_k) * pf_extra * pf_extra + constant_k) *
                           stress_term_plus,
                       phi_i_grads_u) +
                   decompose_stress_rhs *
                       scalar_product(stress_term_minus, phi_i_grads_u)
                   // Pressure terms
                   - (alpha_biot - 1.0) * current_pressure * pf_extra *
                         pf_extra * divergence_u_LinU) *
                  fe_values.JxW(q);

            } else if (comp_i == introspection.component_indices.phase_field) {
              const double phi_i_pf =
                  fe_values[introspection.extractors.phase_field].value(i, q);
              const Tensor<1, dim> phi_i_grads_pf =
                  fe_values[introspection.extractors.phase_field].gradient(i,
                                                                           q);

              // Simple penalization
              local_rhs(i) -= gamma_penal / timestep * 1.0 /
                              (cell->diameter() * cell->diameter()) *
                              pf_minus_old_timestep_pf_plus * phi_i_pf *
                              fe_values.JxW(q);

              // Phase field
              local_rhs(i) -=
                  ((1.0 - constant_k) * scalar_product(stress_term_plus, E) *
                       pf * phi_i_pf -
                   G_c / alpha_eps * (1.0 - pf) * phi_i_pf +
                   G_c * alpha_eps * grad_pf * phi_i_grads_pf
                   // Pressure terms
                   - 2.0 * (alpha_biot - 1.0) * current_pressure * pf *
                         divergence_u * phi_i_pf) *
                  fe_values.JxW(q);
            }

          } // end i

          // end n_q_points
        }

        cell->get_dof_indices(local_dof_indices);
        if (residual_only) {
          constraints_update.distribute_local_to_global(
              local_rhs, local_dof_indices, system_pde_residual);

          if (outer_solver == OuterSolverType::active_set) {
            constraints_hanging_nodes.distribute_local_to_global(
                local_rhs, local_dof_indices, system_total_residual);
          } else {
            constraints_update.distribute_local_to_global(
                local_rhs, local_dof_indices, system_total_residual);
          }
        } else {
          constraints_update.distribute_local_to_global(
              local_matrix, local_rhs, local_dof_indices, system_pde_matrix,
              system_pde_residual);
        }
        // end if (second PDE: STVK material)
      }
      // end cell
    }

  if (residual_only)
    system_total_residual.compress(VectorOperation::add);
  else
    system_pde_matrix.compress(VectorOperation::add);

  system_pde_residual.compress(VectorOperation::add);

  if (!direct_solver && !residual_only) {
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      data.constant_modes = constant_modes;
      data.elliptic = true;
      data.higher_order_elements = true;
      data.smoother_sweeps = 2;
      data.aggregation_threshold = 0.02;
      preconditioner_solid.initialize(system_pde_matrix.block(0, 0), data);
    }
    {
      LA::MPI::PreconditionAMG::AdditionalData data;
      // data.constant_modes = constant_modes;
      data.elliptic = true;
      data.higher_order_elements = true;
      data.smoother_sweeps = 2;
      data.aggregation_threshold = 0.02;
      preconditioner_phase_field.initialize(system_pde_matrix.block(1, 1),
                                            data);
    }
  }
}

// In this function we assemble the semi-linear
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the
// system matrix.
template <int dim> void FracturePhaseFieldProblem<dim>::assemble_nl_residual() {
  assemble_system(true);
}

template <int dim>
void FracturePhaseFieldProblem<dim>::assemble_diag_mass_matrix() {
  diag_mass = 0;

  QGaussLobatto<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  const unsigned int n_q_points = quadrature_formula.size();

  Vector<double> local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      local_rhs = 0;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i != introspection.component_indices.phase_field)
            continue; // only look at phase field

          local_rhs(i) += fe_values.shape_value(i, q_point) *
                          fe_values.shape_value(i, q_point) *
                          fe_values.JxW(q_point);
        }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        diag_mass(local_dof_indices[i]) += local_rhs(i);
    }

  diag_mass.compress(VectorOperation::add);
  diag_mass_relevant = diag_mass;
}

// Here, we impose boundary conditions. If initial_step is true, these are
// non-zero conditions, otherwise they are homogeneous conditions as we solve
// the Newton system in update form.
template <int dim>
void FracturePhaseFieldProblem<dim>::set_boundary_conditions(
    const double time, const bool initial_step, ConstraintMatrix &constraints) {
  compatibility::ZeroFunction<dim> f_zero(introspection.n_components);

  if (dim == 2) {
    if (test_case == TestCase::sneddon ||
        test_case == TestCase::multiple_homo ||
        test_case == TestCase::multiple_het) {
      for (unsigned int bc = 0; bc < 4; ++bc)
        VectorTools::interpolate_boundary_values(
            dof_handler, bc, f_zero, constraints,
            introspection.component_masks.displacements);
    } else if (test_case == TestCase::miehe_tension) {
      // Tension test (e.g., phase-field by Miehe et al. in 2010)
      VectorTools::interpolate_boundary_values(
          dof_handler, 2, f_zero, constraints,
          introspection.component_masks.displacement[1]);

      if (initial_step)
        VectorTools::interpolate_boundary_values(
            dof_handler, 3,
            BoundaryTensionTest<dim>(introspection.n_components, time),
            constraints, introspection.component_masks.displacements);
      else
        VectorTools::interpolate_boundary_values(
            dof_handler, 3, f_zero, constraints,
            introspection.component_masks.displacements);
    } else if (test_case == TestCase::miehe_shear) {
      // Single edge notched shear (e.g., phase-field by Miehe et al. in 2010)
      VectorTools::interpolate_boundary_values(
          dof_handler, 0, f_zero, constraints,
          introspection.component_masks.displacement[1]);
      VectorTools::interpolate_boundary_values(
          dof_handler, 1, f_zero, constraints,
          introspection.component_masks.displacement[1]);
      VectorTools::interpolate_boundary_values(
          dof_handler, 2, f_zero, constraints,
          introspection.component_masks.displacements);
      if (initial_step)
        VectorTools::interpolate_boundary_values(
            dof_handler, 3,
            BoundaryShearTest<dim>(introspection.n_components, time),
            constraints, introspection.component_masks.displacements);
      else
        VectorTools::interpolate_boundary_values(
            dof_handler, 3, f_zero, constraints,
            introspection.component_masks.displacements);

      //      bottom part of crack
      VectorTools::interpolate_boundary_values(
          dof_handler, 4, f_zero, constraints,
          introspection.component_masks.displacement[1]);
    } else if (test_case == TestCase::three_point_bending) {
      // fix y component of left and right bottom corners
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                                .begin_active(),
                                                     endc = dof_handler.end();

      for (; cell != endc; ++cell) {
        if (cell->is_artificial())
          continue;

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
             ++v) {
          if (std::abs(cell->vertex(v)[1]) < 1e-10 &&
              (std::abs(cell->vertex(v)[0] + 4.0) < 1e-10 ||
               std::abs(cell->vertex(v)[0] - 4.0) < 1e-10)) {
            // y displacement
            types::global_dof_index idx = cell->vertex_dof_index(
                v, introspection.component_indices.displacement[1]);
            constraints.add_line(idx);

            // x displacement
            idx = cell->vertex_dof_index(
                v, introspection.component_indices.displacement[0]);
            if (std::abs(cell->vertex(v)[0] + 4.0) < 1e-10)
              constraints.add_line(idx);

            // phasefield: TODO, is this really necessary?
            idx = cell->vertex_dof_index(
                v, introspection.component_indices.phase_field);
            constraints.add_line(idx);
            if (initial_step)
              constraints.set_inhomogeneity(idx, 1.0);
          } else if (std::abs(cell->vertex(v)[0]) < 1e-10 &&
                     std::abs(cell->vertex(v)[1] - 2.0) < 1e-10) {
            types::global_dof_index idx =
                cell->vertex_dof_index(v,
                                       introspection.component_indices
                                           .displacement[0]); // x displacement
            // boundary_values[idx] = 0.0;
            idx =
                cell->vertex_dof_index(v,
                                       introspection.component_indices
                                           .displacement[1]); // y displacement
            constraints.add_line(idx);
            if (initial_step)
              constraints.set_inhomogeneity(idx, -1.0 * time);
          }
        }
      }

    } else
      AssertThrow(false, ExcNotImplemented());

  } // end 2d
  else if (dim == 3) {
    for (unsigned int b = 0; b < 6; ++b)
      VectorTools::interpolate_boundary_values(
          dof_handler, b, f_zero, constraints,
          introspection.component_masks.displacements);
  }
}

template <int dim>
void FracturePhaseFieldProblem<dim>::set_initial_bc(const double time) {
  ConstraintMatrix constraints;
  set_boundary_conditions(time, true, constraints);
  constraints.close();
  constraints.distribute(solution);
}

template <int dim> void FracturePhaseFieldProblem<dim>::set_newton_bc() {
  set_boundary_conditions(time, false, constraints_update);
}

template <class PreconditionerA, class PreconditionerC>
class BlockDiagonalPreconditioner {
public:
  BlockDiagonalPreconditioner(const LA::MPI::BlockSparseMatrix &M,
                              const PreconditionerA &pre_A,
                              const PreconditionerC &pre_C)
      : matrix(M), prec_A(pre_A), prec_C(pre_C) {}

  void vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
    prec_A.vmult(dst.block(0), src.block(0));
    prec_C.vmult(dst.block(1), src.block(1));
  }

  const LA::MPI::BlockSparseMatrix &matrix;
  const PreconditionerA &prec_A;
  const PreconditionerC &prec_C;
};

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration.
template <int dim> unsigned int FracturePhaseFieldProblem<dim>::solve() {
  newton_update = 0;

  if (direct_solver) {
    SolverControl cn;
    TrilinosWrappers::SolverDirect solver(cn);
    solver.solve(system_pde_matrix.block(0, 0), newton_update.block(0),
                 system_pde_residual.block(0));

    constraints_update.distribute(newton_update);

    return 1;
  } else {
    SolverControl solver_control(200, system_pde_residual.l2_norm() * 1e-8);

    SolverGMRES<LA::MPI::BlockVector> solver(solver_control);

    BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
                                LA::MPI::PreconditionAMG>
        preconditioner(system_pde_matrix, preconditioner_solid,
                       preconditioner_phase_field);

    solver.solve(system_pde_matrix, newton_update, system_pde_residual,
                 preconditioner);

    constraints_update.distribute(newton_update);

    return solver_control.last_step();
  }
}

template <int dim> double FracturePhaseFieldProblem<dim>::newton_active_set() {
  pcout << "It.\t#A.Set\t#CycDoF\tResidual\tReduction\tLSrch\t#LinIts"
        << std::endl;

  LA::MPI::BlockVector residual_relevant(partition_relevant);

  set_initial_bc(time);
  constraints_hanging_nodes.distribute(solution);

  assemble_nl_residual();
  residual_relevant = system_total_residual;

  constraints_update.set_zero(system_pde_residual);
  double newton_residual = system_pde_residual.l2_norm();

  double old_newton_residual = newton_residual;
  unsigned int newton_step = 0;

  pcout << "0\t\t\t" << std::scientific << newton_residual << std::endl;
  std::cout.unsetf(std::ios_base::floatfield);

  active_set.clear();
  active_set.set_size(dof_handler.n_dofs());

  // map global_dof_idx -> number of times it switched from inactive to active
  // to detect cycles
  std::map<types::global_dof_index, unsigned int> cycle_counter;

  LA::MPI::BlockVector old_solution_relevant(partition_relevant);
  old_solution_relevant = old_solution;

  unsigned int sum_lin_it = 0;

  double new_newton_residual = 0.0;
  while (true) {
    pcout << newton_step + 1 << std::flush;

    IndexSet active_set_old = active_set;
    unsigned int n_cycling_dofs = 0;

    {
      // compute new active set
      active_set.clear();
      active_set.set_size(dof_handler.n_dofs());
      constraints_update.clear();
      unsigned int owned_active_set_dofs = 0;

      LA::MPI::BlockVector solution_relevant(partition_relevant);
      solution_relevant = solution;

      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                                .begin_active(),
                                                     endc = dof_handler.end();

      for (; cell != endc; ++cell) {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i != introspection.component_indices.phase_field)
            continue; // only look at phase field

          const types::global_dof_index idx = local_dof_indices[i];

          double old_value = old_solution_relevant(idx);
          double new_value = solution_relevant(idx);

          // already processed or a hanging node?
          if (active_set.is_element(idx) ||
              constraints_hanging_nodes.is_constrained(idx))
            continue;

          double c = 1e+1 * E_modulus;
          double massm = diag_mass_relevant(idx);

          double gap = new_value - old_value;
          double active_set_tolarance = 0.0;

          // consider a DoF as cycling after this many inactive->active switches
          const unsigned int n_cycling_threshold = 5;

          if (residual_relevant(idx) / massm + c * (gap) <=
                  active_set_tolarance &&
              (cycle_counter[idx] < n_cycling_threshold))
            continue;

          if (cycle_counter[idx] >= n_cycling_threshold)
            ++n_cycling_dofs;

          // now idx is in the active set
          constraints_update.add_line(idx);
          constraints_update.set_inhomogeneity(idx, 0.0);
          solution(idx) = old_value;
          active_set.add_index(idx);

          if (dof_handler.locally_owned_dofs().is_element(idx))
            ++owned_active_set_dofs;
        }
      }
      solution.compress(VectorOperation::insert);
      // we might have changed values of the solution, so fix the
      // hanging nodes (we ignore in the active set):
      constraints_hanging_nodes.distribute(solution);

      pcout << "\t" << Utilities::MPI::sum(owned_active_set_dofs, mpi_com)
            << "\t" << Utilities::MPI::sum(n_cycling_dofs, mpi_com)
            << std::flush;
    }

    {
      // cycle detection: increment a counter for each DoF that became active
      IndexSet i_before = active_set_old;
      i_before.subtract_set(active_set);
      for (IndexSet::ElementIterator it = i_before.begin();
           it != i_before.end(); ++it)
        ++(cycle_counter[*it]);
    }

    set_newton_bc();
    constraints_update.merge(constraints_hanging_nodes,
                             ConstraintMatrix::right_object_wins);
    constraints_update.close();

    int is_my_set_changed = (active_set == active_set_old) ? 0 : 1;
    int num_changed = Utilities::MPI::sum(is_my_set_changed, MPI_COMM_WORLD);

    assemble_system();
    constraints_update.set_zero(system_pde_residual);
    unsigned int no_linear_iterations = solve();
    sum_lin_it += no_linear_iterations;

    LA::MPI::BlockVector saved_solution = solution;

    if (false) {
      solution += newton_update;
      project_back_phase_field();
      // output_results();

      assemble_nl_residual();
      constraints_update.set_zero(system_pde_residual);
      pcout << "full step res: " << system_pde_residual.l2_norm() << " "
            << std::endl;
      solution = saved_solution;
      assemble_nl_residual();
      constraints_update.set_zero(system_pde_residual);
      pcout << "0-size res: " << system_pde_residual.l2_norm() << " "
            << std::endl;
    }

    // line search:
    unsigned int line_search_step = 0;

    for (; line_search_step < max_no_line_search_steps; ++line_search_step) {
      solution += newton_update;

      assemble_nl_residual();
      residual_relevant = system_total_residual;
      constraints_update.set_zero(system_pde_residual);
      new_newton_residual = system_pde_residual.l2_norm();

      if (new_newton_residual < newton_residual)
        break;

      solution = saved_solution;
      newton_update *= line_search_damping;
    }
    pcout << std::scientific << "\t" << new_newton_residual << "\t"
          << new_newton_residual / newton_residual;
    std::cout.unsetf(std::ios_base::floatfield);
    pcout << "\t" << line_search_step << "\t" << no_linear_iterations
          << std::endl;

    old_newton_residual = newton_residual;
    newton_residual = new_newton_residual;

    ++newton_step;

    if (newton_residual < lower_bound_newton_residual && num_changed == 0) {
      // convergence achieved:
      pcout << "\tNewton iterations: " << newton_step
            << " total linear iterations: " << sum_lin_it << std::endl;

      break;
    }

    if (newton_step >= max_no_newton_steps) {
      pcout << "Newton iteration did not converge in " << newton_step
            << " steps." << std::endl;
      throw SolverControl::NoConvergence(0, 0);
    }
  }
  return new_newton_residual / old_newton_residual;
}

template <int dim>
double FracturePhaseFieldProblem<dim>::newton_iteration(const double time) {
  pcout << "It.\tResidual\tReduction\tLSrch\t\t#LinIts" << std::endl;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1;

  // Line search parameters
  unsigned int line_search_step = 0;
  double new_newton_residual = 0.0;

  // Application of the initial boundary conditions to the
  // variational equations:
  set_initial_bc(time);
  assemble_nl_residual();
  constraints_update.set_zero(system_pde_residual);

  double newton_residual = system_pde_residual.linfty_norm();
  double old_newton_residual = newton_residual;
  unsigned int newton_step = 1;
  unsigned int no_linear_iterations = 0;

  pcout << "0\t" << std::scientific << newton_residual << std::endl;

  while (newton_residual > lower_bound_newton_residual &&
         newton_step < max_no_newton_steps) {
    old_newton_residual = newton_residual;

    assemble_nl_residual();
    constraints_update.set_zero(system_pde_residual);
    newton_residual = system_pde_residual.linfty_norm();

    if (newton_residual < lower_bound_newton_residual) {
      pcout << '\t' << std::scientific << newton_residual << std::endl;
      break;
    }

    if (newton_step == 1 ||
        newton_residual / old_newton_residual > nonlinear_rho)
      assemble_system();

    // Solve Ax = b
    no_linear_iterations = solve();

    line_search_step = 0;
    for (; line_search_step < max_no_line_search_steps; ++line_search_step) {
      solution += newton_update;

      assemble_nl_residual();
      constraints_update.set_zero(system_pde_residual);
      new_newton_residual = system_pde_residual.linfty_norm();

      if (new_newton_residual < newton_residual)
        break;
      else
        solution -= newton_update;

      newton_update *= line_search_damping;
    }
    old_newton_residual = newton_residual;
    newton_residual = new_newton_residual;

    pcout << std::setprecision(5) << newton_step << '\t' << std::scientific
          << newton_residual;

    if (!direct_solver)
      pcout << " (" << system_pde_residual.block(0).linfty_norm() << '|'
            << system_pde_residual.block(1).linfty_norm() << ")";

    pcout << '\t' << std::scientific << newton_residual / old_newton_residual
          << '\t';

    if (newton_step == 1 ||
        newton_residual / old_newton_residual > nonlinear_rho)
      pcout << "rebuild" << '\t';
    else
      pcout << " " << '\t';
    pcout << line_search_step << '\t' << std::scientific << no_linear_iterations
          << '\t' << std::scientific << std::endl;

    // Terminate if nothing is solved anymore. After this,
    // we cut the time step.
    if ((newton_residual / old_newton_residual > upper_newton_rho) &&
        (newton_step > 1)) {
      break;
    }

    // Updates
    newton_step++;
  }

  if ((newton_residual > lower_bound_newton_residual) &&
      (newton_step == max_no_newton_steps)) {
    pcout << "Newton iteration did not converge in " << newton_step
          << " steps :-(" << std::endl;
    throw SolverControl::NoConvergence(0, 0);
  }

  return newton_residual / old_newton_residual;
}

template <int dim>
void FracturePhaseFieldProblem<dim>::project_back_phase_field() {
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
        const unsigned int comp_i = fe.system_to_component_index(i).first;
        if (comp_i != introspection.component_indices.phase_field)
          continue; // only look at phase field

        const types::global_dof_index idx = local_dof_indices[i];
        if (!dof_handler.locally_owned_dofs().is_element(idx))
          continue;

        solution(idx) =
            std::max(0.0, std::min(static_cast<double>(solution(idx)), 1.0));
      }
    }

  solution.compress(VectorOperation::insert);
}

//////////////////
template <int dim> void FracturePhaseFieldProblem<dim>::output_results() const {
  static int refinement_cycle = -1;
  ++refinement_cycle;

  LA::MPI::BlockVector relevant_solution(partition_relevant);
  relevant_solution = solution;

  SneddonExactPostProc<dim> exact_sol_sneddon(introspection.n_components,
                                              alpha_eps);
  DataOut<dim> data_out;
  {
    std::vector<std::string> solution_names(dim, "displacement");
    solution_names.push_back("phasefield");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector(dof_handler, relevant_solution, solution_names,
                             data_component_interpretation);
  }

  if (test_case == TestCase::sneddon) {
    data_out.add_data_vector(dof_handler, relevant_solution, exact_sol_sneddon);
  }

  Vector<float> e_mod(triangulation.n_active_cells());
  if (test_case == TestCase::multiple_het) {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();

    unsigned int cellindex = 0;
    for (; cell != endc; ++cell, ++cellindex)
      if (cell->is_locally_owned()) {
        e_mod(cellindex) = 1.0 + func_emodulus->value(cell->center(), 0);
      }
    data_out.add_data_vector(e_mod, "emodulus");
  }

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  TrilinosWrappers::MPI::Vector active_set_vector;

  if (outer_solver == OuterSolverType::active_set) {
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    TrilinosWrappers::MPI::Vector distributed_active_set_vector(
        dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto index : active_set)
      distributed_active_set_vector[index] = 1.;

    active_set_vector.reinit(locally_relevant_dofs, MPI_COMM_WORLD);
    active_set_vector = distributed_active_set_vector;

    data_out.add_data_vector(dof_handler, active_set_vector, "active_set");
  }

  data_out.build_patches();

  // Filename basis comes from parameter file
  std::ostringstream filename;

  pcout << "Write solution " << refinement_cycle << std::endl;

  filename << output_folder << "/" << filename_basis
           << Utilities::int_to_string(refinement_cycle, 5) << "."
           << Utilities::int_to_string(triangulation.locally_owned_subdomain(),
                                       4)
           << ".vtu";

  std::ofstream output(filename.str().c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_com) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_com); ++i)
      filenames.push_back(filename_basis +
                          Utilities::int_to_string(refinement_cycle, 5) + "." +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::string master_name = filename_basis +
                              Utilities::int_to_string(refinement_cycle, 5) +
                              ".pvtu";
    std::ofstream master_output((output_folder + "/" + master_name).c_str());
    data_out.write_pvtu_record(master_output, filenames);

    std::string visit_master_filename =
        (output_folder + "/" + filename_basis +
         Utilities::int_to_string(refinement_cycle, 5) + ".visit");
    std::ofstream visit_master(visit_master_filename.c_str());
    DataOutBase::write_visit_record(visit_master, filenames);

    static std::vector<std::vector<std::string>> output_file_names_by_timestep;
    output_file_names_by_timestep.push_back(filenames);
    std::ofstream global_visit_master(
        (output_folder + "/solution.visit").c_str());
    DataOutBase::write_visit_record(global_visit_master,
                                    output_file_names_by_timestep);

    pcout << "\tas " << visit_master_filename << std::endl;

    std::ofstream global_paraview_master(
        (output_folder + "/solution.pvd").c_str());
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.emplace_back(time, master_name);
    DataOutBase::write_pvd_record(global_paraview_master, times_and_names);
  }
}

// With help of this function, we extract
// point values for a certain component from our
// discrete solution. We use it to gain the
// displacements of the solid in the x- and y-directions.
template <int dim>
double FracturePhaseFieldProblem<dim>::compute_point_value(
    const DoFHandler<dim> &dofh, const LA::MPI::BlockVector &vector,
    const Point<dim> &p, const unsigned int component) const {
  double value = -1e100;
  Assert(component < dofh.get_fe().n_components(), ExcInternalError());
  try {
    Vector<double> tmp_vector(dofh.get_fe().n_components());
    VectorTools::point_value(dofh, vector, p, tmp_vector);
    value = tmp_vector(component);
  } catch (typename VectorTools::ExcPointNotAvailableHere &e) {
  }

  return Utilities::MPI::max(value, mpi_com);
}

template <int dim> void FracturePhaseFieldProblem<dim>::compute_point_stress() {
  // Evaluation point
  Point<dim> p1(0.0, 2.0);

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  // first find the cell in which this point
  // is, initialize a quadrature rule with
  // it, and then a FEValues object
  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>
      cell_point = GridTools::find_active_cell_around_point(
          StaticMappingQ1<dim>::mapping, dof_handler, p1);

  double value = 0.0;
  if (!cell_point.first->is_artificial()) {
    const Quadrature<dim> quadrature(
        GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients);
    fe_values.reinit(cell_point.first);

    std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
        1, std::vector<Tensor<1, dim>>(dim + 1));

    fe_values.get_function_gradients(rel_solution, old_solution_grads);

    // Compute stress of y-comp into y-direction
    value = -1.0 * old_solution_grads[0][1][1];
  }

  pcout << " PStress: " << Utilities::MPI::max(value, mpi_com);
}

template <int dim> void FracturePhaseFieldProblem<dim>::compute_cod_array() {
  // we want to integrate along dim-1 dim faces along the x axis
  // for this we fill buckets representing slices orthogonal to the x axis
  const unsigned int n_buckets = 75;
  std::vector<double> values(n_buckets);
  std::vector<double> volume(n_buckets);

  std::vector<double> exact(n_buckets);
  for (unsigned int i = 0; i < n_buckets; ++i) {
    double x = bucket_to_value(i, n_buckets);
    exact[i] = 1.92e-3 * std::sqrt(std::max(0.0, 1.0 - x * x));
  }

  // this yields 100 quadrature points evenly distributed in the interior of the
  // cell. We avoid points on the faces, as they would be counted more than
  // once.
  const unsigned int n_reps =
      50 + 100.0 * min_cell_diameter / (1.0 / n_buckets);
  const QIterated<dim> quadrature_formula(QMidpoint<1>(), 100);
  const unsigned int n_q_points = quadrature_formula.size();

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  std::vector<Vector<double>> solution_values(n_q_points,
                                              Vector<double>(dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> solution_grads(
      n_q_points, std::vector<Tensor<1, dim>>(dim + 1));

  const double width =
      bucket_to_value(1, n_buckets) - bucket_to_value(0, n_buckets);

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {

      fe_values.reinit(cell);

      fe_values.get_function_values(rel_solution, solution_values);
      fe_values.get_function_gradients(rel_solution, solution_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const int idx_ =
            value_to_bucket(fe_values.quadrature_point(q)[0], n_buckets);
        if (idx_ < 0 || idx_ >= static_cast<int>(n_buckets))
          continue;
        const unsigned int idx = static_cast<unsigned int>(idx_);

        const Tensor<1, dim> u = Tensors::get_u<dim>(q, solution_values);

        const Tensor<1, dim> grad_pf =
            Tensors::get_grad_pf<dim>(q, solution_grads);

        double cod_value =
            // Motivated by Bourdin et al. (2012); SPE Paper
            u * grad_pf;

        values[idx] += cod_value * fe_values.JxW(q);
        volume[idx] += fe_values.JxW(q);
      }
    }

  std::vector<double> values_all(n_buckets);
  std::vector<double> volume_all(n_buckets);
  Utilities::MPI::sum(values, mpi_com, values_all);
  Utilities::MPI::sum(volume, mpi_com, volume_all);
  for (unsigned int i = 0; i < n_buckets; ++i)
    values[i] = values_all[i] / width / 2.0;

  double middle_value = compute_cod(0.0);

  if (Utilities::MPI::this_mpi_process(mpi_com) == 0) {
    static unsigned int no = 0;
    ++no;
    std::ostringstream filename;
    filename << "cod-" << Utilities::int_to_string(no, 2) << ".txt";
    pcout << "writing " << filename.str() << std::endl;
    std::ofstream f(filename.str().c_str());

    double error = 0.0;
    for (unsigned int i = 0; i < n_buckets; ++i) {
      error += std::pow(values[i] - exact[i], 2.0);
      f << bucket_to_value(i, n_buckets) << " " << values[i] << " " << exact[i]
        << std::endl;
    }
    error = std::sqrt(error);
    double err_middle = std::abs(middle_value - 3.84e-4);
    pcout << "ERROR: " << error << " alpha_eps: " << alpha_eps
          << " k: " << constant_k << " hmin: " << min_cell_diameter
          << " errmiddle: " << err_middle << " dofs: " << dof_handler.n_dofs()
          << std::endl;
  }
}

template <int dim>
double FracturePhaseFieldProblem<dim>::compute_cod(const double eval_line) {

  const QGauss<dim - 1> face_quadrature_formula(3);
  FEFaceValues<dim> fe_face_values(
      fe, face_quadrature_formula,
      update_values | update_quadrature_points | update_gradients |
          update_normal_vectors | update_JxW_values);

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Vector<double>> face_solution_values(n_face_q_points,
                                                   Vector<double>(dim + 1));
  std::vector<std::vector<Tensor<1, dim>>> face_solution_grads(
      n_face_q_points, std::vector<Tensor<1, dim>>(dim + 1));

  double cod_value = 0.0;
  double eps = 1.0e-8;

  unsigned int n_faces = 0;

  Tensor<1, dim> n; // normal of the evaluation line
  n[0] = 1.0;

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      const double cell_x = cell->center()[0];
      if (cell_x - cell->diameter() > eval_line ||
          cell_x + cell->diameter() < eval_line)
        continue; // skip cells that are surely too far away

      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face) {
        fe_face_values.reinit(cell, face);
        if (fabs(fe_face_values.normal_vector(0) * n) < 0.5)
          continue; // skip faces not perpendicular to evaluation line

        fe_face_values.get_function_values(rel_solution, face_solution_values);
        fe_face_values.get_function_gradients(rel_solution,
                                              face_solution_grads);

        if ((fe_face_values.quadrature_point(0)[0] < (eval_line + eps)) &&
            (fe_face_values.quadrature_point(0)[0] > (eval_line - eps))) {
          ++n_faces;

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {

            {
              const Tensor<1, dim> u =
                  Tensors::get_u<dim>(q_point, face_solution_values);

              const Tensor<1, dim> grad_pf =
                  Tensors::get_grad_pf<dim>(q_point, face_solution_grads);

              // Motivated by Bourdin et al. (2012); SPE Paper
              cod_value += 0.5 * u * grad_pf * fe_face_values.JxW(q_point);
            }
          }
        }
      }
    }

  // divide by two, because we count each face twice:
  cod_value = Utilities::MPI::sum(cod_value, mpi_com) / 2.0;

  const unsigned int global_n_faces = Utilities::MPI::sum(n_faces, mpi_com);
  if (global_n_faces == 0)
    return -1e300;

  pcout << eval_line << "  " << cod_value << std::endl;

  return cod_value;
}

template <int dim> void FracturePhaseFieldProblem<dim>::compute_tcv() {
  // compute the total crack volume = TCV = \int_\Omega u \cdot \nabla \phi

  const QGauss<dim> quadrature(fe.degree + 2);
  const unsigned int n_q_points = quadrature.size();

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  std::vector<Tensor<1, dim>> displacement_values(n_q_points);
  std::vector<Tensor<1, dim>> phase_field_grads(n_q_points);

  double local_integral = 0.0;

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values[introspection.extractors.displacement].get_function_values(
          rel_solution, displacement_values);
      fe_values[introspection.extractors.phase_field].get_function_gradients(
          rel_solution, phase_field_grads);

      for (unsigned int q = 0; q < n_q_points; ++q)
        local_integral +=
            displacement_values[q] * phase_field_grads[q] * fe_values.JxW(q);
    }

  double tcv = Utilities::MPI::sum(local_integral, mpi_com);

  double ref;
  {
    double l_0 = 1.0;
    double E = 1.0;
    double p = func_pressure.value(Point<1>(time), 0); // 1e-3 in benchmark
    double nu = poisson_ratio_nu;                      // 0.2 in benchmark

    if (dim == 2)
      ref = 2.0 * p * l_0 * l_0 * (1.0 - nu * nu) * numbers::PI /
            E; // = 0.00603186
    else
      ref = 16.0 * p * l_0 * l_0 * l_0 * (1.0 - nu * nu) / E / 3.0; // = 0.00512
  }

  pcout << "TCV: value= " << tcv << " exact= " << ref
        << " error= " << std::abs(tcv - ref) << std::endl;
  statistics.add_value("TCV", tcv);
  statistics.set_precision("TCV", 8);
  statistics.set_scientific("TCV", true);
}

template <int dim> double FracturePhaseFieldProblem<dim>::compute_energy() {
  // What are we computing? In Latex-style it is:
  // bulk energy = [(1+k)phi^2 + k] psi(e)
  // crack energy = \frac{G_c}{2}\int_{\Omega}\Bigl( \frac{(\varphi -
  // 1)^2}{\eps}
  //+ \eps |\nabla \varphi|^2 \Bigr) \, dx
  double local_bulk_energy = 0.0;
  double local_crack_energy = 0.0;

  const QGauss<dim> quadrature_formula(fe.degree + 2);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  std::vector<double> phase_field_values(n_q_points);
  std::vector<Tensor<1, dim>> phase_field_grads(n_q_points);
  std::vector<Tensor<2, dim>> displacement_grads(n_q_points);

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);

      // update lame coefficients based on current cell
      if (test_case == TestCase::multiple_het) {
        E_modulus = func_emodulus->value(cell->center(), 0);

        lame_coefficient_mu = E_modulus / (2.0 * (1 + poisson_ratio_nu));

        lame_coefficient_lambda = (2 * poisson_ratio_nu * lame_coefficient_mu) /
                                  (1.0 - 2 * poisson_ratio_nu);
      }

      fe_values[introspection.extractors.phase_field].get_function_values(
          rel_solution, phase_field_values);
      fe_values[introspection.extractors.phase_field].get_function_gradients(
          rel_solution, phase_field_grads);
      fe_values[introspection.extractors.displacement].get_function_gradients(
          rel_solution, displacement_grads);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const Tensor<2, dim> grad_u = displacement_grads[q];
        const Tensor<1, dim> grad_pf = phase_field_grads[q];

        const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);

        const double pf = phase_field_values[q];

        const double tr_e_2 = trace(E * E);

        const double psi_e = 0.5 * lame_coefficient_lambda * tr_E * tr_E +
                             lame_coefficient_mu * tr_e_2;

        local_bulk_energy += ((1 + constant_k) * pf * pf + constant_k) * psi_e *
                             fe_values.JxW(q);

        local_crack_energy += G_c / 2.0 *
                              ((pf - 1) * (pf - 1) / alpha_eps +
                               alpha_eps * scalar_product(grad_pf, grad_pf)) *
                              fe_values.JxW(q);
      }
    }

  double bulk_energy = Utilities::MPI::sum(local_bulk_energy, mpi_com);
  double crack_energy = Utilities::MPI::sum(local_crack_energy, mpi_com);

  pcout << "No " << timestep_number << " time " << time
        << " bulk energy: " << bulk_energy << " crack energy: " << crack_energy;
  statistics.add_value("Bulk Energy", bulk_energy);
  statistics.set_precision("Bulk Energy", 8);
  statistics.set_scientific("Bulk Energy", true);
  statistics.add_value("Crack Energy", crack_energy);
  statistics.set_precision("Crack Energy", 8);
  statistics.set_scientific("Crack Energy", true);

  return 0;
}

// Evaluate the COD at different lines -1.0=x_0< x_1 < ... < x_N = 1.0
template <int dim>
void FracturePhaseFieldProblem<dim>::compute_functional_values() {
  static unsigned int no = 0;
  ++no;
  std::ostringstream filename;
  filename << "cod-" << Utilities::int_to_string(no, 2) << "b.txt";
  pcout << "writing " << filename.str() << std::endl;

  std::ofstream f(filename.str().c_str());

  const unsigned int N = 16 * 16;
  const double dx = 1.0 / N;
  for (unsigned int i = 0; i <= 3 * N; ++i) {
    const double x = -1.5 + i * dx;
    double value = compute_cod(x);
    if (value > -1e100)
      f << x << " " << value << std::endl;
  }
}

template <int dim> void FracturePhaseFieldProblem<dim>::compute_load() {
  const QGauss<dim - 1> face_quadrature_formula(3);
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_gradients |
                                       update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<std::vector<Tensor<1, dim>>> face_solution_grads(
      n_face_q_points, std::vector<Tensor<1, dim>>(dim + 1));

  Tensor<1, dim> load_value;

  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  const Tensor<2, dim> Identity = Tensors::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 3) {
          fe_face_values.reinit(cell, face);
          fe_face_values.get_function_gradients(rel_solution,
                                                face_solution_grads);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
            const Tensor<2, dim> grad_u =
                Tensors::get_grad_u<dim>(q_point, face_solution_grads);

            const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
            const double tr_E = trace(E);

            Tensor<2, dim> stress_term;
            stress_term = lame_coefficient_lambda * tr_E * Identity +
                          2 * lame_coefficient_mu * E;

            load_value += stress_term * fe_face_values.normal_vector(q_point) *
                          fe_face_values.JxW(q_point);
          }
        } // end boundary 3 for structure
    }

  load_value[0] *= -1.0;

  if (test_case == TestCase::miehe_tension) {
    double load_y = Utilities::MPI::sum(load_value[1], mpi_com);
    pcout << "  Load y: " << load_y;
    statistics.add_value("Load y", load_y);
    statistics.set_precision("Load y", 8);
    statistics.set_scientific("Load y", true);
  } else if (test_case == TestCase::miehe_shear) {
    double load_x = Utilities::MPI::sum(load_value[0], mpi_com);
    pcout << "  Load x: " << load_x;
    statistics.add_value("Load x", load_x);
    statistics.set_precision("Load x", 8);
    statistics.set_scientific("Load x", true);
  } else if (test_case == TestCase::three_point_bending) {
    load_value[1] *= -1.0;
    double load = Utilities::MPI::sum(load_value[1], mpi_com);
    pcout << "  P11: " << load;
    statistics.add_value("Load P11", load);
    statistics.set_precision("Load P11", 8);
    statistics.set_scientific("Load P11", true);
  }
}

// Determine the phase-field regularization parameters
// eps and kappa
template <int dim>
void FracturePhaseFieldProblem<dim>::determine_mesh_dependent_parameters() {
  min_cell_diameter = 1.0e+300;
  {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {
        min_cell_diameter = std::min(cell->diameter(), min_cell_diameter);
      }

    min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, mpi_com);
  }

  // for this test we want to use the h that will be used at the end
  if (test_case == TestCase::miehe_tension ||
      test_case == TestCase::miehe_shear ||
      test_case == TestCase::multiple_homo ||
      test_case == TestCase::three_point_bending) {
    min_cell_diameter = 0.0;

    typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(0),
                                            endc = dof_handler.end(0);

    for (; cell != endc; ++cell) {
      min_cell_diameter = std::max(cell->diameter(), min_cell_diameter);
    }
    min_cell_diameter *=
        std::pow(2.0, -1.0 * (n_global_pre_refine + n_refinement_cycles +
                              n_local_pre_refine));
  }

  // Set additional runtime parameters, the
  // regularization parameters, which
  // are chosen dependent on the present mesh size
  // old relations (see below why now others are used!)

  bool h_and_eps_small_o = false;
  if (h_and_eps_small_o && test_case == TestCase::sneddon) {
    // Bourdin 1999 Image Segmentation gives ideas for
    // choice of parameters w.r.t. h
    // also used in Mikelic, Wheeler, Wick (ICES reports 13-15, 14-18)
    // However, this if-control statement is now redundant since
    // we also adapt the parameters in the parameter file.
    // I kept them because this specific relation was used
    // in our CMAME paper (2013).
    constant_k = 0.25 * std::sqrt(min_cell_diameter);
    alpha_eps = 0.5 * std::sqrt(constant_k);
  } else {
    FunctionParser<1> func;
    prm.enter_subsection("Problem dependent parameters");
    func.initialize("h", prm.get("K reg"), std::map<std::string, double>());
    constant_k = func.value(Point<1>(min_cell_diameter), 0);
    func.initialize("h", prm.get("Eps reg"), std::map<std::string, double>());
    alpha_eps = func.value(Point<1>(min_cell_diameter), 0);
    prm.leave_subsection();
  }

  // sanity check
  //   pcout << "****"
  //   << "h = " << min_cell_diameter
  //   << "k = " << constant_k
  //   << " alpha_eps = " << alpha_eps
  //   << std::endl;
}

template <int dim> bool FracturePhaseFieldProblem<dim>::refine_mesh() {
  LA::MPI::BlockVector relevant_solution(partition_relevant);
  relevant_solution = solution;

  if (refinement_strategy == RefinementStrategy::fixed_preref_sneddon) {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {

        for (unsigned int vertex = 0;
             vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
          Tensor<1, dim> cell_vertex = (cell->vertex(vertex));
          if (cell_vertex[0] <= 2.5 && cell_vertex[0] >= -2.5 &&
              cell_vertex[1] <= 1.25 && cell_vertex[1] >= -1.25) {
            cell->set_refine_flag();
            break;
          }
        }
      }
  } // end Sneddon
  else if (refinement_strategy ==
           RefinementStrategy::fixed_preref_miehe_tension) {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {

        for (unsigned int vertex = 0;
             vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
          Tensor<1, dim> cell_vertex = (cell->vertex(vertex));
          if (cell_vertex[0] <= 0.6 && cell_vertex[0] >= 0.0 &&
              cell_vertex[1] <= 0.55 && cell_vertex[1] >= 0.45) {
            cell->set_refine_flag();
            break;
          }
        }
      }
  } // end Miehe tension
  else if (refinement_strategy ==
           RefinementStrategy::fixed_preref_miehe_shear) {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {

        for (unsigned int vertex = 0;
             vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
          Tensor<1, dim> cell_vertex = (cell->vertex(vertex));
          if (cell_vertex[0] <= 0.6 && cell_vertex[0] >= 0.0 &&
              cell_vertex[1] <= 0.55 && cell_vertex[1] >= 0.0) {
            cell->set_refine_flag();
            break;
          }
        }
      }
  } // end Miehe shear
  else if (refinement_strategy == RefinementStrategy::phase_field_ref) {
    // refine if phase field < constant
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i != introspection.component_indices.phase_field)
            continue; // only look at phase field
          if (relevant_solution(local_dof_indices[i]) <
              value_phase_field_for_refinement) {
            cell->set_refine_flag();
            break;
          }
        }
      }
  } else if (refinement_strategy ==
             RefinementStrategy::phase_field_ref_three_point_top) {
    // refine if phase field < constant
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {
        // Miehe three point top boundary refinement
        for (unsigned int vertex = 0;
             vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex) {
          Assert(dim == 2, ExcNotImplemented());
          Tensor<1, dim> cell_vertex = (cell->vertex(vertex));
          if (cell_vertex[1] >= 1.75) {
            cell->set_refine_flag();
            break;
          }
        }

        // Phase-field
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i != introspection.component_indices.phase_field)
            continue; // only look at phase field
          if (relevant_solution(local_dof_indices[i]) <
              value_phase_field_for_refinement) {
            cell->set_refine_flag();
            break;
          }
        }
      }
  } else if (refinement_strategy == RefinementStrategy::global) {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        cell->set_refine_flag();
  } else if (refinement_strategy == RefinementStrategy::mix) {

    {
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                                .begin_active(),
                                                     endc = dof_handler.end();
      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

      for (; cell != endc; ++cell)
        if (cell->is_locally_owned()) {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
            const unsigned int comp_i = fe.system_to_component_index(i).first;
            if (comp_i != introspection.component_indices.phase_field)
              continue; // only look at phase field
            if (relevant_solution(local_dof_indices[i]) <
                value_phase_field_for_refinement) {
              cell->set_refine_flag();
              break;
            }
          }
        }
    }

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    std::vector<bool> component_mask(dim + 1, true);
    component_mask[dim] = false;

    // estimate displacement:
    KellyErrorEstimator<dim>::estimate(
        dof_handler, QGauss<dim - 1>(fe.degree + 2),
        std::map<types::boundary_id, const Function<dim> *>(),
        relevant_solution, estimated_error_per_cell, component_mask, 0, 0,
        triangulation.locally_owned_subdomain());

    // but ignore cells in the crack:
    {
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                                .begin_active(),
                                                     endc = dof_handler.end();
      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

      unsigned int idx = 0;
      for (; cell != endc; ++cell, ++idx)
        if (cell->refine_flag_set())
          estimated_error_per_cell[idx] = 0.0;
    }

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.0);
  }

  // limit level
  if (test_case != TestCase::sneddon) {
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() &&
          cell->level() ==
              static_cast<int>(n_global_pre_refine + n_refinement_cycles +
                               n_local_pre_refine))
        cell->clear_refine_flag();
  }

  // check if we are doing anything
  {
    bool refine_or_coarsen = false;
    triangulation.prepare_coarsening_and_refinement();

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() &&
          (cell->refine_flag_set() || cell->coarsen_flag_set())) {
        refine_or_coarsen = true;
        break;
      }

    if (Utilities::MPI::sum(refine_or_coarsen ? 1 : 0, mpi_com) == 0)
      return false;
  }

  std::vector<const LA::MPI::BlockVector *> x(3);
  x[0] = &relevant_solution;
  x[1] = &old_solution;
  x[2] = &old_old_solution;

  parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector>
      solution_transfer(dof_handler);

  solution_transfer.prepare_for_coarsening_and_refinement(x);

  triangulation.execute_coarsening_and_refinement();
  setup_system();

  LA::MPI::BlockVector tmp_v(partition);
  LA::MPI::BlockVector tmp_vv(partition);
  std::vector<LA::MPI::BlockVector *> tmp(3);
  tmp[0] = &solution;
  tmp[1] = &tmp_v;
  tmp[2] = &tmp_vv;

  solution_transfer.interpolate(tmp);
  old_solution = tmp_v;
  old_old_solution = tmp_vv;

  determine_mesh_dependent_parameters();
  return true;
}

// As usual, we have to call the run method.
template <int dim> void FracturePhaseFieldProblem<dim>::run() {
  pcout << "Running on " << Utilities::MPI::n_mpi_processes(mpi_com) << " cores"
        << std::endl;

  set_runtime_parameters();
  setup_system();
  determine_mesh_dependent_parameters();

  for (unsigned int i = 0; i < n_local_pre_refine; ++i) {
    pcout << "Prerefinement step with h= " << min_cell_diameter << std::endl;

    ConstraintMatrix constraints;
    constraints.close();

    if (test_case == TestCase::sneddon) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesSneddon<dim>(introspection.n_components,
                                    min_cell_diameter),
          solution);

    } else if (test_case == TestCase::multiple_homo) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesMultipleHomo<dim>(introspection.n_components,
                                         min_cell_diameter),
          solution);

    } else if (test_case == TestCase::multiple_het) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesMultipleHet<dim>(introspection.n_components,
                                        min_cell_diameter),
          solution);

    } else if (test_case == TestCase::miehe_shear ||
               test_case == TestCase::miehe_tension) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesTensionOrShear<dim>(introspection.n_components,
                                           min_cell_diameter),
          solution);
    } else if (test_case == TestCase::three_point_bending) {
      VectorTools::interpolate(
          dof_handler, InitialValuesNoCrack<dim>(introspection.n_components),
          solution);
    }
    refine_mesh();
  }

  if (n_local_pre_refine == 0)
    determine_mesh_dependent_parameters();

  AssertThrow(alpha_eps >= min_cell_diameter,
              ExcMessage("You need to pick eps >= h"));
  AssertThrow(constant_k < 1.0, ExcMessage("You need to pick K < 1"));

  pcout << "\n=============================="
        << "=====================================" << std::endl;
  pcout << "Parameters\n"
        << "==========\n"
        << "h (min):           " << min_cell_diameter << "\n"
        << "k:                 " << constant_k << "\n"
        << "eps:               " << alpha_eps << "\n"
        << "G_c:               " << G_c << "\n"
        << "gamma penal:       " << gamma_penal << "\n"
        << "Poisson nu:        " << poisson_ratio_nu << "\n"
        << "E modulus:         " << E_modulus << "\n"
        << "Lame mu:           " << lame_coefficient_mu << "\n"
        << "Lame lambda:       " << lame_coefficient_lambda << "\n"
        << std::endl;

  {
    ConstraintMatrix constraints;
    constraints.close();

    if (test_case == TestCase::sneddon) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesSneddon<dim>(introspection.n_components,
                                    min_cell_diameter),
          solution);
    } else if (test_case == TestCase::multiple_homo) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesMultipleHomo<dim>(introspection.n_components,
                                         min_cell_diameter),
          solution);
    } else if (test_case == TestCase::multiple_het) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesMultipleHet<dim>(introspection.n_components,
                                        min_cell_diameter),
          solution);

    } else if (test_case == TestCase::miehe_shear ||
               test_case == TestCase::miehe_tension) {
      VectorTools::interpolate(
          dof_handler,
          InitialValuesTensionOrShear<dim>(introspection.n_components,
                                           min_cell_diameter),
          solution);
    } else if (test_case == TestCase::three_point_bending) {
      VectorTools::interpolate(
          dof_handler, InitialValuesNoCrack<dim>(introspection.n_components),
          solution);
    } else
      AssertThrow(false, ExcNotImplemented());

    output_results();
  }

  // Normalize phase-field function between 0 and 1
  project_back_phase_field();

  const unsigned int output_skip = 1;
  unsigned int refinement_cycle = 0;
  double finishing_timestep_loop = 0;
  double tmp_timestep = 0.0;

  // Initialize old and old_old_solutions
  // old_old is needed for extrapolation for pf_extra to avoid pf^2 in
  // block(0,0)
  old_old_solution = solution;
  old_solution = solution;

  // Initialize old and old_old timestep sizes
  old_timestep = timestep;
  old_old_timestep = timestep;

  // Timestep loop
  do {

    {
      // begin timer
      TimerOutput::Scope t(timer, "Time step loop");

      double newton_reduction = 1.0;

      if (timestep_number > switch_timestep && switch_timestep > 0)
        timestep = timestep_size_2;

      tmp_timestep = timestep;
      old_old_timestep = old_timestep;
      old_timestep = timestep;

      // Compute next time step
      old_old_solution = old_solution;
      old_solution = solution;

    redo_step:
      pcout << std::endl;
      pcout << "\n=============================="
            << "=========================================" << std::endl;
      pcout << "Timestep " << timestep_number << ": " << time << " ("
            << timestep << ")"
            << "   "
            << "Cells: " << triangulation.n_global_active_cells() << "   "
            << "DoFs: " << dof_handler.n_dofs();
      pcout << "\n--------------------------------"
            << "---------------------------------------" << std::endl;

      pcout << std::endl;

      if (outer_solver == OuterSolverType::active_set) {
        time += timestep;
        do {
          // The Newton method can either stagnate or the linear solver
          // might not converge. To not abort the program we catch the
          // exception and retry with a smaller step.
          use_old_timestep_pf = false;
          try {
            newton_reduction = newton_active_set();

            break;

          } catch (SolverControl::NoConvergence &e) {
            pcout << "Solver did not converge! Adjusting time step to "
                  << timestep / 10 << std::endl;
          }

          pcout << "Taking old_timestep_pf" << std::endl;
          use_old_timestep_pf = true;
          solution = old_solution;

          if (test_case == TestCase::three_point_bending) {
            // three-point bending tests needs old_timestep_pf, but
            // doesn't converge if we cut the timestep, so just run
            // the nonlinear solver here (and crash if this again
            // fails)
            newton_reduction = newton_active_set();
            break;
          }

          // Time step cut
          time -= timestep;
          timestep = timestep / 10.0;
          time += timestep;

        } while (true);
      } else if (outer_solver == OuterSolverType::simple_monolithic) {
        // Increment time
        time += timestep;

        do {
          // The Newton method can either stagnate or the linear solver
          // might not converge. To not abort the program we catch the
          // exception and retry with a smaller step.
          use_old_timestep_pf = false;
          try {
            // Normalize phase-field function between 0 and 1
            project_back_phase_field();
            newton_reduction = newton_iteration(time);

            while (newton_reduction > upper_newton_rho) {
              use_old_timestep_pf = true;
              time -= timestep;
              timestep = timestep / 10.0;
              time += timestep;
              solution = old_solution;
              newton_reduction = newton_iteration(time);

              if (timestep < 1.0e-9) {
                pcout << "Timestep too small - taking step" << std::endl;
                break;
              }
            }

            break;

          } catch (SolverControl::NoConvergence &e) {
            pcout << "Solver did not converge! Adjusting time step."
                  << std::endl;
          }

          time -= timestep;
          solution = old_solution;
          timestep = timestep / 10.0;
          time += timestep;

        } while (true);

      } else
        AssertThrow(false, ExcNotImplemented());

      // Normalize phase-field function between 0 and 1
      // TODO: this function is not really needed any more
      project_back_phase_field();
      constraints_hanging_nodes.distribute(solution);

      if (test_case != TestCase::sneddon) {
        bool changed = refine_mesh();
        if (changed) {
          // redo the current time step
          pcout << "MESH CHANGED!" << std::endl;
          time -= timestep;
          solution = old_solution;
          goto redo_step;
          continue;
        }
      }

      // Set timestep to original timestep
      timestep = tmp_timestep;

      statistics.add_value("Timestep No", timestep_number);
      statistics.add_value("Time", time);
      statistics.add_value("DoFs", dof_handler.n_dofs());
      statistics.add_value("minimum cell diameter", min_cell_diameter);
      statistics.set_precision("minimum cell diameter", 8);
      statistics.set_scientific("minimum cell diameter", true);

      // Compute statistics and print them in a single line:
      {
        pcout << std::endl;
        compute_energy();

        if (test_case == TestCase::sneddon ||
            test_case == TestCase::multiple_homo ||
            test_case == TestCase::multiple_het) {
          // no extra statistics
        } else {
          compute_load();
          if (test_case == TestCase::three_point_bending)
            compute_point_stress();
        }
        pcout << std::endl;
      }

      // Write solutions
      if ((timestep_number % output_skip == 0))
        output_results();

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        std::ofstream stat_file((output_folder + "/statistics").c_str());
        statistics.write_text(
            stat_file,
            TableHandler::simple_table_with_separate_column_description);
        stat_file.close();
      }

      // is this the residual? rename variable if not
      LA::MPI::BlockVector residual(partition);
      residual = old_solution;
      residual.add(-1.0, solution);

      // Abbruchkriterium time step algorithm
      finishing_timestep_loop = residual.linfty_norm();
      if (test_case == TestCase::sneddon)
        pcout << "Timestep difference linfty: " << finishing_timestep_loop
              << std::endl;

      ++timestep_number;

      if (test_case == TestCase::sneddon && finishing_timestep_loop < 1.0e-5) {
        // compute_cod_array(); // very expensive
        compute_tcv();
        compute_functional_values();

        // Now we compare phi to our reference function
        {
          ExactPhiSneddon<dim> exact(introspection.n_components, alpha_eps);
          Vector<float> error(triangulation.n_active_cells());

          LA::MPI::BlockVector rel_solution(partition_relevant);
          rel_solution = solution;

          if (test_case == TestCase::sneddon) {
            ExactPhiSneddon<dim> exact(introspection.n_components, alpha_eps);
            ComponentSelectFunction<dim> value_select(dim, dim + 1); // phi
            VectorTools::integrate_difference(dof_handler, rel_solution, exact,
                                              error, QGauss<dim>(fe.degree + 2),
                                              VectorTools::L2_norm,
                                              &value_select);
          } else
            AssertThrow(false, ExcNotImplemented());

          const double local_error = error.l2_norm();
          const double L2_error = std::sqrt(
              Utilities::MPI::sum(local_error * local_error, mpi_com));
          pcout << "phi_L2_error: " << L2_error << " h: " << min_cell_diameter
                << std::endl;
          statistics.add_value("phi_L2_error", L2_error);
          statistics.set_precision("phi_L2_error", 8);
          statistics.set_scientific("phi_L2_error", true);
        }

        if (n_refinement_cycles == 0)
          break;

        --n_refinement_cycles;
        // timestep_number = 0;
        pcout << std::endl;
        pcout << "\n================== " << std::endl;
        pcout << "Refinement cycle " << refinement_cycle
              << "\n------------------ " << std::endl;

        refine_mesh();
        solution = 0;
        ++refinement_cycle;
        if (test_case == TestCase::sneddon) {
          VectorTools::interpolate(
              dof_handler,
              InitialValuesSneddon<dim>(introspection.n_components,
                                        min_cell_diameter),
              solution);
        } else if (test_case == TestCase::multiple_homo) {
          VectorTools::interpolate(
              dof_handler,
              InitialValuesMultipleHomo<dim>(introspection.n_components,
                                             min_cell_diameter),
              solution);

        } else if (test_case == TestCase::multiple_het) {
          VectorTools::interpolate(
              dof_handler,
              InitialValuesMultipleHet<dim>(introspection.n_components,
                                            min_cell_diameter),
              solution);

        } else
          VectorTools::interpolate(
              dof_handler,
              InitialValuesTensionOrShear<dim>(introspection.n_components,
                                               min_cell_diameter),
              solution);
      }

    } // end timer

  } while (timestep_number <= max_no_timesteps);

  pcout << std::endl;
  pcout << "Finishing time step loop: " << finishing_timestep_loop << std::endl;

  pcout << std::resetiosflags(std::ios::floatfield) << std::fixed;
  std::cout.precision(2);

  MPI_Barrier(MPI_COMM_WORLD);

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  pcout << "VMPEAK, Resident in kB: " << stats.VmSize << " " << stats.VmRSS
        << std::endl;
}

#endif // CRACKS_PFM_H
