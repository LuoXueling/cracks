# Notes on Codes

## `cracks.cc`

几个`include`，之后分别介绍。

```c++
#ifndef SOURCE_DIR
#define SOURCE_DIR ""
#endif

#define CATCH_CONFIG_RUNNER

#include "dealii_includes.h"
#include "contrib/catch.hpp"
#include "pfm.h"
#include <sstream>
#include <sys/stat.h> // for mkdir
```

`dealii`所使用的命名空间

```c++
using namespace dealii;
```

MPI和PETSc需要进行初始化，其中，参数`1`代表在一个节点上运行。[链接](https://www.dealii.org/developer/doxygen/deal.II/step_17.html#Thecodemaincodefunction)

```c++
int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
```

`deallog`默认不输出前缀，用`depth_console`可以指定前缀长度。[链接](https://www.dealii.org/developer/doxygen/deal.II/step_3.html#Thecodemaincodefunction)

```c++
  try {
    deallog.depth_console(0);
```

`ParameterHandler`是dealii提供的参数读取库，`declare_parameters`通过引用方式设定参数。

```c++
    ParameterHandler prm;
    FracturePhaseFieldProblem<2>::declare_parameters(prm);
```

如果输入了参数路径，则将路径传递给`prm`

```c++
    if (argc > 1) {
      prm.parse_input(argv[1]);
```

在使用MPI时，每个进程都会有一个$[0,N-1]$的编号，由`Utilities::MPI::this_mpi_process`给出，在输出某种信息时，可以选择某个进程输出。[链接](https://www.dealii.org/developer/doxygen/deal.II/step_17.html#ElasticProblemElasticProblem)

```c++
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
```

从Global parameters中获取输出路径，用于记录参数

```c++
        prm.enter_subsection("Global parameters");
        const std::string output_folder = prm.get("Output directory");
        prm.leave_subsection();
```

创建输出文件夹并记录参数

```c++
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
```

当所有进程都调用了`MPI_Barrier`之后，每个进程才会继续运行，否则先调用的进程会原地等待。[链接](https://www.mpich.org/static/docs/latest/www3/MPI_Barrier.html)

```c++
      MPI_Barrier(MPI_COMM_WORLD);
```

如果没有输入参数，则终止程序，并提示输入

```c++
    } else {
      std::ofstream out("default.prm");
      prm.print_parameters(out, ParameterHandler::Text);
      std::cout << "usage: ./cracks <parameter_file>" << std::endl
                << " (created default.prm)" << std::endl;
      return 0;
    }
```

获取维度

```c++
    prm.enter_subsection("Global parameters");
    unsigned int problem_dimension = prm.get_integer("Dimension");
    prm.leave_subsection();
```

让第一个进程输出维度

```c++
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Problem dimension: " << problem_dimension << std::endl;
```

根据维度选择类模板的参数，创建对象并运行程序。其中`EXcNotImplemented`是dealii给出的异常类型。

```c++
    if (problem_dimension == 2) {
      FracturePhaseFieldProblem<2> fracture_problem(prm);
      fracture_problem.run();
    } else if (problem_dimension == 3) {
      FracturePhaseFieldProblem<3> fracture_problem(prm);
      fracture_problem.run();
    } else
      AssertThrow(false, ExcNotImplemented());
```

运行时的异常捕获，及程序返回

```c++
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
```

## `pfm.h`

在这里，将定义`FracturePhaseFieldProblem`类，这与dealii Tutorial的程序结构是一致的。

### `FracturePhaseFieldProblem` 声明

对于维度无关的编程，需要使用类模板。[链接](https://www.dealii.org/developer/doxygen/deal.II/step_4.html#ThecodeStep4codeclasstemplate)

```c++
template <int dim>
class FracturePhaseFieldProblem {
```

#### 变量

下面一一介绍私有变量内容。

##### `MPI_Comm`

是Message Passing Interface(MPI)的信息交换器，[链接](https://www.dealii.org/developer/doxygen/deal.II/DEALGlossary.html#GlossMPICommunicator)

```c++
  MPI_Comm mpi_com;
```

##### `  ParameterHandler`

由dealii给出的参数读取类，这里以指针形式给出。[链接](https://www.dealii.org/developer/doxygen/deal.II/classParameterHandler.html)

```c++
  ParameterHandler &prm;
```

##### `Introspection<dim>`

这是用于设定边界条件的类，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_44.html#Solidmake_constraints)，能够提取设定边界条件的边界上的值。其中包含`ComponentMask`, `ComponentIndices`, `FEValuesExtractors`

```c++
  Introspection<dim> introspection;
```

##### `parallel::distributed::Triangulation<dim> `

在[step-8](https://www.dealii.org/developer/doxygen/deal.II/step_8.html#ThecodeElasticProblemcodeclassimplementation)和[step-17](https://www.dealii.org/developer/doxygen/deal.II/step_17.html#ThecodeElasticProblemcodeclasstemplate)中，都直接使用`Triangulation triangulation;`定义网格，但这意味着每个进程都会保存一份完整的网格，在这里使用`parallel::distributed`(p4est)给出的网格，使每个进程中只保存自己拥有的单元及ghost单元(即包围前者的单元)。[链接](https://www.dealii.org/developer/doxygen/deal.II/step_40.html#Intro)

```c++
  parallel::distributed::Triangulation<dim> triangulation;
```

##### `FESystem`

由于弹性体问题是基于向量的(每个vertex上有两个值)，因此使用`FESystem`代替[step-2](https://www.dealii.org/developer/doxygen/deal.II/step_2.html)中的`FE_Q`，实际上`FESystem`是多个`FE_Q`的组合。[链接](https://www.dealii.org/developer/doxygen/deal.II/step_8.html#ThecodeElasticProblemcodeclasstemplate)

```c++
  FESystem<dim> fe;
```

##### `DoFHandler`

这是自由度处理的类，能够处理单元节点编号相关，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_2.html#CreationofaDoFHandler)。可以在并行环境下运行，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_40.html#LaplaceProblemsetup_system)

```c++
  DoFHandler<dim> dof_handler;
```

##### `ConstraintMatrix`

实际上是`dealii::AffineConstraints<double>`，不知道为什么会这样处理一下(`dealii_includes.h`中`using ConstraintMatrix = dealii::AffineConstraints<double>;`)

`AffineConstraints`是用来处理`hanging node`的类，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_6.html#Howtodealwithhangingnodesinpractice)，关于hanging node[链接](https://www.dealii.org/developer/doxygen/deal.II/group__constraints.html#ga3b4ea7dfd313e388d868c4e4aa685799)

```c++
  ConstraintMatrix constraints_update;
  ConstraintMatrix constraints_hanging_nodes;
```

##### `BlockSparseMatrix` and `BlockVector`

用于保存矩阵、向量，包括求解结果、残差等。这是专为"[Block system](https://www.dealii.org/developer/doxygen/deal.II/step_44.html)"设计的类，我还没理解。

```c++
  LA::MPI::BlockSparseMatrix system_pde_matrix;
  LA::MPI::BlockVector solution, newton_update, old_solution, old_old_solution,
      system_pde_residual;
  LA::MPI::BlockVector system_total_residual;

  LA::MPI::BlockVector diag_mass, diag_mass_relevant;
```

在分块计算中，Precondition也是非常困难的问题，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_40.html#LaplaceProblemsolve)。

```c++
  LA::MPI::PreconditionAMG preconditioner_solid;
  LA::MPI::PreconditionAMG preconditioner_phase_field;
```

`LA`在`dealii_includes.h`中

```c++
namespace LA {
using namespace dealii::LinearAlgebraTrilinos;
}
```

##### ConditionalOStream

这是为多线程设计的输出流，只有第一个进程会进行输出，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_17.html#ThecodeElasticProblemcodeclasstemplate)

```c++
  ConditionalOStream pcout;
```

##### `TimerOutput`

可以方便地对各模块计时，[step-40](https://www.dealii.org/developer/doxygen/deal.II/step_40.html)中有使用。[链接](https://www.dealii.org/developer/doxygen/deal.II/classTimerOutput.html)

```c++
  TimerOutput timer;
```

##### IndexSet

能够给出该线程拥有的自由度、单元等编号信息。

```c++
  IndexSet active_set;
```

##### Function

能返回(一组)标量或者(一组)向量的函数类。[链接](https://www.dealii.org/developer/doxygen/deal.II/classFunction.html)

```c++
  Function<dim> *func_emodulus;
```

##### std::vector

```c++
  std::vector<IndexSet> partition;
  std::vector<IndexSet> partition_relevant;

  std::vector<std::vector<bool>> constant_modes;
```

##### TableHandler

能够处理表格，tutorial中用`ConvergenceTable`代替了，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_7.html)

```c++
  TableHandler statistics;
```

##### 时间步相关变量

```c++
  // Global variables for timestepping scheme
  unsigned int timestep_number;
  unsigned int max_no_timesteps;
  double timestep, timestep_size_2, time;
  unsigned int switch_timestep;
  struct OuterSolverType {
    enum Enum { active_set, simple_monolithic };
  };
  typename OuterSolverType::Enum outer_solver;
```

##### 测试例枚举

```c++
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
```

##### 网格加密方式枚举

```c++
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
```

##### 其他变量

```c++
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
```

#### 函数

公有的函数为构造函数、运行函数和参数声明函数

```c++
public:
  FracturePhaseFieldProblem(ParameterHandler &);

  void run();

  static void declare_parameters(ParameterHandler &prm);
```

##### 构造函数

```c++
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
```

##### `setup_mesh`

首先定义`mesh_info`，设定读取网格或生成网格的命令

```c++
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

```

生成网格，其中`is`记录了网格信息，输出给生成网格的参数。对"rect"，需要生成矩形，`p1` `p2`是对角线上的点，`repetition`是第i个维度上有j个单元，使用了`GridGenerator`，[链接](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html#ac76417d7404b75cf53c732f456e6e971)

`GridIn`则是从ucd或者inp文件中读取，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_5.html#Step5run)

```c++
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
```

三点弯需要特别设置边界

```c++
  if (test_case == TestCase::three_point_bending) {
    // adjust boundary conditions
    double eps_machine = 1.0e-10;
```

网格加密时，母单元并不会删除，而是失效。实际作用的单元/面是"active"的。dealii中，网格、节点、面的循环变量都可以由迭代器给出

```c++
    typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
```

在dealii中，每个边界有一个id，初始值为0，用于设置边界条件。网格加密时，母单元的边界id会继承给子单元，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_3.html#Step3assemble_system)。可以通过下面这种方法设置边界id，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_3.html#Possibilitiesforextensions)

```c++
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
```

##### `declare_parameters`

太长了不放了，可以参考[链接](https://www.dealii.org/developer/doxygen/deal.II/classParameterHandler.html)

##### `set_runtime_parameters`

包含参数设定、网格生成。

##### `setup_system`

初始化刚度矩阵，[链接](https://www.dealii.org/developer/doxygen/deal.II/classParameterHandler.html)

```c++
  system_pde_matrix.clear();
```

初始化DoFHandler，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_2.html#Intro)

```c++
  dof_handler.distribute_dofs(fe);
```

DoFRenumbering用于自由度重新编号，使得稀疏矩阵非零元素集中在对角线附近，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_2.html)。这里是将相场的自由度放在位移自由度后，作为blcok 1，实例:[链接](https://www.dealii.org/developer/doxygen/deal.II/step_22.html)，文档:[链接](https://www.dealii.org/developer/doxygen/deal.II/namespaceDoFRenumbering.html#a52c1941406d1ce2937e29a46edf111f4)

```c++
  std::vector<unsigned int> sub_blocks(introspection.n_components, 0);
  sub_blocks[introspection.component_indices.phase_field] = 1;
  DoFRenumbering::component_wise(dof_handler, sub_blocks);
```

PreconditionAMG 需要知道每个dof属于向量的哪个自由度，实例:[链接](https://www.dealii.org/developer/doxygen/deal.II/step_31.html#BoussinesqFlowProblembuild_stokes_preconditioner)，文档:[链接](https://www.dealii.org/developer/doxygen/deal.II/namespaceDoFTools.html#afc96893388fe1a55c6ae5ae19ba52c6d)

```c++
  constant_modes.clear();
  DoFTools::extract_constant_modes(
      dof_handler, introspection.component_masks.displacements, constant_modes);
```

根据`sub_blocks`计算每个向量(var)的总自由度数，这里相场自由度数`dofs_per_var[1]`是`dofs_per_var[0]`的一半

```c++
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
```

`partition`记录了每个变量的自由度起始和结束位置

```c++
  partition.clear();
  compatibility::split_by_block(dofs_per_block,
                                dof_handler.locally_owned_dofs(), partition);
```

`relevant_set`记录了当前线程的自由度起始和结束位置，[链接](https://www.dealii.org/developer/doxygen/deal.II/namespaceDoFTools.html#acad7e0841b9046eaafddc4c617ab1d9d)

```c++
  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_set);
```

`partition_relevant`记录了当前线程每个变量的自由度起始和结束位置

```c++
  partition_relevant.clear();
  compatibility::split_by_block(dofs_per_block, relevant_set,
                                partition_relevant);
```

`constraints_hanging_nodes` [链接](https://www.dealii.org/developer/doxygen/deal.II/group__constraints.html#ga3b4ea7dfd313e388d868c4e4aa685799)

```c++
  {
    constraints_hanging_nodes.clear();
    constraints_hanging_nodes.reinit(relevant_set);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_hanging_nodes);
    constraints_hanging_nodes.close();
  }
```

设定边界条件

```c++
  {
    constraints_update.clear();
    constraints_update.reinit(relevant_set);

    set_newton_bc();
    constraints_update.merge(constraints_hanging_nodes,
                             ConstraintMatrix::right_object_wins);
    constraints_update.close();
  }
```

使稀疏矩阵非零元素集中在对角线附近，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_2.html)。

```c++
  {
    TrilinosWrappers::BlockSparsityPattern csp(partition, mpi_com);

    DoFTools::make_sparsity_pattern(dof_handler, csp, constraints_update, false,
                                    Utilities::MPI::this_mpi_process(mpi_com));

    csp.compress();
    system_pde_matrix.reinit(csp);
  }
```

##### `assemble_system`

一些初始化，

```c++
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
```

从`func_pressure`中提取当前加载值

```c++
  const double current_pressure = func_pressure.value(Point<1>(time), 0);
```

`LA::MPI::BlockVector rel_solution(partition_relevant);`生成一个[链接](https://www.dealii.org/developer/doxygen/deal.II/classTrilinosWrappers_1_1MPI_1_1BlockVector.html#a0ce108a75cba1dd5e7670be554b08a8e)

```c++
  LA::MPI::BlockVector rel_solution(partition_relevant);
  rel_solution = solution;

  LA::MPI::BlockVector rel_old_solution(partition_relevant);
  rel_old_solution = old_solution;

  LA::MPI::BlockVector rel_old_old_solution(partition_relevant);
  rel_old_old_solution = old_old_solution;
```

生成积分对象，[链接](https://www.dealii.org/developer/doxygen/deal.II/step_3.html#Assemblingthematrixandrighthandsidevector)

```c++
  QGauss<dim> quadrature_formula(fe.degree + 2);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
```

一些矩阵初始化，注意需要使用`active_cell_iterator`

```c++
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
```

对每个单元进行更新

```c++
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
```

提取一些历史值，[链接](https://www.dealii.org/developer/doxygen/deal.II/classFEValuesViews_1_1Vector.html#a48550c939e5c77067ea1f5732c0b9d8d)

```c++
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
```

计算位移、相场和位移梯度、相场梯度

```c++
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
```

这里直接限制相场的范围

```c++
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
```

相场增量

```c++
          double pf_minus_old_timestep_pf_plus =
              std::max(0.0, pf - old_timestep_pf);
```

对相场做外插，还可以

```c++
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
```

准备位移

```c++
         const Tensor<2, dim> grad_u = old_displacement_grads[q];
          const Tensor<1, dim> grad_pf = old_phase_field_grads[q];

          const double divergence_u = Tensors ::get_divergence_u<dim>(grad_u);

          const Tensor<2, dim> Identity = Tensors ::get_Identity<dim>();
```

计算应变

```c++
          const Tensor<2, dim> E = 0.5 * (grad_u + transpose(grad_u));
          const double tr_E = trace(E);
```

计算应力，包括应力分解

```c++
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
```

如果只求residual，则跳过

```c++
          if (!residual_only)
```

每个分量都要单独求

```c++
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
```

```c++
              double pf_minus_old_timestep_pf_plus = 0.0;
              if ((pf - old_timestep_pf) < 0.0)
                pf_minus_old_timestep_pf_plus = 0.0;
              else
                pf_minus_old_timestep_pf_plus = phi_i_pf[i];
```

应变

```c++
              const Tensor<2, dim> E_LinU =
                  0.5 * (phi_i_grads_u[i] + transpose(phi_i_grads_u[i]));
              const double tr_E_LinU = trace(E_LinU);
```

$\nabla^2u$?

```c++
              const double divergence_u_LinU =
                  Tensors ::get_divergence_u<dim>(phi_i_grads_u[i]);
```

应力分解

```c++
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
```

刚度矩阵，第二个循环

```c++
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
```

如果是位移

```c++
                const unsigned int comp_j =
                    fe.system_to_component_index(j).first;
                if (comp_j < dim) {
```

```c++
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
```

注意，该论文求解的方程与最简单的方程大有不同

```c++
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
```

残差矩阵是类似的

```c++
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
```

组装、处理约束

```c++
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
```

在不同线程间组装，[链接](https://www.dealii.org/developer/doxygen/deal.II/classBlockMatrixBase.html#a74954a421ab950fef132131c2eb6b5f9)，[链接](https://www.dealii.org/developer/doxygen/deal.II/DEALGlossary.html#GlossCompress)

```
  if (residual_only)
    system_total_residual.compress(VectorOperation::add);
  else
    system_pde_matrix.compress(VectorOperation::add);

  system_pde_residual.compress(VectorOperation::add);
```

设定precondition

```c++
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
```

##### `assemble_nl_residual`

只对残差积分

```c++
// In this function we assemble the semi-linear
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the
// system matrix.
template <int dim> void FracturePhaseFieldProblem<dim>::assemble_nl_residual() {
  assemble_system(true);
}
```

##### `assemble_diag_mass_matrix`

顾名思义

```c++
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
```

##### `set_boundary_condition`

设定边界条件

```c++
// Here, we impose boundary conditions. If initial_step is true, these are
// non-zero conditions, otherwise they are homogeneous conditions as we solve
// the Newton system in update form.
template <int dim>
void FracturePhaseFieldProblem<dim>::set_boundary_conditions(
    const double time, const bool initial_step, ConstraintMatrix &constraints)
```

边界条件是用函数给出的，首先设定0函数

```c++
  compatibility::ZeroFunction<dim> f_zero(introspection.n_components);
```

对三种算例，所有边界都固定

```c++
  if (dim == 2) {
    if (test_case == TestCase::sneddon ||
        test_case == TestCase::multiple_homo ||
        test_case == TestCase::multiple_het) {
      for (unsigned int bc = 0; bc < 4; ++bc)
        VectorTools::interpolate_boundary_values(
            dof_handler, bc, f_zero, constraints,
            introspection.component_masks.displacements);
```

对miehe tension，下边界y固定，上边界受拉。`initial_step`指每个载荷步的第一次迭代。这里`BoundaryTensionTest`在`boundary.h`中给出，`BoundaryTensionTest.value`给出边界上两个分量上的值

```c++
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
```

miehe shear

```c++
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
```

三点弯，查找左下角和右下角

```c++
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
```

中点

```c++
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
```

三维

```c++
  else if (dim == 3) {
    for (unsigned int b = 0; b < 6; ++b)
      VectorTools::interpolate_boundary_values(
          dof_handler, b, f_zero, constraints,
          introspection.component_masks.displacements);
  }
}
```

##### `set_initial_bc`

```c++
template <int dim>
void FracturePhaseFieldProblem<dim>::set_initial_bc(const double time) {
  ConstraintMatrix constraints;
  set_boundary_conditions(time, true, constraints);
  constraints.close();
  constraints.distribute(solution);
}
```

##### `BlockDiagonalPreconditioner`

用于`solve`函数的

```c++
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
```

##### `solve`

```c++
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
```





