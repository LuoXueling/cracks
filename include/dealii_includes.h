/**
* Xueling Luo @ Shanghai Jiao Tong University, 2022
* This code is for multiscale phase field fracture.
**/

#ifndef CRACKS_DEALII_INCLUDES_H
#define CRACKS_DEALII_INCLUDES_H


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#if DEAL_II_VERSION_GTE(9,1,0)
#  include <deal.II/lac/affine_constraints.h>
using ConstraintMatrix = dealii::AffineConstraints<double>;
#else
#  include <deal.II/lac/constraint_matrix.h>
#  include <deal.II/grid/tria_boundary_lib.h>
#endif

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>


using namespace dealii;

namespace compatibility {
/**
 * Split the set of DoFs (typically locally owned or relevant) in @p whole_set
 * into blocks given by the @p dofs_per_block structure.
 */
void split_by_block(const std::vector<types::global_dof_index> &dofs_per_block,
                    const IndexSet &whole_set,
                    std::vector<IndexSet> &partitioned) {
  const unsigned int n_blocks = dofs_per_block.size();
  partitioned.clear();

  partitioned.resize(n_blocks);
  types::global_dof_index start = 0;
  for (unsigned int i = 0; i < n_blocks; ++i) {
    partitioned[i] = whole_set.get_view(start, start + dofs_per_block[i]);
    start += dofs_per_block[i];
  }
}
} // namespace compatibility

namespace compatibility {
template <int dim> using ZeroFunction = dealii::Functions::ZeroFunction<dim>;
}

// Ref. step-40
namespace LA
{
#if defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#endif //CRACKS_DEALII_INCLUDES_H
