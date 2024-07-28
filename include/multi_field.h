/**
* Xueling Luo @ Shanghai Jiao Tong University, 2022
* This code is for multiscale phase field fracture.
**/

#ifndef MULTI_FIELD_H
#define MULTI_FIELD_H

#include "dealii_includes.h"
#include "parameters.h"
using namespace dealii;

/**
 * @refitem tjhei/cracks
 *
 */
template <int dim> struct MultiFieldCfg {
  MultiFieldCfg(const Parameters::AllParameters &params);

  std::vector<const FiniteElement<dim, dim> *> FE_Q_sequence;
  std::vector<unsigned int> FE_Q_dim_sequence;

  std::vector<unsigned int> block_component;
  unsigned int n_components;
  unsigned int n_blocks;

  const unsigned int disp_dim = dim;

  struct ComponentMasks {
    ComponentMask displacements;
    ComponentMask displacement[dim];
    ComponentMask phase_field;
  };
  ComponentMasks component_masks;

  struct ComponentIndices {
    unsigned int displacement[dim];
    unsigned int phase_field;
  };
  ComponentIndices component_indices;

  struct Extractors {
    FEValuesExtractors::Vector displacement;
    FEValuesExtractors::Vector velocity;
    FEValuesExtractors::Scalar phase_field;
  };
  Extractors extractors;

  std::vector<unsigned int> components_to_blocks;

  int total_components;
  int disp_start_component;
  int phase_start_component;
};

template <int dim>
MultiFieldCfg<dim>::MultiFieldCfg(const Parameters::AllParameters &params)
    : disp_start_component(0), phase_start_component(dim) {

  FE_Q_sequence.push_back(new FE_Q<dim>(params.poly_degree));
  FE_Q_dim_sequence.push_back(disp_dim);

  if (params.enable_phase_field){FE_Q_sequence.push_back(new FE_Q<dim>(params.poly_degree));
    FE_Q_dim_sequence.push_back(1);}

  if (params.enable_phase_field) {
    n_components = disp_dim + 1;
    total_components = disp_dim + 1;

    // block_component[i]=j means the ith component belongs to the jth field.
    block_component = std::vector<unsigned int>(disp_dim+1, 0);
    block_component[phase_start_component] = 1;
  }
  else{
    n_components = disp_dim;
    total_components = disp_dim;
    block_component = std::vector<unsigned int>(disp_dim, 0);
  }
  n_blocks = 1;

  /**
   * @code
   * component_indices.displacement[i] = i for i=0,...,dim-1
   * component_indices.phase_field = dim
   * @endcode
   */
  {
    for (unsigned int d = 0; d < disp_dim; ++d) {
      component_indices.displacement[d] = d;
    }
    component_indices.phase_field = phase_start_component;
  }
  /**
   * @code
   * component_masks.displacement[i][j] == true iff i==j and j<dim
   * component_masks.displacement[i] == true iff i==0,...,dim-1
   * component_masks.phase_field[i][j] == true iff i==j==dim
   * component_masks.phase_field[i] == true iff i==dim
   * @endcode
   **/
  {
    component_masks.displacements = ComponentMask(total_components, false);
    for (unsigned int d = 0; d < disp_dim; ++d) {
      component_masks.displacement[d] = ComponentMask(total_components, false);
      component_masks.displacement[d].set(d, true);
      component_masks.displacements.set(d, true);
    }
    if (params.enable_phase_field) {
      component_masks.phase_field = ComponentMask(total_components, false);
      component_masks.phase_field.set(component_indices.phase_field, true);
    }
  }

  {
    extractors.displacement =
        FEValuesExtractors::Vector(component_indices.displacement[0]);
    extractors.phase_field =
        FEValuesExtractors::Scalar(component_indices.phase_field);
  }
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
//    prec_C.vmult(dst.block(1), src.block(1));
  }

  const LA::MPI::BlockSparseMatrix &matrix;
  const PreconditionerA &prec_A;
  const PreconditionerC &prec_C;
};

#endif
