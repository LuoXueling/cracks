//
// Created by xlluo on 2022/3/26.
//

#ifndef CRACKS_INTROSPECTION_H
#define CRACKS_INTROSPECTION_H
#include "dealii_includes.h"
using namespace dealii;

template <int dim>
struct Introspection
{
    Introspection(ParameterHandler &prm);

    unsigned int displacement_degree;

    unsigned int n_components;
    unsigned int n_blocks;

    struct ComponentMasks
    {
        ComponentMask displacements;
        ComponentMask displacement[dim];
        ComponentMask phase_field;
    };
    ComponentMasks component_masks;

    struct ComponentIndices
    {
        unsigned int displacement[dim];
        unsigned int velocity[dim];
        unsigned int phase_field;
    };
    ComponentIndices component_indices;

    struct Extractors
    {
        FEValuesExtractors::Vector displacement;
        FEValuesExtractors::Vector velocity;
        FEValuesExtractors::Scalar phase_field;
    };
    Extractors extractors;

    std::vector<unsigned int> components_to_blocks;

    std::vector<const FiniteElement<dim,dim>*> fes;
    std::vector<unsigned int> multiplicities;

};

template <int dim>
Introspection<dim>::Introspection(ParameterHandler &prm)
{
    prm.enter_subsection("Global parameters");
    const unsigned int degree = prm.get_integer("FE degree");
    this->displacement_degree = degree;
    prm.leave_subsection();
    prm.enter_subsection("Solver parameters");
    const bool direct_solver = prm.get_bool("Use Direct Inner Solver");
    prm.leave_subsection();


    fes.push_back(new FE_Q<dim>(degree));
    multiplicities.push_back(dim);
    fes.push_back(new FE_Q<dim>(degree));
    multiplicities.push_back(1);

    n_components = dim + 1;
    if (direct_solver)
        n_blocks = 1;
    else
        n_blocks = 1 + 1;

    {
        unsigned int c = 0;
        for (unsigned int d=0; d<dim; ++d)
            component_indices.displacement[d] = c++;
        component_indices.phase_field = c++;
    }

    {
        component_masks.displacements = ComponentMask(n_components, false);
        for (unsigned int d=0; d<dim; ++d)
        {
            component_masks.displacement[d] = ComponentMask(n_components, false);
            component_masks.displacement[d].set(d, true);
            component_masks.displacements.set(d, true);
        }

        component_masks.phase_field = ComponentMask(n_components, false);
        component_masks.phase_field.set(component_indices.phase_field, true);
    }
    {
        extractors.displacement = FEValuesExtractors::Vector(component_indices.displacement[0]);
        extractors.phase_field = FEValuesExtractors::Scalar(component_indices.phase_field);
    }
    {
        components_to_blocks.resize(n_components, 0);
        unsigned int block = 0;
        block += direct_solver ? 0 : 1;
        components_to_blocks[component_indices.phase_field] = block;
    }
}


#endif //CRACKS_INTROSPECTION_H
