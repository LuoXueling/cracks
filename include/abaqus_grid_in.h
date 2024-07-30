/**
* Xueling Luo @ Shanghai Jiao Tong University, 2022
* This code is for multiscale phase field fracture.
**/

#ifndef ABAQUS_GRID_IN_H
#define ABAQUS_GRID_IN_H

#include "dealii_includes.h"
#include "utils.h"
using namespace dealii;

/**
 * Inherit from GridIn to extend its function.
 */
template <int dim, int spacedim = dim>
class AbaqusGridIn : public GridIn<dim, spacedim> {
public:
  void attach_triangulation(Triangulation<dim, spacedim> &t);
  void read_abaqus_inp(const std::string &input_file);
protected:
   SmartPointer<Triangulation<dim, spacedim>, GridIn<dim, spacedim>> tria;
};

template <int dim, int spacedim>
void AbaqusGridIn<dim, spacedim>::attach_triangulation(Triangulation<dim, spacedim> &t)
{
  tria = &t;
}

/**
 * @refitem https://www.dealii.org/developer/doxygen/deal.II/grid__in_8cc_source.html
 */
template <int dim, int spacedim>
void AbaqusGridIn<dim, spacedim>::read_abaqus_inp(
    const std::string &input_file) {
  std::ifstream in(input_file);

  unsigned int n_vertices = 0;
  unsigned int n_cells = 0;

  std::string line;

  /**
   * There is no n_vertices declaration in abaqus .inp file,
   * so dynamic vector should be used.
   */
  std::vector<Point<dim>> vertices;
  std::map<int, int> vertex_indices;

  std::vector<CellData<dim>> cells;
  SubCellData subcelldata;
  std::string cell_type;

  char del = ',';

  while (!in.eof()) {
    std::getline(in, line);
    if (contains(line, "*Node")) {

      /**
       * read nodes
       */
      while (!contains(line, "*Element")) {
        int vertex_number = n_vertices + 1;
        if (n_vertices == 0)
          std::getline(in, line, del);

        for (unsigned int d = 0; d < dim; ++d) {
          std::getline(in, line, del);
          vertices.push_back({});
          vertices[n_vertices](d) = std::atof(line.c_str());
        }
        vertex_indices[vertex_number] = n_vertices;
        n_vertices++;
      }

      /**
       * read elements
       */
      std::getline(in, cell_type, del);

      unsigned int cell_dim;
      if (contains(cell_type, "CPS4R") && dim == 2) {
        cell_dim = 2;
      } else if (contains(cell_type, "C3D4") && dim == 3) {
        cell_dim = 3;
      } else {
        AssertThrow(false, ExcMessage("Cell type not implemented or dim in "
                                      ".prm is incompatible with cell type."))
      }
      line = cell_type;
      while (!contains(line, "*End Part")) {
//        int cell_number = n_cells + 1;
        if (cell_dim == 2) {
          cells.emplace_back();
          for (const unsigned int i : GeometryInfo<dim>::vertex_indices()){
            std::getline(in, line, del);
            cells.back().vertices[GeometryInfo<dim>::ucd_to_deal[i]]=std::stoi(line);
          }
          cells.back().material_id = 0;

          /**
           * transform from ucd to
           * consecutive numbering
           */
          for (const unsigned int i : GeometryInfo<dim>::vertex_indices()) {
            if (vertex_indices.find(cells.back().vertices[i]) !=
                vertex_indices.end())
              // vertex with this index exists
              cells.back().vertices[i] =
                  vertex_indices[cells.back().vertices[i]];
            else {
              // no such vertex index
              AssertThrow(false, ExcMessage("ExcInvalidVertexIndex"));

              cells.back().vertices[i] = numbers::invalid_unsigned_int;
            }
          }
        }
        n_cells++;
      }
    }

  }
  Assert(subcelldata.check_consistency(dim), ExcInternalError());

  // do some clean-up on vertices...
  GridTools::delete_unused_vertices(vertices, cells, subcelldata);
  // ... and cells
  if (dim == spacedim)
    GridTools::invert_all_negative_measure_cells(vertices, cells);
  GridTools::consistently_order_cells(cells);

  tria->create_triangulation(vertices, cells, subcelldata);
}

#endif