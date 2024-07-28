/**
* Xueling Luo @ Shanghai Jiao Tong University, 2022
* This code is for multiscale phase field fracture.
**/

#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "dealii_includes.h"
#include <iostream>

using namespace dealii;
struct PreconditionerCfg{
  PreconditionerCfg();

  std::vector<std::vector<bool>> constant_modes;
};

PreconditionerCfg::PreconditionerCfg() {
  constant_modes.clear();
}

#endif
