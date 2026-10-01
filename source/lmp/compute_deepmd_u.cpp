// SPDX-License-Identifier: LGPL-3.0-or-later
#include "compute_deepmd_u.h"

#include <cstring>

#include "error.h"
#include "force.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeDeepmdU::ComputeDeepmdU(LAMMPS* lmp, int narg, char** arg)
    : Compute(lmp, narg, arg), pair(nullptr) {
  if (narg != 3) {
    error->all(FLERR, "Illegal compute deepmd/u command");
  }
  scalar_flag = 1;
  extscalar = 1;
  timeflag = 1;
}

/* ---------------------------------------------------------------------- */

ComputeDeepmdU::~ComputeDeepmdU() = default;

/* ---------------------------------------------------------------------- */

void ComputeDeepmdU::init() {
  pair = dynamic_cast<PairDeepMD*>(force->pair);
  if (!pair) {
    error->all(FLERR, "compute deepmd/u requires pair_style deepmd");
  }
}

/* ---------------------------------------------------------------------- */

double ComputeDeepmdU::compute_scalar() {
  invoked_scalar = update->ntimestep;
  int dim = 0;
  double* u_ptr = (double*)pair->extract("u", dim);
  if (u_ptr) {
    scalar = *u_ptr;
  } else {
    scalar = 0.0;
  }
  return scalar;
}
