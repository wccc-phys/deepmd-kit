// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(deepmd/u, ComputeDeepmdU)
// clang-format on
#else

#ifndef LMP_COMPUTE_DEEPMD_U_H
#define LMP_COMPUTE_DEEPMD_U_H

#include "compute.h"
#include "pair_deepmd.h"

namespace LAMMPS_NS {

class ComputeDeepmdU : public Compute {
 public:
  ComputeDeepmdU(class LAMMPS*, int, char**);
  ~ComputeDeepmdU() override;
  void init() override;
  double compute_scalar() override;

 private:
  PairDeepMD* pair;
};

}  // namespace LAMMPS_NS

#endif
#endif
