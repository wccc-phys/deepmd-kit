// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(deepmd/uparam/dedn, ComputeDeepmdUparamDedn)
// clang-format on
#else

#ifndef LMP_COMPUTE_DEEPMD_UPARAM_DEDN_H
#define LMP_COMPUTE_DEEPMD_UPARAM_DEDN_H

#include "compute.h"
#include "pair_deepmd.h"

namespace LAMMPS_NS {

class ComputeDeepmdUparamDedn : public Compute {
 public:
  ComputeDeepmdUparamDedn(class LAMMPS*, int, char**);
  ~ComputeDeepmdUparamDedn() override;
  void init() override;
  double compute_scalar() override;

 private:
  enum SourceType { SRC_VAR, SRC_COMPUTE, SRC_FIX };

  SourceType source_type;
  std::string source_id;
  int source_index;
  double delta;
  PairDeepMD* pair;

  double get_source_value();
};

}  // namespace LAMMPS_NS

#endif
#endif
