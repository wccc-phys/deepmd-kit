// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "DeepMLU.h"
#include "neighbor_list.h"

namespace deepmd {

/**
 * @brief PyTorch backend for DeepMLU.
 *
 * Loads a compiled MLU model (.pth) and evaluates the per-frame U
 * prediction each timestep.
 **/
class DeepMLUPT : public DeepMLUBackend {
 public:
  DeepMLUPT();
  DeepMLUPT(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");
  ~DeepMLUPT();

  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "") override;

  void computew(std::vector<double>& u_per_frame,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& lmp_list,
                const int ago,
                const std::vector<double>& fparam) override;

  void computew(std::vector<float>& u_per_frame,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& lmp_list,
                const int ago,
                const std::vector<float>& fparam) override;

  double cutoff() const override;
  int numb_types() const override;
  int numb_types_spin() const override;
  int dim_fparam() const override;
  int dim_aparam() const override;
  void get_type_map(std::string& type_map) override;
  bool is_aparam_nall() const override;
  bool has_default_fparam() const override;
  int dim_uparam() const override;
  bool has_default_uparam() const override;
  const std::string& get_var_name() const override;
  int get_task_dim() const override;

 private:
  /// Thunk that catches PyTorch exceptions and rethrows as deepmd_exception.
  void translate_error(std::function<void()> f);

  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& u_per_frame,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int ago,
               const std::vector<VALUETYPE>& fparam);

  bool inited;
  double rcut;
  int ntypes;
  int dfparam;
  std::string var_name;
  int task_dim_;
  bool intensive_{true};
  torch::jit::script::Module module;
  int gpu_id;
  bool gpu_enabled;
  int num_intra_nthreads, num_inter_nthreads;
  // Cached neighbor list data to avoid reallocation across timesteps.
  // However, since MLU does NOT compute forces, the neighbor list is only
  // used to build the atom graph for descriptor evaluation — we need the
  // nlist for the model forward_lower() call.
  NeighborListData nlist_data;
  at::Tensor firstneigh_tensor;
};

}  // namespace deepmd

#endif  // BUILD_PYTORCH
