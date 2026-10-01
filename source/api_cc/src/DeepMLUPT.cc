// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "DeepMLUPT.h"

#include <cstdint>

#include "common.h"
#include "commonPT.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

// ---------------------------------------------------------------------------
// translate_error — catch PyTorch exceptions
// ---------------------------------------------------------------------------
void DeepMLUPT::translate_error(std::function<void()> f) {
  try {
    f();
  } catch (const c10::Error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  } catch (const std::runtime_error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  }
}

// ---------------------------------------------------------------------------
// ctor / dtor
// ---------------------------------------------------------------------------
DeepMLUPT::DeepMLUPT() : inited(false) {}

DeepMLUPT::DeepMLUPT(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false) {
  try {
    translate_error([&] { init(model, gpu_rank, file_content); });
  } catch (...) {
    throw;
  }
}

DeepMLUPT::~DeepMLUPT() {}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
void DeepMLUPT::init(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  deepmd::load_op_library(deepmd::DPBackend::PyTorch);
  int gpu_num = torch::cuda::device_count();
  gpu_id = (gpu_num > 0) ? (gpu_rank % gpu_num) : 0;
  gpu_enabled = torch::cuda::is_available();
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
    std::cout << "load MLU model from: " << model << " to cpu " << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif
    std::cout << "load MLU model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  std::unordered_map<std::string, std::string> metadata = {{"type", ""}};
  module = torch::jit::load(model, device, metadata);
  module.eval();

  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  if (num_inter_nthreads) {
    try {
      at::set_num_interop_threads(num_inter_nthreads);
    } catch (...) {
    }
  }
  if (num_intra_nthreads) {
    try {
      at::set_num_threads(num_intra_nthreads);
    } catch (...) {
    }
  }

  // Query model properties
  rcut = module.run_method("get_rcut").toDouble();
  ntypes = module.run_method("get_ntypes").toInt();
  dfparam = module.run_method("get_dim_fparam").toInt();

  // var_name — MLU models always output "uparam"
  var_name = module.run_method("get_var_name").toStringRef();
  task_dim_ = module.run_method("get_task_dim").toInt();

  if (var_name != "uparam") {
    throw deepmd::deepmd_exception("DeepMLU: unexpected var_name '" + var_name +
                                   "', expected 'uparam'");
  }
  if (task_dim_ != 1) {
    throw deepmd::deepmd_exception("DeepMLU: unexpected task_dim " +
                                   std::to_string(task_dim_) + ", expected 1");
  }
  if (module.find_method("get_intensive")) {
    intensive_ = module.run_method("get_intensive").toBool();
  } else {
    intensive_ = true;
  }

  inited = true;
}

// ---------------------------------------------------------------------------
// Simple accessors
// ---------------------------------------------------------------------------
double DeepMLUPT::cutoff() const { return rcut; }
int DeepMLUPT::numb_types() const { return ntypes; }
int DeepMLUPT::numb_types_spin() const { return 0; }
int DeepMLUPT::dim_fparam() const { return dfparam; }
int DeepMLUPT::dim_aparam() const { return 0; }
void DeepMLUPT::get_type_map(std::string& type_map) {
  auto type_map_result = module.run_method("get_type_map");
  auto type_map_list = type_map_result.toList();
  type_map.clear();
  for (const torch::IValue& element : type_map_list) {
    if (!type_map.empty()) {
      type_map += " ";
    }
    type_map += torch::str(element);
  }
}
bool DeepMLUPT::is_aparam_nall() const { return false; }
bool DeepMLUPT::has_default_fparam() const { return false; }
int DeepMLUPT::dim_uparam() const { return 0; }
bool DeepMLUPT::has_default_uparam() const { return false; }
const std::string& DeepMLUPT::get_var_name() const { return var_name; }
int DeepMLUPT::get_task_dim() const { return task_dim_; }

// ---------------------------------------------------------------------------
// compute — template implements both double and float
// ---------------------------------------------------------------------------
template <typename VALUETYPE>
void DeepMLUPT::compute(std::vector<VALUETYPE>& u_per_frame,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const int nghost,
                        const InputNlist& lmp_list,
                        const int ago,
                        const std::vector<VALUETYPE>& fparam) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }

  int nall = atype.size();
  auto floatType =
      std::is_same<VALUETYPE, float>::value ? torch::kFloat32 : torch::kFloat64;
  auto options = torch::TensorOptions().dtype(floatType);
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);

  // Select real atoms only, following DeepPotPT convention.  Ghost atoms are
  // handled by the LAMMPS communication layer; the model only needs to see
  // real atoms for predicting a per-frame scalar.
  std::vector<VALUETYPE> dcoord, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nframes = 1;
  // MLU has no aparam input — pass empty vector.
  std::vector<VALUETYPE> aparam;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, nframes, 0, nall, false);

  // coord: (1, nall_real, 3)
  at::Tensor coord_tensor =
      torch::from_blob(dcoord.data(), {1, nall_real, 3}, options).to(device);

  // atype: (1, nall_real)
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_options).to(device);

  // box: (1, 9)
  c10::optional<torch::Tensor> box_tensor;
  if (!box.empty()) {
    box_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options)
            .to(device);
  }

  // fparam: (1, dfparam)
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<int>(fparam.size())}, options)
            .to(device);
  }

  // Build neighbor list from LAMMPS nlist, mirroring DeepPotPT.
  nlist_data.copy_from_nlist(lmp_list, nall - nghost);
  nlist_data.shuffle_exclude_empty(fwd_map);
  nlist_data.padding();
  firstneigh_tensor =
      createNlistTensor(nlist_data.jlist).to(torch::kInt64).to(device);

  c10::optional<torch::Tensor> mapping_tensor;
  c10::optional<torch::Tensor> uparam_tensor;  // None — uparam is output
  c10::optional<torch::Tensor> aparam_tensor;  // None
  bool do_atomic_virial = false;
  c10::optional<torch::Tensor> chg_spin_tensor;

  // forward_lower — the model uses the LAMMPS nlist instead of building its
  // own, matching the training evaluation path.
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module
          .run_method("forward_lower", coord_tensor, atype_tensor,
                      firstneigh_tensor, mapping_tensor, fparam_tensor,
                      uparam_tensor, aparam_tensor, do_atomic_virial,
                      chg_spin_tensor)
          .toGenericDict();

  // Read the U output. In intensive (frame) mode use "uparam" (per-frame
  // reduced). In non-intensive (atomic) mode use "atom_uparam" (per-atom).
  std::string output_key = intensive_ ? var_name : "atom_" + var_name;
  c10::IValue uparam_out;
  if (outputs.contains(output_key)) {
    uparam_out = outputs.at(output_key);
  } else {
    throw deepmd::deepmd_exception(
        "DeepMLU: output dict missing key '" + output_key +
        "'. Available keys should include 'uparam' and 'atom_uparam'.");
  }
  torch::Tensor flat_uparam = uparam_out.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_uparam = flat_uparam.to(torch::kCPU);
  u_per_frame.assign(cpu_uparam.data_ptr<VALUETYPE>(),
                     cpu_uparam.data_ptr<VALUETYPE>() + cpu_uparam.numel());
}

// ---------------------------------------------------------------------------
// computew — public wrappers (translate_error → private template)
// ---------------------------------------------------------------------------
void DeepMLUPT::computew(std::vector<double>& u_per_frame,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const int nghost,
                         const InputNlist& lmp_list,
                         const int ago,
                         const std::vector<double>& fparam) {
  translate_error([&] {
    compute(u_per_frame, coord, atype, box, nghost, lmp_list, ago, fparam);
  });
}

void DeepMLUPT::computew(std::vector<float>& u_per_frame,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const int nghost,
                         const InputNlist& lmp_list,
                         const int ago,
                         const std::vector<float>& fparam) {
  translate_error([&] {
    compute(u_per_frame, coord, atype, box, nghost, lmp_list, ago, fparam);
  });
}

#endif  // BUILD_PYTORCH
