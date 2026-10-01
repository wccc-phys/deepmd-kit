// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepMLU.h"

#include <memory>

#include "BackendPlugin.h"
#include "common.h"

using namespace deepmd;

// ---------------------------------------------------------------------------
// DeepMLU (pimpl)
// ---------------------------------------------------------------------------
DeepMLU::DeepMLU() : DeepBaseModel(), mlu_backend(nullptr) {}

DeepMLU::DeepMLU(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content)
    : DeepBaseModel(), mlu_backend(nullptr) {
  init(model, gpu_rank, file_content);
}

DeepMLU::~DeepMLU() {}

void DeepMLU::init(const std::string& model,
                   const int& gpu_rank,
                   const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  const DPBackend backend = get_backend(model);
  if (deepmd::DPBackend::PyTorch == backend ||
      deepmd::DPBackend::TensorFlow == backend) {
    mlu_backend = create_deepmlu_backend_from_plugin(backend, model, gpu_rank,
                                                     file_content);
  } else {
    throw deepmd::deepmd_exception(
        "DeepMLU: unsupported backend for MLU model");
  }
  // Also init the DeepBaseModel dpbase pointer (needed for cutoff/numb_types/…
  // accessors). The DeepMLUBackend IS a DeepBaseModelBackend, so we can share
  // the pointer.
  dpbase = mlu_backend;
  inited = true;
}

template <typename VALUETYPE>
void DeepMLU::compute(std::vector<VALUETYPE>& u_per_frame,
                      const std::vector<VALUETYPE>& coord,
                      const std::vector<int>& atype,
                      const std::vector<VALUETYPE>& box,
                      const int nghost,
                      const InputNlist& lmp_list,
                      const int ago,
                      const std::vector<VALUETYPE>& fparam) {
  mlu_backend->computew(u_per_frame, coord, atype, box, nghost, lmp_list, ago,
                        fparam);
}

template void DeepMLU::compute<double>(std::vector<double>&,
                                       const std::vector<double>&,
                                       const std::vector<int>&,
                                       const std::vector<double>&,
                                       const int,
                                       const InputNlist&,
                                       const int,
                                       const std::vector<double>&);
template void DeepMLU::compute<float>(std::vector<float>&,
                                      const std::vector<float>&,
                                      const std::vector<int>&,
                                      const std::vector<float>&,
                                      const int,
                                      const InputNlist&,
                                      const int,
                                      const std::vector<float>&);

const std::string& DeepMLU::get_var_name() const {
  return mlu_backend->get_var_name();
}

int DeepMLU::get_task_dim() const { return mlu_backend->get_task_dim(); }
