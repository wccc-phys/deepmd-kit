// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "DeepBaseModel.h"
#include "common.h"

namespace deepmd {
/**
 * @brief DeepMLU backend interface.
 *
 * DeepMLU runs an MLU (Machine-Learning Hubbard U) model to predict the
 * per-frame Hubbard U value from coordinates, atom types, and frame
 * parameters (typically temperature). The predicted U is fed as ``uparam``
 * to the main DP+U potential.
 **/
class DeepMLUBackend : public DeepBaseModelBackend {
 public:
  DeepMLUBackend() {};
  virtual ~DeepMLUBackend() {};

  virtual void init(const std::string& model,
                    const int& gpu_rank = 0,
                    const std::string& file_content = "") = 0;

  /**
   * @brief Evaluate the per-frame U value.
   * @param[out] u_per_frame The predicted U, one scalar per frame.
   * @param[in] coord The coordinates of atoms, size natoms x 3.
   * @param[in] atype The atom types, size natoms.
   * @param[in] box The cell, size 9.
   * @param[in] nghost Number of ghost atoms.
   * @param[in] lmp_list LAMMPS neighbor list.
   * @param[in] ago Recompute flag (-1 for recompute).
   * @param[in] fparam Frame parameters (e.g. temperature), size dim_fparam.
   **/
  virtual void computew(std::vector<double>& u_per_frame,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const int nghost,
                        const InputNlist& lmp_list,
                        const int ago,
                        const std::vector<double>& fparam) = 0;

  virtual void computew(std::vector<float>& u_per_frame,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const int nghost,
                        const InputNlist& lmp_list,
                        const int ago,
                        const std::vector<float>& fparam) = 0;

  /**
   * @brief Get the variable name of the MLU output.
   * @return The variable name (always "uparam").
   **/
  virtual const std::string& get_var_name() const = 0;

  /**
   * @brief Get the task dimension (always 1 for MLU).
   * @return The task dimension.
   **/
  virtual int get_task_dim() const = 0;
};

/**
 * @brief DeepMLU pimpl wrapper — dispatches to the correct backend plugin.
 **/
class DeepMLU : public DeepBaseModel {
 public:
  DeepMLU();
  ~DeepMLU();
  DeepMLU(const std::string& model,
          const int& gpu_rank = 0,
          const std::string& file_content = "");

  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

  /**
   * @brief Evaluate the per-frame U value.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& u_per_frame,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int ago,
               const std::vector<VALUETYPE>& fparam);

  /**
   * @brief Get the variable name of the MLU output.
   **/
  const std::string& get_var_name() const;

  /**
   * @brief Get the task dimension.
   **/
  int get_task_dim() const;

 private:
  std::shared_ptr<DeepMLUBackend> mlu_backend;
};
}  // namespace deepmd
