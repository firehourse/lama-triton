#include "image_preprocessor.h"
#include <cstdint>
#include <cstring>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>
#include <triton/core/tritonserver.h>

namespace triton {
namespace backend {
namespace image_preprocessor {

class ModelState : public BackendModel {
public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model,
                                    ModelState **state);
  virtual ~ModelState() = default;

private:
  ModelState(TRITONBACKEND_Model *triton_model) : BackendModel(triton_model) {}
};

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model,
                                       ModelState **state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException &ex) {
    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("unexpected nullptr error"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr; // Success
}

class ModelInstanceState : public BackendModelInstance {
public:
  static TRITONSERVER_Error *
  Create(ModelState *model_state,
         TRITONBACKEND_ModelInstance *triton_model_instance,
         ModelInstanceState **state);
  virtual ~ModelInstanceState() = default;

  // The core execution logic for the instance.
  TRITONSERVER_Error *Execute(std::vector<TRITONBACKEND_Request *> &requests);

private:
  ModelInstanceState(ModelState *model_state,
                     TRITONBACKEND_ModelInstance *triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance) {}

  TRITONSERVER_Error *ProcessRequest(TRITONBACKEND_Request *request);
};

TRITONSERVER_Error *
ModelInstanceState::Create(ModelState *model_state,
                           TRITONBACKEND_ModelInstance *triton_model_instance,
                           ModelInstanceState **state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException &ex) {
    RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                         std::string("unexpected nullptr error"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr; // Success
}

TRITONSERVER_Error *
ModelInstanceState::Execute(std::vector<TRITONBACKEND_Request *> &requests) {
  for (auto &request : requests) {
    auto err = ProcessRequest(request);

    TRITONBACKEND_Response *response = nullptr;
    if (err != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
      TRITONBACKEND_ResponseNew(&response, request);
      if (response != nullptr) {
        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                         response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                     "failed sending error response");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(
                     request, TRITONSERVER_RESPONSE_COMPLETE_FINAL),
                 "failed to release request");
  }
  return nullptr;
}

TRITONSERVER_Error *
ModelInstanceState::ProcessRequest(TRITONBACKEND_Request *request) {
  TRITONBACKEND_Input *image_input;
  TRITONBACKEND_Input *mask_input;
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInput(request, "IMAGE_RAW", &image_input));
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "MASK_RAW", &mask_input));

  uint64_t img_byte_size;
  const void *image_buffer;
  TRITONSERVER_MemoryType img_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t img_memory_type_id = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(image_input, 0, &image_buffer,
                                            &img_byte_size, &img_memory_type,
                                            &img_memory_type_id));

  uint64_t msk_byte_size;
  const void *mask_buffer;
  TRITONSERVER_MemoryType msk_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t msk_memory_type_id = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(mask_input, 0, &mask_buffer,
                                            &msk_byte_size, &msk_memory_type,
                                            &msk_memory_type_id));

  // Handle Triton STRING format (4-byte length prefix)
  const char *img_ptr = reinterpret_cast<const char *>(image_buffer);
  uint32_t img_len = *(reinterpret_cast<const uint32_t *>(img_ptr));
  std::vector<uint8_t> img_bytes(img_ptr + 4, img_ptr + 4 + img_len);

  const char *msk_ptr = reinterpret_cast<const char *>(mask_buffer);
  uint32_t msk_len = *(reinterpret_cast<const uint32_t *>(msk_ptr));
  std::vector<uint8_t> msk_bytes(msk_ptr + 4, msk_ptr + 4 + msk_len);

  lama_preproc::PreprocResult res;
  try {
    res = lama_preproc::preprocess(img_bytes, msk_bytes);
  } catch (const std::exception &e) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }

  TRITONBACKEND_Response *response;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

  // IMAGE_TENSOR
  TRITONBACKEND_Output *img_out;
  std::vector<int64_t> img_shape = {3, res.padded_h, res.padded_w};
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      response, &img_out, "IMAGE_TENSOR", TRITONSERVER_TYPE_FP32,
      img_shape.data(), 3));

  void *out_buffer;
  TRITONSERVER_MemoryType mem_type = TRITONSERVER_MEMORY_CPU;
  int64_t mem_id = 0;
  uint64_t out_byte_size = res.img_tensor.total() * res.img_tensor.elemSize();
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
      img_out, &out_buffer, out_byte_size, &mem_type, &mem_id));
  std::memcpy(out_buffer, res.img_tensor.data, out_byte_size);

  // MASK_TENSOR
  TRITONBACKEND_Output *msk_out;
  std::vector<int64_t> msk_shape = {1, res.padded_h, res.padded_w};
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      response, &msk_out, "MASK_TENSOR", TRITONSERVER_TYPE_FP32,
      msk_shape.data(), 3));
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(msk_out, &out_buffer,
                                             res.mask_tensor.total() *
                                                 res.mask_tensor.elemSize(),
                                             &mem_type, &mem_id));
  std::memcpy(out_buffer, res.mask_tensor.data,
              res.mask_tensor.total() * res.mask_tensor.elemSize());

  // ORIGINAL_SHAPE
  TRITONBACKEND_Output *shape_out;
  std::vector<int64_t> s_shape = {2};
  int32_t shapes[2] = {res.original_h, res.original_w};
  RETURN_IF_ERROR(
      TRITONBACKEND_ResponseOutput(response, &shape_out, "ORIGINAL_SHAPE",
                                   TRITONSERVER_TYPE_INT32, s_shape.data(), 1));
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
      shape_out, &out_buffer, sizeof(shapes), &mem_type, &mem_id));
  std::memcpy(out_buffer, shapes, sizeof(shapes));

  RETURN_IF_ERROR(TRITONBACKEND_ResponseSend(
      response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));
  return nullptr;
}

extern "C" {

TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend) {
  return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model) {
  ModelState *model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
      model, reinterpret_cast<void *>(model_state)));
  return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vstate);
  delete model_state;
  return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance) {
  TRITONBACKEND_Model *model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void *vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vmodelstate);

  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void *>(instance_state)));
  return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState *instance_state =
      reinterpret_cast<ModelInstanceState *>(vstate);
  delete instance_state;
  return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                   TRITONBACKEND_Request **requests,
                                   const uint32_t request_count) {
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void **>(&instance_state)));

  std::vector<TRITONBACKEND_Request *> request_vec(requests,
                                                   requests + request_count);
  return instance_state->Execute(request_vec);
}

} // extern "C"

} // namespace image_preprocessor
} // namespace backend
} // namespace triton
