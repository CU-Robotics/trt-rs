#pragma once

#include <cstdio>
#include <cstdint>
#include <memory>
#include "NvInfer.h"
#include "cuda_runtime.h"
#include "rust/cxx.h"

class Logger : public nvinfer1::ILogger
{
private:
    Severity log_level;

public:
    Logger(Severity severity) : log_level(severity) {}

    void log(Severity severity, const char *message) noexcept override
    {
        if (severity <= log_level)
        {
            const char *level = "";

            switch (severity)
            {
            case Severity::kINTERNAL_ERROR:
                level = "internal_error";
                break;
            case Severity::kERROR:
                level = "error";
                break;
            case Severity::kWARNING:
                level = "warning";
                break;
            case Severity::kINFO:
                level = "info";
                break;
            case Severity::kVERBOSE:
                level = "verbose";
                break;
            }

            fprintf(stderr, "[trt-rs %s] %s\n", level, message);
        }
    }
};

using Severity = nvinfer1::ILogger::Severity;
using DataType = nvinfer1::DataType;
using TensorLocation = nvinfer1::TensorLocation;
using TensorIoMode = nvinfer1::TensorIOMode;
using TensorFormat = nvinfer1::TensorFormat;
using OptProfileSelector = nvinfer1::OptProfileSelector;

std::unique_ptr<Logger> new_logger(Severity log_level);

std::unique_ptr<nvinfer1::IRuntime> create_infer_runtime(std::unique_ptr<Logger> logger);
std::unique_ptr<nvinfer1::ICudaEngine> runtime_deserialize_cuda_engine(std::unique_ptr<nvinfer1::IRuntime> runtime, std::vector<uint8_t> model);

std::unique_ptr<nvinfer1::Dims> new_dims(rust::Slice<const int32_t> dims_spec);

std::unique_ptr<nvinfer1::IExecutionContext> engine_create_execution_context(nvinfer1::ICudaEngine &engine);
std::unique_ptr<nvinfer1::Dims> engine_get_tensor_shape(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
DataType engine_get_tensor_data_type(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
TensorLocation engine_get_tensor_location(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
TensorIoMode engine_get_tensor_io_mode(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
int32_t engine_get_tensor_bytes_per_component1(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
int32_t engine_get_tensor_bytes_per_component2(nvinfer1::ICudaEngine &engine, rust::Str tensor_name, int32_t profile_index);
int32_t engine_get_tensor_components_per_element1(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
int32_t engine_get_tensor_components_per_element2(nvinfer1::ICudaEngine &engine, rust::Str tensor_name, int32_t profile_index);
TensorFormat engine_get_tensor_format1(nvinfer1::ICudaEngine &engine, rust::Str tensor_name);
TensorFormat engine_get_tensor_format2(nvinfer1::ICudaEngine &engine, rust::Str tensor_name, int32_t profile_index);
int32_t engine_get_tensor_vectorized_dim1(nvinfer1::ICudaEngine &engine, rust::Str tensor_name)
    int32_t engine_get_tensor_vectorized_dim(nvinfer1::ICudaEngine &engine, rust::Str tensor_name, int32_t profile_index);
int32_t engine_get_nb_optimization_profiles(nvinfer1::ICudaEngine &engine);
std::unique_ptr<nvinfer1::Dims> engine_get_profile_shape(nvinfer1::ICudaEngine &engine, rust::Str tensor_name, int32_t profile_index, OptProfileSelector optimization_selector);
int32_t engine_get_nb_io_tensors(nvinfer1::ICudaEngine &engine);
rust::Str engine_get_io_tensor_name(int32_t index);

bool context_set_tensor_address(std::unique_ptr<nvinfer1::IExecutionContext> context, rust::Str tensor_name, void *cuda_ptr);
bool context_set_input_shape(std::unique_ptr<nvinfer1::IExecutionContext> context, rust::Str tensor_name, const std::unique_ptr<nvinfer1::Dims> dims);
bool context_enqueue_v3(std::unique_ptr<nvinfer1::IExecutionContext> context, cudaStream_t stream);
