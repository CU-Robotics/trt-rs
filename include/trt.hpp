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
std::unique_ptr<nvinfer1::Dims> new_dims(rust::Slice<const int32_t> dims_spec);
void reshape_dims(const std::unique_ptr<nvinfer1::Dims> &dims, rust::Slice<const int32_t> dims_spec);
std::unique_ptr<Logger> new_logger(Severity log_level);
std::unique_ptr<nvinfer1::IRuntime> create_infer_runtime(const std::unique_ptr<Logger> &logger);
std::unique_ptr<nvinfer1::ICudaEngine> runtime_deserialize_cuda_engine(const std::unique_ptr<nvinfer1::IRuntime> &runtime, rust::Slice<const rust::u8> serialized_engine);
std::unique_ptr<nvinfer1::Dims> engine_get_tensor_shape(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, const char *tensor_name);
std::unique_ptr<nvinfer1::Dims> engine_get_profile_shape(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, const char *tensor_name, int32_t profile_index, nvinfer1::OptProfileSelector optimization_selector);
bool context_set_tensor_address(const std::unique_ptr<nvinfer1::IExecutionContext> &context, const char *tensor_name, rust::usize addr);
bool context_set_input_shape(const std::unique_ptr<nvinfer1::IExecutionContext> &context, const char *tensor_name, const std::unique_ptr<nvinfer1::Dims> &dims);