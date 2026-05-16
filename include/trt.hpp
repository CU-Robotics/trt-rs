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

std::unique_ptr<nvinfer1::Dims> dims_new(rust::Slice<const int64_t> spec);
void dims_copy_from_slice(const std::unique_ptr<nvinfer1::Dims> &dims, rust::Slice<const int64_t> spec);
void dims_copy(const std::unique_ptr<nvinfer1::Dims> &src, const std::unique_ptr<nvinfer1::Dims> &dest);
rust::Slice<const int64_t> dims_as_slice(const std::unique_ptr<nvinfer1::Dims> &dims);
std::unique_ptr<nvinfer1::Dims> dims_invalid();
std::unique_ptr<nvinfer1::Dims> dims_clone(const std::unique_ptr<nvinfer1::Dims> &dims);
int32_t dims_nb_dims(const std::unique_ptr<nvinfer1::Dims> &dims);
int64_t dims_get_axis(const std::unique_ptr<nvinfer1::Dims> &dims, size_t idx);
void dims_set_axis(const std::unique_ptr<nvinfer1::Dims> &dims, size_t idx, int64_t val);
bool dims_is_invalid(const std::unique_ptr<nvinfer1::Dims> &dims);
bool dims_is_unknown_rank(const std::unique_ptr<nvinfer1::Dims> &dims);

std::unique_ptr<Logger> logger_new(Severity log_level);

std::unique_ptr<nvinfer1::IRuntime> create_infer_runtime(const std::unique_ptr<Logger> &logger);
std::unique_ptr<nvinfer1::ICudaEngine> runtime_deserialize_cuda_engine(const std::unique_ptr<nvinfer1::IRuntime> &runtime, rust::Slice<const rust::u8> serialized_engine);

std::unique_ptr<nvinfer1::IExecutionContext> engine_create_execution_context(const std::unique_ptr<nvinfer1::ICudaEngine> &engine);
std::unique_ptr<nvinfer1::Dims> engine_get_tensor_shape(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, const char *tensor_name);
std::unique_ptr<nvinfer1::Dims> engine_get_profile_shape(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, const char *tensor_name, int32_t profile_index, nvinfer1::OptProfileSelector optimization_selector);

bool context_set_tensor_address(const std::unique_ptr<nvinfer1::IExecutionContext> &context, const char *tensor_name, rust::usize addr);
bool context_set_input_shape(const std::unique_ptr<nvinfer1::IExecutionContext> &context, const char *tensor_name, const std::unique_ptr<nvinfer1::Dims> &dims);