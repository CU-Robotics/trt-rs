#include <algorithm>
#include <ranges>
#include "trt.hpp"

std::unique_ptr<nvinfer1::Dims> dims_new(rust::Slice<const int64_t> spec)
{
    auto dims = std::make_unique<nvinfer1::Dims>(); // can throw
    dims_copy_from_slice(dims, spec);
    return dims;
}

void dims_copy_from_slice(const std::unique_ptr<nvinfer1::Dims> &dims, rust::Slice<const int64_t> spec)
{
    dims->nbDims = std::min(static_cast<int32_t>(spec.length()), dims->MAX_DIMS);
    std::ranges::copy(spec | std::views::take(dims->MAX_DIMS), dims->d);
}

void dims_copy(const std::unique_ptr<nvinfer1::Dims> &src, const std::unique_ptr<nvinfer1::Dims> &dest)
{
    dest->nbDims = src->nbDims;
    std::copy(std::begin(src->d), std::end(src->d), std::begin(dest->d));
}

std::unique_ptr<nvinfer1::Dims> dims_invalid()
{
    nvinfer1::Dims dims{};
    dims.nbDims = -1;
    return std::make_unique<nvinfer1::Dims>(dims); // can throw
}

std::unique_ptr<nvinfer1::Dims> dims_clone(const std::unique_ptr<nvinfer1::Dims> &dims)
{
    return std::make_unique<nvinfer1::Dims>(*dims); // can throw
}

int32_t dims_nb_dims(const std::unique_ptr<nvinfer1::Dims> &dims)
{
    return dims->nbDims;
}

int64_t dims_get_axis(const std::unique_ptr<nvinfer1::Dims> &dims, size_t idx)
{
    if (idx >= dims->MAX_DIMS)
        throw std::out_of_range("axis index is out of range");

    return dims->d[idx];
}

void dims_set_axis(const std::unique_ptr<nvinfer1::Dims> &dims, size_t idx, int64_t val)
{
    if (idx >= dims->MAX_DIMS)
        throw std::out_of_range("axis index is out of range");

    dims->d[idx] = val;
}

bool dims_is_invalid(const std::unique_ptr<nvinfer1::Dims> &dims)
{
    return dims->nbDims == -1 && dims->d[0] == 0;
}

bool dims_is_unknown_rank(const std::unique_ptr<nvinfer1::Dims> &dims)
{
    return dims->nbDims == -1 && dims->d[0] == -1;
}

std::unique_ptr<Logger> logger_new(Severity log_level)
{
    return std::make_unique<Logger>(log_level); // can throw
}

std::unique_ptr<nvinfer1::IRuntime> create_infer_runtime(const std::unique_ptr<Logger> &logger)
{
    auto runtime = nvinfer1::createInferRuntime(*logger);
    if (!runtime)
        throw std::runtime_error("could not create runtime");
    return std::unique_ptr<nvinfer1::IRuntime>(runtime);
}

std::unique_ptr<nvinfer1::ICudaEngine> runtime_deserialize_cuda_engine(const std::unique_ptr<nvinfer1::IRuntime> &runtime, rust::Slice<const rust::u8> serialized_engine)
{
    auto engine = runtime->deserializeCudaEngine(serialized_engine.data(), serialized_engine.length());
    if (!engine)
        throw std::runtime_error("could not create engine");
    return std::unique_ptr<nvinfer1::ICudaEngine>(engine);
}

std::unique_ptr<nvinfer1::IExecutionContext> engine_create_execution_context(const std::unique_ptr<nvinfer1::ICudaEngine> &engine)
{
    auto context = engine->createExecutionContext();
    if (!context)
        throw std::runtime_error("could not create execution context");
    return std::unique_ptr<nvinfer1::IExecutionContext>(context);
}

std::unique_ptr<nvinfer1::Dims> engine_get_tensor_shape(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, const char *tensor_name)
{
    auto dims = engine->getTensorShape(tensor_name);
    return std::make_unique<nvinfer1::Dims>(dims);
}

std::unique_ptr<nvinfer1::Dims> engine_get_profile_shape(const std::unique_ptr<nvinfer1::ICudaEngine> &engine, const char *tensor_name, int32_t profile_index, nvinfer1::OptProfileSelector optimization_selector)
{
    auto dims = engine->getProfileShape(tensor_name, profile_index, optimization_selector);
    return std::make_unique<nvinfer1::Dims>(dims);
}

bool context_set_tensor_address(const std::unique_ptr<nvinfer1::IExecutionContext> &context, const char *tensor_name, rust::usize addr)
{
    auto ptr = reinterpret_cast<void *>(addr);
    return context->setTensorAddress(tensor_name, ptr);
}

bool context_set_input_shape(const std::unique_ptr<nvinfer1::IExecutionContext> &context, const char *tensor_name, const std::unique_ptr<nvinfer1::Dims> dims)
{
    return context->setInputShape(tensor_name, *dims);
}