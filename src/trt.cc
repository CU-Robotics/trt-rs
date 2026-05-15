#include <ranges>
#include "trt.hpp"

std::unique_ptr<nvinfer1::Dims> new_dims(rust::Slice<const int32_t> dims_spec)
{
    nvinfer1::Dims dims{};

    if (dims_spec.length() > dims.MAX_DIMS)
        throw std::runtime_error("dimension count exceeds MAX_DIMS");

    dims.nbDims = static_cast<int32_t>(dims_spec.length());
    std::ranges::copy(dims_spec, dims.d);

    auto dims_ptr = std::make_unique<nvinfer1::Dims>(dims);
    if (!dims_ptr)
        throw std::runtime_error("could not create dims");
    return dims_ptr;
}

void reshape_dims(const std::unique_ptr<nvinfer1::Dims> dims, rust::Slice<const int32_t> dims_spec)
{
    if (dims_spec.length() > dims->MAX_DIMS)
        throw std::runtime_error("dimension count exceeds MAX_DIMS");

    dims->nbDims = static_cast<int32_t>(dims_spec.length());
    std::ranges::copy(dims_spec, dims->d);
}

std::unique_ptr<Logger> new_logger(Severity log_level)
{
    auto logger = std::make_unique<Logger>(log_level);
    if (!logger)
        throw std::runtime_error("could not create logger");
    return logger;
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