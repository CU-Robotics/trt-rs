#include <ranges>
#include "trt.hpp"

std::unique_ptr<Logger> new_logger(Severity log_level)
{
    return std::make_unique<Logger>(log_level);
}

std::unique_ptr<nvinfer1::IRuntime> create_infer_runtime(std::unique_ptr<Logger> logger)
{
    auto runtime = nvinfer1::createInferRuntime(*logger);
    return std::unique_ptr<nvinfer1::IRuntime>(runtime);
}

std::unique_ptr<nvinfer1::ICudaEngine> runtime_deserialize_cuda_engine(std::unique_ptr<nvinfer1::IRuntime> runtime, std::vector<rust::u8> model)
{
    auto engine = runtime->deserializeCudaEngine(model.data(), model.size());
    return std::unique_ptr<nvinfer1::ICudaEngine>(engine);
}

std::unique_ptr<nvinfer1::Dims> new_dims(rust::Slice<const int32_t> dims_spec)
{
    nvinfer1::Dims dims{};

    dims.nbDims = std::min(dims_spec.length(), dims.MAX_DIMS);
    std::ranges::copy(
        dims_spec | std::views::take(dims.MAX_DIMS),
        dims.d);

    return std::unique_ptr<nvinfer1>(dims);
}

std::unique_ptr<nvinfer1::IExecutionContext> engine_create_execution_context(const nvinfer1::icudaengine &engine)
{
    auto context = engine->createExecutionContext();
    return std::unique_ptr<nvinfer1::IExecutionContext>(context);
}

std::unique_ptr<nvinfer1::Dims> engine_get_tensor_shape(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorShape(tensor_name.data());
}

DataType engine_get_tensor_data_type(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorDataType(tensor_name.data());
}

TensorLocation engine_get_tensor_location(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorLocation(tensor_name.data());
}

TensorIoMode engine_get_tensor_io_mode(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorIOMode(tensor_name.data());
}

int32_t engine_get_tensor_bytes_per_component1(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorBytesPerComponent(tensor_name.data());
}

int32_t engine_get_tensor_bytes_per_component2(const nvinfer1::icudaengine &engine, rust::Str tensor_name, int32_t profile_index)
{
    return engine->getTensorBytesPerComponent(tensor_name.data(), profile_index);
}

int32_t engine_get_tensor_components_per_element1(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorComponentsPerElement(tensor_name.data());
}

int32_t engine_get_tensor_components_per_element2(const nvinfer1::icudaengine &engine, rust::Str tensor_name, int32_t profile_index)
{
    return engine->getTensorComponentsPerElement(tensor_name.data(), profile_index);
}

TensorFormat engine_get_tensor_format1(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorFormat(tensor_name.data());
}

TensorFormat engine_get_tensor_format2(const nvinfer1::icudaengine &engine, rust::Str tensor_name, int32_t profile_index)
{
    return engine->getTensorFormat(tensor_name.data(), profile_index);
}

int32_t engine_get_tensor_vectorized_dim1(const nvinfer1::icudaengine &engine, rust::Str tensor_name)
{
    return engine->getTensorVectorizedDim2(tensor_name.data());
}

int32_t engine_get_tensor_vectorized_dim(const nvinfer1::icudaengine &engine, rust::Str tensor_name, int32_t profile_index)
{
    return engine->getTensorVectorizedDim(tensor_name.data(), profile_index);
}

int32_t engine_get_nb_optimization_profiles(const nvinfer1::icudaengine &engine)
{
    return engine->getNbOptimizationProfiles();
}

std::unique_ptr<nvinfer1::Dims> engine_get_profile_shape(const nvinfer1::icudaengine &engine, rust::Str tensor_name, int32_t profile_index, OptProfileSelector optimization_selector)
{
    auto dims = engine->getProfileShape(tensor_name.data(), profile_index, optimization_selector);
    return std::make_unique<nvinfer1::Dims>(dims);
}

int32_t engine_get_nb_io_tensors(const nvinfer1::icudaengine &engine)
{
    return engine->getNbIOTensors();
}

rust::Str engine_get_io_tensor_name(const nvinfer1::icudaengine &engine, int32_t index)
{
    return engine->getIOTensorName(index);
}

bool context_set_tensor_address(std::unique_ptr<nvinfer1::IExecutionContext> context, rust::Str tensor_name, void *cuda_ptr)
{
    return context->setTensorAddress(tensor_name.data(), cuda_ptr);
}

bool context_set_input_shape(std::unique_ptr<nvinfer1::IExecutionContext> context, rust::Str tensor_name, const std::unique_ptr<nvinfer1::Dims> dims)
{
    return context->setInputShape(tensor_name.data(), dims);
}

bool context_enqueue_v3(std::unique_ptr<nvinfer1::IExecutionContext> context, cudaStream_t stream)
{
    return context->enqueueV3(stream);
}
