use cxx::UniquePtr;

#[cxx::bridge]
mod ffi {
    #[repr(i32)]
    enum Severity {
        // An internal error has occurred. Execution is unrecoverable.
        kINTERNAL_ERROR = 0,
        // An application error has occurred.
        kERROR = 1,
        // An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        kWARNING = 2,
        // Informational messages with instructional information.
        kINFO = 3,
        // Verbose messages with debugging information.
        kVERBOSE = 4,
    }

    #[repr(i32)]
    enum DataType {
        // 32-bit floating point format.
        kFLOAT = 0,

        // IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
        kHALF = 1,

        // Signed 8-bit integer representing a quantized floating-point value.
        kINT8 = 2,

        // Signed 32-bit integer format.
        kINT32 = 3,

        // 8-bit boolean. 0 = false, 1 = true, other values undefined.
        kBOOL = 4,

        // Unsigned 8-bit integer format.
        // Cannot be used to represent quantized floating-point values.
        // Use the IdentityLayer to convert kUINT8 network-level inputs to {kFLOAT, kHALF} prior
        // to use with other TensorRT layers, or to convert intermediate output
        // before kUINT8 network-level outputs from {kFLOAT, kHALF} to kUINT8.
        // kUINT8 conversions are only supported for {kFLOAT, kHALF}.
        // kUINT8 to {kFLOAT, kHALF} conversion will convert the integer values
        // to equivalent floating point values.
        // {kFLOAT, kHALF} to kUINT8 conversion will convert the floating point values
        // to integer values by truncating towards zero. This conversion has undefined behavior for
        // floating point values outside the range [0.0F, 256.0F) after truncation.
        // kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
        kUINT8 = 5,

        // Signed 8-bit floating point with
        // 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
        kFP8 = 6,

        // Brain float -- has an 8 bit exponent and 8 bit significand.
        kBF16 = 7,

        // Signed 64-bit integer type.
        kINT64 = 8,

        // Signed 4-bit integer type.
        kINT4 = 9,
    }

    #[repr(i32)]
    enum TensorLocation {
        kDEVICE = 0, //< Data stored on device.
        kHOST = 1,   //< Data stored on host.
    }

    #[repr(i32)]
    enum TensorIoMode {
        // Tensor is not an input or output.
        kNONE = 0,

        // Tensor is input to the engine.
        kINPUT = 1,

        // Tensor is output by the engine.
        kOUTPUT = 2,
    }

    #[repr(i32)]
    enum TensorFormat {
        // Memory layout is similar to an array in C or C++.
        // The stride of each dimension is the product of the dimensions after it.
        // The last dimension has unit stride.
        //
        // For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
        kLINEAR = 0,

        // Vector-major format with two scalars per vector.
        // Vector dimension is third to last.
        //
        // This format requires FP16 or BF16 and at least three dimensions.
        kCHW2 = 1,

        // Vector-minor format with eight scalars per vector.
        // Vector dimension is third to last.
        // This format requires FP16 or BF16 and at least three dimensions.
        kHWC8 = 2,

        // Vector-major format with four scalars per vector.
        // Vector dimension is third to last.
        //
        // This format requires INT8 or FP16 and at least three dimensions.
        // For INT8, the length of the vector dimension must be a build-time constant.
        //
        // Deprecated usage:
        //
        // If running on the DLA, this format can be used for acceleration
        // with the caveat that C must be less than or equal to 4.
        // If used as DLA input and the build option kGPU_FALLBACK is not specified,
        // it needs to meet line stride requirement of DLA format. Column stride in
        // bytes must be a multiple of 64 on Orin.
        kCHW4 = 3,

        // Vector-major format with 16 scalars per vector.
        // Vector dimension is third to last.
        //
        // This format requires INT8 or FP16 and at least three dimensions.
        //
        // For DLA usage, this format maps to the native feature format for FP16,
        // and the tensor sizes are limited to C,H,W in the range [1,8192].
        kCHW16 = 4,

        // Vector-major format with 32 scalars per vector.
        // Vector dimension is third to last.
        //
        // This format requires at least three dimensions.
        //
        // For DLA usage, this format maps to the native feature format for INT8,
        // and the tensor sizes are limited to C,H,W in the range [1,8192].
        kCHW32 = 5,

        // Vector-minor format with eight scalars per vector.
        // Vector dimension is fourth to last.
        //
        // This format requires FP16 or BF16 and at least four dimensions.
        kDHWC8 = 6,

        // Vector-major format with 32 scalars per vector.
        // Vector dimension is fourth to last.
        //
        // This format requires FP16 or INT8 and at least four dimensions.
        kCDHW32 = 7,

        // Vector-minor format where channel dimension is third to last and unpadded.
        //
        // This format requires either FP32 or UINT8 and at least three dimensions.
        kHWC = 8,

        // DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
        // always has unit stride. The stride for stepping along the H axis is
        // rounded up to 64 bytes.
        //
        // The memory layout is equivalent to a C array with dimensions
        // [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
        // 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
        // mapping to array subscript [n][c][h][w].
        kDLA_LINEAR = 9,

        // DLA image format. For a tensor with dimension {N, C, H, W} the C axis
        // always has unit stride. The stride for stepping along the H axis is rounded up
        // to 64 bytes on Orin. C can only be 1, 3 or 4.
        // If C == 1, it will map to grayscale format.
        // If C == 3 or C == 4, it will map to color image format. And if C == 3,
        // the stride for stepping along the W axis needs to be padded to 4 in elements.
        //
        // When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
        // the memory layout is equivalent to a C array with dimensions
        // [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
        // where elementSize is 2 for FP16
        // and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
        // subscript [n][h][w][c].
        kDLA_HWC4 = 10,

        // Vector-minor format with 16 scalars per vector.
        // Vector dimension is third to last.
        //
        // This requires FP16 and at least three dimensions.
        kHWC16 = 11,

        // Vector-minor format with one scalar per vector.
        // Vector dimension is fourth to last.
        //
        // This format requires FP32 and at least four dimensions.
        kDHWC = 12,
    }

    enum OptProfileSelector {
        kMIN = 0, //< This is used to set or get the minimum permitted value for dynamic dimensions etc.
        kOPT = 1, //< This is used to set or get the value that is used in the optimization (kernel selection).
        kMAX = 2, //< This is used to set or get the maximum permitted value for dynamic dimensions etc.
    }

    unsafe extern "C++" {
        include!("NvInfer.h");
        include!("cuda_runtime.h");
        include!("catcher.hpp");
        include!("trt.hpp");

        type Severity;
        type DataType;
        type TensorLocation;
        type TensorIoMode;
        type TensorFormat;
        type OptProfileSelector;

        type Logger;

        #[namespace = "nvinfer1"]
        type Dims;
        #[namespace = "nvinfer1"]
        type IRuntime;
        #[namespace = "nvinfer1"]
        type ICudaEngine;
        #[namespace = "nvinfer1"]
        type IExecutionContext;

        fn new_logger(log_level: Severity) -> UniquePtr<Logger>;

        fn create_infer_runtime(logger: UniquePtr<Logger>) -> UniquePtr<IRuntime>;
        fn runtime_deserialize_cuda_engine(
            runtime: UniquePtr<IRuntime>,
            model: Vec<u8>,
        ) -> UniquePtr<ICudaEngine>;

        fn engine_create_execution_context(
            engine: &ICudaEngine,
        ) -> UniquePtr<IExecutionContext>;
        fn engine_get_tensor_shape(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> UniquePtr<Dims>;
        fn engine_get_tensor_data_type(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> DataType;
        fn engine_get_tensor_location(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> TensorLocation;
        fn engine_get_tensor_io_mode(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> TensorIoMode;
        fn engine_get_tensor_bytes_per_component1(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> i32;
        fn engine_get_tensor_bytes_per_component2(
            engine: &ICudaEngine,
            tensor_name: &str,
            profile_index: i32,
        ) -> i32;
        fn engine_get_tensor_components_per_element1(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> i32;
        fn engine_get_tensor_components_per_element2(
            engine: &ICudaEngine,
            tensor_name: &str,
            profile_index: i32,
        ) -> i32;
        fn engine_get_tensor_format1(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> TensorFormat;
        fn engine_get_tensor_format2(
            engine: &ICudaEngine,
            tensor_name: &str,
            profile_index: i32,
        ) -> TensorFormat;
        fn engine_get_tensor_vectorized_dim1(
            engine: &ICudaEngine,
            tensor_name: &str,
        ) -> i32;
        fn engine_get_tensor_vectorized_dim2(
            engine: &ICudaEngine,
            tensor_name: &str,
            profile_index: i32,
        ) -> i32;
        fn engine_get_nb_optimization_profiles(engine: &ICudaEngine) -> i32;
        fn engine_get_profile_shape(
            engine: &ICudaEngine,
            tensor_name: &str,
            profile_index: i32,
            optimization_selector: OptProfileSelector,
        ) -> UniquePtr<Dims>;
        fn engine_get_nb_io_tensors(engine: &ICudaEngine) -> i32;
        fn engine_get_io_tensor_name(engine: &ICudaEngine) -> &str;

        fn context_set_tensor_address(context: Unique)
    }
}
