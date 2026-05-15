use cxx::UniquePtr;
use std::ffi::{CStr, c_void};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrtError {
    #[error("TensorRT error: {0}")]
    Cxx(#[from] cxx::Exception),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("null pointer error: {0}")]
    NullPtr(&'static str),
    #[error("API failed: {0}")]
    Api(&'static str),
}

pub type TrtResult<T> = Result<T, TrtError>;

#[cxx::bridge]
mod ffi {
    #[repr(i32)]
    enum Severity {
        /// An internal error has occurred. Execution is unrecoverable.
        #[cxx_name = "kINTERNAL_ERROR"]
        InternalError = 0,
        /// An application error has occurred.
        #[cxx_name = "kERROR"]
        Error = 1,
        /// An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        #[cxx_name = "kWARNING"]
        Warning = 2,
        /// Informational messages with instructional information.
        #[cxx_name = "kINFO"]
        Info = 3,
        /// Verbose messages with debugging information.
        #[cxx_name = "kVERBOSE"]
        Verbose = 4,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    enum DataType {
        #[cxx_name = "kFLOAT"]
        Float = 0,
        #[cxx_name = "kHALF"]
        Half = 1,
        #[cxx_name = "kINT8"]
        Int8 = 2,
        #[cxx_name = "kINT32"]
        Int32 = 3,
        #[cxx_name = "kBOOL"]
        Bool = 4,
        #[cxx_name = "kUINT8"]
        Uint8 = 5,
        /// Signed 8-bit floating point with
        /// 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
        #[cxx_name = "kFP8"]
        Fp8 = 6,
        /// Brain float -- has an 8 bit exponent and 8 bit significand.
        #[cxx_name = "kBF16"]
        Bf16 = 7,
        #[cxx_name = "kINT64"]
        Int64 = 8,
        #[cxx_name = "kINT4"]
        Int4 = 9,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    enum TensorLocation {
        #[cxx_name = "kDEVICE"]
        Device = 0,
        #[cxx_name = "kHOST"]
        Host = 1,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    #[cxx_name = "TensorIOMode"]
    enum TensorIoMode {
        #[cxx_name = "kNONE"]
        None = 0,
        #[cxx_name = "kINPUT"]
        Input = 1,
        #[cxx_name = "kOUTPUT"]
        Output = 2,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    enum TensorFormat {
        /// Memory layout is similar to an array in C or C++.
        /// The stride of each dimension is the product of the dimensions after it.
        /// The last dimension has unit stride.
        ///
        /// For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
        #[cxx_name = "kLINEAR"]
        Linear = 0,

        /// Vector-major format with two scalars per vector.
        /// Vector dimension is third to last.
        ///
        /// This format requires FP16 or BF16 and at least three dimensions.
        #[cxx_name = "kCHW2"]
        Chw2 = 1,

        /// Vector-minor format with eight scalars per vector.
        /// Vector dimension is third to last.
        /// This format requires FP16 or BF16 and at least three dimensions.
        #[cxx_name = "kHWC8"]
        Hwc8 = 2,

        /// Vector-major format with four scalars per vector.
        /// Vector dimension is third to last.
        ///
        /// This format requires INT8 or FP16 and at least three dimensions.
        /// For INT8, the length of the vector dimension must be a build-time constant.
        ///
        /// Deprecated usage:
        ///
        /// If running on the DLA, this format can be used for acceleration
        /// with the caveat that C must be less than or equal to 4.
        /// If used as DLA input and the build option kGPU_FALLBACK is not specified,
        /// it needs to meet line stride requirement of DLA format. Column stride in
        /// bytes must be a multiple of 64 on Orin.
        #[cxx_name = "kCHW4"]
        Chw4 = 3,

        /// Vector-major format with 16 scalars per vector.
        /// Vector dimension is third to last.
        ///
        /// This format requires INT8 or FP16 and at least three dimensions.
        ///
        /// For DLA usage, this format maps to the native feature format for FP16,
        /// and the tensor sizes are limited to C,H,W in the range [1,8192].
        #[cxx_name = "kCHW16"]
        Chw16 = 4,

        /// Vector-major format with 32 scalars per vector.
        /// Vector dimension is third to last.
        ///
        /// This format requires at least three dimensions.
        ///
        /// For DLA usage, this format maps to the native feature format for INT8,
        /// and the tensor sizes are limited to C,H,W in the range [1,8192].
        #[cxx_name = "kCHW32"]
        Chw32 = 5,

        /// Vector-minor format with eight scalars per vector.
        /// Vector dimension is fourth to last.
        ///
        /// This format requires FP16 or BF16 and at least four dimensions.
        #[cxx_name = "kDHWC8"]
        Dhwc8 = 6,

        /// Vector-major format with 32 scalars per vector.
        /// Vector dimension is fourth to last.
        ///
        /// This format requires FP16 or INT8 and at least four dimensions.
        #[cxx_name = "kCDHW32"]
        Cdhw32 = 7,

        /// Vector-minor format where channel dimension is third to last and unpadded.
        ///
        /// This format requires either FP32 or UINT8 and at least three dimensions.
        #[cxx_name = "kHWC"]
        Hwc = 8,

        /// DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
        /// always has unit stride. The stride for stepping along the H axis is
        /// rounded up to 64 bytes.
        ///
        /// The memory layout is equivalent to a C array with dimensions
        /// [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
        /// 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
        /// mapping to array subscript [n][c][h][w].
        #[cxx_name = "kDLA_LINEAR"]
        DlaLinear = 9,

        /// DLA image format. For a tensor with dimension {N, C, H, W} the C axis
        /// always has unit stride. The stride for stepping along the H axis is rounded up
        /// to 64 bytes on Orin. C can only be 1, 3 or 4.
        /// If C == 1, it will map to grayscale format.
        /// If C == 3 or C == 4, it will map to color image format. And if C == 3,
        /// the stride for stepping along the W axis needs to be padded to 4 in elements.
        ///
        /// When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
        /// the memory layout is equivalent to a C array with dimensions
        /// [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
        /// where elementSize is 2 for FP16
        /// and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
        /// subscript [n][h][w][c].
        #[cxx_name = "kDLA_HWC4"]
        DlaHwc4 = 10,

        /// Vector-minor format with 16 scalars per vector.
        /// Vector dimension is third to last.
        ///
        /// This requires FP16 and at least three dimensions.
        #[cxx_name = "kHWC16"]
        Hwc16 = 11,

        /// Vector-minor format with one scalar per vector.
        /// Vector dimension is fourth to last.
        ///
        /// This format requires FP32 and at least four dimensions.
        #[cxx_name = "kDHWC"]
        Dhwc = 12,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    enum OptProfileSelector {
        #[cxx_name = "kMIN"]
        Min = 0,
        #[cxx_name = "kOPT"]
        Opt = 1,
        #[cxx_name = "kMAX"]
        Max = 2,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    enum ExecutionContextAllocationStrategy {
        kSTATIC = 0,
        kON_PROFILE_CHANGE = 1,
        kUSER_MANAGED = 2,
    }

    unsafe extern "C++" {
        include!("NvInfer.h");
        include!("cuda_runtime.h");
        include!("catcher.hpp");
        include!("trt.hpp");

        // using types/custom types
        type CUstream_st;
        type Severity;
        type Logger;

        // enums
        #[namespace = "nvinfer1"]
        type DataType;
        #[namespace = "nvinfer1"]
        type TensorLocation;
        #[namespace = "nvinfer1"]
        #[rust_name = "TensorIoMode"]
        type TensorIOMode;
        #[namespace = "nvinfer1"]
        type TensorFormat;
        #[namespace = "nvinfer1"]
        type OptProfileSelector;
        #[namespace = "nvinfer1"]
        type ExecutionContextAllocationStrategy;

        // classes
        #[namespace = "nvinfer1"]
        type Dims;
        #[namespace = "nvinfer1"]
        type IRuntime;
        #[namespace = "nvinfer1"]
        type ICudaEngine;
        #[namespace = "nvinfer1"]
        type IExecutionContext;

        // methods
        fn new_dims(dims_spec: &[i32]) -> Result<UniquePtr<Dims>>;
        fn reshape_dims(dims: &UniquePtr<Dims>, dims_spec: &[i32]) -> Result<()>;

        fn new_logger(log_level: Severity) -> Result<UniquePtr<Logger>>;

        fn create_infer_runtime(logger: &UniquePtr<Logger>) -> Result<UniquePtr<IRuntime>>;

        fn runtime_deserialize_cuda_engine(
            runtime: &UniquePtr<IRuntime>,
            data: &[u8],
        ) -> Result<UniquePtr<ICudaEngine>>;

        #[namespace = "nvinfer1"]
        #[rust_name = "create_execution_context"]
        fn createExecutionContext(
            self: Pin<&mut ICudaEngine>,
            strategy: ExecutionContextAllocationStrategy,
        ) -> *mut IExecutionContext;

        unsafe fn engine_get_tensor_shape(
            engine: &UniquePtr<ICudaEngine>,
            tensor_name: *const c_char,
        ) -> UniquePtr<Dims>;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_tensor_data_type"]
        unsafe fn getTensorDataType(self: &ICudaEngine, tensor_name: *const c_char) -> DataType;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_tensor_location"]
        unsafe fn getTensorLocation(
            self: &ICudaEngine,
            tensor_name: *const c_char,
        ) -> TensorLocation;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_tensor_io_mode"]
        unsafe fn getTensorIOMode(self: &ICudaEngine, tensor_name: *const c_char) -> TensorIoMode;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorBytesPerComponent"]
        unsafe fn get_tensor_bytes_per_component1(
            self: &ICudaEngine,
            tensor_name: *const c_char,
        ) -> i32;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorBytesPerComponent"]
        unsafe fn get_tensor_bytes_per_component2(
            self: &ICudaEngine,
            tensor_name: *const c_char,
            profile_index: i32,
        ) -> i32;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorComponentsPerElement"]
        unsafe fn get_tensor_components_per_element1(
            self: &ICudaEngine,
            tensor_name: *const c_char,
        ) -> i32;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorComponentsPerElement"]
        unsafe fn get_tensor_components_per_element2(
            self: &ICudaEngine,
            tensor_name: *const c_char,
            profile_index: i32,
        ) -> i32;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorFormat"]
        unsafe fn get_tensor_format1(
            self: &ICudaEngine,
            tensor_name: *const c_char,
        ) -> TensorFormat;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorFormat"]
        unsafe fn get_tensor_format2(
            self: &ICudaEngine,
            tensor_name: *const c_char,
            profile_index: i32,
        ) -> TensorFormat;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorVectorizedDim"]
        unsafe fn get_tensor_vectorized_dim1(self: &ICudaEngine, tensor_name: *const c_char)
        -> i32;

        #[namespace = "nvinfer1"]
        #[cxx_name = "getTensorVectorizedDim"]
        unsafe fn get_tensor_vectorized_dim2(
            self: &ICudaEngine,
            tensor_name: *const c_char,
            profile_index: i32,
        ) -> i32;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_nb_optimization_profiles"]
        fn getNbOptimizationProfiles(self: &ICudaEngine) -> i32;

        unsafe fn engine_get_profile_shape(
            engine: &UniquePtr<ICudaEngine>,
            tensor_name: *const c_char,
            profile_index: i32,
            optimization_selector: OptProfileSelector,
        ) -> UniquePtr<Dims>;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_nb_io_tensors"]
        fn getNbIOTensors(self: &ICudaEngine) -> i32;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_io_tensor_name"]
        fn getIOTensorName(self: &ICudaEngine, index: i32) -> *const c_char;

        unsafe fn context_set_tensor_address(
            context: &UniquePtr<IExecutionContext>,
            tensor_name: *const c_char,
            addr: usize,
        ) -> bool;

        unsafe fn context_set_input_shape(
            context: &UniquePtr<IExecutionContext>,
            tensor_name: *const c_char,
            dims: &UniquePtr<Dims>,
        ) -> bool;

        #[namespace = "nvinfer1"]
        #[rust_name = "enqueue_v3"]
        unsafe fn enqueueV3(self: Pin<&mut IExecutionContext>, stream: *mut CUstream_st) -> bool;
    }
}

pub type LogLevel = ffi::Severity;
pub type DataType = ffi::DataType;
pub type TensorLocation = ffi::TensorLocation;
pub type TensorIoMode = ffi::TensorIoMode;
pub type TensorFormat = ffi::TensorFormat;
pub type OptProfileSelector = ffi::OptProfileSelector;

pub struct Dims {
    inner: UniquePtr<ffi::Dims>,
}

impl Dims {
    fn from_raw(dims: UniquePtr<ffi::Dims>) -> Self {
        Self { inner: dims }
    }

    pub fn new(dims_spec: &[i32]) -> TrtResult<Self> {
        Ok(Self {
            inner: ffi::new_dims(dims_spec)?,
        })
    }

    pub fn reshape(&mut self, dims_spec: &[i32]) -> TrtResult<()> {
        ffi::reshape_dims(&self.inner, dims_spec)?;
        Ok(())
    }
}

pub struct Engine {
    engine: UniquePtr<ffi::ICudaEngine>,
    _runtime: UniquePtr<ffi::IRuntime>,
    _logger: UniquePtr<ffi::Logger>,
}

impl Engine {
    pub fn new(path: std::path::PathBuf, log_level: LogLevel) -> TrtResult<Self> {
        let serialized_engine = std::fs::read(path)?;

        let logger = ffi::new_logger(log_level)?;
        let runtime = ffi::create_infer_runtime(&logger)?;
        let engine = ffi::runtime_deserialize_cuda_engine(&runtime, &serialized_engine)?;

        Ok(Self {
            engine,
            _runtime: runtime,
            _logger: logger,
        })
    }

    pub fn create_execution_context(&mut self) -> TrtResult<ExecutionContext> {
        let ctx = self
            .engine
            .as_mut()
            .expect("FFI: engine should be non-null")
            .create_execution_context(ffi::ExecutionContextAllocationStrategy::kSTATIC);

        if ctx.is_null() {
            return Err(TrtError::NullPtr("could not create execution context"));
        }

        Ok(ExecutionContext {
            ctx: unsafe { UniquePtr::from_raw(ctx) },
        })
    }

    pub fn get_tensor_shape(&self, tensor_name: &CStr) -> Dims {
        let dims = unsafe { ffi::engine_get_tensor_shape(&self.engine, tensor_name.as_ptr()) };
        Dims::from_raw(dims)
    }

    pub fn get_tensor_data_type(&self, tensor_name: &CStr) -> DataType {
        unsafe { self.engine.get_tensor_data_type(tensor_name.as_ptr()) }
    }

    pub fn get_tensor_location(&self, tensor_name: &CStr) -> TensorLocation {
        unsafe { self.engine.get_tensor_location(tensor_name.as_ptr()) }
    }

    pub fn get_tensor_io_mode(&self, tensor_name: &CStr) -> TensorIoMode {
        unsafe { self.engine.get_tensor_io_mode(tensor_name.as_ptr()) }
    }

    pub fn get_tensor_bytes_per_component(&self, tensor_name: &CStr) -> i32 {
        unsafe {
            self.engine
                .get_tensor_bytes_per_component1(tensor_name.as_ptr())
        }
    }

    pub fn get_tensor_bytes_per_component_for_profile(
        &self,
        tensor_name: &CStr,
        profile_index: i32,
    ) -> i32 {
        unsafe {
            self.engine
                .get_tensor_bytes_per_component2(tensor_name.as_ptr(), profile_index)
        }
    }

    pub fn get_tensor_components_per_element(&self, tensor_name: &CStr) -> i32 {
        unsafe {
            self.engine
                .get_tensor_components_per_element1(tensor_name.as_ptr())
        }
    }

    pub fn get_tensor_components_per_element_for_profile(
        &self,
        tensor_name: &CStr,
        profile_index: i32,
    ) -> i32 {
        unsafe {
            self.engine
                .get_tensor_components_per_element2(tensor_name.as_ptr(), profile_index)
        }
    }

    pub fn get_tensor_format(&self, tensor_name: &CStr) -> TensorFormat {
        unsafe { self.engine.get_tensor_format1(tensor_name.as_ptr()) }
    }

    pub fn get_tensor_format_for_profile(
        &self,
        tensor_name: &CStr,
        profile_index: i32,
    ) -> TensorFormat {
        unsafe {
            self.engine
                .get_tensor_format2(tensor_name.as_ptr(), profile_index)
        }
    }

    pub fn get_tensor_vectorized_dim(&self, tensor_name: &CStr) -> i32 {
        unsafe { self.engine.get_tensor_vectorized_dim1(tensor_name.as_ptr()) }
    }

    pub fn get_tensor_vectorized_dim_for_profile(
        &self,
        tensor_name: &CStr,
        profile_index: i32,
    ) -> i32 {
        unsafe {
            self.engine
                .get_tensor_vectorized_dim2(tensor_name.as_ptr(), profile_index)
        }
    }

    pub fn get_profile_count(&self) -> i32 {
        self.engine.get_nb_optimization_profiles()
    }

    pub fn get_profile_shape(
        &self,
        tensor_name: &CStr,
        profile_index: i32,
        optimization_selector: OptProfileSelector,
    ) -> Dims {
        let dims = unsafe {
            ffi::engine_get_profile_shape(
                &self.engine,
                tensor_name.as_ptr(),
                profile_index,
                optimization_selector,
            )
        };

        Dims::from_raw(dims)
    }

    pub fn get_io_tensor_count(&self) -> i32 {
        self.engine.get_nb_io_tensors()
    }

    pub fn get_tensor_name<'tensor>(&'tensor self, index: i32) -> &'tensor CStr {
        let tensor_name = self.engine.get_io_tensor_name(index);
        unsafe { CStr::from_ptr(tensor_name) }
    }
}

pub struct ExecutionContext {
    ctx: UniquePtr<ffi::IExecutionContext>,
}

impl ExecutionContext {
    pub fn set_tensor_address(
        &mut self,
        tensor_name: &CStr,
        addr: *const std::ffi::c_void,
    ) -> TrtResult<()> {
        let ok = unsafe {
            ffi::context_set_tensor_address(&self.ctx, tensor_name.as_ptr(), addr as usize)
        };

        ok.then_some(())
            .ok_or(TrtError::Api("could not set tensor address"))
    }

    pub fn set_input_shape(&mut self, tensor_name: &CStr, dims: Dims) -> TrtResult<()> {
        let ok =
            unsafe { ffi::context_set_input_shape(&self.ctx, tensor_name.as_ptr(), &dims.inner) };

        ok.then_some(())
            .ok_or(TrtError::Api("could not set input shape"))
    }

    pub fn enqueue(&mut self, stream: *mut c_void) -> TrtResult<()> {
        let ok = unsafe {
            self.ctx
                .as_mut()
                .expect("FFI: context should be non-null")
                .enqueue_v3(stream.cast())
        };

        ok.then_some(())
            .ok_or(TrtError::Api("could not enqueue inference"))
    }
}
