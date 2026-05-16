use cxx::UniquePtr;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_void},
    sync::Arc,
};
use thiserror::Error;

/*
TensoRT implementation notes:
https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html

"The expected runtime concurrency model is that different threads operate in different execution contexts. The context contains the network state (activation values and so on) during execution, so using a context concurrently in different threads results in undefined behavior.

To support this model, the following operations are thread-safe:
    Nonmodifying operations on a runtime or engine.
    Deserializing an engine from a TensorRT runtime.
    Creating an execution context from an engine.
    Registering and deregistering plugins."
*/

const UNNAMED_TENSOR: &str = "[unnamed tensor]";

#[derive(Debug, Error)]
pub enum ShapeError {
    #[error("axis '{axis}' does not exist for rank-{rank} tensor")]
    NoSuchAxis { axis: &'static str, rank: usize },
}

#[derive(Debug, Error)]
pub enum TrtError {
    #[error("TensorRT error: {0}")]
    Cxx(#[from] cxx::Exception),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Shape error: {0}")]
    Shape(#[from] ShapeError),
    #[error("API error: {0}")]
    Api(String),
    #[error("Allocation error")]
    Alloc,
}

pub type TrtResult<T> = Result<T, TrtError>;

#[cxx::bridge]
mod ffi {
    #[repr(i32)]
    #[derive(Debug, Clone, Copy)]

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
    #[derive(Debug, Clone, Copy)]

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
    #[derive(Debug, Clone, Copy)]

    enum TensorLocation {
        #[cxx_name = "kDEVICE"]
        Device = 0,
        #[cxx_name = "kHOST"]
        Host = 1,
    }

    #[repr(i32)]
    #[namespace = "nvinfer1"]
    #[cxx_name = "TensorIOMode"]
    #[derive(Debug, Clone, Copy)]

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
    #[derive(Debug, Clone, Copy)]

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
    #[derive(Debug, Clone, Copy)]

    enum OptProfileSelector {
        #[cxx_name = "kMIN"]
        Min = 0,
        #[cxx_name = "kOPT"]
        Opt = 1,
        #[cxx_name = "kMAX"]
        Max = 2,
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
        fn dims_new(spec: &[i64]) -> Result<UniquePtr<Dims>>;
        fn dims_copy_from_slice(dims: &UniquePtr<Dims>, spec: &[i64]);
        fn dims_copy(src: &UniquePtr<Dims>, dest: &UniquePtr<Dims>);
        fn dims_as_slice(dims: &UniquePtr<Dims>) -> &[i64];
        fn dims_invalid() -> Result<UniquePtr<Dims>>;
        fn dims_clone(dims: &UniquePtr<Dims>) -> Result<UniquePtr<Dims>>;
        fn dims_nb_dims(dims: &UniquePtr<Dims>) -> i32;
        fn dims_get_axis(dims: &UniquePtr<Dims>, idx: usize) -> Result<i64>;
        fn dims_set_axis(dims: &UniquePtr<Dims>, idx: usize, val: i64) -> Result<()>;
        fn dims_is_invalid(dims: &UniquePtr<Dims>) -> bool;
        fn dims_is_unknown_rank(dims: &UniquePtr<Dims>) -> bool;

        fn logger_new(log_level: Severity) -> Result<UniquePtr<Logger>>;

        fn create_infer_runtime(logger: &UniquePtr<Logger>) -> Result<UniquePtr<IRuntime>>;

        fn runtime_deserialize_cuda_engine(
            runtime: &UniquePtr<IRuntime>,
            data: &[u8],
        ) -> Result<UniquePtr<ICudaEngine>>;

        fn engine_create_execution_context(
            engine: &UniquePtr<ICudaEngine>,
        ) -> Result<UniquePtr<IExecutionContext>>;

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

        #[namespace = "nvinfer1"]
        #[rust_name = "set_optimization_profile_async"]
        unsafe fn setOptimizationProfileAsync(
            self: Pin<&mut IExecutionContext>,
            profile_index: i32,
            stream: *mut CUstream_st,
        ) -> bool;

        #[namespace = "nvinfer1"]
        #[rust_name = "get_max_output_size"]
        unsafe fn getMaxOutputSize(self: &IExecutionContext, tensor_name: *const c_char) -> i64;
    }
}

pub type LogLevel = ffi::Severity;
pub type DataType = ffi::DataType;
pub type TensorLocation = ffi::TensorLocation;
pub type TensorIoMode = ffi::TensorIoMode;
pub type TensorFormat = ffi::TensorFormat;
pub type OptProfileSelector = ffi::OptProfileSelector;

impl DataType {
    pub fn size_bytes(self) -> usize {
        match self {
            DataType::Float => 4,
            DataType::Half => 2,
            DataType::Int8 => 1,
            DataType::Int32 => 4,
            DataType::Bool => 1,
            DataType::Uint8 => 1,
            DataType::Fp8 => 1,
            DataType::Bf16 => 2,
            DataType::Int64 => 8,
            DataType::Int4 => 0, // special case
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Axis {
    Batch,
    Channel,
    Depth,
    Height,
    Width,
}

impl Axis {
    pub fn name(self) -> &'static str {
        match self {
            Axis::Batch => "batch",
            Axis::Channel => "channel",
            Axis::Depth => "depth",
            Axis::Height => "height",
            Axis::Width => "width",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AxisLayout {
    N,
    NC,
    NCL, // 1D conv
    NCHW,
    NHWC,
    NCDHW,
    NDHWC,
}

impl AxisLayout {
    fn from_format_and_rank(format: TensorFormat, rank: i32) -> Self {
        match format {
            // 4D, channel-minor
            TensorFormat::Hwc | TensorFormat::Hwc8 | TensorFormat::Hwc16 => Self::NHWC,

            // volumetric, channel-minor
            TensorFormat::Dhwc | TensorFormat::Dhwc8 => Self::NDHWC,

            // volumetric, channel-major
            TensorFormat::Cdhw32 => Self::NCDHW,

            // linear & CHW* variants, channel-major
            TensorFormat::Linear
            | TensorFormat::Chw2
            | TensorFormat::Chw4
            | TensorFormat::Chw16
            | TensorFormat::Chw32
            | TensorFormat::DlaLinear
            | TensorFormat::DlaHwc4 => match rank {
                1 => Self::N,
                2 => Self::NC,
                3 => Self::NCL,
                4 => Self::NCHW,
                _ => Self::NCHW, // best effort
            },

            _ => Self::N, // best effort for int min/max (not a real enum)
        }
    }

    fn index_of(self, axis: Axis) -> Option<usize> {
        match (self, axis) {
            (_, Axis::Batch) => Some(0), // batch precedes all axes

            (Self::NCHW | Self::NC | Self::NCL | Self::NCDHW, Axis::Channel) => Some(1),
            (Self::NHWC, Axis::Channel) => Some(3),
            (Self::NDHWC, Axis::Channel) => Some(4),

            (Self::NCHW, Axis::Height) => Some(2),
            (Self::NHWC, Axis::Height) => Some(1),
            (Self::NCDHW, Axis::Height) => Some(3),
            (Self::NDHWC, Axis::Height) => Some(2),
            (Self::NCL, Axis::Height) => Some(2), // "length"

            (Self::NCHW, Axis::Width) => Some(3),
            (Self::NHWC, Axis::Width) => Some(2),
            (Self::NCDHW, Axis::Width) => Some(4),
            (Self::NDHWC, Axis::Width) => Some(3),

            (Self::NCDHW, Axis::Depth) => Some(2),
            (Self::NDHWC, Axis::Depth) => Some(1),

            _ => None,
        }
    }
}

pub struct Shape {
    dims: UniquePtr<ffi::Dims>,
    layout: AxisLayout,
}

impl Shape {
    pub fn try_new(spec: &[i64], format: TensorFormat) -> TrtResult<Self> {
        let rank = spec.len();
        let dims = ffi::dims_new(spec)?;
        let layout = AxisLayout::from_format_and_rank(format, rank as i32);

        Ok(Self { dims, layout })
    }

    fn from_dims(dims: UniquePtr<ffi::Dims>, format: TensorFormat) -> Self {
        let rank = ffi::dims_nb_dims(&dims);
        let layout = AxisLayout::from_format_and_rank(format, rank);

        Self { dims, layout }
    }

    pub fn copy(&mut self, other: &Shape) -> TrtResult<()> {
        if self.layout != other.layout {
            return Err(TrtError::Api("shape ranks do not match".to_owned()));
        }

        ffi::dims_copy(&other.dims, &self.dims);
        Ok(())
    }

    pub fn as_slice(&self) -> &[i64] {
        ffi::dims_as_slice(&self.dims)
    }

    pub fn try_clone(&self) -> TrtResult<Self> {
        let dims = ffi::dims_clone(&self.dims)?;
        Ok(Self {
            dims,
            layout: self.layout,
        })
    }

    pub fn is_invalid(&self) -> bool {
        ffi::dims_is_invalid(&self.dims)
    }

    pub fn is_unknown_rank(&self) -> bool {
        ffi::dims_is_unknown_rank(&self.dims)
    }

    pub fn rank(&self) -> usize {
        ffi::dims_nb_dims(&self.dims) as usize
    }

    pub fn get(&self, idx: usize) -> TrtResult<i64> {
        Ok(ffi::dims_get_axis(&self.dims, idx)?)
    }

    pub fn set(&mut self, idx: usize, value: i64) -> TrtResult<()> {
        Ok(ffi::dims_set_axis(&self.dims, idx, value)?)
    }

    fn get_axis(&self, axis: Axis) -> TrtResult<i64> {
        let idx = self.layout.index_of(axis).ok_or(ShapeError::NoSuchAxis {
            axis: axis.name(),
            rank: self.rank(),
        })?;

        self.get(idx)
    }

    fn set_axis(&mut self, axis: Axis, value: i64) -> TrtResult<()> {
        let idx = self.layout.index_of(axis).ok_or(ShapeError::NoSuchAxis {
            axis: axis.name(),
            rank: self.rank(),
        })?;

        self.set(idx, value)
    }

    pub fn batch(&self) -> TrtResult<i64> {
        self.get_axis(Axis::Batch)
    }

    pub fn channels(&self) -> TrtResult<i64> {
        self.get_axis(Axis::Channel)
    }

    pub fn depth(&self) -> TrtResult<i64> {
        self.get_axis(Axis::Depth)
    }

    pub fn height(&self) -> TrtResult<i64> {
        self.get_axis(Axis::Height)
    }

    pub fn width(&self) -> TrtResult<i64> {
        self.get_axis(Axis::Width)
    }

    pub fn set_batch(&mut self, value: i64) -> TrtResult<()> {
        self.set_axis(Axis::Batch, value)
    }

    pub fn set_channels(&mut self, value: i64) -> TrtResult<()> {
        self.set_axis(Axis::Channel, value)
    }

    pub fn set_depth(&mut self, value: i64) -> TrtResult<()> {
        self.set_axis(Axis::Depth, value)
    }

    pub fn set_height(&mut self, value: i64) -> TrtResult<()> {
        self.set_axis(Axis::Height, value)
    }

    pub fn set_width(&mut self, value: i64) -> TrtResult<()> {
        self.set_axis(Axis::Width, value)
    }

    pub fn set_nchw(&mut self, n: i64, c: i64, h: i64, w: i64) -> TrtResult<()> {
        self.set_batch(n)?;
        self.set_channels(c)?;
        self.set_height(h)?;
        self.set_width(w)
    }

    pub fn numel(&self) -> Option<usize> {
        (0..self.rank()).try_fold(1usize, |acc, i| {
            let value = self.get(i).ok()?;
            if value < 0 {
                return None;
            }
            acc.checked_mul(value as usize)
        })
    }

    pub fn numel_except(&self, axis: Axis) -> Option<usize> {
        let skip_idx = self.layout.index_of(axis);

        (0..self.rank()).try_fold(1usize, |acc, i| {
            if Some(i) == skip_idx {
                return Some(acc);
            }
            let value = self.get(i).ok()?;
            if value < 0 {
                return None;
            }
            acc.checked_mul(value as usize)
        })
    }

    pub fn copy_from_slice(&mut self, spec: &[i64]) {
        ffi::dims_copy_from_slice(&self.dims, spec);
    }
}

pub struct ProfileShapes {
    min: Shape,
    opt: Shape,
    max: Shape,
}

impl ProfileShapes {
    pub fn min(&self) -> &Shape {
        &self.min
    }

    pub fn opt(&self) -> &Shape {
        &self.opt
    }

    pub fn max(&self) -> &Shape {
        &self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

pub struct TensorInfo {
    id: TensorId,
    name: String,
    ffi_name: CString,
    dtype: DataType,
    io_mode: TensorIoMode,
    location: TensorLocation,
    default_shape: Shape,
    profiles: Vec<ProfileShapes>,
}

impl TensorInfo {
    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn ffi_name(&self) -> &CStr {
        &self.ffi_name
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn io_mode(&self) -> TensorIoMode {
        self.io_mode
    }

    pub fn location(&self) -> TensorLocation {
        self.location
    }

    pub fn is_input(&self) -> bool {
        self.io_mode == TensorIoMode::Input
    }

    pub fn is_output(&self) -> bool {
        self.io_mode == TensorIoMode::Output
    }

    pub fn has_dynamic_axes(&self) -> bool {
        self.profiles.iter().any(|p| p.opt.numel().is_none())
    }

    pub fn default_shape(&self) -> &Shape {
        &self.default_shape
    }

    pub fn profile(&self, idx: usize) -> Option<&ProfileShapes> {
        self.profiles.get(idx)
    }
}

pub struct Engine {
    tensor_ids: HashMap<String, TensorId>,
    only_input_id: Option<TensorId>,  // fast path
    only_output_id: Option<TensorId>, // fast path
    tensors: Vec<TensorInfo>,
    engine: UniquePtr<ffi::ICudaEngine>,
    _runtime: UniquePtr<ffi::IRuntime>,
    _logger: UniquePtr<ffi::Logger>,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    pub fn try_new<P: AsRef<std::path::Path>>(
        path: P,
        log_level: LogLevel,
    ) -> TrtResult<Arc<Self>> {
        let serialized_engine = std::fs::read(path)?;
        let logger = ffi::logger_new(log_level)?;
        let runtime = ffi::create_infer_runtime(&logger)?;
        let engine = ffi::runtime_deserialize_cuda_engine(&runtime, &serialized_engine)?;

        // snapshot all tensor info
        let opt_profile_count = engine.get_nb_optimization_profiles();
        let tensor_count = engine.get_nb_io_tensors();

        let mut tensor_ids = HashMap::with_capacity(tensor_count as usize);
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let id = TensorId(i as usize);

            // get name
            let ffi_name = engine.get_io_tensor_name(i);
            let ffi_name = unsafe { CStr::from_ptr(ffi_name) };
            let ffi_name = ffi_name.to_owned();
            let name = ffi_name.to_str().unwrap_or(UNNAMED_TENSOR).to_owned();

            // other attributes
            let mut profiles = Vec::new();
            for profile_index in 0..opt_profile_count {
                let [min, opt, max] = [
                    OptProfileSelector::Min,
                    OptProfileSelector::Opt,
                    OptProfileSelector::Max,
                ]
                .map(|selector| {
                    let dims = unsafe {
                        ffi::engine_get_profile_shape(
                            &engine,
                            ffi_name.as_ptr(),
                            profile_index,
                            selector,
                        )
                    };
                    let format =
                        unsafe { engine.get_tensor_format2(ffi_name.as_ptr(), profile_index) };

                    Shape::from_dims(dims, format)
                });

                profiles.push(ProfileShapes { min, opt, max });
            }

            let default_shape = Shape::from_dims(
                unsafe { ffi::engine_get_tensor_shape(&engine, ffi_name.as_ptr()) },
                unsafe { engine.get_tensor_format1(ffi_name.as_ptr()) },
            );

            tensor_ids.insert(name.clone(), id);
            tensors.push(TensorInfo {
                id,
                dtype: unsafe { engine.get_tensor_data_type(ffi_name.as_ptr()) },
                io_mode: unsafe { engine.get_tensor_io_mode(ffi_name.as_ptr()) },
                location: unsafe { engine.get_tensor_location(ffi_name.as_ptr()) },
                name,
                ffi_name,
                default_shape,
                profiles,
            })
        }

        let input_ids: Vec<_> = tensors
            .iter()
            .filter(|t| t.is_input())
            .map(|t| t.id())
            .collect();

        let output_ids: Vec<_> = tensors
            .iter()
            .filter(|t| t.is_output())
            .map(|t| t.id())
            .collect();

        let only_input_id = match input_ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        };

        let only_output_id = match output_ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        };

        Ok(Arc::new(Self {
            only_input_id,
            only_output_id,
            tensor_ids,
            tensors,
            engine,
            _runtime: runtime,
            _logger: logger,
        }))
    }

    pub fn opt_profile_count(&self) -> usize {
        self.engine.get_nb_optimization_profiles() as usize
    }

    pub fn only_input_tensor(&self) -> Option<&TensorInfo> {
        self.only_input_id.and_then(|id| self.tensor(id))
    }

    pub fn only_output_tensor(&self) -> Option<&TensorInfo> {
        self.only_output_id.and_then(|id| self.tensor(id))
    }

    pub fn lookup_tensor_id(&self, name: &str) -> Option<TensorId> {
        Some(*self.tensor_ids.get(name)?)
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn inputs(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors
            .iter()
            .filter(|t| t.io_mode == TensorIoMode::Input)
    }

    pub fn outputs(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors
            .iter()
            .filter(|t| t.io_mode == TensorIoMode::Output)
    }

    pub fn tensor(&self, id: TensorId) -> Option<&TensorInfo> {
        self.tensors.get(id.0)
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        Some(self.tensors.get(self.tensor_ids.get(name)?.0)?)
    }
}

pub trait DeviceBuffer: Drop {
    fn ptr(&self) -> *mut c_void;
    fn size_bytes(&self) -> usize;
}

pub trait DeviceAllocator {
    type Buffer: DeviceBuffer;
    fn allocate(&mut self, size_bytes: usize) -> TrtResult<Self::Buffer>;
}

pub struct TensorBinding<B: DeviceBuffer> {
    tensor_id: TensorId,
    buffer: B,
    scratch_shape: Shape,
    runtime_shape: Shape,
}

impl<B: DeviceBuffer> TensorBinding<B> {
    pub fn buffer(&self) -> &B {
        &self.buffer
    }

    pub fn runtime_shape(&self) -> &Shape {
        &self.runtime_shape
    }
}

pub struct ExecutionContext<A: DeviceAllocator> {
    active_profile: usize,
    allocator: A,
    bindings: Vec<Option<TensorBinding<A::Buffer>>>,
    ctx: UniquePtr<ffi::IExecutionContext>,
    engine: Arc<Engine>,
}

impl<A: DeviceAllocator> ExecutionContext<A> {
    pub fn try_new(engine: &Arc<Engine>, allocator: A) -> TrtResult<Self> {
        // createExecutionContext is non-const but documented as threadsafe
        let ctx = ffi::engine_create_execution_context(&engine.engine)?;

        Ok(Self {
            active_profile: 0,
            allocator,
            bindings: std::iter::repeat_with(|| None)
                .take(engine.tensors().len())
                .collect(),
            ctx,
            engine: Arc::clone(engine),
        })
    }

    pub fn allocate_max(&mut self) -> TrtResult<()> {
        for tensor in self.engine.tensors() {
            let profile = tensor
                .profile(self.active_profile)
                .ok_or(TrtError::Api(format!(
                    "could not find profile {} for tensor with name '{}'",
                    self.active_profile,
                    tensor.name(),
                )))?;

            let max_bytes = if tensor.is_input() {
                profile
                    .max
                    .numel()
                    .map(|n| n * tensor.dtype().size_bytes())
                    .ok_or(TrtError::Api(format!(
                        "could not find max size for tensor '{}'",
                        tensor.name(),
                    )))?
            } else {
                unsafe { self.ctx.get_max_output_size(tensor.ffi_name().as_ptr()) as usize }
            };

            let buffer = self.allocator.allocate(max_bytes)?;

            unsafe {
                ffi::context_set_tensor_address(
                    &self.ctx,
                    tensor.ffi_name().as_ptr(),
                    buffer.ptr() as usize,
                )
                .then_some(())
                .ok_or(TrtError::Api(format!(
                    "could not set address for tensor with name '{}'",
                    tensor.name()
                )))?;
            }

            // old bindings will get dropped if they exist
            self.bindings[tensor.id().0] = Some(TensorBinding {
                tensor_id: tensor.id(),
                buffer,
                scratch_shape: profile.opt.try_clone()?,
                runtime_shape: profile.opt.try_clone()?,
            });
        }

        Ok(())
    }

    pub fn only_input_binding(&self) -> Option<&TensorBinding<A::Buffer>> {
        self.engine.only_input_id.and_then(|id| self.binding(id))
    }

    pub fn only_output_binding(&self) -> Option<&TensorBinding<A::Buffer>> {
        self.engine.only_input_id.and_then(|id| self.binding(id))
    }

    pub fn binding(&self, id: TensorId) -> Option<&TensorBinding<A::Buffer>> {
        self.bindings.get(id.0)?.as_ref()
    }

    pub fn binding_by_name(&self, name: &str) -> Option<&TensorBinding<A::Buffer>> {
        self.binding(self.engine.lookup_tensor_id(name)?)
    }

    pub fn modify_input_shape<F>(&mut self, id: TensorId, f: F) -> TrtResult<()>
    where
        F: FnOnce(&mut Shape) -> TrtResult<()>,
    {
        let binding = self
            .bindings
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or(TrtError::Api(format!(
                "could not find binding with ID {}",
                id.0
            )))?;

        let tensor = self
            .engine
            .tensors
            .get(binding.tensor_id.0)
            .ok_or(TrtError::Api(format!(
                "could not match binding ID {} with tensor",
                id.0
            )))?;

        binding.scratch_shape.copy(&binding.runtime_shape)?;
        f(&mut binding.scratch_shape)?;

        unsafe {
            ffi::context_set_input_shape(
                &self.ctx,
                tensor.ffi_name().as_ptr(),
                &binding.scratch_shape.dims,
            )
            .then_some(())
            .ok_or(TrtError::Api(format!(
                "could not set shape for tensor with ID {}",
                id.0
            )))?;
        }

        // commit result
        binding.runtime_shape.copy(&binding.scratch_shape)?;
        Ok(())
    }

    pub fn modify_named_input_shape<F>(&mut self, name: &str, f: F) -> TrtResult<()>
    where
        F: FnOnce(&mut Shape) -> TrtResult<()>,
    {
        self.modify_input_shape(
            self.engine
                .lookup_tensor_id(name)
                .ok_or(TrtError::Api(format!(
                    "could not find ID for tensor with name '{}'",
                    name
                )))?,
            f,
        )
    }

    pub fn set_input_shape(&mut self, id: TensorId, spec: &[i64]) -> TrtResult<()> {
        self.modify_input_shape(id, |shape| {
            shape.copy_from_slice(spec);
            Ok(())
        })
    }

    pub fn set_named_input_shape(&mut self, name: &str, spec: &[i64]) -> TrtResult<()> {
        self.modify_named_input_shape(name, |shape| {
            shape.copy_from_slice(spec);
            Ok(())
        })
    }

    pub unsafe fn enqueue(&mut self, stream: *mut c_void) -> TrtResult<()> {
        unsafe {
            self.ctx
                .as_mut()
                .expect("ffi: context should be non-null")
                .enqueue_v3(stream.cast())
                .then_some(())
                .ok_or(TrtError::Api("enqueue failed".to_owned()))
        }
    }

    pub unsafe fn set_opt_profile_async(
        &mut self,
        profile_index: i32,
        stream: *mut c_void,
    ) -> TrtResult<()> {
        unsafe {
            self.ctx
                .as_mut()
                .expect("ffi: context should be non-null")
                .set_optimization_profile_async(profile_index, stream.cast())
                .then_some(())
                .ok_or(TrtError::Api("profile select failed".to_owned()))?;
        }

        self.active_profile = profile_index as usize;

        Ok(())
    }
}
