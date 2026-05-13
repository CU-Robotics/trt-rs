use std::env;
use std::path::PathBuf;

fn linux_multiarch_gnu() -> Option<&'static str> {
    let os = std::env::var("CARGO_CFG_TARGET_OS").ok()?;
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").ok()?;
    let env_abi = std::env::var("CARGO_CFG_TARGET_ENV").ok()?;

    match (arch.as_str(), os.as_str(), env_abi.as_str()) {
        ("x86_64", "linux", "gnu") => Some("x86_64-linux-gnu"),
        ("aarch64", "linux", "gnu") => Some("aarch64-linux-gnu"),
        _ => None,
    }
}

fn main() {
    // Re-run build if source changes
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=include/");

    // Linking
    let target = linux_multiarch_gnu().expect("unrecognized or unsupported target");

    println!("cargo:rustc-link-search=/usr/lib/{target}");
    println!("cargo:rustc-link-lib=nvinfer");

    // CXX build
    cxx_build::bridge("src/lib.rs")
        .file("src/trt.cc")
        .cpp(true)
        .std("c++20")
        .include("include")
        .include("/usr/local/cuda/include")
        .compile("trt-rs");
}
