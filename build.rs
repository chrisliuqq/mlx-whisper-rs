fn main() {
    // 如果系統已安裝 MLX（例如 brew install mlx），直接 link，不從原始碼編譯
    // 查找順序：MLX_DIR 環境變數 → Homebrew → 自行編譯（fallback）

    let mlx_lib_dir = if let Ok(dir) = std::env::var("MLX_DIR") {
        // 使用者明確指定路徑
        Some(dir)
    } else if let Some(brew_prefix) = homebrew_prefix() {
        let candidate = format!("{brew_prefix}/lib");
        if std::path::Path::new(&format!("{candidate}/libmlx.dylib")).exists() {
            Some(candidate)
        } else {
            None
        }
    } else {
        None
    };

    if let Some(lib_dir) = mlx_lib_dir {
        // 找到系統 MLX，直接 link，告訴 mlx-sys 不要自行編譯
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib=mlx");
        println!("cargo:warning=mlx-whisper-rs: using system MLX from {lib_dir}");

        // 告訴 mlx-sys 的 build script 跳過編譯
        println!("cargo:rustc-env=MLX_SYS_USE_SYSTEM=1");
    } else {
        println!("cargo:warning=mlx-whisper-rs: MLX not found via Homebrew, building from source (slow first build)");
        println!("cargo:warning=Tip: run `brew install mlx` to skip source compilation");
    }
}

fn homebrew_prefix() -> Option<String> {
    // Apple Silicon Homebrew 預設路徑
    for path in ["/opt/homebrew", "/usr/local"] {
        if std::path::Path::new(&format!("{path}/lib/libmlx.dylib")).exists() {
            return Some(path.to_string());
        }
    }

    // 也試試 brew --prefix 指令
    std::process::Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| format!("{}/lib", s.trim()))
            } else {
                None
            }
        })
}
