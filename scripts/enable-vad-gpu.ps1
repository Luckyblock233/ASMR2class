# 启用/修复 whisper_asmr VAD 的 GPU 推理（onnxruntime-gpu 1.26，CUDA 12 + cuDNN 9）
#
# 用法（在仓库根目录的 PowerShell 中）：
#   powershell -ExecutionPolicy Bypass -File scripts\enable-vad-gpu.ps1
#
# 原理：pyproject.toml 已用 override-dependencies 把 faster-whisper 传递依赖的 CPU 版
# onnxruntime 限定为非 Windows 平台，Windows 上锁定安装 onnxruntime-gpu==1.26.*，
# 所以 uv sync 本身就是安装步骤，且之后任意 uv run / uv sync 都不会再装回 CPU 版。
# 本脚本 = uv sync + 验证；仅当历史上手动换装残留了被覆盖的文件时才走修复分支。

Set-Location (Split-Path -Parent $PSScriptRoot)  # 仓库根目录

function Test-CudaEP {
    .venv\Scripts\python -c "import onnxruntime as ort; ps = ort.get_available_providers(); print('可用执行提供者:', ps); exit(0 if 'CUDAExecutionProvider' in ps else 1)"
    return ($LASTEXITCODE -eq 0)
}

Write-Host "==> uv sync（Windows 上锁定安装 onnxruntime-gpu 1.26 + CUDA 12 组件库）"
uv sync
if ($LASTEXITCODE -ne 0) { Write-Host "✗ uv sync 失败"; exit 1 }

if (Test-CudaEP) {
    Write-Host "`n✓ GPU 版生效。启动时看到「VAD：whisper_asmr（CUDAExecutionProvider）」即 GPU 推理。"
    exit 0
}

Write-Host "`n==> CUDA EP 不可用，可能存在历史手动换装的文件残留，执行修复"
uv pip uninstall onnxruntime onnxruntime-gpu
uv pip install "onnxruntime-gpu==1.26.*" nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12

if (Test-CudaEP) {
    Write-Host "`n✓ 修复完成，GPU 版生效。"
} else {
    Write-Host "`n✗ CUDA EP 仍不可用，请把上面的输出贴给维护者排查。"
    exit 1
}
