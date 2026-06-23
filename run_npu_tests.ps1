#!/usr/bin/env pwsh
# ==============================================================================
# NPU Verilator 自动化测试脚本
#
# 功能:
#   1. 初始化 oss-cad-suite (Verilator + firtool) 和 MinGW64 (gcc) 环境
#   2. 生成 NPU Verilog (通过 Bazel EmitNpuVerilog + firtool)
#   3. 使用 Verilator 编译 Verilog + C++ 测试程序 (功能/性能/边界)
#   4. 运行仿真并收集测试结果
#   5. 调用 generate_report.py 生成 Markdown 测试报告
#
# 用法:
#   .\run_npu_tests.ps1                          # 运行全部测试
#   .\run_npu_tests.ps1 -TestName functional     # 仅运行功能测试
#   .\run_npu_tests.ps1 -TestName performance    # 仅运行性能测试
#   .\run_npu_tests.ps1 -TestName boundary       # 仅运行边界测试
#   .\run_npu_tests.ps1 -SkipVerilogGen          # 跳过 Verilog 生成
#   .\run_npu_tests.ps1 -SkipCompile             # 跳过 Verilator 编译
#   .\run_npu_tests.ps1 -EnableVcd               # 启用 VCD 波形输出
#
# 参考: tests/gpu_sdl3_demo/run_verilator_tests.ps1
# ==============================================================================

param(
[string]$TestName = "all",  # all, functional, performance, boundary
[switch]$SkipVerilogGen = $false,
[switch]$SkipCompile = $false,
[switch]$EnableVcd = $false
)

$ErrorActionPreference = "Stop"

# ==============================================================================
# 0. 全局变量与辅助函数
# ==============================================================================

$ScriptStartTime = Get-Date
$ScriptExitCode = 0

function Write-Info  { param([string]$Msg) Write-Host "[INFO] $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "[ERROR] $Msg" -ForegroundColor Red }
function Write-Step { param([string]$Msg) Write-Host "[STEP] $Msg" -ForegroundColor Cyan }

# 测试配置表: 名称 -> (testbench 文件, 可执行文件名, 日志文件名, 描述)
$TestConfig = @{
  "functional"  = @{
    TbFile   = "npu_test_tb.cpp"
    ExeName  = "Vnpu_test_tb.exe"
    LogFile  = "npu_functional.log"
    Desc     = "功能测试"
  }
  "performance" = @{
    TbFile   = "npu_perf_tb.cpp"
    ExeName  = "Vnpu_perf_tb.exe"
    LogFile  = "npu_performance.log"
    Desc     = "性能测试"
  }
  "boundary"    = @{
    TbFile   = "npu_boundary_tb.cpp"
    ExeName  = "Vnpu_boundary_tb.exe"
    LogFile  = "npu_boundary.log"
    Desc     = "边界测试"
  }
}

# 根据参数选择要运行的测试
function Get-SelectedTests {
  if ($TestName -eq "all") {
    return @("functional", "performance", "boundary")
    } elseif ($TestConfig.ContainsKey($TestName)) {
    return @($TestName)
    } else {
    Write-Err "Unknown TestName: $TestName (valid: all, functional, performance, boundary)"
    exit 1
  }
}

# ==============================================================================
# 1. 环境初始化
# ==============================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  NPU Verilator Automated Test Script" -ForegroundColor Cyan
Write-Host "  Palingenesis EtherealBecoming Project" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# --- 1.1 oss-cad-suite (Verilator + firtool) ---
$OSS_CAD_SUITE = "D:\oss-cad-suite"
if (-not (Test-Path "$OSS_CAD_SUITE\bin\verilator.exe")) {
  Write-Err "Verilator not found at $OSS_CAD_SUITE\bin\verilator.exe"
  Write-Warn "Please run D:\oss-cad-suite\environment.ps1 first or verify installation."
  exit 1
}
$envFile = "$OSS_CAD_SUITE\environment.ps1"
if (Test-Path $envFile) {
  & $envFile
}
$env:YOSYSHQ_ROOT = "$OSS_CAD_SUITE\"
$env:PATH = "$OSS_CAD_SUITE\bin;$OSS_CAD_SUITE\lib;$env:PATH"
Write-Info "oss-cad-suite initialized: $OSS_CAD_SUITE"

# --- 1.2 MSYS2 usr/bin (提供 make, uname 等 Unix 工具) ---
$MSYS2 = "C:\msys64"
if (Test-Path "$MSYS2\usr\bin\make.exe") {
  $env:PATH = "$MSYS2\usr\bin;$env:PATH"
  Write-Info "MSYS2 usr/bin initialized (make, uname): $MSYS2\usr\bin"
  } else {
  Write-Err "make not found at $MSYS2\usr\bin\make.exe"
  Write-Warn "Please install MSYS2 or verify installation."
  exit 1
}

# --- 1.3 MinGW64 (gcc/g++ for C++ compilation) ---
$MinGW64 = "C:\mingw64"
if (-not (Test-Path "$MinGW64\bin\gcc.exe")) {
  # 尝试 MSYS2 内置的 mingw64
  $MsysMinGw = "$MSYS2\mingw64\bin"
  if (Test-Path "$MsysMinGw\gcc.exe") {
    $MinGW64 = "$MSYS2\mingw64"
    } else {
    Write-Err "MinGW64 gcc not found."
    Write-Warn "Please run C:\mingw64\mingwvars.bat first or verify installation."
    exit 1
  }
}
$env:PATH = "$MinGW64\bin;$env:PATH"
$env:CXX = "g++"
$env:CC  = "gcc"
Write-Info "MinGW64 initialized: $MinGW64"

# --- 1.4 验证工具链 ---
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$verilatorVer = (& verilator --version 2>&1 | Select-Object -Last 1)
$gccVer       = (& gcc --version 2>&1 | Select-Object -First 1)
$makeVer      = (& make --version 2>&1 | Select-Object -First 1)
$firtoolVer   = (& firtool --version 2>&1 | Select-Object -First 1)
$ErrorActionPreference = $prevEAP
Write-Host "[INFO] Verilator: $verilatorVer" -ForegroundColor Yellow
Write-Host "[INFO] C++ Compiler: $gccVer" -ForegroundColor Yellow
Write-Host "[INFO] Make: $makeVer" -ForegroundColor Yellow
Write-Host "[INFO] firtool: $firtoolVer" -ForegroundColor Yellow
Write-Host ""

# ==============================================================================
# 路径设置
# ==============================================================================

$PROJECT_ROOT  = "c:\Users\admin\Documents\trae_projects\Palingenesis\EtherealBecoming"
$TEST_DIR      = "$PROJECT_ROOT\tests\npu_verilator_sim"
$SRC_DIR       = "$TEST_DIR\src"
$BUILD_DIR     = "$TEST_DIR\build"
$OBJ_DIR       = "$BUILD_DIR\obj_dir"
$REPORT_DIR    = "$TEST_DIR\reports"
$VERILOG_DIR   = "$BUILD_DIR\verilog"
$CHIRRTL_DIR   = "$PROJECT_ROOT\generated"

# NPU Verilog 模块列表 (由 EmitNpuVerilog 生成)
$NpuModules = @(
"NpuTop",
"NpuMultiCore",
"SystolicCell",
"EccCodec",
"NpuSram"
)

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BUILD_DIR  | Out-Null
New-Item -ItemType Directory -Force -Path $OBJ_DIR    | Out-Null
New-Item -ItemType Directory -Force -Path $REPORT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $VERILOG_DIR | Out-Null

Write-Info "Project Root: $PROJECT_ROOT"
Write-Info "Test Dir:     $TEST_DIR"
Write-Info "Build Dir:    $BUILD_DIR"
Write-Info "Report Dir:   $REPORT_DIR"
Write-Host ""

# ==============================================================================
# 2. 生成 NPU Verilog
# ==============================================================================

if (-not $SkipVerilogGen) {
  Write-Host "================================================================" -ForegroundColor Cyan
  Write-Host "  Phase 1: Generate NPU Verilog" -ForegroundColor Cyan
  Write-Host "================================================================" -ForegroundColor Cyan
  Write-Host ""

  # --- 2.1 通过 Bazel 运行 EmitNpuVerilog 生成 CHIRRTL IR (.fir) ---
  Write-Step "Generating CHIRRTL IR via Bazel (EmitNpuVerilog)..."

  Push-Location $PROJECT_ROOT
  try {
    $prevEAP2 = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & bazel run //hdl/chisel/src:EmitNpuVerilog_emit_verilog -- --output-dir "$PROJECT_ROOT\generated" 2>&1 | ForEach-Object {
      $line = $_
      if ($line -match "error|Error|ERROR") {
        Write-Host $line -ForegroundColor Red
        } elseif ($line -match "warning|Warning|WARNING") {
        Write-Host $line -ForegroundColor Yellow
        } else {
        Write-Host $line -ForegroundColor Gray
      }
    }
    $bazelExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP2

    if ($bazelExitCode -ne 0) {
      Write-Err "Bazel EmitNpuVerilog failed (exit code: $bazelExitCode)"
      Write-Warn "Trying direct scala invocation as fallback..."
      # Fallback: 直接运行 scala (如果 classpath 已配置)
      # 此处保留扩展点, 实际部署时可根据环境补充
      Pop-Location
      exit 1
    }
    Pop-Location
    } catch {
    Pop-Location
    Write-Err "Exception during Bazel run: $_"
    exit 1
  }
  Write-Info "CHIRRTL IR generation completed."
  Write-Host ""

  # --- 2.2 使用 firtool 将 .fir 转换为 .sv ---
  Write-Step "Converting CHIRRTL IR to SystemVerilog via firtool..."

  foreach ($mod in $NpuModules) {
    $firFile = "$CHIRRTL_DIR\$mod.fir"
    $svFile  = "$VERILOG_DIR\$mod.sv"

    if (-not (Test-Path $firFile)) {
      Write-Err "CHIRRTL IR not found: $firFile"
      exit 1
    }

    $prevEAP3 = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    # --default-layer-specialization=disable: 禁用验证层(assert/cover)分离到单独include文件
    # 避免 Verilator 找不到 layers-*-Verification.sv include 文件
    & firtool --verilog $firFile -o $svFile --default-layer-specialization=disable 2>&1 | ForEach-Object {
      Write-Host $_ -ForegroundColor Gray
    }
    $firtoolExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP3

    if ($firtoolExitCode -ne 0 -or -not (Test-Path $svFile)) {
      Write-Err "firtool conversion failed for $mod (exit code: $firtoolExitCode)"
      exit 1
    }
    $fileSize = (Get-Item $svFile).Length
    Write-Host ("  {0,-20} -> {1} bytes" -f $mod, $fileSize) -ForegroundColor Green
  }
  Write-Info "SystemVerilog generation completed."
  Write-Host ""
  } else {
  Write-Warn "Skipping Verilog generation (-SkipVerilogGen)"
  Write-Host ""
}

# ==============================================================================
# 3. Verilator 编译
# ==============================================================================

$CompiledTests = @{}

if (-not $SkipCompile) {
  Write-Host "================================================================" -ForegroundColor Cyan
  Write-Host "  Phase 2: Verilator Compilation" -ForegroundColor Cyan
  Write-Host "================================================================" -ForegroundColor Cyan
  Write-Host ""

  # 清理旧的 obj_dir
  if (Test-Path $OBJ_DIR) {
    Remove-Item -Recurse -Force $OBJ_DIR
  }
  New-Item -ItemType Directory -Force -Path $OBJ_DIR | Out-Null

  $selectedTests = Get-SelectedTests

  foreach ($testKey in $selectedTests) {
    $cfg = $TestConfig[$testKey]
    $tbFile = "$SRC_DIR\$($cfg.TbFile)"
    $apiFile = "$SRC_DIR\npu_verilator_api.cpp"
    $exePath = "$OBJ_DIR\$($cfg.ExeName)"

    Write-Step "Compiling $($cfg.Desc) ($($cfg.TbFile))..."

    if (-not (Test-Path $tbFile)) {
      Write-Err "Testbench not found: $tbFile"
      exit 1
    }
    if (-not (Test-Path $apiFile)) {
      Write-Err "API source not found: $apiFile"
      exit 1
    }

    # 构建 Verilator 参数
    $verilatorArgs = @(
    "--cc",
    "--exe",
    "--build",
    "--public",
    "-Wno-fatal",
    "-Wno-WIDTH",
    "-Wno-UNUSED",
    "-Wno-UNOPTFLAT",
    "-Wno-DECLFILENAME",
    "-Wno-MODDUP",
    "--top-module", "NpuTop",
    "-Mdir", $OBJ_DIR,
    "--no-timing",
    "-CFLAGS", "-std=c++17",
    "-CFLAGS", "-O2"
    )

    # VCD 波形输出
    if ($EnableVcd) {
      $verilatorArgs += "--trace"
    }

    # 添加 NPU Verilog 文件
    # 注意: 只需传递 NpuTop.sv, 因为 firtool 已将所有子模块内联到 NpuTop.sv 中
    # 传递多个 .sv 文件会导致 MODDUP (重复模块声明) 警告
    $verilatorArgs += "$VERILOG_DIR\NpuTop.sv"

    # 添加 C++ 测试文件 (testbench + api 实现)
    $verilatorArgs += $tbFile
    $verilatorArgs += $apiFile

    # 指定输出可执行文件名 (Verilator 的 -o 标志, 不是 -CFLAGS -o)
    $verilatorArgs += "-o"
    $verilatorArgs += "$OBJ_DIR\$($cfg.ExeName)"

    Write-Host "[INFO] Running Verilator for $testKey..." -ForegroundColor Yellow
    Write-Host "[INFO] Top module: NpuTop"
    Write-Host "[INFO] Testbench: $($cfg.TbFile)"

    $prevEAP4 = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & verilator @verilatorArgs 2>&1 | ForEach-Object {
      $line = $_
      if ($line -match "error|Error|ERROR") {
        Write-Host $line -ForegroundColor Red
        } elseif ($line -match "warning|Warning|WARNING") {
        Write-Host $line -ForegroundColor Yellow
        } else {
        Write-Host $line -ForegroundColor Gray
      }
    }
    $verilatorExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP4

    if ($verilatorExitCode -ne 0) {
      Write-Err "Verilator compilation failed for $testKey (exit code: $verilatorExitCode)"
      exit 1
    }

    if (-not (Test-Path $exePath)) {
      Write-Err "Executable not found: $exePath"
      exit 1
    }

    Write-Info "Compilation successful: $exePath"
    $CompiledTests[$testKey] = $exePath
    Write-Host ""
  }
  } else {
  Write-Warn "Skipping Verilator compilation (-SkipCompile)"
  # 使用已编译的可执行文件
  $selectedTests = Get-SelectedTests
  foreach ($testKey in $selectedTests) {
    $cfg = $TestConfig[$testKey]
    $exePath = "$OBJ_DIR\$($cfg.ExeName)"
    if (Test-Path $exePath) {
      $CompiledTests[$testKey] = $exePath
      } else {
      Write-Warn "Pre-built executable not found: $exePath (skipping $testKey)"
    }
  }
  Write-Host ""
}

# ==============================================================================
# 4. 运行测试
# ==============================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Phase 3: Run Simulation Tests" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$TestResults = @{}
$AllLogFiles = @()

foreach ($testKey in $CompiledTests.Keys) {
  $cfg = $TestConfig[$testKey]
  $exePath = $CompiledTests[$testKey]
  $logFile = "$REPORT_DIR\$($cfg.LogFile)"

  Write-Step "Running $($cfg.Desc)..."
  Write-Host "[INFO] Executable: $exePath"
  Write-Host "[INFO] Log file: $logFile"
  Write-Host ""

  # 运行仿真, 输出到控制台和日志文件
  $prevEAP5 = $ErrorActionPreference
  $ErrorActionPreference = "Continue"

  # 设置 VCD 输出目录 (如果启用)
  if ($EnableVcd) {
    $env:VERILATOR_TRACE_FILE = "$REPORT_DIR\${testKey}_waveform.vcd"
  }

  & $exePath 2>&1 | Tee-Object -FilePath $logFile
  $simExitCode = $LASTEXITCODE
  $ErrorActionPreference = $prevEAP5

  Write-Host ""
  Write-Host "[INFO] Simulation exit code: $simExitCode"

  # 解析日志提取摘要
  $totalTests = 0
  $passedTests = 0
  $failedTests = 0
  $skippedTests = 0

  if (Test-Path $logFile) {
    $logContent = Get-Content $logFile -Encoding UTF8
    foreach ($line in $logContent) {
      if ($line -match "Total Tests:\s*(\d+)") {
        $totalTests = [int]$Matches[1]
      }
      if ($line -match "Passed Tests:\s*(\d+)") {
        $passedTests = [int]$Matches[1]
      }
      if ($line -match "Failed Tests:\s*(\d+)") {
        $failedTests = [int]$Matches[1]
      }
      if ($line -match "Skipped Tests:\s*(\d+)") {
        $skippedTests = [int]$Matches[1]
      }
    }
  }

  # 如果未找到摘要行, 通过 [PASS]/[FAIL] 计数
  if ($totalTests -eq 0 -and (Test-Path $logFile)) {
    $logContent = Get-Content $logFile -Encoding UTF8
    $passCount = ($logContent | Select-String -Pattern "\[PASS\]" -SimpleMatch).Count
    $failCount = ($logContent | Select-String -Pattern "\[FAIL\]" -SimpleMatch).Count
    $passedTests = $passCount
    $failedTests = $failCount
    $totalTests = $passCount + $failCount
  }

  $status = if ($failedTests -eq 0 -and $simExitCode -eq 0) { "PASS" } else { "FAIL" }
  $color  = if ($status -eq "PASS") { "Green" } else { "Red" }

  Write-Host ("  {0,-15} {1} (total={2}, passed={3}, failed={4})" -f `
  $testKey, $status, $totalTests, $passedTests, $failedTests) -ForegroundColor $color

  $TestResults[$testKey] = @{
    Status    = $status
    Total     = $totalTests
    Passed    = $passedTests
    Failed    = $failedTests
    Skipped   = $skippedTests
    ExitCode  = $simExitCode
    LogFile   = $logFile
  }
  $AllLogFiles += $logFile

  if ($status -eq "FAIL") {
    $ScriptExitCode = 1
  }
  Write-Host ""
}

# ==============================================================================
# 5. 测试结果摘要
# ==============================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Phase 4: Test Results Summary" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$grandTotal  = 0
$grandPassed = 0
$grandFailed = 0

Write-Host "  Test Module Results:" -ForegroundColor White
foreach ($testKey in @("functional", "performance", "boundary")) {
  if ($TestResults.ContainsKey($testKey)) {
    $r = $TestResults[$testKey]
    $status = $r.Status
    $color  = if ($status -eq "PASS") { "Green" } else { "Red" }
    Write-Host ("    {0,-15} {1} ({2} passed, {3} failed)" -f `
    $testKey, $status, $r.Passed, $r.Failed) -ForegroundColor $color
    $grandTotal  += $r.Total
    $grandPassed += $r.Passed
    $grandFailed += $r.Failed
  }
}

Write-Host ""
Write-Host "  ================================================================" -ForegroundColor Cyan
Write-Host "  SIMULATION SUMMARY" -ForegroundColor Cyan
Write-Host "  ================================================================" -ForegroundColor Cyan
Write-Host "  Total tests:   $grandTotal" -ForegroundColor White
Write-Host "  Passed:        $grandPassed" -ForegroundColor Green
Write-Host "  Failed:        $grandFailed" -ForegroundColor Red
if ($grandTotal -gt 0) {
  $passRate = 100.0 * $grandPassed / $grandTotal
  $rateColor = if ($passRate -eq 100) { "Green" } else { "Yellow" }
  Write-Host ("  Pass rate:     {0:F1}%" -f $passRate) -ForegroundColor $rateColor
}
Write-Host "  ================================================================" -ForegroundColor Cyan

if ($grandFailed -eq 0 -and $grandTotal -gt 0) {
  Write-Host ""
  Write-Host "  *** ALL TESTS PASSED ***" -ForegroundColor Green
  } elseif ($grandFailed -gt 0) {
  Write-Host ""
  Write-Host "  *** $grandFailed TEST(S) FAILED ***" -ForegroundColor Red
}
Write-Host ""

# ==============================================================================
# 6. 生成报告
# ==============================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Phase 5: Generate Test Report" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$reportScript = "$TEST_DIR\generate_report.py"
$markdownReport = "$REPORT_DIR\npu_test_report.md"
$jsonReport = "$REPORT_DIR\npu_test_report.json"

if (Test-Path $reportScript) {
  Write-Step "Generating Markdown report via generate_report.py..."

  # 构建 generate_report.py 参数
  $reportArgs = @(
  $reportScript,
  "--log-files"
  )
  $reportArgs += $AllLogFiles
  $reportArgs += "--output"
  $reportArgs += $markdownReport
  $reportArgs += "--json-output"
  $reportArgs += $jsonReport

  $prevEAP6 = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & python @reportArgs 2>&1 | ForEach-Object {
    Write-Host $_ -ForegroundColor Gray
  }
  $reportExitCode = $LASTEXITCODE
  $ErrorActionPreference = $prevEAP6

  if ($reportExitCode -ne 0) {
    Write-Warn "Report generation returned exit code: $reportExitCode"
    } else {
    Write-Info "Markdown report: $markdownReport"
    Write-Info "JSON report:    $jsonReport"
  }
  } else {
  Write-Warn "Report generator not found: $reportScript"
  Write-Warn "Skipping report generation."
}
Write-Host ""

# ==============================================================================
# 7. 清理与退出
# ==============================================================================

$ScriptEndTime = Get-Date
$duration = $ScriptEndTime - $ScriptStartTime
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Test session completed" -ForegroundColor Cyan
Write-Host "  Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "  Exit code: $ScriptExitCode" -ForegroundColor $(if ($ScriptExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[INFO] Intermediate files kept for debugging:"
Write-Host "  - Object dir: $OBJ_DIR"
Write-Host "  - Verilog:    $VERILOG_DIR"
Write-Host "  - Reports:    $REPORT_DIR"
Write-Host ""

exit $ScriptExitCode
