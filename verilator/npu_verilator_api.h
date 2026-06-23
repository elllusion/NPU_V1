// ==============================================================================
// NPU Verilator API 头文件
//
// 提供 C++ 接口调用 Verilator 编译的 NPU 硬件模块:
//   - SystolicCell: 脉动阵列基本单元 (MAC 操作, 时序逻辑)
//   - EccCodec: SECDED ECC 编解码器 (组合逻辑)
//   - NpuSram: 双端口 SRAM (时序逻辑)
//   - MmaEngine: 矩阵乘法加速引擎 (时序逻辑, FSM)
//   - ConvEngine: 卷积加速引擎 (时序逻辑, FSM)
//   - NpuTop: NPU 顶层集成 (时序逻辑, FSM)
//
// 参考:
//   - IEEE 2992-2024 Standard for Neural Network Accelerator Verification
//   - NVIDIA Jetson NPU 测试方法论
//   - Google TPU Systolic Array 测试方法
//   - Apache TVM VTA 测试框架
//
// 所有函数都是线程不安全的，只能在单线程中使用
// ==============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npu_verilator {

// ==============================================================================
// 初始化/关闭
// ==============================================================================

// 初始化 NPU 仿真（创建 DUT 实例，复位）
void init();

// 关闭 NPU 仿真（销毁 DUT 实例）
void shutdown();

// 检查是否已初始化
bool is_initialized();

// 全局时钟步进
void clock_step(uint32_t cycles = 1);

// ==============================================================================
// SystolicCell - 脉动阵列基本单元
// MAC: c_out = c_in + a_in * b_in
// 时序逻辑，单周期完成
// ==============================================================================

struct SystolicCellResult {
    int32_t  cOut;    // 累加器输出
    int8_t   aOut;    // a 信号右移输出
    int8_t   bOut;    // b 信号下移输出
};

// SystolicCell MAC 操作 (单周期)
// a_in * b_in + c_in, 可选 clear 清零
SystolicCellResult systolic_cell_mac(
    int8_t aIn, int8_t bIn, int32_t cIn, bool clear, bool en);

// ==============================================================================
// EccCodec - SECDED ECC 编解码器
// 组合逻辑，单周期完成
// ==============================================================================

enum class EccErrorType : uint8_t {
    NO_ERROR    = 0,  // 无错误
    SINGLE_BIT  = 1,  // 单比特错误 (可纠正)
    DOUBLE_BIT  = 2,  // 双比特错误 (不可纠正)
    MULTI_BIT   = 3   // 多比特错误
};

struct EccResult {
    uint32_t     decoded;       // 解码后的数据
    EccErrorType errorType;     // 错误类型
    uint8_t      syndrome;      // 错误综合征
};

// ECC 编码 (8-bit 数据 -> 13-bit 编码)
uint16_t ecc_encode(uint8_t data);

// ECC 解码 (13-bit 编码 -> 8-bit 数据 + 错误状态)
EccResult ecc_decode(uint16_t encoded);

// 注入单比特错误 (用于测试 SEC)
uint16_t ecc_inject_single_bit_error(uint16_t encoded, int bitPos);

// 注入双比特错误 (用于测试 DED)
uint16_t ecc_inject_double_bit_error(uint16_t encoded, int bit1, int bit2);

// ==============================================================================
// NpuSram - 双端口 SRAM
// 时序逻辑，1 周期读延迟
// ==============================================================================

// SRAM 写入 (Port A)
void sram_write_a(uint32_t addr, uint32_t data, bool byteEnable = false,
                  uint8_t wstrb = 0xFF);

// SRAM 读取 (Port A, 1 周期延迟)
uint32_t sram_read_a(uint32_t addr);

// SRAM 写入 (Port B)
void sram_write_b(uint32_t addr, uint32_t data, bool byteEnable = false,
                  uint8_t wstrb = 0xFF);

// SRAM 读取 (Port B, 1 周期延迟)
uint32_t sram_read_b(uint32_t addr);

// 清空 SRAM (全部写 0)
void sram_clear();

// ==============================================================================
// MmaEngine - 矩阵乘法加速引擎
// C[M][N] = A[M][K] * B[K][N]
// 时序逻辑, FSM: sIdle -> sLoad -> sCompute -> sStoreResult -> sDone
// ==============================================================================

enum class NpuOpcode : uint8_t {
    OP_MATMUL  = 0,  // 矩阵乘法
    OP_CONV    = 1,  // 卷积
    OP_ATTN    = 2,  // 注意力
    OP_SPARSE  = 3,  // 稀疏矩阵
    OP_DMA_LD  = 4,  // DMA 加载
    OP_DMA_ST  = 5   // DMA 存储
};

struct MmaCommand {
    NpuOpcode opcode;
    uint32_t  srcAddr;   // 源地址 (A 矩阵)
    uint32_t  weightAddr; // 权重地址 (B 矩阵)
    uint32_t  dstAddr;   // 目标地址 (C 矩阵)
    uint16_t  dimM;      // M 维度
    uint16_t  dimN;      // N 维度
    uint16_t  dimK;      // K 维度
};

// 启动矩阵乘法 (阻塞调用, 等待完成)
bool mma_execute(const MmaCommand& cmd, uint32_t timeoutCycles = 10000);

// 获取 MMA 性能计数器
uint32_t mma_get_op_count();
uint32_t mma_get_block_count();

// ==============================================================================
// ConvEngine - 卷积加速引擎
// 时序逻辑, FSM: sIdle -> sLoad -> sCompute -> sPostProcess -> sStoreResult -> sDone
// ==============================================================================

// ==============================================================================
// QuantMode - 量化精度模式 (Phase 3.1)
// ==============================================================================

enum class QuantMode : uint8_t {
    INT4  = 0,  // 4-bit 量化
    INT8  = 1,  // 8-bit 量化 (默认)
    INT16 = 2   // 16-bit 量化
};

struct ConvCommand {
    NpuOpcode opcode;
    uint32_t  inputAddr;    // 输入特征图地址
    uint32_t  weightAddr;   // 卷积核地址
    uint32_t  biasAddr;     // 偏置地址
    uint32_t  outputAddr;   // 输出地址
    uint16_t  inputH;       // 输入高度
    uint16_t  inputW;       // 输入宽度
    uint16_t  kernelSize;   // 卷积核大小
    uint16_t  stride;       // 步长
    bool      useRelu;      // 是否使用 ReLU 激活
    bool      useBias;      // 是否使用偏置
    bool      depthwise;    // 是否为 Depthwise 卷积
    // Phase 2.7: 算子融合 BN 参数 (默认关闭, 向后兼容)
    bool      useBatchNorm = false; // 是否融合 BatchNorm
    uint32_t  bnScaleAddr = 0;  // BN scale 参数地址
    uint32_t  bnShiftAddr = 0;  // BN shift 参数地址
    // Phase 3.1: INT4 量化模式 (默认 INT8, 向后兼容)
    QuantMode quantMode = QuantMode::INT8;  // 量化精度模式
    // Phase 3.2: 结构化稀疏支持 (默认关闭, 向后兼容)
    bool      useSparsity = false;  // 是否启用零值检测统计
};

// 启动卷积运算 (阻塞调用, 等待完成)
bool conv_execute(const ConvCommand& cmd, uint32_t timeoutCycles = 50000);

// 获取 Conv 性能计数器
uint32_t conv_get_op_count();
uint32_t conv_get_iter_count();

// Phase 3.2: 获取 Conv 稀疏统计
uint32_t conv_get_zero_weight_count();
uint8_t  conv_get_sparsity_rate();

// ==============================================================================
// SparseEngine - 稀疏矩阵加速引擎 (Phase 3.2)
// 支持模式: 0=稠密, 1=2:4结构化, 2=非结构化, 3=动态零值检测
// ==============================================================================

enum class SparseMode : uint8_t {
    DENSE       = 0,  // 稠密模式
    STRUCT_2_4  = 1,  // 2:4 结构化稀疏
    UNSTRUCT    = 2,  // 非结构化稀疏
    DYNAMIC     = 3   // 动态零值检测 (Phase 3.2)
};

struct SparseCommand {
    SparseMode mode;
    uint32_t   sparseAddr;  // 稀疏矩阵地址
    uint32_t   denseAddr;   // 稠密矩阵地址
    uint32_t   resultAddr;  // 结果地址
    uint32_t   maskAddr;     // mask 地址 (非结构化模式)
    uint16_t   vectorLen;   // 向量长度
};

// 启动稀疏矩阵运算 (阻塞调用, 等待完成)
bool sparse_execute(const SparseCommand& cmd, uint32_t timeoutCycles = 10000);

// 获取 Sparse 统计
uint32_t sparse_get_skip_count();
uint8_t  sparse_get_sparsity_rate();

// ==============================================================================
// NpuTop - NPU 顶层集成
// 时序逻辑, FSM
// ==============================================================================

struct NpuCommand {
    NpuOpcode opcode;
    uint32_t  srcAddr;
    uint32_t  dstAddr;
    uint16_t  dimM;
    uint16_t  dimN;
    uint16_t  dimK;
};

// 启动 NPU 任务 (阻塞调用, 等待完成)
bool npu_execute(const NpuCommand& cmd, uint32_t timeoutCycles = 100000);

// 获取 NPU 完成状态
bool npu_is_done();
bool npu_is_busy();

// 获取 NPU 性能计数器
uint32_t npu_get_perf_matmul_count();
uint32_t npu_get_perf_conv_count();
uint32_t npu_get_perf_attn_count();
uint32_t npu_get_perf_sparse_count();

// ==============================================================================
// Phase 3.3: Winograd 卷积加速 API
// Winograd F(2,3) 算法: 3x3 卷积优化, 输入 4x4 tile -> 输出 2x2 tile
// ============================================================================

struct WinogradCommand {
    uint32_t inputAddr;    // 输入 4x4 tile 地址 (16 bytes)
    uint32_t weightAddr;   // 权重 3x3 地址 (9 bytes)
    uint32_t outputAddr;   // 输出 2x2 tile 地址 (4 bytes)
};

// 启动 Winograd 卷积运算 (阻塞调用, 等待完成)
bool winograd_execute(const WinogradCommand& cmd, uint32_t timeoutCycles = 5000);

// 获取 Winograd 性能计数器
uint32_t winograd_get_op_count();

// ============================================================================
// Phase 3.4: DVFS 动态调频调压 API
// ============================================================================

enum class DvfsLevel : uint8_t {
    P0_TURBO     = 0,  // 1.0V, 1000MHz, div=1
    P1_NOMINAL   = 1,  // 0.9V, 500MHz,  div=2
    P2_EFFICIENT = 2,  // 0.8V, 250MHz,  div=4
    P3_ULTRA_LOW = 3   // 0.7V, 125MHz,  div=8
};

struct DvfsCommand {
    DvfsLevel targetLevel;
};

bool dvfs_request(const DvfsCommand& cmd, uint32_t timeoutCycles = 1000);
uint32_t dvfs_get_transition_count();
uint32_t dvfs_get_current_freq();
uint8_t  dvfs_get_current_voltage();
uint8_t  dvfs_get_current_divisor();

// ============================================================================
// Phase 3.5: 电源域划分 API
// ============================================================================

enum class PowerDomain : uint8_t {
    PD_ALWAYS_ON = 0,  // 始终开启域
    PD_SRAM      = 1,  // SRAM 存储域
    PD_NPU_CORE  = 2,  // NPU 计算核心域
    PD_IO        = 3   // IO 接口域
};

struct PowerDomainCommand {
    PowerDomain targetDomain;
    bool        powerOn;  // true=上电, false=下电
};

bool power_domain_request(const PowerDomainCommand& cmd, uint32_t timeoutCycles = 1000);
uint32_t power_domain_get_up_count();
uint32_t power_domain_get_down_count();
bool power_domain_is_on(PowerDomain domain);

// ============================================================================
// Phase 3.6: 温度监控与调度 API
// ============================================================================

enum class ThermalAlert : uint8_t {
    NORMAL    = 0,  // < 75°C
    WARNING   = 1,  // >= 75°C
    CRITICAL  = 2,  // >= 85°C
    EMERGENCY = 3   // >= 95°C
};

bool thermal_sample(uint8_t tempCelsius);
uint8_t  thermal_get_current_temp();
uint8_t  thermal_get_avg_temp();
ThermalAlert thermal_get_alert_level();
bool thermal_get_warn_intr();
bool thermal_get_crit_intr();
bool thermal_get_emerg_intr();
bool thermal_get_suggest_throttle();
bool thermal_get_suggest_shutdown();
uint32_t thermal_get_sample_count();
uint32_t thermal_get_warn_count();
uint32_t thermal_get_crit_count();
uint32_t thermal_get_emerg_count();

// ============================================================================
// Phase 3.7: 阵列扩展 API
// ============================================================================

enum class ArraySize : uint8_t {
    SIZE_1X1   = 0,
    SIZE_2X2   = 1,
    SIZE_4X4   = 2,
    SIZE_8X8   = 3,
    SIZE_16X16 = 4,
    SIZE_32X32 = 5
};

struct ArrayConfigCommand {
    ArraySize targetSize;
};

bool array_config_request(const ArrayConfigCommand& cmd, uint32_t timeoutCycles = 1000);
uint32_t array_get_rows();
uint32_t array_get_cols();
uint32_t array_get_est_tops();   // MTOPS (TOPS * 1000)
uint32_t array_get_est_gmacs();
uint32_t array_get_config_count();

// ==============================================================================
// 测试统计
// ==============================================================================

struct TestStats {
    uint32_t totalTests;
    uint32_t passedTests;
    uint32_t failedTests;
    uint32_t skippedTests;
};

// 获取测试统计
TestStats get_stats();

// 重置统计
void reset_stats();

// 打印统计摘要
void print_stats_summary();

} // namespace npu_verilator

// ==============================================================================
// 测试辅助宏
// ==============================================================================

#define NPU_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            g_npu_fail_count++; \
        } else { \
            printf("[PASS] %s\n", (msg)); \
            g_npu_pass_count++; \
        } \
    } while(0)

#define NPU_CHECK_EQ(actual, expected, msg) \
    do { \
        if ((actual) != (expected)) { \
            printf("[FAIL] %s:%d: %s (got=0x%llx, expected=0x%llx)\n", \
                   __FILE__, __LINE__, (msg), \
                   (unsigned long long)(actual), \
                   (unsigned long long)(expected)); \
            g_npu_fail_count++; \
        } else { \
            printf("[PASS] %s\n", (msg)); \
            g_npu_pass_count++; \
        } \
    } while(0)

#define NPU_CHECK_NEAR(actual, expected, tolerance, msg) \
    do { \
        int32_t diff = (int32_t)(actual) - (int32_t)(expected); \
        if (diff < 0) diff = -diff; \
        if (diff > (tolerance)) { \
            printf("[FAIL] %s:%d: %s (got=%d, expected=%d, tol=%d)\n", \
                   __FILE__, __LINE__, (msg), \
                   (int32_t)(actual), (int32_t)(expected), (tolerance)); \
            g_npu_fail_count++; \
        } else { \
            printf("[PASS] %s (got=%d, expected=%d)\n", \
                   (msg), (int32_t)(actual), (int32_t)(expected)); \
            g_npu_pass_count++; \
        } \
    } while(0)

extern uint32_t g_npu_pass_count;
extern uint32_t g_npu_fail_count;
