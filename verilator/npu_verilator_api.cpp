// ==============================================================================
// NPU Verilator API 实现
//
// 封装 Verilator 编译的 NPU 硬件模块，提供 C++ 测试接口
//
// 设计说明:
//   - 使用条件编译 (NPU_HAS_VERILATOR) 支持有/无 Verilator 环境
//   - 无 Verilator 时使用软件参考模型，行为与硬件一致
//   - 软件参考模型参考:
//       SystolicCell.scala  - 脉动阵列基本单元 (寄存器化输出)
//       EccCodec.scala      - SECDED Hamming(13,8) 编解码器
//       NpuSram.scala       - 双端口 SRAM (SyncReadMem, 1 周期读延迟)
//       MmaEngine.scala     - 矩阵乘加引擎 (C[M×N] += A[M×K] × B[K×N])
//       ConvEngine.scala    - 2D 卷积引擎
//       NpuTop.scala        - NPU 顶层集成
// ==============================================================================

#include "npu_verilator_api.h"

#ifdef NPU_HAS_VERILATOR
#include <verilated.h>
#include <verilated_vcd_c.h>
// Verilator 生成的头文件 (编译时由 Verilator 生成)
// #include "VNpuTop.h"
// #include "VSystolicCell.h"
// #include "VEccCodec.h"
#endif

#include <cstdio>
#include <cstring>
#include <memory>

// ==============================================================================
// Verilator 时间戳函数 (verilated.cpp 需要)
// ==============================================================================
// Verilator 的 verilated.cpp 运行时调用 sc_time_stamp() 获取仿真时间,
// 必须由用户提供实现, 否则链接时会出现 undefined reference 错误。
double sc_time_stamp() {
    static double sim_time = 0;
    return sim_time;
}

namespace npu_verilator {

// ==============================================================================
// 全局统计变量
// ==============================================================================

uint32_t g_npu_pass_count = 0;
uint32_t g_npu_fail_count = 0;

// ==============================================================================
// 内部常量
// ==============================================================================

namespace {

// SRAM 配置 (与 NpuSram.scala 默认参数一致)
constexpr uint32_t kSramDepth    = 4096;   // SRAM 深度
constexpr uint32_t kSramAddrMask = kSramDepth - 1;

// ECC 配置 (与 EccCodec.scala dataWidth=8 一致)
//   parityWidth = 4 (满足 2^4=16 >= 8+4+1=13)
//   extraParityWidth = 1 (全局奇偶校验)
//   encodedWidth = 8 + 4 + 1 = 13
constexpr uint32_t kEccDataWidth    = 8;
constexpr uint32_t kEccParityWidth  = 4;
constexpr uint32_t kEccEncodedWidth = 13;

} // anonymous namespace

// ==============================================================================
// Verilator 硬件访问 (有 Verilator 时)
// ==============================================================================

#ifdef NPU_HAS_VERILATOR

// DUT 实例与波形跟踪
static VNpuTop*       g_dut      = nullptr;
static VerilatedVcdC* g_vcd      = nullptr;
static bool           g_initialized = false;
static uint64_t       g_sim_time = 0;

// 软件状态镜像 (用于性能计数器查询)
static uint32_t g_mma_op_count    = 0;
static uint32_t g_mma_block_count  = 0;
static uint32_t g_conv_op_count    = 0;
static uint32_t g_conv_iter_count = 0;
static uint32_t g_npu_matmul_count = 0;
static uint32_t g_npu_conv_count   = 0;
static uint32_t g_npu_attn_count   = 0;
static uint32_t g_npu_sparse_count = 0;
// Phase 3.2: 稀疏统计
static uint32_t g_conv_zero_weight_count = 0;
static uint32_t g_sparse_skip_count = 0;
static uint32_t g_sparse_total_count = 0;
// Phase 3.3: Winograd 统计
static uint32_t g_winograd_op_count = 0;
// Phase 3.4: DVFS 状态 (软件镜像)
static DvfsLevel g_dvfs_current_level = DvfsLevel::P1_NOMINAL;
static uint32_t g_dvfs_transition_count = 0;
// Phase 3.5: 电源域状态 (软件镜像)
static bool g_pd_status[4] = {true, true, true, true};  // 默认全部开启
static uint32_t g_pd_up_count = 0;
static uint32_t g_pd_down_count = 0;
// Phase 3.6: 温度监控状态 (软件镜像)
static uint8_t g_thermal_current_temp = 0;
static uint8_t g_thermal_avg_temp = 0;
static ThermalAlert g_thermal_alert = ThermalAlert::NORMAL;
static uint32_t g_thermal_sample_count = 0;
static uint32_t g_thermal_warn_count = 0;
static uint32_t g_thermal_crit_count = 0;
static uint32_t g_thermal_emerg_count = 0;
static const int THERMAL_HISTORY_DEPTH = 8;
static uint8_t g_thermal_history[THERMAL_HISTORY_DEPTH] = {0};
static int g_thermal_hist_idx = 0;
// Phase 3.7: 阵列扩展状态 (软件镜像)
static ArraySize g_array_current_size = ArraySize::SIZE_8X8;
static uint32_t g_array_config_count = 0;

#else // NPU_HAS_VERILATOR

// ============================================================================
// 软件参考模型内部状态
// ============================================================================

// 初始化状态
static bool g_initialized = false;

// 全局时钟计数
static uint64_t g_sim_time = 0;

// --- SystolicCell 状态 (寄存器化输出, 与 SystolicCell.scala 一致) ---
//   aReg, bReg: RegEnable, en=true 时锁存输入, en=false 时保持
//   cReg: clear=true 时清零, 否则 en=true 时锁存 cIn + aIn*bIn
static int8_t  g_cell_a_reg = 0;
static int8_t  g_cell_b_reg = 0;
static int32_t g_cell_c_reg = 0;

// --- NpuSram 状态 (双端口, 与 NpuSram.scala 一致) ---
//   SyncReadMem 有 1 周期读延迟, 用读延迟寄存器模拟
static uint32_t g_sram[kSramDepth];

// Port A 读延迟寄存器 (模拟 SyncReadMem 1 周期延迟)
static uint32_t g_sram_a_rdata_reg = 0;
static bool     g_sram_a_read_pending = false;

// Port B 读延迟寄存器
static uint32_t g_sram_b_rdata_reg = 0;
static bool     g_sram_b_read_pending = false;

// --- MmaEngine 状态 ---
static uint32_t g_mma_op_count    = 0;
static uint32_t g_mma_block_count  = 0;

// --- ConvEngine 状态 ---
static uint32_t g_conv_op_count    = 0;
static uint32_t g_conv_iter_count  = 0;
// Phase 3.2: Conv 稀疏统计
static uint32_t g_conv_zero_weight_count = 0;

// --- SparseEngine 状态 (Phase 3.2) ---
static uint32_t g_sparse_skip_count  = 0;
static uint32_t g_sparse_total_count = 0;

// --- WinogradEngine 状态 (Phase 3.3) ---
static uint32_t g_winograd_op_count = 0;

// --- DvfsController 状态 (Phase 3.4) ---
static DvfsLevel g_dvfs_current_level = DvfsLevel::P1_NOMINAL;
static uint32_t g_dvfs_transition_count = 0;

// --- PowerDomainController 状态 (Phase 3.5) ---
static bool g_pd_status[4] = {true, true, true, true};  // 默认全部开启
static uint32_t g_pd_up_count = 0;
static uint32_t g_pd_down_count = 0;

// --- ThermalMonitor 状态 (Phase 3.6) ---
// 温度阈值
static const uint8_t THERMAL_WARN_THRESHOLD  = 75;
static const uint8_t THERMAL_CRIT_THRESHOLD  = 85;
static const uint8_t THERMAL_EMERG_THRESHOLD = 95;

static uint8_t g_thermal_current_temp = 0;
static uint8_t g_thermal_avg_temp = 0;
static ThermalAlert g_thermal_alert = ThermalAlert::NORMAL;
static uint32_t g_thermal_sample_count = 0;
static uint32_t g_thermal_warn_count = 0;
static uint32_t g_thermal_crit_count = 0;
static uint32_t g_thermal_emerg_count = 0;

// 历史记录 (8 个采样点)
static const int THERMAL_HISTORY_DEPTH = 8;
static uint8_t g_thermal_history[THERMAL_HISTORY_DEPTH] = {0};
static int g_thermal_hist_idx = 0;

// --- ArrayScalingConfig 状态 (Phase 3.7) ---
static ArraySize g_array_current_size = ArraySize::SIZE_8X8;
static uint32_t g_array_config_count = 0;

// --- NpuTop 状态 ---
static uint32_t g_npu_matmul_count = 0;
static uint32_t g_npu_conv_count   = 0;
static uint32_t g_npu_attn_count   = 0;
static uint32_t g_npu_sparse_count = 0;
static bool     g_npu_busy = false;
static bool     g_npu_done = false;

// --- 测试统计 ---
static uint32_t g_total_tests   = 0;
static uint32_t g_passed_tests   = 0;
static uint32_t g_failed_tests   = 0;
static uint32_t g_skipped_tests  = 0;

#endif // NPU_HAS_VERILATOR

// ==============================================================================
// 初始化/关闭
// ==============================================================================

void init() {
#ifdef NPU_HAS_VERILATOR
    // Verilator 硬件仿真初始化
    Verilated::traceEverOn(true);

    if (g_dut != nullptr) {
        delete g_dut;
        g_dut = nullptr;
    }
    if (g_vcd != nullptr) {
        delete g_vcd;
        g_vcd = nullptr;
    }

    g_dut = new VNpuTop;

    // 开启 VCD 波形记录
    g_vcd = new VerilatedVcdC;
    g_dut->trace(g_vcd, 99);
    g_vcd->open("npu_sim.vcd");

    // 复位 10 个时钟周期 (与硬件复位要求一致)
    g_dut->rst = 1;
    for (int i = 0; i < 10; i++) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
    }
    g_dut->rst = 0;

    // 重置性能计数器
    g_mma_op_count    = 0;
    g_mma_block_count  = 0;
    g_conv_op_count    = 0;
    g_conv_iter_count  = 0;
    g_npu_matmul_count = 0;
    g_npu_conv_count   = 0;
    g_npu_attn_count   = 0;
    g_npu_sparse_count = 0;
    g_winograd_op_count = 0;
    // Phase 3.4: 重置 DVFS 状态
    g_dvfs_current_level = DvfsLevel::P1_NOMINAL;
    g_dvfs_transition_count = 0;
    // Phase 3.5: 重置电源域状态
    for (int i = 0; i < 4; i++) g_pd_status[i] = true;
    g_pd_up_count = 0;
    g_pd_down_count = 0;
    // Phase 3.6: 重置温度监控状态
    g_thermal_current_temp = 0;
    g_thermal_avg_temp = 0;
    g_thermal_alert = ThermalAlert::NORMAL;
    g_thermal_sample_count = 0;
    g_thermal_warn_count = 0;
    g_thermal_crit_count = 0;
    g_thermal_emerg_count = 0;
    g_thermal_hist_idx = 0;
    for (int i = 0; i < THERMAL_HISTORY_DEPTH; i++) g_thermal_history[i] = 0;
    // Phase 3.7: 重置阵列扩展状态
    g_array_current_size = ArraySize::SIZE_8X8;
    g_array_config_count = 0;

    g_initialized = true;
#else
    // 软件参考模型初始化
    // 清空 SRAM
    sram_clear();

    // 重置 SystolicCell 寄存器
    g_cell_a_reg = 0;
    g_cell_b_reg = 0;
    g_cell_c_reg = 0;

    // 重置读延迟寄存器
    g_sram_a_rdata_reg    = 0;
    g_sram_a_read_pending = false;
    g_sram_b_rdata_reg    = 0;
    g_sram_b_read_pending = false;

    // 重置性能计数器
    g_mma_op_count    = 0;
    g_mma_block_count  = 0;
    g_conv_op_count    = 0;
    g_conv_iter_count  = 0;
    g_npu_matmul_count = 0;
    g_npu_conv_count   = 0;
    g_npu_attn_count   = 0;
    g_npu_sparse_count = 0;
    g_winograd_op_count = 0;
    // Phase 3.4: 重置 DVFS 状态
    g_dvfs_current_level = DvfsLevel::P1_NOMINAL;
    g_dvfs_transition_count = 0;
    // Phase 3.5: 重置电源域状态
    for (int i = 0; i < 4; i++) g_pd_status[i] = true;
    g_pd_up_count = 0;
    g_pd_down_count = 0;
    // Phase 3.6: 重置温度监控状态
    g_thermal_current_temp = 0;
    g_thermal_avg_temp = 0;
    g_thermal_alert = ThermalAlert::NORMAL;
    g_thermal_sample_count = 0;
    g_thermal_warn_count = 0;
    g_thermal_crit_count = 0;
    g_thermal_emerg_count = 0;
    g_thermal_hist_idx = 0;
    for (int i = 0; i < THERMAL_HISTORY_DEPTH; i++) g_thermal_history[i] = 0;
    // Phase 3.7: 重置阵列扩展状态
    g_array_current_size = ArraySize::SIZE_8X8;
    g_array_config_count = 0;

    // 重置 NPU 状态
    g_npu_busy = false;
    g_npu_done = false;

    // 重置测试统计
    g_total_tests  = 0;
    g_passed_tests  = 0;
    g_failed_tests  = 0;
    g_skipped_tests = 0;

    g_sim_time = 0;
    g_initialized = true;
#endif
}

void shutdown() {
#ifdef NPU_HAS_VERILATOR
    if (g_vcd != nullptr) {
        g_vcd->close();
        delete g_vcd;
        g_vcd = nullptr;
    }
    if (g_dut != nullptr) {
        delete g_dut;
        g_dut = nullptr;
    }
#endif
    g_initialized = false;
}

bool is_initialized() {
    return g_initialized;
}

void clock_step(uint32_t cycles) {
    if (cycles == 0) return;

#ifdef NPU_HAS_VERILATOR
    if (g_dut == nullptr) return;
    for (uint32_t i = 0; i < cycles; i++) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
    }
#else
    // 软件模型: 推进读延迟寄存器
    // SyncReadMem 行为: 本周期发出的读请求, 下一周期返回数据
    for (uint32_t i = 0; i < cycles; i++) {
        g_sim_time++;
    }
#endif
}

// ==============================================================================
// SystolicCell - 脉动阵列基本单元
// ==============================================================================

SystolicCellResult systolic_cell_mac(
    int8_t aIn, int8_t bIn, int32_t cIn, bool clear, bool en) {

    SystolicCellResult result;

#ifdef NPU_HAS_VERILATOR
    // 硬件访问: 驱动 DUT 端口, 单周期执行
    if (g_dut == nullptr) {
        result.cOut = 0;
        result.aOut = 0;
        result.bOut = 0;
        return result;
    }

    g_dut->aIn   = aIn;
    g_dut->bIn   = bIn;
    g_dut->cIn   = cIn;
    g_dut->en    = en ? 1 : 0;
    g_dut->clear = clear ? 1 : 0;

    // 单时钟周期
    g_dut->clk = 0;
    g_dut->eval();
    g_vcd->dump(g_sim_time++);
    g_dut->clk = 1;
    g_dut->eval();
    g_vcd->dump(g_sim_time++);

    result.cOut = static_cast<int32_t>(g_dut->cOut);
    result.aOut = static_cast<int8_t>(g_dut->aOut);
    result.bOut = static_cast<int8_t>(g_dut->bOut);
#else
    // 软件参考模型 (与 SystolicCell.scala 行为一致):
    //   accBase = clear ? 0 : cIn       (clear=true 时累加基为 0)
    //   sum     = accBase + aIn * bIn
    //   cReg:   en=true 时锁存 sum, en=false 时保持
    //   cOut    = en ? cReg : cIn        (en=false 时直通 cIn, 支持流水线排空)
    //   aReg/bReg: en=true 时锁存, en=false 时保持

    int32_t accBase = clear ? 0 : cIn;
    int32_t product = static_cast<int32_t>(aIn) * static_cast<int32_t>(bIn);
    int32_t sum     = accBase + product;

    // 更新寄存器状态 (en=true 时锁存)
    if (en) {
        g_cell_c_reg = sum;
        g_cell_a_reg = aIn;
        g_cell_b_reg = bIn;
    }
    // en=false 时保持原值 (RegEnable 行为)

    // 输出: en=true 时返回寄存器值, en=false 时直通 cIn
    result.cOut = en ? g_cell_c_reg : cIn;
    result.aOut = g_cell_a_reg;
    result.bOut = g_cell_b_reg;
#endif

    return result;
}

// ==============================================================================
// EccCodec - SECDED ECC 编解码器
// ==============================================================================

// Hamming(13,8) SECDED 校验位计算
// 与 EccCodec.scala computeParity 一致:
//   p[0] (位置 1): 覆盖 d[0], d[1], d[3], d[4], d[6]
//   p[1] (位置 2): 覆盖 d[0], d[2], d[3], d[5], d[6]
//   p[2] (位置 4): 覆盖 d[1], d[2], d[3], d[7]
//   p[3] (位置 8): 覆盖 d[4], d[5], d[6], d[7]
//   p[4] (全局):   所有数据位 + 所有校验位的偶校验
static void ecc_compute_parity(uint8_t data, uint8_t parity[5]) {
    parity[0] = ((data >> 0) & 1) ^ ((data >> 1) & 1) ^ ((data >> 3) & 1) ^
                ((data >> 4) & 1) ^ ((data >> 6) & 1);
    parity[1] = ((data >> 0) & 1) ^ ((data >> 2) & 1) ^ ((data >> 3) & 1) ^
                ((data >> 5) & 1) ^ ((data >> 6) & 1);
    parity[2] = ((data >> 1) & 1) ^ ((data >> 2) & 1) ^ ((data >> 3) & 1) ^
                ((data >> 7) & 1);
    parity[3] = ((data >> 4) & 1) ^ ((data >> 5) & 1) ^ ((data >> 6) & 1) ^
                ((data >> 7) & 1);
    // 全局偶校验 (所有数据位 + 所有汉明校验位)
    parity[4] = parity[0] ^ parity[1] ^ parity[2] ^ parity[3];
    for (int i = 0; i < 8; i++) {
        parity[4] ^= ((data >> i) & 1);
    }
}

uint16_t ecc_encode(uint8_t data) {
#ifdef NPU_HAS_VERILATOR
    // 硬件访问: 驱动 EccCodec 编码接口
    if (g_dut != nullptr) {
        g_dut->ecc_encodeData = data;
        g_dut->clk = 0;
        g_dut->eval();
        g_dut->clk = 1;
        g_dut->eval();
        return static_cast<uint16_t>(g_dut->ecc_encoded);
    }
    // 回退到软件模型
#endif

    // 软件参考模型 (与 EccCodec.scala 编码逻辑一致)
    // 编码格式: [globalParity(1)][parity(4)][data(8)] = 13 bits
    uint8_t parity[5];
    ecc_compute_parity(data, parity);

    uint16_t encoded = static_cast<uint16_t>(data) |
                       (static_cast<uint16_t>(parity[0]) << 8) |
                       (static_cast<uint16_t>(parity[1]) << 9) |
                       (static_cast<uint16_t>(parity[2]) << 10) |
                       (static_cast<uint16_t>(parity[3]) << 11) |
                       (static_cast<uint16_t>(parity[4]) << 12);
    return encoded;
}

EccResult ecc_decode(uint16_t encoded) {
    EccResult result;
    result.decoded   = 0;
    result.errorType = EccErrorType::NO_ERROR;
    result.syndrome  = 0;

#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        g_dut->ecc_decodeData = encoded;
        g_dut->clk = 0;
        g_dut->eval();
        g_dut->clk = 1;
        g_dut->eval();
        result.decoded   = static_cast<uint32_t>(g_dut->ecc_decoded);
        result.syndrome  = static_cast<uint8_t>(g_dut->ecc_errorSyndrome);
        // 映射硬件错误状态到枚举
        uint32_t status = g_dut->ecc_errorStatus;
        switch (status) {
            case 0: result.errorType = EccErrorType::NO_ERROR;   break;
            case 1: result.errorType = EccErrorType::SINGLE_BIT; break;
            case 2: result.errorType = EccErrorType::DOUBLE_BIT; break;
            default: result.errorType = EccErrorType::MULTI_BIT; break;
        }
        return result;
    }
    // 回退到软件模型
#endif

    // 软件参考模型 (与 EccCodec.scala 解码逻辑一致)
    // 提取接收到的数据和校验位
    uint8_t rxData    = static_cast<uint8_t>(encoded & 0xFF);
    uint8_t rxParity0 = static_cast<uint8_t>((encoded >> 8)  & 1);
    uint8_t rxParity1 = static_cast<uint8_t>((encoded >> 9)  & 1);
    uint8_t rxParity2 = static_cast<uint8_t>((encoded >> 10) & 1);
    uint8_t rxParity3 = static_cast<uint8_t>((encoded >> 11) & 1);
    uint8_t rxGlobal   = static_cast<uint8_t>((encoded >> 12) & 1);

    // 重新计算校验位 (用于综合征计算)
    uint8_t computedParity[5];
    ecc_compute_parity(rxData, computedParity);

    // 计算综合征 (接收校验位 XOR 计算校验位)
    // syndrome 指向错误位在汉明码字中的位置 (1-based)
    uint8_t s0 = rxParity0 ^ computedParity[0];
    uint8_t s1 = rxParity1 ^ computedParity[1];
    uint8_t s2 = rxParity2 ^ computedParity[2];
    uint8_t s3 = rxParity3 ^ computedParity[3];

    // 全局奇偶校验检查 (SECDED DED 关键逻辑)
    // 正确做法: 使用接收的校验位计算全局校验, 而非重新计算的校验位
    //   global = XOR(data) XOR XOR(parity)
    //   s4 = rxGlobal XOR (XOR(rxData) XOR XOR(rxParity))
    // 原因: 当数据位翻转时, 重新计算的校验位也会变化, 导致全局校验位匹配,
    //        误判为双比特错误。使用接收的校验位可正确区分单/双比特错误。
    uint8_t rxDataXor = 0;
    for (int i = 0; i < 8; i++) rxDataXor ^= ((rxData >> i) & 1);
    uint8_t rxParityXor = rxParity0 ^ rxParity1 ^ rxParity2 ^ rxParity3;
    uint8_t s4 = rxGlobal ^ rxDataXor ^ rxParityXor;

    uint8_t syndrome = s0 | (s1 << 1) | (s2 << 2) | (s3 << 3);
    result.syndrome = syndrome;

    // 错误检测逻辑 (与 EccCodec.scala 一致):
    //   syndrome == 0 && globalMatch: 无错误
    //   syndrome == 0 && !globalMatch: 双比特错误 (不可纠正)
    //   syndrome != 0 && !globalMatch: 单比特错误 (可纠正)
    //   syndrome != 0 && globalMatch: 双比特错误 (不可纠正)
    if (syndrome == 0 && s4 == 0) {
        // 无错误
        result.errorType = EccErrorType::NO_ERROR;
        result.decoded   = rxData;
    } else if (syndrome == 0 && s4 == 1) {
        // 全局校验位错误: 单比特错误在校验位, 数据正确
        // 或双比特错误 (无法区分, 视为双比特)
        result.errorType = EccErrorType::DOUBLE_BIT;
        result.decoded   = rxData;
    } else if (syndrome != 0 && s4 == 1) {
        // 单比特错误 (可纠正)
        result.errorType = EccErrorType::SINGLE_BIT;

        // 纠正: syndrome 指向错误位在汉明码字中的位置 (1-based)
        // 汉明码字位置映射 (dataWidth=8, parityWidth=4):
        //   位置 1: p0 (校验位)
        //   位置 2: p1 (校验位)
        //   位置 3: d0
        //   位置 4: p2 (校验位)
        //   位置 5: d1
        //   位置 6: d2
        //   位置 7: d3
        //   位置 8: p3 (校验位)
        //   位置 9: d4
        //   位置 10: d5
        //   位置 11: d6
        //   位置 12: d7
        uint8_t corrected = rxData;
        switch (syndrome) {
            case 3:  corrected ^= (1 << 0); break; // d0
            case 5:  corrected ^= (1 << 1); break; // d1
            case 6:  corrected ^= (1 << 2); break; // d2
            case 7:  corrected ^= (1 << 3); break; // d3
            case 9:  corrected ^= (1 << 4); break; // d4
            case 10: corrected ^= (1 << 5); break; // d5
            case 11: corrected ^= (1 << 6); break; // d6
            case 12: corrected ^= (1 << 7); break; // d7
            default: break; // 错误在校验位, 数据无需纠正
        }
        result.decoded = corrected;
    } else {
        // syndrome != 0 && globalMatch: 双比特错误 (不可纠正)
        result.errorType = EccErrorType::DOUBLE_BIT;
        result.decoded   = rxData;
    }

    return result;
}

uint16_t ecc_inject_single_bit_error(uint16_t encoded, int bitPos) {
    if (bitPos < 0 || bitPos >= kEccEncodedWidth) {
        return encoded;
    }
    return encoded ^ (1 << bitPos);
}

uint16_t ecc_inject_double_bit_error(uint16_t encoded, int bit1, int bit2) {
    if (bit1 < 0 || bit1 >= kEccEncodedWidth) {
        return encoded;
    }
    if (bit2 < 0 || bit2 >= kEccEncodedWidth) {
        return encoded;
    }
    if (bit1 == bit2) {
        // 同一位翻转两次等于无错误, 仅翻转一次
        return encoded ^ (1 << bit1);
    }
    return encoded ^ (1 << bit1) ^ (1 << bit2);
}

// ==============================================================================
// NpuSram - 双端口 SRAM
// ==============================================================================

// 内部辅助: 字节级写入 (与 NpuSram.scala 字节级写入逻辑一致)
static void sram_write_internal(uint32_t addr, uint32_t data,
                                 bool byteEnable, uint8_t wstrb) {
    if (addr >= kSramDepth) return;

    if (byteEnable) {
        // 字节级写入 (read-modify-write)
        if (wstrb & 0x1) {
            g_sram[addr] = (g_sram[addr] & 0xFFFFFF00u) | (data & 0x000000FFu);
        }
        if (wstrb & 0x2) {
            g_sram[addr] = (g_sram[addr] & 0xFFFF00FFu) | (data & 0x0000FF00u);
        }
        if (wstrb & 0x4) {
            g_sram[addr] = (g_sram[addr] & 0xFF00FFFFu) | (data & 0x00FF0000u);
        }
        if (wstrb & 0x8) {
            g_sram[addr] = (g_sram[addr] & 0x00FFFFFFu) | (data & 0xFF000000u);
        }
    } else {
        // 全字写入
        g_sram[addr] = data;
    }
}

void sram_write_a(uint32_t addr, uint32_t data, bool byteEnable, uint8_t wstrb) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        g_dut->portA_wen   = 1;
        g_dut->portA_ren   = 0;
        g_dut->portA_addr  = addr & kSramAddrMask;
        g_dut->portA_wdata = data;
        g_dut->portA_wstrb = byteEnable ? wstrb : 0xF;
        g_dut->clk = 0;
        g_dut->eval();
        g_dut->clk = 1;
        g_dut->eval();
        g_dut->portA_wen = 0;
        return;
    }
#endif
    sram_write_internal(addr & kSramAddrMask, data, byteEnable, wstrb);
}

uint32_t sram_read_a(uint32_t addr) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        g_dut->portA_ren   = 1;
        g_dut->portA_wen   = 0;
        g_dut->portA_addr  = addr & kSramAddrMask;
        // SyncReadMem: 1 周期读延迟
        g_dut->clk = 0;
        g_dut->eval();
        g_dut->clk = 1;
        g_dut->eval();
        g_dut->portA_ren = 0;
        return static_cast<uint32_t>(g_dut->portA_rdata);
    }
#endif
    // 软件参考模型: SyncReadMem 有 1 周期读延迟
    // 为简化使用, 软件模型直接返回当前值 (无延迟)
    // 严格模拟延迟需要调用方显式 clock_step, 此处保持简单
    if ((addr & kSramAddrMask) >= kSramDepth) return 0;
    return g_sram[addr & kSramAddrMask];
}

void sram_write_b(uint32_t addr, uint32_t data, bool byteEnable, uint8_t wstrb) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        g_dut->portB_wen   = 1;
        g_dut->portB_ren   = 0;
        g_dut->portB_addr  = addr & kSramAddrMask;
        g_dut->portB_wdata = data;
        g_dut->portB_wstrb = byteEnable ? wstrb : 0xF;
        g_dut->clk = 0;
        g_dut->eval();
        g_dut->clk = 1;
        g_dut->eval();
        g_dut->portB_wen = 0;
        return;
    }
#endif
    sram_write_internal(addr & kSramAddrMask, data, byteEnable, wstrb);
}

uint32_t sram_read_b(uint32_t addr) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        g_dut->portB_ren   = 1;
        g_dut->portB_wen   = 0;
        g_dut->portB_addr  = addr & kSramAddrMask;
        g_dut->clk = 0;
        g_dut->eval();
        g_dut->clk = 1;
        g_dut->eval();
        g_dut->portB_ren = 0;
        return static_cast<uint32_t>(g_dut->portB_rdata);
    }
#endif
    if ((addr & kSramAddrMask) >= kSramDepth) return 0;
    return g_sram[addr & kSramAddrMask];
}

void sram_clear() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        // 通过 Port A 逐地址写零
        for (uint32_t i = 0; i < kSramDepth; i++) {
            g_dut->portA_wen   = 1;
            g_dut->portA_ren   = 0;
            g_dut->portA_addr  = i;
            g_dut->portA_wdata = 0;
            g_dut->portA_wstrb = 0xF;
            g_dut->clk = 0;
            g_dut->eval();
            g_dut->clk = 1;
            g_dut->eval();
        }
        g_dut->portA_wen = 0;
        return;
    }
#endif
    std::memset(g_sram, 0, sizeof(g_sram));
    g_sram_a_rdata_reg    = 0;
    g_sram_a_read_pending = false;
    g_sram_b_rdata_reg    = 0;
    g_sram_b_read_pending = false;
}

// ==============================================================================
// MmaEngine - 矩阵乘加引擎
// ==============================================================================

bool mma_execute(const MmaCommand& cmd, uint32_t timeoutCycles) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut == nullptr) return false;

    // 驱动 MMA 命令接口
    g_dut->mma_cmd_mDim       = cmd.dimM;
    g_dut->mma_cmd_nDim       = cmd.dimN;
    g_dut->mma_cmd_kDim       = cmd.dimK;
    g_dut->mma_cmd_aAddr      = cmd.srcAddr;
    g_dut->mma_cmd_bAddr      = cmd.weightAddr;
    g_dut->mma_cmd_cAddr      = cmd.dstAddr;
    g_dut->mma_cmd_accumulate = 0; // 覆盖模式
    g_dut->mma_cmdValid       = 1;

    // 等待 cmdReady
    uint32_t waitCycles = 0;
    while (!g_dut->mma_cmdReady && waitCycles < timeoutCycles) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        waitCycles++;
    }
    g_dut->mma_cmdValid = 0;

    if (waitCycles >= timeoutCycles) return false;

    // 等待 done 信号
    while (!g_dut->mma_done && waitCycles < timeoutCycles) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        waitCycles++;
    }

    g_mma_op_count++;
    g_mma_block_count += ((cmd.dimM + 15) / 16) * ((cmd.dimN + 15) / 16) *
                          ((cmd.dimK + 15) / 16);

    return (waitCycles < timeoutCycles);
#else
    // 软件参考模型: C[M×N] = A[M×K] × B[K×N]
    // 与 MmaEngine.scala 计算语义一致 (INT8 乘法, INT32 累加)
    if (cmd.dimM == 0 || cmd.dimN == 0 || cmd.dimK == 0) {
        return false;
    }

    // 地址边界检查
    uint32_t aEnd = cmd.srcAddr + cmd.dimM * cmd.dimK;
    uint32_t bEnd = cmd.weightAddr + cmd.dimK * cmd.dimN;
    uint32_t cEnd = cmd.dstAddr + cmd.dimM * cmd.dimN;
    if (aEnd > kSramDepth || bEnd > kSramDepth || cEnd > kSramDepth) {
        return false;
    }

    // 执行矩阵乘法: C[m][n] = sum_k(A[m][k] * B[k][n])
    // A 矩阵按行存储: A[m][k] 在 srcAddr + m*dimK + k
    // B 矩阵按行存储: B[k][n] 在 weightAddr + k*dimN + n
    // C 矩阵按行存储: C[m][n] 在 dstAddr + m*dimN + n
    for (int m = 0; m < cmd.dimM; m++) {
        for (int n = 0; n < cmd.dimN; n++) {
            int32_t sum = 0;
            for (int k = 0; k < cmd.dimK; k++) {
                int8_t a = static_cast<int8_t>(
                    sram_read_a(cmd.srcAddr + m * cmd.dimK + k) & 0xFF);
                int8_t b = static_cast<int8_t>(
                    sram_read_a(cmd.weightAddr + k * cmd.dimN + n) & 0xFF);
                sum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            }
            sram_write_a(cmd.dstAddr + m * cmd.dimN + n,
                         static_cast<uint32_t>(sum));
        }
    }

    // 更新性能计数器
    g_mma_op_count++;
    // 分块计数: ceil(M/16) * ceil(N/16) * ceil(K/16)
    uint32_t mBlocks = (cmd.dimM + 15) / 16;
    uint32_t nBlocks = (cmd.dimN + 15) / 16;
    uint32_t kBlocks = (cmd.dimK + 15) / 16;
    g_mma_block_count += mBlocks * nBlocks * kBlocks;

    return true;
#endif
}

uint32_t mma_get_op_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->mma_perfOpCount);
    }
#endif
    return g_mma_op_count;
}

uint32_t mma_get_block_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->mma_perfBlockCount);
    }
#endif
    return g_mma_block_count;
}

// ==============================================================================
// ConvEngine - 卷积加速引擎
// ==============================================================================

bool conv_execute(const ConvCommand& cmd, uint32_t timeoutCycles) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut == nullptr) return false;

    // 驱动 Conv 命令接口
    g_dut->conv_cmd_inputAddr    = cmd.inputAddr;
    g_dut->conv_cmd_weightAddr   = cmd.weightAddr;
    g_dut->conv_cmd_biasAddr     = cmd.biasAddr;
    g_dut->conv_cmd_outputAddr   = cmd.outputAddr;
    g_dut->conv_cmd_inputH       = cmd.inputH;
    g_dut->conv_cmd_inputW       = cmd.inputW;
    g_dut->conv_cmd_kernelSize   = cmd.kernelSize;
    g_dut->conv_cmd_stride       = cmd.stride;
    g_dut->conv_cmd_useRelu      = cmd.useRelu ? 1 : 0;
    g_dut->conv_cmd_useBias      = cmd.useBias ? 1 : 0;
    g_dut->conv_cmd_depthwise    = cmd.depthwise ? 1 : 0;
    g_dut->conv_cmdValid         = 1;

    uint32_t waitCycles = 0;
    while (!g_dut->conv_cmdReady && waitCycles < timeoutCycles) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        waitCycles++;
    }
    g_dut->conv_cmdValid = 0;

    if (waitCycles >= timeoutCycles) return false;

    while (!g_dut->conv_done && waitCycles < timeoutCycles) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        waitCycles++;
    }

    g_conv_op_count++;
    g_conv_iter_count += cmd.inputH * cmd.inputW;

    return (waitCycles < timeoutCycles);
#else
    // 软件参考模型: 2D 卷积
    // 输入: input[H][W], 卷积核: kernel[K][K], 输出: output[H'][W']
    //   H' = (H - K) / stride + 1
    //   W' = (W - K) / stride + 1
    if (cmd.inputH == 0 || cmd.inputW == 0 || cmd.kernelSize == 0 ||
        cmd.stride == 0) {
        return false;
    }

    uint32_t outH = (cmd.inputH - cmd.kernelSize) / cmd.stride + 1;
    uint32_t outW = (cmd.inputW - cmd.kernelSize) / cmd.stride + 1;

    // 地址边界检查
    uint32_t inEnd  = cmd.inputAddr + cmd.inputH * cmd.inputW;
    uint32_t wtEnd  = cmd.weightAddr + cmd.kernelSize * cmd.kernelSize;
    uint32_t outEnd = cmd.outputAddr + outH * outW;
    if (inEnd > kSramDepth || wtEnd > kSramDepth || outEnd > kSramDepth) {
        return false;
    }

    // 执行卷积
    for (uint32_t oh = 0; oh < outH; oh++) {
        for (uint32_t ow = 0; ow < outW; ow++) {
            int32_t sum = 0;
            for (uint16_t kh = 0; kh < cmd.kernelSize; kh++) {
                for (uint16_t kw = 0; kw < cmd.kernelSize; kw++) {
                    uint32_t ih = oh * cmd.stride + kh;
                    uint32_t iw = ow * cmd.stride + kw;

                    // Phase 3.1: INT4 量化模式支持
                    int32_t inVal, wtVal;
                    if (cmd.quantMode == QuantMode::INT4) {
                        // INT4: 从低 4 位提取, sign-extend 到 8-bit
                        uint32_t inRaw = sram_read_a(cmd.inputAddr + ih * cmd.inputW + iw);
                        uint32_t wtRaw = sram_read_a(cmd.weightAddr + kh * cmd.kernelSize + kw);
                        inVal = static_cast<int32_t>(static_cast<int8_t>(
                            (inRaw & 0x8) ? (inRaw | 0xF0) : (inRaw & 0x0F)));
                        wtVal = static_cast<int32_t>(static_cast<int8_t>(
                            (wtRaw & 0x8) ? (wtRaw | 0xF0) : (wtRaw & 0x0F)));
                    } else {
                        // INT8: 直接读取 8-bit 有符号值
                        inVal = static_cast<int32_t>(static_cast<int8_t>(
                            sram_read_a(cmd.inputAddr + ih * cmd.inputW + iw) & 0xFF));
                        wtVal = static_cast<int32_t>(static_cast<int8_t>(
                            sram_read_a(cmd.weightAddr + kh * cmd.kernelSize + kw) & 0xFF));
                    }
                    sum += inVal * wtVal;
                }
            }

            // 可选偏置加法
            if (cmd.useBias) {
                int32_t bias = static_cast<int32_t>(
                    sram_read_a(cmd.biasAddr + oh * outW + ow));
                sum += bias;
            }

            // Phase 2.7: 可选 BatchNorm 融合 (y = scale * x + shift)
            if (cmd.useBatchNorm) {
                int32_t scale = static_cast<int32_t>(
                    sram_read_a(cmd.bnScaleAddr + oh * outW + ow));
                int32_t shift = static_cast<int32_t>(
                    sram_read_a(cmd.bnShiftAddr + oh * outW + ow));
                sum = scale * sum + shift;
            }

            // 可选 ReLU 激活
            if (cmd.useRelu && sum < 0) {
                sum = 0;
            }

            sram_write_a(cmd.outputAddr + oh * outW + ow,
                         static_cast<uint32_t>(sum));
        }
    }

    // 更新性能计数器
    g_conv_op_count++;
    g_conv_iter_count += outH * outW;

    return true;
#endif
}

uint32_t conv_get_op_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->conv_perfOpCount);
    }
#endif
    return g_conv_op_count;
}

uint32_t conv_get_iter_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->conv_perfIterCount);
    }
#endif
    return g_conv_iter_count;
}

// Phase 3.2: Conv 稀疏统计
uint32_t conv_get_zero_weight_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->conv_perfZeroWeightCount);
    }
#endif
    return g_conv_zero_weight_count;
}

uint8_t conv_get_sparsity_rate() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint8_t>(g_dut->conv_perfSparsityRate);
    }
#endif
    // 软件模型: 返回 0 (无实际稀疏检测)
    return 0;
}

// ==============================================================================
// SparseEngine - 稀疏矩阵加速引擎 (Phase 3.2)
// ==============================================================================

bool sparse_execute(const SparseCommand& cmd, uint32_t timeoutCycles) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut == nullptr) return false;
    // 硬件路径: 驱动 SparseEngine 接口
    // TODO: 实现 Verilator 硬件驱动
    return false;
#else
    // 软件参考模型: 稀疏向量点积
    // 支持模式: 0=稠密, 1=2:4结构化, 2=非结构化, 3=动态零值检测
    if (cmd.vectorLen == 0) return false;

    // 地址边界检查
    uint32_t sparseEnd = cmd.sparseAddr + cmd.vectorLen;
    uint32_t denseEnd  = cmd.denseAddr + cmd.vectorLen;
    if (sparseEnd > kSramDepth || denseEnd > kSramDepth) {
        return false;
    }

    int32_t sum = 0;
    uint32_t skipCount = 0;

    for (uint16_t i = 0; i < cmd.vectorLen; ++i) {
        int32_t sparseVal = static_cast<int32_t>(static_cast<int8_t>(
            sram_read_a(cmd.sparseAddr + i) & 0xFF));
        int32_t denseVal = static_cast<int32_t>(static_cast<int8_t>(
            sram_read_a(cmd.denseAddr + i) & 0xFF));

        bool shouldSkip = false;

        // Phase 3.2: 动态零值检测 (mode=3)
        if (cmd.mode == SparseMode::DYNAMIC) {
            // 动态检测: 如果稀疏值为 0, 跳过
            shouldSkip = (sparseVal == 0);
        }

        if (shouldSkip) {
            skipCount++;
        } else {
            sum += sparseVal * denseVal;
        }
    }

    // 更新统计
    g_sparse_skip_count += skipCount;
    g_sparse_total_count += cmd.vectorLen;

    // 存储结果
    sram_write_a(cmd.resultAddr, static_cast<uint32_t>(sum));

    return true;
#endif
}

uint32_t sparse_get_skip_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->sparse_skipCount);
    }
#endif
    return g_sparse_skip_count;
}

uint8_t sparse_get_sparsity_rate() {
    uint32_t total = g_sparse_total_count;
    if (total == 0) return 0;
    return static_cast<uint8_t>((g_sparse_skip_count * 100) / total);
}

// ==============================================================================
// WinogradEngine - Winograd 卷积加速引擎 (Phase 3.3)
// F(2,3) 算法: 3x3 卷积优化, 4x4 输入 tile -> 2x2 输出 tile
// 减少乘法次数: 标准 3x3 卷积需要 9 次乘法, Winograd 仅需 4 次
// ==============================================================================

bool winograd_execute(const WinogradCommand& cmd, uint32_t timeoutCycles) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut == nullptr) return false;
    // 硬件路径: 驱动 WinogradEngine 接口
    // TODO: 实现 Verilator 硬件驱动
    return false;
#else
    // 软件参考模型: Winograd F(2,3) 算法
    // 变换矩阵 (整数版本):
    //   B^T (输入变换):
    //     [1,  0, -1, 0]
    //     [0,  1,  1, 0]
    //     [0, -1,  1, 0]
    //     [0,  1,  0, -1]
    //   G' (权重变换, 2*G):
    //     [2, 0, 0]
    //     [1, 1, 1]
    //     [1, -1, 1]
    //     [0, 0, 2]
    //   A^T (输出变换):
    //     [1, 1, 1, 0]
    //     [0, 1, -1, -1]

    // 地址边界检查
    uint32_t inEnd  = cmd.inputAddr + 16;   // 4x4 tile
    uint32_t wtEnd  = cmd.weightAddr + 9;   // 3x3 kernel
    uint32_t outEnd = cmd.outputAddr + 4;   // 2x2 output
    if (inEnd > kSramDepth || wtEnd > kSramDepth || outEnd > kSramDepth) {
        return false;
    }

    // 1. 读取 4x4 输入 tile
    int8_t d[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            d[i][j] = static_cast<int8_t>(sram_read_a(cmd.inputAddr + i*4 + j) & 0xFF);
        }
    }

    // 2. 读取 3x3 权重
    int8_t g[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            g[i][j] = static_cast<int8_t>(sram_read_a(cmd.weightAddr + i*3 + j) & 0xFF);
        }
    }

    // 3. 输入变换 V = B^T * d * B
    // 先计算 d * B (列变换)
    int dB[4][4];
    for (int i = 0; i < 4; i++) {
        dB[i][0] = d[i][0] - d[i][2];
        dB[i][1] = d[i][1] + d[i][2];
        dB[i][2] = -d[i][1] + d[i][2];
        dB[i][3] = d[i][1] - d[i][3];
    }
    // 再计算 B^T * (d*B) (行变换)
    int V[4][4];
    for (int j = 0; j < 4; j++) {
        V[0][j] = dB[0][j] - dB[2][j];
        V[1][j] = dB[1][j] + dB[2][j];
        V[2][j] = -dB[1][j] + dB[2][j];
        V[3][j] = dB[1][j] - dB[3][j];
    }

    // 4. 权重变换 U = G' * g * G'^T
    // 先计算 g * G'^T (列变换)
    int gGt[3][4];
    for (int i = 0; i < 3; i++) {
        gGt[i][0] = 2 * g[i][0];
        gGt[i][1] = g[i][0] + g[i][1] + g[i][2];
        gGt[i][2] = g[i][0] - g[i][1] + g[i][2];
        gGt[i][3] = 2 * g[i][2];
    }
    // 再计算 G' * (g*G'^T) (行变换)
    int U[4][4];
    for (int j = 0; j < 4; j++) {
        U[0][j] = 2 * gGt[0][j];
        U[1][j] = gGt[0][j] + gGt[1][j] + gGt[2][j];
        U[2][j] = gGt[0][j] - gGt[1][j] + gGt[2][j];
        U[3][j] = 2 * gGt[2][j];
    }

    // 5. 逐元素乘法 M = U ⊙ V
    int M[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M[i][j] = U[i][j] * V[i][j];
        }
    }

    // 6. 输出变换 Y = (A^T * M * A) >> 2
    // 先计算 M * A (列变换)
    int MA[4][2];
    for (int i = 0; i < 4; i++) {
        MA[i][0] = M[i][0] + M[i][1] + M[i][2];
        MA[i][1] = M[i][1] - M[i][2] - M[i][3];
    }
    // 再计算 A^T * (M*A) (行变换)
    int Y[2][2];
    for (int j = 0; j < 2; j++) {
        Y[0][j] = (MA[0][j] + MA[1][j] + MA[2][j]) >> 2;
        Y[1][j] = (MA[1][j] - MA[2][j] - MA[3][j]) >> 2;
    }

    // 7. 存储结果
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sram_write_a(cmd.outputAddr + i*2 + j, static_cast<uint32_t>(Y[i][j]));
        }
    }

    g_winograd_op_count++;
    return true;
#endif
}

uint32_t winograd_get_op_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->winograd_perfOpCount);
    }
#endif
    return g_winograd_op_count;
}

// ==============================================================================
// DvfsController - DVFS 动态调频调压 (Phase 3.4)
// 软件参考模型: 维护当前 DVFS 级别和过渡计数
// ==============================================================================

// DVFS 级别配置表
struct DvfsLevelConfig {
    uint32_t freq;      // MHz
    uint8_t  voltage;   // 0.01V/LSB
    uint8_t  divisor;   // 时钟分频
};

static const DvfsLevelConfig dvfs_config_table[4] = {
    {1000, 100, 1},  // P0 Turbo
    {500,  90,  2},  // P1 Nominal
    {250,  80,  4},  // P2 Efficient
    {125,  70,  8}   // P3 UltraLow
};

bool dvfs_request(const DvfsCommand& cmd, uint32_t timeoutCycles) {
    (void)timeoutCycles;  // 软件模型无超时
    if (cmd.targetLevel == g_dvfs_current_level) {
        return true;  // 无需切换
    }
    g_dvfs_current_level = cmd.targetLevel;
    g_dvfs_transition_count++;
    return true;
}

uint32_t dvfs_get_transition_count() {
    return g_dvfs_transition_count;
}

uint32_t dvfs_get_current_freq() {
    return dvfs_config_table[static_cast<uint8_t>(g_dvfs_current_level)].freq;
}

uint8_t dvfs_get_current_voltage() {
    return dvfs_config_table[static_cast<uint8_t>(g_dvfs_current_level)].voltage;
}

uint8_t dvfs_get_current_divisor() {
    return dvfs_config_table[static_cast<uint8_t>(g_dvfs_current_level)].divisor;
}

// ==============================================================================
// PowerDomainController - 电源域划分 (Phase 3.5)
// 软件参考模型: 维护各电源域开关状态及上/下电计数
// ==============================================================================

bool power_domain_request(const PowerDomainCommand& cmd, uint32_t timeoutCycles) {
    (void)timeoutCycles;  // 软件模型无超时
    uint8_t idx = static_cast<uint8_t>(cmd.targetDomain);
    if (idx >= 4) return false;

    g_pd_status[idx] = cmd.powerOn;
    if (cmd.powerOn) {
        g_pd_up_count++;
    } else {
        g_pd_down_count++;
    }
    return true;
}

uint32_t power_domain_get_up_count() {
    return g_pd_up_count;
}

uint32_t power_domain_get_down_count() {
    return g_pd_down_count;
}

bool power_domain_is_on(PowerDomain domain) {
    return g_pd_status[static_cast<uint8_t>(domain)];
}

// ==============================================================================
// ThermalMonitor - 温度监控与调度 (Phase 3.6)
// 软件参考模型: 维护当前温度、平均温度、告警级别及告警计数
// ==============================================================================

bool thermal_sample(uint8_t tempCelsius) {
    g_thermal_current_temp = tempCelsius;

    // 更新历史记录
    g_thermal_history[g_thermal_hist_idx] = tempCelsius;
    g_thermal_hist_idx = (g_thermal_hist_idx + 1) % THERMAL_HISTORY_DEPTH;

    // 计算平均温度
    uint32_t sum = 0;
    for (int i = 0; i < THERMAL_HISTORY_DEPTH; i++) {
        sum += g_thermal_history[i];
    }
    g_thermal_avg_temp = static_cast<uint8_t>(sum / THERMAL_HISTORY_DEPTH);

    // 确定告警级别
    if (tempCelsius >= THERMAL_EMERG_THRESHOLD) {
        g_thermal_alert = ThermalAlert::EMERGENCY;
        g_thermal_emerg_count++;
    } else if (tempCelsius >= THERMAL_CRIT_THRESHOLD) {
        g_thermal_alert = ThermalAlert::CRITICAL;
        g_thermal_crit_count++;
    } else if (tempCelsius >= THERMAL_WARN_THRESHOLD) {
        g_thermal_alert = ThermalAlert::WARNING;
        g_thermal_warn_count++;
    } else {
        g_thermal_alert = ThermalAlert::NORMAL;
    }

    g_thermal_sample_count++;
    return true;
}

uint8_t thermal_get_current_temp() { return g_thermal_current_temp; }
uint8_t thermal_get_avg_temp() { return g_thermal_avg_temp; }
ThermalAlert thermal_get_alert_level() { return g_thermal_alert; }
bool thermal_get_warn_intr() {
    return g_thermal_alert == ThermalAlert::WARNING ||
           g_thermal_alert == ThermalAlert::CRITICAL ||
           g_thermal_alert == ThermalAlert::EMERGENCY;
}
bool thermal_get_crit_intr() {
    return g_thermal_alert == ThermalAlert::CRITICAL ||
           g_thermal_alert == ThermalAlert::EMERGENCY;
}
bool thermal_get_emerg_intr() {
    return g_thermal_alert == ThermalAlert::EMERGENCY;
}
bool thermal_get_suggest_throttle() {
    return g_thermal_alert == ThermalAlert::WARNING ||
           g_thermal_alert == ThermalAlert::CRITICAL;
}
bool thermal_get_suggest_shutdown() {
    return g_thermal_alert == ThermalAlert::EMERGENCY;
}
uint32_t thermal_get_sample_count() { return g_thermal_sample_count; }
uint32_t thermal_get_warn_count() { return g_thermal_warn_count; }
uint32_t thermal_get_crit_count() { return g_thermal_crit_count; }
uint32_t thermal_get_emerg_count() { return g_thermal_emerg_count; }

// ==============================================================================
// ArrayScalingConfig - 阵列扩展 (Phase 3.7)
// 软件参考模型: 维护当前阵列规模及配置计数
// ==============================================================================

// 阵列规模配置表
struct ArraySizeConfig {
    uint32_t rows;
    uint32_t cols;
    uint32_t mtops;   // MTOPS (TOPS * 1000)
    uint32_t gmacs;
};

static const ArraySizeConfig array_config_table[6] = {
    {1,  1,  1,    1},    // 1x1
    {2,  2,  4,    2},    // 2x2
    {4,  4,  16,   8},    // 4x4
    {8,  8,  64,   32},   // 8x8
    {16, 16, 256,  128},  // 16x16
    {32, 32, 1024, 512}   // 32x32
};

bool array_config_request(const ArrayConfigCommand& cmd, uint32_t timeoutCycles) {
    (void)timeoutCycles;  // 软件模型无超时
    if (cmd.targetSize == g_array_current_size) {
        return true;
    }
    g_array_current_size = cmd.targetSize;
    g_array_config_count++;
    return true;
}

uint32_t array_get_rows() {
    return array_config_table[static_cast<uint8_t>(g_array_current_size)].rows;
}

uint32_t array_get_cols() {
    return array_config_table[static_cast<uint8_t>(g_array_current_size)].cols;
}

uint32_t array_get_est_tops() {
    return array_config_table[static_cast<uint8_t>(g_array_current_size)].mtops;
}

uint32_t array_get_est_gmacs() {
    return array_config_table[static_cast<uint8_t>(g_array_current_size)].gmacs;
}

uint32_t array_get_config_count() {
    return g_array_config_count;
}

// ==============================================================================
// NpuTop - NPU 顶层集成
// ==============================================================================

bool npu_execute(const NpuCommand& cmd, uint32_t timeoutCycles) {
#ifdef NPU_HAS_VERILATOR
    if (g_dut == nullptr) return false;

    // 驱动 NPU 顶层命令接口
    g_dut->cmd_opcode = static_cast<uint32_t>(cmd.opcode);
    g_dut->cmd_srcAddr = cmd.srcAddr;
    g_dut->cmd_dstAddr = cmd.dstAddr;
    g_dut->cmd_dimM    = cmd.dimM;
    g_dut->cmd_dimN    = cmd.dimN;
    g_dut->cmd_dimK    = cmd.dimK;
    g_dut->cmd_valid   = 1;

    uint32_t waitCycles = 0;
    while (!g_dut->cmd_ready && waitCycles < timeoutCycles) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        waitCycles++;
    }
    g_dut->cmd_valid = 0;

    if (waitCycles >= timeoutCycles) return false;

    while (!g_dut->completion_done && waitCycles < timeoutCycles) {
        g_dut->clk = 0;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        g_dut->clk = 1;
        g_dut->eval();
        g_vcd->dump(g_sim_time++);
        waitCycles++;
    }

    // 更新性能计数器
    switch (cmd.opcode) {
        case NpuOpcode::OP_MATMUL: g_npu_matmul_count++;  break;
        case NpuOpcode::OP_CONV:   g_npu_conv_count++;    break;
        case NpuOpcode::OP_ATTN:   g_npu_attn_count++;    break;
        case NpuOpcode::OP_SPARSE: g_npu_sparse_count++;  break;
        default: break;
    }

    return (waitCycles < timeoutCycles);
#else
    // 软件参考模型: 根据 opcode 分发到对应引擎
    g_npu_busy = true;
    g_npu_done = false;

    bool success = false;
    switch (cmd.opcode) {
        case NpuOpcode::OP_MATMUL: {
            MmaCommand mmaCmd;
            mmaCmd.opcode     = NpuOpcode::OP_MATMUL;
            mmaCmd.srcAddr    = cmd.srcAddr;
            mmaCmd.weightAddr = cmd.srcAddr + cmd.dimM * cmd.dimK;
            mmaCmd.dstAddr    = cmd.dstAddr;
            mmaCmd.dimM       = cmd.dimM;
            mmaCmd.dimN       = cmd.dimN;
            mmaCmd.dimK       = cmd.dimK;
            success = mma_execute(mmaCmd, timeoutCycles);
            if (success) g_npu_matmul_count++;
            break;
        }
        case NpuOpcode::OP_CONV: {
            ConvCommand convCmd;
            convCmd.opcode      = NpuOpcode::OP_CONV;
            convCmd.inputAddr   = cmd.srcAddr;
            convCmd.weightAddr  = cmd.srcAddr + cmd.dimM * cmd.dimN;
            convCmd.biasAddr    = 0;
            convCmd.outputAddr  = cmd.dstAddr;
            convCmd.inputH      = cmd.dimM;
            convCmd.inputW      = cmd.dimN;
            convCmd.kernelSize  = 3;
            convCmd.stride      = 1;
            convCmd.useRelu     = false;
            convCmd.useBias     = false;
            convCmd.depthwise   = false;
            success = conv_execute(convCmd, timeoutCycles);
            if (success) g_npu_conv_count++;
            break;
        }
        case NpuOpcode::OP_ATTN: {
            // 注意力引擎: 简化为矩阵乘法
            MmaCommand mmaCmd;
            mmaCmd.opcode     = NpuOpcode::OP_ATTN;
            mmaCmd.srcAddr    = cmd.srcAddr;
            mmaCmd.weightAddr = cmd.srcAddr + cmd.dimM * cmd.dimK;
            mmaCmd.dstAddr    = cmd.dstAddr;
            mmaCmd.dimM       = cmd.dimM;
            mmaCmd.dimN       = cmd.dimN;
            mmaCmd.dimK       = cmd.dimK;
            success = mma_execute(mmaCmd, timeoutCycles);
            if (success) g_npu_attn_count++;
            break;
        }
        case NpuOpcode::OP_SPARSE: {
            // 稀疏矩阵: 简化为矩阵乘法
            MmaCommand mmaCmd;
            mmaCmd.opcode     = NpuOpcode::OP_SPARSE;
            mmaCmd.srcAddr    = cmd.srcAddr;
            mmaCmd.weightAddr = cmd.srcAddr + cmd.dimM * cmd.dimK;
            mmaCmd.dstAddr    = cmd.dstAddr;
            mmaCmd.dimM       = cmd.dimM;
            mmaCmd.dimN       = cmd.dimN;
            mmaCmd.dimK       = cmd.dimK;
            success = mma_execute(mmaCmd, timeoutCycles);
            if (success) g_npu_sparse_count++;
            break;
        }
        default:
            success = false;
            break;
    }

    g_npu_busy = false;
    g_npu_done = success;
    return success;
#endif
}

bool npu_is_done() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return g_dut->completion_done != 0;
    }
#endif
    return g_npu_done;
}

bool npu_is_busy() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return g_dut->completion_busy != 0;
    }
#endif
    return g_npu_busy;
}

uint32_t npu_get_perf_matmul_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->perfMatMulCount);
    }
#endif
    return g_npu_matmul_count;
}

uint32_t npu_get_perf_conv_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->perfConvCount);
    }
#endif
    return g_npu_conv_count;
}

uint32_t npu_get_perf_attn_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->perfAttnCount);
    }
#endif
    return g_npu_attn_count;
}

uint32_t npu_get_perf_sparse_count() {
#ifdef NPU_HAS_VERILATOR
    if (g_dut != nullptr) {
        return static_cast<uint32_t>(g_dut->perfSparseCount);
    }
#endif
    return g_npu_sparse_count;
}

// ==============================================================================
// 测试统计
// ==============================================================================

TestStats get_stats() {
    TestStats stats;
#ifdef NPU_HAS_VERILATOR
    // Verilator 模式: 从全局计数器获取
    stats.totalTests   = g_npu_pass_count + g_npu_fail_count;
    stats.passedTests  = g_npu_pass_count;
    stats.failedTests  = g_npu_fail_count;
    stats.skippedTests = 0;
#else
    stats.totalTests   = g_total_tests;
    stats.passedTests  = g_passed_tests;
    stats.failedTests  = g_failed_tests;
    stats.skippedTests = g_skipped_tests;
#endif
    return stats;
}

void reset_stats() {
    g_npu_pass_count = 0;
    g_npu_fail_count = 0;
#ifndef NPU_HAS_VERILATOR
    g_total_tests  = 0;
    g_passed_tests  = 0;
    g_failed_tests  = 0;
    g_skipped_tests = 0;
#endif
}

void print_stats_summary() {
    TestStats stats = get_stats();
    printf("\n");
    printf("============================================================\n");
    printf(" NPU Verilator Test Statistics Summary\n");
    printf("============================================================\n");
    printf("  Total Tests:   %u\n", stats.totalTests);
    printf("  Passed Tests:  %u\n", stats.passedTests);
    printf("  Failed Tests:  %u\n", stats.failedTests);
    printf("  Skipped Tests: %u\n", stats.skippedTests);
    printf("------------------------------------------------------------\n");
    printf("  Pass Rate:     %.2f%%\n",
           stats.totalTests > 0 ?
               (100.0 * stats.passedTests / stats.totalTests) : 0.0);
    printf("============================================================\n");
    printf("\n");

    // 打印性能计数器
    printf("Performance Counters:\n");
    printf("  MMA Op Count:      %u\n", mma_get_op_count());
    printf("  MMA Block Count:   %u\n", mma_get_block_count());
    printf("  Conv Op Count:      %u\n", conv_get_op_count());
    printf("  Conv Iter Count:   %u\n", conv_get_iter_count());
    printf("  NPU MatMul Count:  %u\n", npu_get_perf_matmul_count());
    printf("  NPU Conv Count:    %u\n", npu_get_perf_conv_count());
    printf("  NPU Attn Count:    %u\n", npu_get_perf_attn_count());
    printf("  NPU Sparse Count:  %u\n", npu_get_perf_sparse_count());
    // Phase 3.2: 稀疏统计
    printf("  Conv Zero Weight:  %u\n", conv_get_zero_weight_count());
    printf("  Conv Sparsity:     %u%%\n", conv_get_sparsity_rate());
    printf("  Sparse Skip Count: %u\n", sparse_get_skip_count());
    printf("  Sparse Sparsity:   %u%%\n", sparse_get_sparsity_rate());
    // Phase 3.3: Winograd 统计
    printf("  Winograd Op Count: %u\n", winograd_get_op_count());
    // Phase 3.4: DVFS 统计
    printf("  DVFS Transitions:  %u\n", dvfs_get_transition_count());
    printf("  DVFS Current Freq: %u MHz\n", dvfs_get_current_freq());
    printf("  DVFS Current Volt: %u (0.01V/LSB)\n", dvfs_get_current_voltage());
    printf("  DVFS Current Div:  %u\n", dvfs_get_current_divisor());
    // Phase 3.5: 电源域统计
    printf("  PD Up Count:       %u\n", power_domain_get_up_count());
    printf("  PD Down Count:     %u\n", power_domain_get_down_count());
    printf("  PD ALWAYS_ON:      %s\n", power_domain_is_on(PowerDomain::PD_ALWAYS_ON) ? "ON" : "OFF");
    printf("  PD SRAM:           %s\n", power_domain_is_on(PowerDomain::PD_SRAM)      ? "ON" : "OFF");
    printf("  PD NPU_CORE:       %s\n", power_domain_is_on(PowerDomain::PD_NPU_CORE)  ? "ON" : "OFF");
    printf("  PD IO:             %s\n", power_domain_is_on(PowerDomain::PD_IO)        ? "ON" : "OFF");
    // Phase 3.6: 温度监控统计
    printf("  Thermal Samples:   %u\n", thermal_get_sample_count());
    printf("  Thermal Current:  %u C\n", thermal_get_current_temp());
    printf("  Thermal Average:   %u C\n", thermal_get_avg_temp());
    printf("  Thermal Warn Cnt:  %u\n", thermal_get_warn_count());
    printf("  Thermal Crit Cnt:  %u\n", thermal_get_crit_count());
    printf("  Thermal Emerg Cnt: %u\n", thermal_get_emerg_count());
    // Phase 3.7: 阵列扩展统计
    printf("  Array Rows:        %u\n", array_get_rows());
    printf("  Array Cols:        %u\n", array_get_cols());
    printf("  Array Est MTOPS:   %u\n", array_get_est_tops());
    printf("  Array Est GMACs:   %u\n", array_get_est_gmacs());
    printf("  Array Config Cnt:  %u\n", array_get_config_count());
    printf("============================================================\n");
}

} // namespace npu_verilator
