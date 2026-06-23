// ==============================================================================
// NPU 功能测试 Testbench
//
// 测试覆盖:
//   - SystolicCell MAC 操作 (8 个测试)
//   - EccCodec 编解码 (7 个测试)
//   - NpuSram 读写 (6 个测试)
//   - MmaEngine 矩阵乘法 (5 个测试)
//   - ConvEngine 卷积 (4 个测试)
//
// 参考:
//   - IEEE 2992-2024 Neural Network Accelerator Verification
//   - NVIDIA Jetson NPU 测试方法论
//   - Google TPU Systolic Array 测试
//   - Apache TVM VTA 测试框架
// ==============================================================================

#include "npu_verilator_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace npu_verilator;

// ==============================================================================
// 全局统计
// ==============================================================================

uint32_t g_npu_pass_count = 0;
uint32_t g_npu_fail_count = 0;

// SRAM 深度常量 (参考 NpuSramParams.depth 默认值 4096)
static const uint32_t NPU_SRAM_DEPTH    = 4096;
static const uint32_t NPU_SRAM_MAX_ADDR = NPU_SRAM_DEPTH - 1;

// ==============================================================================
// 断言宏 (若头文件未定义则提供回退实现)
// ==============================================================================

#ifndef NPU_CHECK
#define NPU_CHECK(cond, msg) do { \
    if ((cond)) { \
        g_npu_pass_count++; \
        printf("  [PASS] %s\n", msg); \
    } else { \
        g_npu_fail_count++; \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
    } \
} while (0)
#endif

#ifndef NPU_CHECK_EQ
#define NPU_CHECK_EQ(actual, expected, msg) do { \
    auto _npa = (actual); \
    auto _npe = (expected); \
    if (_npa == _npe) { \
        g_npu_pass_count++; \
        printf("  [PASS] %s\n", msg); \
    } else { \
        g_npu_fail_count++; \
        printf("  [FAIL] %s (line %d): actual=%lld, expected=%lld\n", \
               msg, __LINE__, (long long)(_npa), (long long)(_npe)); \
    } \
} while (0)
#endif

#ifndef NPU_CHECK_NEAR
#define NPU_CHECK_NEAR(actual, expected, tolerance, msg) do { \
    auto _npa = (actual); \
    auto _npe = (expected); \
    auto _npt = (tolerance); \
    long long _npdiff = (long long)(_npa) - (long long)(_npe); \
    if (_npdiff < 0) _npdiff = -_npdiff; \
    if (_npdiff <= (long long)(_npt)) { \
        g_npu_pass_count++; \
        printf("  [PASS] %s\n", msg); \
    } else { \
        g_npu_fail_count++; \
        printf("  [FAIL] %s (line %d): actual=%lld, expected=%lld, tol=%lld, diff=%lld\n", \
               msg, __LINE__, (long long)(_npa), (long long)(_npe), \
               (long long)(_npt), _npdiff); \
    } \
} while (0)
#endif

// ==============================================================================
// 测试函数声明
// ==============================================================================

void test_systolic_cell();
void test_ecc_codec();
void test_npu_sram();
void test_mma_engine();
void test_conv_engine();
void test_sparse_engine();  // Phase 3.2: 稀疏加速测试
void test_winograd_engine();  // Phase 3.3: Winograd 卷积测试
void test_dvfs_controller();  // Phase 3.4: DVFS 测试
void test_power_domain();  // Phase 3.5: 电源域测试
void test_thermal_monitor();  // Phase 3.6: 温度监控测试
void test_array_scaling();  // Phase 3.7: 阵列扩展测试

// ==============================================================================
// 主函数
// ==============================================================================

int main(int argc, char** argv) {
    printf("======================================================================\n");
    printf("NPU Verilator 功能测试\n");
    printf("======================================================================\n\n");

    // 初始化
    init();
    reset_stats();

    // 运行测试
    test_systolic_cell();
    test_ecc_codec();
    test_npu_sram();
    test_mma_engine();
    test_conv_engine();
    test_sparse_engine();  // Phase 3.2: 稀疏加速测试
    test_winograd_engine();  // Phase 3.3: Winograd 卷积测试
    test_dvfs_controller();  // Phase 3.4: DVFS 测试
    test_power_domain();  // Phase 3.5: 电源域测试
    test_thermal_monitor();  // Phase 3.6: 温度监控测试
    test_array_scaling();  // Phase 3.7: 阵列扩展测试

    // 打印摘要
    print_stats_summary();

    // 关闭
    shutdown();

    return (g_npu_fail_count > 0) ? 1 : 0;
}

// ==============================================================================
// A. SystolicCell 测试 (8 个)
// ==============================================================================

void test_systolic_cell() {
    printf("\n--- SystolicCell 测试 ---\n");

    // 1. clear 重置累加基: clear=true 时 accBase=0, cOut=0+a*b=35
    {
        auto r = systolic_cell_mac(5, 7, 100, true, true);
        NPU_CHECK_EQ(r.cOut, 35, "SystolicCell #1: clear resets accBase, cOut=0+5*7=35");
    }

    // 2. 基本 MAC: 2*3=6, cIn=0 -> cOut=6
    {
        auto r = systolic_cell_mac(2, 3, 0, false, true);
        NPU_CHECK_EQ(r.cOut, 6, "SystolicCell #2: 2*3+0=6");
    }

    // 3. 累加: 2*3+10=16, cIn=10 -> cOut=16
    {
        auto r = systolic_cell_mac(2, 3, 10, false, true);
        NPU_CHECK_EQ(r.cOut, 16, "SystolicCell #3: 2*3+10=16");
    }

    // 4. clear 重置累加基: clear=true 时 accBase=0, cOut=0+a*b=35
    {
        auto r = systolic_cell_mac(5, 7, 100, true, true);
        NPU_CHECK_EQ(r.cOut, 35, "SystolicCell #4: clear resets accBase, cOut=0+5*7=35");
    }

    // 5. en=false 保持: en=false 时不执行 MAC, cOut=cIn
    {
        auto r = systolic_cell_mac(5, 7, 42, false, false);
        NPU_CHECK_EQ(r.cOut, 42, "SystolicCell #5: en=false preserves cIn");
    }

    // 6. 负数乘法: (-2)*3=-6
    {
        auto r = systolic_cell_mac(-2, 3, 0, false, true);
        NPU_CHECK_EQ(r.cOut, -6, "SystolicCell #6: (-2)*3=-6");
    }

    // 7. 数据流传播: aOut=aIn, bOut=bIn
    {
        auto r = systolic_cell_mac(42, 99, 0, false, true);
        NPU_CHECK_EQ((int)r.aOut, 42, "SystolicCell #7: aOut=aIn");
        NPU_CHECK_EQ((int)r.bOut, 99, "SystolicCell #7: bOut=bIn");
    }

    // 8. 有符号乘法: (-1)*(-1)=1
    {
        auto r = systolic_cell_mac(-1, -1, 0, false, true);
        NPU_CHECK_EQ(r.cOut, 1, "SystolicCell #8: (-1)*(-1)=1");
    }
}

// ==============================================================================
// B. EccCodec 测试 (7 个)
// ==============================================================================

void test_ecc_codec() {
    printf("\n--- EccCodec 测试 ---\n");

    // 9. 编码正确性: ecc_encode(0x00) 和 ecc_encode(0xFF) 结果不同
    {
        uint16_t enc0  = ecc_encode(0x00);
        uint16_t encFF = ecc_encode(0xFF);
        NPU_CHECK(enc0 != encFF, "EccCodec #9: encode(0x00) != encode(0xFF)");
    }

    // 10. 无错误解码: encode->decode, errorType=NO_ERROR
    {
        uint16_t enc = ecc_encode(0x5A);
        EccResult r  = ecc_decode(enc);
        NPU_CHECK_EQ((uint32_t)r.errorType, (uint32_t)EccErrorType::NO_ERROR,
                     "EccCodec #10: no error on clean codeword");
        NPU_CHECK_EQ(r.decoded, (uint32_t)0x5A,
                     "EccCodec #10: decoded data matches original");
    }

    // 11. SEC 单比特纠正: 注入单比特错误后解码, errorType=SINGLE_BIT, decoded 正确
    {
        uint8_t  data = 0x3C;
        uint16_t enc  = ecc_encode(data);
        uint16_t cor  = ecc_inject_single_bit_error(enc, 4);
        EccResult r   = ecc_decode(cor);
        NPU_CHECK_EQ((uint32_t)r.errorType, (uint32_t)EccErrorType::SINGLE_BIT,
                     "EccCodec #11: single bit error detected");
        NPU_CHECK_EQ(r.decoded, (uint32_t)data,
                     "EccCodec #11: single bit error corrected");
    }

    // 12. DED 双比特检测: 注入双比特错误后解码, errorType=DOUBLE_BIT
    {
        uint16_t enc = ecc_encode(0xA5);
        uint16_t cor = ecc_inject_double_bit_error(enc, 2, 7);
        EccResult r  = ecc_decode(cor);
        NPU_CHECK_EQ((uint32_t)r.errorType, (uint32_t)EccErrorType::DOUBLE_BIT,
                     "EccCodec #12: double bit error detected (uncorrectable)");
    }

    // 13. 错误综合征: 不同位错误产生不同 syndrome
    {
        uint16_t enc = ecc_encode(0x5A);
        uint16_t e1  = ecc_inject_single_bit_error(enc, 1);
        uint16_t e2  = ecc_inject_single_bit_error(enc, 6);
        EccResult r1 = ecc_decode(e1);
        EccResult r2 = ecc_decode(e2);
        NPU_CHECK(r1.syndrome != r2.syndrome,
                  "EccCodec #13: different bit errors yield different syndromes");
    }

    // 14. 多数据模式: 测试 0x00, 0xFF, 0x55, 0xAA, 0x12, 0x34
    {
        const uint8_t patterns[] = {0x00, 0xFF, 0x55, 0xAA, 0x12, 0x34};
        const int num = sizeof(patterns) / sizeof(patterns[0]);
        bool all_ok = true;
        for (int i = 0; i < num; ++i) {
            uint16_t enc = ecc_encode(patterns[i]);
            EccResult r  = ecc_decode(enc);
            if ((uint32_t)r.errorType != (uint32_t)EccErrorType::NO_ERROR ||
                r.decoded != (uint32_t)patterns[i]) {
                all_ok = false;
                printf("    pattern 0x%02X failed: err=%u decoded=0x%08X\n",
                       patterns[i], (uint32_t)r.errorType, r.decoded);
            }
        }
        NPU_CHECK(all_ok, "EccCodec #14: multiple data patterns round-trip");
    }

    // 15. 编解码往返: encode->decode 往返一致性
    {
        bool all_ok = true;
        for (uint32_t d = 0; d <= 0xFF; ++d) {
            uint16_t enc = ecc_encode((uint8_t)d);
            EccResult r  = ecc_decode(enc);
            if (r.decoded != d ||
                (uint32_t)r.errorType != (uint32_t)EccErrorType::NO_ERROR) {
                all_ok = false;
                break;
            }
        }
        NPU_CHECK(all_ok, "EccCodec #15: full 0x00-0xFF round-trip consistency");
    }
}

// ==============================================================================
// C. NpuSram 测试 (6 个)
// ==============================================================================

void test_npu_sram() {
    printf("\n--- NpuSram 测试 ---\n");

    // 16. 基本写入读取: write(0x100, 0xDEAD) -> read(0x100)=0xDEAD
    {
        sram_clear();
        sram_write_a(0x100, 0xDEAD);
        uint32_t val = sram_read_a(0x100);
        NPU_CHECK_EQ(val, (uint32_t)0xDEAD, "NpuSram #16: basic write/read");
    }

    // 17. Port A/B 独立: A 写, B 读
    {
        sram_clear();
        sram_write_a(0x200, 0xCAFE);
        uint32_t val = sram_read_b(0x200);
        NPU_CHECK_EQ(val, (uint32_t)0xCAFE,
                     "NpuSram #17: port A write visible to port B");
    }

    // 18. 字节级写入: byteEnable=true, wstrb=0x01 只写最低字节
    {
        sram_clear();
        // 先写入全零
        sram_write_a(0x150, 0x00000000);
        // 仅写最低字节 (0x78 来自 0x12345678)
        sram_write_a(0x150, 0x12345678, true, 0x01);
        uint32_t val = sram_read_a(0x150);
        NPU_CHECK_EQ(val, (uint32_t)0x00000078,
                     "NpuSram #18: byte write wstrb=0x01 writes LSB only");
    }

    // 19. 地址范围: 写入地址 0 和最大地址
    {
        sram_clear();
        sram_write_a(0x000, 0x1111);
        sram_write_a(NPU_SRAM_MAX_ADDR, 0x2222);
        uint32_t v0  = sram_read_a(0x000);
        uint32_t vMax = sram_read_a(NPU_SRAM_MAX_ADDR);
        NPU_CHECK_EQ(v0, (uint32_t)0x1111, "NpuSram #19: address 0 accessible");
        NPU_CHECK_EQ(vMax, (uint32_t)0x2222, "NpuSram #19: max address accessible");
    }

    // 20. 覆盖写入: 同一地址多次写入
    {
        sram_clear();
        sram_write_a(0x080, 0xAAAA);
        sram_write_a(0x080, 0xBBBB);
        sram_write_a(0x080, 0xCCCC);
        uint32_t val = sram_read_a(0x080);
        NPU_CHECK_EQ(val, (uint32_t)0xCCCC,
                     "NpuSram #20: last write wins on overwrite");
    }

    // 21. 清空 SRAM: sram_clear() 后所有地址为 0
    {
        sram_write_a(0x100, 0xDEAD);
        sram_write_a(0x200, 0xBEEF);
        sram_clear();
        uint32_t v1 = sram_read_a(0x100);
        uint32_t v2 = sram_read_a(0x200);
        uint32_t v3 = sram_read_a(0x300);
        NPU_CHECK_EQ(v1, (uint32_t)0, "NpuSram #21: clear zeroes addr 0x100");
        NPU_CHECK_EQ(v2, (uint32_t)0, "NpuSram #21: clear zeroes addr 0x200");
        NPU_CHECK_EQ(v3, (uint32_t)0, "NpuSram #21: clear zeroes addr 0x300");
    }
}

// ==============================================================================
// D. MmaEngine 矩阵乘法测试 (5 个)
//
// 数据布局 (row-major, 紧密存储, 与软件模型一致):
//   A[m][k] 位于 srcAddr    + m*dimK + k
//   B[k][n] 位于 weightAddr + k*dimN + n
//   C[m][n] 位于 dstAddr    + m*dimN + n
// 输入数据为 int8_t (经 uint32_t 写入), 输出为 int32_t
// ==============================================================================

void test_mma_engine() {
    printf("\n--- MmaEngine 矩阵乘法测试 ---\n");

    // 22. 1x1 矩阵乘法: A=[[2]], B=[[3]], C=[[6]]
    {
        sram_clear();
        // A=[[2]]
        sram_write_a(0x000, 2);
        // B=[[3]]
        sram_write_a(0x100, 3);

        MmaCommand cmd;
        cmd.opcode     = NpuOpcode::OP_MATMUL;
        cmd.srcAddr    = 0x000;
        cmd.weightAddr = 0x100;
        cmd.dstAddr    = 0x200;
        cmd.dimM       = 1;
        cmd.dimN       = 1;
        cmd.dimK       = 1;

        bool ok = mma_execute(cmd);
        NPU_CHECK(ok, "MmaEngine #22: 1x1 matmul execute");

        uint32_t result = sram_read_a(0x200);
        NPU_CHECK_EQ(result, (uint32_t)6, "MmaEngine #22: 1x1 matmul result = 2*3 = 6");
    }

    // 23. 2x2 矩阵乘法: A=[[1,2],[3,4]], B=[[5,6],[7,8]], C=[[19,22],[43,50]]
    {
        sram_clear();
        // A = [[1,2],[3,4]] (row-major, stride=dimK=2)
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // B = [[5,6],[7,8]] (row-major, stride=dimN=2)
        sram_write_a(0x100, 5); sram_write_a(0x101, 6);
        sram_write_a(0x102, 7); sram_write_a(0x103, 8);

        MmaCommand cmd;
        cmd.opcode     = NpuOpcode::OP_MATMUL;
        cmd.srcAddr    = 0x000;
        cmd.weightAddr = 0x100;
        cmd.dstAddr    = 0x200;
        cmd.dimM       = 2;
        cmd.dimN       = 2;
        cmd.dimK       = 2;

        bool ok = mma_execute(cmd);
        NPU_CHECK(ok, "MmaEngine #23: 2x2 matmul execute");

        // C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        uint32_t c00 = sram_read_a(0x200);
        uint32_t c01 = sram_read_a(0x201);
        uint32_t c10 = sram_read_a(0x202);
        uint32_t c11 = sram_read_a(0x203);
        NPU_CHECK_EQ(c00, (uint32_t)19, "MmaEngine #23: C[0][0] = 1*5+2*7 = 19");
        NPU_CHECK_EQ(c01, (uint32_t)22, "MmaEngine #23: C[0][1] = 1*6+2*8 = 22");
        NPU_CHECK_EQ(c10, (uint32_t)43, "MmaEngine #23: C[1][0] = 3*5+4*7 = 43");
        NPU_CHECK_EQ(c11, (uint32_t)50, "MmaEngine #23: C[1][1] = 3*6+4*8 = 50");
    }

    // 24. 含零矩阵: A=[[0,0],[0,0]], B=[[1,2],[3,4]], C=[[0,0],[0,0]]
    {
        sram_clear();
        // A 全零 (sram_clear 后)
        // B = [[1,2],[3,4]] (row-major, stride=dimN=2)
        sram_write_a(0x100, 1); sram_write_a(0x101, 2);
        sram_write_a(0x102, 3); sram_write_a(0x103, 4);

        MmaCommand cmd;
        cmd.opcode     = NpuOpcode::OP_MATMUL;
        cmd.srcAddr    = 0x000;  // A 全零
        cmd.weightAddr = 0x100;
        cmd.dstAddr    = 0x200;
        cmd.dimM       = 2;
        cmd.dimN       = 2;
        cmd.dimK       = 2;

        mma_execute(cmd);
        NPU_CHECK_EQ(sram_read_a(0x200), (uint32_t)0,
                     "MmaEngine #24: zero matrix C[0][0]=0");
        NPU_CHECK_EQ(sram_read_a(0x201), (uint32_t)0,
                     "MmaEngine #24: zero matrix C[0][1]=0");
        NPU_CHECK_EQ(sram_read_a(0x202), (uint32_t)0,
                     "MmaEngine #24: zero matrix C[1][0]=0");
        NPU_CHECK_EQ(sram_read_a(0x203), (uint32_t)0,
                     "MmaEngine #24: zero matrix C[1][1]=0");
    }

    // 25. 负数矩阵: A=[[-1,2],[3,-4]], B=[[5,-6],[-7,8]], C=[[-19,22],[43,-50]]
    {
        sram_clear();
        // A = [[-1,2],[3,-4]] (int8_t, stride=dimK=2)
        sram_write_a(0x000, (uint32_t)(uint8_t)(-1)); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, (uint32_t)(uint8_t)(-4));
        // B = [[5,-6],[-7,8]] (stride=dimN=2)
        sram_write_a(0x100, 5); sram_write_a(0x101, (uint32_t)(uint8_t)(-6));
        sram_write_a(0x102, (uint32_t)(uint8_t)(-7)); sram_write_a(0x103, 8);

        MmaCommand cmd;
        cmd.opcode     = NpuOpcode::OP_MATMUL;
        cmd.srcAddr    = 0x000;
        cmd.weightAddr = 0x100;
        cmd.dstAddr    = 0x200;
        cmd.dimM       = 2;
        cmd.dimN       = 2;
        cmd.dimK       = 2;

        mma_execute(cmd);
        // C = [[-1*5+2*(-7), -1*(-6)+2*8],[3*5+(-4)*(-7), 3*(-6)+(-4)*8]]
        //   = [[-19, 22],[43, -50]]
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), -19,
                     "MmaEngine #25: neg C[0][0]=-19");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x201), 22,
                     "MmaEngine #25: neg C[0][1]=22");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x202), 43,
                     "MmaEngine #25: neg C[1][0]=43");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x203), -50,
                     "MmaEngine #25: neg C[1][1]=-50");
    }

    // 26. 非方矩阵: A(2x3) * B(3x2) = C(2x2)
    //     A = [[1,2,3],[4,5,6]]
    //     B = [[1,2],[3,4],[5,6]]
    //     C = [[1+6+15, 2+8+18],[4+15+30, 8+20+36]] = [[22,28],[49,64]]
    {
        sram_clear();
        // A (2x3, stride=dimK=3)
        sram_write_a(0x000, 1); sram_write_a(0x001, 2); sram_write_a(0x002, 3);
        sram_write_a(0x003, 4); sram_write_a(0x004, 5); sram_write_a(0x005, 6);
        // B (3x2, stride=dimN=2)
        sram_write_a(0x100, 1); sram_write_a(0x101, 2);
        sram_write_a(0x102, 3); sram_write_a(0x103, 4);
        sram_write_a(0x104, 5); sram_write_a(0x105, 6);

        MmaCommand cmd;
        cmd.opcode     = NpuOpcode::OP_MATMUL;
        cmd.srcAddr    = 0x000;
        cmd.weightAddr = 0x100;
        cmd.dstAddr    = 0x200;
        cmd.dimM       = 2;
        cmd.dimN       = 2;
        cmd.dimK       = 3;

        bool ok = mma_execute(cmd);
        NPU_CHECK(ok, "MmaEngine #26: non-square matmul execute");

        // C = [[22,28],[49,64]] (stride=dimN=2)
        NPU_CHECK_EQ(sram_read_a(0x200), (uint32_t)22,
                     "MmaEngine #26: C[0][0] = 1*1+2*3+3*5 = 22");
        NPU_CHECK_EQ(sram_read_a(0x201), (uint32_t)28,
                     "MmaEngine #26: C[0][1] = 1*2+2*4+3*6 = 28");
        NPU_CHECK_EQ(sram_read_a(0x202), (uint32_t)49,
                     "MmaEngine #26: C[1][0] = 4*1+5*3+6*5 = 49");
        NPU_CHECK_EQ(sram_read_a(0x203), (uint32_t)64,
                     "MmaEngine #26: C[1][1] = 4*2+5*4+6*6 = 64");
    }
}

// ==============================================================================
// E. ConvEngine 卷积测试 (4 个)
//
// 数据布局 (row-major, 连续存储):
//   input[i][j]  位于 inputAddr  + i*inputW     + j
//   kernel[i][j] 位于 weightAddr + i*kernelSize + j
//   output[i][j] 位于 outputAddr + i*outputW    + j
//   outputW = (inputW - kernelSize)/stride + 1
//   outputH = (inputH - kernelSize)/stride + 1
// ==============================================================================

void test_conv_engine() {
    printf("\n--- ConvEngine 卷积测试 ---\n");

    // 27. 1x1 卷积 (等同矩阵乘法):
    //     input=[[1,2],[3,4]], kernel=[[1]], output=[[1,2],[3,4]]
    {
        sram_clear();
        // input 2x2 at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // kernel 1x1 at 0x100
        sram_write_a(0x100, 1);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = false;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #27: 1x1 conv execute");

        // output 2x2: each element = input * 1
        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)1,
                     "ConvEngine #27: out[0][0]=1");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)2,
                     "ConvEngine #27: out[0][1]=2");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)3,
                     "ConvEngine #27: out[1][0]=3");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)4,
                     "ConvEngine #27: out[1][1]=4");
    }

    // 28. 3x3 卷积: input=4x4, kernel=3x3(全1), stride=1, output=2x2
    //     input = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    //     kernel = [[1,1,1],[1,1,1],[1,1,1]]
    //     out[0][0] = 1+2+3+5+6+7+9+10+11 = 54
    //     out[0][1] = 2+3+4+6+7+8+10+11+12 = 63
    //     out[1][0] = 5+6+7+9+10+11+13+14+15 = 90
    //     out[1][1] = 6+7+8+10+11+12+14+15+16 = 99
    {
        sram_clear();
        // input 4x4 at 0x000 (contiguous)
        sram_write_a(0x000, 1);  sram_write_a(0x001, 2);  sram_write_a(0x002, 3);  sram_write_a(0x003, 4);
        sram_write_a(0x004, 5);  sram_write_a(0x005, 6);  sram_write_a(0x006, 7);  sram_write_a(0x007, 8);
        sram_write_a(0x008, 9);  sram_write_a(0x009, 10); sram_write_a(0x00A, 11); sram_write_a(0x00B, 12);
        sram_write_a(0x00C, 13); sram_write_a(0x00D, 14); sram_write_a(0x00E, 15); sram_write_a(0x00F, 16);
        // kernel 3x3 at 0x100 (all ones)
        for (int i = 0; i < 9; ++i) {
            sram_write_a(0x100 + i, 1);
        }

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 4;
        cmd.inputW     = 4;
        cmd.kernelSize = 3;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = false;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #28: 3x3 conv execute");

        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)54,
                     "ConvEngine #28: out[0][0]=54");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)63,
                     "ConvEngine #28: out[0][1]=63");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)90,
                     "ConvEngine #28: out[1][0]=90");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)99,
                     "ConvEngine #28: out[1][1]=99");
    }

    // 29. ReLU 激活: useRelu=true, 负数输出变为 0
    //     input=[[1,2],[3,4]], kernel=[[-1]]
    //     无 ReLU: output=[[-1,-2],[-3,-4]]
    //     有 ReLU: output=[[0,0],[0,0]]
    {
        sram_clear();
        // input 2x2 at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // kernel 1x1 = -1 at 0x100
        sram_write_a(0x100, (uint32_t)(uint8_t)(-1));

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = true;
        cmd.useBias     = false;
        cmd.depthwise   = false;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #29: ReLU conv execute");

        // 所有输出 <= 0, ReLU 后应为 0
        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)0,
                     "ConvEngine #29: ReLU out[0][0]=0");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)0,
                     "ConvEngine #29: ReLU out[0][1]=0");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)0,
                     "ConvEngine #29: ReLU out[1][0]=0");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)0,
                     "ConvEngine #29: ReLU out[1][1]=0");
    }

    // 30. Depthwise 卷积: depthwise=true
    //     单通道 1x1 depthwise 等同于逐元素乘法
    //     input=[[1,2],[3,4]], kernel=[[2]], output=[[2,4],[6,8]]
    {
        sram_clear();
        // input 2x2 at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // kernel 1x1 = 2 at 0x100
        sram_write_a(0x100, 2);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = true;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #30: depthwise conv execute");

        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)2,
                     "ConvEngine #30: depthwise out[0][0]=1*2=2");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)4,
                     "ConvEngine #30: depthwise out[0][1]=2*2=4");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)6,
                     "ConvEngine #30: depthwise out[1][0]=3*2=6");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)8,
                     "ConvEngine #30: depthwise out[1][1]=4*2=8");
    }

    // ============================================================
    // Phase 2.7: 算子融合 Conv+BN+ReLU 测试
    // ============================================================

    // 31. BN 融合: Conv + BN (scale=2, shift=1)
    //     input=[[1,2],[3,4]], kernel=[[1]], scale=2, shift=1
    //     conv: [[1,2],[3,4]], bn: 2*1+1=3, 2*2+1=5, 2*3+1=7, 2*4+1=9
    {
        sram_clear();
        // input 2x2 at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // kernel 1x1 = 1 at 0x100
        sram_write_a(0x100, 1);
        // BN scale at 0x400: [2, 2, 2, 2]
        sram_write_a(0x400, 2); sram_write_a(0x401, 2);
        sram_write_a(0x402, 2); sram_write_a(0x403, 2);
        // BN shift at 0x500: [1, 1, 1, 1]
        sram_write_a(0x500, 1); sram_write_a(0x501, 1);
        sram_write_a(0x502, 1); sram_write_a(0x503, 1);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = false;
        cmd.useBatchNorm = true;
        cmd.bnScaleAddr  = 0x400;
        cmd.bnShiftAddr  = 0x500;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #31: BN fusion execute");

        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)3,
                     "ConvEngine #31: BN out[0][0]=2*1+1=3");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)5,
                     "ConvEngine #31: BN out[0][1]=2*2+1=5");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)7,
                     "ConvEngine #31: BN out[1][0]=2*3+1=7");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)9,
                     "ConvEngine #31: BN out[1][1]=2*4+1=9");
    }

    // 32. 完整融合: Conv + Bias + BN + ReLU
    //     input=[[1,2],[3,4]], kernel=[[1]], bias=[10,10,10,10]
    //     scale=1, shift=-5
    //     conv: [[1,2],[3,4]], +bias: [[11,12,13,14]]
    //     bn: 1*11-5=6, 1*12-5=7, 1*13-5=8, 1*14-5=9
    {
        sram_clear();
        // input 2x2 at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // kernel 1x1 = 1 at 0x100
        sram_write_a(0x100, 1);
        // bias at 0x200: [10, 10, 10, 10]
        sram_write_a(0x200, 10); sram_write_a(0x201, 10);
        sram_write_a(0x202, 10); sram_write_a(0x203, 10);
        // BN scale at 0x400: [1, 1, 1, 1]
        sram_write_a(0x400, 1); sram_write_a(0x401, 1);
        sram_write_a(0x402, 1); sram_write_a(0x403, 1);
        // BN shift at 0x500: [-5, -5, -5, -5]
        sram_write_a(0x500, (uint32_t)(int32_t)(-5));
        sram_write_a(0x501, (uint32_t)(int32_t)(-5));
        sram_write_a(0x502, (uint32_t)(int32_t)(-5));
        sram_write_a(0x503, (uint32_t)(int32_t)(-5));

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x200;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = true;
        cmd.depthwise   = false;
        cmd.useBatchNorm = true;
        cmd.bnScaleAddr  = 0x400;
        cmd.bnShiftAddr  = 0x500;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #32: Conv+Bias+BN fusion execute");

        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)6,
                     "ConvEngine #32: BN out[0][0]=1*(1+10)-5=6");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)7,
                     "ConvEngine #32: BN out[0][1]=1*(2+10)-5=7");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)8,
                     "ConvEngine #32: BN out[1][0]=1*(3+10)-5=8");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)9,
                     "ConvEngine #32: BN out[1][1]=1*(4+10)-5=9");
    }

    // 33. 完整融合 + ReLU: Conv + Bias + BN + ReLU (负数变 0)
    //     input=[[1,2],[3,4]], kernel=[[-1]], bias=[0,0,0,0]
    //     scale=1, shift=0
    //     conv: [[-1,-2],[-3,-4]], bn: -1, -2, -3, -4
    //     relu: 0, 0, 0, 0
    {
        sram_clear();
        // input 2x2 at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // kernel 1x1 = -1 at 0x100
        sram_write_a(0x100, (uint32_t)(uint8_t)(-1));
        // BN scale at 0x400: [1, 1, 1, 1]
        sram_write_a(0x400, 1); sram_write_a(0x401, 1);
        sram_write_a(0x402, 1); sram_write_a(0x403, 1);
        // BN shift at 0x500: [0, 0, 0, 0]
        sram_write_a(0x500, 0); sram_write_a(0x501, 0);
        sram_write_a(0x502, 0); sram_write_a(0x503, 0);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = true;
        cmd.useBias     = false;
        cmd.depthwise   = false;
        cmd.useBatchNorm = true;
        cmd.bnScaleAddr  = 0x400;
        cmd.bnShiftAddr  = 0x500;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #33: Conv+BN+ReLU fusion execute");

        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)0,
                     "ConvEngine #33: ReLU out[0][0]=0");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)0,
                     "ConvEngine #33: ReLU out[0][1]=0");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)0,
                     "ConvEngine #33: ReLU out[1][0]=0");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)0,
                     "ConvEngine #33: ReLU out[1][1]=0");
    }

    // ==================================================================
    // Phase 3.1: INT4 量化测试
    // ==================================================================

    // 34. INT4 1x1 卷积 (正值): input=[[2,3],[4,5]], kernel=[[2]], output=[[4,6],[8,10]]
    //     INT4 打包: 值 v (-8~7) 存储为 (v & 0xF) 在 SRAM 低 4 位
    {
        sram_clear();
        // input 2x2 INT4 at 0x000 (每个地址存一个 INT4 值在低 4 位)
        sram_write_a(0x000, 0x2); sram_write_a(0x001, 0x3);
        sram_write_a(0x002, 0x4); sram_write_a(0x003, 0x5);
        // kernel 1x1 INT4 at 0x100
        sram_write_a(0x100, 0x2);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = false;
        cmd.quantMode   = QuantMode::INT4;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #34: INT4 1x1 conv execute");

        // output 2x2: each element = input * 2
        NPU_CHECK_EQ(sram_read_a(0x300), (uint32_t)4,
                     "ConvEngine #34: INT4 out[0][0]=2*2=4");
        NPU_CHECK_EQ(sram_read_a(0x301), (uint32_t)6,
                     "ConvEngine #34: INT4 out[0][1]=3*2=6");
        NPU_CHECK_EQ(sram_read_a(0x302), (uint32_t)8,
                     "ConvEngine #34: INT4 out[1][0]=4*2=8");
        NPU_CHECK_EQ(sram_read_a(0x303), (uint32_t)10,
                     "ConvEngine #34: INT4 out[1][1]=5*2=10");
    }

    // 35. INT4 1x1 卷积 (含负值): input=[[-2,3],[-4,5]], kernel=[[2]]
    //     output=[[-4,6],[-8,10]]
    //     INT4 负值: -2 -> 0xE, -4 -> 0xC (4-bit two's complement)
    {
        sram_clear();
        // input 2x2 INT4 (负值用 4-bit two's complement)
        sram_write_a(0x000, 0xE); sram_write_a(0x001, 0x3);  // -2, 3
        sram_write_a(0x002, 0xC); sram_write_a(0x003, 0x5);  // -4, 5
        // kernel 1x1 INT4
        sram_write_a(0x100, 0x2);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = false;
        cmd.quantMode   = QuantMode::INT4;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #35: INT4 conv with negatives execute");

        // output: -2*2=-4, 3*2=6, -4*2=-8, 5*2=10
        NPU_CHECK_EQ((int32_t)sram_read_a(0x300), (int32_t)-4,
                     "ConvEngine #35: INT4 out[0][0]=(-2)*2=-4");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x301), (int32_t)6,
                     "ConvEngine #35: INT4 out[0][1]=3*2=6");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x302), (int32_t)-8,
                     "ConvEngine #35: INT4 out[1][0]=(-4)*2=-8");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x303), (int32_t)10,
                     "ConvEngine #35: INT4 out[1][1]=5*2=10");
    }

    // 36. INT4 3x3 卷积: input=4x4, kernel=3x3(全1), stride=1, output=2x2
    //     INT4 范围: -8 ~ +7, 所有输入值必须在 INT4 范围内
    //     input = [[1,2,3,4],[5,6,7,7],[-3,-2,-1,0],[1,1,1,1]]
    //     kernel = [[1,1,1],[1,1,1],[1,1,1]]
    //     out[0][0] = 1+2+3+5+6+7+(-3)+(-2)+(-1) = 18
    //     out[0][1] = 2+3+4+6+7+7+(-2)+(-1)+0 = 26
    //     out[1][0] = 5+6+7+(-3)+(-2)+(-1)+1+1+1 = 15
    //     out[1][1] = 6+7+7+(-2)+(-1)+0+1+1+1 = 20
    {
        sram_clear();
        // input 4x4 INT4 at 0x000 (所有值在 INT4 范围 -8~7 内)
        sram_write_a(0x000, 0x1); sram_write_a(0x001, 0x2); sram_write_a(0x002, 0x3); sram_write_a(0x003, 0x4);
        sram_write_a(0x004, 0x5); sram_write_a(0x005, 0x6); sram_write_a(0x006, 0x7); sram_write_a(0x007, 0x7);  // 最后一列用 7 (INT4 max)
        sram_write_a(0x008, 0xD); sram_write_a(0x009, 0xE); sram_write_a(0x00A, 0xF); sram_write_a(0x00B, 0x0);  // -3,-2,-1,0
        sram_write_a(0x00C, 0x1); sram_write_a(0x00D, 0x1); sram_write_a(0x00E, 0x1); sram_write_a(0x00F, 0x1);
        // kernel 3x3 INT4 at 0x100 (all ones)
        for (int i = 0; i < 9; ++i) {
            sram_write_a(0x100 + i, 0x1);
        }

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 4;
        cmd.inputW     = 4;
        cmd.kernelSize = 3;
        cmd.stride      = 1;
        cmd.useRelu     = false;
        cmd.useBias     = false;
        cmd.depthwise   = false;
        cmd.quantMode   = QuantMode::INT4;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #36: INT4 3x3 conv execute");

        // out[0][0] = 1+2+3+5+6+7+(-3)+(-2)+(-1) = 18
        NPU_CHECK_EQ((int32_t)sram_read_a(0x300), (int32_t)18,
                     "ConvEngine #36: INT4 out[0][0]=18");
        // out[0][1] = 2+3+4+6+7+7+(-2)+(-1)+0 = 26
        NPU_CHECK_EQ((int32_t)sram_read_a(0x301), (int32_t)26,
                     "ConvEngine #36: INT4 out[0][1]=26");
        // out[1][0] = 5+6+7+(-3)+(-2)+(-1)+1+1+1 = 15
        NPU_CHECK_EQ((int32_t)sram_read_a(0x302), (int32_t)15,
                     "ConvEngine #36: INT4 out[1][0]=15");
        // out[1][1] = 6+7+7+(-2)+(-1)+0+1+1+1 = 20
        NPU_CHECK_EQ((int32_t)sram_read_a(0x303), (int32_t)20,
                     "ConvEngine #36: INT4 out[1][1]=20");
    }

    // 37. INT4 卷积 + ReLU: input=[[-2,3],[-4,5]], kernel=[[1]], output=[[0,3],[0,5]]
    {
        sram_clear();
        // input 2x2 INT4 (含负值)
        sram_write_a(0x000, 0xE); sram_write_a(0x001, 0x3);  // -2, 3
        sram_write_a(0x002, 0xC); sram_write_a(0x003, 0x5);  // -4, 5
        // kernel 1x1 INT4 = 1
        sram_write_a(0x100, 0x1);

        ConvCommand cmd;
        cmd.opcode     = NpuOpcode::OP_CONV;
        cmd.inputAddr  = 0x000;
        cmd.weightAddr = 0x100;
        cmd.biasAddr   = 0x000;
        cmd.outputAddr = 0x300;
        cmd.inputH     = 2;
        cmd.inputW     = 2;
        cmd.kernelSize = 1;
        cmd.stride      = 1;
        cmd.useRelu     = true;
        cmd.useBias     = false;
        cmd.depthwise   = false;
        cmd.quantMode   = QuantMode::INT4;

        bool ok = conv_execute(cmd);
        NPU_CHECK(ok, "ConvEngine #37: INT4 conv + ReLU execute");

        // output: ReLU(-2)=0, ReLU(3)=3, ReLU(-4)=0, ReLU(5)=5
        NPU_CHECK_EQ((int32_t)sram_read_a(0x300), (int32_t)0,
                     "ConvEngine #37: INT4 ReLU out[0][0]=0");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x301), (int32_t)3,
                     "ConvEngine #37: INT4 ReLU out[0][1]=3");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x302), (int32_t)0,
                     "ConvEngine #37: INT4 ReLU out[1][0]=0");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x303), (int32_t)5,
                     "ConvEngine #37: INT4 ReLU out[1][1]=5");
    }
}

// ==============================================================================
// Phase 3.2: SparseEngine 稀疏加速测试
// ==============================================================================

void test_sparse_engine() {
    printf("\n--- SparseEngine 稀疏加速测试 ---\n");

    // 38. 稠密模式点积: sparse=[1,2,3,4], dense=[5,6,7,8], result=1*5+2*6+3*7+4*8=70
    {
        sram_clear();
        // sparse vector at 0x000
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // dense vector at 0x100
        sram_write_a(0x100, 5); sram_write_a(0x101, 6);
        sram_write_a(0x102, 7); sram_write_a(0x103, 8);

        SparseCommand cmd;
        cmd.mode       = SparseMode::DENSE;
        cmd.sparseAddr = 0x000;
        cmd.denseAddr  = 0x100;
        cmd.resultAddr = 0x200;
        cmd.maskAddr   = 0;
        cmd.vectorLen  = 4;

        bool ok = sparse_execute(cmd);
        NPU_CHECK(ok, "SparseEngine #38: dense dot product execute");

        // 1*5+2*6+3*7+4*8 = 5+12+21+32 = 70
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)70,
                     "SparseEngine #38: dense dot product = 70");
    }

    // 39. 动态零值检测: sparse=[0,2,0,4], dense=[5,6,7,8]
    //     跳过 2 个零值, result=0*5+2*6+0*7+4*8=12+32=44
    {
        sram_clear();
        // sparse vector with zeros
        sram_write_a(0x000, 0); sram_write_a(0x001, 2);
        sram_write_a(0x002, 0); sram_write_a(0x003, 4);
        // dense vector
        sram_write_a(0x100, 5); sram_write_a(0x101, 6);
        sram_write_a(0x102, 7); sram_write_a(0x103, 8);

        SparseCommand cmd;
        cmd.mode       = SparseMode::DYNAMIC;
        cmd.sparseAddr = 0x000;
        cmd.denseAddr  = 0x100;
        cmd.resultAddr = 0x200;
        cmd.maskAddr   = 0;
        cmd.vectorLen  = 4;

        bool ok = sparse_execute(cmd);
        NPU_CHECK(ok, "SparseEngine #39: dynamic zero skip execute");

        // 2*6+4*8 = 12+32 = 44 (零值跳过)
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)44,
                     "SparseEngine #39: dynamic zero skip result = 44");

        // 验证跳过计数 (2 个零值被跳过)
        NPU_CHECK_EQ(sparse_get_skip_count(), (uint32_t)2,
                     "SparseEngine #39: skip count = 2");
    }

    // 40. 全零稀疏向量: sparse=[0,0,0,0], dense=[1,2,3,4], result=0
    {
        sram_clear();
        // all-zero sparse vector
        sram_write_a(0x000, 0); sram_write_a(0x001, 0);
        sram_write_a(0x002, 0); sram_write_a(0x003, 0);
        // dense vector
        sram_write_a(0x100, 1); sram_write_a(0x101, 2);
        sram_write_a(0x102, 3); sram_write_a(0x103, 4);

        SparseCommand cmd;
        cmd.mode       = SparseMode::DYNAMIC;
        cmd.sparseAddr = 0x000;
        cmd.denseAddr  = 0x100;
        cmd.resultAddr = 0x200;
        cmd.maskAddr   = 0;
        cmd.vectorLen  = 4;

        bool ok = sparse_execute(cmd);
        NPU_CHECK(ok, "SparseEngine #40: all-zero sparse execute");

        // 全零: result = 0
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)0,
                     "SparseEngine #40: all-zero result = 0");
    }

    // 41. 无零值稀疏向量: sparse=[1,2,3,4], dense=[1,1,1,1], result=10
    {
        sram_clear();
        // no-zero sparse vector
        sram_write_a(0x000, 1); sram_write_a(0x001, 2);
        sram_write_a(0x002, 3); sram_write_a(0x003, 4);
        // dense vector (all ones)
        sram_write_a(0x100, 1); sram_write_a(0x101, 1);
        sram_write_a(0x102, 1); sram_write_a(0x103, 1);

        SparseCommand cmd;
        cmd.mode       = SparseMode::DYNAMIC;
        cmd.sparseAddr = 0x000;
        cmd.denseAddr  = 0x100;
        cmd.resultAddr = 0x200;
        cmd.maskAddr   = 0;
        cmd.vectorLen  = 4;

        bool ok = sparse_execute(cmd);
        NPU_CHECK(ok, "SparseEngine #41: no-zero sparse execute");

        // 1+2+3+4 = 10 (无跳过)
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)10,
                     "SparseEngine #41: no-zero result = 10");
    }

    // 42. 负值稀疏向量: sparse=[-1,0,-3,2], dense=[4,5,6,7]
    //     result=(-1)*4+0+(-3)*6+2*7=-4-18+14=-8
    {
        sram_clear();
        // sparse vector with negatives and zero
        sram_write_a(0x000, 0xFF); sram_write_a(0x001, 0);    // -1, 0
        sram_write_a(0x002, 0xFD); sram_write_a(0x003, 2);    // -3, 2
        // dense vector
        sram_write_a(0x100, 4); sram_write_a(0x101, 5);
        sram_write_a(0x102, 6); sram_write_a(0x103, 7);

        SparseCommand cmd;
        cmd.mode       = SparseMode::DYNAMIC;
        cmd.sparseAddr = 0x000;
        cmd.denseAddr  = 0x100;
        cmd.resultAddr = 0x200;
        cmd.maskAddr   = 0;
        cmd.vectorLen  = 4;

        bool ok = sparse_execute(cmd);
        NPU_CHECK(ok, "SparseEngine #42: negative sparse execute");

        // (-1)*4 + (-3)*6 + 2*7 = -4 - 18 + 14 = -8
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)-8,
                     "SparseEngine #42: negative result = -8");
    }
}

// ==============================================================================
// Phase 3.3: WinogradEngine Winograd 卷积加速测试
//
// Winograd F(2,3) 算法: 4x4 输入 tile + 3x3 权重 -> 2x2 输出 tile
// 等价于 stride=1, padding=0 的 3x3 卷积
// 数据布局:
//   input[i][j]  位于 inputAddr  + i*4 + j  (4x4 tile)
//   weight[i][j] 位于 weightAddr + i*3 + j  (3x3 kernel)
//   output[i][j] 位于 outputAddr + i*2 + j  (2x2 output)
// ==============================================================================

void test_winograd_engine() {
    printf("\n--- WinogradEngine Winograd 卷积加速测试 ---\n");

    // 43. Winograd 3x3 卷积 (简单):
    // 输入 4x4 tile (全1), 权重 3x3 (全1), 输出 2x2
    // 标准卷积: out[0][0] = 1+1+1+1+1+1+1+1+1 = 9
    //           out[0][1] = 9, out[1][0] = 9, out[1][1] = 9
    // Winograd 应得到相同结果
    {
        sram_clear();
        // 输入 4x4 tile (全1)
        for (int i = 0; i < 16; i++) sram_write_a(0x000 + i, 1);
        // 权重 3x3 (全1)
        for (int i = 0; i < 9; i++) sram_write_a(0x100 + i, 1);

        WinogradCommand cmd;
        cmd.inputAddr = 0x000;
        cmd.weightAddr = 0x100;
        cmd.outputAddr = 0x200;

        bool ok = winograd_execute(cmd);
        NPU_CHECK(ok, "Winograd #43: 3x3 conv execute");

        // 3x3 卷积全1: 每个输出 = 9
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)9, "Winograd #43: out[0][0]=9");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x201), (int32_t)9, "Winograd #43: out[0][1]=9");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x202), (int32_t)9, "Winograd #43: out[1][0]=9");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x203), (int32_t)9, "Winograd #43: out[1][1]=9");
    }

    // 44. Winograd 3x3 卷积 (不同值):
    // 输入 4x4 tile, 权重 3x3, 验证与标准卷积一致
    {
        sram_clear();
        // 输入 4x4 tile
        int8_t input[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                sram_write_a(0x000 + i*4 + j, (uint32_t)input[i][j]);

        // 权重 3x3
        int8_t weight[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                sram_write_a(0x100 + i*3 + j, (uint32_t)weight[i][j]);

        WinogradCommand cmd;
        cmd.inputAddr = 0x000;
        cmd.weightAddr = 0x100;
        cmd.outputAddr = 0x200;

        bool ok = winograd_execute(cmd);
        NPU_CHECK(ok, "Winograd #44: 3x3 conv execute");

        // 标准卷积: out[i][j] = sum(input[i+k][j+l] * weight[k][l])
        // out[0][0] = 1*1 + 6*1 + 11*1 = 18
        // out[0][1] = 2*1 + 7*1 + 12*1 = 21
        // out[1][0] = 5*1 + 10*1 + 15*1 = 30
        // out[1][1] = 6*1 + 11*1 + 16*1 = 33
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)18, "Winograd #44: out[0][0]=18");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x201), (int32_t)21, "Winograd #44: out[0][1]=21");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x202), (int32_t)30, "Winograd #44: out[1][0]=30");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x203), (int32_t)33, "Winograd #44: out[1][1]=33");
    }

    // 45. Winograd 3x3 卷积 (负值):
    {
        sram_clear();
        // 输入 4x4 tile (含负值)
        int8_t input[4][4] = {{-1,2,-3,4},{5,-6,7,-8},{9,10,-11,12},{-13,14,15,-16}};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                sram_write_a(0x000 + i*4 + j, (uint32_t)(int8_t)input[i][j]);

        // 权重 3x3 (全1)
        for (int i = 0; i < 9; i++) sram_write_a(0x100 + i, 1);

        WinogradCommand cmd;
        cmd.inputAddr = 0x000;
        cmd.weightAddr = 0x100;
        cmd.outputAddr = 0x200;

        bool ok = winograd_execute(cmd);
        NPU_CHECK(ok, "Winograd #45: 3x3 conv with negatives execute");

        // 标准卷积全1权重: out[i][j] = sum of 3x3 input tile
        // out[0][0] = -1+2-3+5-6+7+9+10-11 = 12
        // out[0][1] = 2-3+4-6+7-8+10-11+12 = 7
        // out[1][0] = 5-6+7+9+10-11-13+14+15 = 30
        // out[1][1] = -6+7-8+10-11+12+14+15-16 = 17
        NPU_CHECK_EQ((int32_t)sram_read_a(0x200), (int32_t)12, "Winograd #45: out[0][0]=12");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x201), (int32_t)7, "Winograd #45: out[0][1]=7");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x202), (int32_t)30, "Winograd #45: out[1][0]=30");
        NPU_CHECK_EQ((int32_t)sram_read_a(0x203), (int32_t)17, "Winograd #45: out[1][1]=17");
    }
}

// ==============================================================================
// Phase 3.4: DvfsController DVFS 动态调频调压测试
//
// 测试覆盖:
//   - P0/P1/P2/P3 四个 DVFS 级别切换
//   - 频率/电压/分频参数验证
//   - 过渡计数验证
// ==============================================================================

void test_dvfs_controller() {
    printf("\n--- DvfsController DVFS 测试 ---\n");

    // 46. DVFS 切换到 P0 Turbo
    {
        DvfsCommand cmd;
        cmd.targetLevel = DvfsLevel::P0_TURBO;
        bool ok = dvfs_request(cmd);
        NPU_CHECK(ok, "DvfsController #46: switch to P0 Turbo");
        NPU_CHECK_EQ(dvfs_get_current_freq(), (uint32_t)1000, "DvfsController #46: freq=1000MHz");
        NPU_CHECK_EQ(dvfs_get_current_voltage(), (uint8_t)100, "DvfsController #46: voltage=1.0V");
        NPU_CHECK_EQ(dvfs_get_current_divisor(), (uint8_t)1, "DvfsController #46: divisor=1");
    }

    // 47. DVFS 切换到 P2 Efficient
    {
        DvfsCommand cmd;
        cmd.targetLevel = DvfsLevel::P2_EFFICIENT;
        bool ok = dvfs_request(cmd);
        NPU_CHECK(ok, "DvfsController #47: switch to P2 Efficient");
        NPU_CHECK_EQ(dvfs_get_current_freq(), (uint32_t)250, "DvfsController #47: freq=250MHz");
        NPU_CHECK_EQ(dvfs_get_current_voltage(), (uint8_t)80, "DvfsController #47: voltage=0.8V");
        NPU_CHECK_EQ(dvfs_get_current_divisor(), (uint8_t)4, "DvfsController #47: divisor=4");
    }

    // 48. DVFS 切换到 P3 UltraLow
    {
        DvfsCommand cmd;
        cmd.targetLevel = DvfsLevel::P3_ULTRA_LOW;
        bool ok = dvfs_request(cmd);
        NPU_CHECK(ok, "DvfsController #48: switch to P3 UltraLow");
        NPU_CHECK_EQ(dvfs_get_current_freq(), (uint32_t)125, "DvfsController #48: freq=125MHz");
        NPU_CHECK_EQ(dvfs_get_current_voltage(), (uint8_t)70, "DvfsController #48: voltage=0.7V");
        NPU_CHECK_EQ(dvfs_get_current_divisor(), (uint8_t)8, "DvfsController #48: divisor=8");
    }

    // 49. DVFS 切换回 P1 Nominal
    {
        DvfsCommand cmd;
        cmd.targetLevel = DvfsLevel::P1_NOMINAL;
        bool ok = dvfs_request(cmd);
        NPU_CHECK(ok, "DvfsController #49: switch to P1 Nominal");
        NPU_CHECK_EQ(dvfs_get_current_freq(), (uint32_t)500, "DvfsController #49: freq=500MHz");
        NPU_CHECK_EQ(dvfs_get_current_voltage(), (uint8_t)90, "DvfsController #49: voltage=0.9V");
        NPU_CHECK_EQ(dvfs_get_current_divisor(), (uint8_t)2, "DvfsController #49: divisor=2");
    }

    // 50. DVFS 过渡计数验证 (应该有 4 次过渡)
    {
        NPU_CHECK_EQ(dvfs_get_transition_count(), (uint32_t)4, "DvfsController #50: transition count=4");
    }
}

// ==============================================================================
// Phase 3.5: PowerDomainController 电源域划分测试
//
// 测试覆盖:
//   - NPU Core / SRAM / IO 域上/下电控制
//   - ALWAYS_ON 域始终开启验证
//   - 上/下电过渡计数验证
// ==============================================================================

void test_power_domain() {
    printf("\n--- PowerDomainController 电源域测试 ---\n");

    // 51. 下电 NPU Core
    {
        PowerDomainCommand cmd;
        cmd.targetDomain = PowerDomain::PD_NPU_CORE;
        cmd.powerOn = false;
        bool ok = power_domain_request(cmd);
        NPU_CHECK(ok, "PowerDomain #51: power down NPU Core");
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_NPU_CORE), false, "PowerDomain #51: NPU Core is OFF");
        NPU_CHECK_EQ(power_domain_get_down_count(), (uint32_t)1, "PowerDomain #51: down count=1");
    }

    // 52. 上电 NPU Core
    {
        PowerDomainCommand cmd;
        cmd.targetDomain = PowerDomain::PD_NPU_CORE;
        cmd.powerOn = true;
        bool ok = power_domain_request(cmd);
        NPU_CHECK(ok, "PowerDomain #52: power up NPU Core");
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_NPU_CORE), true, "PowerDomain #52: NPU Core is ON");
        NPU_CHECK_EQ(power_domain_get_up_count(), (uint32_t)1, "PowerDomain #52: up count=1");
    }

    // 53. 下电 SRAM
    {
        PowerDomainCommand cmd;
        cmd.targetDomain = PowerDomain::PD_SRAM;
        cmd.powerOn = false;
        bool ok = power_domain_request(cmd);
        NPU_CHECK(ok, "PowerDomain #53: power down SRAM");
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_SRAM), false, "PowerDomain #53: SRAM is OFF");
    }

    // 54. 上电 SRAM
    {
        PowerDomainCommand cmd;
        cmd.targetDomain = PowerDomain::PD_SRAM;
        cmd.powerOn = true;
        bool ok = power_domain_request(cmd);
        NPU_CHECK(ok, "PowerDomain #54: power up SRAM");
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_SRAM), true, "PowerDomain #54: SRAM is ON");
    }

    // 55. 下电 IO
    {
        PowerDomainCommand cmd;
        cmd.targetDomain = PowerDomain::PD_IO;
        cmd.powerOn = false;
        bool ok = power_domain_request(cmd);
        NPU_CHECK(ok, "PowerDomain #55: power down IO");
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_IO), false, "PowerDomain #55: IO is OFF");
    }

    // 56. 上电 IO
    {
        PowerDomainCommand cmd;
        cmd.targetDomain = PowerDomain::PD_IO;
        cmd.powerOn = true;
        bool ok = power_domain_request(cmd);
        NPU_CHECK(ok, "PowerDomain #56: power up IO");
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_IO), true, "PowerDomain #56: IO is ON");
    }

    // 57. 验证 ALWAYS_ON 始终开启
    {
        NPU_CHECK_EQ(power_domain_is_on(PowerDomain::PD_ALWAYS_ON), true, "PowerDomain #57: ALWAYS_ON is ON");
    }

    // 58. 过渡计数验证
    {
        NPU_CHECK_EQ(power_domain_get_up_count(), (uint32_t)3, "PowerDomain #58: up count=3");
        NPU_CHECK_EQ(power_domain_get_down_count(), (uint32_t)3, "PowerDomain #58: down count=3");
    }
}

// ==============================================================================
// Phase 3.6: ThermalMonitor 温度监控与调度测试
//
// 测试覆盖:
//   - 正常/告警/严重/紧急 四个温度级别
//   - 告警中断信号验证
//   - 调度建议 (降频/关机) 验证
//   - 采样计数与告警计数验证
// ==============================================================================

void test_thermal_monitor() {
    printf("\n--- ThermalMonitor 温度监控测试 ---\n");

    // 59. 正常温度 (50°C)
    {
        bool ok = thermal_sample(50);
        NPU_CHECK(ok, "ThermalMonitor #59: sample 50C");
        NPU_CHECK_EQ(thermal_get_current_temp(), (uint8_t)50, "ThermalMonitor #59: temp=50");
        NPU_CHECK_EQ((int)thermal_get_alert_level(), (int)ThermalAlert::NORMAL, "ThermalMonitor #59: alert=NORMAL");
        NPU_CHECK_EQ(thermal_get_warn_intr(), false, "ThermalMonitor #59: no warn intr");
        NPU_CHECK_EQ(thermal_get_suggest_throttle(), false, "ThermalMonitor #59: no throttle");
    }

    // 60. 告警温度 (80°C)
    {
        bool ok = thermal_sample(80);
        NPU_CHECK(ok, "ThermalMonitor #60: sample 80C");
        NPU_CHECK_EQ(thermal_get_current_temp(), (uint8_t)80, "ThermalMonitor #60: temp=80");
        NPU_CHECK_EQ((int)thermal_get_alert_level(), (int)ThermalAlert::WARNING, "ThermalMonitor #60: alert=WARNING");
        NPU_CHECK_EQ(thermal_get_warn_intr(), true, "ThermalMonitor #60: warn intr");
        NPU_CHECK_EQ(thermal_get_suggest_throttle(), true, "ThermalMonitor #60: suggest throttle");
        NPU_CHECK_EQ(thermal_get_suggest_shutdown(), false, "ThermalMonitor #60: no shutdown");
    }

    // 61. 严重温度 (90°C)
    {
        bool ok = thermal_sample(90);
        NPU_CHECK(ok, "ThermalMonitor #61: sample 90C");
        NPU_CHECK_EQ(thermal_get_current_temp(), (uint8_t)90, "ThermalMonitor #61: temp=90");
        NPU_CHECK_EQ((int)thermal_get_alert_level(), (int)ThermalAlert::CRITICAL, "ThermalMonitor #61: alert=CRITICAL");
        NPU_CHECK_EQ(thermal_get_crit_intr(), true, "ThermalMonitor #61: crit intr");
        NPU_CHECK_EQ(thermal_get_suggest_throttle(), true, "ThermalMonitor #61: suggest throttle");
    }

    // 62. 紧急温度 (100°C)
    {
        bool ok = thermal_sample(100);
        NPU_CHECK(ok, "ThermalMonitor #62: sample 100C");
        NPU_CHECK_EQ(thermal_get_current_temp(), (uint8_t)100, "ThermalMonitor #62: temp=100");
        NPU_CHECK_EQ((int)thermal_get_alert_level(), (int)ThermalAlert::EMERGENCY, "ThermalMonitor #62: alert=EMERGENCY");
        NPU_CHECK_EQ(thermal_get_emerg_intr(), true, "ThermalMonitor #62: emerg intr");
        NPU_CHECK_EQ(thermal_get_suggest_shutdown(), true, "ThermalMonitor #62: suggest shutdown");
    }

    // 63. 恢复正常温度 (60°C)
    {
        bool ok = thermal_sample(60);
        NPU_CHECK(ok, "ThermalMonitor #63: sample 60C");
        NPU_CHECK_EQ((int)thermal_get_alert_level(), (int)ThermalAlert::NORMAL, "ThermalMonitor #63: alert=NORMAL");
        NPU_CHECK_EQ(thermal_get_warn_intr(), false, "ThermalMonitor #63: no warn intr");
    }

    // 64. 采样计数验证
    {
        NPU_CHECK_EQ(thermal_get_sample_count(), (uint32_t)5, "ThermalMonitor #64: sample count=5");
        NPU_CHECK_EQ(thermal_get_warn_count(), (uint32_t)1, "ThermalMonitor #64: warn count=1");
        NPU_CHECK_EQ(thermal_get_crit_count(), (uint32_t)1, "ThermalMonitor #64: crit count=1");
        NPU_CHECK_EQ(thermal_get_emerg_count(), (uint32_t)1, "ThermalMonitor #64: emerg count=1");
    }
}

// ==============================================================================
// Phase 3.7: ArrayScalingConfig 阵列扩展测试
//
// 测试覆盖:
//   - 32x32 / 16x16 / 4x4 / 1x1 / 8x8 五种阵列规模配置
//   - 行列数、MTOPS、GMACs 性能估算验证
//   - 配置计数验证
// ==============================================================================

void test_array_scaling() {
    printf("\n--- ArrayScalingConfig 阵列扩展测试 ---\n");

    // 65. 配置 32x32 阵列
    {
        ArrayConfigCommand cmd;
        cmd.targetSize = ArraySize::SIZE_32X32;
        bool ok = array_config_request(cmd);
        NPU_CHECK(ok, "ArrayScaling #65: config 32x32");
        NPU_CHECK_EQ(array_get_rows(), (uint32_t)32, "ArrayScaling #65: rows=32");
        NPU_CHECK_EQ(array_get_cols(), (uint32_t)32, "ArrayScaling #65: cols=32");
        NPU_CHECK_EQ(array_get_est_tops(), (uint32_t)1024, "ArrayScaling #65: MTOPS=1024");
        NPU_CHECK_EQ(array_get_est_gmacs(), (uint32_t)512, "ArrayScaling #65: GMACs=512");
    }

    // 66. 配置 16x16 阵列
    {
        ArrayConfigCommand cmd;
        cmd.targetSize = ArraySize::SIZE_16X16;
        bool ok = array_config_request(cmd);
        NPU_CHECK(ok, "ArrayScaling #66: config 16x16");
        NPU_CHECK_EQ(array_get_rows(), (uint32_t)16, "ArrayScaling #66: rows=16");
        NPU_CHECK_EQ(array_get_cols(), (uint32_t)16, "ArrayScaling #66: cols=16");
        NPU_CHECK_EQ(array_get_est_tops(), (uint32_t)256, "ArrayScaling #66: MTOPS=256");
    }

    // 67. 配置 4x4 阵列
    {
        ArrayConfigCommand cmd;
        cmd.targetSize = ArraySize::SIZE_4X4;
        bool ok = array_config_request(cmd);
        NPU_CHECK(ok, "ArrayScaling #67: config 4x4");
        NPU_CHECK_EQ(array_get_rows(), (uint32_t)4, "ArrayScaling #67: rows=4");
        NPU_CHECK_EQ(array_get_cols(), (uint32_t)4, "ArrayScaling #67: cols=4");
        NPU_CHECK_EQ(array_get_est_tops(), (uint32_t)16, "ArrayScaling #67: MTOPS=16");
    }

    // 68. 配置 1x1 阵列 (最小)
    {
        ArrayConfigCommand cmd;
        cmd.targetSize = ArraySize::SIZE_1X1;
        bool ok = array_config_request(cmd);
        NPU_CHECK(ok, "ArrayScaling #68: config 1x1");
        NPU_CHECK_EQ(array_get_rows(), (uint32_t)1, "ArrayScaling #68: rows=1");
        NPU_CHECK_EQ(array_get_cols(), (uint32_t)1, "ArrayScaling #68: cols=1");
    }

    // 69. 恢复 8x8 阵列 (默认)
    {
        ArrayConfigCommand cmd;
        cmd.targetSize = ArraySize::SIZE_8X8;
        bool ok = array_config_request(cmd);
        NPU_CHECK(ok, "ArrayScaling #69: config 8x8");
        NPU_CHECK_EQ(array_get_rows(), (uint32_t)8, "ArrayScaling #69: rows=8");
        NPU_CHECK_EQ(array_get_cols(), (uint32_t)8, "ArrayScaling #69: cols=8");
    }

    // 70. 配置计数验证
    {
        NPU_CHECK_EQ(array_get_config_count(), (uint32_t)5, "ArrayScaling #70: config count=5");
    }
}
