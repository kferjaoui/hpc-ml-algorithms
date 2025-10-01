#include <cstdio>
#include <cstdint>

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline void cpuid_count(uint32_t leaf, uint32_t subleaf,
                                 uint32_t &eax, uint32_t &ebx, uint32_t &ecx, uint32_t &edx) {
      int regs[4];
      __cpuidex(regs, (int)leaf, (int)subleaf);
      eax = (uint32_t)regs[0];
      ebx = (uint32_t)regs[1];
      ecx = (uint32_t)regs[2];
      edx = (uint32_t)regs[3];
  }
  static inline uint32_t cpuid_max(uint32_t leaf_base) {
      int regs[4];
      __cpuid(regs, (int)leaf_base);
      return (uint32_t)regs[0];
  }
#else
  #include <cpuid.h>
  static inline void cpuid_count(uint32_t leaf, uint32_t subleaf,
                                 uint32_t &eax, uint32_t &ebx, uint32_t &ecx, uint32_t &edx) {
      __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  }
  static inline uint32_t cpuid_max(uint32_t leaf_base) {
      return __get_cpuid_max(leaf_base, nullptr);
  }
#endif

static bool print_caches_for_leaf(uint32_t leaf) {
    bool printed = false;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t eax=0, ebx=0, ecx=0, edx=0;
        cpuid_count(leaf, i, eax, ebx, ecx, edx);

        uint32_t cache_type  =  eax        & 0x1F;   // 0 => no more caches
        if (!cache_type) break;

        uint32_t cache_level = (eax >> 5)  & 0x7;    // L1=1, L2=2, L3=3...
        uint32_t line_size   = (ebx        & 0xFFF) + 1;
        uint32_t partitions  = ((ebx >> 12) & 0x3FF) + 1;
        uint32_t ways        = ((ebx >> 22) & 0x3FF) + 1;
        uint32_t sets        = ecx + 1;

        if (!line_size || !partitions || !ways || !sets) break;

        uint32_t cache_size  = ways * partitions * line_size * sets;

        const char* type_str =
            (cache_type == 1) ? "Data" :
            (cache_type == 2) ? "Instruction" :
            (cache_type == 3) ? "Unified" : "Unknown";

        std::printf("Leaf 0x%08X: L%u %s cache: %u KB, line: %u bytes\n",
                    leaf, cache_level, type_str, cache_size / 1024, line_size);
        printed = true;
    }
    return printed;
}

int main() {
    uint32_t max_std = cpuid_max(0x0);
    uint32_t max_ext = cpuid_max(0x80000000u);

    std::printf("CPUID max std leaf: 0x%08X, max ext leaf: 0x%08X\n", max_std, max_ext);

    // Show raw subleaf 0 for both leaves for debugging (optional)
    {
        uint32_t eax=0, ebx=0, ecx=0, edx=0;
        cpuid_count(0x00000004u, 0, eax, ebx, ecx, edx);
        std::printf("Raw 0x00000004 subleaf 0: EAX=%08X EBX=%08X ECX=%08X EDX=%08X\n", eax, ebx, ecx, edx);
    }
    {
        uint32_t eax=0, ebx=0, ecx=0, edx=0;
        cpuid_count(0x8000001Du, 0, eax, ebx, ecx, edx);
        std::printf("Raw 0x8000001D subleaf 0: EAX=%08X EBX=%08X ECX=%08X EDX=%08X\n", eax, ebx, ecx, edx);
    }

    bool any = false;
    if (max_std >= 0x4)          any |= print_caches_for_leaf(0x4);           // Intel leaf
    if (max_ext >= 0x8000001Du)  any |= print_caches_for_leaf(0x8000001D);    // AMD leaf

    if (!any) {
        std::puts("No usable deterministic cache info via CPUID (masked or unsupported).");
    }
    return 0;
}
