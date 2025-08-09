#include <stdio.h>
#include <cpuid.h>

int main() {
    unsigned int eax, ebx, ecx, edx;

    for (int i = 0; ; i++) {
        __cpuid_count(4, i, eax, ebx, ecx, edx);

        unsigned cache_type = eax & 0x1F; // 0 = no more caches
        if (cache_type == 0) break;

        unsigned cache_level = (eax >> 5) & 0x7; // L1=1, L2=2, L3=3...
        unsigned line_size   = (ebx & 0xFFF) + 1;
        unsigned partitions  = ((ebx >> 12) & 0x3FF) + 1;
        unsigned ways        = ((ebx >> 22) & 0x3FF) + 1;
        unsigned sets        = ecx + 1;
        unsigned cache_size  = ways * partitions * line_size * sets;

        const char* type_str =
            (cache_type == 1) ? "Data" :
            (cache_type == 2) ? "Instruction" :
            (cache_type == 3) ? "Unified" : "Unknown";

        printf("L%u cache (%s): %u KB, line size: %u bytes\n",
               cache_level, type_str, cache_size / 1024, line_size);
    }

    return 0;
}
