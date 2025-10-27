#pragma once
#include <ostream>
#include <cstdint>


namespace mx{

// Standard index type for BLAS/LAPACK compatibility
using index_t = std::int32_t;

enum class LUStatus: int {
    SUCCESS = 0,   // factorization completed (may still have tiny pivots; check info)
    BAD_ARG = -1,  // invalid shapes, nulls...
    SINGULAR = 1   // zero pivot encountered (info holds index)
};

struct LUInfo {
    LUStatus status = LUStatus::SUCCESS;
    index_t first_zero_pivot = -1; // -1 means none

    constexpr bool ok() const        { return status == LUStatus::SUCCESS; }
    constexpr bool singular() const  { return status == LUStatus::SINGULAR; }
};

inline const char* to_string(LUStatus s) {
    switch (s) {
        case LUStatus::SUCCESS:       return "ok";
        case LUStatus::BAD_ARG:  return "bad_arg";
        case LUStatus::SINGULAR: return "singular";
    }
    return "unknown";
}

inline std::ostream& operator<<(std::ostream& os, LUInfo result) {
    return os << "LUInfo{status=" << to_string(result.status)
                << ", first_zero_pivot=" << result.first_zero_pivot 
                << "}";
}

}