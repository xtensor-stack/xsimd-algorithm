/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_BENCHMARK_BENCH_UTILS_HPP
#define XSIMD_ALGORITHM_BENCHMARK_BENCH_UTILS_HPP

#include <cstdint>
#include <type_traits>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace xsimd::bench
{
    template <typename T>
    std::vector<std::int64_t> bench_sizes()
    {
        constexpr auto batch_size = static_cast<std::int64_t>(xsimd::batch<T>::size);

        return std::vector<std::int64_t> {
            64, 67, 1024, 1027, 65536, 2097152
        };
    }

    template <typename T>
    constexpr auto type_name()
    {
        // Avoid failing compiler check such as std::is_same<std::uint64_t, long long> due
        // to it not being an alias.
        constexpr bool is_int = std::is_integral_v<T> && !std::is_same_v<T, bool>;
        constexpr bool is_sint = is_int && std::is_signed_v<T>;
        constexpr bool is_uint = is_int && std::is_unsigned_v<T>;

        if constexpr (std::is_same_v<T, bool>)
        {
            return "bool";
        }
        else if constexpr (is_sint && sizeof(T) == 1)
        {
            return "i8";
        }
        else if constexpr (is_uint && sizeof(T) == 1)
        {
            return "u8";
        }
        else if constexpr (is_sint && sizeof(T) == 2)
        {
            return "i16";
        }
        else if constexpr (is_uint && sizeof(T) == 2)
        {
            return "u16";
        }
        else if constexpr (is_sint && sizeof(T) == 4)
        {
            return "i32";
        }
        else if constexpr (is_uint && sizeof(T) == 4)
        {
            return "u32";
        }
        else if constexpr (is_sint && sizeof(T) == 8)
        {
            return "i64";
        }
        else if constexpr (is_uint && sizeof(T) == 8)
        {
            return "u64";
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return "f32";
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return "f64";
        }
    }
}

#endif
