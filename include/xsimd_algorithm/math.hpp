/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_MATH_HPP
#define XSIMD_ALGORITHM_MATH_HPP

#include <xsimd/xsimd.hpp>

#include "./map.hpp"

namespace xsimd::algo
{
    template <
        xsimd::alignment_options align = xsimd::alignment_options{},
        typename Arch = xsimd::default_arch,
        typename T>
    void sqrt(std::span<T const> in, std::span<T> out)
    {
        constexpr map_options opts = { .unroll_factor = 4, .pure = true };
        return xsimd::map_unary<align, opts, Arch>(
            in, out, [](auto x)
            { return sqrt(x); });
    }

    template <
        xsimd::alignment_options align = xsimd::alignment_options{},
        typename Arch = xsimd::default_arch,
        typename T>
    void abs(std::span<T const> in, std::span<T> out)
    {
        constexpr map_options opts = { .unroll_factor = 4, .pure = true };
        return xsimd::map_unary<align, opts, Arch>(
            in, out, [](auto x)
            { return abs(x); });
    }

    template <
        xsimd::alignment_options align = xsimd::alignment_options{},
        typename Arch = xsimd::default_arch,
        typename T>
    void exp(std::span<T const> in, std::span<T> out)
    {
        constexpr map_options opts = { .unroll_factor = 4, .pure = true };
        return xsimd::map_unary<align, opts, Arch>(
            in, out, [](auto x)
            { return exp(x); });
    }

    template <
        xsimd::alignment_options align = xsimd::alignment_options{},
        typename Arch = xsimd::default_arch,
        typename T>
    void add(std::span<T const> lhs, std::span<T const> rhs, std::span<T> out)
    {
        constexpr map_options opts = { .unroll_factor = 4, .pure = true };
        return xsimd::map_binary<align, opts, Arch>(
            lhs, rhs, out, [](auto x, auto y)
            { return x + y; });
    }

    template <
        xsimd::alignment_options align = xsimd::alignment_options{},
        typename Arch = xsimd::default_arch,
        typename T>
    void multiply(std::span<T const> lhs, std::span<T const> rhs, std::span<T> out)
    {
        constexpr map_options opts = { .unroll_factor = 4, .pure = true };
        return xsimd::map_binary<align, opts, Arch>(
            lhs, rhs, out, [](auto x, auto y)
            { return x * y; });
    }
}

#endif
