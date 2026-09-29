/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_TEST_UTILS_MATH_OPS_HPP
#define XSIMD_ALGORITHM_TEST_UTILS_MATH_OPS_HPP

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <xsimd_algorithm/map.hpp>
#include <xsimd_algorithm/math.hpp>

#include "xsimd_test_utils/utils.hpp"

namespace xsimd::test
{
    template <typename T>
    struct sqrt_op
    {
        using value_type = T;

        static constexpr auto name = "sqrt";

        template <xsimd::alignment_options aligned = xsimd::alignment_options{}>
        static void apply(std::span<T const> in, std::span<T> out)
        {
            xsimd::algo::sqrt<aligned>(in, out);
        }

        static T scalar(T x)
        {
            return std::sqrt(x);
        }

        template <typename Alloc>
        static std::vector<T, Alloc> input(std::size_t size)
        {
            return xsimd::test::make_arange<T, Alloc>(size);
        }
    };

    template <typename T>
    struct abs_op
    {
        using value_type = T;

        static constexpr auto name = "abs";

        template <xsimd::alignment_options aligned = xsimd::alignment_options{}>
        static void apply(std::span<T const> in, std::span<T> out)
        {
            xsimd::algo::abs<aligned>(in, out);
        }

        static T scalar(T x)
        {
            return std::abs(x);
        }

        template <typename Alloc>
        static std::vector<T, Alloc> input(std::size_t size)
        {
            return xsimd::test::make_arange<T, Alloc>(size, -static_cast<T>(size) / 2);
        }
    };
}

#endif
