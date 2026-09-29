/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_TEST_UTILS_MAP_BINARY_DATA_HPP
#define XSIMD_ALGORITHM_TEST_UTILS_MAP_BINARY_DATA_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "xsimd_algorithm/map.hpp"

#include "xsimd_test_utils/utils.hpp"

namespace xsimd::test
{
    using xsimd::alignment_options;
    using xsimd::map_options;

    /// Derives the scalar range application from the element-wise Derived::apply.
    template <typename Derived, typename Lhs, typename Rhs = Lhs, typename Out = Lhs>
    struct binary_op
    {
        using lhs_t = Lhs;
        using rhs_t = Rhs;
        using output_t = Out;

        template <typename Alloc, typename T>
        using rebind_allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;

        static void range_apply_scalar(std::span<lhs_t const> lhs, std::span<rhs_t const> rhs, std::span<output_t> out)
        {
            for (std::size_t i = 0; i < lhs.size(); ++i)
            {
                out[i] = Derived::apply_scalar(lhs[i], rhs[i]);
            }
        }

        template <
            alignment_options aligned = alignment_options {},
            map_options opts = map_options {},
            typename Arch = xsimd::default_arch>
        static void range_apply_map_binary(std::span<lhs_t const> lhs, std::span<rhs_t const> rhs, std::span<output_t> out)
        {
            return xsimd::map_binary<aligned, opts, Arch>(
                lhs, rhs, out, [](auto x, auto y)
                { return Derived::apply_batch(x, y); });
        }

        template <typename Alloc>
        static auto make_input_output(std::size_t size)
            -> std::tuple<
                std::vector<lhs_t, rebind_allocator<Alloc, lhs_t>>,
                std::vector<rhs_t, rebind_allocator<Alloc, rhs_t>>,
                std::vector<output_t, rebind_allocator<Alloc, output_t>>>
        {
            auto lhs = make_arange<lhs_t, rebind_allocator<Alloc, lhs_t>>(size, -static_cast<lhs_t>(size / 2));
            auto rhs = make_arange<rhs_t, rebind_allocator<Alloc, rhs_t>>(size, static_cast<rhs_t>(3));
            auto output = std::vector<output_t, rebind_allocator<Alloc, output_t>>(size);
            return { std::move(lhs), std::move(rhs), std::move(output) };
        }
    };

    /*******************
     *  Test fixtures  *
     *******************/

    template <typename T>
    struct add_op : binary_op<add_op<T>, T>
    {
        static constexpr auto name = "add";
        static constexpr bool pure = true;

        static auto apply_scalar(T x, T y) { return static_cast<T>(x + y); }

        template <typename A>
        static auto apply_batch(xsimd::batch<T, A> x, xsimd::batch<T, A> y) { return x + y; }
    };

    template <typename T>
    struct multiply_op : binary_op<multiply_op<T>, T>
    {
        static constexpr auto name = "multiply";
        static constexpr bool pure = true;

        static auto apply_scalar(T x, T y) { return static_cast<T>(x * y); }

        template <typename A>
        static auto apply_batch(xsimd::batch<T, A> x, xsimd::batch<T, A> y) { return x * y; }
    };

    /// Multiply an int32 by a double, one int32 batch with two double batches.
    struct mixed_multiply_op : binary_op<mixed_multiply_op, std::int32_t, double, double>
    {
        static constexpr auto name = "mixed_multiply";
        static constexpr bool pure = true;

        static auto apply_scalar(std::int32_t x, double y) { return static_cast<double>(x) * y; }

        template <typename A>
        static auto apply_batch(xsimd::batch<std::int32_t, A> x, std::array<xsimd::batch<double, A>, 2> const& y)
        {
            auto const wide = xsimd::widen(x);
            return std::array {
                xsimd::batch_cast<double>(wide[0]) * y[0],
                xsimd::batch_cast<double>(wide[1]) * y[1],
            };
        }
    };
}

#endif
