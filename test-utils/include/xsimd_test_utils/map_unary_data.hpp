/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_TEST_UTILS_MATH_DATA_HPP
#define XSIMD_ALGORITHM_TEST_UTILS_MATH_DATA_HPP

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "xsimd_algorithm/map.hpp"
#include "xsimd_algorithm/stl/transform.hpp"

#include "xsimd_test_utils/utils.hpp"

namespace xsimd::test
{
    using xsimd::alignment_options;
    using xsimd::map_options;

    /// Derives the scalar range application from the element-wise Derived::apply.
    template <typename Derived, typename In, typename Out = In>
    struct unary_op
    {
        using input_t = In;
        using output_t = Out;

        /// Allocator of output_t matching an allocator of input_t.
        template <typename Alloc>
        using output_allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<output_t>;

        static void range_apply_scalar(std::span<input_t const> in, std::span<output_t> out)
        {
            for (std::size_t i = 0; i < in.size(); ++i)
            {
                out[i] = Derived::apply_scalar(in[i]);
            }
        }

        template <
            alignment_options aligned = alignment_options {},
            map_options opts = map_options {},
            typename Arch = xsimd::default_arch>
        static void range_apply_map_unary(std::span<input_t const> in, std::span<output_t> out)
        {
            return xsimd::map_unary<aligned, opts, Arch>(
                in, out, [](auto x)
                { return Derived::apply_batch(x); });
        }

        template <typename Arch = xsimd::default_arch>
        inline static void range_apply_transform(std::span<input_t const> in, std::span<output_t> out)
        {
            return xsimd::transform<Arch>(
                in.data(), in.data() + in.size(), out.data(),
                []<typename T>(T x)
                {
                    if constexpr (xsimd::is_batch<T>::value)
                    {
                        return Derived::apply_batch(x);
                    }
                    else
                    {
                        return Derived::apply_scalar(x);
                    }
                });
        }

        template <typename Alloc>
        static auto make_input_output(std::size_t size)
            -> std::pair<std::vector<input_t, Alloc>, std::vector<output_t, output_allocator<Alloc>>>
        {
            auto input = Derived::template make_input<Alloc>(size);
            auto output = std::vector<output_t, output_allocator<Alloc>>(input.size());
            return { std::move(input), std::move(output) };
        }
    };

    /*******************
     *  Test fixtures  *
     *******************/

    template <typename T>
    struct sqrt_op : unary_op<sqrt_op<T>, T>
    {
        static constexpr auto name = "sqrt";
        static constexpr bool pure = true;

        static auto apply_scalar(T x) { return std::sqrt(x); }

        template <typename A>
        static auto apply_batch(xsimd::batch<T, A> x) { return xsimd::sqrt(x); }

        template <typename Alloc>
        static std::vector<T, Alloc> make_input(std::size_t size)
        {
            return make_arange<T, Alloc>(size);
        }
    };

    template <typename T>
    struct abs_op : unary_op<abs_op<T>, T>
    {
        static constexpr auto name = "abs";
        static constexpr bool pure = true;

        static auto apply_scalar(T x) { return std::abs(x); }

        template <typename A>
        static auto apply_batch(xsimd::batch<T, A> x) { return xsimd::abs(x); }

        template <typename Alloc>
        static std::vector<T, Alloc> make_input(std::size_t size)
        {
            return make_arange<T, Alloc>(size, -static_cast<T>(size) / 2);
        }
    };

    template <typename T>
    struct exp_op : unary_op<exp_op<T>, T>
    {
        static constexpr auto name = "exp";
        static constexpr bool pure = true;

        static auto apply_scalar(T x) { return std::exp(x); }

        template <typename A>
        static auto apply_batch(xsimd::batch<T, A> x) { return xsimd::exp(x); }

        template <typename Alloc>
        static std::vector<T, Alloc> make_input(std::size_t size)
        {
            // exp overflows past a small range, so wrap the values back into [-10, 10).
            auto input = make_arange<T, Alloc>(size);
            for (auto& x : input)
            {
                x = std::fmod(x, T { 20 }) - T { 10 };
            }
            return input;
        }
    };

    /// Sign-extend to the type with twice as many bytes, one input batch to two output batches.
    template <typename T>
    struct widen_op : unary_op<widen_op<T>, T, xsimd::widen_t<T>>
    {
        static constexpr auto name = "widen";
        static constexpr bool pure = true;

        static auto apply_scalar(T x) { return static_cast<xsimd::widen_t<T>>(x); }

        template <typename A>
        static auto apply_batch(xsimd::batch<T, A> x) { return xsimd::widen(x); }

        template <typename Alloc>
        static std::vector<T, Alloc> make_input(std::size_t size)
        {
            return make_arange<T, Alloc>(size, -static_cast<T>(size / 2));
        }
    };
}

#endif
