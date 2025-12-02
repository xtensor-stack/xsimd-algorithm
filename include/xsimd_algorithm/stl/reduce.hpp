/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHMS_REDUCE_HPP
#define XSIMD_ALGORITHMS_REDUCE_HPP

#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "xsimd/xsimd.hpp"

namespace xsimd
{
    // TODO: Remove this once we drop C++11 support
    namespace detail
    {
        struct plus
        {
            template <class X, class Y>
            auto operator()(X&& x, Y&& y) noexcept -> decltype(x + y) { return x + y; }
        };
    }

    template <class Arch = default_arch, class Iterator1, class Iterator2, class Init, class BinaryFunction = detail::plus>
    Init reduce(Iterator1 first, Iterator2 last, Init init, BinaryFunction&& binfun = detail::plus {}) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;

        std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        constexpr std::size_t simd_size = batch_type::size;

        if (size < simd_size)
        {
            while (first != last)
            {
                init = binfun(init, *first++);
            }
            return init;
        }

        const auto* const ptr_begin = &(*first);

        std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        // reduce initial unaligned part
        for (std::size_t i = 0; i < align_begin; ++i)
        {
            init = binfun(init, first[i]);
        }

        // reduce aligned part
        auto ptr = ptr_begin + align_begin;
        batch_type batch_init = batch_type::load_aligned(ptr);
        ptr += simd_size;
        for (auto const end = ptr_begin + align_end; ptr < end; ptr += simd_size)
        {
            batch_type batch = batch_type::load_aligned(ptr);
            batch_init = binfun(batch_init, batch);
        }

        // reduce across batch
        alignas(batch_type) std::array<value_type, simd_size> arr;
        xsimd::store_aligned(arr.data(), batch_init);
        for (auto x : arr)
            init = binfun(init, x);

        // reduce final unaligned part
        for (std::size_t i = align_end; i < size; ++i)
        {
            init = binfun(init, first[i]);
        }

        return init;
    }

}

#endif
