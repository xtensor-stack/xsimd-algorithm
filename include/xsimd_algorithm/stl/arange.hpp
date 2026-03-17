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

#ifndef XSIMD_ALGORITHMS_ARANGE_HPP
#define XSIMD_ALGORITHMS_ARANGE_HPP

#include "xsimd/xsimd.hpp"

#include <iterator>

namespace xsimd
{
    namespace detail
    {
        template <class Arch = default_arch, class ForwardIterator, class T>
        T sequential_arange(ForwardIterator first, ForwardIterator last, T value, T step) noexcept
        {
            for (; first != last; ++first, value += step)
            {
                *first = value;
            }

            return value;
        }
    }

    template <class Arch = default_arch, class ContiguousIterator, class T>
    void arange(ContiguousIterator first, ContiguousIterator last, T value, T step) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;

        const std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        constexpr std::size_t simd_size = batch_type::size;

        if (size < simd_size)
        {
            detail::sequential_arange(first, last, value, step);
            return;
        }

        const auto* const ptr_begin = &(*first);
        const std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        const std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        const auto align_begin_it = std::next(first, align_begin);
        const auto align_end_it = std::next(first, align_end);

        value = detail::sequential_arange(first, align_begin_it, value, step);

        alignas(batch_type::arch_type::alignment()) value_type init_tmp[simd_size];
        detail::sequential_arange(init_tmp, init_tmp + simd_size, value, step);
        batch_type batch_val = batch_type::load_aligned(init_tmp);

        const batch_type step_batch(static_cast<value_type>(simd_size));
        for (auto current = align_begin_it; current != align_end_it; std::advance(current, simd_size))
        {
            batch_val.store_aligned(&(*current));
            batch_val = batch_val + (step_batch * step);
        }

        value = *std::next(align_end_it, -1) + step;

        detail::sequential_arange(align_end_it, last, value, step);
    }

} // namespace xsimd

#endif // XSIMD_ALGORITHMS_ARANGE_HPP
