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

#ifndef XSIMD_ALGORITHMS_ANY_OF_HPP
#define XSIMD_ALGORITHMS_ANY_OF_HPP

#include "xsimd/xsimd.hpp"

#include <iterator>

namespace xsimd
{
    namespace detail
    {
        template <class Arch = default_arch, class ForwardIterator, class Predicate>
        bool sequential_any_of(ForwardIterator first, ForwardIterator last, Predicate&& p) noexcept
        {
            for (; first != last; ++first)
            {
                if (p(*first))
                {
                    return true;
                }
            }
            return false;
        }
    }

    template <class Arch = default_arch, class ContiguousIterator, class Predicate>
    bool any_of(ContiguousIterator first, ContiguousIterator last, Predicate&& p) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;

        const std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        constexpr std::size_t simd_size = batch_type::size;

        // quick exit for small sizes
        if (size < simd_size)
        {
            return detail::sequential_any_of(first, last, std::forward<Predicate>(p));
        }

        // prologue: process elements until the first aligned one
        // compute alignment offsets
        const auto* const ptr_begin = &(*first);
        const std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        const std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        const auto align_begin_it = std::next(first, align_begin);
        const auto align_end_it = std::next(first, align_end);

        const auto prologue_result = detail::sequential_any_of(first, align_begin_it, std::forward<Predicate>(p));
        if (prologue_result)
        {
            return true;
        }

        // main loop: process aligned elements
        for (auto current = align_begin_it; current != align_end_it; std::advance(current, simd_size))
        {
            batch_type batch_val = batch_type::load_aligned(&(*current));
            if (xsimd::any(p(batch_val)))
            {
                return true;
            }
        }
        // epilogue: process remaining elements
        const auto epilogue_result = detail::sequential_any_of(align_end_it, last, std::forward<Predicate>(p));
        if (epilogue_result)
        {
            return true;
        }

        return false;
    }

}

#endif // XSIMD_ALGORITHMS_ANY_OF_HPP