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

#ifndef XSIMD_ALGORITHMS_MAX_ELEMENT_HPP
#define XSIMD_ALGORITHMS_MAX_ELEMENT_HPP

#include "xsimd/xsimd.hpp"

#include <iterator>

namespace xsimd
{
    namespace detail
    {
        template <class Arch = default_arch, class ForwardIterator, class Compare>
        ForwardIterator sequential_max_element(ForwardIterator first, ForwardIterator last, Compare&& comp) noexcept
        {
            if (first == last)
            {
                return last;
            }

            ForwardIterator max_it = first;

            for (++first; first != last; ++first)
            {
                if (comp(*max_it, *first))
                {
                    max_it = first;
                }
            }

            return max_it;
        }
    }

    template <class Arch = default_arch, class ContiguousIterator, class Compare>
    ContiguousIterator max_element(ContiguousIterator first, ContiguousIterator last, Compare&& comp) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;
        
        const std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        constexpr std::size_t simd_size = batch_type::size;

        // Quick return for small ranges
        if (size < simd_size)
        {
            return detail::sequential_max_element(first, last, std::forward<Compare>(comp));
        }

        // prologue: process elements until the first aligned one
        // compute alignment offsets
        const auto* const ptr_begin = &(*first);
        const std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        const std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        const auto align_begin_it = std::next(first, align_begin);
        const auto align_end_it = std::next(first, align_end);

        auto max_it = detail::sequential_max_element(first, align_begin_it, std::forward<Compare>(comp));
        auto current_max = *max_it;
        auto current_max_index = std::distance(first, max_it);

        // Generate a batch of the current max value
        batch_type current_max_batch = batch_type(current_max);

        // main loop: process aligned elements
        for (auto current = align_begin_it; current != align_end_it; std::advance(current, simd_size))
        {
            batch_type batch_val = batch_type::load_aligned(&(*current));

            // Compare the current batch with the current max batch
            auto mask = comp(current_max_batch, batch_val);

            // Update ?
            if (xsimd::any(mask))
            {
                // Update the current max batch
                current_max_batch = xsimd::select(mask, batch_val, current_max_batch);
                // extract the max value
                current_max = xsimd::reduce_max(current_max_batch);
                //retrieve position of the max value in the current batch
                auto max_pos = std::find(current, current + simd_size, current_max);
                if (max_pos != current + simd_size)
                {
                    current_max_index = std::distance(first, max_pos);
                }
            }
        }
    
        max_it = std::next(first, current_max_index);

        for (auto current = align_end_it; current != last; ++current)
        {
            if (comp(current_max, *current))
            {
                current_max = *current;
                max_it = current;
            }
        }
        return max_it;
    }

    template <class Arch = default_arch, class ContiguousIterator>
    ContiguousIterator max_element(ContiguousIterator first, ContiguousIterator last) noexcept
    {
        return xsimd::max_element<Arch>(first, last, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
    }

} // namespace xsimd

#endif // XSIMD_ALGORITHMS_MAX_ELEMENT_HPP