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

#ifndef XSIMD_ALGORITHMS_MIN_ELEMENT_HPP
#define XSIMD_ALGORITHMS_MIN_ELEMENT_HPP

#include "xsimd/xsimd.hpp"

#include <iterator>

namespace xsimd
{
    namespace detail
    {
        template <class Arch = default_arch, class ForwardIterator, class Compare>
        ForwardIterator sequential_min_element(ForwardIterator first, ForwardIterator last, Compare&& comp) noexcept
        {
            if (first == last)
            {
                return last;
            }

            ForwardIterator min_it = first;

            for (++first; first != last; ++first)
            {
                if (comp(*min_it, *first))
                {
                    min_it = first;
                }
            }

            return min_it;
        }
    }

    template <class Arch = default_arch, class ContiguousIterator, class Compare>
    ContiguousIterator min_element(ContiguousIterator first, ContiguousIterator last, Compare&& comp) noexcept
    {
        using value_type = typename std::decay<decltype(*first)>::type;
        using batch_type = batch<value_type, Arch>;
        
        const std::size_t size = static_cast<std::size_t>(std::distance(first, last));
        constexpr std::size_t simd_size = batch_type::size;

        // Quick return for small ranges
        if (size < simd_size)
        {
            return detail::sequential_min_element(first, last, std::forward<Compare>(comp));
        }

        // prologue: process elements until the first aligned one
        // compute alignment offsets
        const auto* const ptr_begin = &(*first);
        const std::size_t align_begin = xsimd::get_alignment_offset(ptr_begin, size, simd_size);
        const std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        const auto align_begin_it = std::next(first, align_begin);
        const auto align_end_it = std::next(first, align_end);

        auto min_it = detail::sequential_min_element(first, align_begin_it, std::forward<Compare>(comp));
        auto current_min = *min_it;
        auto current_min_index = std::distance(first, min_it);

        // Generate a batch of the current min value
        batch_type current_min_batch = batch_type(current_min);

        // main loop: process aligned elements
        for (auto current = align_begin_it; current != align_end_it; std::advance(current, simd_size))
        {
            batch_type batch_val = batch_type::load_aligned(&(*current));

            // Compare the current batch with the current min batch
            auto mask = comp(current_min_batch, batch_val);

            // Update ?
            if (xsimd::any(mask))
            {
                // Update the current min batch
                current_min_batch = xsimd::select(mask, batch_val, current_min_batch);
                // extract the min value
                current_min = xsimd::reduce_min(current_min_batch);
                //retrieve position of the min value in the current batch
                auto min_pos = std::find(current, current + simd_size, current_min);
                if (min_pos != current + simd_size)
                {
                    current_min_index = std::distance(first, min_pos);
                }
            }
        }
    
        min_it = std::next(first, current_min_index);

        for (auto current = align_end_it; current != last; ++current)
        {
            if (comp(current_min, *current))
            {
                current_min = *current;
                min_it = current;
            }
        }
        return min_it;
    }

    template <class Arch = default_arch, class ContiguousIterator>
    ContiguousIterator min_element(ContiguousIterator first, ContiguousIterator last) noexcept
    {
        return xsimd::min_element<Arch>(first, last, [](const auto& lhs, const auto& rhs) { return lhs > rhs; });
    }

} // namespace xsimd

#endif // XSIMD_ALGORITHMS_MIN_ELEMENT_HPP