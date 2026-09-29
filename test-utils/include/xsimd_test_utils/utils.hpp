/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_TEST_UTILS_UTILS_HPP
#define XSIMD_ALGORITHM_TEST_UTILS_UTILS_HPP

#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace xsimd::test
{
    template <typename T, typename A = xsimd::default_arch>
    using aligned_vector = std::vector<T, xsimd::aligned_allocator<T, A::alignment()>>;

    /// An allocator returning memory guaranteed not to be aligned on @p Align.
    ///
    /// Over-allocates by @p Offset elements and shifts the returned pointer, so that the data
    /// starts @p Offset * sizeof(T) bytes past an aligned address.
    template <typename T, std::size_t Align = xsimd::default_arch::alignment(), std::size_t Offset = 1>
    struct unaligned_allocator : private xsimd::aligned_allocator<T, Align>
    {
        static_assert((Offset * sizeof(T)) % Align != 0, "Shifted pointer would still be aligned");

        using base_type = xsimd::aligned_allocator<T, Align>;
        using value_type = T;

        // The non-type Align parameter defeats the default allocator_traits rebind.
        template <typename U>
        struct rebind
        {
            using other = unaligned_allocator<U, Align, Offset>;
        };

        unaligned_allocator() = default;

        template <typename U>
        unaligned_allocator(unaligned_allocator<U, Align, Offset> const&)
        {
        }

        T* allocate(std::size_t n) { return base_type::allocate(n + Offset) + Offset; }

        void deallocate(T* p, std::size_t n) { base_type::deallocate(p - Offset, n + Offset); }

        friend bool operator==(unaligned_allocator const&, unaligned_allocator const&) { return true; }
    };

    template <typename T, typename A = xsimd::default_arch>
    using unaligned_vector = std::vector<T, unaligned_allocator<T, A::alignment()>>;

    template <typename T, typename A>
    std::span<T const> as_span(std::vector<T, A> const& v)
    {
        return std::span<T const> { v.data(), v.size() };
    }

    template <typename T, typename A>
    std::span<T> as_span(std::vector<T, A>& v)
    {
        return std::span<T> { v.data(), v.size() };
    }

    template <typename T, typename Alloc = typename aligned_vector<T>::allocator_type>
    std::vector<T, Alloc> make_arange(std::size_t size, T start = T { 0 })
    {
        std::vector<T, Alloc> data(size);
        std::iota(data.begin(), data.end(), start);
        return data;
    }
}

#endif
