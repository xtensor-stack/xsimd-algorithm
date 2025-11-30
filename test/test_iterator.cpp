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

#include "xsimd_algorithm/algorithms.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

template <class T>
using test_allocator_type = xsimd::aligned_allocator<T>;

#if XSIMD_X86_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE || XSIMD_ARM_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE
TEST_CASE("algorithms - iterator")
{
    std::vector<float, test_allocator_type<float>> a(10 * 16, 0.2), b(1000, 2.), c(1000, 3.);

    std::iota(a.begin(), a.end(), 0.f);
    std::vector<float> a_cpy(a.begin(), a.end());

    using batch_type = xsimd::batch<float>;
    auto begin = xsimd::aligned_iterator<batch_type>(&a[0]);
    auto end = xsimd::aligned_iterator<batch_type>(&a[0] + a.size());

    for (; begin != end; ++begin)
    {
        *begin = *begin / 2.f;
    }

    for (auto& el : a_cpy)
    {
        el /= 2.f;
    }

    CHECK(a.size() == a_cpy.size());
    CHECK(std::equal(a.begin(), a.end(), a_cpy.begin()));

    begin = xsimd::aligned_iterator<batch_type>(&a[0]);
    *begin = sin(*begin);

    for (std::size_t i = 0; i < batch_type::size; ++i)
    {
        CHECK(a[i] == doctest::Approx(sinf(a_cpy[i])).epsilon(1e-6));
    }

#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
    std::vector<std::complex<double>, test_allocator_type<std::complex<double>>> ca(10 * 16, std::complex<double>(0.2));
    using cbatch_type = xsimd::batch<std::complex<double>>;
    auto cbegin = xsimd::aligned_iterator<cbatch_type>(&ca[0]);
    auto cend = xsimd::aligned_iterator<cbatch_type>(&ca[0] + a.size());

    for (; cbegin != cend; ++cbegin)
    {
        *cbegin = (*cbegin + std::complex<double>(0, .3)) / 2.;
    }

    cbegin = xsimd::aligned_iterator<cbatch_type>(&ca[0]);
    *cbegin = sin(*cbegin);
    *cbegin = sqrt(*cbegin);
    auto real_part = abs(*(cbegin));
    (void)real_part;
#endif
}
#endif
#endif
