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

#include "xsimd_algorithm/stl/reduce.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <numeric>
#include <vector>

template <class T>
using test_allocator_type = xsimd::aligned_allocator<T>;

#if XSIMD_WITH_NEON && !XSIMD_WITH_NEON64
using test_value_type = float;
#else
using test_value_type = double;
#endif

struct multiply
{
    template <class T>
    T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

TEST_CASE("xsimd_reduce - unaligned_begin_unaligned_end")
{
    using aligned_vec_t = std::vector<test_value_type, test_allocator_type<test_value_type>>;
    constexpr std::size_t num_elements = 4 * xsimd::batch<test_value_type>::size;
    constexpr std::size_t small_num = xsimd::batch<test_value_type>::size - 1;

    aligned_vec_t vec(num_elements, 123.);
    aligned_vec_t small_vec(small_num, 42.);
    test_value_type init = 1337.;

    auto const begin = std::next(vec.begin());
    auto const end = std::prev(vec.end());

    CHECK_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if (small_vec.size() > 1)
    {
        auto const sbegin = std::next(small_vec.begin());
        auto const send = std::prev(small_vec.end());

        CHECK_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_CASE("xsimd_reduce - unaligned_begin_aligned_end")
{
    using aligned_vec_t = std::vector<test_value_type, test_allocator_type<test_value_type>>;
    constexpr std::size_t num_elements = 4 * xsimd::batch<test_value_type>::size;
    constexpr std::size_t small_num = xsimd::batch<test_value_type>::size - 1;

    aligned_vec_t vec(num_elements, 123.);
    aligned_vec_t small_vec(small_num, 42.);
    test_value_type init = 1337.;

    auto const begin = std::next(vec.begin());
    auto const end = vec.end();

    CHECK_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if (small_vec.size() > 1)
    {
        auto const sbegin = std::next(small_vec.begin());
        auto const send = small_vec.end();

        CHECK_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_CASE("xsimd_reduce - aligned_begin_unaligned_end")
{
    using aligned_vec_t = std::vector<test_value_type, test_allocator_type<test_value_type>>;
    constexpr std::size_t num_elements = 4 * xsimd::batch<test_value_type>::size;
    constexpr std::size_t small_num = xsimd::batch<test_value_type>::size - 1;

    aligned_vec_t vec(num_elements, 123.);
    aligned_vec_t small_vec(small_num, 42.);
    test_value_type init = 1337.;

    auto const begin = vec.begin();
    auto const end = std::prev(vec.end());

    CHECK_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if (small_vec.size() > 1)
    {
        auto const sbegin = small_vec.begin();
        auto const send = std::prev(small_vec.end());

        CHECK_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_CASE("xsimd_reduce - aligned_begin_aligned_end")
{
    using aligned_vec_t = std::vector<test_value_type, test_allocator_type<test_value_type>>;
    constexpr std::size_t num_elements = 4 * xsimd::batch<test_value_type>::size;
    constexpr std::size_t small_num = xsimd::batch<test_value_type>::size - 1;

    aligned_vec_t vec(num_elements, 123.);
    aligned_vec_t small_vec(small_num, 42.);
    test_value_type init = 1337.;

    auto const begin = vec.begin();
    auto const end = vec.end();

    CHECK_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));

    if (small_vec.size() > 1)
    {
        auto const sbegin = small_vec.begin();
        auto const send = small_vec.end();

        CHECK_EQ(std::accumulate(sbegin, send, init), xsimd::reduce(sbegin, send, init));
    }
}

TEST_CASE("xsimd_reduce - using_custom_binary_function")
{
    using aligned_vec_t = std::vector<test_value_type, test_allocator_type<test_value_type>>;
    constexpr std::size_t num_elements = 4 * xsimd::batch<test_value_type>::size;
    constexpr std::size_t small_num = xsimd::batch<test_value_type>::size - 1;

    aligned_vec_t vec(num_elements, 123.);
    aligned_vec_t small_vec(small_num, 42.);
    test_value_type init = 1337.;

    auto const begin = vec.begin();
    auto const end = vec.end();

    if (std::is_same<aligned_vec_t::value_type, double>::value)
    {
        CHECK(std::accumulate(begin, end, init, multiply {}) == doctest::Approx(xsimd::reduce(begin, end, init, multiply {})));
    }
    else
    {
        CHECK(std::accumulate(begin, end, init, multiply {}) == doctest::Approx(xsimd::reduce(begin, end, init, multiply {})));
    }

    if (small_vec.size() > 1)
    {
        auto const sbegin = small_vec.begin();
        auto const send = small_vec.end();

        if (std::is_same<aligned_vec_t::value_type, double>::value)
        {
            CHECK(std::accumulate(sbegin, send, init, multiply {}) == doctest::Approx(xsimd::reduce(sbegin, send, init, multiply {})));
        }
        else
        {
            CHECK(std::accumulate(sbegin, send, init, multiply {}) == doctest::Approx(xsimd::reduce(sbegin, send, init, multiply {})));
        }
    }
}

#endif
