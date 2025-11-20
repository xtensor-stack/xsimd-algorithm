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

#include "xsimd_algo/algorithms.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <numeric>

#if XSIMD_WITH_NEON && !XSIMD_WITH_NEON64
#define ALGORITHMS_TYPES float, std::complex<float>
#else
#define ALGORITHMS_TYPES float, double, std::complex<float>, std::complex<double>
#endif

template <typename Type>
struct transform_test
{
    using vector = std::vector<Type>;
    using aligned_vector = std::vector<Type, xsimd::aligned_allocator<Type>>;
    struct binary_functor
    {
        template <class T>
        T operator()(const T& a, const T& b) const
        {
            return a + b;
        }
    };

    struct unary_functor
    {
        template <class T>
        T operator()(const T& a) const
        {
            return -a;
        }
    };

    void test_binary_transform() const
    {
        vector expected(93);
        vector a(93, 123), b(93, 123), c(93);
        aligned_vector aa(93, 123), ba(93, 123), ca(93);

        std::transform(a.begin(), a.end(), b.begin(), expected.begin(),
                       binary_functor {});

        xsimd::transform(a.begin(), a.end(), b.begin(), c.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
        std::fill(c.begin(), c.end(), -1); // erase

        xsimd::transform(aa.begin(), aa.end(), ba.begin(), c.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
        std::fill(c.begin(), c.end(), -1); // erase

        xsimd::transform(aa.begin(), aa.end(), b.begin(), c.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
        std::fill(c.begin(), c.end(), -1); // erase

        xsimd::transform(a.begin(), a.end(), ba.begin(), c.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
        std::fill(c.begin(), c.end(), -1); // erase

        xsimd::transform(aa.begin(), aa.end(), ba.begin(), ca.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), ca.begin()));
        CHECK(expected.size() == ca.size());
        std::fill(ca.begin(), ca.end(), -1); // erase

        xsimd::transform(aa.begin(), aa.end(), b.begin(), ca.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), ca.begin()));
        CHECK(expected.size() == ca.size());
        std::fill(ca.begin(), ca.end(), -1); // erase

        xsimd::transform(a.begin(), a.end(), ba.begin(), ca.begin(),
                         binary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), ca.begin()));
        CHECK(expected.size() == ca.size());
        std::fill(ca.begin(), ca.end(), -1); // erase
    }

    void test_unary_transform() const
    {
        vector expected(93);
        vector a(93, 123), c(93);
        aligned_vector aa(93, 123), ca(93);

        std::transform(a.begin(), a.end(), expected.begin(),
                       unary_functor {});

        xsimd::transform(a.begin(), a.end(), c.begin(),
                         unary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
        std::fill(c.begin(), c.end(), -1); // erase

        xsimd::transform(aa.begin(), aa.end(), c.begin(),
                         unary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
        std::fill(c.begin(), c.end(), -1); // erase

        xsimd::transform(a.begin(), a.end(), ca.begin(),
                         unary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), ca.begin()));
        CHECK(expected.size() == ca.size());
        std::fill(ca.begin(), ca.end(), -1); // erase

        xsimd::transform(aa.begin(), aa.end(), ca.begin(),
                         unary_functor {});
        CHECK(std::equal(expected.begin(), expected.end(), ca.begin()));
        CHECK(expected.size() == ca.size());
        std::fill(ca.begin(), ca.end(), -1); // erase
    }
};

TEST_CASE_TEMPLATE("transform test", T, ALGORITHMS_TYPES)
{
    transform_test<T> Test;

    SUBCASE("unary") { Test.test_unary_transform(); }
    SUBCASE("binary") { Test.test_binary_transform(); }
}

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