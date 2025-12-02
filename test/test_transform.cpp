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

#include "xsimd_algorithm/stl/transform.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <vector>

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

#endif
