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

#ifndef XSIMD_ALGORITHMS_FIND_HPP
#define XSIMD_ALGORITHMS_FIND_HPP

#include "xsimd/xsimd.hpp"
#include "xsimd_algorithm/stl/find_if.hpp"

#include <iostream>
#include <iterator>

namespace xsimd
{
    template <class Arch = default_arch, class ContiguousIterator, class T = typename std::iterator_traits<ContiguousIterator>::value_type>
    ContiguousIterator find(ContiguousIterator first, ContiguousIterator last, const T& value) noexcept
    {
        return find_if<Arch>(first, last, [value](const auto& x) { return x == value; });
    }
}
#endif // XSIMD_ALGORITHMS_FIND_HPP