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

#ifndef XSIMD_ALGORITHMS_NONE_OF_HPP
#define XSIMD_ALGORITHMS_NONE_OF_HPP

#include "xsimd/xsimd.hpp"
#include "xsimd_algorithm/stl/any_of.hpp"

#include <iterator>

namespace xsimd
{

    template <class Arch = default_arch, class ContiguousIterator, class Predicate>
    bool none_of(ContiguousIterator first, ContiguousIterator last, Predicate&& p) noexcept
    {
        return !any_of<Arch>(first, last, std::forward<Predicate>(p));
    }

}

#endif // XSIMD_ALGORITHMS_NONE_OF_HPP