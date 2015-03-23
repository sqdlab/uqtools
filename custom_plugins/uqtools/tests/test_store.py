from pytest import fixture, raises, mark

import pandas as pd
import numpy as np

from uqtools.store import index_concat


def mkindex(dim):
    if dim == '0d':
        return pd.Int64Index([0])
    elif dim == '1d':
        return pd.Float64Index([1., 2., 3.])
    elif dim == '2d':
        return pd.MultiIndex(levels = ([0, 1], ['A', 'B']),
                             labels = ([1, 0, 1], [0, 1, 1]),
                             names = ['#', '@'])

@mark.parametrize('n1, n2',
                  [('0d', '1d'), ('1d', '0d'), ('0d', '2d'), ('2d', '0d'),
                   ('1d', '1d'), ('1d', '2d'), ('2d', '1d'), ('2d', '2d')])
def test_index_concat(n1, n2):
    ''' concatenate indices of zero to two dimensions '''
    index1 = mkindex(n1)
    index2 = mkindex(n2)
    concat = index_concat(index1, index2)

    if n1 == '0d':
        index1, index2 = index2, index1
    if (n2 == '0d') or (n1 == '0d'):
        assert concat.nlevels == index1.nlevels
        for level in range(index1.nlevels):
            assert np.all(concat.get_level_values(level) ==
                          index1.get_level_values(level))
        assert concat.names == index1.names
    else:
        assert concat.nlevels == index1.nlevels + index2.nlevels
        for index, offset in [(index1, 0), (index2, index1.nlevels)]:
            for level in range(index.nlevels):
                assert np.all(concat.get_level_values(offset + level) ==
                              index.get_level_values(level))
        assert concat.names == index1.names + index2.names
