import numpy as np
import pandas as pd
import torch
from torch import Tensor

from grelu.utils import get_aggfunc, get_compare_func, get_transform_func, make_list

arr = np.expand_dims(np.array([[0.0, 5.0, 3.0], [4.0, 2.0, 6.0]]), 0)  # 1, 2, 3
tens = Tensor(arr)


def test_get_aggfunc():
    # Array input

    # max, array
    out = get_aggfunc("max")(arr, axis=1)
    assert out.shape == (1, 3)
    assert np.allclose(out.squeeze(), [4, 5, 6])

    # min, array
    out = get_aggfunc("min")(arr, axis=2, keepdims=True)
    assert out.shape == (1, 2, 1)
    assert np.allclose(out.squeeze(), [0, 2])

    # mean, array
    out = get_aggfunc("mean")(arr)
    assert np.allclose(out, 20 / 6)

    # sum, array
    out = get_aggfunc("sum")(arr, axis=-1, keepdims=True)
    assert out.shape == (1, 2, 1)
    assert np.allclose(out.squeeze(), [8, 12])

    # None
    assert get_aggfunc(None) is None

    # custom function
    out = get_aggfunc(np.sum)(arr, axis=-1, keepdims=True)
    assert out.shape == (1, 2, 1)
    assert np.allclose(out.squeeze(), [8, 12])

    # Tensor input

    # max, tensor
    out = get_aggfunc("max", tensor=True)(tens, axis=1)
    assert out.shape == (1, 3)
    assert np.allclose(out.squeeze().numpy(), [4, 5, 6])

    # min, tensor
    out = get_aggfunc("min", tensor=True)(tens, axis=2, keepdims=True)
    assert out.shape == (1, 2, 1)
    assert np.allclose(out.squeeze().numpy(), [0, 2])

    # mean, tensor
    out = get_aggfunc("mean", tensor=True)(tens)
    assert np.allclose(out.numpy(), 20 / 6)

    # sum, tensor
    out = get_aggfunc("sum", tensor=True)(tens, axis=-1, keepdims=True)
    assert out.shape == (1, 2, 1)
    assert np.allclose(out.squeeze().numpy(), [8, 12])


def test_get_compare_func():
    # Array input

    # subtract, array
    out = get_compare_func("subtract")(arr, arr)
    assert out.shape == (1, 2, 3)
    assert np.unique(out).squeeze() == 0

    # divide, array
    out = get_compare_func("divide")(arr, arr)
    assert out.shape == (1, 2, 3)
    assert np.unique(out[:, 1:]).squeeze() == 1

    # custom function
    out = get_compare_func(np.subtract)(arr, arr)
    assert out.shape == (1, 2, 3)
    assert np.unique(out).squeeze() == 0

    # Tensor input

    # subtract, tensor
    out = get_compare_func("subtract", tensor=True)(tens, tens)
    assert out.shape == (1, 2, 3)
    assert torch.unique(out).squeeze() == 0
    # divide, tensor
    out = get_compare_func("divide", tensor=True)(tens, tens)
    assert out.shape == (1, 2, 3)
    assert torch.unique(out[:, 1:]).squeeze() == 1


def test_get_transform_func():
    # Array input

    # log, array
    out = get_transform_func("log")(arr)
    assert out.shape == (1, 2, 3)
    assert np.allclose(out, np.log(arr))

    # log1p, array
    out = get_transform_func("log1p")(arr)
    assert out.shape == (1, 2, 3)
    assert np.allclose(out, np.log(arr + 1))

    # custom function
    out = get_transform_func(np.log)(arr)
    assert out.shape == (1, 2, 3)
    assert np.allclose(out, np.log(arr))

    # None
    assert get_transform_func(None) is None

    # Tensor input

    # log, tensor
    out = get_transform_func("log", tensor=True)(tens)
    assert out.shape == (1, 2, 3)
    assert np.allclose(out.numpy(), np.log(arr))
    # log1p, tensor
    out = get_transform_func("log1p", tensor=True)(tens)
    assert out.shape == (1, 2, 3)
    assert np.allclose(out.numpy(), np.log(arr + 1))


def test_make_list():
    # string
    assert make_list("AA") == ["AA"]

    # int
    assert make_list(3) == [3]

    # float
    assert make_list(3.1) == [3.1]

    # list string
    assert make_list(["AA"]) == ["AA"]

    # list int
    assert make_list([3]) == [3]

    # numpy
    assert make_list(np.array([1, 2])) == [1, 2]

    # pandas
    assert make_list(pd.Series([1, 2])) == [1, 2]

    # tensor
    assert make_list(Tensor([1, 2])) == [1.0, 2.0]

    # matrix
    assert make_list(np.matrix([1, 2])) == [1, 2]
