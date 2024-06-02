from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

TENSOR_TYPES = (float, int, list, tuple, torch.Tensor, np.ndarray)
VALUE_TYPES = (str, bool)


class TensorProperties(nn.Module):
    """
    A mix-in class for storing tensors as properties with helper methods.
    """

    def __init__(self, __shared_keys__=(), **kwargs):
        """
        Args:
            dtype: data type to set for the inputs
            device: str or torch.device
            kwargs: any number of keyword arguments. Any arguments which are
                of type (float/int/tuple/tensor/array) are broadcasted and
                other keyword arguments are set as attributes.
        """
        super().__init__()
        self._N = 0
        assert isinstance(__shared_keys__, (list, tuple)), "'__shared_keys__' has wrong type: {}".format(
            type(__shared_keys__)
        )
        if kwargs is not None:
            for k, v in kwargs.items():
                # format into tensor
                if v is not None and isinstance(v, TENSOR_TYPES):
                    v = format_tensor(v)
                # set the prop
                setattr(self, k, v)
                # check batch size
                if (v is not None) and (isinstance(v, TENSOR_TYPES)) and (k not in __shared_keys__):
                    bsz = v.shape[0]
                    if self._N == 0:
                        self._N = bsz
                    assert self._N == bsz, f"'{k}' has different batch size {bsz} with others {self._N}"

    def __len__(self) -> int:
        return self._N

    def isempty(self) -> bool:
        return self._N == 0

    @property
    def batch_size(self) -> int:
        return self._N

    def _apply(self, fn):
        is_fn_move_device = str(fn).find("cuda") >= 0 or str(fn).find("cpu") >= 0 or str(fn).find("to") >= 0
        if is_fn_move_device:
            for k in dir(self):
                v = getattr(self, k)
                if torch.is_tensor(v):
                    v = fn(v)
                    self.device = v.device
                    setattr(self, k, v)
        # super
        return super()._apply(fn)

    def __getitem__(self, index: Union[int, slice]):
        """

        Args:
            index: an int or slice used to index all the fields.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original camera.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(class_object=self, index=(index,))

        msg = "Expected index of type int or slice; got %r"
        raise ValueError(msg % type(index))

    def extend(self, n: int):
        if self._N != 1:
            raise ValueError("You can only extend tensor with batch_size 1, however {} is found".format(self._N))
        if isinstance(n, int):
            return TensorAccessor(class_object=self, index=(), extend=n)

        msg = "Expected extend int; got %r"
        raise ValueError(msg % type(n))

    def __getattr__(self, key) -> torch.Tensor:
        ret: torch.Tensor = super().__getattr__(key)  # type: ignore
        return ret

    def clone(self):
        raise NotImplementedError()

    def update_attr(self, attr, **kwargs):
        new_val = kwargs.get(attr)
        if new_val is None:
            new_val = getattr(self, attr)
        setattr(self, attr, new_val)


def format_tensor(input) -> torch.Tensor:
    """
    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if input.dim() == 0:
        input = input.view(1)
    return input


class TensorAccessor(nn.Module):
    """
    A helper class to be used with the __getitem__ method. This can be used for
    getting/setting the values for an attribute of a class at one particular
    index.  This is useful when the attributes of a class are batched tensors
    and one element in the batch needs to be modified.
    """

    def __init__(self, class_object, index: Tuple[Union[int, slice], ...], extend=None):
        """
        Args:
            class_object: this should be an instance of a class which has
                attributes which are tensors representing a batch of
                values.
            index: int/slice, an index indicating the position in the batch.
                In __setattr__ and __getattr__ only the value of class
                attributes at this index will be accessed.
        """
        self.__dict__["class_object"] = class_object
        self.__dict__["index"] = index
        self.__dict__["extend"] = extend

    def __setattr__(self, name: str, value: Any):
        """
        Update the attribute given by `name` to the value given by `value`
        at the index specified by `self.index`.

        Args:
            name: str, name of the attribute.
            value: value to set the attribute to.
        """
        v = getattr(self.class_object, name)
        if not torch.is_tensor(v):
            msg = "Can only set values on attributes which are tensors; got %r"
            raise AttributeError(msg % type(v))

        # Convert the attribute to a tensor if it is not a tensor.
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=v.device, dtype=v.dtype, requires_grad=v.requires_grad)

        # Check the shapes match the existing shape and the shape of the index.
        if v.dim() > 1 and value.dim() > 1 and value.shape[1:] != v.shape[1:]:
            msg = "Expected value to have shape %r; got %r"
            raise ValueError(msg % (v.shape, value.shape))
        # if (
        #     v.dim() == 0
        #     and isinstance(self.index, tuple)
        #     and len(value) != len(self.index)
        # ):
        #     msg = "Expected value to have len %r; got %r"
        #     raise ValueError(msg % (len(self.index), len(value)))

        # accessor = getattr(self.class_object, name)
        accessor = self.class_object.__dict__[name]
        if self.extend is not None:
            accessor = accessor.expand(self.extend, *[-1 * (accessor.ndim - 1)])
        if len(self.index) == 0:
            accessor = value
        else:
            for i, idx in enumerate(self.index):
                if i + 1 == len(self.index):
                    accessor[idx] = value
                else:
                    accessor = accessor[idx]

    def __getattr__(self, name: str):
        """
        Return the value of the attribute given by "name" on self.class_object
        at the index specified in self.index.

        Args:
            name: string of the attribute name
        """
        if hasattr(self.class_object, name):
            # ret = getattr(self.class_object, name)
            ret = self.class_object.__dict__[name]
            if self.extend is not None:
                ret = ret.expand(self.extend, *[-1 for _ in range(ret.ndim - 1)])
            for idx in self.index:
                ret = ret[idx]
            return ret
        else:
            msg = "Attribute %s not found on %r"
            return AttributeError(msg % (name, self.class_object.__name__))

    def __getitem__(self, index: Union[int, slice]):
        """

        Args:
            index: an int or slice used to index all the fields.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original camera.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(
                class_object=self.class_object,
                index=self.index + (index,),
                extend=self.extend,
            )

        msg = "Expected index of type int or slice; got %r"
        raise ValueError(msg % type(index))

    def __str__(self):
        return "{}[{}]".format(self.class_object, self.index)
