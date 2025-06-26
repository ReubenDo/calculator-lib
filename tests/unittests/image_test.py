import numpy as np

from calculator.image import Image


def test_image_instantiation():
    """Test the correct instantiation of the Image class with a 2D Numpy array."""
    assert isinstance(Image(np.array([[1, 2], [3, 4]])), Image)

def test_image_data_set_correctly():
    """Test that the _data attribute of the Image class is set correctly by the constructor."""
    from numpy.testing import assert_array_equal

    raw_data = np.array([[1, 2], [3, 4]])
    assert_array_equal(
        Image(raw_data)._data,
        raw_data,
    )


def test_bad_type_in_image_instantiation():
    """Test that a proper TypeError is raised when providing data in other form than a 2D Numpy array."""
    import pytest

    with pytest.raises(
        TypeError,
        match="Image data should be a 2D Numpy array.",
    ):
        Image(10)

