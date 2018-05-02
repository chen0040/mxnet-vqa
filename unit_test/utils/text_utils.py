import unittest
import numpy as np
from mxnet_vqa.utils.text_utils import pad_sequence


class TestClass(unittest.TestCase):
    
    def test_pad_sequence(self):
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 2, 'left'), np.array([1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 3, 'left'), np.array([1, 1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 4, 'left'), np.array([1, 1, 1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 5, 'left'), np.array([0, 1, 1, 1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 6, 'left'), np.array([0, 0, 1, 1, 1, 1]))

        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 2, 'right'), np.array([1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 3, 'right'), np.array([1, 1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 4, 'right'), np.array([1, 1, 1, 1]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 5, 'right'), np.array([1, 1, 1, 1, 0]))
        np.testing.assert_array_equal(pad_sequence([1, 1, 1, 1], 6, 'right'), np.array([1, 1, 1, 1, 0, 0]))


if __name__ == '__main__':
    unittest.main()