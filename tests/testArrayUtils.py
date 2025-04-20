from src import arrayUtils
import unittest


class ArrayUtilsTest(unittest.TestCase):
    def test_array(self):
        self.assertTrue(arrayUtils.contains([0,1,2,3], 2))
        self.assertFalse(arrayUtils.contains([0,1,2,3], 4))
        self.assertTrue(arrayUtils.contains([2], 2))
        self.assertFalse(arrayUtils.contains([], 4))
        self.assertTrue(arrayUtils.zeros(2), [0,0])
        self.assertTrue(arrayUtils.ones(2), [1,1])


if __name__ == '__main__':
    unittest.main()
