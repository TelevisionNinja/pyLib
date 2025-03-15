from src import mathUtils
import unittest

class MathUtilsTest(unittest.TestCase):
    def test_math(self):
        self.assertEqual(mathUtils.linear_space(0, 1, 2), [0, 1])
        self.assertEqual(mathUtils.linear_space(0, 1, 3), [0, 1/2, 1])
        self.assertEqual(mathUtils.linear_space(0, 1, 0), [])
        self.assertEqual(mathUtils.linear_space(0, 1, 1), [0])

if __name__ == '__main__':
    unittest.main()
