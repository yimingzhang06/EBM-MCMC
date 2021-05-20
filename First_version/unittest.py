import unittest
import torch
import time
from langevin import *
from optimizers import *


# For testing the SGLD, the train.py has tested the SGLD function. For testing the langevin.py, the tests.py has tested it.

# This .py file will be used for test energy function in the next week.

class MyTestCase(unittest.TestCase):
    def test_something(self,):
        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
