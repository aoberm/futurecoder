# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class NumpyInPython(Page):
    title = "Numpy in Python"

    class importnumpy(VerbatimStep):
        """
    NumPy is a fundamental package for scientific computing in Python.
    It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
    Begin by importing the NumPy library, commonly abbreviated as np.

        __copyable__
        import numpy as np

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

        program_in_text = False


    final_text = """
    Good job!
    You have now mastered the first important steps with Python. Based on your newly acquired knowledge, you will be able to use Python as a tool for various tasks!
"""










