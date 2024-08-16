# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class SklearnInPython(Page):
    title = "Sklearn in Python"

    class importSklearn(VerbatimStep):
        """
    Pandas is a powerful and popular Python library used for data manipulation, analysis, and cleaning.
    It provides data structures like Series and DataFrame, which are essential for handling and analyzing data in tabular form.
    Run the code below to load the package.

        __copyable__
        import pandas as pd
        import pyodide_http

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

        program_in_text = False


    final_text = """
    Good job!
    You have now mastered the first important steps with Python. Based on your newly acquired knowledge, you will be able to use Python as a tool for various tasks!
"""










