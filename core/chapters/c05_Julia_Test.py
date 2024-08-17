# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep

class Playground(Page):
    title = "This is a test"

    class SimpleCalculator(VerbatimStep):
        """
    You can use Python to do calculations. Let's do some simple calculations. Copy the code and run it.

    __program_intended__       
        """
        program = "print(4 + 8)"

    final_text = """
Good job!
    """

