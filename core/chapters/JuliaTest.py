# flake8: NOQA E501
import ast
import re

from astcheck import is_ast_like
from core.text import MessageStep, Page, Step, VerbatimStep
from core import translation as t


class TestClass(Page):
    title = "This is a test"

    class SimpleCalculator(VerbatimStep):
        """
        You can use Python to do calculations. Let's do some simple calculations.
        
        __program_intended__
        
        Copy the code and run it.
        """

        program = "4 + 8"

   