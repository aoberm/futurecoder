# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class Anleitung(Page):
    title = "Anleitung"

    class CodeSchnipsel(VerbatimStep):
        """
    Text

    __program_indented__

        Text im Code
        """

        program = "print('Hello Obi')"

        #wenn es hints gibt braucht es requirements

        requirements = "Here you can write requirements"

        hints = [
            "Hint1",
            "Hint2",
        ]

    class Auswahlmöglichkeiten(VerbatimStep):
        """
    Text

    __program_indented__

    Zeigt Wahlmögichkeiten an

        """

        predicted_output_choices = [
            "Welcome\n"
            "Welcome to Python!",
            "Welcome!\n"
            "Welcome to Python!",
            "Welcome\n"
            "Welcometo Python!",
        ]

        def program(self):
            welcome = 'Welcome'
            print(welcome)
            python = welcome + 'to Python!'
            print(python)


    final_text = """
Hier muss finaler Text stehen.
"""

