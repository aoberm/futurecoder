# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class Hannah(Page):
    title = "Welcome to Python"

    class FirstSteps(VerbatimStep):
        """
        Python is one of the most popular programming languages — it's been used to write millions of computer programs.
        This is a very simple Python program. Copy the code and press “Run” to see what it does.

    __program_indented__
        """
        program = "print('Hello World')"

    class SecondStep(VerbatimStep):
        """
    Try changing the text of the program and running it again.

    __program_indented__

    Run the Code
        """
        program = "print('Hello Python')"


    class firstQuestion(VerbatimStep):
        """
    What do you think this program will output?

    __program_indented__

    Run the Code to find out.
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

    class FourthStep(VerbatimStep):
        """
        The + operator combines strings together. If you want a space between the strings, you'll need to add it to one of the strings before combining.
        In Python, there are often multiple ways to accomplish a task.
        Alternatively, you can list the strings in the print function using the , operator which inserts a space.

    __program_indented__
        """
        program = "print('Welcome to ', 'Python!')"

    final_text = """
    Good job!
    In the next lesson, we'll learn how to manipulate numbers in Python.
"""

class Obi(Page):
    title = "Welcome to Python again"

    class SecondFirstSteps(VerbatimStep):
        """
        Python is one of the most popular programming languages — it's been used to write millions of computer programs.
        This is a very simple Python program. Copy the code and press “Run” to see what it does.

    __program_indented__
        """
        program = "print('Hello Obi')"


    final_text = """
Good job!
In Python, there are often multiple ways to accomplish a task. In the next lesson, we'll learn how to manipulate numbers in Python.
"""












class Anleitung(Page):
    title = "Anleitung"

    class CodeSchnipsel(VerbatimStep):
        """
    Text

    __program_indented__

        Text im Code
        """
        program = "print('Hello Obi')"

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

