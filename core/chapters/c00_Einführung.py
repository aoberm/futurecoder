# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class IntroductionToPython(Page):
    title = "Welcome to Python"

    class HelloWorld(VerbatimStep):
        """
    Python is one of the most popular programming languages — it's been used to write millions of computer programs.
    This is a very simple Python program. Copy the code and press "run” to see what it does.

    __program_indented__
        """
        program = "print('Hello World')"

    class HelloPython(VerbatimStep):
        """
    Try changing the text of the program such that it will say "Hello Python" and run it again.

        """
        program = "print('Hello Python')"
        program_in_text = False

    class WelcomePythonSpace(VerbatimStep):
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


    class WelcomeToPYthonSpace2(VerbatimStep):
            """
    You can see that `+` combines or joins two strings together end to end. Technically, this is called concatenation.

    Here's an exercise: change the previous code slightly so that the result is the string `'hello world'`, i.e. with a space between the words.

    By the way, if you get stuck, you can click the lightbulb icon in the bottom right for a hint.
            """
            requirements = "Use `+` to add two or more strings together, so that the result is the string `'Welcome to Python'`."

            hints = [
                "A space is a character just like any other, like `o` or `w`.",
                "The space character must be somewhere inside quotes.",
            ]

            program = "'Welcome to  ' + 'Python'"
            program_in_text = False


    class WelcometoPytonComma(VerbatimStep):
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












