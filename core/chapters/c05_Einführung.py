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

        requirements = "hints"

        hints = """
        Change the text betwen the ''
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
            "Welcome to Python!",
            "Welcome to Python",
            "Welcometo Python!",
        ]

        def program(self):
            welcome = 'Welcome'
            python = welcome + 'to Python!'
            print(python)




    class else_full_stop(ExerciseStep):
        """
If `excited` is true then `char` is defined and everything runs fine. But otherwise
`char` never gets assigned a value, so trying to use it in `sentence += char` fails.

Fix this by adding an `else` clause to the `if` so that if `excited` is false, a full stop (`.`)
is added to the end of the sentence instead of an exclamation mark (`!`).
        """

        hints = """
Don't change anything that's already there, just add a bit more code.
`else` needs to come immediately after the `if` body, with nothing in between.
`sentence += char` needs to run whether `excited` is `True` or `False`.
You *could* have a copy of `sentence += char` in both the `if` and `else` blocks, but there's a better way.
Use `else` to assign a different value to `char`.
If `excited` is `False`, then `char` should be `'.'` instead of `'!'`.
"""

        parsons_solution = True

        def solution(self, sentence: str, excited: bool):
            if excited:
                char = '!'
            else:
                char = '.'
            sentence += char

            print(sentence)

        tests = {
            ('Hello there', True): 'Hello there!',
            ('Goodbye', False): 'Goodbye.',
        }





    class WelcometoPytonComma(VerbatimStep):
        """
    If you want a space between the strings, you'll need to add it to one of the strings before combining.
    In Python, there are often multiple ways to accomplish a task.
    Alternatively, you can list the strings in the print function using the , operator which inserts a space.

    __program_indented__
        """
        program = "print('Welcome to ', 'Python!')"


    class FixError(VerbatimStep):
        """
    The program `print('Welcome to Python!)` is causing an error. Copy this Code and see if you can fix it.
        """
        requirements = "hints"

        hints = """
        add a '
                """


        def program(self):
            print('Welcome to Python!')

        program_in_text = False

    final_text = """
    Good job!
    In the next lesson, we'll learn how to manipulate numbers in Python.
"""




class NumbersInPython(Page):
    title = "Numbers in Python"

    class Calculation1(VerbatimStep):
        """
    So far, we've made programs that display text, but many programs need to display numbers.

    __program_indented__

    What do you think this program will display?
        """
        program = "print(7)"

        predicted_output_choices = [
            '7\nNone',
            '7',
            "It will give an error",
            "Nothing",
        ]

    class Calculation2(VerbatimStep):
        """
    So far, we've made programs that display text, but many programs need to display numbers.

    __program_indented__

    What do you think this program will display?
        """
        program = "print(7 + 12)"

        predicted_output_choices = [
            '19\nNone',
            '19',
            "The number 19",
            "'19'",
        ]

    class Calculation3(VerbatimStep):
        """
    Here's another program that manipulates numbers. Run the code.

    __program_indented__

    Python follows the standard order of operations used in arithmetic — so multiplication and division are performed before addition and subtraction.
        """

        program = "print(1 / 2 * 3 + 4 - 5)"

    final_text = """
Good job!
    """



class CombiningNumbersandStrings(Page):

    title = "Combining Numbers and Strings"

    class Combine1(VerbatimStep):
        """
    What if we want to display text along with a number? Here's one approach you could use.

    __program_indented__

    Run the Code
        """
        program = "print('My age is 27')"


    class Combine2(VerbatimStep):
        """
    Here are two other approaches:

    print("My age is " + 27) and print("My age is", 27)

    Which of these approaches do you think would work? Run the code which works and try.
        """

        def program(self):
            print("My age is", 27)

        requirements = "hints"

        hints = """
                        Delete one of the lines
                                """
        program_in_text = False


    class Combine3(VerbatimStep):
        """
    We can use the function str() to produce the desired output with the '+' sign. Try to write a code which uses '+' and str().
        """

        def program(self):
            print("My age is" + str(27))

        requirements = "hints"

        hints = """str(7) converts the number 7 to a string.
        """
        program_in_text = False


    class Combine4(VerbatimStep):
        """
    Here's another program that uses concatenation.

    __program_indented__

    Since both parts of the expression are strings of text, the program concatenates them.
        """

        def program(self):
            print("My age is" + "27")



    final_text = """
Good job! Numbers and strings are two of the core data types in Python, and Python treats them differently.
Understanding how different data types behave and what rules they follow is a key skill when working in Python.
    """












