# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class WorkingWithPandas(Page):
    title = "Working with Pandas"

    class importPandas(VerbatimStep):
        """
    Pandas is a powerful and popular Python library used for data manipulation, analysis, and cleaning.
    It provides data structures like Series and DataFrame, which are essential for handling and analyzing data in tabular form.
    Run the code below to load the package.

        __no_auto_translate__
        import pandas as pd
        import pyodide_http

    For the following steps, it is important that you always add to the code and do not delete the previous steps from the editor.
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

        program_in_text = False


    class LoadDataset(VerbatimStep):
        """
    Download the csv file cereals.csv from Moodle, read it in and save it in the variable data.
    To get a first impression of your data set, display the first 5 lines.
    Remember to add to the code and do not delete the previous steps.

        __no_auto_translate__
        pyodide_http.patch_all() #Notwendig damit Download geht
        churn_data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
        print(churn_data.head())


        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http
            pyodide_http.patch_all() #Notwendig damit Download geht
            churn_data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
            print(churn_data.head())

        program_in_text = False


    class RowsCols(ExerciseStep):
        """
    To gain an understanding of the structure of the data set, determine the number of columns and rows in the entire data set.
    To do this, save the values in the variables "rows" and "cols" and print them.
    Also output a complete sentence: “This data set has ... rows and ... columns."

        __no_auto_translate__
        rows = churn_data.shape[0]
        cols = churn_data.shape[1]
        print(rows)
        print(cols)
        print("This data set has", rows, "rows and", cols , "columns.")

    Remember to add to the code and do not delete the previous steps.
        """

        requirements = "hints"

        hints = """ test """

        parsons_solution = False

        def solution(self):
            print(10000)
            print(7)
            print("This data set has", rows, "rows and", cols, "columns.")

        tests = {
            (): """\
10000
7
This data set has 10000 rows and 7 columns.
""",
        }



    class else_full_stop(ExerciseStep):
        """
Change the code above such that the output will be "Welcome to Python!
        """

        hints = """
        hint 1
"""

        parsons_solution = True

        def solution(self):
            welcome = 'Welcome'
            python = welcome + ' to Python!'
            print(python)

        tests = {
            (): 'Welcome to Python!',
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
        correct_output = "Error"

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












