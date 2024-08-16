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


    class LoadDataset(VerbatimStep):
        """
    Download the csv file cereals.csv from Moodle, read it in and save it in the variable data.
    To get a first impression of your data set, display the first 5 lines.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        churn_data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
        print(churn_data.head())

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            churn_data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
            print(churn_data.head())

        program_in_text = False

    class RowsCols(VerbatimStep):
        """
    To gain an understanding of the structure of the data set, determine the number of columns and rows in the entire data set.
    To do this, save the values in the variables "rows" and "cols" and print them.
    Also output a complete sentence: “This data set has ... rows and ... columns."

    Write the correct solution instead of the question marks.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        churn_data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
        churn_data.head()

        rows = churn_data.shape[?]
        cols = churn_data.shape[?]
        print(rows)
        print(cols)
        print("This data set has", ? , "rows and", ? , "columns.")

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            churn_data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
            churn_data.head()

            rows = churn_data.shape[0]
            cols = churn_data.shape[1]
            print(rows)
            print(cols)
            print("This data set has", rows, "rows and", cols, "columns.")

        program_in_text = False



    class ColNames(VerbatimStep):
        """
    Now that you know the structure, you are interested in the names of the individual columns.
    To do this, display the names of the columns. What role do the columns play in this data set?

    Write the correct solution instead of the question marks.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        churn_data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
        print(churn_data.head())

        rows = churn_data.shape[?]
        cols = churn_data.shape[?]
        print(rows)
        print(cols)
        print("This data set has", ? , "rows and", ? , "columns.")

        print(churn_data.columns)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            churn_data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
            print(churn_data.head())

            rows = churn_data.shape[0]
            cols = churn_data.shape[1]
            print(rows)
            print(cols)
            print("This data set has", rows, "rows and", cols, "columns.")

            print(churn_data.columns)

        program_in_text = False



    class Info(VerbatimStep):
        """
     Before we continue with the analysis, we want to get a summary of the data set to deepen our understanding.
     To do this, enter data.info() in the next code block and study the output.

    Write the correct solution instead of the question marks.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        churn_data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
        print(churn_data.head())

        rows = churn_data.shape[?]
        cols = churn_data.shape[?]
        print(rows)
        print(cols)
        print("This data set has", ? , "rows and", ? , "columns.")

        print(churn_data.columns)

        churn_data.info()

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            churn_data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/churn_dataset.csv')
            print(churn_data.head())

            rows = churn_data.shape[0]
            cols = churn_data.shape[1]
            print(rows)
            print(cols)
            print("This data set has", rows, "rows and", cols, "columns.")

            print(churn_data.columns)

            churn_data.info()

        program_in_text = False

    final_text = """
    Good job!
    In the next lesson, we'll learn how to manipulate numbers in Python.
"""










