# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class ExploreDataset(Page):
    title = "Explore a Dataset"

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
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
        print(data.head())

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
            print(data.head())

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
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
        data.head()

        rows = data.shape[?]
        cols = data.shape[?]
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
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
            data.head()

            rows = data.shape[0]
            cols = data.shape[1]
            print(rows)
            print(cols)
            print("This data set has", rows, "rows and", cols, "columns.")

        program_in_text = False



    class ColNames(VerbatimStep):
        """
    Now that you know the structure, you are interested in the names of the individual columns.
    To do this, display the names of the columns. What role do the columns play in this data set?

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
        print(data.head())

        rows = data.shape[0]
        cols = data.shape[1]
        print(rows)
        print(cols)
        print("This data set has", rows , "rows and", cols , "columns.")

        print(data.columns)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
            print(data.head())

            rows = data.shape[0]
            cols = data.shape[1]
            print(rows)
            print(cols)
            print("This data set has", rows, "rows and", cols, "columns.")

            print(data.columns)

        program_in_text = False



    class Info(VerbatimStep):
        """
     Before we continue with the analysis, we want to get a summary of the data set to deepen our understanding.
     To do this, enter data.info() in the next code block and study the output.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
        print(data.head())

        rows = data.shape[0]
        cols = data.shape[1]
        print(rows)
        print(cols)
        print("This data set has", rows , "rows and", cols , "columns.")

        print(data.columns)

        data.info()

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
            print(data.head())

            rows = data.shape[0]
            cols = data.shape[1]
            print(rows)
            print(cols)
            print("This data set has", rows, "rows and", cols, "columns.")

            print(data.columns)

            data.info()

        program_in_text = False

    final_text = """
    Good job!
    In the next lesson, we'll learn different dimensions of data quality.
"""






class DataQuality(Page):
    title = "Working with Pandas"

    class Introduction(VerbatimStep):
        """
    Data quality plays a decisive role in the analysis of large amounts of data.
    In the following, various dimensions of data quality will be examined.
        """

    class MissingVal(VerbatimStep):
        """
    First of all, we want to test for missing data. Missing entries are marked with NULL in databases.
    Output the percentage of missing data in columns or rows by cleverly linking the .isnull() and .sum() functions.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()  # Notwendig damit Download geht
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        print(round(100*data.isnull().sum()/len(data)),2)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            print(round(100 * data.isnull().sum() / len(data)), 2)

        program_in_text = False


    class RedundantData(VerbatimStep):
        """
    Redundant data is often caused by data records being saved multiple times. Remove all duplicates to restore order.
    Display the number of duplicates by using the number of data records before and after removing the duplicates.


        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()  # Notwendig damit Download geht
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        print(round(100 * data.isnull().sum() / len(data)), 2)

        number_of_rows = len(data)
        data.drop_duplicates(subset=None, inplace=True)
        number_of_rows_new = len(data)

        number_of_duplicates = number_of_rows - number_of_rows_new
        print(number_of_duplicates)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            print(round(100 * data.isnull().sum() / len(data)), 2)

            number_of_rows = len(data)
            data.drop_duplicates(subset=None, inplace=True)
            number_of_rows_new = len(data)

            number_of_duplicates = number_of_rows - number_of_rows_new
            print(number_of_duplicates)

        program_in_text = False


    final_text = """
    Good job!
    In the next lesson, we'll learn to work with the data.
"""










