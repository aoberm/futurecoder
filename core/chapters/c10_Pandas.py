# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class PandasInPython(Page):
    title = "Pandas in Python"

    class ImportingPandas(VerbatimStep):
        """
    Pandas is a powerful library for data manipulation and analysis in Python.
    It provides tools to work with structured data, particularly in the form of DataFrames and Series.
    Start by importing the Pandas library. This is essential to access its features.
    Here, pd is a common alias used for Pandas to make the code more concise.

        __copyable__
        import pandas as pd
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

        program_in_text = False

    class CreatingSeries(VerbatimStep):
        """
    A Pandas Series is a one-dimensional labeled array capable of holding any data type.

        __copyable__
        import pandas as pd

        data = [10, 20, 30, 40]
        series = pd.Series(data)
        print(series)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = [10, 20, 30, 40]
            series = pd.Series(data)
            print(series)

        program_in_text = False


    class CreatingDataframes(VerbatimStep):
        """
    A DataFrame is a two-dimensional labeled data structure with columns that can be of different types.

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)
        print(df)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)
            print(df)

        program_in_text = False



    class BasicDataFrameOperations(VerbatimStep):
        """
    Now we want to learn the Basic DataFrame Operations.
    Viewing Data:
    Use head() to view the first few rows of the DataFrame.

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)

        print(df.head())

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)

            print(df.head())

        program_in_text = False


    class BasicDataFrameOperations2(VerbatimStep):
        """
    Basic Information:
    Get a summary of the data using info().

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)

        df.info()
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)

            df.info()

        program_in_text = False

    class BasicDataFrameOperations3(VerbatimStep):
        """
    Selecting Columns:
    Access specific columns:

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)

        print(df['Name'])
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)

            print(df['Name'])

        program_in_text = False


    class BasicDataFrameOperations4(VerbatimStep):
        """
    Filtering Rows:
    Filter rows based on conditions.

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)

        adults = df[df['Age'] > 30]
        print(adults)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)

            adults = df[df['Age'] > 30]
            print(adults)

        program_in_text = False



    class ModifyDataFrames(VerbatimStep):
        """
    We can add or drop a column:

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)

        df['Salary'] = [50000, 60000, 70000]

        df = df.drop('Age', axis=1)
        print(df)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)

            df['Salary'] = [50000, 60000, 70000]

            df = df.drop('Age', axis=1)
            print(df)

        program_in_text = False


    final_text = """
    Good job!
    To deepen your understanding, we will explore a real-world dataset in the next step.
"""





class ExploreRealWorldDataset(Page):
    title = "Explore a Dataset"

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

            data.info()

        program_in_text = False


    class MissingVal(VerbatimStep):
        """
    Data quality plays a decisive role in the analysis of large amounts of data.
    In the following, various dimensions of data quality will be examined.

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

            number_of_rows = len(data)
            data.drop_duplicates(subset=None, inplace=True)
            number_of_rows_new = len(data)

            number_of_duplicates = number_of_rows - number_of_rows_new
            print(number_of_duplicates)

        program_in_text = False

    class NewColumn(VerbatimStep):
        """
    In the last step, we would like to add an additional column to the data set.
    You may have noticed that the existing data distinguishes between carbohydrates and sugars.
    Since sugar is a subtype of carbohydrates, we want to add a new column for “Carbohydrates including sugar”.
    In addition, we will save the extended data set to make the changes permanent.

    Create a new column carbs incl. sugar in which you add the amounts of carbohydrates and sugar.


        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()  # Notwendig damit Download geht
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        data['carbs incl. sugar'] = data['carbo'] + data['sugars']

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            data['carbs incl. sugar'] = data['carbo'] + data['sugars']

        program_in_text = False



    class CheckNewColumn(VerbatimStep):
        """
        Check whether the new column has been created correctly by displaying the top 5 rows of the data record again.
        Replace the ?? with an already known command

            __copyable__
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            data['carbs incl. sugar'] = data['carbo'] + data['sugars']
            data.??
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Notwendig damit Download geht
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            data['carbs incl. sugar'] = data['carbo'] + data['sugars']
            data.head()

        program_in_text = False


    final_text = """
    Good job!
    In the next lesson, we'll learn different dimensions of data quality.
"""




