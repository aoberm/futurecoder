# flake8: NOQA E501
import ast
import random
from collections import Counter
from typing import List, Dict

from core import translation as t
from core.exercises import assert_equal
from core.exercises import generate_string, generate_dict
from core.text import (
    ExerciseStep,
    Page,
    Step,
    VerbatimStep,
)

class IntroducingDictionaries(Page):
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

        hints = """There are no hints"""

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

        hints = """There are no hints"""

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

        hints = """There are no hints"""

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

    final_text = """
    Great job!
    """


class UsingDictionaries(Page):
    title = "Pandas in Python (advanced)"

    class BasicDataFrameOperations(VerbatimStep):
        """
    Now we want to learn the Basic DataFrame Operations. \n
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

        hints = """There are no hints"""

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

        hints = """There are no hints"""

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

    class BasicDataFrameOperations21(VerbatimStep):
        """
    Information about the structure:
    Using the `shape` function, determine the number of rows and columns of our DataFrame. Like in matrix notation, the argument `shape[0]` gives you the number of rows, `shape[1]` gives you the number of columns and just `shape` gives you both.

        __copyable__
        import pandas as pd

        data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago'],
                'Gender': ['f', 'm', 'f']
        }
        df = pd.DataFrame(data)

        print("size: ", df.shape)
        print("Number of rows: ", df.shape[0])
        print("Number of cols: ", df.shape[1])


        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago'],
                'Gender': ['f', 'm', 'f']
            }
            df = pd.DataFrame(data)

            print("size: ", df.shape)
            print("Number of rows: ", df.shape[0])
            print("Number of cols: ", df.shape[1])

        program_in_text = False

    class BasicDataFrameOperations3(VerbatimStep):
        """
    Selecting Columns:
    Access specific columns and get column names:

        __copyable__
        import pandas as pd

        data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }
        df = pd.DataFrame(data)

        print(df['Name'])
        print(df.columns)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            df = pd.DataFrame(data)

            print(df['Name'])
            print(df.columns)

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

        hints = """There are no hints"""

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

        hints = """There are no hints"""

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
        To deepen your understanding, we will explore a real-world dataset in the next step and apply what we just learned.
    """

class DictionaryKeysAndValues(Page):
    title = "Show Your Skills: Explore a real world dataset"

    class LoadDataset(VerbatimStep):
        """
    Download the csv file cereals.csv from Moodle, read it in and save it in the variable data.


        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        program_in_text = False

    class LoadDataset2(VerbatimStep):
        """
    To get a first impression of your data set, display the first 5 lines. To directly apply your knowledge of data frames, replace the “?” with the correct code.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
        print(data.?())

        """

        requirements = "hints"

        hints = [
            "To display the first 5 lines you have to insert the correct function after data.",
            "You will find online the correct function, that lets you look at the first few lines of data sets.",
        ]

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')
            print(data.head())

        program_in_text = False

    class RowsCols(VerbatimStep):
        """
    To gain an understanding of the structure of the data set, determine the number of columns and rows in the entire data set.
    To do this, save the values in the variables "rows" and "cols" and print them.
    Also output a complete sentence: “This data set has ... rows and ... columns."

    To directly apply your knowledge of data frames, replace the “?” with the correct code.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        rows = data.shape[?]
        cols = data.shape[?]
        print(rows)
        print(cols)
        print("This data set has", ? , "rows and", ? , "columns.")

        """

        requirements = "hints"

        hints = [
            "The 'shape' function returns the number of rows and columns in your DataFrame as a tuple e.g. (5,9).",
            "To access only the number of rows you have to call the 'shape' function at the index 0.",
            "To complete the sentece, use the variables 'rows' and 'cols' accordingly.",
        ]

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
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

    To directly apply your knowledge of data frames, replace the “?” with the correct code.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        print(data.?)

        """

        requirements = "hints"

        hints = [
            "Try to remember the name of the function that shows you the names of the columns.",
            "Replace the ? with the correct function (columns).",
        ]

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            print(data.columns)

        program_in_text = False

    class Info(VerbatimStep):
        """
    Before we continue with the analysis, we want to get a summary of the data set to deepen our understanding.
    To directly apply your knowledge of data frames, replace the “?” with the correct code.

        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        data.?()

        """

        requirements = "hints"

        hints = [
            "You are looking for a functon that gives you information about the DataFrame.",
            "What could be a short (4 letters) name for a function that gives you information?",
        ]

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
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

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        print(round(100*data.isnull().sum()/len(data)),2)

        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
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

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        number_of_rows = len(data)
        data.drop_duplicates(subset=None, inplace=True) 
        #subset = None considers all columns of the DataFrame for duplicate detection
        #inplace = True modifies the original DataFrame in-place, removing the duplicate rows
        number_of_rows_new = len(data)

        number_of_duplicates = number_of_rows - number_of_rows_new
        print(number_of_duplicates)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            number_of_rows = len(data)
            data.drop_duplicates(subset=None, inplace=True)
            #subset = None considers all columns of the DataFrame for duplicate detection
            #inplace = True modifies the original DataFrame in-place, removing the duplicate rows
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
    To directly apply your knowledge of data frames, replace the “?” with the correct code.


        __copyable__
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

        data['carbs incl. sugar'] = data['carbo'] + data[?]

        """

        requirements = "hints"

        hints = [
            "The name of the column you are looking for is 'sugars'.",
            "Replace the ? with the columnname 'sugars' including the single quotation marks.",
        ]

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            data['carbs incl. sugar'] = data['carbo'] + data['sugars']

        program_in_text = False

    class CheckNewColumn(VerbatimStep):
        """
        Check whether the new column has been created correctly by displaying the top 5 rows of the data record again.
        To directly apply your knowledge of data frames, replace the “?” with the correct code.

            __copyable__
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            data['carbs incl. sugar'] = data['carbo'] + data['sugars']
            print(data.??)
        """

        requirements = "hints"

        hints = [
            "Call the function displaying the first rows of a DataFrame and do not forget the parenthesis.",
            "Replace the first ? with the name of the function and replace the second ? with the parenthesis ().",
        ]

        def program(self):
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/cereals.csv')

            data['carbs incl. sugar'] = data['carbo'] + data['sugars']
            print(data.head())

        program_in_text = False

    final_text = """
        Good job!
    """
