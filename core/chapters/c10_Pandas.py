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



