# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class SklearnInPython(Page):
    title = "Sklearn in Python"

    class ImportSklearn(VerbatimStep):
        """
    Scikit-learn is a powerful library in Python for machine learning.
    It provides simple and efficient tools for data mining and data analysis. The tools are useful for classification, regression, clustering, dimensionality reduction, model selection, perprocessing, and much more.
    First we have to import the packages we will need later on.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd

        program_in_text = False

    class CreateData(VerbatimStep):
        """
    If you want to train a model you first need data. We will download a DataFrame about some students, their study time, and their resulting grades.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')
        
        # Show DataFrame
        print(df)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all() #Notwendig damit Download geht
            df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')

            # Show DataFrame
            print(df)
            
        program_in_text = False

    class SplittingData(VerbatimStep):
        """
    Before training a model, you need to separate features and target variable (here: variable 'Grade', which indicates whether a customer has switched insurance providers).
    Sklearn offers for that a function called `train_test_split` which splits datasets into random train and test subsets.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')
        
        # Separate features and target variable
        X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
        y = df['Grade']  # y will contain the Grade, which is the target for prediction
        
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('X_train:', X_train)
        print('X_test:', X_test)
        print('y_train:', y_train)
        print('y_test:', y_test)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all() #Notwendig damit Download geht
            df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')
            
            # Separate features and target variable
            X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
            y = df['Grade']  # y will contain the Grade, which is the target for prediction
            
            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print('X_train:', X_train)
            print('X_test:', X_test)
            print('y_train:', y_train)
            print('y_test:', y_test)

        program_in_text = False


    class LinearRegression(VerbatimStep):
        """
    Sklearn provides multiple functions for machine learning models such as `LinearRegression`, `LogisticRegression`, `DecisionTree`, and many more. 
    Let's have a look at `LinearRegression` which performs a linear regression to model the relationship between a dependent variable and one ore more dependent variables.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')
        
        # Separate features and target variable
        X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
        y = df['Grade']  # y will contain the Grade, which is the target for prediction

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        print('Model was successfully fitted to your training data')
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all() #Notwendig damit Download geht
            df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')

            # Separate features and target variable
            X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
            y = df['Grade']  # y will contain the Grade, which is the target for prediction

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            print('Model was successfully fitted to your training data')

        program_in_text = False
        
        
    class ExampleStudent(VerbatimStep):
        """
    First you can try out the model. Feel free to change the number in our variable `new_study_time` to look at the different predictions of our model.
    This might take a moment to execute.
    
        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')

        # Separate features and target variable
        X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
        y = df['Grade']  # y will contain the Grade, which is the target for prediction

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Try different study times
        new_study_time = 25 # Example study time that can be changed
        
        new_student_df = pd.DataFrame({'Study time (Hours)': [new_study_time]})

        new_student_pred = model.predict(new_student_df)[0]
        print(new_student_pred)
        """
        
        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all() #Notwendig damit Download geht
            df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')

            # Separate features and target variable
            X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
            y = df['Grade']  # y will contain the Grade, which is the target for prediction

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Try different study times
            new_study_time = 25 # Example study time that can be changed

            new_student_df = pd.DataFrame({'Study time (Hours)': [new_study_time]})

            new_student_pred = model.predict(new_student_df)[0]
            print(new_student_pred)

        program_in_text = False
      

    class MakingPredictions(VerbatimStep):
        """
    Now let our newly fitted model do some predictions for the grades in our `X_test` data set.
    The `predict` function uses the given data set (here X_test) and predicts the y values (here 'Grade'). 
    This might take a moment to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all() #Notwendig damit Download geht
        df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')
        
        # Separate features and target variable
        X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
        y = df['Grade']  # y will contain the Grade, which is the target for prediction

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Making predictions
        y_pred = model.predict(X_test)
        
        # Add predictions to the X_test DataFrame
        # This code snippet is solely for visualization of the predictions
        X_test['Predicted Grade'] = y_pred.round(1)

        # Add the corresponding student names to the X_test DataFrame
        X_test['Student'] = df.loc[X_test.index, 'Student']
        
        # Add true grades to the X_test DataFrame
        X_test['True Grade'] = y_test.values

        # Reorder the columns in the desired order: Student, Study time (Hours), Predicted Grade
        X_test = X_test[['Student', 'Study time (Hours)', 'Predicted Grade', 'True Grade']]

        # Look at the predictions
        print(X_test)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all() #Notwendig damit Download geht
            df = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_grades.csv')

            # Separate features and target variable
            X = df.drop(columns=['Student', 'Grade'])  # X will only contain the Study time
            y = df['Grade']  # y will contain the Grade, which is the target for prediction

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Add predictions to the X_test DataFrame
            # This code snippet is solely for visualization of the predictions
            X_test['Predicted Grade'] = y_pred.round(1)

            # Add the corresponding student names to the X_test DataFrame
            X_test['Student'] = df.loc[X_test.index, 'Student']
            
            # Add true grades to the X_test DataFrame
            X_test['True Grade'] = y_test.values

            # Reorder the columns in the desired order: Student, Study time (Hours), Predicted Grade
            X_test = X_test[['Student', 'Study time (Hours)', 'Predicted Grade', 'True Grade']]

            # Look at the predictions
            print(X_test)
            
        program_in_text = False

    
    final_text = """
    Good job!
    This quick course should give you a understanding of how to use scikit-learn for machine learning tasks.
    scikit-learn is a versatile library with a wide range of features, so exploring its documentation further is highly recommended.
"""


class PracticeSklearn(Page):
    title = "Practice: Sklearn in Python"

    class loadDataset(VerbatimStep):
        """
    Now it is time to practice what you just learned with a new dataset.
    The dataset has been adjusted to reflect a scenario involving cars, where the goal is to predict car prices solely based on miles driven.
    The DataFrame includes the variables `Car Model`, `Miles Driven (Thousands)`, and a target variable `Price (Thousands of dollars)`.
    Load the dataset and print the first few lines to get an impression.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

        # Print the first few lines
        print(data.?())
        """

        requirements = "hints"

        hints = [
            "Call the function displaying the first rows of a data set.",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

            # Print the first few lines
            print(data.head())

        program_in_text = False
        

    class SplitData(VerbatimStep):
        """
    Now you should separate the features and target variable. Drop from your DataFrame `data` certain columns, so that the only variable left is `Miles Driven (Thousands)`
    Split your data into training and test sets using `train_test_split`. Use 80% of the data as training data.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

        # Separate features and target variable
        X = data.drop(columns=['?','?'])
        y = data[?]

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(?, ?, test_size=?, random_state=42)
        """

        requirements = "hints"
        hints = [
            "Drop the variable 'Car Model' and 'Price (Thousands of dollars)' from your data set for X and save the latter variable in your data set y.",
            "Use for the train_test_split() your data sets X and y and set the correct percentage for your test data set (0.2).",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

            # Separate features and target variable
            X = data.drop(columns=['Car Model', 'Price (Thousands of dollars)'])
            y = data['Price (Thousands of dollars)']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        program_in_text = False


    class TrainLinearRegression(VerbatimStep):
        """
    Now train a LinearRegression. Fot that you have to define the correct model and fit it.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

        # Separate features and target variable
        X = data.drop(columns=['Car Model','Price (Thousands of dollars)'])
        y = data['Price (Thousands of dollars)']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #Training a LinearRegression model
        model = ?
        ?.fit(X_train, ?)
        """

        requirements = "hints"

        hints = [
            "First you have to save in the variable 'model' the model you want to use for your predictions.",
            "You should apply the 'fit' function to your model. After that you have to use the two 'train' data sets, because you want to fit your model on them.",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

            # Separate features and target variable
            X = data.drop(columns=['Car Model', 'Price (Thousands of dollars)'])
            y = data['Price (Thousands of dollars)']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)

        program_in_text = False
        
        
    class MakingPrecitions(VerbatimStep):
        """
    Now that you successfully trained your model it is time to make some predictions.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

        # Separate features and target variable
        X = data.drop(columns=['Car Model','Price (Thousands of dollars)'])
        y = data['Price (Thousands of dollars)']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Making predictions
        y_pred = model.?
        
        print('Predictions made')
        """
        
        requirements = "hints"

        hints = [
            "To make predictions you not only have to call the correct function, but you also have to specify which dataset you want to use for your predictions.",
            "You trained your model with the 'X_train' and 'y_train' datasets. You need the 'y_test' dataset to evaluate how well your model predicts values. Hence the only dataset left to make predictions is your 'X_test' dataset.",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

            # Separate features and target variable
            X = data.drop(columns=['Car Model', 'Price (Thousands of dollars)'])
            y = data['Price (Thousands of dollars)']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Making predictions
            y_pred = model.predict(X_test)
            
            print('Predictions made')

        program_in_text = False
        
        
    class PrintPredictions(VerbatimStep):
        """
    Let's have a look at the predictions of our model. Just copy the following code so that you can compare the predictions and the true values.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

        # Separate features and target variable
        X = data.drop(columns=['Car Model','Price (Thousands of dollars)'])
        y = data['Price (Thousands of dollars)']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Making predictions
        y_pred = model.predict(X_test)
        
        # Add predictions to the X_test DataFrame
        X_test['Predicted Price'] = y_pred

        # Add the corresponding car models to the X_test DataFrame
        X_test['Car Model'] = data.loc[X_test.index, 'Car Model']
        # Add the true prices from y_test for comparison
        X_test['True Price'] = y_test.values

        # Reorder the columns in the desired order: Car Model, Miles Driven, Predicted Price, True Price
        X_test = X_test[['Car Model', 'Miles Driven (Thousands)', 'Predicted Price', 'True Price']]

        # Look at the predictions
        print(X_test)
        """
        
        requirements = "hints"

        hints = [
            "There are not hints.",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/car_prices.csv')

            # Separate features and target variable
            X = data.drop(columns=['Car Model', 'Price (Thousands of dollars)'])
            y = data['Price (Thousands of dollars)']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Making predictions
            y_pred = model.predict(X_test)
            
            # Add predictions to the X_test DataFrame
            X_test['Predicted Price'] = y_pred

            # Add the corresponding car models to the X_test DataFrame
            X_test['Car Model'] = data.loc[X_test.index, 'Car Model']
            # Add the true prices from y_test for comparison
            X_test['True Price'] = y_test.values

            # Reorder the columns in the desired order: Car Model, Miles Driven, Predicted Price, True Price
            X_test = X_test[['Car Model', 'Miles Driven (Thousands)', 'Predicted Price', 'True Price']]

            # Look at the predictions
            print(X_test)

        program_in_text = False
        
    final_text = """
        Good job! As you have noticed we only used the variable `Miles Driven` for our predictions. Of course if this was a real dataset this would not make very much sense to only use this variable for price predictions.
        However the idea is always the same: You have a variable you want to predict and you have variables to make the predictions. Based on that you have to design your machine learning algorithm.
    """
