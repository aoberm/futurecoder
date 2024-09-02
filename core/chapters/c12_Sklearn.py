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
    It provides simple and efficient tools for data mining and data analysis.
    This course will cover the most essential functions that you need to get started with scikit-learn.
    Before you start using scikit-learn, you need to import it along with other necessary libraries like numpy and pandas.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

        program_in_text = False

    class LoadingData(VerbatimStep):
        """
    In this analysis, we explore a dataset from an insurance company, which includes various customer details such as age, gender, type of insurance, and history of complaints.
    Notably, the dataset also indicates whether a customer has switched insurance providers, a critical piece of information for the company.
    By training a model to predict this behavior, we aim to help the company identify potential future customers who might switch to their services.
    You can load your dataset using pandas.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        program_in_text = False

    class SplittingData(VerbatimStep):
        """
    Before training a model, separate features and target variable (here: variable "Change", which indicates whether a customer has switched insurance providers).
    Split your data into training and test sets using train_test_split.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        # Separate features and target variable
        X = data.drop('Change', axis=1)
        y = data['Change']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

            # Separate features and target variable
            X = data.drop('Change', axis=1)
            y = data['Change']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        program_in_text = False

    class DataPreprocessing(VerbatimStep):
        """
    Preprocessing your data is a crucial step. You can standardize your features using StandardScaler.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        # Separate features and target variable
        X = data.drop('Change', axis=1)
        y = data['Change']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

            # Separate features and target variable
            X = data.drop('Change', axis=1)
            y = data['Change']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        program_in_text = False

    class TrainingModel(VerbatimStep):
        """
    You can train various machine learning models. Here’s how to train a simple random forest.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        # Separate features and target variable
        X = data.drop('Change', axis=1)
        y = data['Change']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

            # Separate features and target variable
            X = data.drop('Change', axis=1)
            y = data['Change']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Random Forest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

        program_in_text = False

    class MakingPredictions(VerbatimStep):
        """
    Once the model is trained, you can make predictions on the test set and evaluate the performance of your model using metrics like accuracy.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        # Separate features and target variable
        X = data.drop('Change', axis=1)
        y = data['Change']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

            # Separate features and target variable
            X = data.drop('Change', axis=1)
            y = data['Change']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Random Forest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Evaluating the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

        program_in_text = False

    class CrossValidation(VerbatimStep):
        """
    The next steps are really advanced.
    Use cross-validation to assess how well your model generalizes to an independent dataset.
    Optimize your model’s hyperparameters using grid search.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

        # Separate features and target variable
        X = data.drop('Change', axis=1)
        y = data['Change']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f'Cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Best parameters
        print(f'Best parameters: {grid_search.best_params_}')

        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')

            # Separate features and target variable
            X = data.drop('Change', axis=1)
            y = data['Change']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Random Forest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Evaluating the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            print(f'Cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

            # Define parameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }

            # Initialize GridSearchCV
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            # Best parameters
            print(f'Best parameters: {grid_search.best_params_}')

        program_in_text = False

    final_text = """
    Good job!
    This quick course should give you a foundational understanding of how to use scikit-learn for machine learning tasks.
    scikit-learn is a versatile library with a wide range of features, so exploring its documentation further is highly recommended.
"""


class PracticeSklearn(Page):
    title = "Practice: Train your own model"

    class loadDataset(VerbatimStep):
        """
    Now it is time to practice what you just learned with a new dataset.
    The dataset has been adjusted to reflect a scenario involving students who either pass or fail a test.
    The features include age, study hours per week, attendance rate, number of courses taken, assignments completed, test scores, and a target variable indicating whether a student passed the test.
    Load the dataset and print the first few lines to get an impression.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

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
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

            # Print the first few lines
            print(data.head())

        program_in_text = False

    class SplitData(VerbatimStep):
        """
    Separate features and target variable (here: variable "passed", which indicates whether a student passed the test).
    Split your data into training and test sets using train_test_split. Use 80% of the data as training data.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

        # Separate features and target variable
        X = data.drop(?, axis=1)
        y = data[?]

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(?, ?, test_size=?, random_state=42)

        """

        requirements = "hints"
        hints = [
            "Drop the variable 'passed' from you data set for X and save it in your data set y.",
            "Use for the train_test_split() your data sets X and y and set the correct percentage for your test data set (0.2).",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

            # Separate features and target variable
            X = data.drop('passed', axis=1)
            y = data['passed']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        program_in_text = False

    class TrainModel(VerbatimStep):
        """
    Now train a simple random forest. Before you do so, preprocess your data using StandardScaler.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

        # Separate features and target variable
        X = data.drop('passed', axis=1)
        y = data['passed']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(?)
        X_test = scaler.transform(?)

        # Training a Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(?, ?)
        """

        requirements = "hints"

        hints = [
            "For preprocessing you need the corresponding training and test data set. Keep in mind to look for spelling errors.",
            "If you want to use the scaler on your X_train data set you need to fill the first ? with 'X_train'.",
            "To fit your model you need the two training data sets you created earlier.",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

            # Separate features and target variable
            X = data.drop('passed', axis=1)
            y = data['passed']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Random Forest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

        program_in_text = False

    class Predict(VerbatimStep):
        """
    Now you can make predictions on the test set and evaluate the performance of your model using metrics like accuracy.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

        # Separate features and target variable
        X = data.drop('passed', axis=1)
        y = data['passed']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(?)

        # Evaluating the model
        accuracy = accuracy_score(y_test, ?)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        """

        requirements = "hints"
        hints = [
            "For prediction you need the unseen test data from X, where your model shall now make predictions for the previously dropped y-values ('passed').",
            "To calculate the accuracy, the function needs the true values 'y_test' to compare it with the predictions you just created in 'y_pred'. Check if you spelled everything correctly.",
            "Replace the first ? with 'X_test' and the second ? with 'y_pred'.",
        ]

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            # Load data using pandas
            data = pd.read_csv(
                'https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

            # Separate features and target variable
            X = data.drop('passed', axis=1)
            y = data['passed']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features (optional for Random Forest, but we'll keep it for consistency)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Random Forest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Evaluating the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

        program_in_text = False

    final_text = """
        Good job! You have managed to train your own model. Now you can apply your knowledge to various data analysis tasks!
    """
