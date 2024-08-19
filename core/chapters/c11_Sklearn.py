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
    scikit-learn is a powerful library in Python for machine learning.
    It provides simple and efficient tools for data mining and data analysis.
    This course will cover the most essential functions that you need to get started with scikit-learn.
    Before you start using scikit-learn, you need to import it along with other necessary libraries like numpy and pandas.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

        program_in_text = False

    class LoadingData(VerbatimStep):
        """
    You can load your dataset using pandas.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        # Load data using pandas
        data = pd.read_csv('your_dataset.csv')

        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

            # Load data using pandas
            data = pd.read_csv('your_dataset.csv')

        program_in_text = False

    class SplittingData(VerbatimStep):
        """
    Before training a model, separate features and target variable and split your data into training and test sets using train_test_split.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        # Load data using pandas
        data = pd.read_csv('your_dataset.csv')

        # Separate features and target variable
        X = data.drop('target_column', axis=1)
        y = data['target_column']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

            # Load data using pandas
            data = pd.read_csv('your_dataset.csv')

            # Separate features and target variable
            X = data.drop('target_column', axis=1)
            y = data['target_column']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        program_in_text = False


    class DataPreprocessing(VerbatimStep):
        """
    Preprocessing your data is a crucial step. You can standardize your features using StandardScaler.


        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        # Load data using pandas
        data = pd.read_csv('your_dataset.csv')

        # Separate features and target variable
        X = data.drop('target_column', axis=1)
        y = data['target_column']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

            # Load data using pandas
            data = pd.read_csv('your_dataset.csv')

            # Separate features and target variable
            X = data.drop('target_column', axis=1)
            y = data['target_column']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        program_in_text = False



    class TrainingModel(VerbatimStep):
        """
    You can train various machine learning models. Here’s how to train a simple logistic regression model.
        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        # Load data using pandas
        data = pd.read_csv('your_dataset.csv')

        # Separate features and target variable
        X = data.drop('target_column', axis=1)
        y = data['target_column']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

            # Load data using pandas
            data = pd.read_csv('your_dataset.csv')

            # Separate features and target variable
            X = data.drop('target_column', axis=1)
            y = data['target_column']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

        program_in_text = False


    class MakingPredictions(VerbatimStep):
        """
    Once the model is trained, you can make predictions on the test set and Evaluate the performance of your model using metrics like accuracy.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        # Load data using pandas
        data = pd.read_csv('your_dataset.csv')

        # Separate features and target variable
        X = data.drop('target_column', axis=1)
        y = data['target_column']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

            # Load data using pandas
            data = pd.read_csv('your_dataset.csv')

            # Separate features and target variable
            X = data.drop('target_column', axis=1)
            y = data['target_column']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Evaluating the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

        program_in_text = False


    class CrossValidation(VerbatimStep):
        """
    Use cross-validation to assess how well your model generalizes to an independent dataset.
    Optimize your model’s hyperparameters using grid search.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV

        # Load data using pandas
        data = pd.read_csv('your_dataset.csv')

        # Separate features and target variable
        X = data.drop('target_column', axis=1)
        y = data['target_column']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f'Cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

        # Define parameter grid
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear']}

        # Initialize GridSearchCV
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Best parameters
        print(f'Best parameters: {grid_search.best_params_}')

        """

        requirements = "hints"
        hints = """ test """
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV

            # Load data using pandas
            data = pd.read_csv('your_dataset.csv')

            # Separate features and target variable
            X = data.drop('target_column', axis=1)
            y = data['target_column']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardizing the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Evaluating the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy * 100:.2f}%')

            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            print(f'Cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

            # Define parameter grid
            param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear']}

            # Initialize GridSearchCV
            grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            # Best parameters
            print(f'Best parameters: {grid_search.best_params_}')

        program_in_text = False

    final_text = """
    Good job!
    This quick course should give you a foundational understanding of how to use scikit-learn for machine learning tasks.
    scikit-learn is a versatile library with a wide range of features, so exploring its documentation further is highly recommended.
"""










