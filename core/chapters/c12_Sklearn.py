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
    If you want to train a model you first need data. We will create a DataFrame about some students, their study time, and their resulting grades.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        # Set seed
        np.random.seed(42)

        # Number of students
        n_students = 50

        # Generate names
        names = list(range(n_students))

        # Amount of hours each student spent learning (ranges from 0 to 40)
        h_learned = np.random.randint(0, 41, size=n_students)

        # Function for grades based on learning time
        def generate_grade(h_learned):
            return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

        # Generate grades
        grades = [generate_grade(lz) for lz in h_learned]

        # Create DataFrame
        df = pd.DataFrame({
            "Student": names,
            "Study time (Hours)": h_learned,
            "Grade": grades
        })

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

            # Set seed
            np.random.seed(42)

            # Number of students
            n_students = 50

            # Generate names
            names = list(range(n_students))

            # Amount of hours each student spent learning (ranges from 0 to 40)
            h_learned = np.random.randint(0, 41, size=n_students)

            # Function for grades based on learning time
            def generate_grade(h_learned):
                return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

            # Generate grades
            grades = [generate_grade(lz) for lz in h_learned]

            # Create DataFrame
            df = pd.DataFrame({
                "Student": names,
                "Study time (Hours)": h_learned,
                "Grade": grades
            })

            # Show DataFrame
            print(df)

        program_in_text = False

    class SplittingData(VerbatimStep):
        """
    Before training a model, you need to separate features and target variable (here: variable "Grade", which indicates whether a customer has switched insurance providers).
    Sklearn offers for that a function called `train_test_split` which splits datasets into random train and test subsets.
    This might take a moment for the code to execute.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        # Set seed
        np.random.seed(42)

        # Number of students
        n_students = 50

        # Generate names
        names = list(range(n_students))

        # Amount of hours each student spent learning (ranges from 0 to 40)
        h_learned = np.random.randint(0, 41, size=n_students)

        # Function for grades based on learning time
        def generate_grade(h_learned):
            return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

        # Generate grades
        grades = [generate_grade(lz) for lz in h_learned]

        # Create DataFrame
        df = pd.DataFrame({
            "Student": names,
            "Study time (Hours)": h_learned,
            "Grade": grades
        })
        
        # Separate features and target variable
        X = df.drop('Grade', axis=1)
        y = df['Grade']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            # Set seed
            np.random.seed(42)

            # Number of students
            n_students = 50

            # Generate names
            names = list(range(n_students))

            # Amount of hours each student spent learning (ranges from 0 to 40)
            h_learned = np.random.randint(0, 41, size=n_students)

            # Function for grades based on learning time
            def generate_grade(h_learned):
                return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

            # Generate grades
            grades = [generate_grade(lz) for lz in h_learned]

            # Create DataFrame
            df = pd.DataFrame({
                "Student": names,
                "Study time (Hours)": h_learned,
                "Grade": grades
            })

            # Separate features and target variable
            X = df.drop('Grade', axis=1)
            y = df['Grade']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        
        # Set seed
        np.random.seed(42)

        # Number of students
        n_students = 50

        # Generate names
        names = list(range(n_students))

        # Amount of hours each student spent learning (ranges from 0 to 40)
        h_learned = np.random.randint(0, 41, size=n_students)

        # Function for grades based on learning time
        def generate_grade(h_learned):
            return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

        # Generate grades
        grades = [generate_grade(lz) for lz in h_learned]

        # Create DataFrame
        df = pd.DataFrame({
            "Student": names,
            "Study time (Hours)": h_learned,
            "Grade": grades
        })
        
        # Separate features and target variable
        X = df.drop('Grade', axis=1)
        y = df['Grade']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            # Set seed
            np.random.seed(42)

            # Number of students
            n_students = 50

            # Generate names
            names = list(range(n_students))

            # Amount of hours each student spent learning (ranges from 0 to 40)
            h_learned = np.random.randint(0, 41, size=n_students)

            # Function for grades based on learning time
            def generate_grade(h_learned):
                return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

            # Generate grades
            grades = [generate_grade(lz) for lz in h_learned]

            # Create DataFrame
            df = pd.DataFrame({
                "Student": names,
                "Study time (Hours)": h_learned,
                "Grade": grades
            })

            # Separate features and target variable
            X = df.drop('Grade', axis=1)
            y = df['Grade']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
        program_in_text = False

    class MakingPredictions(VerbatimStep):
        """
    Now let our newly fitted model do some predictions for the grades in our X_test data set.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        # Set seed
        np.random.seed(42)

        # Number of students
        n_students = 50

        # Generate names
        names = list(range(n_students))

        # Amount of hours each student spent learning (ranges from 0 to 40)
        h_learned = np.random.randint(0, 41, size=n_students)

        # Function for grades based on learning time
        def generate_grade(h_learned):
            return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

        # Generate grades
        grades = [generate_grade(lz) for lz in h_learned]

        # Create DataFrame
        df = pd.DataFrame({
            "Student": names,
            "Study time (Hours)": h_learned,
            "Grade": grades
        })
        
        # Separate features and target variable
        X = df.drop('Grade', axis=1)
        y = df['Grade']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training a LinearRegression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Making predictions
        y_pred = model.predict(X_test)
        """

        requirements = "hints"

        hints = """There are no hints"""

        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            # Set seed
            np.random.seed(42)

            # Number of students
            n_students = 50

            # Generate names
            names = list(range(n_students))

            # Amount of hours each student spent learning (ranges from 0 to 40)
            h_learned = np.random.randint(0, 41, size=n_students)

            # Function for grades based on learning time
            def generate_grade(h_learned):
                return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

            # Generate grades
            grades = [generate_grade(lz) for lz in h_learned]

            # Create DataFrame
            df = pd.DataFrame({
                "Student": names,
                "Study time (Hours)": h_learned,
                "Grade": grades
            })

            # Separate features and target variable
            X = df.drop('Grade', axis=1)
            y = df['Grade']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            #Making predictions
            y_pred = model.predict(X_test)

        program_in_text = False

    class Exploration(VerbatimStep):
            """
        After our model predicted some grades based on the `Study time` of our students, we will now look at the predicted values and the corresponding study time.

            __copyable__
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            # Set seed
            np.random.seed(42)

            # Number of students
            n_students = 50

            # Generate names
            names = list(range(n_students))

            # Amount of hours each student spent learning (ranges from 0 to 40)
            h_learned = np.random.randint(0, 41, size=n_students)

            # Function for grades based on learning time
            def generate_grade(h_learned):
                return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

            # Generate grades
            grades = [generate_grade(lz) for lz in h_learned]

            # Create DataFrame
            df = pd.DataFrame({
                "Student": names,
                "Study time (Hours)": h_learned,
                "Grade": grades
            })

            # Separate features and target variable
            X = df.drop('Grade', axis=1)
            y = df['Grade']

            # Splitting the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training a LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)
            
            # Create new column in our X_test DataFrame for our predictions
            X_test['Predicted Grade'] = y_pred
            
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

                # Set seed
                np.random.seed(42)

                # Number of students
                n_students = 50

                # Generate names
                names = list(range(n_students))

                # Amount of hours each student spent learning (ranges from 0 to 40)
                h_learned = np.random.randint(0, 41, size=n_students)

                # Function for grades based on learning time
                def generate_grade(h_learned):
                    return np.clip(5.0 - h_learned * 0.1 + np.random.normal(0, 0.5), 1.0, 5.0).round(1)

                # Generate grades
                grades = [generate_grade(lz) for lz in h_learned]

                # Create DataFrame
                df = pd.DataFrame({
                    "Student": names,
                    "Study time (Hours)": h_learned,
                    "Grade": grades
                })

                # Separate features and target variable
                X = df.drop('Grade', axis=1)
                y = df['Grade']

                # Splitting the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Training a LinearRegression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                #Making predictions
                y_pred = model.predict(X_test)
                
                # Create new column in our X_test DataFrame for our predictions
                X_test['Predicted Grade'] = y_pred

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
    The dataset has been adjusted to reflect a scenario involving students who either pass or fail a test.
    The features include age, study hours per week, attendance rate, number of courses taken, assignments completed, test scores, and a target variable indicating whether a student passed the test.
    Load the dataset and print the first few lines to get an impression.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
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
    Split your data into training and test sets using `train_test_split`. Use 80% of the data as training data.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
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


    class TrainRandomForest(VerbatimStep):
        """
    Now train a simple random forest. Before you do so, preprocess your data using StandardScaler.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

        # Separate features and target variable
        X = data.drop('passed', axis=1)
        y = data['passed']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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

    class TrainLogisticRegression(VerbatimStep):
        """
    In this step you should train a logistic regression.
    To directly apply your knowledge, replace the “?” with the correct code.
    
        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

        # Separate features and target variable
        X = data.drop('passed', axis=1)
        y = data['passed']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a LogisticRegression model
        model = ?()
        model.fit(X_train, y_train)
        """
        
        requirements = "hints"

        hints = [
            "Remember what the name of the function was.",
        ]
        
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            #Previous steps
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')
            X = data.drop('Change', axis=1)
            y = data['Change']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a LogisticRegression model
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
        program_in_text = False
        
        
    class TrainLinearRegression(VerbatimStep):
        """
    In this step you should train a linear regression.
    To directly apply your knowledge, replace the “?” with the correct code.
    
        __copyable__
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import ?Regression
        import pyodide_http

        pyodide_http.patch_all()  # Necessary for downloading

        # Load data using pandas
        data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/student_classification_dataset.csv')

        # Separate features and target variable
        X = data.drop('passed', axis=1)
        y = data['passed']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Training a LinearRegression model
        model = ?()
        model.fit(X_train, y_train)
        """
        
        requirements = "hints"

        hints = [
            "You have to import the correct package from sklearn.",
            "Remember what the name of the function was that estimates a linear regression.",
        ]
        
        def program(self):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression
            import pyodide_http

            pyodide_http.patch_all()  # Necessary for downloading

            #Previous steps
            data = pd.read_csv('https://raw.githubusercontent.com/aoberm/futurecoder/master/Datasets/Customer.csv')
            X = data.drop('Change', axis=1)
            y = data['Change']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Training a LogisticRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
        program_in_text = False
        
        
    final_text = """
        Good job! You have learned to fit different models using Sklearn. Now you can apply your knowledge to various data analysis tasks!
    """
