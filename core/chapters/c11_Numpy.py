# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class NumpyInPython(Page):
    title = "Numpy in Python"

    class importNumpy(VerbatimStep):
        """
    NumPy is a powerful library in Python for numerical computations.
    It provides support for large, multi-dimensional arrays and matrices,
    along with a collection of mathematical functions to operate on these arrays.
    This course will cover the most essential functions that you need to get started with NumPy.
    Begin by importing the NumPy library, commonly abbreviated as np.

        __copyable__
        import numpy as np

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

        program_in_text = False

    class CreatingArrays(VerbatimStep):
        """
    You can create a NumPy array from a Python list using the np.array() function.

        __copyable__
        import numpy as np

        # Creating a 1D array
        array_1d = np.array([1, 2, 3, 4, 5])
        print('array_1d: ', array_1d)

        # Creating a 2D array
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        print('array_2d: ', array_2d)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 1D array
            array_1d = np.array([1, 2, 3, 4, 5])
            print('array_1d: ', array_1d)

            # Creating a 2D array
            array_2d = np.array([[1, 2, 3], [4, 5, 6]])
            print('array_2d: ', array_2d)

        program_in_text = False

    class ArrayProperties(VerbatimStep):
        """
    Once you have created an array, you can check its properties.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array : ', array)

        # Shape of the array (rows, columns)
        shape = array.shape
        print('shape : ', shape)

        # Number of elements
        size = array.size
        print('size : ', size)

        # Number of dimensions
        dimensions = array.ndim
        print('dimensions : ', dimensions)

        # Data type of the elements
        dtype = array.dtype
        print('dtype : ', dtype)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array : ', array)

            # Shape of the array (rows, columns)
            shape = array.shape
            print('shape : ', shape)

            # Number of elements
            size = array.size
            print('size : ', size)

            # Number of dimensions
            dimensions = array.ndim
            print('dimensions : ', dimensions)

            # Data type of the elements
            dtype = array.dtype
            print('dtype : ', dtype)

        program_in_text = False

    class ArrayInitialization(VerbatimStep):
        """
    NumPy provides several methods to initialize arrays with specific values or structures:

        __copyable__
        import numpy as np

        # Array of zeros
        zeros_array = np.zeros((3, 3))
        print('zeros_array: ', zeros_array)

        # Array of ones
        ones_array = np.ones((2, 4))
        print('ones_array: ', ones_array)

        # Identity matrix
        identity_matrix = np.eye(3)
        print('identity_matrix: ', identity_matrix)

        # Array with random values
        random_array = np.random.random((2, 3))
        print('random_array: ', random_array)

        # Array with a range of values
        range_array = np.arange(0, 10, 2)
        print('range_array: ', range_array)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Array of zeros
            zeros_array = np.zeros((3, 3))
            print('zeros_array: ', zeros_array)

            # Array of ones
            ones_array = np.ones((2, 4))
            print('ones_array: ', ones_array)

            # Identity matrix
            identity_matrix = np.eye(3)
            print('identity_matrix: ', identity_matrix)

            # Array with random values
            random_array = np.random.random((2, 3))
            print('random_array: ', random_array)

            # Array with a range of values
            range_array = np.arange(0, 10, 2)
            print('range_array: ', range_array)

        program_in_text = False

    class IndexingAndSlicing(VerbatimStep):
        """
    You can access elements in an array using indexing and slicing.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array: ', array)

        # Accessing a single element
        element = array[1, 2]  # Element at second row, third column
        print('element: ', element)

        # Slicing a subarray
        subarray = array[0:2, 1:3]  # First two rows and columns 2-3
        print('subarray: ', subarray)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array: ', array)

            # Accessing a single element
            element = array[1, 2]  # Element at second row, third column
            print('element: ', element)

            # Slicing a subarray
            subarray = array[0:2, 1:3]  # First two rows and columns 2-3
            print('subarray: ', subarray)

        program_in_text = False

    class BasicOperations(VerbatimStep):
        """
    NumPy allows you to perform element-wise operations easily.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array: ', array)

        # Element-wise addition
        sum_array = array + 10
        print('sum_array: ', sum_array)

        # Element-wise multiplication
        product_array = array * 2
        print('product_array: ', product_array)

        # Element-wise square
        squared_array = array ** 2
        print('squared_array: ', squared_array)

        # Mathematical operations
        mean_value = np.mean(array)
        print('mean_value: ', mean_value)
        sum_value = np.sum(array)
        print('sum_value: ', sum_value)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array: ', array)

            # Element-wise addition
            sum_array = array + 10
            print('sum_array: ', sum_array)

            # Element-wise multiplication
            product_array = array * 2
            print('product_array: ', product_array)

            # Element-wise square
            squared_array = array ** 2
            print('squared_array: ', squared_array)

            # Mathematical operations
            mean_value = np.mean(array)
            print('mean_value: ', mean_value)
            sum_value = np.sum(array)
            print('sum_value: ', sum_value)

        program_in_text = False

    class ReshapingAndTransposing(VerbatimStep):
        """
    You can change the shape of an array or transpose it.

        __copyable__
        import numpy as np

        # Creating a 1D array
        array_1d = np.array([1, 2, 3, 4, 5])
        print('array_1d: ', array_1d)

        # Creating a 2D array
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        print('array_2d: ', array_2d)

        # Reshaping a 1D array to 2D
        reshaped_array = np.reshape(array_1d, (5, 1))
        print('reshaped_array: ', reshaped_array)

        # Transposing a 2D array
        transposed_array = array_2d.T
        print('transposed_array: ', transposed_array)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 1D array
            array_1d = np.array([1, 2, 3, 4, 5])
            print('array_1d: ', array_1d)

            # Creating a 2D array
            array_2d = np.array([[1, 2, 3], [4, 5, 6]])
            print('array_2d: ', array_2d)

            # Reshaping a 1D array to 2D
            reshaped_array = np.reshape(array_1d, (5, 1))
            print('reshaped_array: ', reshaped_array)

            # Transposing a 2D array
            transposed_array = array_2d.T
            print('transposed_array: ', transposed_array)

        program_in_text = False

    class MatrixOperations(VerbatimStep):
        """
    For linear algebra, NumPy provides matrix operations.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array: ', array)

        # Matrix multiplication
        matrix_product = np.dot(array, array.T)
        print('matrix_product: ', matrix_product)

        # Determinant of a matrix
        determinant = np.linalg.det(array[:2, :2])
        print('determinant: ', determinant)

        # Inverse of a matrix
        inverse_matrix = np.linalg.inv(array[:2, :2])
        print('inverse_matrix: ', inverse_matrix)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array: ', array)

            # Matrix multiplication
            matrix_product = np.dot(array, array.T)
            print('matrix_product: ', matrix_product)

            # Determinant of a matrix
            determinant = np.linalg.det(array[:2, :2])
            print('determinant: ', determinant)

            # Inverse of a matrix
            inverse_matrix = np.linalg.inv(array[:2, :2])
            print('inverse_matrix: ', inverse_matrix)

        program_in_text = False

    final_text = """
    Good job!
    This short course should give you a solid foundation to start working with NumPy.
    The library is vast, so I recommend exploring its documentation to discover more advanced features and functions.
"""










