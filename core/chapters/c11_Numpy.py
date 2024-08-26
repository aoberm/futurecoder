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
    Here, np is a common alias used for NumPy to make the code more concise.

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
        print('array_1d: ')
        print(array_1d)

        # Creating a 2D array
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        print('array_2d: ')
        print(array_2d)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 1D array
            array_1d = np.array([1, 2, 3, 4, 5])
            print('array_1d: ')
            print(array_1d)

            # Creating a 2D array
            array_2d = np.array([[1, 2, 3], [4, 5, 6]])
            print('array_2d: ')
            print(array_2d)

        program_in_text = False

    class ArrayProperties1(VerbatimStep):
        """
    Once you have created an array, you can check its properties. First display the shape and dimension of the array.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array : ')
        print(array)

        # Shape of the array (rows, columns)
        shape = array.shape
        print('shape : ')
        print(shape)

        # Number of dimensions
        dimensions = array.ndim
        print('dimensions : ')
        print(dimensions)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array : ')
            print(array)

            # Shape of the array (rows, columns)
            shape = array.shape
            print('shape : ')
            print(shape)

            # Number of dimensions
            dimensions = array.ndim
            print('dimensions : ')
            print(dimensions)

        program_in_text = False


    class ArrayProperties2(VerbatimStep):
        """
    Second, print the number of elements of the array.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array : ')
        print(array)

        # Number of elements
        size = array.size
        print('size : ')
        print(size)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array : ')
            print(array)

            # Number of elements
            size = array.size
            print('size : ')
            print(size)

        program_in_text = False


    class ArrayProperties3(VerbatimStep):
        """
    Finally, investigate the data type of the elements of the array.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array : ')
        print(array)

        # Data type of the elements
        dtype = array.dtype
        print('dtype : ')
        print(dtype)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array : ')
            print(array)

            # Data type of the elements
            dtype = array.dtype
            print('dtype : ')
            print(dtype)

        program_in_text = False

    final_text = """
    Great!
    Moving on to more advanced operations in NumPy on the next page will allow you to harness the full power of this library.
"""

class NumpyInPythonAdvanced(Page):
    title = "Numpy in Python (advanced)"

    class ArrayInitialization(VerbatimStep):
        """
    Let's explore some advanced operations in NumPy.
    NumPy provides several methods to initialize arrays with specific values or structures:

        __copyable__
        import numpy as np

        # Array of zeros
        zeros_array = np.zeros((3, 3))
        print('zeros_array: ')
        print(zeros_array)

        # Array of ones
        ones_array = np.ones((2, 4))
        print('ones_array: ')
        print(ones_array)

        # Identity matrix
        identity_matrix = np.eye(3)
        print('identity_matrix: ')
        print(identity_matrix)

        # Array with random values
        random_array = np.random.random((2, 3))
        print('random_array: ')
        print(random_array)

        # Array with a range of values
        range_array = np.arange(0, 10, 2)
        print('range_array: ')
        print(range_array)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Array of zeros
            zeros_array = np.zeros((3, 3))
            print('zeros_array: ')
            print(zeros_array)

            # Array of ones
            ones_array = np.ones((2, 4))
            print('ones_array: ')
            print(ones_array)

            # Identity matrix
            identity_matrix = np.eye(3)
            print('identity_matrix: ')
            print(identity_matrix)

            # Array with random values
            random_array = np.random.random((2, 3))
            print('random_array: ')
            print(random_array)

            # Array with a range of values
            range_array = np.arange(0, 10, 2)
            print('range_array: ')
            print(range_array)

        program_in_text = False

    class IndexingAndSlicing(VerbatimStep):
        """
    You can access elements in an array using indexing and slicing.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array: ')
        print(array)

        # Accessing a single element
        element = array[1, 2]  # Element at second row, third column
        print('element: ')
        print(element)

        # Slicing a subarray
        subarray = array[0:2, 1:3]  # First two rows and columns 2-3
        print('subarray: ')
        print(subarray)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array: ')
            print(array)

            # Accessing a single element
            element = array[1, 2]  # Element at second row, third column
            print('element: ')
            print(element)

            # Slicing a subarray
            subarray = array[0:2, 1:3]  # First two rows and columns 2-3
            print('subarray: ')
            print(subarray)

        program_in_text = False

    final_text = """
    Great!
    Let's dive into some basic operations that you can perform on NumPy arrays.
"""


class NumpyMatrixOperations(Page):
    title = "Matrix Operations in Numpy"

    class BasicOperations(VerbatimStep):
        """
    Let's focus on basic matrix operations in NumPy.
    Understanding how to work with matrices is crucial for many applications in data science, machine learning, and scientific computing.

    NumPy allows you to perform element-wise operations easily.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array: ')
        print(array)

        # Element-wise addition
        sum_array = array + 10
        print('sum_array: ')
        print(sum_array)

        # Element-wise multiplication
        product_array = array * 2
        print('product_array: ')
        print(product_array)

        # Element-wise square
        squared_array = array ** 2
        print('squared_array: ')
        print(squared_array)

        # Mathematical operations
        mean_value = np.mean(array)
        print('mean_value: ')
        print(mean_value)
        sum_value = np.sum(array)
        print('sum_value: ')
        print(sum_value)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array: ')
            print(array)

            # Element-wise addition
            sum_array = array + 10
            print('sum_array: ')
            print(sum_array)

            # Element-wise multiplication
            product_array = array * 2
            print('product_array: ')
            print(product_array)

            # Element-wise square
            squared_array = array ** 2
            print('squared_array: ')
            print(squared_array)

            # Mathematical operations
            mean_value = np.mean(array)
            print('mean_value: ')
            print(mean_value)
            sum_value = np.sum(array)
            print('sum_value: ')
            print(sum_value)

        program_in_text = False

    class Reshaping(VerbatimStep):
        """
    You can change the shape of an array.

        __copyable__
        import numpy as np

        # Creating a 1D array
        array_1d = np.array([1, 2, 3, 4, 5])
        print('array_1d: ')
        print(array_1d)

        # Reshaping a 1D array to 2D
        reshaped_array = np.reshape(array_1d, (5, 1))
        print('reshaped_array: ')
        print(reshaped_array)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 1D array
            array_1d = np.array([1, 2, 3, 4, 5])
            print('array_1d: ')
            print(array_1d)

            # Reshaping a 1D array to 2D
            reshaped_array = np.reshape(array_1d, (5, 1))
            print('reshaped_array: ')
            print(reshaped_array)

        program_in_text = False



    class Transposing(VerbatimStep):
        """
    You can change the transpose the array.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        print('array_2d: ')
        print(array_2d)

        # Transposing a 2D array
        transposed_array = array_2d.T
        print('transposed_array: ')
        print(transposed_array)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array_2d = np.array([[1, 2, 3], [4, 5, 6]])
            print('array_2d: ')
            print(array_2d)

            # Transposing a 2D array
            transposed_array = array_2d.T
            print('transposed_array: ')
            print(transposed_array)

        program_in_text = False

    class MatrixOperations(VerbatimStep):
        """
    For linear algebra, NumPy provides matrix operations.

        __copyable__
        import numpy as np

        # Creating a 2D array
        array = np.array([[1, 2, 3], [4, 5, 6]])
        print('array: ')
        print(array)

        # Matrix multiplication
        matrix_product = np.dot(array, array.T)
        print('matrix_product: ')
        print(matrix_product)

        # Determinant of a matrix
        determinant = np.linalg.det(array[:2, :2])
        print('determinant: ')
        print(determinant)

        # Inverse of a matrix
        inverse_matrix = np.linalg.inv(array[:2, :2])
        print('inverse_matrix: ')
        print(inverse_matrix)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            # Creating a 2D array
            array = np.array([[1, 2, 3], [4, 5, 6]])
            print('array: ')
            print(array)

            # Matrix multiplication
            matrix_product = np.dot(array, array.T)
            print('matrix_product: ')
            print(matrix_product)

            # Determinant of a matrix
            determinant = np.linalg.det(array[:2, :2])
            print('determinant: ')
            print(determinant)

            # Inverse of a matrix
            inverse_matrix = np.linalg.inv(array[:2, :2])
            print('inverse_matrix: ')
            print(inverse_matrix)

        program_in_text = False

    final_text = """
    Good job!
    The library is vast, so we recommend exploring its documentation to discover more advanced features and functions.
"""




class PracticeNumpy(Page):
    title = "Practice: Numpy in Python"

    class importNumpy(VerbatimStep):
        """
    This quiz is designed to test your understanding of fundamental NumPy operations.
    Each question requires you to perform a common task in NumPy. To demonstrate your knowledge, replace each "?" with the correct code snippet.
    Your goal is to accurately fill in the missing code to complete the task.

    Begin by importing the NumPy as np.

        __copyable__
        import ? as ?

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

        program_in_text = False


    '''class Quiz1(VerbatimStep):
        """
        Can you guess the correct output for the following code snippet? Run the code.

            __copyable__
            import numpy as np
            array_1d = np.arange(10, 16)
            print(array_1d)

        """
        requirements = "hints"

        hints = """ test """

        predicted_output_choices = [
            "[10 11 12 13 14 15]",
            "[9 10 11 12 13 14 15 16]",
            "[9 10 11 12 13 14 15]",
            "[10 11 12 13 14 15 16]"
        ]

        def program(self):
            import numpy as np
            array_1d = np.arange(10, 16)
            print(array_1d)

        program_in_text = False'''


    class ArrayCreation(VerbatimStep):
        """
    Create a 1D NumPy array containing the integers from 10 to 20 (inclusive).
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        array_1d = np.?(10, ?)
        print('array_1d: ')
        print(array_1d)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np
            array_1d = np.arange(10, 21)
            print('array_1d: ')
            print(array_1d)

        program_in_text = False

    '''class Quiz2(VerbatimStep):
        """
        Can you guess the correct output for the following code snippet? Run the code.

            __copyable__
            import numpy as np
            array_2d = np.array([[1, 2], [3, 4], [5, 6]])
            print(array_2d.shape)

        """
        requirements = "hints"

        hints = """ test """

        predicted_output_choices = [
            '(3, 2)',
            '(2, 3)',
            '(6,)',
            '(3, 3)',
        ]

        def program(self):
            import numpy as np
            array_2d = np.array([[1, 2], [3, 4], [5, 6]])
            print(array_2d.shape)

        program_in_text = False'''

    class ShapeAndSize(VerbatimStep):
        """
    Given the array_2d, find out the shape, size, and the number of dimensions of the array.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])

        shape = array_2d.?
        size = array_2d.?
        dimensions = array_2d.?
        print(f"Shape: {shape}, Size: {size}, Dimensions: {dimensions}")

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np
            array_2d = np.array([[1, 2, 3], [4, 5, 6]])

            shape = array_2d.shape
            size = array_2d.size
            dimensions = array_2d.ndim
            print(f"Shape: {shape}, Size: {size}, Dimensions: {dimensions}")

        program_in_text = False

    '''
    class Quiz2(VerbatimStep):
        """
        Can you guess the correct output for the following code snippet? Run the code.

            __copyable__
            import numpy as np
            array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            subarray = array_2d[1:, 1:]
            print(subarray)

        """
        requirements = "hints"

        hints = """ test """

        predicted_output_choices = [
            '[[4 5], [7 8]]',
            '[[4 5 6], [7 8 9]]',
            '[[5 6], [7 8]]',
            '[[5 6] [8 9]]'
        ]

        def program(self):
            import numpy as np
            array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            subarray = array_2d[1:, 1:]
            print(subarray)

        program_in_text = False
    '''

    class Slicing(VerbatimStep):
        """
    From the following 2D array array_2d, extract the subarray containing the elements [[4, 5], [7, 8]].
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        subarray = array_2d[?, ?]
        print('subarray:')
        print(subarray)
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

            subarray = array_2d[1:3, 0:2]
            print('subarray:')
            print(subarray)

        program_in_text = False

    class ElementwiseOperations(VerbatimStep):
        """
    Given the array array_1d, square each element in the array.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        array_1d = np.array([1, 2, 3, 4, 5])

        squared_array = array_1d ?? ?
        print('squared_array: ')
        print(squared_array)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            array_1d = np.array([1, 2, 3, 4, 5])

            squared_array = array_1d ** 2
            print('squared_array: ')
            print(squared_array)

        program_in_text = False

    class Create7Array(VerbatimStep):
        """
    Create a 2D array of shape (3, 3) where all elements are 7.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        array_2d = np.full((?, ?), ?)
        print('array_2d: ')
        print(array_2d)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            array_2d = np.full((3, 3), 7)
            print('array_2d: ')
            print(array_2d)

        program_in_text = False

    class Reshaping(VerbatimStep):
        """
    Reshape array_1d into a 3x3 matrix.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        reshaped_array = array_1d.?(3, ?)
        print('reshaped_array:')
        print(reshaped_array)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

            reshaped_array = array_1d.reshape(3, 3)
            print('reshaped_array:')
            print(reshaped_array)

        program_in_text = False

    class StatisticalOperations(VerbatimStep):
        """
    Given the array_2d, calculate the mean and the sum of all the elements in the array.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        mean_value = np.?(array_2d)
        sum_value = ?.sum(array_2d)
        print(f"Mean: {mean_value}, Sum: {sum_value}")
        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

            mean_value = np.mean(array_2d)
            sum_value = np.sum(array_2d)
            print(f"Mean: {mean_value}, Sum: {sum_value}")

        program_in_text = False

    class MatrixMultiplication(VerbatimStep):
        """
    Perform matrix multiplication on the following arrays:
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        matrix_product = np.?(?, ?)
        print('matrix_product:')
        print(matrix_product)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])

            matrix_product = np.dot(A, B)
            print('matrix_product:')
            print(matrix_product)

        program_in_text = False

    class InverseMatrix(VerbatimStep):
        """
    Find the inverse of the matrix C.
    To directly apply your knowledge, replace the “?” with the correct code.

        __copyable__
        import numpy as np

        C = np.array([[1, 2], [3, 4]])

        inverse_C = np.?.?(C)
        print('inverse_C:')
        print(inverse_C)

        """

        requirements = "hints"

        hints = """ test """

        def program(self):
            import numpy as np

            C = np.array([[1, 2], [3, 4]])

            inverse_C = np.linalg.inv(C)
            print('inverse_C:')
            print(inverse_C)

        program_in_text = False


    final_text = """
    Good job!
    This short course should give you a solid foundation to start working with NumPy.
"""








