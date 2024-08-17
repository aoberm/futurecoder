# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class Playground(Page):
    title = "This is a test"
    
    class ndarray(VerbatimStep):
        """
    Eines der zentralen Merkmal von NumPY ist sein N-dinesionales Array-Objekt oder "ndarray", ein schneller, flexibler Container für große Datenmengen in Python.
    Importiere zuerst NumPy und dann erzeuge ein kleines Array:
    
        __copyable__
        import numpy as np
        data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
    
        """
        
        requirements = "hints"

        hints = """ Am einfachsten erzeugst du ein Array mit der *array*-Funktion. """

        def program(self):
            import numpy as np
            data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])

        program_in_text = False

    final_text = """
<<<<<<< HEAD
    Sehr gut! Du kannst jetzt mit dem Paket NumPy verschiedene Arrays erzeugen. Welche Vorteile das bringt, lernst du in den nächsten Kapiteln.
"""
    
=======
    Good job!
"""
>>>>>>> parent of 9dfed9b (Added some things)
