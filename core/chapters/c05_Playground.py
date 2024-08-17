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
    Eines der zentralen Merkmal von NumPy ist sein N-dinesionales Array-Objekt oder "ndarray", ein schneller, flexibler Container für große Datenmengen in Python.
    Importiere zuerst NumPy und dann erzeuge ein kleines Array:
    
        __copyable__
        import numpy as np
        array = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
    
        """
        
        requirements = "hints"

        hints = """ Am einfachsten erzeugst du ein Array mit der *array*-Funktion. """

        def program(self):
            import numpy as np
            array = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])

        program_in_text = False
        
    class numpy_zeros(VerbatimStep):
        """
    Neben *np.array* gibt es eine Reihe weiterer Funktionen zum Anlegen neuer Arrays.
    *np.zeros* und *np.ones* erzeugen zum Beispiel Arrays aus Nullen bzw. Einsen bei vorgegebener Länge oder Form.
    Erzeuge ein Array bestehend aus 10 Nullen und ein Array bestehend aus 5 Einsen und lasse sie dir dann anzeigen:
    
        __copyable__
        import numpy as np
        
        array = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
        array0 = np.zeros(10)
        array1 = np.ones(5)
        
        print(array, array0, array1)
        """
        
        requirements = "hints"
        
        hints = """ *np.zeros(n)* erzeugt ein Array mit Länge n bestehend aus Nullen """
        
        def program(self):
            import numpy as np
        
            array = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
            array0 = np.zeros(10)
            array1 = np.ones(5)
        
            print(array, array0, array1)
        
        
        program_in_text = False
        
    final_text = """
    Sehr gut! Du kannst jetzt mit dem Paket NumPy verschiedene Arrays erzeugen. Welche Vorteile das bringt, lernst du in den nächsten Kapiteln.
"""
    
