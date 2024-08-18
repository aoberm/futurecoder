# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class Playground(Page):
    title = "Arrays in NumPy"
    
    class ndarray(VerbatimStep):
        """
    Eines der zentralen Merkmal von NumPy ist sein N-dinesionales Array-Objekt oder "ndarray", ein schneller, flexibler Container für große Datenmengen in Python.
    Importiere zuerst NumPy und dann erzeuge ein kleines Array:
    
        __copyable__
        import numpy as np
        array = np.array([1.5, -0.1, 3])
    
        """
        
        requirements = "hints"

        hints = """ Am einfachsten erzeugst du ein Array mit der *array*-Funktion. """

        def program(self):
            import numpy as np
            array = np.array([1.5, -0.1, 3])

        program_in_text = False
        
        
    class numpy_zeros(VerbatimStep):
        """
    Neben `np.array` gibt es eine Reihe weiterer Funktionen zum Anlegen neuer Arrays.
    `np.zeros` und `np.ones` erzeugen zum Beispiel Arrays aus Nullen bzw. Einsen bei vorgegebener Länge oder Form.
    Erzeuge ein Array bestehend aus 10 Nullen und ein Array bestehend aus 3 Einsen und lasse sie dir dann anzeigen:
    
        __copyable__
        import numpy as np
        
        array = np.array([1.5, -0.1, 3])
        array0 = np.zeros(10)
        array1 = np.ones(3)
        
        print(array, array0, array1)
        """
        
        requirements = "hints"
        
        hints = """ `np.zeros(n)` erzeugt ein Array mit Länge *n* bestehend aus Nullen """
        
        def program(self):
            import numpy as np
        
            array = np.array([1.5, -0.1, 3])
            array0 = np.zeros(10)
            array1 = np.ones(5)
        
            print(array, array0, array1)
        
        
        program_in_text = False
        
        
    class array_maths(VerbatimStep):
        """
    Arrays sind wichtig, weil sie es dir erlauben, viele Operationen auf Daten auszuführen, ohne dass man for-Schleifen schreiben muss. Jede arithmetische Operation zwischen Arrays gleicher Größe führt ihre Arbeit elementweise durch.
    Führe mit deinen Arrays *array* und *array1* verschiedene arithmetische Oparationen durch. Angefangen mit der Substraktion. Nutze die Funktion `np.substract()` um von `array` das Array `array1` abzuziehen:
    
        __copyable__
        import numpy as np
        
        array = np.array([1.5, -0.1, 3])
        array1 = np.ones(3)
        
        np.substract(array, array1)
        """
        
        requirements = "hints"
        
        hints = """ `np.substract(a,b)` subtrahiert von Array a das Array b """
        
        def program(self):
            import numpy as np
        
            array = np.array([1.5, -0.1, 3])
            array1 = np.ones(3)

            np.substract(array, array1)
            
        
    final_text = """
    Sehr gut! Du kannst jetzt mit dem Paket NumPy verschiedene Arrays erzeugen und mathematische Oparationen durchführen.
"""
    
