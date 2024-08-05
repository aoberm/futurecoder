# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class Anleitung(Page):
    title = "Anleitung"

    class CodeNurCopyPastenUndAusführen(VerbatimStep):
        """
     Einfacher Text um Aufgabe zu beschreiben: Kopiere den Code in die Konsole und führe ihn aus!
     Wenn man 'Text' hinterlegen will dann schreib man ihn im Code zwischen zwei '.

    __program_indented__

        Wenn man Text in die Konsole schreiben will, rückt man ihn ein.
        """

        # das program wird dann oben im Text an die Stelle wo "__program_indented__" steht gesetzt.
        program = "print('Hello Obi')"

        #wenn es hints gibt braucht es requirements
        requirements = "Here you can write requirements"

        # Hier werden Hinweise eigefügt
        hints = [
            "Hint1",
            "Hint2",
        ]

        # Als Solutions wird hier einfach das oben eingegebene Programm angezeigt.


    class Auswahlmöglichkeiten(VerbatimStep):
        """
    Text

    __program_indented__

    Zeigt Wahlmögichkeiten an

        """

        predicted_output_choices = [
            "Welcome\n"
            "Welcome to Python!",
            "Welcome!\n"
            "Welcome to Python!",
            "Welcome\n"
            "Welcometo Python!",
        ]

        def program(self):
            welcome = 'Welcome'
            print(welcome)
            python = welcome + 'to Python!'
            print(python)


    final_text = """
Hier muss finaler Text stehen.
"""

