# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep

# Pro Seite muss man eine Überklasse erstellen: class Name(page)

class AnleitungSeite1(Page):
    # Name der Seite festlegen
    title = "Anleitung Seite 1 - VerbatimStep"

    # Jede untergeordnete Klasse stellt eine Aufgabe dar, die automatisch erst aufpopt wenn man die Aufgabe davor gelöst hat.
    # Für einfache Aufgaben class name(VerbatimStep):
    class CodeNurCopyPastenUndAusführen(VerbatimStep):
        """
     Aufgabentyp 1: Kopiere den Code in die Konsole und führe ihn aus!
     Wenn man `Text` hinterlegen will dann schreibt man ihn im Code zwischen zwei `.

    __program_indented__

        Wenn man Text in die Konsole schreiben will, rückt man ihn ein.

    Wenn nicht dann nicht.
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

        # Als Solutions wird hier einfach 'program' angezeigt.
        # Um Aufgabe abzuschließen muss genau der Code von 'programm' ausgeführt werden


    class Auswahlmöglichkeiten(VerbatimStep):
        """
    Führe den Code aus und wähle eine Auswahlmöglichkeit:

    __program_indented__

    In der Konsole wird dann anstatt den Code auszuführen Wahlmöglichkeiten angezeigt, die unter 'predicted_output_choices' fetsgelegt wurden.
    Error ist auch immer eine Option. Auserdem muss die richtige Lösung auch zur Auswahl stehen. Nächste Aufgabe wird automatisch geladen, wenn man die richtige Auswahl auswählt oder 2x falsch lag.

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


    class ProgramNichtInTextZeigen(VerbatimStep):
        """
    Programm wird nicht im Text angezeigt, sondern bezieht sich auf den Code von der vorherigen Aufgabe
    Zum Beispiel: Gebe 'Hello Python' aus.
        """

        requirements = "hints"

        hints = """
        Use print()
        """

        program = "print('Hello Python')"
        # das verhindert dass program in Text angezeigt wird. Ansonsten meckert er, dass __program_indented__ fehlt.
        program_in_text = False

    final_text = """
Für jede Seite muss es einen finalen Text geben
"""


class AnleitungSeite2(Page):
    # Name der Seite festlegen
    title = "Anleitung Seite 2 - Exercise"


    # Für Aufgaben bei denen Studenten selber coden müssen class name(ExerciseStep):
    class SelbstCoden(ExerciseStep):
        """
Hier steht die Aufforderung an den Studenten selbst zu coden: Beispiel:
Gebe 'Hello World` aus. Da der Student das Problem auf verschiedene Art & Weisen Lösen kann, wird zwar Musterlösung angegeben aber die Aufgabe skipt zur nächsten wenn die Tests erfolgreich sind.
        """

        hints = """
            hint 1
    """

        parsons_solution = True

        def solution(self):
            pritn('Hello World')

        tests = {
            (): 'Hello World',
        }


    final_text = """
Hier muss wieder ein finaler Text stehen.
"""


class AnleitungSeite3(Page):
    # Name der Seite festlegen
    title = "Anleitung Seite 3 - Lösungen und Hints"


    # Für Aufgaben bei denen Studenten selber coden müssen class name(ExerciseStep):
    class SelbstCoden(ExerciseStep):
        """
Man kann die Lösungen auf verschiedenen Art und Weise anzeigen. Entweder man zeigt direkt die verdeckt Lösung an und deckt sie nach und nach auf, oder man baut einen Schritt davor ein, in der die Lösung zwar angezeigt wird, aber nicht in der richtigen Reihenfolge.

        """

        hints = [
            "Hint1",
            "Hint2",
        ]

        # Wenn das true ist dann wird Lösung in vertauschter Reihenfolge angezeigt.
        parsons_solution = True

        def solution(self):
            print('Hello World')
            print('Hello World 2')
            print('Hello World 3')


        tests = {
            (): """\
Hello World
Hello World 2
Hello World 3
""",
        }


    final_text = """
Hier muss wieder ein finaler Text stehen.
"""
