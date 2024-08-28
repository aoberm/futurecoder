# flake8: NOQA E501
import ast
from contextlib import suppress
from random import randint
from typing import List

from core import translation as t
from core.exercises import generate_string
from core.text import (
    ExerciseStep,
    MessageStep,
    Page,
    VerbatimStep,
    Disallowed,
    search_ast,
    Step,
)


class IntroducingNestedLoops(Page):
    class first_nested_loop(VerbatimStep):
        """
You've seen that the indented body of an `if` or a loop can contain any kind of statement, including more `if` statements and loops. In particular a loop can contain another loop. Here's an example:

__program_indented__

This is called a *nested loop*. Nothing about it is really new, it's just worth understanding properly because it can be very useful for writing interesting programs.
        """

        def program(self):
            for letter in "ABC":
                print(letter)
                for number in range(4):
                    print(f'{letter} {number}')
                print('---')

        translate_output_choices = False
        predicted_output_choices = [
            """\
A 0
A 1
A 2
A 3
---
B 0
B 1
B 2
B 3
---
C 0
C 1
C 2
C 3
---
""", """\
A
A 0
A 1
A 2
A 3
---
B
B 0
B 1
B 2
B 3
---
C
C 0
C 1
C 2
C 3
---
""", """\
A 1
A 2
A 3
A 4
---
B 1
B 2
B 3
B 4
---
C 1
C 2
C 3
C 4
---
""", """\
A
B
C
---
A 0
B 0
C 0
---
A 1
B 1
C 1
---
A 2
B 2
C 2
---
A 3
B 3
C 3
"""
        ]



    final_text = """
You have mastered nested lists and how to combine them with nested loops.
Brilliant! You now have extremely powerful programming tools in your tool belt.
    """
