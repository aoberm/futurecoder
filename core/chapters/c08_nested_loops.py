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

    final_text = """
You have mastered nested lists and how to combine them with nested loops.
Brilliant! You now have extremely powerful programming tools in your tool belt.
    """
