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
    title = "Numpy in Python"


    final_text = """
Excellent! The solution goes like this:

    players = ['Charlie', 'Alice', 'Dylan', 'Bob']
    for i in range(len(players)):
        for j in range(len(players)):
            if i < j:
                print(f'{players[i]} vs {players[j]}')
"""


class IntroducingBirdseye(Page):
    title = "Understanding Programs with `birdseye`"

    final_text = """
Note that:

1. There's a pair of arrows next to the for loop. Click on them to navigate through the loop in time and see what happened in different iterations.
2. Code that doesn't run in an iteration because of the `if` is greyed out. The expressions within have no values because they weren't evaluated.
3. The values recorded for the expressions `vowels` and `consonants` depend on which box you look at. In the lines after the loop, they contain all the letters, but inside the loop they only contain some, and exactly how many depends on which iteration you're on.
4. In `vowels.append(letter)`, you see what the values of those variables were *at that moment*. That means that `letter` is about to be appended to `vowels` but this hasn't happened yet, so `vowels` doesn't contain `letter`.
        """


class IntroducingNestedLists(Page):


    final_text = """
Excellent! You now understand nested subscripting very well.

We can still use all the list methods and functions we learned before.
For example we can add a new word to the last sublist of `strings` with `append`,
to come after `'you'`:

    strings[1].append("today?")

After all, the sublist `strings[1]` is still a list like any other!

On the next page we will learn about looping over nested lists.
        """


class LoopingOverNestedLists(Page):

    final_text = """
You have mastered nested lists and how to combine them with nested loops.
Brilliant! You now have extremely powerful programming tools in your tool belt.
    """
