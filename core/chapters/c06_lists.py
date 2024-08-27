# flake8: NOQA E501
import ast
import random
from textwrap import dedent
from typing import List

from core.exercises import generate_list, generate_string
from core.text import (
    ExerciseStep,
    MessageStep,
    Page,
    Step,
    VerbatimStep,
    search_ast,
    Disallowed,
)


class IntroducingLists(Page):
    class first_list(VerbatimStep):
        """
It's time to learn about a powerful new type of value called *lists*. Here's an example:

__program_indented__
        """

        auto_translate_program = False

        def program(self):
            words = ['This', 'is', 'a', 'list']

            for word in words:
                print(word)

    class can_contain_anything(VerbatimStep):
        """
A list is a *sequence* (an ordered collection/container) of any number of values.
The values are often referred to as *elements*.
They can be anything: numbers, strings, booleans, even lists! They can also be a mixture of types.

To create a list directly, like above:

1. Write some square brackets: `[]`
2. If you don't want an empty list, write some expressions inside to be the elements.
3. Put commas (`,`) between elements to separate them.

Here's another example of making a list:

__program_indented__
        """

        def program(self):
            x = 1
            things = ['Hello', x, x + 3]
            print(things)

    class numbers_sum(VerbatimStep):
        """
As you saw above, lists are *iterable*, meaning you can iterate over them with a `for loop`.
Here's a program that adds up all the numbers in a list:

__program_indented__
        """

        def program(self):
            numbers = [3, 1, 4, 1, 5, 9]

            total = 0
            for number in numbers:
                total += number

            print(total)

    class strings_sum(ExerciseStep):
        """
Now modify the program so that it can add up a list of strings instead of numbers.
For example, given:

    __no_auto_translate__
    words = ['This', 'is', 'a', 'list']

it should print:

    __no_auto_translate__
    Thisisalist
        """

        hints = """
This is very similar to the exercises you've done building up strings character by character.
The solution is very similar to the program that adds numbers.
In fact, what happens if you try running that program with a list of strings?
The problem is that 0. You can't add 0 to a string because numbers and strings are incompatible.
Is there a similar concept among strings to 0? A blank initial value?
"""

        def solution(self, words: List[str]):
            total = ''
            for word in words:
                total += word

            print(total)

        tests = [
            (['This', 'is', 'a', 'list'], 'Thisisalist'),
            (['The', 'quick', 'brown', 'fox', 'jumps'], 'Thequickbrownfoxjumps'),
        ]

    class strings_sum_bonus(ExerciseStep):
        """
Excellent!

If you'd like, you can just continue to the [next page](#BuildingNewLists) now.

For an optional bonus challenge: extend the program to insert a separator string *between* each word.
For example, given

    __no_auto_translate__
    words = ['This', 'is', 'a', 'list']
    separator = ' - '

it would output:

    __no_auto_translate__
    This - is - a - list
        """

        hints = """
This is similar to the previous exercise. You can start with your solution from that.
This exercise doesn't require anything fancy and the final solution can be quite simple. But it's tricky to get it right and you need to think about the approach carefully.
In each iteration, in addition to a word in the list, you also have to add the separator.
But you don't want to add the separator after adding the last word in the list.
Unfortunately there is no "subtraction" with strings; you can't add the last separator then remove it.
Let's back up. The final result should contain each word, and `n - 1` separators, where `n` is the number of words.
So you want to add a separator in every iteration except one.
You can skip adding the separator in one particular iteration using an `if` statement.
Later on you will learn a way to iterate over a list and check if you're in the last iteration, but right now you have no way of doing that.
However, the iteration you skip doesn't have to be the last one!
You *can* write a program that checks if you're in the *first* iteration of a loop.
Just make a boolean variable to keep track of this. No need for any comparison operators or numbers.
We looked at programs that did something like this [here](#UnderstandingProgramsWithSnoop).
So if you only skip adding the separator in the first iteration, you will have `n - 1` separators. Now you just need to think carefully about how to make sure the separators are in the right place.
Forgetting the loop for a moment, you need to add the following to the string in this order: the first word, the separator, the second word, the separator, the third word, etc.
That means that in the first iteration, you just add the first word. In the second iteration, you add the separator, then the second word. In the third iteration, you add the separator, then the third word. And so on.
So inside your loop, add the separator first, add the word after.
Skip adding the separator in the first iteration by checking a boolean variable.
Create the boolean variable before the loop, then change it inside the loop.
Only change it in the loop after checking it, or you won't be able to skip the first iteration.
        """

        # TODO message: catch the "obvious solution" where the user adds the separator after the last word?

        parsons_solution = True

        def solution(self, words: List[str], separator: str):
            total = ''
            not_first = False

            for word in words:
                if not_first:
                    total += separator
                total += word
                not_first = True

            print(total)

        tests = [
            ((['This', 'is', 'a', 'list'], ' - '), 'This - is - a - list'),
            ((['The', 'quick', 'brown', 'fox', 'jumps'], '**'), 'The**quick**brown**fox**jumps'),
        ]

    final_text = """
Congratulations! That was very tricky! One solution looks like this:

    __no_auto_translate__
    words = ['This', 'is', 'a', 'list']
    separator = ' - '
    total = ''
    not_first = False

    for word in words:
        if not_first:
            total += separator
        total += word
        not_first = True

    print(total)
        """


class BuildingNewLists(Page):
    class double_numbers(ExerciseStep):
        """
Lists and strings have a lot in common.
For example, you can add two lists to combine them together into a new list.
You can also create an empty list that has no elements.
Check for yourself:

    __copyable__
    numbers = [1, 2] + [3, 4]
    print(numbers)
    new_numbers = []
    new_numbers += numbers
    new_numbers += [5]
    print(new_numbers)

With that knowledge, write a program which takes a list of numbers
and prints a list where each number has been doubled. For example, given:

    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

it would print:

    [6, 2, 8, 2, 10, 18, 4, 12, 10]
        """

        hints = """
Remember that you can multiply numbers using `*`.
This program is structurally very similar to the programs you've written to build up strings character by character.
Make a new list, and then build it up element by element in a for loop.
Start with an empty list.
You can make a list with one element `x` by just writing `[x]`.
You can add an element to a list by adding a list containing one element.
        """

        def solution(self, numbers: List[int]):
            double = []
            for number in numbers:
                double += [number * 2]
            print(double)

        tests = [
            ([3, 1, 4, 1, 5, 9, 2, 6, 5], [6, 2, 8, 2, 10, 18, 4, 12, 10]),
            ([0, 1, 2, 3], [0, 2, 4, 6]),
        ]

    class filter_numbers(ExerciseStep):
        """
Great!

When you want to add a single element to the end of a list, instead of:

    some_list += [element]

it's actually more common to write:

    some_list.append(element)

There isn't really a big difference between these, but `.append`
will be more familiar and readable to most people.

Now use `.append` to write a program which takes a list of numbers and
prints a new list containing only the numbers bigger than 5.

For example, given:

    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

it would print:

    [9, 6]
        """

        hints = """
This is very similar to the previous exercise.
The difference is that sometimes you should skip appending to the new list.
Use an `if` statement.
Use a comparison operator to test if a number is big enough to add.
        """

        disallowed = Disallowed(
            ast.AugAssign,
            message="Well done, that's correct! However, you should use `.append()` instead of `+=`.",
        )

        parsons_solution = True

        def solution(self, numbers: List[int]):
            big_numbers = []
            for number in numbers:
                if number > 5:
                    big_numbers.append(number)
            print(big_numbers)

        tests = [
            ([3, 1, 4, 1, 5, 9, 2, 6, 5], [9, 6]),
            ([0, 2, 4, 6, 8, 10], [6, 8, 10]),
        ]

    final_text = """
Fantastic! We're making great progress.
"""

class GettingElementsAtPositionExercises(Page):
    title = "Exercises with `range()` and `len()`"

    class index_exercise(ExerciseStep):
        """
Given a list `things` and a value `to_find`,
print the first index of `to_find` in the list, i.e. the lowest number `i` such that
`things[i]` is `to_find`. For example, for

    __no_auto_translate__
    things = ['on', 'the', 'way', 'to', 'the', 'store']
    to_find = 'the'

your program should print `1`.

You can assume that `to_find` appears at least once.
        """

        hints = """
You will need to look at all the possible indices of `things` and check which one is the answer.
To look at all possible indices, you will need a loop over `range(len(things))`.
To check if an index is the answer, you will need to use:
- `if`
- the index in a subscript
- `==`
Since you're looking for the first index, you need to stop the loop once you find one.
You learned how to stop a loop in the middle recently.
You need to use `break`.
        """

        class all_indices(MessageStep, ExerciseStep):
            """
            You're almost there! However, this prints all the indices,
            not just the first one.
            """

            def solution(self, things, to_find):
                for i in range(len(things)):
                    if to_find == things[i]:
                        print(i)

            tests = [
                ((['on', 'the', 'way', 'to', 'the', 'store'], 'the'), "1\n4"),
                (([0, 1, 2, 3, 4, 5, 6, 6], 6), "6\n7"),
            ]

        class last_index(MessageStep, ExerciseStep):
            """
            You're almost there! However, this prints the *last* index,
            not the first one.
            """

            def solution(self, things, to_find):
                answer = None
                for i in range(len(things)):
                    if to_find == things[i]:
                        answer = i
                print(answer)

            tests = [
                ((['on', 'the', 'way', 'to', 'the', 'store'], 'the'), 4),
                (([0, 1, 2, 3, 4, 5, 6, 6], 6), 7),
            ]

        def solution(self, things, to_find):
            for i in range(len(things)):
                if to_find == things[i]:
                    print(i)
                    break

        tests = [
            ((['on', 'the', 'way', 'to', 'the', 'store'], 'the'), 1),
            (([0, 1, 2, 3, 4, 5, 6, 6], 6), 6),
        ]

        @classmethod
        def generate_inputs(cls):
            things = generate_list(str)
            to_find = generate_string()
            things += [to_find] * random.randint(1, 3)
            random.shuffle(things)
            return dict(
                things=things,
                to_find=to_find,
            )

    class zip_exercise(ExerciseStep):
        """
Nice!

By the way, indexing and `len()` also work on strings. Try them out in the shell.

Here's another exercise. Given two strings of equal length, e.g:

    __no_auto_translate__
    string1 = 'Hello'
    string2 = 'World'

print them vertically side by side, with a space between each character:

    H W
    e o
    l r
    l l
    o d
        """

        hints = """
Did you experiment with indexing and `len()` with strings in the shell?
Forget loops for a moment. How would you print just the first line, which has the first character of each of the two strings?
In the second line you want to print the second character of each string, and so on.
You will need a `for` loop.
You will need indexing (subscripting).
You will need `range`.
You will need `len`.
You will need `+`.
You will need to index both strings.
You will need to pass the same index to both strings each time to retrieve matching characters.
"""

        def solution(self, string1, string2):
            for i in range(len(string1)):
                char1 = string1[i]
                char2 = string2[i]
                print(char1 + ' ' + char2)

        tests = {
            ("Hello", "World"): dedent("""\
                    H W
                    e o
                    l r
                    l l
                    o d
                    """),
            ("Having", "ablast"): dedent("""\
                    H a
                    a b
                    v l
                    i a
                    n s
                    g t
                    """),
        }

        @classmethod
        def generate_inputs(cls):
            length = random.randrange(5, 11)
            return dict(
                string1=generate_string(length),
                string2=generate_string(length),
            )

    class zip_longest_exercise(ExerciseStep):
        """
Incredible!

Your solution probably looks something like this:

    for i in range(len(string1)):
        char1 = string1[i]
        char2 = string2[i]
        print(char1 + ' ' + char2)

This doesn't work so well if the strings have different lengths.
In fact, it goes wrong in different ways depending on whether `string1` or `string2` is longer.
Your next challenge is to fix this problem by filling in 'missing' characters with spaces.

For example, for:

    __no_auto_translate__
    string1 = 'Goodbye'
    string2 = 'World'

output:

    G W
    o o
    o r
    d l
    b d
    y
    e

and for:

    __no_auto_translate__
    string1 = 'Hello'
    string2 = 'Elizabeth'

output:

    H E
    e l
    l i
    l z
    o a
      b
      e
      t
      h
        """

        hints = [
            "The solution has the same overall structure and "
            "essential elements of the previous solution, "
            "but it's significantly longer and will require "
            "a few additional ideas and pieces.",
            dedent("""
            In particular, it should still contain something like:

                for i in range(...):
                    ...
                    print(char1 + ' ' + char2)
            """),
            "What should go inside `range()`? Neither `len(string1)` nor `len(string2)` is good enough.",
            "You want a loop iteration for every character in the longer string.",
            "That means you need `range(<length of the longest string>)`",
            "In other words you need to find the biggest of the two values "
            "`len(string1)` and `len(string2)`. You've already done an exercise like that.",
            "Once you've sorted out `for i in range(...)`, `i` will sometimes be too big "
            "to be a valid index for both strings. You will need to check if it's too big before indexing.",
            "Remember, the biggest valid index for `string1` is `len(string1) - 1`. "
            "`len(string1)` is too big.",
            "You will need two `if` statements, one for each string.",
            "You will need to set e.g. `char1 = ' '` when `string1[i]` is not valid.",
        ]

        # TODO message: catch user writing string1 < string2 instead of comparing lengths

        parsons_solution = True

        def solution(self, string1, string2):
            length1 = len(string1)
            length2 = len(string2)

            if length1 > length2:
                length = length1
            else:
                length = length2

            for i in range(length):
                if i < len(string1):
                    char1 = string1[i]
                else:
                    char1 = ' '

                if i < len(string2):
                    char2 = string2[i]
                else:
                    char2 = ' '

                print(char1 + ' ' + char2)

        tests = {
            ("Goodbye", "World"): dedent("""\
                    G W
                    o o
                    o r
                    d l
                    b d
                    y
                    e
                    """),
            ("Hello", "Elizabeth"): dedent("""\
                    H E
                    e l
                    l i
                    l z
                    o a
                      b
                      e
                      t
                      h
                    """),
        }

        @classmethod
        def generate_inputs(cls):
            length1 = random.randrange(5, 11)
            length2 = random.randrange(12, 20)
            if random.choice([True, False]):
                length1, length2 = length2, length1
            return dict(
                string1=generate_string(length1),
                string2=generate_string(length2),
            )

    final_text = """
Magnificent! Take a break, you've earned it!
    """


class CallingFunctionsTerminology(Page):
    title = "Terminology: Calling functions and methods"

    class print_functions(VerbatimStep):
        """
It's time to expand your vocabulary some more.

`print` and `len` are ***functions***. See for yourself:

__program_indented__
        """

        def program(self):
            print(len)
            print(print)

    class introducing_callable(VerbatimStep):
        """
An expression like `len(things)` or `print(things)` is a function ***call*** - when you write that, you are ***calling*** the function `len` or `print`. The fact that this is possible means that functions are ***callable***:

__program_indented__
        """

        def program(self):
            print(callable(len))

    class not_callable(VerbatimStep):
        """
Most things are not callable, so trying to call them will give you an error:

__program_indented__
        """

        # noinspection PyCallingNonCallable
        def program(self):
            f = 'a string'
            print(callable(f))
            f()

    class print_returns_none(VerbatimStep):
        """
In the call `len(things)`, `things` is an ***argument***. Sometimes you will also see the word ***parameter***, which means basically the same thing as argument. It's a bit like you're giving the argument to the function - specifically we say that the argument `things` is *passed* to `len`, and `len` *accepts* or *receives* the argument.

`len(things)` will evaluate to a number such as 3, in which case we say that `len` ***returned*** 3.

All calls have to return something...even if it's nothing. For example, `print`'s job is to display something on screen, not to return a useful value. So it returns something useless instead:

__program_indented__
        """

        # noinspection PyNoneFunctionAssignment
        def program(self):
            things = [1, 2, 3]
            length = len(things)
            printed = print(length)
            print(printed)

    class len_of_none(VerbatimStep):
        """
`None` is a special 'null' value which can't do anything interesting. It's a common placeholder that represents the lack of a real useful value. Functions that don't want to return anything return `None` by default. If you see an error message about `None` or `NoneType`, it often means you assigned the wrong thing to a variable:

__program_indented__
        """

        # noinspection PyNoneFunctionAssignment,PyUnusedLocal,PyTypeChecker
        def program(self):
            things = print([1, 2, 3])
            length = len(things)

    class methods_of_str(VerbatimStep):
        """
A ***method*** is a function which belongs to a type, and can be called on all values of that type using `.`. For example, `upper` and `lower` are methods of strings, which are called with e.g. `word.upper()`:

__program_indented__
        """

        def program(self):
            word = 'Hello'
            print(word.upper)
            print(word.upper())

    class no_append_for_str(VerbatimStep):
        """
Another example is that `append` is a method of lists. But you can't use `.upper` on a list or `.append` on a string:

__program_indented__
        """

        # noinspection PyUnresolvedReferences
        def program(self):
            word = 'Hello'
            word.append('!')

    final_text = """
    The word 'attribute' in the error message refers to the use of `.` - the error actually comes just from `word.append`, without even a call.
    """

