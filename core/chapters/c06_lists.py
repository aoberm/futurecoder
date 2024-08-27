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



class GettingElementsAtPosition(Page):
    title = "Getting elements at a position, `range()`, and `len()`"

    class introducing_subscripting(VerbatimStep):
        """
Looping is great, but often you just want to retrieve a single element from the list at a known position.
Here's how:

__program_indented__
        """

        auto_translate_program = False

        def program(self):
            words = ['This', 'is', 'a', 'list']

            print(words[0])
            print(words[1])
            print(words[2])
            print(words[3])

    class index_error(Step):
        """
In general, you can get the element at the position `i` with `words[i]`. The operation is called *subscripting* or *indexing*, and the position is called the *index*.

You've probably noticed that the first index is 0, not 1. In programming, counting starts at 0. It seems weird, but that's how most programming languages do it, and it's generally agreed to be better.

This also means that the last index in this list of 4 elements is 3. What happens if you try getting an index greater than that?
        """

        auto_translate_program = False

        requirements = "Run something like `words[3]` but replace `3` with a bigger number."

        program = "words[4]"

        def check(self):
            return "IndexError" in self.result

    class introducing_len_and_range(VerbatimStep):
        """
There you go. `words[4]` and beyond don't exist, so trying that will give you an error.
That first program is a bit repetitive. Let's improve it with a list and a loop!

__program_indented__
        """

        auto_translate_program = False

        predicted_output_choices = ["""\
This
is
a
list
""", """\
0
1
2
3
""", """\
0
This
1
is
2
a
3
list
""", """\
This
0
is
1
a
2
list
3
""", """\
0
1
2
3
This
is
a
list
""", """\
This
is
a
list
0
1
2
3
"""
                                    ]

        def program(self):
            words = ['This', 'is', 'a', 'list']
            indices = [0, 1, 2, 3]

            for index in indices:
                print(index)
                print(words[index])

    class range_len(VerbatimStep):
        """
That's a bit better, but writing out `[0, 1, 2, ...]` isn't great, especially if it gets long.
There's a handy function `range` to do that part for you. Replace `[0, 1, 2, 3]` with `range(4)`,
i.e. `indices = range(4)`.
        """

        program_in_text = False
        requirements = "Run the same program from the previous step, but replace the second line `indices = [0, 1, 2, 3]` with `indices = range(4)`."
        auto_translate_program = False

        def program(self):
            words = ['This', 'is', 'a', 'list']
            indices = range(4)

            for index in indices:
                print(index)
                print(words[index])

    class printing_the_range(VerbatimStep):
        """
As you can see, the result is the same. Try this:

    __copyable__
    __program_indented__
        """

        predicted_output_choices = ["""\
0
1
2
3
""", """\
1
2
3
4
""", """\
[0]
[1]
[2]
[3]
""", """\
[1]
[2]
[3]
[4]
""", """\
This
is
a
list
""",
                                    ]

        def program(self):
            indices = range(4)

            print(indices[0])
            print(indices[1])
            print(indices[2])
            print(indices[3])

    class indices_out_of_bounds(VerbatimStep):
        """
Now try `__program__` in the shell.
        """

        predicted_output_choices = ["0", "1", "2", "3", "4"]

        correct_output = "Error"

        expected_code_source = "shell"

        program = "indices[4]"

    class range_almost_the_same_as_list(VerbatimStep):
        """
`range(4)` is the same thing as `[0, 1, 2, 3]` ... almost. Try `__program__` in the shell.
        """

        expected_code_source = "shell"

        program = "range(4)"

    class range_versus_list(VerbatimStep):
        """
That's probably a bit surprising. If you're curious, the `0` represents the start of the range.
`0` is the default start, so `range(4)` is equal to `range(0, 4)`.
`4` is the end of the range, but the end is always excluded, so the last value is `3`.
If you're confused now, don't worry about it.

There's a good reason for why `range(4)` is not actually a list - it makes programs faster and more efficient.
It's not worth explaining that more right now.

But you can easily convert it to a list: try `__program__` in the shell.
        """
        predicted_output_choices = [
            "range(4)",
            "range(0, 4)",
            "list(range(4))",
            "list(range(0, 4))",
            "range(0, 1, 2, 3)",
            "(0, 1, 2, 3)",
            "[0, 1, 2, 3]"
        ]

        expected_code_source = "shell"

        program = "list(range(4))"

    class using_len_first_time(VerbatimStep):
        """
That's just a demonstration to let you see a range in a more familiar form.
You should almost never actually do that.

If you're feeling overwhelmed, don't worry! All you need to know is that `range(n)`
is very similar to the list:

    [0, 1, 2, ..., n - 2, n - 1]

By the way, you can get the number of elements in a list (commonly called the *length*) using the `len` function.
Try it by running this code:

    __copyable__
    __program_indented__
        """

        auto_translate_program = False

        predicted_output_choices = ["0", "1", "2", "3", "4", "5"]

        def program(self):
            words = ['This', 'is', 'a', 'list']
            print(len(words))

    class print_last_element(ExerciseStep):
        """
Exercise: for any non-empty list `words`, print the last element. For example, if

    __no_auto_translate__
    words = ['This', 'is', 'a', 'list']

your program should print `list`.
        """

        hints = """
To access the last element of the list, you'll need the index of the last position.
If the list has 2 elements, the first element is at index 0, so the last element is at index 1.
Likewise, if the list had 3 elements, the last element would be at index 2.
Do you see a pattern between those numbers? How can you express it?
Can you come up with a general solution that works for any length?
        """

        def solution(self, words: List[str]):
            print(words[len(words) - 1])

        tests = [
            (["Python"], "Python"),
            (['Hello', 'world'], "world"),
            (['futurecoder', 'is', 'cool!'], "cool!"),
            (['This', 'is', 'a', 'list'], "list")
        ]

    class print_indices_and_words(ExerciseStep):
        """
So in general, the valid indices are:

    [0, 1, 2, ..., len(words) - 2, len(words) - 1]

Now we can fix the program from earlier to work with any list. Fill in the `...`:

    __copyable__
    __no_auto_translate__
    words = ['This', 'is', 'a', 'list']

    for index in ...:
        print(index)
        print(words[index])

For the given example value of `words` it should print:

    0
    This
    1
    is
    2
    a
    3
    list
        """

        hints = """
Remember that earlier we used `range(4)`.
This time, it should work for any list. What if the list has 5 elements, or 10?
Combine the two functions you learned!
        """

        def solution(self, words: List[str]):
            for index in range(len(words)):
                print(index)
                print(words[index])

        tests = [
                   (['Python'], """\
0
Python
                        """),
                    (['Hello', 'world'], """\
0
Hello
1
world
                    """),
                    (['futurecoder', 'is', 'cool!'], """\
0
futurecoder
1
is
2
cool!
                    """),
                    (['This', 'is', 'a', 'list'], """\
0
This
1
is
2
a
3
list
                    """),
                ]

    final_text = """
If you're still not quite comfortable with `range` and/or `len`, practice and experiment with it for a bit.
Here are some simple exercises you can try on your own if you want.

- Print the numbers from `1` to `100` inclusive.
- Print your name 100 times.
- Print each word in a list `words` except for the last one.
- Print each word in `words` in reverse order, i.e. print the last word, then the second last word, etc.
- Revisit the bonus problem at the end of the [Introducing Lists page](#IntroducingLists),
whether or not you completed it. It's now much easier with `range` and `len`!

When you're ready, continue to the next page for something a bit more challenging.
"""
