# flake8: NOQA E501
import ast
from textwrap import dedent
from typing import List
import random

from core.exercises import assert_equal, generate_string
from core.text import ExerciseStep, Page, VerbatimStep, Disallowed, MessageStep


class Hannah(Page):
    title = "Welcome to Python"

    class FirstSteps(VerbatimStep):
        """
        Python is one of the most popular programming languages — it's been used to write millions of computer programs.

        This is a very simple Python program. Copy the code and press “Run” to see what it does.

    __program_indented__
        """
        program = "print('Hello World')"

    class SecondStep(VerbatimStep):
        """
        Try changing the text of the program and running it again.

    __program_indented__
        """
        program = "print('Hello Python')"

    class ThirdStep(VerbatimStep):

        def program(self):
            print('Hello' +  ' Python')

        """
        What do you think this program will output? Try.

    __copyable__

        """

    class FourthStep(VerbatimStep):
        """
        The + operator combines strings together.
        If you want a space between the strings, you'll need to add it to one of the strings before combining.
        Alternatively, you can list the strings in the print function using the , operator which inserts a space.

    __program_indented__
        """
        program = "print('Welcome to', 'Python!')"

    final_text = """
Good job!
In Python, there are often multiple ways to accomplish a task. In the next lesson, we'll learn how to manipulate numbers in Python.
"""



class Hannah_2(Page):
    title = "Welcome to Python 2"

    final_text = """
    Good job!
    """

class IntroducingAnd(Page):
    title = "Introducing `and`"

    class TrueAndTrue(VerbatimStep):
        """
Another boolean operator in Python is `and`.
The expression `A and B` is `True` only if BOTH `A` and `B` are `True`. Otherwise it's `False`.
Try it in the shell:

__program_indented__
        """
        program = "True and True"
        predicted_output_choices = ["True", "False"]
        expected_code_source = "shell"

    class TrueAndFalse(VerbatimStep):
        """
Good, now try:

__program_indented__

What do you expect?
        """
        program = "True and False"
        predicted_output_choices = ["True", "False"]
        expected_code_source = "shell"

    class FalseAndFalse(VerbatimStep):
        """
Finally, try:

__program_indented__
        """
        program = "False and False"
        predicted_output_choices = ["True", "False"]
        expected_code_source = "shell"

    # noinspection PyChainedComparisons
    class AndExercise(ExerciseStep):
        """
Let's practice now. Previously we wrote a function `is_valid_percentage` using `or`. Here's an example
solution:

    __copyable__
    def is_valid_percentage(x):
        if x < 0 or x > 100:
            return False
        else:
            return True

    assert_equal(is_valid_percentage(-1), False)
    assert_equal(is_valid_percentage(0), True)
    assert_equal(is_valid_percentage(50), True)
    assert_equal(is_valid_percentage(100), True)
    assert_equal(is_valid_percentage(101), False)

Rewrite this function using `and` instead.
        """

        hints = """
If you have something like `x < 0 and x > 100`, you're on the wrong track. That's going to be `False` for *any* value of `x`!
The solution with `and` is different in several ways from the solution with `or`.
Our solution with `or` first determines if `x` is an invalid percentage, else concludes validity. Using `and` will do this in reverse.
You will have to reverse the `return` statements accordingly.
You will have to change the comparison operators too.
        """

        disallowed = [
            Disallowed(ast.Or, label="`or`"),
            Disallowed(ast.In, label="`in`"),
            Disallowed(ast.If, label="`if`", max_count=1),
        ]

        def solution(self):
            def is_valid_percentage(x: int):
                if 0 <= x and x <= 100:
                    return True
                else:
                    return False

            return is_valid_percentage

        tests = {
            -1: False,
            0: True,
            50: True,
            100: True,
            101: False,
        }

    class TicTacToeWinningRow(ExerciseStep):
        """
Awesome! Here's one possible solution:

    def is_valid_percentage(x):
        if 0 <= x and x <= 100:
            return True
        else:
            return False

As before, we can simplify this solution to:

    def is_valid_percentage(x):
        return 0 <= x and x <= 100

There's another trick to improve this further called comparison chaining. Any condition like this:

    a < b and b < c

can be shortened by removing the extra `and b` into:

    a < b < c

This works for any comparison operators, including `==`, and the two operators can even be different.
So the solution can be simplified to:

    def is_valid_percentage(x):
        return 0 <= x <= 100

Next exercise: given a list of three elements, check if all three elements are equal.

    __copyable__
    def all_equal(row):
        ...

    assert_equal(all_equal(["X", "X", "X"]), True)
    assert_equal(all_equal(["O", "O", "O"]), True)
    assert_equal(all_equal(["X", "O", "X"]), False)
        """

        hints = """
The list will always have 3 elements.
That means you don't need to use a loop.
Remember that you can get the first element using `row[0]`.
The first element, second element, and third element all need to be equal.
That means the first element should be equal to the second element and also the third element.
                """

        def solution(self):
            def all_equal(row: List[str]):
                return row[0] == row[1] and row[0] == row[2]

            return all_equal

        tests = [
            (["O", "O", "O"], True),
            (["X", "X", "X"], True),
            (["O", "X", "O"], False),
            (["O", "O", "X"], False),
            (["X", "O", "O"], False)
        ]

        @classmethod
        def generate_inputs(cls):
            if random.random() < 0.5:
                row = [generate_string()] * 3
            else:
                row = [generate_string() for _ in range(3)]
            return {"row": row}

    final_text = """
Good job. There are many possible correct solutions here:

    def all_equal(row):
        return row[0] == row[1] and row[0] == row[2]

or using comparison chaining again:

        return row[0] == row[1] == row[2]

or check that it's equal to a list containing the first element three times:

        return row == [row[0], row[0], row[0]]
"""


class MultiLineExpressions(Page):
    title = "Multi-line statements"

    class invalid_multiline(VerbatimStep):
        """
Our code lines are starting to get quite long.
Thankfully Python offers a few ways to spread out one statement across many lines,
but it's not automatic. You have to make sure Python understands that's what you're doing.
For example, this code is invalid syntax and will give you an error:

__program_indented__
        """
        program = """\
is_friend = name == "Alice" or
            name == "Bob"
"""

        def check(self):
            return "SyntaxError: invalid syntax" in self.result

    class valid_multiline(VerbatimStep):
        """
Python tries to interpret this as two separate lines of code and gets confused. You need to tell it that
the first line is continuing onto the second line.

One way to do this is by adding `\\` at the end of the line to 'escape' the line break.

Another way is to ensure that the line break is contained within some kind of brackets. Then the line
continuation is implied because Python will wait till all brackets have been closed before
considering a line to be complete. If you already have brackets because for example you're calling a function
or making a list, you may not need to do anything! Otherwise you can add brackets to any expression
to imply the line continuation.

Here are some examples. Pay close attention to the details.

    __copyable__
    __program_indented__
        """

        def program(self):
            name = "Bob"

            is_friend = name == "Alice" or \
                        name == "Bob"
            print(is_friend)

            is_friend = (name == "Alice" or
                         name == "Bob")
            print(is_friend)

            is_friend = [name == "Alice",
                         name == "Bob"]
            print(is_friend)

            print(name == "Alice" or
                  name == "Bob")

    final_text = """
So if you get a mysterious `SyntaxError`, make sure that you haven't improperly broken up any lines!
    """


class CombiningAndAndOr(Page):
    title = "Combining `and` and `or`"

    class CombiningAndOr(VerbatimStep):
        """
If you use both `and` and `or` in a single expression, it's a lot like combining `*` and `+`.
The operators are evaluated in a specific order.

For example, try the following code in the shell.
What do you expect?

__program_indented__
        """

        expected_code_source = "shell"
        program = "True or False and False"

    class AndHasHigherPriority(ExerciseStep):
        """
If you read it casually from left to right, you may think that:

    True or False and False

is equivalent to

    (True or False) and False

but it's actually equivalent to

    True or (False and False)

This is because `and` has a higher priority than `or`.
This is important because the first interpretation reduces to `True and False` which is `False`, while the second
interpretation reduces to `True or False` which is `True`!
You can try both options with parentheses in the shell to confirm.

**The lesson here is to be extra careful when combining operators.** Either add parentheses to be safe or
break up your expression into smaller parts and assign each part to a variable.
This will make your code clear, readable, and unambiguous, and will save you from painful mistakes.

Time for an exercise. Suppose you're writing a program to play tic-tac-toe,
also known as noughts and crosses or Xs and Os. If you've never heard of tic-tac-toe, you can read the rules
and play a few games [here](https://gametable.org/games/tic-tac-toe/).

We need to check if someone has won a game. Our function `all_equal` is already helpful for checking rows.

Write a function to check if someone has won a game by placing 3 of the same pieces on one of the diagonal lines.
The board is given as a nested list `board` of 3 sublists, each sublist containing 3 strings, representing a row. For example:

    board = [
        ['X', 'O', 'X'],
        ['X', 'X', 'O'],
        ['O', 'O', 'X']
    ]

The function should return a boolean: `True` if one of the diagonals have 3 of the same pieces, `False` otherwise.
Click the Copy button to get started with the code below.
We provided some tests for you, your job is to replace the `...` with your code.

    __copyable__
    def diagonal_winner(board):
        ...

    assert_equal(
        diagonal_winner(
            [
                ['X', 'O', 'X'],
                ['X', 'X', 'O'],
                ['O', 'O', 'X']
            ]
        ),
        True
    )

    assert_equal(
        diagonal_winner(
            [
                ['X', 'X', 'O'],
                ['X', 'O', 'O'],
                ['O', 'X', 'X']
            ]
        ),
        True
    )

    assert_equal(
        diagonal_winner(
            [
                ['O', 'X', 'O'],
                ['X', 'X', 'X'],
                ['O', 'O', 'X']
            ]
        ),
        False
    )
        """
        hints = """
How many diagonals are there on the board?
Which entries of the three sublists make up each diagonal? How can you access these entries?
Every list always has 3 entries, so no need for a loop.
There are two problems to solve here: checking for a win in a specific diagonal, and combining the checks for each diagonal.
One problem can be solved using `and`, the other using `or`.
There's a lot of similarity with the `all_equal` function. You can even call that function to help! But then you have to include its definition.
Similar to `all_equal`, check that the 3 entries on a diagonal are equal to each other, e.g. by using `and`.
Check the two diagonals together, using `or`.
        """

        disallowed = [
            Disallowed(ast.If, label="`if`"),
        ]

        def solution(self):
            def diagonal_winner(board: List[List[str]]):
                middle = board[1][1]
                return (
                        (middle == board[0][0] and middle == board[2][2]) or
                        (middle == board[0][2] and middle == board[2][0])
                )

            return diagonal_winner

        @classmethod
        def generate_inputs(cls):
            row1 = [random.choice(["X", "O"]) for _ in range(3)]
            row2 = [random.choice(["X", "O"]) for _ in range(3)]
            row3 = [random.choice(["X", "O"]) for _ in range(3)]
            return {
                "board": [row1, row2, row3]
            }

        tests = [
            ([["X", "O", "X"],
              ["X", "X", "O"],
              ["O", "O", "X"]], True),
            ([["X", "O", "O"],
              ["X", "O", "X"],
              ["O", "X", "X"]], True),
            ([["X", "O", "X"],
              ["X", "O", "X"],
              ["O", "O", "X"]], False),
        ]

    final_text = """
Well done! This was a hard one. Here are some possible solutions:

    def diagonal_winner(board):
        middle = board[1][1]
        return (
                (middle == board[0][0] and middle == board[2][2]) or
                (middle == board[0][2] and middle == board[2][0])
        )

or:

        diagonal1 = all_equal([board[0][0], board[1][1], board[2][2]])
        diagonal2 = all_equal([board[2][0], board[1][1], board[0][2]])
        return diagonal1 or diagonal2
"""


class IntroducingNotPage(Page):
    title = "Introducing `not`"

    class IntroducingNot(VerbatimStep):
        """
Unlike the other two boolean operators `and` and `or`,
which are used in between two booleans (called *binary* operators),
`not` is used before only one boolean (called a *unary* operator).
It negates the expression to which it is applied, a bit like a minus sign. Try in the shell:

__program_indented__
        """
        program = "not True"
        predicted_output_choices = ["True", "False"]

    class NotFalse(VerbatimStep):
        """
Now try the following:

__program_indented__
        """
        program = "not False"
        predicted_output_choices = ["True", "False"]

    class NotTrueOrTrue(VerbatimStep):
        """
What is the priority of `not` compared to `and` and `or`? Try the following in `birdseye`:

    __program_indented__
        """

        expected_code_source = "birdseye"

        def program(self):
            b = True
            print(not b or b)

    class NotPriority(ExerciseStep):
        """
You can see in `birdseye` that

    not True or True

is interpreted by Python as

    (not True) or True

rather than:

    not (True or True)

So, `not` has higher priority than `or` if there are no parentheses. It's the same as how

    -1 + 2

means:

    (-1) + 2

rather than

    -(1 + 2)

`not` also has higher priority than `and`.

Again, the main thing to remember is to use parentheses or extra variables when in doubt.

Exercise: Suppose you're writing a program which processes images. Only certain types of file can be processed.
If the user gives you a file that can't be processed, you want to show an error:

    if invalid_image(filename):
        print("I can't process " + filename)

Suppose that .png and .jpg files can be processed, but other file types cannot.
Here's an example function to do that:

    __copyable__
    def invalid_image(filename):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            return False
        else:
            return True

    assert_equal(invalid_image("dog.png"), False)
    assert_equal(invalid_image("cat.jpg"), False)
    assert_equal(invalid_image("invoice.pdf"), True)

This is longer than it needs to be. Rewrite `invalid_image` so that the body is a single line `return <expression>`,
i.e. no `if` statement. It should pass the same tests.
        """

        hints = [
            dedent("""
            What if you were instead asked to simplify this related but opposite function?

                def valid_image(filename):
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        return True
                    else:
                        return False

                assert_equal(valid_image("dog.png"), True)
                assert_equal(valid_image("cat.jpg"), True)
                assert_equal(valid_image("invoice.pdf"), False)
            """),
            "In that case there is a standard simplification trick you can apply that we discussed a few pages ago.",
            'In particular the `returns` are redundant because `filename.endswith(".png") or filename.endswith(".jpg")` '
            'is already the desired boolean.',
            dedent("""
            So you can just write:

                def valid_image(filename):
                    return filename.endswith(".png") or filename.endswith(".jpg")
            """),
            "For the real exercise, you can do something similar.",
            "The difference in the real exercise is that the result is reversed.",
            "That is, `invalid_image` returns `True` when `valid_image` returns `False` and vice versa.",
            "Remember what `not` does?",
        ]

        disallowed = [
            Disallowed(ast.If, label="`if`"),
        ]

        def solution(self):
            def invalid_image(filename: str):
                return not (filename.endswith(".png") or filename.endswith(".jpg"))

            return invalid_image

        tests = {
            "dog.png": False,
            "cat.jpg": False,
            "invoice.pdf": True,
        }

        @classmethod
        def generate_inputs(cls):
            result = generate_string()
            if random.random() < 0.5:
                result += random.choice([".png", ".jpg"])
            return {"filename": result}

    final_text = """
Well done! Here are two valid solutions:

    def invalid_image(filename):
        return not (filename.endswith(".png") or filename.endswith(".jpg"))

    def invalid_image(filename):
        return not filename.endswith(".png") and not filename.endswith(".jpg")

(if you're curious, these are equivalent because of something called De Morgan's law)

Also notice that this is another general pattern that can be simplified: if your code has the form:

    if x:
        return False
    else:
        return True

where `x` itself is a boolean, then it can be simplified to:

    return not x
    """
