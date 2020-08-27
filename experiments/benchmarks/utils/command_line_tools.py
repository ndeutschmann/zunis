"""Tools for command line interfaces with click"""
import ast

import click


class PythonLiteralOption(click.Option):
    """Option allowing a python literal to be passed as a string which is then parsed

    Notes
    -----
    Small variation on a `solution`_ proposed by stackoverflow user Stephen Rauch

    .. _solution: https://stackoverflow.com/a/47730333
    """

    def type_cast_value(self, ctx, value):
        try:
            # Convert the literal
            return ast.literal_eval(value)
        except ValueError:
            # The default value is allowed to not be a litteral for readability
            # Manifest by the fact that value does not have string type
            return value
        except SyntaxError:
            # Raise a proper error if the string is not parsable
            raise click.BadParameter(value)
