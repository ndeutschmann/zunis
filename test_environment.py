"""Test the current python environment"""

import sys
import unittest
from pathlib import Path
import pkg_resources
from pkg_resources import yield_lines,Requirement,WorkingSet
import os
import sys

_REQUIREMENTS_PATH = Path(__file__).with_name("requirements.txt")
REQUIRED_PYTHON = "python3"


def parse_requirements(strs):
    """Yield ``Requirement`` objects for each specification in `strs`

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    This is a slight modification of pkg_resources.parse_requirements to
    ignore the requirement "-e ." used here
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(strs))

    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if ' #' in line:
            line = line[:line.find(' #')]
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith('\\'):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return

        # The local library editable install is replaced with its name: "zunis"
        if line.strip() in [".", "-e ."]:
            line = "zunis"

        yield Requirement(line)


class TestEnvironment(unittest.TestCase):
    """Test availability of required packages."""

    def test_requirements(self):
        """Test that each required package is available."""
        # Ref: https://stackoverflow.com/a/45474387/
        print("Checking requirements.txt")
        with _REQUIREMENTS_PATH.open() as file:
            requirements = parse_requirements(file)
            path_without_wd = sys.path[1:]
            working_set = WorkingSet(entries=path_without_wd)
            for requirement in requirements:
                requirement = str(requirement)
                with self.subTest(requirement=requirement):
                    working_set.require(requirement)

    def test_executable(self):
        """Check that the python executable is the right major version"""
        system_major = sys.version_info.major
        if REQUIRED_PYTHON == "python":
            required_major = 2
        elif REQUIRED_PYTHON == "python3":
            required_major = 3
        else:
            raise ValueError("Unrecognized python interpreter: {}".format(
                REQUIRED_PYTHON))

        if system_major != required_major:
            raise TypeError(
                "This project requires Python {}. Found: Python {}".format(
                    required_major, sys.version))


if __name__ == '__main__':
    unittest.main()
