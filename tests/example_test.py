"""
PYTEST RECAP — ONE FILE EDITION 🧪

Save this file as: test_pytest_recap.py
Run with: pytest -v

This file demonstrates:
- Test discovery
- Basic assertions
- Fixtures
- Fixture scopes
- Setup / teardown via yield
- Parametrization
- Exception testing
- Marks (skip / xfail)
- Temporary paths
"""

import pytest

# --------------------------------------------------
# BASIC TEST — pytest discovers functions named test_*
# --------------------------------------------------


def test_basic_assertion():
    """
    pytest uses plain Python assert statements.
    No special assertion methods required.
    """
    assert 1 + 1 == 2


def test_string_membership():
    """
    pytest gives very readable error messages when asserts fail.
    """
    assert "py" in "pytest"


# --------------------------------------------------
# FIXTURES — reusable test data / setup
# --------------------------------------------------


@pytest.fixture
def sample_list():
    """
    A fixture is a function decorated with @pytest.fixture.
    The return value is injected into tests that ask for it.
    """
    return [1, 2, 3]


def test_fixture_usage(sample_list):
    """
    pytest sees the function argument name (sample_list),
    finds a fixture with that name, and injects it.
    """
    assert len(sample_list) == 3
    assert sample_list[0] == 1


# --------------------------------------------------
# FIXTURE WITH SETUP + TEARDOWN (yield pattern)
# --------------------------------------------------


@pytest.fixture
def resource():
    """
    Code before 'yield' = setup
    Code after 'yield' = teardown
    """
    print("\n[setup] creating resource")
    data = {"status": "ready"}

    yield data  # test runs here

    print("[teardown] destroying resource")


def test_resource_fixture(resource):
    """
    The test receives whatever was yielded.
    """
    assert resource["status"] == "ready"


# --------------------------------------------------
# FIXTURE SCOPES
# --------------------------------------------------


@pytest.fixture(scope="module")
def shared_value():
    """
    scope="module" means:
    - created once per test file
    - shared across all tests in this module
    """
    return 42


def test_shared_value_one(shared_value):
    assert shared_value == 42


def test_shared_value_two(shared_value):
    assert shared_value > 0


# --------------------------------------------------
# PARAMETRIZED TESTS — same test, many inputs
# --------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (2, 3, 5),
        (0, 0, 0),
        (-1, 1, 0),
    ],
)
def test_addition(a, b, expected):
    """
    pytest will run this test once per tuple.
    Each case shows up as a separate test in output.
    """
    assert a + b == expected


# --------------------------------------------------
# TESTING EXCEPTIONS
# --------------------------------------------------


def test_exception_is_raised():
    """
    Use pytest.raises to assert exceptions.
    """
    with pytest.raises(ValueError):
        int("not-a-number")


def test_exception_message():
    """
    You can also match the exception message.
    """
    with pytest.raises(ValueError, match="invalid literal"):
        int("still-not-a-number")


# --------------------------------------------------
# MARKS — skip and xfail
# --------------------------------------------------


@pytest.mark.skip(reason="Example of a skipped test")
def test_skipped():
    """
    This test will be skipped entirely.
    """
    assert False


@pytest.mark.xfail(reason="Known bug, expected to fail")
def test_expected_failure():
    """
    xfail means:
    - failure does NOT fail the test suite
    - success is reported as XPASS
    """
    assert 1 == 2


# --------------------------------------------------
# TEMPORARY FILES & DIRECTORIES
# --------------------------------------------------


def test_tmp_path(tmp_path):
    """
    tmp_path is a built-in pytest fixture.
    It provides a unique temporary directory per test.
    """
    file_path = tmp_path / "example.txt"

    # Write to the file
    file_path.write_text("hello pytest")

    # Read from the file
    content = file_path.read_text()

    assert content == "hello pytest"


# --------------------------------------------------
# TEST CLASSES (optional, no __init__)
# --------------------------------------------------


class TestMath:
    """
    pytest will collect classes named Test*
    No __init__ method allowed.
    """

    def test_multiplication(self):
        assert 2 * 3 == 6

    def test_division(self):
        assert 10 / 2 == 5


# --------------------------------------------------
# FINAL NOTES (not code, just vibes)
# --------------------------------------------------
"""
Key pytest ideas to remember:

- Tests are just functions
- Fixtures are passed as arguments
- Assertions are plain assert
- pytest handles ordering, setup, teardown, and reporting
"""
