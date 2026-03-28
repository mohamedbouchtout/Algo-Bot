"""
Root module for running all tests. This can be expanded to include more test classes as needed.
"""

from tests.test_retest_200ma import TestRetest200MA

def main():
    test_bot = TestRetest200MA()
    test_bot.test_retest_200ma()

if __name__ == "__main__":
    main()