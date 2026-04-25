"""
Main for running all the tests
"""

from tests.run_tests import RunTests

def main():
    tests = RunTests()
    tests.run()

if __name__ == "__main__":
    main()