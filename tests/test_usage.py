import unittest
from time import sleep

from tests.functions import fibonacci, recursive_fibonacci
from timed.nested_timed import nested_timed
from timed.simple_timed import timed
from timed.utils import build_decorated_fn


class UsageTest(unittest.TestCase):
    def test_simple_timed(self):
        fn = build_decorated_fn(fibonacci, timed)
        fn(1000)

        fn = build_decorated_fn(fibonacci, timed, collect_gc=False)
        fn(1000)

        fn = build_decorated_fn(recursive_fibonacci, timed)
        fn(30)

        fn = build_decorated_fn(recursive_fibonacci, timed, collect_gc=False)
        fn(30)

        fn = build_decorated_fn(recursive_fibonacci, timed, collect_gc=False, use_seconds=True)
        fn(30)

        fn = build_decorated_fn(recursive_fibonacci, timed, collect_gc=False, use_seconds=True, precision=5)
        fn(30)

        fn = build_decorated_fn(fibonacci, timed, show_args=True, display_level=0)
        fn(1000)

        fn = build_decorated_fn(fibonacci, timed, show_args=True, display_level=2)
        fn(1000)

    def test_nested_timed(self):
        @nested_timed()
        def nested_fn():

            @nested_timed()
            def sleeping_fn(x):
                sleep(x)

            @nested_timed()
            def other_fn():
                sleep(0.5)
                sleeping_fn(0.5)

            sleep(1)
            sleeping_fn(1)
            other_fn()
            sleeping_fn(1)

        nested_fn()


if __name__ == '__main__':
    unittest.main()
