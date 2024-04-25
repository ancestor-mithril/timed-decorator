import unittest

from tests.functions import fibonacci, recursive_fibonacci
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


if __name__ == '__main__':
    unittest.main()
