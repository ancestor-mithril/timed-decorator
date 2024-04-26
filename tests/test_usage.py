import logging
import os
import unittest
from io import StringIO
from time import sleep

from tests.functions import fibonacci, recursive_fibonacci
from timed_decorator.nested_timed import nested_timed
from timed_decorator.simple_timed import timed
from timed_decorator.utils import build_decorated_fn


class UsageTest(unittest.TestCase):
    @staticmethod
    def test_simple_timed():
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

    @staticmethod
    def test_nested_timed():
        @nested_timed(collect_gc=False, use_seconds=True, precision=3)
        def nested_fn():
            @nested_timed(collect_gc=False, use_seconds=True, precision=3)
            def sleeping_fn(x):
                sleep(x)

            @nested_timed(collect_gc=False, use_seconds=True, precision=3)
            def other_fn():
                sleep(0.5)
                sleeping_fn(0.5)

            sleep(1)
            sleeping_fn(1)
            other_fn()
            sleeping_fn(1)

        nested_fn()

    def test_file_usage(self):
        filename = 'file.txt'

        @timed(file_path=filename)
        def fn():
            sleep(1)

        try:
            fn()
            fn()
            with open(filename, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
                for line in lines:
                    self.assertEqual(line[:2], fn.__name__)
        finally:
            if os.path.isfile(filename):
                os.remove(filename)

    def test_logger_usage(self):
        logger_name = 'TEST_LOGGER'
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        logging.root.setLevel(logging.NOTSET)
        logging.getLogger(logger_name).addHandler(log_handler)

        @timed(logger_name=logger_name)
        def fn():
            sleep(1)

        fn()
        fn()

        logged = log_stream.getvalue().split('\n')[:-1]
        self.assertEqual(len(logged), 2)
        self.assertIn(fn.__name__, logged[0])
        self.assertIn(fn.__name__, logged[1])


if __name__ == '__main__':
    unittest.main()
