import logging
import os
import unittest
from io import StringIO
from time import sleep

from tests.functions import fibonacci, recursive_fibonacci
from timed_decorator.builder import create_timed_decorator, get_timed_decorator
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

        fn = build_decorated_fn(fibonacci, timed, show_args=True, disable_gc=True)
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
                sleep(0.1)
                sleeping_fn(0.1)

            sleep(0.1)
            sleeping_fn(0.1)
            other_fn()
            sleeping_fn(0.1)

        nested_fn()

    def test_file_usage(self):
        filename = 'file.txt'

        @timed(file_path=filename, stdout=False)
        def fn():
            sleep(0.1)

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

        @timed(logger_name=logger_name, stdout=False)
        def fn_1():
            sleep(0.1)

        fn_1()

        @timed(logger_name=logger_name, stdout=False, use_qualname=True)
        def fn_2():
            sleep(0.1)

        fn_2()

        logged = log_stream.getvalue().split('\n')[:-1]
        self.assertEqual(len(logged), 2)
        self.assertIn(fn_1.__name__, logged[0])
        self.assertNotIn(fn_1.__qualname__, logged[0])
        self.assertIn(fn_2.__name__, logged[1])
        self.assertIn(fn_2.__qualname__, logged[1])

    def test_dict_output(self):
        out = {}

        @timed(out=out, stdout=False)
        def fn():
            sleep(0.1)

        fn()

        counts, elapsed, own_time = out[fn.__qualname__]
        self.assertEqual(counts, 1)
        self.assertIsInstance(elapsed, int)
        self.assertGreater(elapsed, 1e+8)
        self.assertEqual(elapsed, own_time)

        fn()
        counts, elapsed, own_time = out[fn.__qualname__]
        self.assertEqual(counts, 2)
        self.assertIsInstance(elapsed, int)
        self.assertGreater(elapsed, 2e+8)
        self.assertEqual(elapsed, own_time)

    def test_dict_output_with_nested_timed(self):
        out = {}

        @nested_timed(out=out, stdout=False)
        def fn1():
            @nested_timed(out=out, stdout=False)
            def fn2():
                sleep(0.1)

            fn2()

        fn1()

        counts_1, elapsed_1, own_time_1 = out[fn1.__qualname__]
        counts_2, elapsed_2, own_time_2 = out[fn1.__qualname__ + '.<locals>.fn2']

        self.assertTrue(counts_1 == counts_2 == 1)
        self.assertEqual(own_time_2, elapsed_2)
        self.assertGreater(elapsed_1, own_time_1)

    def test_qualname(self):
        out = {}

        class ClassA:
            @timed(out=out)
            def wait(self, x):
                sleep(x)

        ClassA().wait(0.1)
        key = next(iter(out.keys()))
        self.assertIn(ClassA.__name__, key)

    def test_return_time(self):
        seconds = 0.1

        @timed(return_time=True, stdout=False)
        def fn(x):
            sleep(x)

        _, elapsed = fn(seconds)

        self.assertIsInstance(elapsed, int)
        self.assertGreater(elapsed / 1e+9, seconds)

    def test_create_timed_decorator(self):
        create_timed_decorator('same name')
        self.assertRaises(KeyError, create_timed_decorator, 'same name')
        create_timed_decorator('other name')

        get_timed_decorator('same name')(fibonacci)(10000)
        get_timed_decorator('other name')(fibonacci)(10000)
        fn = get_timed_decorator('no name')(fibonacci)  # Does not raise error if not called
        self.assertRaises(KeyError, fn, 10000)

        # Defer instantiation
        fn = get_timed_decorator('lazy name')(fibonacci)
        create_timed_decorator('lazy name')
        fn(10000)


if __name__ == '__main__':
    unittest.main()
