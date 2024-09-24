import logging
import unittest
from io import StringIO

import torch
from torch import Tensor

from timed_decorator.simple_timed import timed


def exp_plus_mean(x: Tensor) -> Tensor:
    return x.exp() + x.mean()


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.logger_name = "TEST_LOGGER"
        self.log_stream = StringIO()
        log_handler = logging.StreamHandler(self.log_stream)
        logging.root.setLevel(logging.NOTSET)
        logging.getLogger(self.logger_name).addHandler(log_handler)
        self.fn = exp_plus_mean

    def do_tensor_op(self, x: Tensor):
        timed(logger_name=self.logger_name, stdout=False, show_args=True)(self.fn)(x)

    def test_tensor_name(self):
        cpu_tensor = torch.rand((10, 50))
        self.do_tensor_op(cpu_tensor)
        logged = self.log_stream.getvalue().split("\n")[0]
        self.assertIn("CpuTensor[10, 50]", logged)

    def test_tensor_name_cuda(self):
        if torch.cuda.is_available():
            cuda_tensor = torch.rand((10, 50), device="cuda:0")
            self.fn(cuda_tensor)
            logged = self.log_stream.getvalue().split("\n")[0]
            self.assertIn("CudaTensor[10, 50]", logged)

    def test_traced_fn(self):
        self.fn = torch.jit.trace(exp_plus_mean, example_inputs=torch.rand((10, 50)))

        cpu_tensor = torch.rand((10, 50))
        self.do_tensor_op(cpu_tensor)
        logged = self.log_stream.getvalue().split("\n")[0]
        self.assertIn("CpuTensor[10, 50]", logged)

    def test_scripted_fn(self):
        self.fn = torch.jit.script(exp_plus_mean)

        cpu_tensor = torch.rand((10, 50))
        self.do_tensor_op(cpu_tensor)
        logged = self.log_stream.getvalue().split("\n")[0]
        self.assertIn("CpuTensor[10, 50]", logged)


if __name__ == "__main__":
    unittest.main()
