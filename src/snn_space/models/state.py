import torch


class NetworkState:

    def __init__(self):
        self.reset()

    def reset(self):
        self.mem1 = None
        self.mem2 = None
        self.mem3 = None
        self.mem4 = None
        self.mem5 = None
        self.mem6 = None
        self.mem_out = None