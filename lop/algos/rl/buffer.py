import numpy as np
import collections as c

import torch


class Buffer(object):
    def __init__(self, o_dim, a_dim, bs, device='cpu'):
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.bs = bs
        self.device = device
        self.o_buf, self.a_buf, self.r_buf, self.logpb_buf, self.distb_buf, self.done_buf = \
            c.deque(), c.deque(), c.deque(), c.deque(), c.deque(), c.deque()
        self.op = np.zeros((1, o_dim), dtype=np.float32)

    def store(self, o, a, r, op, logpb, dist, done):
        self.o_buf.append(o)
        self.a_buf.append(a)
        self.r_buf.append(r)
        self.logpb_buf.append(logpb)
        self.distb_buf.append(dist)
        self.done_buf.append(float(done))
        self.op[:] = op

    def pop(self):
        self.o_buf.popleft()
        self.a_buf.popleft()
        self.r_buf.popleft()
        self.logpb_buf.popleft()
        self.distb_buf.popleft()
        self.done_buf.popleft()

    def clear(self):
        self.o_buf.clear()
        self.a_buf.clear()
        self.r_buf.clear()
        self.logpb_buf.clear()
        self.distb_buf.clear()
        self.done_buf.clear()

    def get(self, dist_stack):
        rang = range(self.bs)
        os = torch.as_tensor(np.array([self.o_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, self.o_dim)
        acts = torch.as_tensor(np.array([self.a_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, self.a_dim)
        rs = torch.as_tensor(np.array([self.r_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, 1)
        op = torch.as_tensor(self.op, device=self.device).view(-1, self.o_dim)
        logpbs = torch.as_tensor(np.array([self.logpb_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, 1)
        distbs = dist_stack([self.distb_buf[i] for i in rang], device=self.device)
        dones = torch.as_tensor(np.array([self.done_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, 1)

        return os, acts, rs, op, logpbs, distbs, dones