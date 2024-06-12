from copy import copy
import numpy as np
from models.manipulator_model import ManiuplatorModel

class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.m1 = ManiuplatorModel(Tp, m3=0.1, r3=0.05)
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        debug = self.state.T
        z_h_dot = (self.A @ np.expand_dims(self.state,axis=1) + self.B @ np.atleast_2d(u) + self.L @ (q - self.W @ np.expand_dims(self.state,axis=1)))
        self.state = self.Tp * z_h_dot.reshape((z_h_dot.shape[0],)) + self.state
    def get_state(self):
        return self.state
