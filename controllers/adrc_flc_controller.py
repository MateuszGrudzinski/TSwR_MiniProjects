import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3*p[0], 0],[0, 3*p[1]],[3*p[0]**2, 0],[0, 3*p[1]**2],[p[0]**3, 0],[0, p[1]**3]])
        W = np.zeros((2,6))
        A = np.zeros((6,6))
        B = np.zeros((6,2))
        rows = np.arange(4)
        cols = np.arange(2, 6)
        A[rows, cols] = 1

        W[0,0] = 1
        W[1,1] = 1

        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        C = self.model.C(x)

        A = np.zeros((6,6))
        B = np.zeros((6, 2))
        I = np.eye(2)
        A[0:2, 2:4] = I
        A[2:4, 4:6] = I
        A[2:4, 2:4] = -np.linalg.inv(M) @ C
        B[2:4, :] = np.linalg.inv(M)
        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q1, q2, q1_dot, q2_dot = x
        z_h = self.eso.get_state()
        x_h  = z_h[0:2]
        x_h_dot = z_h[2:4]
        f= z_h[4:]

        v = q_d_ddot + self.Kd @ (q_d_dot - x_h_dot) + self.Kp @ (q_d - np.array([q1,q2]))
        u = self.model.M(x) @ (v - f) + self.model.C(x) @ x_h_dot

        self.update_params(x_h, x_h_dot)
        self.eso.update(np.expand_dims(np.array([q1,q2]),1), np.expand_dims(u,1))

        return u
