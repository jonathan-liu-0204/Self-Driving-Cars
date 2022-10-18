import numpy as np

class KalmanFilter():
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # Error matrix
        self.P = np.identity(3) * 1
        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
        # State transition error covariance
        self.Q = None
        # Measurement error
        self.R = None

    def predict(self, u):
        raise NotImplementedError

    def update(self, z):
        raise NotImplementedError
        return self.x, self.P
