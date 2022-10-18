import numpy as np

class KalmanFilter():
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # [1.0, 0.0, 0.0]
        # [0.0, 1.0, 0.0]
        # [0.0, 0.0, 1.0]

        # Error matrix
        self.P = np.identity(3) * 1
        # [1, 0, 0]
        # [0, 1, 0]
        # [0, 0, 1]

        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
        
        self.n = self.A.shape[1]
        
        # State transition error covariance
        self.Q = np.eye(self.n)
        # Measurement error
        self.R = np.eye(self.n)

    def predict(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.R
        # raise NotImplementedError

    def update(self, z):
        
        S = self.R + np.dot(self.H.T, np.dot(self.H, self.P))
        K = np.dot(np.linalg.inv(S), np.dot(self.P, self.H.T))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x, self.P
        # raise NotImplementedError
