import numpy as np

class Kalman1D:
    """
    Scalar Kalman filter for a single measurement variable (e.g., range, angle).
    State is just the scalar value x.
    """
    def __init__(self, x0=0.0, P0=1.0, Q=1e-5, R=1e-2):
        self.x = float(x0)   # state estimate
        self.P = float(P0)   # estimate covariance
        self.Q = float(Q)    # process noise variance
        self.R = float(R)    # measurement noise variance

    def predict(self):
        # For a static scalar state model x_k = x_{k-1} + w, so prediction is identity
        self.P += self.Q

    def update(self, z):
        # Kalman gain
        K = self.P / (self.P + self.R)
        # state update
        self.x = self.x + K * (z - self.x)
        # covariance update (Joseph form avoided for brevity)
        self.P = (1 - K) * self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return self.x

# Example usage:
if __name__ == "__main__":
    import random, time
    kf = Kalman1D(x0=0.0, P0=1.0, Q=1e-4, R=0.05)
    while True:
        time.sleep(5)
        measurements = [np.sin(i*0.1) + random.gauss(0, 0.2) for i in range(100)]
        filtered = [kf.step(z) for z in measurements]
        print("\n\n", 20*"=","\nMeasured:\n", measurements[:10])
        print("\n\n", 20*"=","\nFiltered:\n", filtered[:10])
