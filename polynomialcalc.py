import numpy as np
import matplotlib.pyplot as plt

class Polynomial_Calc:

    def __init__(self, points, degree=3):
        self.points = points
        self.degree = degree
        self.coeffs_x, self.coeffs_y, self.coeffs_z= self.fit_polynomial()
    def fit_polynomial(self):
        t = np.linspace(0, 1, len(self.points))
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]

        coeffs_x = np.polyfit(t, x, self.degree)
        coeffs_y = np.polyfit(t, y, self.degree)
        coeffs_z = np.polyfit(t, z, self.degree)

        return coeffs_x, coeffs_y, coeffs_z
    
    def evaluate_polynomial(self):
        num_points = len(self.points)
        t_vals = np.linspace(0, 1, num_points)

        x_fit = np.polyval(self.coeffs_x, t_vals)
        y_fit = np.polyval(self.coeffs_y, t_vals)
        z_fit = np.polyval(self.coeffs_z, t_vals)
        
        return np.vstack((x_fit, y_fit, z_fit)).T
    
    def plot_fit(self):
        fitted_points = self.evaluate_polynomial()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], color='blue', label="Original Points")

        ax.plot(fitted_points[:, 0], fitted_points[:, 1], fitted_points[:, 2], color='red', linewidth=2, label="Polynomial Fit")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()