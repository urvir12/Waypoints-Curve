import math
from collections import namedtuple
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from ellipse import Ellipse
from convexhull import ConvexHullCalc
from polynomialcalc import Polynomial_Calc
from fittest import FitTest
Point = namedtuple("Point", ["x", "y", "z"])


def ellipse_points(a, b, center, normal, num_points = 70):
    theta = np.linspace(0, 2 * np.pi, num_points) #generating 100 different points with different angles
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = 0.05 * np.sin(2 * theta)

    #noise
    x += np.random.normal(0, 0.2, size=theta.shape)
    y += np.random.normal(0, 0.2, size=theta.shape)
    z += np.random.normal(0, 0.2, size=theta.shape)


    ellipse_2d = np.vstack((x, y, z)).T

    u, v = np.linalg.svd(np.random.randn(3, 3))[0][:, :2].T
    if np.dot(u, normal) > np.dot(v, normal):
        u, v = v, u
    
    basis = np.vstack((u, v, normal))

    ellipse_3d = ellipse_2d @ basis + center
    return ellipse_3d
def resample_curve(curve, num_points):
    t_old = np.linspace(0, 1, curve.shape[0])
    t_new = np.linspace(0, 1, num_points)
    interpolator = interp1d(t_old, curve, axis=0, kind='linear')
    return interpolator(t_new)
def main():
    theta = np.linspace(0, 2 * np.pi, 60)
    a = 5
    b = 3

    center = np.array([1, 2, 3])
    normal = np.array([0, 0, 1])
    

    points = ellipse_points(a, b, center, normal)

    hullanalyzer = ConvexHullCalc(points)
    max_dist, pair = hullanalyzer.diameter(points)

    print("Diameter of convex hull: ", max_dist)
    if pair:
        hullanalyzer.plotdiameter(points, pair)
    
    ellipsefit = Ellipse(points)
    ellipse_coefficients = ellipsefit.fit2d()
    ellipse_curve = ellipsefit.transformback(ellipse_coefficients)
    ellipse_curve = resample_curve(ellipse_curve, len(points))
    ellipse_3d = ellipsefit.transformback(ellipse_coefficients)
    ellipsefit.plot_ellipse(ellipse_3d)

    #poly
    poly_fit = Polynomial_Calc(points, degree=3)
    polynomial_curve = poly_fit.evaluate_polynomial()
    poly_fit.plot_fit()

    fit_test = FitTest(points, ellipse_curve, polynomial_curve)
    fit_test.compare()


if __name__ == "__main__":
    main()
    
"""
def ellipse(points):
    theta = np.linspace(0, 2*np.pi, 100)
    major = 5
    minor = 3

    x = major * np.cos(theta)
    y = minor * np.sin(theta)

    noise_level = 0.2

    x_noisy = x + np.random.normal(0, noise_level, size = x.shape)
    y_noisy = x + np.random.normal(0, noise_level, size = y.shape)

    plt.scatter(x_noisy, y_noisy)




def conic_trajectory(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    X = np.column_stack((x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, np.ones(x.shape)))
    _, _, V = np.linalg.svd(X)
    coeffs = V[-1, :]
    return coeffs

def plot_conic(coeffs, points):
    A, B, C, D, E, F, G, H, I, J = coeffs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Points')

    t_vals = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    x_vals = t_vals
    y_vals = (-G * x_vals - J) / H
    z_vals = np.sqrt(np.abs(-(A*x_vals**2 + B * y_vals**2 + D * x_vals * y_vals + G * x_vals + H * y_vals + J) / C))
    
    
    ax.plot(x_vals, y_vals, z_vals, color='red', linewidth=2, label = "Conic Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

"""
