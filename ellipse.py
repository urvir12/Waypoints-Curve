import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

#
class Ellipse: 
    def __init__(self, points):
        self.points = points
        self.centroid, self.normal = self.fit_plane()
        #transforms the 3d points into a 2d plane and stores the new 
        #2d points and basis vectors for the plane.
        self.points2d, self.basis = self.plane()

   
    #returns centroid and the normal vector
    def fit_plane(self):
         #finds the centroid of the points (average of the coordinates)
        centroid = np.mean(self.points, axis=0)
        #performs SVD(singular value decomposition) on the points after 
        #subtracting the centroid to find the plane's normal vector
        _, _, Vt = np.linalg.svd(self.points - centroid) #finds normal vector
        normal = Vt[2]
        return centroid, normal

    #computes the covariance matrix of the points (helps determine orientation of the points)
    def plane(self):
         #performs SVD to extract u and v which define 2d plane
        u, v = np.linalg.svd(np.cov(self.points.T))[0][:, :2].T
        basis = np.vstack([u, v])
        #returns 2d transformed points and basis vectors of the plane
        return np.dot(self.points - self.centroid, basis.T), basis
    
    #fits conic (ellipse) to the 2d points using least squares
    def fit2d(self):
        #gets 2d coordinates
        x, y = self.points2d[:, 0], self.points2d[:, 1]
        #constructs matrix D from 2d coordinates
        D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
        #uses svd to solve for the coefficients that define the ellipse
        _, _, Vt = np.linalg.svd(D)
        #returns coefficients that define the ellipse equation in 2d
        return Vt[-1]

    #takes the coefficients of the fitted ellipse and transforms back to 3d space
    def transformback(self, coeffs):
        #coeffs from previous method (fit2d)
        A, B, C, D, E, F = coeffs
        #creates an array of angles from 0 to 2pi for parametric form
        theta = np.linspace(0, 2 * np.pi, 100)
        #semi major and minor axes of ellipse
        a, b = np.sqrt(1 / np.abs([A, C]))
        x, y = a * np.cos(theta), b * np.sin(theta)

        ellipse2d = np.vstack([x, y])
        return ellipse2d.T @ self.basis + self.centroid
    """
    def set_axes_equal(self, ax):
        xlimits = ax.get_xlim()
        ylimits = ax.get_ylim()
        zlimits = ax.get_zlim()

        xrange = np.abs(xlimits[1] - xlimits[0])
        yrange = np.abs(ylimits[1] - ylimits[0])
        zrange = np.abs(zlimits[1] - zlimits[0])

        max_range = max(xrange, yrange, zrange)

        x_mid = np.mean(xlimits)
        y_mid = np.mean(ylimits)
        z_mid = np.mean(zlimits)

        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    """
    #plots 3d points and ellipse
    def plot_ellipse(self, ellipse3D):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], color='blue', label="Original Points")
        ax.plot(ellipse3D[:, 0], ellipse3D[:, 1], ellipse3D[:, 2], color='red', label="Fitted Ellipse")

        #self.set_axes_equal(ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()
