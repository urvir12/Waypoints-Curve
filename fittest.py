import numpy as np
#using residual error to judge
class FitTest:
    #accepts original points, ellipse curve, polynomial curve

    def __init__(self, orginal_points, ellipse_curve, polynomial_curve):
        self.original_points = orginal_points
        self.ellipse_curve = ellipse_curve
        self.polynomial_curve = polynomial_curve

    def nearestdistance(self, original_points, curve):
        """
        Compute the distance from each original point to the nearest point on the fitted curve.
        
        """
        total_error = 0

        for point in original_points:
            distances = np.linalg.norm(curve - point, axis=1)  # Compute distance to all curve points
            min_distance = np.min(distances)  # Find the closest one
            total_error += min_distance ** 2  # Sum of squared distances

        return total_error


    def ellipse_fit(self) -> int:
        return self.nearestdistance(self.original_points, self.ellipse_curve)

    def polynomial_fit(self) -> int:
        return self.nearestdistance(self.original_points, self.polynomial_curve)

    def compare(self):
        ellipse_error = self.ellipse_fit()
        poly_error = self.polynomial_fit()

        print(f"Ellipse Fit Error: {ellipse_error:.4f}")
        print(f"Polynomial Fit Error: {poly_error:.4f}")

        if ellipse_error < poly_error:
            print("Ellipse fit is more accurate")
        elif poly_error < ellipse_error:
            print("The polynomial fit is more accurate")
        else:
            print("They are both the same.")
    