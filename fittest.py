import numpy as np
#using residual error to judge
class FitTest:
    #accepts original points, ellipse curve, polynomial curve

    def __init__(self, orginal_points, ellipse_curve, polynomial_curve):
        self.original_points = orginal_points
        self.ellipse_curve = ellipse_curve
        self.polynomial_curve = polynomial_curve

    def ellipse_fit(self) -> int:
        return np.sum(np.linalg.norm(self.original_points - self.ellipse_curve, axis=1) ** 2)

    def polynomial_fit(self) -> int:
        return np.sum(np.linalg.norm(self.original_points - self.polynomial_curve, axis=1) ** 2)

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
    