import pandas as pd
import matplotlib.pyplot as plt
import os

class Plot:
    def __init__(self, roofs) -> None:
        self.roofs = roofs

    def scatterplot_3d(self, x, y, z, c, s):
        # 3D scatter plot with all points, different color for each segment in roof
        name = "scatterplot_3d"
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c, s)
        plt.show()

    def plot_2D(self, roof_segments):
        # Plot the roof in 2D with different colors
        df = self.roofs.alpha_shapes(roof_segments)
        for i in range(len(df)):
            poly = df.iloc[i].tolist()[0]
            plt.plot(*poly.exterior.xy)
        plt.show()