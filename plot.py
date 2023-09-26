import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import get_coords, is_within_limit
from mpl_toolkits.mplot3d import Axes3D

class Plot:
    def __init__(self, roofs) -> None:
        self.roofs = roofs

    def scatterplot_3d(self):
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                x, y, z, c = get_coords(roof)
                ax = plt.axes(projection='3d')
                ax.scatter(x, y, z, c = c/255)
                plt.show()

    def plot_2D(self, roof_segments):
        # Plot the roof in 2D with different colors
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                df = self.roofs.alpha_shapes(roof_segments)
                for i in range(len(df)):
                    poly = df.iloc[i].tolist()[0]
                    plt.plot(*poly.exterior.xy)
                plt.show()

    def plane_3D(self, roof):
        # https://saturncloud.io/blog/plotting-3d-plane-equations-with-python-matplotlib-a-guide/
        x = np.linspace(-10, 10, 1000)
        y = np.linspace(-10, 10, 1000)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        for i in range(len(roof)):
            params = roof.iloc[i][2]
            
            a, b, c, d = params[0], params[1], params[2], params[3]
            Z = (d - a*X - b*Y) / c

            # Plot the surface
            ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def scatter_with_plane_3D_all(self, df_planes, ids):
        count = 0
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                x, y, z, c = get_coords(roof)
                ax = plt.axes(projection='3d')
                ax.scatter(x, y, z, c = c/255)
                df = df_planes.loc[df_planes["roof_id"] == ids[count]]

                # Plot planes from df2
                for idx, row in df.iterrows():
                    a, b, c, d = row[2][0], row[2][1], row[2][2], row[2][3]
                    xx, yy = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
                    zz = (-a * xx - b * yy - d) / c
                    ax.plot_surface(xx, yy, zz, alpha=0.5)

                # Set labels and legend
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.legend()

                # Show the plot
                plt.show()
                count += 1

    def scatter_with_plane_3D_segments(self, df_all_roofs, df_planes):
        count = 0
        for i in range(len(df_all_roofs)):
            if count < 5:
                plane_params = df_planes.loc[i][2]
                x_coords = []
                y_coords = []
                z_coords = []
                # Extract coordinates from the polygon
                for point in df_all_roofs.iloc[i][1].exterior.coords:
                    x, y, z = point
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)

                red_value = 255 
                colors = np.zeros((len(x_coords), 3), dtype=np.uint8)  
                colors[:, 0] = red_value

                a, b, c, d = plane_params[0], plane_params[1], plane_params[2], plane_params[3]
                xx, yy = np.meshgrid(np.linspace(min(x_coords), max(x_coords), 100), np.linspace(min(y_coords), max(y_coords), 100))
                zz = (-a * xx - b * yy - d) / c
                ax = plt.axes(projection='3d')
                ax.scatter(x_coords, y_coords, z_coords, c = colors /255)

                ax.plot_surface(xx, yy, zz, alpha=0.5)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.legend()

                # Show the plot
                plt.show()
                count += 1

    def plane_with_intersections(self, ids, df_planes, df_lines):
        # Funker ikke 
        count = 0
        for id in ids: 
            if count < 1:
                df_planes_curr = df_planes.loc[df_planes['roof_id'] == id]
                df_lines_curr = df_lines.loc[df_lines['roof_id'] == id]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                if not df_lines_curr.empty:
                    for i in range(len(df_lines_curr)):
                        direction = np.array(df_lines_curr.iloc[i][3])
                        point_on_plane = df_lines_curr.iloc[i][4]
                        t_values = np.linspace(-10000,10000,100)
                        intersection_points = []

                        # Calculate the intersection points for each value of t
                        for t in t_values:
                            intersection_point = point_on_plane + t * direction
                            intersection_points.append(intersection_point)

                        # Convert the list of points to a NumPy array
                        intersection_points = np.array(intersection_points)
                        ax.plot(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='r', linewidth=3)

                names = self.roofs.roofs.keys()
                data = None
                for name in names:
                    if f'{ids[count]}.las' in self.roofs.roofs[name]:
                        data = self.roofs.roofs[name][f'{ids[count]}.las']

                for idx, row in df_planes_curr.iterrows():
                    plane_params1 = df_planes_curr.loc[df_planes_curr['segment'] == row[1]]['plane_param'].iloc[0]
                    a1, b1, c1, d1 = plane_params1[0], plane_params1[1], plane_params1[2], plane_params1[3]

                    # xx, yy = np.meshgrid(np.linspace(min(data.X), max(data.X), 100), np.linspace(min(data.Y), max(data.Y), 100))
                    xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
                    X, Y = np.meshgrid(xx, yy)

                    zz1 = (-a1 * X - b1 * Y - d1) / c1

                    ax.plot_surface(X, Y, zz1, alpha=0.5)

                x_min, x_max = min(data.X), max(data.X)
                y_min, y_max = min(data.Y), max(data.Y)
                z_min, z_max = min(data.Z), max(data.Z)
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_min, z_max])
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

                # Show the plot
                plt.show()
                count += 1

    def plot_intersections(self, df_lines, ids):
        count = 0
        for id in ids: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            df_curr = df_lines.loc[df_lines['roof_id'] == id]
            if not df_curr.empty:
                for i in range(len(df_curr)):
                    direction = np.array(df_curr.iloc[i][3])
                    point_on_plane = df_curr.iloc[i][4]
                    t_values = np.linspace(-10000,10000,10000)
                    intersection_points = []

                    # Calculate the intersection points for each value of t
                    for t in t_values:
                        intersection_point = point_on_plane + t * direction
                        intersection_points.append(intersection_point)

                    # Convert the list of points to a NumPy array
                    intersection_points = np.array(intersection_points)
                    ax.plot(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='r', linewidth=3)

            # print(row)
            names = self.roofs.roofs.keys()
            data = None
            for name in names:
                if f'{ids[count]}.las' in self.roofs.roofs[name]:
                    data = self.roofs.roofs[name][f'{ids[count]}.las']
            
            x_min, x_max = min(data.X), max(data.X)
            y_min, y_max = min(data.Y), max(data.Y)
            z_min, z_max = min(data.Z), max(data.Z)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            count += 1
            plt.show()

    def line_scatter(self, df_lines, ids):
        counter = 0 
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                df_curr = df_lines.loc[df_lines['roof_id'] == ids[counter]]
                print(df_curr)
                x, y, z, c = get_coords(roof)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c = c/255)

                names = self.roofs.roofs.keys()
                data = None
                for name in names:
                    if f'{ids[counter]}.las' in self.roofs.roofs[name]:
                        data = self.roofs.roofs[name][f'{ids[counter]}.las']

                if not df_curr.empty:
                    for i in range(len(df_curr)):
                        direction = np.array(df_curr.iloc[i][3])
                        point_on_plane = df_curr.iloc[i][4]
                        t_values = np.linspace(-10000,10000,10000)
                        intersection_points = []

                        # Calculate the intersection points for each value of t
                        for t in t_values:
                            intersection_point = point_on_plane + t * direction
                            intersection_points.append(intersection_point)

                        # Convert the list of points to a NumPy array
                        intersection_points = np.array(intersection_points)
                        ax.plot(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='r', linewidth=3)

                x_min, x_max = min(data.X), max(data.X)
                y_min, y_max = min(data.Y), max(data.Y)
                z_min, z_max = min(data.Z), max(data.Z)
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_min, z_max])

                counter += 1
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.legend()
                plt.show()

    #{'corner_element': 
        # {'182448567.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 3176 points, 0 vlrs)>, 
        # '182464406.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 3235 points, 0 vlrs)>}, 
    # 'flat': 
        # {'182341061.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 2725 points, 0 vlrs)>, 
        # '300557684.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 3958 points, 0 vlrs)>}, 
    # 't-element': 
        # {'10498821.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 3107 points, 0 vlrs)>, 
        # '10477107.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 1558 points, 0 vlrs)>}, 
    # 'hipped': 
        # {'300429640.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 2803 points, 0 vlrs)>, 
        # '182282537.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 2312 points, 0 vlrs)>}, 
    # cross_element': 
        # {'182448729.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 2596 points, 0 vlrs)>, 
        # '21088358.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 3609 points, 0 vlrs)>}, 
    # 'gabled': 
        # {'182338605.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 1147 points, 0 vlrs)>, 
        # '10472350.las': <LasData(1.2, point fmt: <PointFormat(3, 0 bytes of extra dims)>, 2336 points, 0 vlrs)>}}