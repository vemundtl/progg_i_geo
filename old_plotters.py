import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import get_coords, is_within_limit
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely import Polygon
import os

class Plot:
    def __init__(self, roofs) -> None:
        self.roofs = roofs

        if not os.path.exists("./plots"):
            os.mkdir("./plots")

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
                plt.title(ids[count])
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
                    if f'{ids[count]}' in self.roofs.roofs[name]:
                        data = self.roofs.roofs[name][f'{ids[count]}']

                for idx, row in df_planes_curr.iterrows():
                    plane_params1 = df_planes_curr.loc[df_planes_curr['segment'] == row[1]]['plane_param'].iloc[0]
                    a1, b1, c1, d1 = plane_params1[0], plane_params1[1], plane_params1[2], plane_params1[3]

                    # xx, yy = np.meshgrid(np.linspace(min(data.X), max(data.X), 100), np.linspace(min(data.Y), max(data.Y), 100))
                    xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
                    X, Y = np.meshgrid(xx, yy)

                    zz1 = (-a1 * X - b1 * Y - d1) / c1

                    ax.plot_surface(X, Y, zz1, alpha=0.5)

                x_min, x_max = min(data.x), max(data.x)
                y_min, y_max = min(data.y), max(data.y)
                z_min, z_max = min(data.z), max(data.z)
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

            names = self.roofs.roofs.keys()
            data = None
            for name in names:
                if f'{ids[count]}' in self.roofs.roofs[name]:
                    data = self.roofs.roofs[name][f'{ids[count]}']
            
            x_min, x_max = min(data.x), max(data.x)
            y_min, y_max = min(data.y), max(data.y)
            z_min, z_max = min(data.z), max(data.z)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            count += 1
            plt.title(id)
            plt.show()

    def test(self, points):
        counter = 0
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            keys = list(self.roofs.roofs[roofs].keys())
            for i, roof in enumerate(roof_segments):
                x, y, z, c = get_coords(roof)
                ax = plt.axes(projection='3d')
                ax.scatter(x, y, z, c = c/255)
                for point in points[counter]:
                    ax.scatter(point[0], point[1], point[2], color='r', linewidth=10)
                plt.title(keys[i])
                plt.show()
                counter += 1

    def test2(self, multi_polygon, id):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ["red", "blue", "green", "orange", "brown", "black"]
        for i, polygon in enumerate(multi_polygon):
            x, y, z = zip(*polygon.exterior.coords)
            ax.plot(x, y, z, color=colors[i], alpha=0.7)

        # Set axis limits and labels
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)
        # ax.set_zlim(0, 1)  # Adjust the z-axis limits as needed
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title(id)
        # Show the 3D plot
        plt.show()

    def test3(self, inter_section_points, extra):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ["red", "blue", "green", "black", "brown", "orange"]
        for i, point in enumerate(inter_section_points):
            # for point in polygon:
            x, y, z = point[0], point[1], point[2]
            ax.scatter(x, y, z, color=colors[1])
        for ex in extra:
            x, y, z = ex[0], ex[1], ex[2]
            ax.scatter(x, y, z, color=colors[0], linewidth=10)

        # Set axis limits and labels
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)
        # ax.set_zlim(0, 1)  # Adjust the z-axis limits as needed
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Show the 3D plot
        plt.show()

    def test4(self, scatter_points, extra_points11, extra_points1, extra_points2, extra3, id):
        ax = plt.axes(projection='3d')
        color=['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan"]
        # for i, seg in enumerate(scatter_points):
        #     for point in seg:
        #         ax.scatter(point[0], point[1], point[2], color=color[0])
        
        for i, seg in enumerate(extra_points11):
            ax.scatter(seg[0], seg[1], seg[2], c = color[0], linewidth=3)
        for i, seg in enumerate(extra_points1):
            ax.scatter(seg[0], seg[1], seg[2], c = color[1], linewidth=3)
        for i, seg in enumerate(extra_points2):
            for j, point in enumerate(seg):
                ax.scatter(point[0], point[1], point[2], c = color[2+i+j], linewidth=3)
        for i, seg in enumerate(extra3):
            for j, point in enumerate(seg):
                ax.scatter(point[0], point[1], point[2], c = color[6+i+j], linewidth=3)

        plt.title(id)

        plt.show()

    def test44(self, scatter_points, extra_points1, extra_points2):
        ax = plt.axes(projection='3d')
        color=['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan"]
        # for i, seg in enumerate(scatter_points):
        #     for point in seg:
        #         ax.scatter(point[0], point[1], point[2], color=color[0])
        
        for i, seg in enumerate(extra_points1):
            ax.scatter(seg[0], seg[1], seg[2], c = color[1+i], linewidth=3)
        for i, seg in enumerate(extra_points2):
            for point in seg:
                ax.scatter(point[0], point[1], point[2], c = color[2], linewidth=3)
        # for i, seg in enumerate(extra3):
        #     for point in seg:
        #         ax.scatter(point[0], point[1], point[2], c = color[3], linewidth=3)

        plt.title(id)

        plt.show()

    def test5(self, multi_polygon, id):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates and z-values from the MultiPolygon
        for polygon in multi_polygon.exterior.coords:
            x, y, z = polygon
            # z = [0] * len(x)  # Replace with the actual z-values for each polygon

            # Plot the polygons in 3D
            ax.scatter(x, y, z, linewidth=1)

        # Set labels and display the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(id)
        plt.show()

    def test6(self, multi_polygon, id):

        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ["red", "blue", "green", "orange", "brown", "black"]
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                x, y, z, c = get_coords(roof)
                ax.scatter(x, y, z, c = c/255)        
                if id == "10498821":
                    for i, polygon in enumerate(multi_polygon):
                        x, y, z = zip(*polygon.exterior.coords)
                        ax.plot(x, y, z, color=colors[i], alpha=0.7)

                ax.set_xlabel('X-axis')
                ax.set_ylabel('Y-axis')
                ax.set_zlabel('Z-axis')
                plt.title(id)
                plt.show()

    def test7(self, main, ips):
        ax = plt.axes(projection='3d')
        color=['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan"]
        # for i, seg in enumerate(scatter_points):
        #     for point in seg:
        #         ax.scatter(point[0], point[1], point[2], color=color[0])
        
        for i, seg in enumerate(main):
            ax.scatter(seg[0], seg[1], seg[2], c = color[i], linewidth=3)
        for i, seg in enumerate(ips):
            for point in seg:
                ax.scatter(point[0], point[1], point[2], c = color[6+i], linewidth=3)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

    def visualize_polygons(self, multi, id):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
        for i, polygon in enumerate(multi[0]):
            x, y, z = zip(*polygon.exterior.coords)
            ax.plot(x, y, z, color=colors[1], alpha=0.7)

        # Set axis limits and labels
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)
        # ax.set_zlim(0, 1)  # Adjust the z-axis limits as needed
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title(id)
        # Show the 3D plot
        plt.show()

    def test9(self, multi, id):
    
        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
        for roof in multi:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if type(roof[0]) == Polygon:
                x, y, z = zip(*roof[0].exterior.coords)
                ax.plot(x, y, z, color=colors[1], alpha=0.7)
            else:
                for i, polygon in enumerate(roof[0].geoms):
                    x, y, z = zip(*polygon.exterior.coords)
                    ax.plot(x, y, z, color=colors[1], alpha=0.7)

            # Set axis limits and labels
            # ax.set_xlim(0, 3)
            # ax.set_ylim(0, 3)
            # ax.set_zlim(0, 1)  # Adjust the z-axis limits as needed
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            plt.title(roof[1])
            # Show the 3D plot
            plt.show()


    def plot_footprints_elevated(self, liste, footprint, curr_poly, roof_ids, upper, new_polygon):
            colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna", 'red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
            for id in [roof_ids]:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                z_use = 160
                xmins, xmaxs = [], []
                ymins, ymaxs = [], []
                zmins, zmaxs = [], []
                # for seg in curr_poly["geometry"]:
                #     x, y, z = zip(*seg.exterior.coords)
                #     z_use = z[0]-8
                #     ax.plot(x, y, z, color=colors[1])
                
                for poly in new_polygon:
                    roof_3d = [(x, y, z) for x, y, z in poly.exterior.coords]

                    xmin, xmax = min([coord[0] for coord in roof_3d]), max([coord[0] for coord in roof_3d])
                    ymin, ymax = min([coord[1] for coord in roof_3d]), max([coord[1] for coord in roof_3d])
                    zmin, zmax = min([coord[2] for coord in roof_3d]), max([coord[2] for coord in roof_3d])
                    xmins.append(xmin)
                    xmaxs.append(xmax)
                    ymins.append(ymin)
                    ymaxs.append(ymax)
                    zmins.append(zmin)
                    zmaxs.append(zmax)
                    poly_collection = Poly3DCollection([roof_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)

                    ax.add_collection3d(poly_collection)


                x, y = zip(footprint.exterior.xy)
                ax.scatter(x, y, min(zmins) - 6, color=colors[1])

                for i, p in enumerate(liste):
                    ax.scatter(p[0], p[1], p[2], c = colors[2+i], linewidth=3)
                
                # for i, p in enumerate(upper):
                #     ax.scatter(p[0], p[1], p[2], c = colors[0], linewidth=3)

                ax.set_xlabel('X-axis')
                ax.set_ylabel('Y-axis')
                ax.set_zlabel('Z-axis')
                # fpx, fpy = footprint.exterior.xy
                # footprint_coords = [(x, y, min(zmins)-5) for x, y in zip(fpx, fpy)]
                # footprint_collection = Poly3DCollection([footprint_coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
                # ax.add_collection3d(footprint_collection)
                
                # ax.set_title(f"{roof_type} - {roof_id}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                ax.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
                ax.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
                ax.set_zlim([min(zmins) - 6, max(zmaxs) + 1])
                plt.show()

    def plot_footprints_elevated2(self, liste, footprint, roof, roof_ids, upper, polygon_hm):
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')

        xmins, xmaxs = [], []
        ymins, ymaxs = [], []
        zmins, zmaxs = [], []

        for polygon in polygon_hm:
            roof_3d = [(x, y, z) for x, y, z in polygon.exterior.coords]

            xmin, xmax = min([coord[0] for coord in roof_3d]), max([coord[0] for coord in roof_3d])
            ymin, ymax = min([coord[1] for coord in roof_3d]), max([coord[1] for coord in roof_3d])
            zmin, zmax = min([coord[2] for coord in roof_3d]), max([coord[2] for coord in roof_3d])
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            zmins.append(zmin)
            zmaxs.append(zmax)

            poly_collection = Poly3DCollection([roof_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)

            ax.add_collection3d(poly_collection)

        # fpx, fpy = footprint.exterior.xy
        # footprint_coords = [(x, y, min(zmins)-5) for x, y in zip(fpx, fpy)]
        # footprint_collection = Poly3DCollection([footprint_coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
        # ax.add_collection3d(footprint_collection)
        
        # ax.set_title(f"{roof_type} - {roof_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
        ax.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
        ax.set_zlim([min(zmins) - 6, max(zmaxs) + 1])

        plt.show()

    def visualize_polygons2(self, multi, id):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
        for poly in multi:
            x, y, z = zip(*poly.exterior.coords)
            ax.plot(x, y, z, color=colors[1], alpha=0.7)

        # Set axis limits and labels
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)
        # ax.set_zlim(0, 1)  # Adjust the z-axis limits as needed
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title(id)
        # Show the 3D plot
        plt.show()

    def visualize_things(self, polygon, footprint_points, roof, id):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Iterate through the individual polygons in the MultiPolygon and plot each one
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
        for point in footprint_points:
            x, y, z = point
            ax.scatter(x, y, z, color=colors[0], alpha=0.7)
        
        for point in footprint_points:
            x, y, z = point
            ax.scatter(x, y, z-5, color=colors[1], alpha=0.7)

        for poly in polygon:
            x,y,z = zip(*poly.exterior.coords)
            ax.plot(x, y, z, color=colors[2], alpha=0.7)

        # for poly in roof:
        #     x,y,z = zip(*poly.exterior.coords)
        #     ax.plot(x, y, z, color=colors[3], alpha=0.7)

        # Set axis limits and labels
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)
        # ax.set_zlim(0, 1)  # Adjust the z-axis limits as needed
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title(id)
        # Show the 3D plot
        plt.show()

