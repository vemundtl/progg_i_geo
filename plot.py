import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import get_coords
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely import Polygon
import os

class Plot:
    def __init__(self, roofs) -> None:
        self.roofs = roofs

        if not os.path.exists("./plots"):
            os.mkdir("./plots")

    def scatterplot_3d(self):
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            keys = list(self.roofs.roofs[roofs].keys())
            
            for i, roof in enumerate(roof_segments):
                x, y, z, c = get_coords(roof)
                ax = plt.axes(projection='3d')
                ax.scatter(x, y, z, c = c/255)

                plt.title(keys[i])
                if not os.path.exists("./plots/scatterplots"):
                    os.mkdir("./plots/scatterplots")
                plt.savefig(f"./plots/scatterplots/{roofs}_{keys[i]}.png", bbox_inches='tight')

                plt.show()

    def plane_3D(self, roof_ids, df_planes, df_all_roofs):
        for id in roof_ids:
            if id == "182341061":
                continue

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            colors = ['red', 'cyan', 'green', 'yellow', 'purple', 'orange', 'blue']

            min_boundaries = np.inf * np.ones(3)
            max_boundaries = -np.inf * np.ones(3)

            planes = df_planes.loc[df_planes["roof_id"] == id]["plane_param"]
            roof_type = df_all_roofs.loc[(df_all_roofs["roof_id"] == id)]["roof_type"].iloc[0]

            for i, plane_params in enumerate(planes):
                A, B, C, D = plane_params

                curr_points = df_all_roofs.loc[(df_all_roofs["roof_id"] == id) & (df_all_roofs["segment"] == i)]["geometry"].iloc[0].exterior.coords

                x, y, z = [point[0] for point in curr_points], [point[1] for point in curr_points], [point[2] for point in curr_points]

                min_coord = [np.min(x), np.min(y), np.min(z)]
                max_coord = [np.max(x), np.max(y), np.max(z)]

                min_boundaries = np.minimum(min_coord, min_boundaries)
                max_boundaries = np.maximum(max_coord, max_boundaries)

                x_grid = np.linspace(min_coord[0], max_coord[0])
                y_grid = np.linspace(min_coord[1], max_coord[1])
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = (-A*X - B*Y - D) / C

                ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.4)
                ax.scatter(x, y, z, s=10/4, c=colors[-i])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim([min_boundaries[0] - 1, max_boundaries[0] + 1])
            ax.set_ylim([min_boundaries[1] - 1, max_boundaries[1] + 1])
            ax.set_zlim([min_boundaries[2] - 1, max_boundaries[2] + 1])

            plt.title(id)
            if not os.path.exists("./plots/plane_3D"):
                os.mkdir("./plots/plane_3D")
            plt.savefig(f"./plots/plane_3D/{roof_type}_{id}.png", bbox_inches='tight')


            plt.show()

    def line_scatter(self, df_lines, ids):
        counter = 0 
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                df_curr = df_lines.loc[df_lines['roof_id'] == ids[counter]]
                x, y, z, c = get_coords(roof)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c = c/255)

                names = self.roofs.roofs.keys()
                data = None
                for name in names:
                    if f'{ids[counter]}' in self.roofs.roofs[name]:
                        data = self.roofs.roofs[name][f'{ids[counter]}']

                if not df_curr.empty:
                    for i in range(len(df_curr)):
                        direction = np.array(df_curr.iloc[i][3])
                        point_on_plane = df_curr.iloc[i][4]
                        t_values = np.linspace(-10000,10000,1000)
                        intersection_points = []

                        for t in t_values:
                            intersection_point = point_on_plane + t * direction
                            intersection_points.append(intersection_point)

                        intersection_points = np.array(intersection_points)
                        ax.plot(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='r', linewidth=3)

                x_min, x_max = min(data.x), max(data.x)
                y_min, y_max = min(data.y), max(data.y)
                z_min, z_max = min(data.z), max(data.z)
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_min, z_max])

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.title(ids[counter])
                ax.legend()
                if not os.path.exists("./plots/line_scatter"):
                    os.mkdir("./plots/line_scatter")
                plt.savefig(f"./plots/line_scatter/{roofs}_{ids[counter]}.png", bbox_inches='tight')
                plt.show()
                counter += 1

    def line_scatter_intersection_points(self, df_lines, ids, df_intersection_points):
        counter = 0 
        for roofs in self.roofs.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs.roofs[roofs].values()]
            for roof in roof_segments:
                df_curr = df_lines.loc[df_lines['roof_id'] == ids[counter]]
                closest_points = df_intersection_points.loc[df_intersection_points['roof_id'] == ids[counter]].iloc[0]['points']
                x, y, z, c = get_coords(roof)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c = c/255)

                names = self.roofs.roofs.keys()
                data = None
                for name in names:
                    if f'{ids[counter]}' in self.roofs.roofs[name]:
                        data = self.roofs.roofs[name][f'{ids[counter]}']

                    for point in closest_points:
                        ax.scatter(point[0], point[1], point[2], color='r', linewidth=15)
                x_min, x_max = min(data.x), max(data.x)
                y_min, y_max = min(data.y), max(data.y)
                z_min, z_max = min(data.z), max(data.z)
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                ax.set_zlim([z_min, z_max])

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.title(ids[counter])
                if not os.path.exists("./plots/intersection_points"):
                    os.mkdir("./plots/intersection_points")
                plt.savefig(f"./plots/intersection_points/{roofs}_{ids[counter]}.png", bbox_inches='tight')
                plt.show()
                counter += 1

    def plot_roof_polygons(self, roof, roof_ids, df_all_roofs):
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
        for id in roof_ids:
            if id == "182341061":
                continue
            curr_roof = roof.loc[roof["roof_id"] == id]["geometry"]
            roof_type = df_all_roofs.loc[df_all_roofs["roof_id"] == id].iloc[0]["roof_type"]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for seg in curr_roof:
                x, y, z = zip(*seg.exterior.coords)
                ax.plot(x, y, z, color=colors[1], alpha=0.7)

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            plt.title(id)

            if not os.path.exists("./plots/roof_polygons"):
                os.mkdir("./plots/roof_polygons")
            plt.savefig(f"./plots/roof_polygons/{roof_type}_{id}.png", bbox_inches='tight')
            plt.show()

    def plot_footprint_and_roof(self, footprint, roof, roof_ids):
        colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
        for id in roof_ids:
            curr_roof = roof.loc[roof["roof_id"] == id]["geometry"]
            curr_footprint = footprint.loc[footprint["roof_id"] == id]["footprint"].iloc[0]
            roof_type = footprint.loc[footprint["roof_id"] == id]["type"].iloc[0]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            z_use = 0
            for seg in curr_roof:
                x, y, z = zip(*seg.exterior.coords)
                z_use = z[0]-8
                ax.plot(x, y, z, color=colors[1], alpha=0.7)

            x, y = curr_footprint.exterior.xy
            z = [z_use for i in range(len(x))]
            ax.plot(x, y, z_use, color=colors[0], alpha=0.7)

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            plt.title(id)
            if not os.path.exists("./plots/roof_with_footprint"):
                os.mkdir("./plots/roof_with_footprint")
            plt.savefig(f"./plots/roof_with_footprint/{roof_type}_{id}.png", bbox_inches='tight')
            # plt.show()

    def plot_entire_roof(self, df_walls, df_polygon, roof_ids, df_adjusted_roof_planewise, roof_obj):
        for id in roof_ids:
            if id == "182341061":
                continue
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection='3d')
            xmins, xmaxs = [], []
            ymins, ymaxs = [], []
            zmins, zmaxs = [], []
            curr_poly = df_polygon.loc[df_polygon["roof_id"] == id]["geometry"].iloc[0]

            curr_poly = roof_obj.adjusted_roof.loc[roof_obj.adjusted_roof["roof_id"] == id]["geometry"].iloc[0]
            roof_type = roof_obj.df_footprints.loc[roof_obj.df_footprints["roof_id"] == id]["type"].iloc[0]

            if len(curr_poly) == 0:
                polygon = df_adjusted_roof_planewise.loc[df_adjusted_roof_planewise["roof_id"] == id]["lower_roof_top_points"].iloc[0]
                roof_3d = [(x, y, z) for x, y, z in polygon]
                poly_collection = Poly3DCollection([roof_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
                ax.add_collection3d(poly_collection)

                xmin, xmax = min([coord[0] for coord in roof_3d]), max([coord[0] for coord in roof_3d])
                ymin, ymax = min([coord[1] for coord in roof_3d]), max([coord[1] for coord in roof_3d])
                zmin, zmax = min([coord[2] for coord in roof_3d]), max([coord[2] for coord in roof_3d])

                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                zmins.append(zmin)
                zmaxs.append(zmax)

            for polygon in curr_poly:
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

            curr_wall = df_walls.loc[df_walls["roof_id"] == id]["walls"].iloc[0]

            for poly in curr_wall:
                roof_3d = [(x, y, z) for x, y, z in poly.exterior.coords]

                poly_collection = Poly3DCollection([roof_3d], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)

                ax.add_collection3d(poly_collection)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
            ax.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
            ax.set_zlim([min(zmins) - 10, max(zmaxs) + 1])

            plt.title(id)
            if not os.path.exists("./plots/entire_roof"):
                os.mkdir("./plots/entire_roof")
            plt.savefig(f"./plots/entire_roof/{roof_type}_{id}.png", bbox_inches='tight')

            # plt.show()


    def plot_entire_roof2(self, df_walls, df_polygon, roof_ids, df_adjusted_roof_planewise, roof_obj):
        for id in roof_ids:
            if id == "182341061":
                continue
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection='3d')
            xmins, xmaxs = [], []
            ymins, ymaxs = [], []
            zmins, zmaxs = [], []
            curr_poly = df_polygon
            roof_type = roof_obj.df_footprints.loc[roof_obj.df_footprints["roof_id"] == id]["type"].iloc[0]
            if len(curr_poly) == 0:
                polygon = df_adjusted_roof_planewise.loc[df_adjusted_roof_planewise["roof_id"] == id]["lower_roof_top_points"].iloc[0]
                roof_3d = [(x, y, z) for x, y, z in polygon]
                poly_collection = Poly3DCollection([roof_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
                ax.add_collection3d(poly_collection)

                xmin, xmax = min([coord[0] for coord in roof_3d]), max([coord[0] for coord in roof_3d])
                ymin, ymax = min([coord[1] for coord in roof_3d]), max([coord[1] for coord in roof_3d])
                zmin, zmax = min([coord[2] for coord in roof_3d]), max([coord[2] for coord in roof_3d])

                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                zmins.append(zmin)
                zmaxs.append(zmax)

            for polygon in curr_poly:
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

            curr_wall = df_walls.loc[df_walls["roof_id"] == id]["walls"].iloc[0]

            for poly in curr_wall:
                roof_3d = [(x, y, z) for x, y, z in poly.exterior.coords]

                poly_collection = Poly3DCollection([roof_3d], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)

                ax.add_collection3d(poly_collection)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
            ax.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
            ax.set_zlim([min(zmins) - 10, max(zmaxs) + 1])

            plt.title(id)
            if not os.path.exists("./plots/entire_roof"):
                os.mkdir("./plots/entire_roof")
            plt.savefig(f"./plots/entire_roof/{roof_type}_{id}.png", bbox_inches='tight')

            # plt.show()

