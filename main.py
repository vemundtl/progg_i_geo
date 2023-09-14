import os
import laspy
import pandas as pd
import geopandas as gpd
import numpy as np
from alphashape import alphashape
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from plot import Plot
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import itertools
from tqdm import tqdm

def get_coords(roof_seg):
    # Extract all points to get a nice scatter plot

    x = roof_seg.X
    y = roof_seg.Y
    z = roof_seg.Z
    c = np.array([[color.red, color.green, color.blue] for color in roof_seg])
    return x, y, z, c

def get_segmented_roof_colors(las):
    c = [[color.red, color.green, color.blue] for color in las]
    unique_lists = set()

    for inner_list in c:
        inner_tuple = tuple(inner_list)

        if inner_tuple not in unique_lists:
            unique_lists.add(inner_tuple) 

    return list(unique_lists)

class Roofs:
    def __init__(self) -> None:
        self.roofs = {}
        self.plot = Plot(self)

    # Punkt 1 er (roof.X, roof.Y, roof.Z) og får fargen [roof.red, roof.green, roof.blue]

    def create_df_all_roofs(self):
        df_all_roofs = gpd.GeoDataFrame()
        for roof_type in self.roofs:
            for key in self.roofs[roof_type].keys():
                red = self.roofs[roof_type][key].red
                green = self.roofs[roof_type][key].green
                blue = self.roofs[roof_type][key].blue
                X = self.roofs[roof_type][key].X
                Y = self.roofs[roof_type][key].Y
                Z = self.roofs[roof_type][key].Z
                merged_data = [(x, y, z, r, g, b) for r, g, b, x, y, z in zip(red, green, blue, X, Y, Z)]
                
                rgb_to_label = {}
                label_counter = 0
                for r, g, b, x, y, z in merged_data:
                    rgb_tuple = (r, g, b)
                    if rgb_tuple not in rgb_to_label:
                        rgb_to_label[rgb_tuple] = label_counter
                        label_counter += 1

                # Create a list of labels corresponding to each point
                labels = [rgb_to_label[(r, g, b)] for r, g, b, x, y, z in merged_data]

                # Create a pandas DataFrame with the data and labels
                df = pd.DataFrame(merged_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B'])

                df['Label'] = labels

                grouped = df.groupby(['R', 'G', 'B'])
                polygons_data = []

                for (r, g, b), group_data in grouped:
                    # Extract X, Y, Z values from the group
                    points = [(x, y, z) for x, y, z in zip(group_data['X'], group_data['Y'], group_data['Z'])]

                    # Create a Shapely polygon from the points
                    polygon = Polygon(points)

                    # Create a dictionary for the new DataFrame row
                    row_data = {
                        "roof_id": key[:-4],
                        'Geometry': polygon,
                        'R': r,
                        'G': g,
                        'B': b
                    }

                    # Append the dictionary to the list
                    polygons_data.append(row_data)

                # Create a new DataFrame from the list of dictionaries
                polygon_df = pd.DataFrame(polygons_data)
                df_all_roofs = pd.concat([df_all_roofs, polygon_df])

        df_all_roofs = df_all_roofs.reset_index(drop=True)
        self.df_all_roofs = df_all_roofs

    def run_all(self, FOLDER):
        count = 0
        folder_names = os.listdir(FOLDER)
        for folder in folder_names:
            if folder != '.DS_Store' and folder != 'roof categories.jpg':
                self.roofs[folder] = {}
                roofs = os.listdir(f'{FOLDER}/{folder}')
                for roof in roofs:
                    FULL_PATH = f'{FOLDER}/{folder}'
                    las = laspy.read(f'{FULL_PATH}/{roof}')
                    # points = pd.DataFrame(las.xyz, columns=['x', 'y', 'z'])
                    self.roofs[folder][roof] = las

        self.create_df_all_roofs()
        # self.clustering()

    def plot_result(self, type):
        match type:
            case "scatterplot_3d":
                for roofs in self.roofs:
                    roof_segments = [roof_segment for roof_segment in self.roofs[roofs].values()]
                    for roof in roof_segments:
                        self.plot.scatterplot_3d(*get_coords(roof))
            case "segmented_roof_2d":
                for roof in self.roofs:
                    roof_segments = [roof_segment for roof_segment in self.roofs[roof].values()]
                    self.plot.plot_2D(roof_segments)

if __name__ == '__main__':
    roofs = Roofs()
    FOLDER = "las_roofs"
    roofs.run_all(FOLDER)
    # roofs.plot_result("segmented_roof_2d")
    # roofs.plot_result("scatterplot_3d")

    # Corner element - 6
    # Flat - 1
    # T-element - 4
    # Hipped - 4
    # Cross-element - 6
    # Gabled - 2
