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

def get_coords(roof_seg, segmented_first):
    # Extract all points to get a nice scatter plot

    if segmented_first:
        x = list(np.concatenate([list(cloud.X) for cloud in roof_seg]).flat)
        y = list(np.concatenate([list(cloud.Y) for cloud in roof_seg]).flat)
        z = list(np.concatenate([list(cloud.Z) for cloud in roof_seg]).flat)
        c = list(np.concatenate([[roof_seg.index(cloud) for i in list(cloud.X)] for cloud in roof_seg]).flat)

        # # Used to adjust the size of the scatterplot points
        ys = np.random.randint(100, 200, len(x))
        zs = np.random.randint(25, 150, len(y))
        s = zs / ((ys * 0.01) ** 2)
    else: 
        x = roof_seg.X
        y = roof_seg.Y
        z = roof_seg.Z
        c = list(np.zeros(len(roof_seg.X)))
        ys = np.random.randint(100, 200, len(x))
        zs = np.random.randint(25, 150, len(y))
        s = zs / ((ys * 0.05) ** 2)
    return x, y, z, c, s

class Roofs:
    def __init__(self) -> None:
        self.roofs = {}
        self.plot = Plot(self)

    def create_df_all_roofs(self):
        df_all_roofs = gpd.GeoDataFrame()

        for roof in self.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs[roof].values()]
            df = self.alpha_shapes(roof_segments)
            df['roof_id'] = roof
            z_coords = [point[2] for point in list(df.iloc[0].geometry.exterior.coords)]
            df["min_z"] = min(z_coords)
            df_all_roofs = pd.concat([df_all_roofs, df])

        df_all_roofs = df_all_roofs.reset_index(drop=True)
        self.df_all_roofs = df_all_roofs

    def alpha_shapes(self, segments):
        alpha_shape = [alphashape(seg.xyz, 0) for seg in segments]
        list = [{'geometry': alpha_shape[i], 'classification': i} for i in range(len(alpha_shape))]
        df = gpd.GeoDataFrame(list)
        threshold = lambda x: 0.7 if x.area > 10 else 0.4
        df.geometry = df.geometry.apply(lambda x: x.simplify(threshold(x), preserve_topology=False))
        return df
    
    def clustering(self):
        # Gå gjennom hvert tak, og for hvert tak lag en liste med punkter 

        print(self.df_all_roofs)
        for roof in self.df_all_roofs.roof_id.unique():
            segments = self.df_all_roofs.loc[self.df_all_roofs["roof_id"] == roof]
            coords = []

            for i in range(len(segments)):
                temp_coords = [coord for coord in segments.iloc[i]]
                coords.append(temp_coords)
            
            # print(roof)
            # print("__")
            # print(self.df_all_roofs.loc[self.df_all_roofs["roof_id"] == roof])
            break


    def run_all(self, FOLDER):
        count = 0
        folder_names = os.listdir(FOLDER)

        for folder in folder_names:
            if folder != '.DS_Store' and folder != 'roof categories.jpg' and count < 100:
                self.roofs[folder] = {}
                roofs = os.listdir(f'{FOLDER}/{folder}')
                for roof in roofs:
                    FULL_PATH = f'{FOLDER}/{folder}'
                    las = laspy.read(f'{FULL_PATH}/{roof}')
                    # points = pd.DataFrame(las.xyz, columns=['x', 'y', 'z'])
                    self.roofs[folder][roof] = las
                count += 1

        self.create_df_all_roofs()
        self.clustering()

    def plot_result(self, type):
        match type:
            case "scatterplot_3d":
                for roofs in self.roofs:
                    roof_segments = [roof_segment for roof_segment in self.roofs[roofs].values()]
                    for roof in roof_segments:
                        print(roof)
                        self.plot.scatterplot_3d(*get_coords(roof, False))
                
                # When we have the roofs segmented beforehand  
                # for roof in self.roofs:
                #     roof_segments = [roof_segment for roof_segment in self.roofs[roof].values()]
                #     self.plot.scatterplot_3d(*get_coords(roof_segments, True))
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