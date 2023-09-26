import os
import laspy
import pandas as pd
import geopandas as gpd
import numpy as np
from alphashape import alphashape
from shapely.geometry import Polygon, MultiPolygon
from plot import Plot
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers import get_coords, get_segmented_roof_colors, fkb_to_csv, get_plane_params_for_segment, get_keys, find_highest_z_coord, closest_dist_between_planes


class Roofs:
    def __init__(self) -> None:
        self.roofs = {}
        self.plot = Plot(self)

    # Punkt 1 fra las-fila er (roof.X[0], roof.Y[0], roof.Z[0]) og får fargen [roof.red[0], roof.green[0], roof.blue[0]]
    def create_df_all_roofs(self):
        df_all_roofs = gpd.GeoDataFrame()
        for roof_type in self.roofs:
            for key in self.roofs[roof_type].keys():
                current_seg = self.roofs[roof_type][key]
                red = current_seg.red
                green = current_seg.green
                blue = current_seg.blue
                X = current_seg.X
                Y = current_seg.Y
                Z = current_seg.Z
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
                polygon_df['segment'] = polygon_df.groupby(['R', 'G', 'B']).ngroup()
                grouped = polygon_df.groupby(['roof_id', 'segment'])
                
                df_all_roofs = pd.concat([df_all_roofs, polygon_df])

        df_all_roofs = df_all_roofs.reset_index(drop=True)
        self.df_all_roofs = df_all_roofs

    def join_segments(self):
        for row in self.df_all_roofs.roof_id.unique():
            roof_segments = MultiPolygon()

            # Group the data by RGB values
            grouped = self.df_all_roofs.groupby(['R', 'G', 'B'])

            # Iterate through the groups and create MultiPolygons
            for _, group_data in grouped:
                polygons = list(group_data['Polygon'])
                multi_polygon = MultiPolygon(polygons)
                roof_segments = roof_segments.union(multi_polygon)

            # Extract individual polygons from the merged MultiPolygon
            roof_polygons = list(roof_segments)

    def find_plane_params_to_segment(self):
        planes = gpd.GeoDataFrame()
        for i in range(len(self.df_all_roofs)):
            current_coords = np.array(list(self.df_all_roofs.iloc[i][1].exterior.coords))
            liste = np.array([[c[0] for c in current_coords], [c[1] for c in current_coords], [c[2] for c in current_coords]])
            max_z = find_highest_z_coord(current_coords)
            sol = get_plane_params_for_segment(liste)
            
            df_pd = pd.DataFrame({
                "roof_id": self.df_all_roofs.iloc[i][0],
                "segment": self.df_all_roofs.iloc[i][5],
                "plane_param": [sol],
                "max_z_coordinate": [max_z]
            })
            df = gpd.GeoDataFrame(df_pd)

            planes = pd.concat([planes, df])

        planes = planes.reset_index(drop=True)
        self.df_planes = planes

    def find_plane_intersections(self):
        df_lines = gpd.GeoDataFrame()
        for id in self.roof_ids:
            df = self.df_planes.loc[self.df_planes['roof_id'] == id]
            checked = set()
            segment_number1_counter = 0
            for idx1, segment1 in df.iterrows():
                plane_params1 = segment1[2]
                segment_number2_counter = 0
                plane1 = self.df_all_roofs.loc[self.df_all_roofs['roof_id'] == id]
                plane_1_coords = np.array(plane1.loc[plane1['segment'] == segment_number1_counter].iloc[0][1].exterior.coords)
                for idx2, segment2 in df.iterrows():
                    if (idx1, idx2) not in checked and (idx2, idx1) not in checked and idx1 != idx2:
                        checked.add((idx1, idx2))
                        plane_2_coords = np.array(plane1.loc[plane1['segment'] == segment_number2_counter].iloc[0][1].exterior.coords)
                        closest_dist, i = closest_dist_between_planes(plane_1_coords.tolist(), plane_2_coords.tolist())
                        if closest_dist < 25:
                            plane_params2 = segment2[2]

                            a1, b1, c1, d1 = plane_params1[0], plane_params1[1], plane_params1[2], plane_params1[3]
                            a2, b2, c2, d2 = plane_params2[0], plane_params2[1], plane_params2[2], plane_params2[3]
                            
                            # Step 1: Find the direction vector of the intersection line
                            normal_vector1 = np.array([a1, b1, c1])
                            normal_vector2 = np.array([a2, b2, c2])
                            direction_vector = np.cross(normal_vector1, normal_vector2)

                            # Step 2: Normalize the direction vector
                            # direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
                            
                            # Step 3: Find a point on the intersection line
                            # For simplicity, let's set x = 0 and solve for y and z

                            df_pd = pd.DataFrame({
                                "roof_id": id,
                                "segment_number1": segment_number1_counter,
                                "segment_number2": segment_number2_counter,
                                "dir_vec": [direction_vector],
                                "point_on_plane": [plane_1_coords[i]]
                            })
                            
                            df_gpd = gpd.GeoDataFrame(df_pd)

                            df_lines = pd.concat([df_lines, df_gpd])
                    segment_number2_counter += 1
                segment_number1_counter += 1
                
        df_lines = df_lines.reset_index(drop=True)
        self.df_lines = df_lines
    
    def remove_buildings_not_relevant():
        # find_relevant_buildings()s
        print()

    def run_all(self, FOLDER):
        count = 0
        folder_names = os.listdir(FOLDER)
        for folder in folder_names:
            if folder != '.DS_Store' and folder != 'roof categories.jpg':
                self.roofs[folder] = {}
                roofs = os.listdir(f'{FOLDER}/{folder}')
                for roof in roofs:
                    FULL_PATH = f'{FOLDER}/{folder}/{roof}'
                    las = laspy.read(f'{FULL_PATH}')
                    self.roofs[folder][roof] = las        
        fkb_path = "fkb_processed"
        
        if not os.path.exists(fkb_path):
            os.mkdir(fkb_path)
            fkb_to_csv()
        
        # self.buildings = gpd.read_file(fkb_path)
        self.roof_ids = get_keys(self.roofs)
        self.create_df_all_roofs()
        self.find_plane_params_to_segment()
        self.find_plane_intersections()
        # self.join_segments()

    def plot_result(self, type):      
        match type:
            case "scatterplot_3d":
                self.plot.scatterplot_3d()
            case "segmented_roof_2d":
                    self.plot.plot_2D()
            case "plane_3D":
                for id in self.df_all_roofs.roof_id.unique():
                    roof = self.df_planes.loc[self.df_planes["roof_id"] == id]
                    self.plot.plane_3D(roof)
            case "scatterplot_with_plane_3D_all":
                self.plot.scatterwith_plane_3D_all(self.df_planes, self.roof_ids)
            case "scatter_with_plane_3D_segments":
                self.plot.scatter_with_plane_3D_segments(self.df_all_roofs, self.df_planes)
            case "plane_with_intersections":
                self.plot.plane_with_intersections(self.roof_ids, self.df_planes, self.df_lines)
            case "intersections":
                self.plot.plot_intersections(self.df_lines, self.roof_ids)
            case "line_scatter":
                # self.plot.test( 
                self.plot.line_scatter(self.df_lines, self.roof_ids)

if __name__ == '__main__':
    roofs = Roofs()
    FOLDER = "las_roofs"
    roofs.run_all(FOLDER)
    # roofs.plot_result("scatterplot_3d")
    # roofs.plot_result("segmented_roof_2d")
    # roofs.plot_result("plane_3D")
    # roofs.plot_result("scatterplot_with_plane_3D_all")
    roofs.plot_result("scatter_with_plane_3D_segments")
    # roofs.plot_result("plane_with_intersections")
    # roofs.plot_result("intersections")
    # roofs.plot_result("line_scatter")

    # df_all_roofs
    #           roof_id                                           Geometry    R    G    B  segment
    # 0   182448567  POLYGON Z ((56729848 702656700 16715, 56729867...   31  120  180        0
    # 1   182448567  POLYGON Z ((56731622 702656412 16714, 56731606...   51  160   44        1
    # 2   182448567  POLYGON Z ((56731557 702656356 16734, 56731474...  166  206  227        2
    # 3   182448567  POLYGON Z ((56730023 702657189 16852, 56729966...  178  223  138        3
    # 4   182448567  POLYGON Z ((56730647 702656822 16731, 56730667...  227   26   28        4
    # 5   182448567  POLYGON Z ((56729697 702657517 16750, 56729725...  251  154  153        5
    # 6   182464406  POLYGON Z ((56727064 702564693 15823, 56727111...   31  120  180        0
    # 7   182464406  POLYGON Z ((56727089 702564602 15847, 56727065...   51  160   44        1
    # 8   182464406  POLYGON Z ((56727207 702563155 15827, 56727288...  166  206  227        2
    # ...
    # 45

    # df_planes
    #     roof_id  segment                                        plane_param
    # 0   182448567        0  [0.07889062854689015, -0.3685117045966265, 0.9...
    # 1   182448567        1  [0.3709616943533618, 0.07972128156193396, 0.92...
    # 2   182448567        2  [0.07941985858260067, -0.37753103622809625, 0....
    # 3   182448567        3  [-0.37103021860466256, -0.08133264747545217, 0...
    # 4   182448567        4  [-0.37205612230525026, -0.07853547039464426, 0...
    # 5   182448567        5  [-0.08089365584866737, 0.3710103107331585, 0.9...
    # 6   182464406        0  [0.16591583551180933, 0.42428604395530706, 0.8...
    # 7   182464406        1  [0.42300995866136054, -0.16284619379879833, -0...
    # 8   182464406        2  [-0.16741696920728807, -0.42630100209769334, 0...
    # 9   182464406        3  [-0.16507805921552943, -0.42738005309814964, 0...
    # 10  182464406        4  [-0.4245817765980372, 0.16716106330933364, 0.8...
    # 11  182464406        5  [0.4231040508402451, -0.16549364508384048, 0.8...
    # ...
    # 45

    # df_lines
    # roof_id  segment_number1  segment_number2                                   line_params
    # 0   182448567                0                1  [0, -739669843.5256134, -19573650.557696994]
    # 1   182448567                0                2  [0, -697981820.8423828, -2988153.8566924576]
    # 2   182448567                0                3   [0, -613648621.5145261, 30563641.340343133]
    # 3   182448567                0                4   [0, -614289408.4239951, 30308705.546103183]
    # 4   182448567                0                5   [0, -690400326.6675401, 28128.483703645652]
    # ..        ...              ...              ...                                           ...
    # 81   21088358               39               40  [0, -686603086.3245313, -29653960.403395396]
    # 82   21088358               39               41   [0, -724048670.7930555, -317332.5485034122]
    # 83   21088358               40               41    [0, -723992439.907873, -273318.2947286003]
    # 84  182338605               42               43   [0, -701393281.3805825, -44716.63428679128]
    # 85   10472350               44               45    [0, -830543972.6353568, 26396.53729101586]

    # Corner element - 6
    # Flat - 1
    # T-element - 4
    # Hipped - 4
    # Cross-element - 6
    # Gabled - 2
