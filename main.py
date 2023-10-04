import os
import laspy
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from plot import Plot
from tqdm import tqdm
from helpers import get_coords, get_segmented_roof_colors, fkb_to_csv, get_plane_params_for_segment, get_keys, find_highest_lowest_z_coord, closest_dist_between_planes, closest_dist_lines, check_distance
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import copy

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
                        'geometry': polygon,
                        'R': r,
                        'G': g,
                        'B': b
                    }

                    # Append the dictionary to the list
                    polygons_data.append(row_data)

                # Create a new DataFrame from the list of dictionaries
                polygon_df = gpd.GeoDataFrame(polygons_data)
                polygon_df['segment'] = polygon_df.groupby(['R', 'G', 'B']).ngroup()
                grouped = polygon_df.groupby(['roof_id', 'segment'])
                
                df_all_roofs = pd.concat([df_all_roofs, polygon_df])

        df_all_roofs = df_all_roofs.reset_index(drop=True)
        self.df_all_roofs = gpd.GeoDataFrame(df_all_roofs)

    def find_plane_params_to_segment(self):
        planes = gpd.GeoDataFrame()
        for i in range(len(self.df_all_roofs)):
            current_coords = np.array(list(self.df_all_roofs.iloc[i][1].exterior.coords))
            liste = np.array([[c[0] for c in current_coords], [c[1] for c in current_coords], [c[2] for c in current_coords]])
            max_z, min_z = find_highest_lowest_z_coord(current_coords)
            sol = get_plane_params_for_segment(liste)
            
            df_pd = pd.DataFrame({
                "roof_id": self.df_all_roofs.iloc[i][0],
                "segment": self.df_all_roofs.iloc[i][5],
                "plane_param": [sol],
                "max_z_coordinate": [max_z],
                "min_z_coordinate": [min_z]
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
                        if closest_dist < 35:
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
    
    def find_intersection_points(self):
        df_intersection_points = gpd.GeoDataFrame()
        for id in self.roof_ids:
            df_curr = self.df_lines.loc[self.df_lines['roof_id'] == id]
            segment_numbers = list(set(df_curr['segment_number1']).union(df_curr['segment_number2']))
            points = []
            for number in segment_numbers:
                curr_plane = self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes["segment"] == number)]
                min_z, max_z = curr_plane["min_z_coordinate"].tolist()[0][2], curr_plane["max_z_coordinate"].tolist()[0][2]
                df_curr_segments = df_curr.loc[df_curr['segment_number1'] == number]
                checked = set()
                for i in range(len(df_curr_segments)):
                    for j in range(len(df_curr_segments)):
                        if (i, j) not in checked and (j, i) not in checked and i != j:
                            checked.add((i, j))
                            closest_point = closest_dist_lines(df_curr_segments.iloc[i], df_curr_segments.iloc[j], min_z, max_z)
                            if closest_point[0][2] <= (max_z + 20) and closest_point[0][2] >= (min_z - 20):
                                points.append(closest_point[0])
            df = pd.DataFrame({
                "roof_id": id,
                "points": [points]
                })
            df_intersection_points = pd.concat([df_intersection_points, df])

        df_intersection_points = df_intersection_points.reset_index(drop=True)
        for idx, row in df_intersection_points.iterrows():
            curr_points = row[1]
            checked = set()
            # Checks wheter there are multiple points within a certain range which 
            # are supposed to be the same intersection point
            for i in range(len(curr_points)):
                for j in range(len(curr_points)):
                    if (i, j) not in checked and (j, i) not in checked and i != j:
                        checked.add((i, j))
                        dist = check_distance(curr_points[i], curr_points[j])
                        if dist < 40: 
                            current = [curr_points[i], curr_points[j]]
                            avg = np.mean(current, axis = 0)
                            new_point_for_curr = copy.deepcopy(curr_points)
                            new_point_for_curr.pop(i)
                            new_point_for_curr.pop(j - 1)
                            new_point_for_curr.append(avg)
                            df_intersection_points.at[idx, "points"] = new_point_for_curr
            updated_curr_points = df_intersection_points.iloc[idx]['points']
            
            # For cross element, which only is supposed to have one intersection point
            if idx == 8 or idx == 9:
                avg_point = np.mean(np.array(updated_curr_points), axis = 0)
                df_intersection_points.iloc[idx]['points'] = [avg_point]
            elif len(updated_curr_points) > 1:
                # For T-elements which have low t-element, and there shouldn't be more than 1 intersection point 
                # It drops the highest z-value intersection points 
                if (idx == 4 or idx == 5):
                    lowest_z_point = None
                    lowest_z_value = float('inf')
                    for point in updated_curr_points: 
                        z = point[2]
                        if z < lowest_z_value:
                            lowest_z_value = z
                            lowest_z_point = point

                    df_intersection_points.at[idx, 'points'] = [lowest_z_point]
                # For all other roofs, the top z-value of the intersection point is set to the same
                else:
                    updated_curr_points = np.array(updated_curr_points)
                    average_z = np.mean(updated_curr_points[:, 2])
                    new_points = np.column_stack((updated_curr_points[:, 0], updated_curr_points[:, 1], np.full(updated_curr_points.shape[0], average_z)))
                    df_intersection_points.at[idx, 'points'] = new_points


        self.df_intersection_points = df_intersection_points

    def find_relevant_buildings_for_modelling(self):
        df_all_roofs = self.df_all_roofs.copy()
        df_all_roofs.crs = 25832
        df_all_roofs = df_all_roofs.to_crs(4326)
        # print(df_all_roofs)

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
        fkb_path = "fkb_trondheim"
        
        if not os.path.exists(fkb_path):
            os.mkdir(fkb_path)
            fkb_to_csv()
        
        self.buildings_from_file = gpd.read_file(fkb_path)
        self.roof_ids = get_keys(self.roofs)
        self.create_df_all_roofs()
        self.find_plane_params_to_segment()
        self.find_plane_intersections()
        self.find_intersection_points()
        # self.find_relevant_buildings_for_modelling()

    def plot_result(self, type):      
        match type:
            case "scatterplot_3d":
                self.plot.scatterplot_3d()
            case "segmented_roof_2d":
                    self.plot.plot_2D()
            case "plane_3D":
                for id in self.df_all_roofs.roof_id.unique():
                    roof = self.df_planes.loc[self.df_planes["roof_id"] == id]
                    self.plot.plane_3D(roof, id)
            case "scatterplot_with_plane_3D_all":
                self.plot.scatterwith_plane_3D_all(self.df_planes, self.roof_ids)
            case "scatter_with_plane_3D_segments":
                self.plot.scatter_with_plane_3D_segments(self.df_all_roofs, self.df_planes)
            case "plane_with_intersections":
                self.plot.plane_with_intersections(self.roof_ids, self.df_planes, self.df_lines)
            case "intersections":
                self.plot.plot_intersections(self.df_lines, self.roof_ids)
            case "line_scatter":
                self.plot.line_scatter(self.df_lines, self.roof_ids)
            case "line_scatter_intersection_points":
                self.plot.line_scatter_intersection_points(self.df_lines, self.roof_ids, self.df_intersection_points)


if __name__ == '__main__':
    roofs = Roofs()
    FOLDER = "las_roofs"
    roofs.run_all(FOLDER)
    # roofs.plot_result("scatterplot_3d")
    # roofs.plot_result("segmented_roof_2d")
    # roofs.plot_result("plane_3D")
    # roofs.plot_result("scatterplot_with_plane_3D_all")
    # roofs.plot_result("scatter_with_plane_3D_segments")
    # roofs.plot_result("plane_with_intersections")
    # roofs.plot_result("intersections")
    # roofs.plot_result("line_scatter")
    # roofs.plot_result("line_scatter_intersection_points")

    # df_all_roofs
    #       roof_id                                           Geometry    R    G    B  segment
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
    #       roof_id  segment                                        plane_param                    max_z_coordinate
    # 0   182448567        0  [-0.2595999960440571, 1.2127394753910943, -3.0...  [56730833.0, 702657350.0, 16901.0]
    # 1   182448567        1  [-15.765312162402822, -3.388012132379369, -39....  [56731081.0, 702656798.0, 16910.0]
    # 2   182448567        2  [-0.5159573970110686, 2.4524695076884435, -5.9...  [56731115.0, 702656650.0, 16893.0]
    # 3   182448567        3  [11.675750804652164, 2.5596137097414777, -29.1...  [56730142.0, 702657200.0, 16906.0]
    # 4   182448567        4  [4.158039255256885, 0.8776678960567065, -10.33...  [56730965.0, 702657347.0, 16905.0]
    # 5   182448567        5  [0.4281721854354697, -1.9637740063686988, -4.8...  [56730313.0, 702657252.0, 16908.0]
    # 6   182464406        0  [-0.07841090311460423, -0.20050697222486438, -...  [56727451.0, 702564107.0, 16042.0]
    # 7   182464406        1  [20.039118763673624, -7.714463344694007, -42.2...  [56727270.0, 702564158.0, 16016.0]
    # ...
    # 45

    # df_lines
    #       roof_id  segment_number1  segment_number2                                            dir_vec                      point_on_plane
    # 0   182448567                0                3  [-27.500302609886827, -43.147498456997425, -14...  [56729917.0, 702656818.0, 16772.0]
    # 1   182448567                0                4  [-9.859597130018683, -15.357976616166322, -5.2...  [56730787.0, 702657091.0, 16810.0]
    # 2   182448567                0                5  [-11.92444499431495, -2.576348338814526, -0.00...  [56730833.0, 702657350.0, 16901.0]
    # 3   182448567                1                2  [116.73596492964535, -74.18947588176178, -40.4...  [56731541.0, 702656391.0, 16754.0]
    # 4   182448567                1                4  [69.5291066172534, -326.4477241891532, 0.25077...  [56731019.0, 702657102.0, 16905.0]
    # 5   182448567                1                5  [-60.626791313394854, -94.03337294550472, 32.4...  [56731261.0, 702657887.0, 16741.0]
    # 6   182448567                2                4  [-20.089244808542563, -30.250962756923574, -10...  [56730819.0, 702656283.0, 16758.0]
    # 7   182448567                3                5  [-69.69868964957784, 44.708176642661094, -24.0...  [56730123.0, 702657241.0, 16899.0]
    # 8   182464406                0                1  [5.221402546752168, -11.741135660910127, 4.622...  [56727264.0, 702564261.0, 15992.0]
    # 9   182464406                0                3  [0.05390384194116137, -0.02095248848083775, -6...  [56727668.0, 702564017.0, 16031.0]
    # 10  182464406                0                5  [1.638531036876076, 0.7137894753837106, -0.645...  [56728430.0, 702564168.0, 15822.0]

    # Corner element - 6
    # Flat - 1
    # T-element - 4
    # Hipped - 4
    # Cross-element - 6
    # Gabled - 2
