import os
import laspy
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Polygon, MultiPolygon, Point, MultiPoint
from plot import Plot
from helpers import *
import copy
from shapely.ops import unary_union

class Roofs:
    def __init__(self) -> None:
        self.roofs = {}
        self.plot = Plot(self)

    def create_df_all_roofs(self):
        """Creates a dataframe for each segment of each roof """
        df_all_roofs = pd.DataFrame()
        for roof_type in self.roofs:
            for key, current_seg in self.roofs[roof_type].items():
                merged_data = [(x, y, z, r, g, b) for r, g, b, x, y, z in zip(current_seg.red, current_seg.green, current_seg.blue, current_seg.x, current_seg.y, current_seg.z)]
                rgb_to_label = {}
                label_counter = 0
                labels = []
                polygons_data = []

                for r, g, b, x, y, z in merged_data:
                    rgb_tuple = (r, g, b)
                    if rgb_tuple not in rgb_to_label:
                        rgb_to_label[rgb_tuple] = label_counter
                        label_counter += 1
                    labels.append(rgb_to_label[rgb_tuple])

                df = pd.DataFrame(merged_data, columns=['X', 'Y', 'Z', 'R', 'G', 'B'])
                df['Label'] = labels

                grouped = df.groupby(['R', 'G', 'B'])

                for (r, g, b), group_data in grouped:
                    points = [Point(x, y, z) for x, y, z in zip(group_data['X'], group_data['Y'], group_data['Z'])]
                    polygon = Polygon(points)
                    row_data = {"roof_id": key, 'geometry': polygon, 'R': r, 'G': g, 'B': b}
                    polygons_data.append(row_data)

                polygon_df = pd.DataFrame(polygons_data)
                polygon_df['segment'] = polygon_df.groupby(['R', 'G', 'B']).ngroup()
                polygon_df["roof_type"] = roof_type
                grouped = polygon_df.groupby(['roof_id', 'segment'])
                df_all_roofs = pd.concat([df_all_roofs, polygon_df])

        df_all_roofs.reset_index(drop=True, inplace=True)
        self.df_all_roofs = gpd.GeoDataFrame(df_all_roofs)

    def find_plane_params_to_segment(self):
        """Finds the plane parameters of a segment in each roof segment where the parameters are Ax + By + Cz = D"""
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
                "min_z_coordinate": [min_z],
                "roof_type": self.df_all_roofs.iloc[i]["roof_type"]
            })
            df = gpd.GeoDataFrame(df_pd)

            planes = pd.concat([planes, df])

        planes = planes.reset_index(drop=True)
        self.df_planes = planes

    def find_plane_intersections(self):
        """Finds intersections between each plane, creating lines on the intersection point"""
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
                        if closest_dist < 0.4:
                            plane_params2 = segment2[2]

                            a1, b1, c1, d1 = plane_params1[0], plane_params1[1], plane_params1[2], plane_params1[3]
                            a2, b2, c2, d2 = plane_params2[0], plane_params2[1], plane_params2[2], plane_params2[3]

                            normal_vector1 = np.array([a1, b1, c1])
                            normal_vector2 = np.array([a2, b2, c2])
                            direction_vector = np.cross(normal_vector1, normal_vector2)

                            direction_vector /= np.linalg.norm(direction_vector)
                            
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
        """Find intersection points, aka where there are supposed to be intersections between plane segments"""
        intersection_points = gpd.GeoDataFrame()
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
                            if closest_point[0][2] <= (max_z + 1) and closest_point[0][2] >= (min_z - 1):
                                points.append(closest_point[0])
            df = pd.DataFrame({
                "roof_id": id,
                "points": [points], 
                "roof_type": self.df_all_roofs.loc[self.df_all_roofs["roof_id"] == id].iloc[0]["roof_type"]
                })
            intersection_points = pd.concat([intersection_points, df])

        intersection_points = intersection_points.reset_index(drop=True)
        for idx, row in intersection_points.iterrows():
            curr_points = row[1]
            checked = set()
            for i in range(len(curr_points)):
                for j in range(len(curr_points)):
                    if (i, j) not in checked and (j, i) not in checked and i != j:
                        checked.add((i, j))
                        dist = check_distance(curr_points[i], curr_points[j])
                        if dist < 1: 
                            current = [curr_points[i], curr_points[j]]
                            avg = np.mean(current, axis = 0)
                            new_point_for_curr = copy.deepcopy(curr_points)
                            new_point_for_curr.pop(i)
                            new_point_for_curr.pop(j - 1)
                            new_point_for_curr.append(avg)
                            intersection_points.at[idx, "points"] = new_point_for_curr
            updated_curr_points = intersection_points.iloc[idx]['points']
            
            if row["roof_type"] == "cross_element":
                avg_point = np.mean(np.array(updated_curr_points), axis = 0)
                intersection_points.iloc[idx]['points'] = [avg_point]
            elif len(updated_curr_points) > 1:
                if (row["roof_type"] == "t-element"):
                    lowest_z_point = None
                    lowest_z_value = float('inf')
                    for point in updated_curr_points: 
                        z = point[2]
                        if z < lowest_z_value:
                            lowest_z_value = z
                            lowest_z_point = point

                    intersection_points.at[idx, 'points'] = [lowest_z_point]
                else:
                    updated_curr_points = np.array(updated_curr_points)
                    average_z = np.mean(updated_curr_points[:, 2])
                    new_points = np.column_stack((updated_curr_points[:, 0], updated_curr_points[:, 1], np.full(updated_curr_points.shape[0], average_z)))
                    intersection_points.at[idx, 'points'] = new_points

        self.intersection_points = intersection_points

    def find_segments_to_match_intersection_point(self):
        """Finds out which segments that should intersect in which intersection point"""
        df_intersection_with_segments = pd.DataFrame()
        for id in self.roof_ids:
            curr_inter = self.intersection_points.loc[self.intersection_points['roof_id'] == id].iloc[0][1]
            segments_to_point = []
            points = []
            for point in curr_inter:
                curr_segments_to_point = set()
                curr_lines = self.df_lines.loc[self.df_lines['roof_id'] == id]
                for idx, row in curr_lines.iterrows():
                    dist = shortest_dist_point_line(row[3], row[4], point)
                    if dist < 1:
                        curr_segments_to_point.add(row[1])
                        curr_segments_to_point.add(row[2])
                segments_to_point.append(list(curr_segments_to_point))
                points.append(point)
            
            temp_df = pd.DataFrame({
                "roof_id": id,
                "points": [points],
                "segments": [segments_to_point]
            })

            df_intersection_with_segments = pd.concat([df_intersection_with_segments, temp_df])

        df_intersection_with_segments = df_intersection_with_segments.reset_index(drop=True)
        self.df_intersection_with_segments = df_intersection_with_segments

    def find_polygons(self):
        """Find polygons"""
        df_polygons = pd.DataFrame()
        df_polygons_roof = pd.DataFrame()
        for id in self.roof_ids:
            polygons_final = []
            plane_df = pd.DataFrame()
            curr_roof = self.df_planes.loc[self.df_planes['roof_id'] == id]
            points_on_plane_min_z = []
            for idx1, row1 in curr_roof.iterrows():
                # Finds the intersection lines of this segment
                curr_lines = self.df_lines.loc[(self.df_lines['roof_id'] == id) & (self.df_lines['segment_number1'] == row1[1])]
                for idx2, row2 in curr_lines.iterrows():
                    dir_vec = row2[3]
                    norm_dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    # Checks wheter the direction vector in z-direction is larger than almost 0 so that it 
                    # doesn't count horizontal lines
                    if abs(dir_vec[2]) > 0.01:
                        min_z = row1[4]
                        point_on_plane = row2[4]
                        scale_factor = (min_z[2] - point_on_plane[2]) / norm_dir_vec[2]
                        point = np.array(point_on_plane) + norm_dir_vec*scale_factor

                        temp_df = pd.DataFrame({
                            "segment_number1": row2[1],
                            "segment_number2": row2[2],
                            "point": [point]
                            }
                        )
                        plane_df = pd.concat([plane_df, temp_df])

                        points_on_plane_min_z.append([point, row2[1], row2[2]])

            roof_type = self.df_all_roofs.loc[self.df_all_roofs["roof_id"] == id]["roof_type"].iloc[0]
            if roof_type == "hipped" or roof_type == "corner_element":
                roof_polygons = []
                main_sub_gable = []
                for idx, row in curr_roof.iterrows():
                    points = plane_df.loc[(plane_df['segment_number1'] == row[1]) | (plane_df['segment_number2'] == row[1])]
                    intersection_points = []
                    segment_list = self.df_intersection_with_segments.loc[self.df_intersection_with_segments["roof_id"] == id]
                    for seg_list in segment_list.iloc[0][2]:
                        if row[1] in seg_list:
                            intersection_points.append(segment_list.iloc[0][1][segment_list.iloc[0][2].index(seg_list)])
                    
                    for idx3, row3 in points.iterrows():
                        intersection_points.append(row3[2])

                    roof_polygons.append(MultiPoint(sort_points_clockwise(intersection_points)).convex_hull)
                    main_sub_gable.append(False)

                polygons_final.append(roof_polygons)
            
            if roof_type == "gabled":
                roof_polygons = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]
                main_sub_gable = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]

                plane_param1, plane_param2, p1, p2, isp1, isp2, _, _, _, _, _, _ = find_main_gable_params(self, 0, 1, id, [0])

                non_outlier1_main = find_closest_points(isp1)
                non_outlier2_main = find_closest_points(isp2)

                roof_polygons[main_gable[0]] = MultiPoint(sort_points_clockwise([p1, p2, isp1[non_outlier2_main[0]], isp2[non_outlier1_main[0]]])).convex_hull
                roof_polygons[main_gable[1]] = MultiPoint(sort_points_clockwise([p1, p2, isp1[non_outlier2_main[1]], isp2[non_outlier1_main[1]]])).convex_hull
                polygons_final.append(roof_polygons)
                main_sub_gable[0] = True
                main_sub_gable[1] = True

            if roof_type == "t-element":
                df = self.df_lines.loc[(self.df_lines['roof_id'] == id)] 
                main_gable, sub_gable = get_main_and_sub_gables(df)
                roof_polygons = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]
                main_sub_gable = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]

                plane_param1, plane_param2, p1, p2, isp1, isp2, min_x, max_x, _, _, max_x_1, _ = find_main_gable_params(self, main_gable[0], main_gable[1], id, main_gable)
                
                non_outlier1_main = find_closest_points(isp1)
                non_outlier2_main = find_closest_points(isp2)

                roof_polygons[main_gable[0]] = MultiPoint(sort_points_clockwise([p1, p2, isp1[non_outlier2_main[0]], isp2[non_outlier1_main[0]]])).convex_hull
                roof_polygons[main_gable[1]] = MultiPoint(sort_points_clockwise([p1, p2, isp1[non_outlier2_main[1]], isp2[non_outlier1_main[1]]])).convex_hull
                main_sub_gable[main_gable[0]] = True
                main_sub_gable[main_gable[1]] = True
                plane1_param_sub, plane2_param_sub, _, _ , _, _ , min_x_sub, max_x_sub, _, _, _, min_x_sub_1 = find_main_gable_params(self, sub_gable[0], sub_gable[1], id, main_gable)

                closest_main_gable = main_gable[1] if check_distance(max_x, max_x_sub) > check_distance(max_x_1, max_x_sub) else main_gable[0]
                seg_to_use = [0, 1] if check_distance(min_x_sub, min_x) < check_distance(min_x_sub_1, min_x) else [1, 0]

                p1_sub, ips1_sub, dir_vec_sub = find_intersections_for_gabled(plane1_param_sub, plane2_param_sub, plane_param1 if closest_main_gable == main_gable[1] else plane_param1, self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub_gable[0])].iloc[0][4][2])

                edge_point = min_x_sub if check_distance(min_x_sub, p1_sub) > check_distance(max_x_sub, p1_sub) else max_x_sub
                edge_point_plane = [dir_vec_sub[0][0], dir_vec_sub[0][1], 0, -dir_vec_sub[0][0]*edge_point[0] - dir_vec_sub[0][1]*edge_point[1]]

                p2_sub, ips2_sub, _ = find_intersections_for_gabled(plane1_param_sub, plane2_param_sub, edge_point_plane, self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub_gable[0])].iloc[0][4][2])

                non_outlier1 = find_closest_points(ips1_sub)
                non_outlier2 = find_closest_points(ips2_sub)

                roof_polygons[sub_gable[seg_to_use[1]]] = MultiPoint(sort_points_clockwise([p1_sub, p2_sub, ips1_sub[non_outlier1[0]], ips2_sub[non_outlier2[0]]])).convex_hull
                roof_polygons[sub_gable[seg_to_use[0]]] = MultiPoint(sort_points_clockwise([p1_sub, p2_sub, ips1_sub[non_outlier1[1]], ips2_sub[non_outlier2[1]]])).convex_hull
                main_sub_gable[sub_gable[seg_to_use[1]]] = False
                main_sub_gable[sub_gable[seg_to_use[0]]] = False
                polygons_final.append(roof_polygons)

            if roof_type == "cross_element":
                roof_polygons = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]
                main_sub_gable = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]

                main_gable, sub_gable = get_main_and_sub_gables(self.df_lines.loc[(self.df_lines['roof_id'] == id)], True)
                plane_param1, plane_param2, p1, p2, isp1, isp2, min_x, max_x, _, _, max_x_1, _ = find_main_gable_params(self, main_gable[0], main_gable[1], id, main_gable)
                roof_polygons[main_gable[0]] = MultiPoint(sort_points_clockwise([p1, p2, isp1[0], isp2[0]])).convex_hull
                roof_polygons[main_gable[1]] = MultiPoint(sort_points_clockwise([p1, p2, isp1[1], isp2[1]])).convex_hull
                main_sub_gable[main_gable[0]] = True
                main_sub_gable[main_gable[1]] = True
                
                for sub in sub_gable:
                    plane_param1_sub, plane_param2_sub, _, _, _, _, min_x_sub, max_x_sub, _, _, _, minx_main1_sub = find_main_gable_params(self, sub[0], sub[1], id, sub)

                    closest_main_gable = main_gable[1] if check_distance(max_x, max_x_sub) > check_distance(max_x_1, max_x_sub) else main_gable[0]
                    seg_to_use = [0, 1] if check_distance(min_x_sub, min_x) < check_distance(minx_main1_sub, min_x) else [1, 0]

                    p1_sub, ips1_sub, dir_vec_sub = find_intersections_for_gabled(plane_param1_sub, plane_param2_sub, plane_param2 if closest_main_gable == main_gable[1] else plane_param1, self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub[0])].iloc[0][4][2] )
                    
                    ep = min_x_sub if check_distance(min_x_sub, p1_sub) > check_distance(max_x_sub, p1_sub) else max_x_sub
                    ep_plane = [dir_vec_sub[0][0], dir_vec_sub[0][1], 0, -dir_vec_sub[0][0]*ep[0] - dir_vec_sub[0][1]*ep[1]]

                    p2_sub, ips2_sub, _ = find_intersections_for_gabled(plane_param1_sub, plane_param2_sub, ep_plane, self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub[0])].iloc[0][4][2] )

                    non_outlier1 = find_closest_points(ips1_sub)
                    non_outlier2 = find_closest_points(ips2_sub)

                    roof_polygons[sub[seg_to_use[1]]] = MultiPoint(sort_points_clockwise([p1_sub, p2_sub, ips1_sub[non_outlier1[0]], ips2_sub[non_outlier2[0]]])).convex_hull
                    roof_polygons[sub[seg_to_use[0]]] = MultiPoint(sort_points_clockwise([p1_sub, p2_sub, ips1_sub[non_outlier1[1]], ips2_sub[non_outlier2[1]]])).convex_hull
                    main_sub_gable[sub[seg_to_use[1]]] = False
                    main_sub_gable[sub[seg_to_use[0]]] = False

                polygons_final.append(roof_polygons)

            if roof_type == "flat":
                main_sub_gable = [False]

                plane1_coords = self.df_all_roofs.loc[(self.df_all_roofs['roof_id'] == id) & (self.df_all_roofs['segment'] == 0)].iloc[0][1]

                plane1_param = self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == 0)].iloc[0][2]

                x_coordinates = [point[0] for point in plane1_coords.exterior.coords]
                y_coordinates = [point[1] for point in plane1_coords.exterior.coords]
                z_coordinates = [point[2] for point in plane1_coords.exterior.coords]

                dir_vec1 = self.df_planes.loc[self.df_planes["roof_id"] == id].iloc[0][2][:3]

                dir_vec2 = np.cross(dir_vec1, [0, 0, 1])

                p1, p3, _ , p2, p4, _ = find_min_max_values(x_coordinates, y_coordinates, z_coordinates)

                edge_plane_1 = [dir_vec1[0], dir_vec1[1], 0, -dir_vec1[0]*p3[0]-dir_vec1[1]*p3[1]]
                edge_plane_2 = [dir_vec1[0], dir_vec1[1], 0, -dir_vec1[0]*p4[0]-dir_vec1[1]*p4[1]]
                edge_plane_3 = [dir_vec2[0], dir_vec2[1], 0, -dir_vec2[0]*p1[0]-dir_vec2[1]*p1[1]]
                edge_plane_4 = [dir_vec2[0], dir_vec2[1], 0, -dir_vec2[0]*p2[0]-dir_vec2[1]*p2[1]]

                p1, _, _ = find_flat_roof_points(plane1_param, edge_plane_1, edge_plane_3)
                p2, _, _ = find_flat_roof_points(plane1_param, edge_plane_1, edge_plane_4)
                p3, _, _ = find_flat_roof_points(plane1_param, edge_plane_2, edge_plane_3)
                p4, _, _ = find_flat_roof_points(plane1_param, edge_plane_2, edge_plane_4)

                res = MultiPoint(sort_points_clockwise([p1, p2, p3, p4])).convex_hull
                polygons_final.append([res])

            for i, p in enumerate(polygons_final[0]):
                temp_df = pd.DataFrame({
                                "roof_id": id,
                                "geometry": [p],
                                "classification": i,
                                "main_gable": main_sub_gable[i]
                            }
                        )
                df_polygons = pd.concat([df_polygons, temp_df])

            df = pd.DataFrame({
                "roof_id": id, 
                "geometry": [MultiPolygon(polygons_final[0])]
            })
            df_polygons_roof = pd.concat([df_polygons_roof, df])
        
        df_polygons_roof = df_polygons_roof.reset_index(drop=True)
        df_polygons = df_polygons.reset_index(drop=True)
        self.df_polygons_roof = gpd.GeoDataFrame(df_polygons_roof)
        self.df_polygons = gpd.GeoDataFrame(df_polygons)

    def merge_buildings_for_roofs_with_multiple_buildings(self, buildings, roofs_with_buildings, roofs_with_several_buildings):
        roofs_several_buildings = roofs_with_buildings.query("roof_id in @roofs_with_several_buildings").sort_values(by=["roof_id", "classification", "index_right"])
        
        for roof in roofs_several_buildings.roof_id.unique():
            _roof = roofs_with_buildings.query("roof_id == @roof")
            building_indices = _roof.index_right.unique()
            _buildings = buildings.iloc[building_indices]
            merged = unary_union(_buildings.geometry.to_list())
            for b in _buildings.index:
                buildings.at[b, 'geometry'] = merged
        return buildings

    def find_relevant_buildings_for_modelling(self):
        building_data = gpd.read_file(self.buildings_path)

        building_data.crs=4326
        building_data = building_data.to_crs(25832)
        df_all_roofs = self.df_polygons

        joined_data = gpd.sjoin_nearest(df_all_roofs, building_data)

        roofs_with_multiple_buildings = joined_data[["roof_id", "index_right"]].groupby("roof_id").describe().index_right.query("std > 0").index
        merged_buildings = self.merge_buildings_for_roofs_with_multiple_buildings(building_data, joined_data, roofs_with_multiple_buildings)
        relevant_roof_ids = joined_data.index_right.unique()
        relevant_roofs = merged_buildings.iloc[relevant_roof_ids]

        matched_roofs_with_buildings = joined_data.merge(relevant_roofs, on="osm_id")
        matched_roofs_with_buildings = matched_roofs_with_buildings.drop(["geometry_x", "classification", "code_x", "fclass_x", "name_x", "type_x", "code_y", "fclass_y", "name_y", "type_y"], axis=1)
        matched_roofs_with_buildings = matched_roofs_with_buildings.drop_duplicates(subset='roof_id', keep='first')

        matched_roofs_with_buildings = gpd.GeoDataFrame(matched_roofs_with_buildings, geometry=list(matched_roofs_with_buildings["geometry_y"]))

        self.df_footprints = pd.DataFrame({
            "roof_id": self.roof_ids,
            "type": self.roof_types
        })

        for i in range(len(matched_roofs_with_buildings)):
            matched_roof_id = matched_roofs_with_buildings.iloc[i]["roof_id"]
            footprint_geometry = matched_roofs_with_buildings.iloc[i]["geometry"]
            self.df_footprints.loc[self.df_footprints["roof_id"] == matched_roof_id, "footprint"] = footprint_geometry

    def match_footprints(self):
        ids_points = {"corner_element": 6, "cross_element": 12, "flat": 4, "gabled": 4, "hipped": 4, "t-element": 8}
        df_adjusted_roof_planewise = pd.DataFrame()
        adjusted_roof = pd.DataFrame()
        for id in self.roof_ids:
            if id == "182341061":
                continue

            footprint = self.df_footprints.loc[self.df_footprints["roof_id"] == id].iloc[0]["footprint"]
            roof_type = self.df_footprints.loc[self.df_footprints["roof_id"] == id].iloc[0]["type"]
            curr_roof_plane = self.df_planes.loc[self.df_planes["roof_id"] == id]
            curr_roof_poly = self.df_polygons.loc[self.df_polygons["roof_id"] == id]
            curr_intersection_points = self.df_intersection_with_segments.loc[self.df_intersection_with_segments["roof_id"] == id].iloc[0]
            min_z, max_z = find_global_min_max_z(curr_roof_plane)

            footprint_points = update_polygon_points_with_footprint(curr_roof_plane, footprint, min_z, max_z)
            upper_points = find_upper_roof_points(curr_roof_poly)
            points_shifted_z_from_footprints = footprint_points
            new_polys = []
            if ids_points[roof_type] == len(points_shifted_z_from_footprints) and roof_type != "flat":
                for i in range(len(curr_roof_poly["geometry"])):
                    unique_elems = list(set(curr_roof_poly["geometry"].iloc[i].exterior.coords))
                    new_points = []
                    for k, point1 in enumerate(unique_elems):
                        temp = None
                        min_distance = float('inf')
                        for j, point2 in enumerate(points_shifted_z_from_footprints):
                            if point2[2] <= curr_roof_plane.iloc[i]["max_z_coordinate"][2] - 0.3:
                                distance = check_distance(point1, point2)

                                if distance < min_distance:
                                    min_distance = distance
                                    temp = [point1, list(point2)]
                        if temp != None and point1[2] < curr_roof_plane.iloc[i]["max_z_coordinate"][2] - 0.3:
                            new_point = [temp[1][0], temp[1][1], min_z]
                            new_points.append(new_point)

                    curr_upper = []
                    for f in range(len(curr_intersection_points["segments"])):
                        for l in range(len(upper_points)):
                                curr_upper.append([check_distance(upper_points[l], curr_intersection_points["points"][f]), [l, f]])

                    curr_upper.sort(key=lambda x: x[0])
                    upper_to_use = []
                    if roof_type == "hipped" or roof_type == "corner_element":
                        for j, segments in enumerate(curr_intersection_points["segments"]):
                            if i in segments:
                                for k, index in enumerate(curr_upper[:3]):
                                    if index[1][1] == j:
                                        upper_to_use.append(upper_points[index[1][0]])
                                        break
                    else:
                        for j in range(len(curr_roof_poly.iloc[i]["geometry"].exterior.coords)):
                            for k in range(len(upper_points)): 
                                if curr_roof_poly.iloc[i]["geometry"].exterior.coords[j] == upper_points[k]:
                                    upper_to_use.append(upper_points[k])
                    distances = []
                    checked = set() 
                    for k in range(len(new_points)):
                        for j in range(len(new_points)):
                            if (k,j) not in checked and (j,k) not in checked and k != j:
                                checked.add((k,j))
                                checked.add((j,k))
                                d = check_distance(new_points[k], new_points[j])
                                distances.append([d, new_points[k], new_points[j]])
                    distances.sort(key=lambda x: x[0])

                    liste = new_points + upper_to_use
                    poly = MultiPoint(sort_points_clockwise(liste)).convex_hull
                    new_polys.append(poly)

                    temp_df = pd.DataFrame({
                        "roof_id": id,
                        "classification": i,
                        "lower_roof_top_points": [new_points],
                        "upper_roof_top_points": [upper_to_use],
                        "footprint": [footprint],
                        "main_gable": curr_roof_poly.iloc[i]["main_gable"],
                        }
                    )
                    df_adjusted_roof_planewise = pd.concat([df_adjusted_roof_planewise, temp_df])
            else:
                if roof_type == "flat": 
                    polygon = Polygon(footprint_points)
                    temp_df = pd.DataFrame({
                        "roof_id": id,
                        "classification": 0,
                        "lower_roof_top_points": [footprint_points],
                        "upper_roof_top_points": [polygon],
                        "footprint": [footprint],
                        "main_gable": False
                        }
                    )
                    df_adjusted_roof_planewise = pd.concat([df_adjusted_roof_planewise, temp_df])
                else:
                    for idx, row in curr_roof_poly.iterrows():
                        seg = row["classification"]
                        curr_upper = []
                        for f in range(len(curr_intersection_points["segments"])):
                            for l in range(len(upper_points)):
                                    curr_upper.append([check_distance(upper_points[l], curr_intersection_points["points"][f]), [l, f]])

                        curr_upper.sort(key=lambda x: x[0])
                        upper_to_use = []
                        if roof_type == "hipped" or roof_type == "corner_element":
                            for j, segments in enumerate(curr_intersection_points["segments"]):
                                if seg in segments:
                                    for k, index in enumerate(curr_upper[:3]):
                                        if index[1][1] == j:
                                            upper_to_use.append(upper_points[index[1][0]])
                                            break
                        else:
                            for j in range(len(curr_roof_poly.iloc[seg]["geometry"].exterior.coords)):
                                for k in range(len(upper_points)): 
                                    if curr_roof_poly.iloc[seg]["geometry"].exterior.coords[j] == upper_points[k]:
                                        upper_to_use.append(upper_points[k])

                        footprint_points2 = []
                        curr_plane = curr_roof_plane.loc[curr_roof_plane["segment"] == seg].iloc[0]

                        current_coords = np.array(list(self.df_all_roofs.loc[self.df_all_roofs["roof_id"] == id].iloc[seg][1].exterior.coords))
                        liste = np.array([[c[0] for c in current_coords], [c[1] for c in current_coords], [c[2] for c in current_coords]])
                        min_x, max_x = min(liste[0]), max(liste[0])

                        for r, corner_point in enumerate(footprint.exterior.coords[:-1]):
                            x, y = corner_point
                            A, B, C, D = curr_plane["plane_param"]
                            intersection_z = (-A * x - B * y - D) / C

                            if min_z < intersection_z < max_z and min_x - 2 < x < max_x + 2:
                                # new_point = [x, y, intersection_z]
                                new_point = [x, y, min_z]
                                footprint_points2.append(new_point)

                        liste = footprint_points2 + upper_to_use
                        polygon = Polygon(sort_points_clockwise(liste))
                        new_polys.append(polygon)
                        temp_df = pd.DataFrame({
                            "roof_id": id,
                            "classification": seg,
                            "lower_roof_top_points": [footprint_points2],
                            "upper_roof_top_points": [upper_to_use],
                            "footprint": [footprint],
                            "main_gable": curr_roof_poly.loc[curr_roof_poly["classification"] == seg].iloc[0]["main_gable"], 
                            }
                        )
                        df_adjusted_roof_planewise = pd.concat([df_adjusted_roof_planewise, temp_df])

            df = pd.DataFrame({
                "roof_id": id,
                "roof_type": [roof_type],
                "geometry": [new_polys]
            })

            adjusted_roof = pd.concat([adjusted_roof, df])
        adjusted_roof = adjusted_roof.reset_index(drop=True)
        df_adjusted_roof_planewise = df_adjusted_roof_planewise.reset_index(drop=True)

        self.adjusted_roof = gpd.GeoDataFrame(adjusted_roof)
        self.df_adjusted_roof_planewise = gpd.GeoDataFrame(df_adjusted_roof_planewise)

    def create_building_walls(self):
        df_walls = pd.DataFrame()
        for id in self.roof_ids:
            if id == "182341061":
                continue
            curr_adjusted_roof = self.df_adjusted_roof_planewise.loc[self.df_adjusted_roof_planewise["roof_id"] == id]
            curr_roof_plane = self.df_planes.loc[self.df_planes["roof_id"] == id]
            roof_type = self.adjusted_roof.loc[self.adjusted_roof["roof_id"] == id]["roof_type"].iloc[0]
            no_gables_roofs = ["flat", "hipped", "corner_element"]

            if roof_type not in no_gables_roofs:
                if roof_type == "cross_element":
                    output = get_main_and_sub_gables(self.df_lines.loc[self.df_lines['roof_id'] == id], True)
                    gables = [output[0]]
                    sub_gable = output[1]
                    for sub in sub_gable:
                        gables.append(sub)
                elif roof_type == "gabled":
                    gables = [[0,1]]
                else:
                    main_gable, sub_gable = get_main_and_sub_gables(self.df_lines.loc[self.df_lines['roof_id'] == id])
                    gables = [main_gable] + [sub_gable]
            else:
                gables = []

            min_z, _ = find_global_min_max_z(curr_roof_plane)
            walls = []
            if roof_type != "flat":
                if len(gables) != 0:
                    for num, gable in enumerate(gables):
                        flattened_list1 = [sublist_2d for sublist_2d in curr_adjusted_roof["lower_roof_top_points"].iloc[gable[0]]]
                        flattened_list2 = [sublist_2d for sublist_2d in curr_adjusted_roof["lower_roof_top_points"].iloc[gable[1]]]
                        flattened_upper = [tuple(coord) for sublist in curr_adjusted_roof["upper_roof_top_points"] for coord in sublist]
                        unique_coordinates = list(set(flattened_upper))

                        dist = []
                        checked = set()
                        for i, point1 in enumerate(flattened_list1):
                            for j, point2 in enumerate(flattened_list2):
                                if (tuple(point1), tuple(point2)) not in checked and (tuple(point1), tuple(point2)) not in checked and tuple(point1) != tuple(point2):
                                    checked.add((tuple(point1),tuple(point2)))
                                    checked.add((tuple(point2),tuple(point1)))
                                    dist.append([i, j, check_distance(point1, point2)])
                        dist.sort(key=lambda x: x[2], reverse=True)
                        indexes = [dist[0][0], dist[0][1], dist[1][0], dist[1][1]]

                        poly1, poly2 = get_wall_polys(indexes, flattened_list1, flattened_list2, unique_coordinates, min_z)

                        if num > 0:
                            dist1 = walls[0].distance(poly1)
                            dist2 = walls[0].distance(poly2)
                            walls.append(poly1) if dist1 >= dist2 else walls.append(poly2)
                        else:
                            walls.append(poly1)
                            walls.append(poly2)

                df_exploded = curr_adjusted_roof.explode('lower_roof_top_points').reset_index(drop=True)
                all_values = list(df_exploded['lower_roof_top_points'])
                all_points = sort_points_clockwise(all_values)
                valid_walls = []
                for curr_index in range(len(all_points)):
                    point1 = all_points[curr_index]
                    if curr_index == len(all_points) - 1:
                        point2 = all_points[0]
                    else:
                        point2 = all_points[curr_index + 1]
                    new_point1 = [point1[0], point1[1], min_z - 8]
                    new_point2 = [point2[0], point2[1], min_z - 8]
                    point = [point1, new_point1, new_point2, point2]
                    poly = Polygon(sort_points_clockwise(point))
                    matching_coords = [does_poly_match(wall, poly) for wall in walls]
                    if not any(matching_coords):
                        valid_walls.append(poly)
                for wall in valid_walls: walls.append(wall)
            else:
                flattened_list = [sublist_2d for sublist_2d in curr_adjusted_roof.iloc[0]["lower_roof_top_points"]]
                points_in_wall = []
                for i in range(len(flattened_list) - 1):
                    point1 = flattened_list[i]
                    point2 = flattened_list[i + 1]

                    new_point1 = [point1[0], point1[1], min_z - 8]
                    new_point2 = [point2[0], point2[1], min_z - 8]
                    point = [point1, new_point1, new_point2, point2]

                    walls.append(Polygon(sort_points_clockwise(point)))

                    if len(flattened_list) - i == 2:
                        point1 = flattened_list[0]
                        point2 = flattened_list[len(flattened_list)-1]
                        new_point1 = [point1[0], point1[1], min_z - 8]
                        new_point2 = [point2[0], point2[1], min_z - 8]

                        point = [point1, new_point1, new_point2, point2]

                        walls.append(Polygon(sort_points_clockwise(point)))

            temp_df = pd.DataFrame({
                "roof_id": id,
                "walls": [walls]
            })
            df_walls = pd.concat([df_walls, temp_df])
        df_walls = df_walls.reset_index(drop=True)
        self.df_walls = gpd.GeoDataFrame(df_walls)

    def run_all(self, FOLDER):
        folder_names = os.listdir(FOLDER)
        for folder in folder_names:
            if folder != '.DS_Store' and folder != 'roof categories.jpg':
                self.roofs[folder] = {}
                roofs = os.listdir(f'{FOLDER}/{folder}')
                for roof in roofs:
                    FULL_PATH = f'{FOLDER}/{folder}/{roof}'
                    laz = laspy.read(FULL_PATH)
                    las = laspy.convert(laz)
                    # las = laspy.read(f'{FULL_PATH}')
                    self.roofs[folder][roof[:-4]] = las

        fkb_path = "fkb_trondheim"
        self.buildings_path = fkb_path
        if not os.path.exists(fkb_path):
            os.mkdir(fkb_path)
            fkb_to_csv()

        self.buildings_from_file = gpd.read_file(fkb_path)
        self.roof_ids, self.roof_types = get_keys(self.roofs)
        self.create_df_all_roofs()
        self.find_plane_params_to_segment()
        self.find_plane_intersections()
        self.find_intersection_points()
        self.find_segments_to_match_intersection_point()
        # self.find_polygons()
        # self.find_relevant_buildings_for_modelling()
        # self.match_footprints()
        # self.create_building_walls()

    def plot_result(self, type):      
        match type:
            case "scatterplot_3d":
                self.plot.scatterplot_3d()
            case "plane_3D":
                self.plot.plane_3D(self.roof_ids, self.df_planes, self.df_all_roofs)
            case "line_scatter":
                self.plot.line_scatter(self.df_lines, self.roof_ids)
            case "line_scatter_intersection_points":
                self.plot.line_scatter_intersection_points(self.df_lines, self.roof_ids, self.intersection_points)
            case "roof_polygons":
                self.plot.plot_roof_polygons(self.df_polygons, self.roof_ids, self.df_all_roofs)
            case "footprint_with_roof":
                self.plot.plot_footprint_and_roof(self.df_footprints, self.df_polygons, self.roof_ids)
            case "plot_entire_roof":
                self.plot.plot_entire_roof(self.df_walls, self.adjusted_roof, self.roof_ids, self.df_adjusted_roof_planewise, self)

if __name__ == '__main__':
    roofs = Roofs()
    FOLDER = "laz_roofs"
    roofs.run_all(FOLDER)
    # roofs.plot_result("scatterplot_3d")
    # roofs.plot_result("plane_3D")
    # roofs.plot_result("line_scatter")
    # roofs.plot_result("line_scatter_intersection_points")
    # roofs.plot_result("roof_polygons")
    roofs.plot_result("footprint_with_roof")
    roofs.plot_result("plot_entire_roof")
