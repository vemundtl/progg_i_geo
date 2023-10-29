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

    # Punkt 1 fra las-fila er (roof.X[0], roof.Y[0], roof.Z[0]) og får fargen [roof.red[0], roof.green[0], roof.blue[0]]
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
                    # polygon = MultiPoint(points).convex_hull
                    row_data = {"roof_id": key, 'geometry': polygon, 'R': r, 'G': g, 'B': b}
                    polygons_data.append(row_data)

                polygon_df = pd.DataFrame(polygons_data)
                polygon_df['segment'] = polygon_df.groupby(['R', 'G', 'B']).ngroup()
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
                "min_z_coordinate": [min_z]
            })
            df = gpd.GeoDataFrame(df_pd)

            planes = pd.concat([planes, df])

        planes = planes.reset_index(drop=True)
        self.df_planes = planes

    def find_plane_intersections(self):
        """Finds inter sections between each plane, creating lines on the intersection point"""
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
                "points": [points]
                })
            intersection_points = pd.concat([intersection_points, df])

        intersection_points = intersection_points.reset_index(drop=True)
        for idx, row in intersection_points.iterrows():
            curr_points = row[1]
            checked = set()
            # Checks wheter there are multiple points within a certain range which 
            # are supposed to be the same intersection point
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
            
            # For cross element, which only is supposed to have one intersection point
            if idx == 8 or idx == 9:
                avg_point = np.mean(np.array(updated_curr_points), axis = 0)
                intersection_points.iloc[idx]['points'] = [avg_point]
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

                    intersection_points.at[idx, 'points'] = [lowest_z_point]
                # For all other roofs, the top z-value of the intersection point is set to the same
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

            # HIPPED, CORNER_ELEMENT
            if id == '182464406' or id == "182448567" or id == "182282537" or id == "300429640":
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
            
            # GABLED
            if id == "182338605" or id == "10472350":
                roof_polygons = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]
                main_sub_gable = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]

                plane_param1, plane_param2, main_tip1, main_tip2, main_ips1, main_ips2, minx_main0, maxx_main0, minz_main0, minz_main1, maxx_main1, _ = find_main_gable_params(self, 0, 1, id, [0])

                non_outlier1_main = find_closest_points(main_ips1)
                non_outlier2_main = find_closest_points(main_ips2)

                roof_polygons[main_gable[0]] = MultiPoint(sort_points_clockwise([main_tip1, main_tip2, main_ips1[non_outlier2_main[0]], main_ips2[non_outlier1_main[0]]])).convex_hull
                roof_polygons[main_gable[1]] = MultiPoint(sort_points_clockwise([main_tip1, main_tip2, main_ips1[non_outlier2_main[1]], main_ips2[non_outlier1_main[1]]])).convex_hull
                polygons_final.append(roof_polygons)
                main_sub_gable[0] = True
                main_sub_gable[1] = True

            # T-ELEMENT
            if id == "10498821" or id == "10477107":
                df = self.df_lines.loc[(self.df_lines['roof_id'] == id)] 
                main_gable, sub_gable = get_main_and_sub_gables(df)
                roof_polygons = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]
                main_sub_gable = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]

                plane_param1, plane_param2, main_tip1, main_tip2, main_ips1, main_ips2, minx_main0, maxx_main0, minz_main0, minz_main1, maxx_main1, _ = find_main_gable_params(self, main_gable[0], main_gable[1], id, main_gable)
                
                non_outlier1_main = find_closest_points(main_ips1)
                non_outlier2_main = find_closest_points(main_ips2)

                roof_polygons[main_gable[0]] = MultiPoint(sort_points_clockwise([main_tip1, main_tip2, main_ips1[non_outlier2_main[0]], main_ips2[non_outlier1_main[0]]])).convex_hull
                roof_polygons[main_gable[1]] = MultiPoint(sort_points_clockwise([main_tip1, main_tip2, main_ips1[non_outlier2_main[1]], main_ips2[non_outlier1_main[1]]])).convex_hull
                main_sub_gable[main_gable[0]] = True
                main_sub_gable[main_gable[1]] = True
                plane1_param_sub, plane2_param_sub, main_tip1_sub, main_tip2_sub, main_ips1_sub, main_ips2_sub, minx_main0_sub, maxx_main0_sub, minz_main0_sub, minz_main1_sub, maxx_main1_sub, minxx_main1_sub = find_main_gable_params(self, sub_gable[0], sub_gable[1], id, main_gable)

                closest_main_gable = main_gable[1] if check_distance(maxx_main0, maxx_main0_sub) > check_distance(maxx_main1, maxx_main0_sub) else main_gable[0]
                sub_seg_sides = [0, 1] if check_distance(minx_main0_sub, minx_main0) < check_distance(minxx_main1_sub, minx_main0) else [1, 0]

                sub_tip1, sub_ips1, sub_dvs1 = find_intersections_for_gabled(
                    plane1_param_sub,
                    plane2_param_sub,
                    plane_param1 if closest_main_gable == main_gable[1] else plane_param1,
                    self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub_gable[0])].iloc[0][4][2]
                )

                edge_point = minx_main0_sub if check_distance(minx_main0_sub, sub_tip1) > check_distance(maxx_main0_sub, sub_tip1) else maxx_main0_sub
                edge_point_plane = [sub_dvs1[0][0], sub_dvs1[0][1], 0, -sub_dvs1[0][0]*edge_point[0] - sub_dvs1[0][1]*edge_point[1]]

                sub_tip2, sub_ips2, sub_dvs2 = find_intersections_for_gabled(
                    plane1_param_sub,
                    plane2_param_sub,
                    edge_point_plane,
                    self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub_gable[0])].iloc[0][4][2]
                )

                non_outlier1 = find_closest_points(sub_ips1)
                non_outlier2 = find_closest_points(sub_ips2)

                roof_polygons[sub_gable[sub_seg_sides[1]]] = MultiPoint(sort_points_clockwise([sub_tip1, sub_tip2, sub_ips1[non_outlier1[0]], sub_ips2[non_outlier2[0]]])).convex_hull
                roof_polygons[sub_gable[sub_seg_sides[0]]] = MultiPoint(sort_points_clockwise([sub_tip1, sub_tip2, sub_ips1[non_outlier1[1]], sub_ips2[non_outlier2[1]]])).convex_hull
                main_sub_gable[sub_gable[sub_seg_sides[1]]] = False
                main_sub_gable[sub_gable[sub_seg_sides[0]]] = False
                polygons_final.append(roof_polygons)

            # CROSS
            if id == "182448729" or id == "21088358":
                roof_polygons = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]
                main_sub_gable = [None for _ in range(len(self.df_planes.loc[self.df_planes["roof_id"] == id]))]

                main_gable, sub_gable = get_main_and_sub_gables(self.df_lines.loc[(self.df_lines['roof_id'] == id)], True)
                plane_param1, plane_param2, main_tip1, main_tip2, main_ips1, main_ips2, minx_main0, maxx_main0, minz_main0, minz_main1, maxx_main1, _ = find_main_gable_params(self, main_gable[0], main_gable[1], id, main_gable)
                roof_polygons[main_gable[0]] = MultiPoint(sort_points_clockwise([main_tip1, main_tip2, main_ips1[0], main_ips2[0]])).convex_hull
                roof_polygons[main_gable[1]] = MultiPoint(sort_points_clockwise([main_tip1, main_tip2, main_ips1[1], main_ips2[1]])).convex_hull
                main_sub_gable[main_gable[0]] = True
                main_sub_gable[main_gable[1]] = True
                
                for sub in sub_gable:
                    plane_param1_sub, plane_param2_sub, _, _, _, _, minx_main0_sub, maxx_main0_sub, minz_main0_sub, minz_main1_sub, maxx_main1_sub, minx_main1_sub = find_main_gable_params(self, sub[0], sub[1], id, sub)

                    closest_main_gable = main_gable[1] if check_distance(maxx_main0, maxx_main0_sub) > check_distance(maxx_main1, maxx_main0_sub) else main_gable[0]
                    sub_seg_sides = [0, 1] if check_distance(minx_main0_sub, minx_main0) < check_distance(minx_main1_sub, minx_main0) else [1, 0]

                    sub_tip1, sub_ips1, sub_dvs1 = find_intersections_for_gabled(
                        plane_param1_sub,
                        plane_param2_sub,
                        plane_param2 if closest_main_gable == main_gable[1] else plane_param1,
                        self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub[0])].iloc[0][4][2]
                    )
                    
                    edge_point = minx_main0_sub if check_distance(minx_main0_sub, sub_tip1) > check_distance(maxx_main0_sub, sub_tip1) else maxx_main0_sub
                    edge_point_plane = [sub_dvs1[0][0], sub_dvs1[0][1], 0, -sub_dvs1[0][0]*edge_point[0] - sub_dvs1[0][1]*edge_point[1]]

                    sub_tip2, sub_ips2, sub_dvs2 = find_intersections_for_gabled(
                        plane_param1_sub,
                        plane_param2_sub,
                        edge_point_plane,
                        self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == sub[0])].iloc[0][4][2]
                    )

                    non_outlier1 = find_closest_points(sub_ips1)
                    non_outlier2 = find_closest_points(sub_ips2)

                    roof_polygons[sub[sub_seg_sides[1]]] = MultiPoint(sort_points_clockwise([sub_tip1, sub_tip2, sub_ips1[non_outlier1[0]], sub_ips2[non_outlier2[0]]])).convex_hull
                    roof_polygons[sub[sub_seg_sides[0]]] = MultiPoint(sort_points_clockwise([sub_tip1, sub_tip2, sub_ips1[non_outlier1[1]], sub_ips2[non_outlier2[1]]])).convex_hull
                    main_sub_gable[sub[sub_seg_sides[1]]] = False
                    main_sub_gable[sub[sub_seg_sides[0]]] = False

                polygons_final.append(roof_polygons)

            # FLAT 
            if id == "182341061" or id == '300557684':
                main_sub_gable = [False]

                plane1_coords = self.df_all_roofs.loc[(self.df_all_roofs['roof_id'] == id) & (self.df_all_roofs['segment'] == 0)].iloc[0][1]

                plane1_param = self.df_planes.loc[(self.df_planes['roof_id'] == id) & (self.df_planes['segment'] == 0)].iloc[0][2]

                x_coordinates = [point[0] for point in plane1_coords.exterior.coords]
                y_coordinates = [point[1] for point in plane1_coords.exterior.coords]
                z_coordinates = [point[2] for point in plane1_coords.exterior.coords]

                dv1 = dv1 = self.df_planes.loc[self.df_planes["roof_id"] == id].iloc[0][2][:3]

                dv2 = np.cross(dv1, [0, 0, 1])

                p1, p2, p3, p4, _, _ = find_min_max_values(x_coordinates, y_coordinates, z_coordinates)

                edge_plane_1 = [dv1[0], dv1[1], 0, -dv1[0]*p3[0]-dv1[1]*p3[1]]
                edge_plane_2 = [dv1[0], dv1[1], 0, -dv1[0]*p4[0]-dv1[1]*p4[1]]
                edge_plane_3 = [dv2[0], dv2[1], 0, -dv2[0]*p1[0]-dv2[1]*p1[1]]
                edge_plane_4 = [dv2[0], dv2[1], 0, -dv2[0]*p2[0]-dv2[1]*p2[1]]

                tip1, _, _ = find_flat_roof_points(plane1_param, edge_plane_1, edge_plane_3)
                tip2, _, _ = find_flat_roof_points(plane1_param, edge_plane_1, edge_plane_4)
                tip3, _, _ = find_flat_roof_points(plane1_param, edge_plane_2, edge_plane_3)
                tip4, _, _ = find_flat_roof_points(plane1_param, edge_plane_2, edge_plane_4)

                res = MultiPoint(sort_points_clockwise([tip1, tip2, tip3, tip4])).convex_hull
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
        """
        Merge buildings when a roof has multiple associated buildings.
        
        Args:
            buildings: GeoDataFrame of buildings
            roofs_with_buildings: GeoDataFrame of roofs with associated buildings
            roofs_with_several_buildings: List of roof IDs with multiple associated buildings
        
        Returns:
            buildings: GeoDataFrame with merged buildings
        """
        # Select roofs with multiple associated buildings and sort by roof_id
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
        # FLAT - 182341061 fucka punktsky 
        # FLAT - 300557684 må endre strukturen til polygonet siden det ikke har alle punktene 
        # T-ELEMENT - 10477107 har et dobbeltpunkt
        # GABLED - 10472350 har ikke riktig struktur på sidene liksom, må legge til punkter her også 
        ids_points = {"corner_element": 6, "cross_element": 12, "flat": 4, "gabled": 4, "hipped": 4, "t-element": 8}
        df_adjusted_roof_planewise = pd.DataFrame()
        adjusted_roof = pd.DataFrame()
        for id in self.roof_ids:
            if id == "182341061":
                continue
            # if id != "182448729":
            #     continue
            # if id != "182448729" and id != "21088358":
            #     continue
            print("Id", id)

            footprint = self.df_footprints.loc[self.df_footprints["roof_id"] == id].iloc[0]["footprint"]
            roof_type = self.df_footprints.loc[self.df_footprints["roof_id"] == id].iloc[0]["type"]
            curr_roof_plane = self.df_planes.loc[self.df_planes["roof_id"] == id]
            curr_roof_poly = self.df_polygons.loc[self.df_polygons["roof_id"] == id]
            curr_intersection_points = self.df_intersection_with_segments.loc[self.df_intersection_with_segments["roof_id"] == id].iloc[0]
            min_z, max_z = find_global_min_max_z(curr_roof_plane)

            footprint_points = update_polygon_points_with_footprint(curr_roof_plane, roof_type, curr_roof_poly, footprint, min_z, max_z)
            upper_points = find_upper_roof_points(curr_roof_poly)
            points_shifted_z_from_footprints = footprint_points
            shifted_points = []
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
                    print(curr_upper)
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

                    # upper_to_use.append(new_points[0])
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
                    # for p in curr_roof_poly.iloc[i]["geometry"].exterior.coords: print(p)

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
                        print("SEGMENT:", seg)
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
                        min_y, max_y = min(liste[1]), max(liste[1])

                        for r, corner_point in enumerate(footprint.exterior.coords[:-1]):
                            x, y = corner_point
                            A, B, C, D = curr_plane["plane_param"]
                            intersection_z = (-A * x - B * y - D) / C

                            if min_z < intersection_z < max_z and min_x - 2 < x < max_x + 2:
                                new_point = [x, y, intersection_z]
                                footprint_points2.append(new_point)
                                
                        dist = []
                        checked = set() 
                        for k in range(len(footprint_points2)):
                            for j in range(len(footprint_points2)):
                                if (k,j) not in checked and (j,k) not in checked and k != j:
                                    checked.add((k,j))
                                    checked.add((j,k))
                                    d = check_distance(footprint_points2[k], footprint_points2[j])
                                    if d < 0.2:
                                        dist.append(k)

                        if len(dist) > 0:
                            dist.sort(reverse=True)
                            for index in dist:
                                footprint_points2.pop(index)

                        # print(min_z, max_z)
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

                        # visualize_things2([polygon], footprint_points2, curr_roof_poly["geometry"], upper_to_use, id)

                # Kanskje regne avstanden mellom hvert punkt for hver side og finne par av punkter som 
                # burde "høre" sammen, kanskje det funker for flate hvert fall

                        # visualize_things2(new_polys, footprint_points2, curr_roof_poly["geometry"], upper_to_use, id)

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

        # for i in range(len(self.df_adjusted_roof_planewise)): print(self.df_adjusted_roof_planewise["roof_id"].iloc[i], self.df_adjusted_roof_planewise["upper_roof_top_points"].iloc[i])

    def create_building_walls(self):
        df_walls = pd.DataFrame()

        for id in self.roof_ids:
            if id == "182341061":
                continue
            print("Roof_id:", id)

            # TROR JEG SKAL KLARE Å SKILLE UT SUBGABLES WALL DER DET OVERLAPPER MED MAIN GABLE WALL VED Å IKKE TA MED HVIS 
            # VEGGEN VI FINNER ER INNEHOLDT I MAIN GABLE WALL ELLER NOE LIGNENDE 
            
            # MANGLER BARE 350 - gabled  --> DONE
            # og sjekke hva som har skjedd 729 - cross

            curr_adjusted_roof = self.df_adjusted_roof_planewise.loc[self.df_adjusted_roof_planewise["roof_id"] == id]
            # curr_footprint = self.df_footprints.loc[self.df_footprints["roof_id"] == id]["footprint"].iloc[0]
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

            min_z, max_z = find_global_min_max_z(curr_roof_plane)
            # curr_footprint_with_z = [[p[0], p[1], min_z - 8] for p in curr_footprint.exterior.coords]

            walls = []
            # for i, segment in curr_adjusted_roof.iterrows(): 
            if len(curr_adjusted_roof) > 0 and roof_type != "flat":
                for i in range(len(curr_adjusted_roof)): 
                    flattened_list = [sublist_2d for sublist_2d in curr_adjusted_roof["lower_roof_top_points"].iloc[i]]
                    points_in_wall = []
                    if i == 0: 
                        item = flattened_list[1]
                        flattened_list.pop(1)
                        flattened_list.append(item)
                    for curr_index in range(len(flattened_list) - 1):
                        point1 = flattened_list[curr_index]
                        point2 = flattened_list[curr_index + 1]

                        new_point1 = [point1[0], point1[1], min_z - 8]
                        new_point2 = [point2[0], point2[1], min_z - 8]
                        point = [point1, new_point1, new_point2, point2]
                        walls.append(Polygon(sort_points_clockwise(point)))


                    # checked = set()
                    # dist = []
                    # for idx1 in range(len(flattened_list)):
                    #     for idx2 in range(len(flattened_list)):
                    #         if (idx1, idx2) not in checked and (idx2, idx1) not in checked and idx1 != idx2:
                    #             checked.add((idx1, idx2))
                    #             checked.add((idx2, idx1))
                    #             dist.append([check_distance(flattened_list[idx1], flattened_list[idx2]), idx1, idx2])

                    # dist.sort(key=lambda x: x[0])
                    # if (dist[-1][1], dist[-1][2]) != (0, len(flattened_list)-1):
                    #     endpoint1 = flattened_list[dist[-1][1]]
                    #     endpoint2 = flattened_list[dist[-1][2]]

                    #     # Define a custom sorting key based on the distance along the main line
                    #     def distance_along_line(point):
                    #         vector = np.array(endpoint2) - np.array(endpoint1)
                    #         to_point = np.array(point) - np.array(endpoint1)
                    #         distance = np.dot(to_point, vector) / np.dot(vector, vector)
                    #         return distance

                    #     flattened_list.sort(key=distance_along_line)

                    # for curr_index in range(len(flattened_list) - 1):
                    #     point1 = flattened_list[curr_index]
                    #     point2 = flattened_list[curr_index + 1]

                    #     new_point1 = [point1[0], point1[1], min_z - 8]
                    #     new_point2 = [point2[0], point2[1], min_z - 8]
                    #     point = [point1, new_point1, new_point2, point2]
                    #     walls.append(Polygon(sort_points_clockwise(point)))

                if len(gables) != 0: 
                    for gable in gables:
                        flattened_list1 = [sublist_2d for sublist_2d in curr_adjusted_roof["lower_roof_top_points"].iloc[gable[0]]]
                        flattened_list2 = [sublist_2d for sublist_2d in curr_adjusted_roof["lower_roof_top_points"].iloc[gable[1]]]
                        dist = []
                        checked = set()
                        for i, point1 in enumerate(flattened_list1):
                            for j, point2 in enumerate(flattened_list2):
                                if (tuple(point1), tuple(point2)) not in checked and (tuple(point1), tuple(point2)) not in checked and tuple(point1) != tuple(point2):
                                    checked.add((tuple(point1),tuple(point2)))
                                    checked.add((tuple(point2),tuple(point1)))
                                    dist.append([i, j, check_distance(point1, point2)])
                        
                        dist.sort(key=lambda x: x[2], reverse=True)
                        index1, index2, index3, index4 = dist[0][0], dist[0][1], dist[1][0], dist[1][1]
                        points_in_wall = [flattened_list1[index1], [flattened_list1[index1][0], flattened_list1[index1][1], min_z -8], [flattened_list2[index4][0], flattened_list2[index4][1], min_z - 8], flattened_list2[index4]]
                        walls.append(Polygon(sort_points_clockwise(points_in_wall)))
                        points_in_wall = [flattened_list1[index3], [flattened_list1[index3][0], flattened_list1[index3][1], min_z -8], [flattened_list2[index2][0], flattened_list2[index2][1], min_z - 8], flattened_list2[index2]]
                        walls.append(Polygon(sort_points_clockwise(points_in_wall)))

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
            if len(curr_adjusted_roof) > 0: visualize_polygons3(walls, id, self.adjusted_roof.loc[self.adjusted_roof["roof_id"] == id]["geometry"].iloc[0])

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
        self.find_polygons()
        self.find_relevant_buildings_for_modelling()
        # VÆR OBS PÅ AT SEGMENTNUMBER (CLASSIFICATION) I DF_POLYGONS KANSKJE IKKE ER RIKTIG HVIS DET SKJER MYE RART SENERE
        self.match_footprints()
        self.create_building_walls()

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
                self.plot.scatter_with_plane_3D_all(self.df_planes, self.roof_ids)
            case "scatter_with_plane_3D_segments":
                self.plot.scatter_with_plane_3D_segments(self.df_all_roofs, self.df_planes)
            case "plane_with_intersections":
                self.plot.plane_with_intersections(self.roof_ids, self.df_planes, self.df_lines)
            case "intersections":
                self.plot.plot_intersections(self.df_lines, self.roof_ids)
            case "line_scatter":
                self.plot.line_scatter(self.df_lines, self.roof_ids)
            case "line_scatter_intersection_points":
                self.plot.line_scatter_intersection_points(self.df_lines, self.roof_ids, self.intersection_points)
            case "footprint_with_roof":
                self.plot.plot_footprint_and_roof(self.df_footprints, self.df_polygons, self.roof_ids)
            case "plot_entire_roof":
                self.plot.plot_entire_roof(self.df_walls, self.adjusted_roof, self.roof_ids, self.df_adjusted_roof_planewise, self)


if __name__ == '__main__':
    roofs = Roofs()
    FOLDER = "laz_roofs"
    roofs.run_all(FOLDER)
    # roofs.plot_result("scatterplot_3d")
    # roofs.plot_result("segmented_roof_2d")
    # roofs.plot_result("plane_3D")
    # roofs.plot_result("scatterplot_with_plane_3D_all")
    # roofs.plot_result("scatter_with_plane_3D_segments")
    # roofs.plot_result("plane_with_intersections")
    
    # INTERSECTION LINES
    # roofs.plot_result("intersections")

    # SCATTER ALL, INTERSECTION LINES
    # roofs.plot_result("line_scatter")
    # SCATTER ALL, INTERSECTION LINES AND INTERSECTION POINTS
    # roofs.plot_result("line_scatter_intersection_points")

    # roofs.plot_result("footprint_with_roof")

    roofs.plot_result("plot_entire_roof")
