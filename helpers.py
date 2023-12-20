import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from math import sqrt
from shapely import Polygon, Point

def get_coords(roof_seg):
    x = roof_seg.x
    y = roof_seg.y
    z = roof_seg.z
    c = np.array([[color.red, color.green, color.blue] for color in roof_seg])
    return x, y, z, c

def fkb_to_csv():
    shape = gpd.read_file("fkb/norway-latest-free/gis_osm_buildings_a_free_1.shp")
    trondheim_polygon = gpd.read_file("fkb/Basisdata_5001_Trondheim_25832_Kommuner_GeoJSON.geojson").to_crs(4326).iloc[0].geometry
    trondheim_buildings = shape.within(trondheim_polygon)
    trondheim_buildings = shape.loc[trondheim_buildings].copy()
    trondheim_buildings.to_file("fkb_trondheim")

def get_plane_params_for_segment(liste):
    # https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points/12301583#12301583
    p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]

    def f_min(X,p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)

    def residuals(params, signals, X):
        return f_min(X, params)

    sol = leastsq(residuals, p0, args=(None, liste))[0]
    
    return sol

def get_keys(roofs):
    ids = []
    types = []
    for roof_type, roof_data in roofs.items():
        for name in roof_data.keys():
            ids.append(name)
            types.append(roof_type)

    return ids, types

def is_within_limit(x_min, x_max, y_min, y_max, z_min, z_max, point):
    x, y, z = point
    x_within_bounds = x_min <= x <= x_max
    y_within_bounds = y_min <= y <= y_max
    z_within_bounds = z_min <= z <= z_max
    return (x_within_bounds and y_within_bounds and z_within_bounds)

def find_highest_lowest_z_coord(liste):
    max_z = float('-inf')
    min_z = float('inf') 
    max_z_index = None
    min_z_index = None

    for index, coord in enumerate(liste):
        x, y, z = coord  
        if z > max_z:
            max_z = z  
            max_z_index = index  
        if z < min_z:
            min_z = z  
            min_z_index = index  

    return liste[max_z_index], liste[min_z_index]

def check_distance(point1, point2, twod=False):
    if twod:
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    else:
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2] - point2[2])**2)

def closest_dist_between_planes(coords1, coords2):
    checked = set()
    closest = 10000
    closest_index = 0
    for i, coord1 in enumerate(coords1):
        for j, coord2 in enumerate(coords2):
            if (i, j) not in checked and (j, i) not in checked and i != j:
                checked.add((i, j))
                dist = check_distance(coord1, coord2)
                if dist < closest:
                    closest = dist
                    closest_index = i
    return closest, closest_index

def closest_dist_lines(line1, line2, min_z, max_z):
    # https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    line1_start = line1[4]
    line1_direction = line1[3]
    line2_start = line2[4]
    line2_direction = line2[3]

    a0 = line1_start + min_z*line1_direction
    a1 = line1_start + max_z*line1_direction
    b0 = line2_start + min_z*line2_direction
    b1 = line2_start + max_z*line2_direction

    clampA0=False
    clampA1=False
    clampB0=False
    clampB1=False

    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2

    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)

        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B

    return pA,pB,np.linalg.norm(pA-pB)

def shortest_dist_point_line(D, P0, P):
    V = P - P0
    Projection = np.dot(V, D) / np.dot(D, D)
    ClosestPointOnLine = P0 + Projection * D
    dist = np.linalg.norm(P - ClosestPointOnLine)

    return dist

def sort_points_clockwise(list_of_points):
    intersection_points = np.array(list_of_points)
    center = np.mean(intersection_points, axis=0)
    angles = np.arctan2(intersection_points[:, 1] - center[1], intersection_points[:, 0] - center[0])

    angle_point_tuples = [(angle, point) for angle, point in zip(angles, intersection_points)]

    sorted_tuples = sorted(angle_point_tuples, key=lambda x: x[0])

    sorted_points = [point for _, point in sorted_tuples]

    return sorted_points

def find_closest_points(points):
    if len(points) == 2:
        return [0,1]
    point1, point2, point3 = points 

    distance12 = check_distance(point1, point2)
    distance13 = check_distance(point1, point3)
    distance23 = check_distance(point2, point3)

    if distance12 <= distance13 and distance12 <= distance23:
        index1 = 0
        index2 = 1
    elif distance13 <= distance12 and distance13 <= distance23:
        index1 = 0
        index2 = 2
    else:
        index1 = 1
        index2 = 2

    return [index1, index2]

def intersect_2planes(plane1, plane2):
    planes = [plane1, plane2]
    
    A = np.array([plane[:3] for plane in planes])
    b = np.array([-plane[3] for plane in planes])
    
    ip1 = np.dot(np.linalg.inv(A[:,:2]), b)
    ip2 = np.append(ip1, 0, axis=None)

    dir_vec = np.cross(A[0], A[1])

    return ip2, dir_vec

def find_main_gable_params(roof, seg_num1, seg_num2, id, gable, sub=False):
    
    plane1_coords = roof.df_all_roofs.loc[(roof.df_all_roofs['roof_id'] == id) & (roof.df_all_roofs['segment'] == seg_num1)].iloc[0][1]
    plane2_coords = roof.df_all_roofs.loc[(roof.df_all_roofs['roof_id'] == id) & (roof.df_all_roofs['segment'] == seg_num2)].iloc[0][1]

    plane1_param = roof.df_planes.loc[(roof.df_planes['roof_id'] == id) & (roof.df_planes['segment'] == seg_num1)].iloc[0][2]
    plane2_param = roof.df_planes.loc[(roof.df_planes['roof_id'] == id) & (roof.df_planes['segment'] == seg_num2)].iloc[0][2]

    x_coordinates = [point[0] for point in plane1_coords.exterior.coords]
    y_coordinates = [point[1] for point in plane1_coords.exterior.coords]
    z_coordinates = [point[2] for point in plane1_coords.exterior.coords]
    
    x2_coordinates = [point[0] for point in plane2_coords.exterior.coords]
    y2_coordinates = [point[1] for point in plane2_coords.exterior.coords]
    z2_coordinates = [point[2] for point in plane2_coords.exterior.coords]

    min_x, _, min_z, max_x, _, _ = find_min_max_values(x_coordinates, y_coordinates, z_coordinates)
    min_x_1, _, min_z_1, max_x_1, _, _ = find_min_max_values(x2_coordinates, y2_coordinates, z2_coordinates)

    if not sub: 
        _, isp_dir_vec = intersect_2planes(plane1_param, plane2_param)
        plane1 = [isp_dir_vec[0], isp_dir_vec[1], 0, -isp_dir_vec[0]*min_x[0] - isp_dir_vec[1]*min_x[1] - 0*min_x[2]]
        plane2 = [isp_dir_vec[0], isp_dir_vec[1], 0, -isp_dir_vec[0]*max_x[0] - isp_dir_vec[1]*max_x[1] - 0*max_x[2]]

        p1, ips1, _ = find_intersections_for_gabled(plane1_param, plane2_param, plane1, roof.df_planes.loc[(roof.df_planes['roof_id'] == id) & (roof.df_planes['segment'] == gable[0])].iloc[0][4][2] )

        p2, ips2, _ = find_intersections_for_gabled(plane1_param, plane2_param, plane2, roof.df_planes.loc[(roof.df_planes['roof_id'] == id) & (roof.df_planes['segment'] == gable[0])].iloc[0][4][2] )

    return plane1_param, plane2_param, p1, p2, ips1, ips2, min_x, max_x, min_z, min_z_1, max_x_1, min_x_1

def get_main_and_sub_gables(df, cross=False):
    combined_segment_numbers = pd.concat([df['segment_number1'], df['segment_number2']])
    segment_counts = combined_segment_numbers.value_counts()
    max_count = segment_counts.max()
    min_count = segment_counts.min()
    most_common_segments = segment_counts[segment_counts == max_count].index.tolist()
    least_common_segments = segment_counts[segment_counts == min_count].index.tolist()

    all_segments = combined_segment_numbers.unique()

    other_segments = [seg for seg in all_segments if seg not in most_common_segments and seg not in least_common_segments]

    result = [most_common_segments[0], least_common_segments[0]]
    
    if cross: 
        least_common_segments_res = []
        checked = set()
        for k1 in least_common_segments:
            if k1 not in checked:
                temp = df.loc[((df["segment_number1"] == k1) | (df["segment_number2"] == k1)) & ((df["segment_number1"] != most_common_segments[0]) & (df["segment_number1"] != most_common_segments[1]) & ((df["segment_number2"] != most_common_segments[0]) & (df["segment_number2"] != most_common_segments[1])))]
                checked.add(temp.iloc[0][1])
                checked.add(temp.iloc[0][2])
                least_common_segments_res.append([temp.iloc[0][1], temp.iloc[0][2]])
        result = [most_common_segments, least_common_segments_res]
        return result

    return result, other_segments

def find_intersections_for_gabled(plane1, plane2, plane3, z_value):
    planes = [plane1, plane2, plane3]

    A = np.array([plane[:3] for plane in planes])
    b = np.array([-plane[3] for plane in planes])

    isp_1 = np.dot(np.linalg.pinv(A), b)

    A_sub = [A[:2, :2], A[1:, :2], np.array([A[0][:2], A[2][:2]])]
    b_sub = [b[:2], b[1:], np.array([b[0], b[2]])]

    isps = [np.dot(np.linalg.pinv(sub_A), sub_b) for sub_A, sub_b in zip(A_sub, b_sub)]

    dir_vecs = [np.cross(A[0], A[1]), np.cross(A[1], A[2]), np.cross(A[0], A[2])]

    isp_2 = [np.append(ip, 0) for ip in isps]

    isp_2 = [p + z_value/dv[2] * dv for p, dv in zip(isp_2, dir_vecs) if abs(dv[2]) > 0.003]

    return isp_1, isp_2, dir_vecs

def find_min_max_values(x, y, z):
    min_indices = [x.index(min(x)), y.index(min(y)), z.index(min(z))]
    max_indices = [x.index(max(x)), y.index(max(y)), z.index(max(z))]

    min_points = [(x[min_indices[0]], y[min_indices[0]], z[min_indices[0]]), (x[min_indices[1]], y[min_indices[1]], z[min_indices[1]]), (x[min_indices[2]], y[min_indices[2]], z[min_indices[2]])]
    max_points = [(x[max_indices[0]], y[max_indices[0]], z[max_indices[0]]), (x[max_indices[1]], y[max_indices[1]], z[max_indices[1]]), (x[max_indices[2]], y[max_indices[2]], z[max_indices[2]])]

    return min_points + max_points

def find_flat_roof_points(plane1, plane2, plane3):
    planes = [plane1, plane2, plane3]

    A = np.array([plane[:3] for plane in planes])
    b = np.array([-plane[3] for plane in planes])

    isp_1 = np.dot(np.linalg.pinv(A), b)

    A_sub = [A[:2, :2], A[1:, :2], np.array([A[0][:2], A[2][:2]])]
    b_sub = [b[:2], b[1:], np.array([b[0], b[2]])]

    isp_2 = [np.dot(np.linalg.pinv(sub_A), sub_b) for sub_A, sub_b in zip(A_sub, b_sub)]

    dir_vecs = [np.cross(A[0], A[1]), np.cross(A[1], A[2]), np.cross(A[0], A[2])]

    isp_2 = [np.append(ip, 0) for ip in isp_2]

    return isp_1, isp_2, dir_vecs

def find_global_min_max_z(df):
    max_z = float('-inf') 
    min_z = float('inf')

    min_z_vals = df["min_z_coordinate"].tolist()
    max_z_vals = df["max_z_coordinate"].tolist()

    for coord in min_z_vals:
        if coord[2] < min_z:
            min_z = coord[2]

    for coord in max_z_vals:
        if coord[2] > max_z:
            max_z = coord[2]
    
    return min_z, max_z

def update_polygon_points_with_footprint(curr_roof_plane, footprint, min_z, max_z):
    footprint_points = []

    for _, curr_plane in curr_roof_plane.iterrows():
        for corner_point in footprint.exterior.coords[:-1]:
            x, y = corner_point
            A, B, C, D = curr_plane["plane_param"]
            intersection_z = (-A * x - B * y - D) / C

            if min_z < intersection_z < max_z:
                new_point = [x, y, intersection_z]
                footprint_points.append(new_point)

    return footprint_points

def find_upper_roof_points(roof):
    upper_points = []

    for i in range(len(roof)):
        polygon = roof["geometry"].iloc[i]
        exterior_coords = list(polygon.exterior.coords)

        lowest_z = min(p[2] for p in exterior_coords)
        for coord in exterior_coords:
            if abs(coord[2] - lowest_z) > 1:
                if coord not in upper_points:
                    upper_points.append(coord)

    return upper_points

def find_upper_point_to_use(poly, list):
    upper_dists = []
    for index, point in enumerate(list):
        d = poly.distance(Point(list[index]))
        upper_dists.append([index, d])
    upper_dists.sort(key=lambda x: x[1])
    return list[upper_dists[0][0]]


def get_wall_polys(indexes, list1, list2, unique_coordinates, min_z):
    index1, index2, index3, index4 = indexes

    points_in_wall = [list1[index1], [list1[index1][0], list1[index1][1], min_z -8], [list2[index4][0], list2[index4][1], min_z - 8], list2[index4]]
    poly1 = Polygon(points_in_wall)
    upper_point = find_upper_point_to_use(poly1, unique_coordinates)
    points_in_wall = [upper_point, list1[index1], [list1[index1][0], list1[index1][1], min_z -8], [list2[index4][0], list2[index4][1], min_z - 8], list2[index4]]
    poly1 = Polygon(points_in_wall)

    points_in_wall = [list1[index3], [list1[index3][0], list1[index3][1], min_z -8], [list2[index2][0], list2[index2][1], min_z - 8], list2[index2]]
    poly2 = Polygon(points_in_wall)
    upper_point = find_upper_point_to_use(poly2, unique_coordinates)
    points_in_wall = [upper_point, list1[index3], [list1[index3][0], list1[index3][1], min_z -8], [list2[index2][0], list2[index2][1], min_z - 8], list2[index2]]
    poly2 = Polygon(points_in_wall)

    return poly1, poly2

def does_poly_match(poly1, poly2):
    coords1 = list(poly1.exterior.coords)
    coords2 = list(poly2.exterior.coords)

    matching_coords_count = sum(coord in coords2 for coord in coords1)

    return matching_coords_count == 4