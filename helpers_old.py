import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from math import sqrt

def get_coords(roof_seg):
    # Extract all points to get a nice scatter plot

    x = roof_seg.x
    y = roof_seg.y
    z = roof_seg.z
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
    
    # Check if x coordinate is within bounds
    x_within_bounds = x_min <= x <= x_max
    
    # Check if y coordinate is within bounds
    y_within_bounds = y_min <= y <= y_max
    
    # Check if z coordinate is within bounds
    z_within_bounds = z_min <= z <= z_max

    return (x_within_bounds and y_within_bounds and z_within_bounds)

def find_highest_lowest_z_coord_globally(liste):
    max_z = float('-inf') 
    min_z = float('inf')  

    for plane in liste:
        temp_max, temp_min = find_highest_lowest_z_coord(plane)
        if temp_max > max_z:
            max_z = temp_max 
        if temp_min < min_z:
            min_z = temp_min 
    return min_z, max_z

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
        
    # Calculate the vector from P0 to P
    V = P - P0

    # Calculate the projection of V onto D
    Projection = np.dot(V, D) / np.dot(D, D)

    # Calculate the closest point on the line to P0
    ClosestPointOnLine = P0 + Projection * D

    # Calculate the distance between P and ClosestPointOnLine
    dist = np.linalg.norm(P - ClosestPointOnLine)

    return dist

def find_intersections_for_gabled(plane1, plane2, plane3, z_value):
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    A3, B3, C3, D3 = plane3

    A = np.array([[A1, B1, C1], [A2, B2, C2], [A3, B3, C3]])
    b = np.array([-D1, -D2, -D3])

    triple_intersection = np.dot(np.linalg.pinv(A), b)

    # Find intersection points where z=0
    A1, A2, A3 = A[:2,:2], A[1:,:2], np.array([A[0][:2], A[2][:2]])
    b1, b2, b3 = b[:2], b[1:], np.array([b[0], b[2]])

    ip1 = np.dot(np.linalg.pinv(A1), b1)
    ip2 = np.dot(np.linalg.pinv(A2), b2)
    ip3 = np.dot(np.linalg.pinv(A3), b3)

    dv1 = np.cross(A[0], A[1])
    dv2 = np.cross(A[1], A[2])
    dv3 = np.cross(A[0], A[2])

    double_intersections = [np.append(ip1, 0, axis=None), np.append(ip2, 0, axis=None), np.append(ip3, 0, axis=None)]
    
    direction_vectors = [dv1, dv2, dv3]

    double_intersections = [p + z_value/dv[2] * dv for p, dv in zip(double_intersections, direction_vectors) if abs(dv[2]) > 0.003]

    return triple_intersection, double_intersections, direction_vectors

def find_min_max_values(x, y, z):
    x_min_index = list(x).index(min(x))
    x_max_index = list(x).index(max(x))
    y_min_index = list(y).index(min(y))
    y_max_index = list(y).index(max(y))
    z_min_index = list(z).index(min(z))
    z_max_index = list(z).index(max(z))

    xmin_point = (x[x_min_index], y[x_min_index], z[x_min_index])
    xmax_point = (x[x_max_index], y[x_max_index], z[x_max_index])
    ymin_point = (x[y_min_index], y[y_min_index], z[y_min_index])
    ymax_point = (x[y_max_index], y[y_max_index], z[y_max_index])
    zmin_point = (x[z_min_index], y[z_min_index], z[z_min_index])
    zmax_point = (x[z_max_index], y[z_max_index], z[z_max_index])

    return xmin_point, xmax_point, ymin_point, ymax_point, zmin_point, zmax_point

def intersect_2planes(plane1, plane2):
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    
    A = np.array([[A1, B1, C1], [A2, B2, C2]])
    b = np.array([-D1, -D2])

    intersection = np.dot(np.linalg.inv(A[:,:2]), b)
    double_intersection = np.append(intersection, 0, axis=None)

    direction_vector = np.cross(A[0], A[1])

    return double_intersection, direction_vector

def get_main_and_sub_gables(df, cross=False):
    combined_segment_numbers = pd.concat([df['segment_number1'], df['segment_number2']])

    # Count the occurrences of each segment number
    segment_counts = combined_segment_numbers.value_counts()
    max_count = segment_counts.max()
    min_count = segment_counts.min()

    # Get the segment numbers with the most and least occurrences
    most_common_segments = segment_counts[segment_counts == max_count].index.tolist()
    least_common_segments = segment_counts[segment_counts == min_count].index.tolist()

    all_segments = combined_segment_numbers.unique()

    # Filter the segments that are not in the most or least common lists
    other_segments = [seg for seg in all_segments if seg not in most_common_segments and seg not in least_common_segments]

    # Combine all segments into one list
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

def sort_points_clockwise(list_of_points):
    intersection_points = np.array(list_of_points)
    center = np.mean(intersection_points, axis=0)
    angles = np.arctan2(intersection_points[:, 1] - center[1], intersection_points[:, 0] - center[0])

    # Create a list of (angle, point) tuples
    angle_point_tuples = [(angle, point) for angle, point in zip(angles, intersection_points)]

    # Sort the tuples based on angles
    sorted_tuples = sorted(angle_point_tuples, key=lambda x: x[0])

    # Extract the sorted points
    sorted_points = [point for angle, point in sorted_tuples]

    return sorted_points

def find_intersection(plane_normal, plane_point, line_direction, line_point):
    # Calculate the plane parameters
    A, B, C = plane_normal
    D = -(A * plane_point[0] + B * plane_point[1] + C * plane_point[2])

    # Calculate the line parameters
    U, V, W = line_direction
    x1, y1, z1 = line_point

    # Calculate the intersection parameter 't'
    t = (-D - A * x1 - B * y1 - C * z1) / (A * U + B * V + C * W)

    # Calculate the intersection point
    x_int = x1 + t * U
    y_int = y1 + t * V
    z_int = z1 + t * W

    return x_int, y_int, z_int

def find_flat_roof_points(plane1, plane2, plane3):
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    A3, B3, C3, D3 = plane3

    A = np.array([[A1, B1, C1], [A2, B2, C2], [A3, B3, C3]])
    b = np.array([-D1, -D2, -D3])

    triple_intersection = np.dot(np.linalg.pinv(A), b)

    A1, A2, A3 = A[:2,:2], A[1:,:2], np.array([A[0][:2], A[2][:2]])
    b1, b2, b3 = b[:2], b[1:], np.array([b[0], b[2]])
    ip1 = np.dot(np.linalg.pinv(A1), b1)
    ip2 = np.dot(np.linalg.pinv(A2), b2)
    ip3 = np.dot(np.linalg.pinv(A3), b3)

    dv1 = np.cross(A[0], A[1])
    dv2 = np.cross(A[1], A[2])
    dv3 = np.cross(A[0], A[2])

    double_intersections = [np.append(ip1, 0, axis=None), np.append(ip2, 0, axis=None), np.append(ip3, 0, axis=None)]
    direction_vectors = [dv1, dv2, dv3]

    return triple_intersection, double_intersections, direction_vectors

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

    minx_main0, maxx_main0, _, _, minz_main0, _ = find_min_max_values(x_coordinates, y_coordinates, z_coordinates)
    minxx_main1, maxx_main1, _, _, minz_main1, _ = find_min_max_values(x2_coordinates, y2_coordinates, z2_coordinates)
    if not sub: 
        _, main_intersection_dv = intersect_2planes(plane1_param, plane2_param)
        main_edge_plane1 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*minx_main0[0] - main_intersection_dv[1]*minx_main0[1] - 0*minx_main0[2]]
        main_edge_plane2 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*maxx_main0[0] - main_intersection_dv[1]*maxx_main0[1] - 0*maxx_main0[2]]

        main_tip1, main_ips1, _ = find_intersections_for_gabled(
            plane1_param,
            plane2_param,
            main_edge_plane1,
            roof.df_planes.loc[(roof.df_planes['roof_id'] == id) & (roof.df_planes['segment'] == gable[0])].iloc[0][4][2]
        )

        main_tip2, main_ips2, _ = find_intersections_for_gabled(
            plane1_param,
            plane2_param,
            main_edge_plane2,
            roof.df_planes.loc[(roof.df_planes['roof_id'] == id) & (roof.df_planes['segment'] == gable[0])].iloc[0][4][2]
        )

    return plane1_param, plane2_param, main_tip1, main_tip2, main_ips1, main_ips2, minx_main0, maxx_main0, minz_main0, minz_main1, maxx_main1, minxx_main1

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

def find_global_min_max_z(df):
    max_z = float('-inf') 
    min_z = float('inf')  
    max_z_index = None
    min_z_index = None

    min_z_vals = df["min_z_coordinate"].tolist()
    max_z_vals = df["max_z_coordinate"].tolist()

    for coord in min_z_vals:
        if coord[2] < min_z:
            min_z = coord[2]

    for coord in max_z_vals:
        if coord[2] > max_z:
            max_z = coord[2]
    
    return min_z, max_z

def update_polygon_points_with_footprint(curr_roof_plane, roof_type, curr_roof_poly, footprint, min_z, max_z):
    footprint_points = []

    for idx, curr_plane in curr_roof_plane.iterrows():
        for i, corner_point in enumerate(footprint.exterior.coords[:-1]):
            x, y = corner_point
            A, B, C, D = curr_plane["plane_param"]
            intersection_z = (-A * x - B * y - D) / C

            if min_z < intersection_z < max_z:
                new_point = [x, y, intersection_z]
                footprint_points.append(new_point)

    return footprint_points

def visualize_polygons(multi, id, roof, footprint=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna", 'red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna", 'red', 'blue', 'green', 'orange', 'brown', 'yellow', 'black',"lime", "magenta", "purple", "navy", "cyan", "thistle", "indigo", "steelblue", "wheat", "tan", "darkorange", "grey", "maroon", "sienna"]
    for i, poly in enumerate(multi):
        x, y, z = zip(*poly.exterior.coords)
        ax.plot(x, y, z, color=colors[2], alpha=0.7)

    for poly in roof:
        x,y,z = zip(*poly.exterior.coords)
        ax.plot(x, y, z, color=colors[i], alpha=0.7)

    if footprint is not None:
        for point in footprint:
            x, y, z = point
            ax.scatter(x, y, z, color=colors[2], alpha=0.7)


    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title(id)
    plt.show()

def sort_3d_points_around_center(list_of_points):
    intersection_points = np.array(list_of_points)

    center = np.mean(intersection_points, axis=0)

    def sorting_key(point):
        vector = point - center

        azimuthal_angle = np.arctan2(vector[1], vector[0])
        polar_angle = np.arctan2(vector[2], np.sqrt(vector[0]**2 + vector[1]**2))

        return (azimuthal_angle, polar_angle)

    sorted_points = sorted(intersection_points, key=sorting_key)

    return sorted_points