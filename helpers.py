import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import leastsq
from math import sqrt

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
    for roof_type, roof_data in roofs.items():
        for name in roof_data.keys():
            ids.append(name[:-4])

    return ids 

def is_within_limit(x_min, x_max, y_min, y_max, z_min, z_max, point):
    x, y, z = point
    
    # Check if x coordinate is within bounds
    x_within_bounds = x_min <= x <= x_max
    
    # Check if y coordinate is within bounds
    y_within_bounds = y_min <= y <= y_max
    
    # Check if z coordinate is within bounds
    z_within_bounds = z_min <= z <= z_max

    return (x_within_bounds and y_within_bounds and z_within_bounds)

def find_highest_lowest_z_coord(liste):
    max_z = float('-inf')  # Start with negative infinity to ensure any z-coordinate is higher
    min_z = float('inf')  # Start with negative infinity to ensure any z-coordinate is higher
    max_z_index = None
    min_z_index = None

    # Iterate through the list of coordinates
    for index, coord in enumerate(liste):
        x, y, z = coord  # Unpack the coordinates
        if z > max_z:
            max_z = z  # Update the maximum z-coordinate
            max_z_index = index  # Update the index of the maximum z-coordinate
        if z < min_z:
            min_z = z  # Update the maximum z-coordinate
            min_z_index = index  # Update the index of the maximum z-coordinate

    return liste[max_z_index], liste[min_z_index]

def check_distance(point1, point2):
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


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
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
