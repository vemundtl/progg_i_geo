import pandas as pd
import geopandas as gpd
import numpy as np
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
    trondheim_buildings.to_file("output")

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

def find_highest_z_coord(liste):
    max_z = float('-inf')  # Start with negative infinity to ensure any z-coordinate is higher
    max_z_index = None

    # Iterate through the list of coordinates
    for index, coord in enumerate(liste):
        x, y, z = coord  # Unpack the coordinates
        if z > max_z:
            max_z = z  # Update the maximum z-coordinate
            max_z_index = index  # Update the index of the maximum z-coordinate

    return liste[max_z_index]

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
