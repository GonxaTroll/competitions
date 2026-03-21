from decimal import Decimal, getcontext
import numpy as np
import pandas as pd
from shapely import affinity#, touches
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from tree_utils import ChristmasTree, scale_factor

def _get_rightest_coordinate_box_1tree():
    tree_base_horizontal = ChristmasTree(angle = 270)
    x_base_h, _ = tree_base_horizontal.polygon.exterior.xy
    lowest_x_index = np.argmin(x_base_h)
    offset_x = - x_base_h[lowest_x_index]

    translated_tree_base_horizontal = affinity.translate(
        tree_base_horizontal.polygon,
        xoff = offset_x,
        yoff = 0)

    x_base_th, _ = translated_tree_base_horizontal.exterior.xy
    return np.max(x_base_th)

def _get_top_coordinate_box_1tree():
    tree_base_vertical = ChristmasTree(angle = 0)
    _, y_base_v = tree_base_vertical.polygon.exterior.xy

    lowest_y_index = np.argmin(y_base_v)
    offset_y = - y_base_v[lowest_y_index]

    tree_base_vertical = affinity.translate(
        tree_base_vertical.polygon,
        xoff = 0,
        yoff = offset_y)

    _, y_base_tv = tree_base_vertical.exterior.xy
    return np.max(y_base_tv)

def get_top_right_corner_box(n_trees):
    return _get_rightest_coordinate_box_1tree() * n_trees, _get_top_coordinate_box_1tree() * n_trees


def translate_tree(tree, offset_x, offset_y):
    tree.center_x, tree.center_y = tree.center_x + Decimal(offset_x), tree.center_y + Decimal(offset_y)
    tree.polygon = affinity.translate(
        tree.polygon,
        xoff = float(offset_x),
        yoff = float(offset_y)
    )
    return tree

def adjust_to_top_right_corner(x_coord, y_coord, tree):
    # get current max x and y of the polygon
    x_coords, y_coords = tree.polygon.exterior.xy
    current_max_x = np.max(x_coords)
    current_max_y = np.max(y_coords)

    # calculate offsets
    offset_x = x_coord - current_max_x
    offset_y = y_coord - current_max_y

    # translate polygon
    tree = translate_tree(tree, offset_x, offset_y)
    return tree

def _intersects(placed_polygons, placed_polygons_tree, candidate_poly):
    possible_indices = placed_polygons_tree.query(candidate_poly)
    # This is the collision detection step
    if any((candidate_poly.intersects(placed_polygons[i]) and not
            candidate_poly.touches(placed_polygons[i]))
            for i in possible_indices):
        return True

def get_extreme_coordinates(polygon: Polygon) -> tuple:
    x_coords, y_coords = polygon.exterior.xy
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    return min_x, max_x, min_y, max_y

def fit(placed_trees: list, new_tree, xct, yct):
    max_cy = 5

    placed_polygons = [tree.polygon for tree in placed_trees]
    placed_polygons_tree = STRtree(placed_polygons)

    xt, yt = 0, 0
    for i in range(2):
        if i == 0:
            considered_x, considered_y = xct, yct
        else:
            considered_x, considered_y = new_tree.polygon.exterior.xy
            considered_x, considered_y = max(np.min(considered_x), 1), max(np.min(considered_y), 1) # CHECK THIS OUT!
        for proportion_step in reversed([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]):
            # xstep, ystep = proportion_step * xct, proportion_step * yct
            xstep, ystep = proportion_step * considered_x, proportion_step * considered_y
            cy = 0

            # loop for left_bottom movement
            while max_cy > cy:
                cy += 1
                while True:
                    xt += 1
                    # move new_polygon to the left by xstep
                    new_tree = translate_tree(new_tree, -xstep, 0)
                    leftest_x, _, _, _ = get_extreme_coordinates(new_tree.polygon)
                    if _intersects(placed_polygons, placed_polygons_tree, new_tree.polygon)\
                        or leftest_x < 0:
                        # put back to last valid position
                        new_tree = translate_tree(new_tree, xstep, 0)
                        break
                while True:
                    yt += 1
                    # move new_polygon down by ystep
                    new_tree = translate_tree(new_tree, 0, -ystep)
                    _, _, bottom_y, _ = get_extreme_coordinates(new_tree.polygon)
                    if _intersects(placed_polygons, placed_polygons_tree, new_tree.polygon) or bottom_y < 0:
                        # put back to last valid position
                        new_tree = translate_tree(new_tree, 0, ystep)
                        break
    return new_tree


def blf(new_trees: list, placed_trees = None) -> list:
    if placed_trees is None:
        i = 1
        i_stock = 1
        xct, yct = get_top_right_corner_box(n_trees = i_stock)
        trees = new_trees.copy()
        placed_trees, placed_polygons, placed_polygons_tree = [], [], STRtree([])
    else:
        placed_trees = [ChristmasTree(x.center_x / scale_factor, x.center_y / scale_factor, x.angle) for x in placed_trees]
        new_trees = [ChristmasTree(x.center_x / scale_factor, x.center_y / scale_factor, x.angle) for x in new_trees]
        trees = placed_trees + new_trees
        i = len(placed_trees)
        for t, tree in enumerate(placed_trees):
            if t == 0:
                _, rightest_x, _, top_y = get_extreme_coordinates(tree.polygon)
                continue
            _, t_rightest_x, _, t_top_y = get_extreme_coordinates(tree.polygon)
            rightest_x = max(rightest_x, t_rightest_x)
            top_y = max(top_y, t_top_y)
        for n_tree in range(1, i+1):
            xct, yct = get_top_right_corner_box(n_trees = n_tree)
            if xct >= rightest_x or yct >= top_y:
                i_stock = n_tree
                break
        placed_polygons = [x.polygon for x in placed_trees]
        placed_polygons_tree = STRtree(placed_polygons)

    while i < len(trees) + 1:
        adjusted_tree = adjust_to_top_right_corner(xct, yct, trees[i-1])
        adjusted_tree = fit(placed_trees, adjusted_tree, xct, yct)
        if _intersects(placed_polygons, placed_polygons_tree, adjusted_tree.polygon): # reinitialize problem with bigger stock
            i = 1
            i_stock += 1
            xct, yct = get_top_right_corner_box(n_trees = i_stock)
            placed_trees, placed_polygons, placed_polygons_tree = [], [], STRtree([])
        else:
            leftest_x_adjusted, _, lowest_y_adjusted, top_y_adjusted = get_extreme_coordinates(adjusted_tree.polygon)
            if top_y_adjusted > yct:
                yct = top_y_adjusted
                xoff = xct - leftest_x_adjusted if leftest_x_adjusted < xct else leftest_x_adjusted - xct
                adjusted_tree = translate_tree(adjusted_tree, xoff, -lowest_y_adjusted)
                adjusted_tree = fit(placed_polygons, adjusted_tree, xct, yct)
                _, rightest_x_adjusted, _, top_y_adjusted = get_extreme_coordinates(adjusted_tree.polygon)
                if rightest_x_adjusted > xct:
                    xct = rightest_x_adjusted
                    yct = top_y_adjusted
            placed_trees.append(adjusted_tree)
            placed_polygons.append(adjusted_tree.polygon)
            placed_polygons_tree = STRtree(placed_polygons)
            i += 1
            # import matplotlib.pyplot as plt
            # for i, tree in enumerate(placed_trees):
            #     x, y = tree.polygon.exterior.xy
            #     plt.plot(x, y, label=f'Tree {i+1}')
            # plt.legend()
            # plt.show()

    return placed_trees
