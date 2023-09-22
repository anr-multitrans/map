import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes



## setup
local_dataroot = '/home/li/Documents/NuScenes/data/sets/nuscenes'

# init NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot=local_dataroot, verbose=False)

# init NuScenes Map
# (`singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`, `boston-seaport`)
nusc_map_one = NuScenesMap(dataroot=local_dataroot, map_name='singapore-onenorth')
bitmap_one = BitMap(nusc_map_one.dataroot, nusc_map_one.map_name, 'basemap')

# nusc_map_hol = NuScenesMap(dataroot=local_dataroot, map_name='singepore-hollandvillage')
# nusc_map_que = NuScenesMap(dataroot=local_dataroot, map_name='singapore-queenstown')
# nusc_map_bos = NuScenesMap(dataroot=local_dataroot, map_name='boston-seaport')


## visulaization
# rendering multiple layers
# fig, ax = nusc_map_one.render_layers(nusc_map_one.non_geometric_layers, figsize=1)
# plt.show()


# rendering bitmap - the lidar basemap
# fig, ax = nusc_map_one.render_layers(['lane'], figsize=1, bitmap=bitmap_one)
# plt.show()


# rendering a particular record of the layer
# fig, ax = nusc_map_one.render_record('stop_line', nusc_map_one.stop_line[14]['token'], other_layers=[], bitmap=bitmap_one)
# plt.show()


# rendering binary map mask layers
# patch_box = (300, 1700, 100, 100)
# patch_angle = 0  # Default orientation where North is up
# layer_names = ['drivable_area', 'walkway']
# canvas_size = (1000, 1000)

# map_mask = nusc_map_one.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
# print(map_mask[0])

# fig, ax = nusc_map_one.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=(12, 4), n_row=1)
# fig, ax = nusc_map_one.render_map_mask(patch_box, 45, layer_names, canvas_size, figsize=(12, 4), n_row=1)
# plt.show()


## rendering layers on top of the camera images
## pick a sample and render the front camera image
# sample_token = nusc.sample[9]['token']
# layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
# camera_channel = 'CAM_FRONT'
# nusc_map_one.render_map_in_image(nusc, sample_token, layer_names=layer_names, camera_channel=camera_channel)
# plt.show()


## rendering ego poses
# ego_poses = nusc_map_bos.render_egoposes_on_fancy_map(nusc, scene_tokens=[nusc.scene[1]['token']], verbose=False)
# plt.show()


## Navigation
# road layers: lane, road_block, road_segment
# x = 873
# y = 1286
# print('Road objects on selected point:', nusc_map_one.layers_on_point(x, y), '\n')
# print('Next road objects:', nusc_map_one.get_next_roads(x, y))    # there are 3 adjacent roads to the intersection specified by (x,y), so the lane has 6 tokens.

# nusc_map_one.render_next_roads(x, y, figsize=1, bitmap=bitmap_one)
# plt.show()

# working w/ lanes
# nusc_map_one.render_centerlines(resolution_meters=0.5, figsize=1, bitmap=bitmap_one)
# plt.show()

# x, y, yaw = 395, 1095, 0
# closest_lane = nusc_map_one.get_closest_lane(x, y, radius=2)
# print(closest_lane)

# lane_record = nusc_map_one.get_arcline_path(closest_lane)
# print(lane_record)

# print(nusc_map_one.get_incoming_lane_ids(closest_lane))
# print(nusc_map_one.get_outgoing_lane_ids(closest_lane))

# poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
# print(poses)

# closest_pose_on_lane, distance_along_lane = arcline_path_utils.project_pose_to_lane((x, y, yaw), lane_record)
# print(closest_pose_on_lane, '\n', distance_along_lane)

# print(arcline_path_utils.length_of_lane(lane_record))

# print(arcline_path_utils.get_curvature_at_distance_along_lane(distance_along_lane, lane_record)) # 0 means it is a straight lane


## data exploration
# my_patch = (300, 1000, 500, 1200) # The rectangular patch coordinates (x_min, y_min, x_max, y_max)
# fig, ax = nusc_map_one.render_map_patch(my_patch, nusc_map_one.non_geometric_layers, figsize=(10, 10), bitmap=bitmap_one)
# plt.show()

# records_within_patch = nusc_map_one.get_records_in_patch(my_patch, nusc_map_one.non_geometric_layers, mode='within')
# records_intersect_patch = nusc_map_one.get_records_in_patch(my_patch, nusc_map_one.non_geometric_layers, mode='intersect')

# layer = 'road_segment'
# print('Found %d records of %s (within).' % (len(records_within_patch[layer]), layer))
# print('Found %d records of %s (intersect).' % (len(records_intersect_patch[layer]), layer))

# my_point = (390, 1100)
# layers = nusc_map_one.layers_on_point(my_point[0], my_point[1])

# assert len(layers['stop_line']) > 0, 'Error: No stop line found!'
# rec_sl = nusc_map_one.record_on_point(my_point[0], my_point[1], 'stop_line')
# print(rec_sl)
# print(nusc_map_one.get_bounds('stop_line', rec_sl))


## layers
# print(nusc_map_one.layer_names)

# -Geometric layers-
# print(nusc_map_one.geometric_layers)

# 1.node {token, x, y}
# print(nusc_map_one.node[0])

# 2.line {token, node_token[ , , ]}
# print(nusc_map_one.line[2])

# 3.polygon, dict_keys(['token', 'exterior_node_tokens', 'holes'])
# sample_polygon = nusc_map_one.polygon[3]
# print(sample_polygon.keys())
# print(sample_polygon['exterior_node_tokens'][:10])
# print(sample_polygon['holes'][0])

# -non geometric layers-
# print(nusc_map_one.non_geometric_layers)

# 1.'drivable_area' {token, polygon_token}
# sample_drivable_area = nusc_map_one.drivable_area[0]
# print(sample_drivable_area)

# fig, ax = nusc_map_one.render_record('drivable_area', sample_drivable_area['token'], other_layers=[])
# plt.show()

# 2.'road_segment' {token, polygon_token, is_intersection, drivable_area_token, exterior_node_tokens, holes}
# print(nusc_map_one.road_segment[100])

# sample_intersection_road_segment = nusc_map_one.road_segment[3]
# print(sample_intersection_road_segment)

# fig, ax = nusc_map_one.render_record('road_segment', sample_intersection_road_segment['token'], other_layers=[])
# plt.show()

# 3.'road_block' {token, polygon_token, from_edge_line_token, to_edge_line_token, road_segment_token, exterior_node_tokens, holes}
# sample_road_block = nusc_map_one.road_block[0]
# print(sample_road_block)

# fig, ax = nusc_map_one.render_record('road_block', sample_road_block['token'], other_layers=[])
# plt.show()

# 4.'lane' {token, polygon_token, lane_type, from_edge_line_token, to_edge_line_token, left_lane_divider_segments, right_lane_divider_segments, exterior_node_tokens, holes, left_lane_divider_segment_nodes, right_lane_divider_segment_nodes}
# sample_lane_record = nusc_map_one.lane[600]
# print(sample_lane_record)

# fig, ax = nusc_map_one.render_record('lane', sample_lane_record['token'], other_layers=[])
# plt.show()

# 5.'ped_crossing' {token, polygon_token, road_segment_token, exterior_node_tokens, holes}
# sample_ped_crossing_record = nusc_map_one.ped_crossing[0]
# print(sample_ped_crossing_record)

# fig, ax = nusc_map_one.render_record('ped_crossing', sample_ped_crossing_record['token'])
# plt.show()

# 6.'walkway' {token, polygon_token, exterior_node_tokens, holes}
# sample_walkway_record = nusc_map_one.walkway[0]
# print(sample_walkway_record)

# fig, ax = nusc_map_one.render_record('walkway', sample_walkway_record['token'])
# plt.show()

# 7.'stop_line' {token, polygon_token, stop_line_type, ped_crossing_tokens, traffic_light_tokens, road_block_token, exterior_node_tokens, holes, cue, holes}
# sample_stop_line_record = nusc_map_one.stop_line[1]
# print(sample_stop_line_record)

# fig, ax = nusc_map_one.render_record('stop_line', sample_stop_line_record['token'])
# plt.show()

# 8.'carpark_area' {token, polygon_token, orientation, road_block_token, exterior_node_tokens, holes}
# sample_carpark_area_record = nusc_map_one.carpark_area[1]
# print(sample_carpark_area_record)

# fig, ax = nusc_map_one.render_record('carpark_area', sample_carpark_area_record['token'])
# plt.show()

# 9.'road_divider' {token, line_token, road_segment_token, node_tokens}
# sample_road_divider_record = nusc_map_one.road_divider[0]
# print(sample_road_divider_record)

# fig, ax = nusc_map_one.render_record('road_divider', sample_road_divider_record['token'])
# plt.show()

# 10.'lane_divider' {token, line_token, lane_divider_segments[{node_token, segment_type}, {}, ...], node_tokens}
# sample_lane_divider_record = nusc_map_one.lane_divider[0]
# print(sample_lane_divider_record)

# fig, ax = nusc_map_one.render_record('lane_divider', sample_lane_divider_record['token'])
# plt.show()

# 11.'traffic_light' {token, line_token, traffic_light_type, from_road_block_token, items[{color, shape, rel_pos{tx, ty, tz, rx, ry, rz}, to_road_block_tokens}, {}, ...], pose{tx, ty, tz, rx, ry, rz}, node_tokens}
# sample_traffic_light_record = nusc_map_one.traffic_light[0]
# print(sample_traffic_light_record)

# print(sample_traffic_light_record['items'])
