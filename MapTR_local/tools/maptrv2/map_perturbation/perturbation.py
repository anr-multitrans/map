import copy
import random
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString, Polygon, Point

from typing import Dict, List, Tuple


class MapTransform:
    def __init__(self, map_explorer: NuScenesMapExplorer, layer_names=['ped_crossing', 'road_segment', 'road_block']):
        self.map_explorer = map_explorer
        self.layer_names = layer_names   # 'road_divider' are lines
        self.patch_angle = 0

    def patch_coords_2_box(self, coords):

        box = ((coords[0] + coords[2])/2, (coords[1] + coords[3]
                                           )/2, coords[3] - coords[1], coords[2] - coords[0])

        return box

    def patch_box_2_coords(self, box):

        coors = (box[0] - box[3]/2, box[1] - box[2]/2,
                 box[0] + box[3]/2, box[1] + box[2]/2)

        return coors

    def transfor_layer(self, patch_box, patch_angle, layer, token, rotate=0, scale=[1, 1], skew=[0, 0], shift=[0, 0]):

        # Convert patch_box to shapely Polygon coordinates
        # patch_box = self.patch_coords_2_box(patch_coords)
        patch_coords = self.patch_box_2_coords(patch_box)

        patch = self.map_explorer.get_patch_coord(patch_box, self.patch_angle)

        # Get neede non geometric layers records intersects a particular rectangular patch
        records = self.map_explorer.map_api.get_records_in_patch(patch_coords, [
                                                                 layer])

        polygon_list = []
        for lay_token in records[layer]:
            lay_record = self.map_explorer.map_api.get(layer, lay_token)
            polygon = self.map_explorer.map_api.extract_polygon(
                lay_record['polygon_token'])
            # plt.plot(*polygon.exterior.xy)
            if polygon.is_valid:
                if lay_token == token:
                    if rotate:
                        polygon = affinity.rotate(polygon, rotate)
                    if scale != [1, 1]:
                        polygon = affinity.scale(polygon, scale[0], scale[1])
                    if skew != [0, 0]:
                        polygon = affinity.skew(polygon, skew[0], skew[1])
                    if shift != [0, 0]:
                        polygon = affinity.translate(
                            polygon, shift[0], shift[1])

                    # plt.plot(*polygon.exterior.xy)

                    # Each pedestrian crossing record has to be on a road segment.
                    if layer == 'ped_crossing':
                        roa_seg_record = self.map_explorer.map_api.get(
                            'road_segment', lay_record['road_segment_token'])
                        road_seg_polygon = self.map_explorer.map_api.extract_polygon(
                            roa_seg_record['polygon_token'])
                        polygon = polygon.intersection(road_seg_polygon)

                    polygon = polygon.intersection(patch)

                if polygon.is_valid:
                    if polygon.geom_type == 'Polygon':
                        polygon = MultiPolygon([polygon])
                    polygon_list.append(polygon)
                else:
                    print(
                        'The transformed layer leaves the patch range and is considered deleted.')

        layer_names = copy.copy(self.layer_names)
        layer_names.remove(layer)
        patch_geo_list = self.map_explorer.map_api.get_map_geom(
            patch_box, self.patch_angle, layer_names)

        polygon_list = patch_geo_list + [(layer, polygon_list)]

        return polygon_list

    def creat_ped_polygon(self, road_segment_token=None):

        min_x, min_y, max_x, max_y = self.map_explorer.map_api.get_bounds(
            'road_segment', road_segment_token)

        x_range = max_x - min_x
        y_range = max_y - min_y

        if max([x_range, y_range]) <= 4:
            new_polygon = self.map_explorer.map_api.extract_polygon(
                self.map_explorer.map_api.get('road_segment', road_segment_token)['polygon_token'])
        else:
            if x_range > y_range:
                rand = random.uniform(min_x, max_x - 4)
                left_bottom = Point([rand, min_y])
                left_top = Point([rand, max_y])
                right_bottom = Point([rand + 4, min_y])
                right_top = Point([rand + 4, max_y])
            else:
                rand = random.uniform(min_y, max_y - 4)
                left_bottom = Point([min_x, rand])
                left_top = Point([min_x, rand + 4])
                right_bottom = Point([max_x, rand])
                right_top = Point([max_x, rand + 4])

            new_polygon = Polygon(
                [left_top, left_bottom, right_bottom, right_top])

        # print('creat a ped_crosing polygon')

        return new_polygon


def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):
    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon


class PerturbedVectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }

    def __init__(self,
                 nusc_map,
                 map_explorer,
                 patch_size,
                 map_classes=['divider', 'ped_crossing',
                              'boundary', 'centerline'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 centerline_classes=['lane_connector', 'lane'],
                 use_simplify=True,
                 ):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.centerline_classes = centerline_classes
        self.patch_size = patch_size
        self.map_trans = MapTransform(self.map_explorer)

    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        map_dict = {'divider': [], 'ped_crossing': [],
                    'boundary': [], 'centerline': []}
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'centerline':
                centerline_geom = self.get_centerline_geom(
                    patch_box, patch_angle, self.centerline_classes)
                centerline_list = self.centerline_geoms_to_instances(
                    centerline_geom)
                for instance in centerline_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        return map_dict

    def gen_trans_vectorized_samples(self, lidar2global_translation, lidar2global_rotation, trans_args):
        '''
        get transformed gt map layers
        '''

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        if trans_args['shi_map'][0]:
            # Move randomly within certain ration of the side length
            patch_box = (map_pose[0] + random.randint(-self.patch_size[1]*trans_args['shi_map'][1], self.patch_size[1]*trans_args['shi_map'][1]),
                         map_pose[1] + random.randint(-self.patch_size[0]*trans_args['shi_map']
                                                      [1], self.patch_size[0]*trans_args['shi_map'][1]),
                         self.patch_size[0], self.patch_size[1])
        else:
            patch_box = (map_pose[0], map_pose[1],
                         self.patch_size[0], self.patch_size[1])

        if trans_args['rot_map'][0]:
            # Rotate instantly within a certain degrees
            patch_angle = quaternion_yaw(
                rotation) / np.pi * 180 + random.randint(-trans_args['rot_map'][1], trans_args['rot_map'][1])
            # print('patch_box is shiffted (%d, %d) and rotated %d degree'%(trans_args['patch_args'][0][1][0], trans_args['patch_args'][0][1][1], trans_args['patch_args'][0][0]))
        else:
            patch_angle = quaternion_yaw(rotation) / np.pi * 180

        map_dict = {'divider': [], 'ped_crossing': [],
                    'boundary': [], 'centerline': []}

        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes, trans_args)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes, trans_args)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, trans_args)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'centerline':
                centerline_geom = self.get_centerline_geom(
                    patch_box, patch_angle, self.centerline_classes)
                centerline_list = self.centerline_geoms_to_instances(
                    centerline_geom)
                for instance in centerline_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        return map_dict

    def get_centerline_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.centerline_classes:
                return_token = False
                layer_centerline_dict = self.map_explorer._get_centerline(
                    patch_box, patch_angle, layer_name, return_token=return_token)
                if len(layer_centerline_dict.keys()) == 0:
                    continue
                map_geom.update(layer_centerline_dict)
        return map_geom

    def get_map_geom(self, patch_box, patch_angle, layer_names, trans_args=None):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                if trans_args is not None:
                    geoms = self.get_trans_divider_line(
                        patch_box, patch_angle, layer_name, trans_args)
                else:
                    geoms = self.get_divider_line(
                        patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                # if trans_args is not None:
                #     geoms = self.get_trans_contour_line(
                #         patch_box, patch_angle, layer_name, trans_args)
                # else:
                #     geoms = self.get_contour_line(
                #         patch_box, patch_angle, layer_name)
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                if trans_args is not None:
                    geoms = self.get_trans_ped_crossing_line(
                        patch_box, patch_angle, trans_args)
                else:
                    geoms = self.get_ped_crossing_line(patch_box, patch_angle)
                map_geom[layer_name] = geoms
        return map_geom

    def valid_line(self, line, patch, patch_angle, patch_x, patch_y, line_list):
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.rotate(
                new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_line = affinity.affine_transform(new_line,
                                                 [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            line_list.append(new_line)

        return line_list

    def get_divider_line(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_trans_divider_line(self, patch_box, patch_angle, layer_name, trans_args=None):
        # 'road_divider', 'lane_divider', 'traffic_light'
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_coords = self.map_trans.patch_box_2_coords(patch_box)
        records = self.map_explorer.map_api.get_records_in_patch(
            patch_coords, [layer_name])[layer_name]

        if trans_args['del_ped'][0]:
            if len(records):
                # del_ped_token.append(random.choice(ped_records))
                records.pop(random.randrange(len(records)))

        line_list = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        for token in records:
            record = self.map_explorer.map_api.get(layer_name, token)
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            line_list = self.valid_line(
                line, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def get_contour_line(self, patch_box, patch_angle, layer_name):
        # 'road_segment', 'lane'
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer.map_api.extract_polygon(
                    polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer.map_api.extract_polygon(
                    record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def valid_polygon(self, polygon, patch, patch_angle, patch_x, patch_y, polygon_list):
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                              origin=(patch_x, patch_y), use_radians=False)
                new_polygon = affinity.affine_transform(new_polygon,
                                                        [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)

        return polygon_list

    def get_ped_crossing_line(self, patch_box, patch_angle):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer.map_api, 'ped_crossing')
        for record in records:
            polygon = self.map_explorer.map_api.extract_polygon(
                record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def get_trans_ped_crossing_line(self, patch_box, patch_angle, trans_args):

        patch_coords = self.map_trans.patch_box_2_coords(patch_box)
        records = self.map_explorer.map_api.get_records_in_patch(
            patch_coords, ['ped_crossing'])['ped_crossing']

        if trans_args['del_ped'][0]:
            if len(records):
                # del_ped_token.append(random.choice(ped_records))
                records.pop(random.randrange(len(records)))

        polygon_list = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        # TODO
        # if trans_args['shi_ped'][0]:
        #     if len(records):
        #         shi_record = records.pop(random.randrange(len(records)))
        #         polygon = self.map_trans.shift_ped_polygon(shi_record)
        #         polygon_list = self.valid_polygon(
        #             polygon, patch, patch_x, patch_y, polygon_list)

        for token in records:
            record = self.map_explorer.map_api.get('ped_crossing', token)
            polygon = self.map_explorer.map_api.extract_polygon(
                record['polygon_token'])
            polygon_list = self.valid_polygon(
                polygon, patch, patch_angle, patch_x, patch_y, polygon_list)

        if trans_args['add_ped'][0]:
            rod_seg_records = self.map_explorer.map_api.get_records_in_patch(patch_coords, [
                'road_segment'])['road_segment']
            if len(rod_seg_records):
                record = random.choice(rod_seg_records)
                polygon = self.map_trans.creat_ped_polygon(record)
                polygon_list = self.valid_polygon(
                    polygon, patch, patch_angle, patch_x, patch_y, polygon_list)

        return polygon_list

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom.items():
            one_type_instances = self._one_type_line_geom_to_instances(
                a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom['ped_crossing']
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom['road_segment']
        lanes = polygon_geom['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def centerline_geoms_to_instances(self, geoms_dict):
        centerline_geoms_list, pts_G = self.union_centerline(geoms_dict)
        # vectors_dict = self.centerline_geoms2vec(centerline_geoms_list)
        # import ipdb;ipdb.set_trace()
        # if len(centerline_geoms_list):
        # print('not emtpy')

        return self._one_type_line_geom_to_instances(centerline_geoms_list)

    def centerline_geoms2vec(self, centerline_geoms_list):
        vector_dict = {}
        vectors = self._geom_to_vectors(
            centerline_geoms_list)
        vector_dict.update({'centerline': ('centerline', vectors)})
        return vector_dict

    def union_centerline(self, centerline_geoms):
        pts_G = nx.DiGraph()
        junction_pts_list = []
        for key, value in centerline_geoms.items():
            centerline_geom = value['centerline']
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(
                    centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(
                    centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx, pt in enumerate(single_geom_pts[:-1]):
                        pts_G.add_edge(tuple(single_geom_pts[idx]), tuple(
                            single_geom_pts[idx+1]))
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx, pts in enumerate(centerline_pts[:-1]):
                    pts_G.add_edge(tuple(centerline_pts[idx]), tuple(
                        centerline_pts[idx+1]))
            else:
                raise NotImplementedError
            valid_incoming_num = 0
            for idx, pred in enumerate(value['incoming_tokens']):
                if pred in centerline_geoms.keys():
                    valid_incoming_num += 1
                    pred_geom = centerline_geoms[pred]['centerline']
                    if pred_geom.geom_type == 'MultiLineString':
                        pred_pt = np.array(
                            pred_geom.geoms[-1].coords).round(3)[-1]
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
                    else:
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))

            valid_outgoing_num = 0
            for idx, succ in enumerate(value['outgoing_tokens']):
                if succ in centerline_geoms.keys():
                    valid_outgoing_num += 1
                    succ_geom = centerline_geoms[succ]['centerline']
                    if succ_geom.geom_type == 'MultiLineString':
                        succ_pt = np.array(
                            succ_geom.geoms[0].coords).round(3)[0]
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
                    else:
                        succ_pt = np.array(succ_geom.coords).round(3)[0]
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))

        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            try:
                paths = nx.all_simple_paths(pts_G, root, leaves)
                all_paths.extend(paths)
            except:
                continue

        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_G


def obtain_perturb_vectormap(nusc_maps, map_explorer, info, point_cloud_range):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(
        info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = info['ego2global_translation']

    lidar2global = ego2global @ lidar2ego

    lidar2global_translation = list(lidar2global[:3, 3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

    location = info['map_location']

    patch_h = point_cloud_range[4]-point_cloud_range[1]
    patch_w = point_cloud_range[3]-point_cloud_range[0]
    patch_size = (patch_h, patch_w)
    vector_map = PerturbedVectorizedLocalMap(
        nusc_maps[location], map_explorer[location], patch_size)
    map_anns = vector_map.gen_vectorized_samples(
        lidar2global_translation, lidar2global_rotation)
    info["annotation"] = map_anns

    trans_args = {'add_ped': [1, None],  # add a ped_crossing in a road_segment
                  'del_ped': [1, None],  # delet a ped_crossing
                  # shift a ped_crossing in its road_segment
                  'shi_ped': [1, None],
                  'del_roa_div': [1, None],  # delete a road_divider
                  'del_lan_div': [1, None],  # delete a lane_divier
                  # [shift the map, the ration of the length that the shift range does not exceed ]
                  'shi_map': [1, 0.3],
                  # [rotate the map, the degree of the rotate range]
                  'rot_map': [1, 54]
                  }

    map_anns_tran = vector_map.gen_trans_vectorized_samples(
        lidar2global_translation, lidar2global_rotation, trans_args)
    info["annotation_1"] = map_anns_tran

    return info
    # - setup
    local_dataroot = '/home/li/Documents/map/data/sets/nuscenes'
    # local_dataroot = '/../../../../data/sets/nuscenes'
    data_version = 'v1.0-mini'
    # `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`, `boston-seaport`
    map_ocation = 'singapore-onenorth'

    # - init NuScenes
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=local_dataroot, verbose=False)

    # - init NuScenes Map
    # nusc_map = NuScenesMap(dataroot=local_dataroot, map_name=map_ocation)
    # bitmap_one = BitMap(local_dataroot, map_ocation, 'basemap')  # for render
    map_explorer = NuScenesMapExplorer(NuScenesMap(
        dataroot=local_dataroot, map_name=map_ocation))

    map_trans = MapTransform(map_explorer)
    # print(map_trans.layer_names)

    # take a patch
    patch_box = [300, 1700, 100, 100]  # x_center, y_center, height, width
    patch_coords = map_trans.patch_box_2_coords(patch_box)
    # patch_coords = (250, 1650, 350, 1750) # x_min, y_min, x_max, y_max
    patch_angle = 0
    # layer_names = ['ped_crossing', 'road_segment', 'road_block', 'road_divider']

    ped_cro_records = map_explorer.map_api.get_records_in_patch(
        patch_coords, ['ped_crossing'])  # intersect

    # shift a ped_crossing
    ped_cro_token = ped_cro_records['ped_crossing'][0]

    new_polygon_list_shi = map_trans.transfor_layer(
        patch_box, patch_angle, 'ped_crossing', ped_cro_token, shift=[4, 0])

    # delet a ped_crossing
    new_polygon_list_del = map_trans.delte_layers(
        patch_box, patch_angle, ['ped_crossing'], ped_cro_token)

    # add a ped_crossing
    roa_seg_records = map_explorer.map_api.get_records_in_patch(patch_coords, [
                                                                'road_segment'])
    roa_seg_token = roa_seg_records['road_segment'][0]

    new_polygon_list_add = map_trans.add_ped_crossing_layer(
        patch_box, patch_angle, roa_seg_token)

    print('FINISH')
