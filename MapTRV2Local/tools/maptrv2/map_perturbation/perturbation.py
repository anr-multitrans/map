import copy
import os
import random

import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import MultiLineString, MultiPolygon, Point, Polygon, box
from visualization import RenderMap


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

        return new_polygon


class PerturbedVectorizedLocalMap(object):

    def __init__(self,
                 nusc_map,
                 map_explorer,
                 patch_size,
                 map_classes=['divider', 'ped_crossing',
                              'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane']  # 'lane_connector'
                 ):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.patch_size = patch_size
        self.map_trans = MapTransform(self.map_explorer)

    def gen_trans_vectorized_samples(self, lidar2global_translation, lidar2global_rotation, trans_args):
        '''
        get transformed gt map layers
        '''

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        if trans_args.shi_map[0]:
            # Move randomly within certain ration of the side length
            patch_box = (map_pose[0] + random.randint(-self.patch_size[1]*trans_args.shi_map[1], self.patch_size[1]*trans_args.shi_map[1]),
                         map_pose[1] + random.randint(-self.patch_size[0]*trans_args.shi_map
                                                      [1], self.patch_size[0]*trans_args.shi_map[1]),
                         self.patch_size[0], self.patch_size[1])
        else:
            patch_box = (map_pose[0], map_pose[1],
                         self.patch_size[0], self.patch_size[1])

        if trans_args.rot_map[0]:
            # Rotate instantly within a certain degrees
            patch_angle = quaternion_yaw(
                rotation) / np.pi * 180 + random.randint(-trans_args.rot_map[1], trans_args.rot_map[1])
        else:
            patch_angle = quaternion_yaw(rotation) / np.pi * 180

        map_dict = {'divider': [], 'ped_crossing': [],
                    'boundary': []}
        map_perturb = {'divider': [], 'ped_crossing': [],
                       'boundary': []}
        geom = {}
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom, map_perturb[vec_class] = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes, trans_args)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                geom.update(line_instances_dict)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'ped_crossing':
                ped_geom, map_perturb[vec_class] = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes, trans_args)
                geom.update(ped_geom)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom, map_perturb[vec_class] = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, trans_args)
                geom.update(polygon_geom)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        return {'map_dict': map_dict, 'map_geom': geom, 'patch_box': patch_box, 'patch_angle': patch_angle, 'map_perturb': map_perturb}

    def get_map_geom(self, patch_box, patch_angle, layer_names, trans_args=None):
        map_geom = {}
        map_perturb = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms, map_pt = self.get_trans_divider_line(
                    patch_box, patch_angle, layer_name, trans_args)
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                geoms, map_pt = self.get_trans_contour_line(
                    patch_box, patch_angle, layer_name, trans_args)  # TODO
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                geoms, map_pt = self.get_trans_ped_crossing_line(
                    patch_box, patch_angle, trans_args)
                map_geom[layer_name] = geoms
            map_perturb.append(map_pt)

        return map_geom, map_perturb

    def valid_polygon(self, polygon, patch, patch_angle, patch_x, patch_y, polygon_list):
        valid = False
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
                valid = True

        return polygon_list, valid

    def valid_line(self, line, patch, patch_angle, patch_x, patch_y, line_list):
        new_line = line.intersection(patch)
        valid = False
        if not new_line.is_empty:
            new_line = affinity.rotate(
                new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_line = affinity.affine_transform(new_line,
                                                 [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            line_list.append(new_line)
            valid = True

        return line_list, valid

    def get_trans_divider_line(self, patch_box, patch_angle, layer_name, trans_args=None):
        # 'road_divider', 'lane_divider', 'traffic_light'
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_coords = self.map_trans.patch_box_2_coords(patch_box)
        records = self.map_explorer.map_api.get_records_in_patch(
            patch_coords, [layer_name])[layer_name]

        ind_list = [i for i in range(len(records))]
        map_pt = [None]*len(records)
        if trans_args.del_lan_div[0]:
            for _ in range(trans_args.del_lan_div[1]):
                if len(ind_list):
                    ind = ind_list.pop(random.randrange(len(ind_list)))
                    records[ind] = None
                    map_pt[ind] = -1

        line_list = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        for ind in ind_list:
            record = self.map_explorer.map_api.get(layer_name, records[ind])
            line = self.map_explorer.map_api.extract_line(
                record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            line_list, valid = self.valid_line(
                line, patch, patch_angle, patch_x, patch_y, line_list)
            if valid:
                map_pt[ind] = 0

        return line_list, map_pt

    def get_trans_contour_line(self, patch_box, patch_angle, layer_name, trans_args=None):
        # 'road_segment', 'lane'
        if layer_name not in self.map_explorer.map_api.lookup_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_coords = self.map_trans.patch_box_2_coords(patch_box)
        records = self.map_explorer.map_api.get_records_in_patch(
            patch_coords, [layer_name])[layer_name]

        ind_list = [i for i in range(len(records))]
        map_pt = [None]*len(records)

        polygon_list = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        for ind in ind_list:
            record = self.map_explorer.map_api.get(layer_name, records[ind])
            polygon = self.map_explorer.map_api.extract_polygon(
                record['polygon_token'])

            polygon_list, valid = self.valid_polygon(
                polygon, patch, patch_angle, patch_x, patch_y, polygon_list)
            if valid:
                map_pt[ind] = 0

        return polygon_list, map_pt

    def get_trans_ped_crossing_line(self, patch_box, patch_angle, trans_args=None):

        patch_coords = self.map_trans.patch_box_2_coords(patch_box)
        records = self.map_explorer.map_api.get_records_in_patch(
            patch_coords, ['ped_crossing'])['ped_crossing']

        ind_list = [i for i in range(len(records))]
        map_pt = [None]*len(records)
        if trans_args.del_ped[0]:
            for _ in range(trans_args.del_ped[1]):
                if len(ind_list):
                    ind = ind_list.pop(random.randrange(len(ind_list)))
                    records[ind] = None
                    map_pt[ind] = -1

        polygon_list = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        if trans_args.shi_ped[0]:
            for _ in range(trans_args.del_ped[1]):
                if len(ind_list):
                    ind = ind_list.pop(random.randrange(len(ind_list)))
                    token = records[ind]
                    record = self.map_explorer.map_api.get(
                        'ped_crossing', token)
                    polygon = self.map_trans.creat_ped_polygon(
                        record['road_segment_token'])
                    polygon_list, valid = self.valid_polygon(
                        polygon, patch, patch_angle, patch_x, patch_y, polygon_list)
                    if valid:
                        map_pt[ind] = 2

        for ind in ind_list:
            record = self.map_explorer.map_api.get(
                'ped_crossing', records[ind])
            polygon = self.map_explorer.map_api.extract_polygon(
                record['polygon_token'])
            polygon_list, valid = self.valid_polygon(
                polygon, patch, patch_angle, patch_x, patch_y, polygon_list)
            if valid:
                map_pt[ind] = 0

        if trans_args.add_ped[0]:
            for _ in range(trans_args.add_ped[1]):
                rod_seg_records = self.map_explorer.map_api.get_records_in_patch(patch_coords, [
                    'road_segment'])['road_segment']
                if len(rod_seg_records):
                    record = random.choice(rod_seg_records)
                    polygon = self.map_trans.creat_ped_polygon(record)
                    polygon_list, valid = self.valid_polygon(
                        polygon, patch, patch_angle, patch_x, patch_y, polygon_list)
                    if valid:
                        map_pt.append(1)

        return polygon_list, map_pt

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


class PerturbParameters():
    def __init__(self,
                 add_ped=[0, 1],  # add a ped_crossing in a road_segment
                 del_ped=[0, 1],  # delet a ped_crossing
                 shi_ped=[0, 1],  # shift a ped_crossing in its road_segment
                 del_roa_div=[0, 1],  # delete a road_divider
                 del_lan_div=[0, 1],  # delete a lane_divier
                 del_lan=[0, 1],  # delete a lane
                 shi_lan=[0, 1],
                 def_lan=[0, 1],
                 # [shift the map, the ration of the length that the shift range does not exceed ]
                 shi_map=[0, 1],
                 # [rotate the map, the degree of the rotate range]
                 rot_map=[0, 1]
                 ):

        self.add_ped = add_ped
        self.del_ped = del_ped
        self.shi_ped = shi_ped
        self.del_roa_div = del_roa_div
        self.del_lan_div = del_lan_div
        self.del_lan = del_lan
        self.shi_lan = shi_lan
        self.def_lan = def_lan
        self.shi_map = shi_map
        self.rot_map = rot_map


def re_index(lis_trans, start_ind):
    ind = start_ind
    ind_list = []
    for tran in lis_trans:
        if tran in [None, -1]:
            ind_list.append(None)
        else:
            ind_list.append(ind)
            ind += 1

    return ind_list


def map_correspondence(ann_org_pert, ann_pt_pert, map_version):
    corr_dic = {}
    for layers in ['divider', 'ped_crossing']:  # TODO Boundaries are more complicated
        layers_org = ann_org_pert[layers]
        layers_pert = ann_pt_pert[layers]

        pt_corr_ind = []
        start_ind = 0
        for ind, lis in enumerate(layers_pert):
            layer_org_ind_list = re_index(layers_org[ind], start_ind)
            start_ind = len(layer_org_ind_list)

            for id, tran in enumerate(lis):
                if tran in [None, -1]:
                    continue

                if id < len(layer_org_ind_list):
                    pt_corr_ind.append(layer_org_ind_list[id])
                else:
                    pt_corr_ind.append(-1)

        corr_dic[layers] = pt_corr_ind

    return corr_dic


def perturb_map(vector_map, lidar2global_translation, lidar2global_rotation, trans_args, info, map_version, visual):

    trans_dic = vector_map.gen_trans_vectorized_samples(
        lidar2global_translation, lidar2global_rotation, trans_args)
    info[map_version] = trans_dic['map_dict']

    if '_' in map_version:
        corr_dic = map_correspondence(
            info['annotation_perturb'], trans_dic['map_perturb'], map_version)
        corr_dic['boundary'] = [i for i in range(
            len(info['annotation']['boundary']))]
        info[map_version+'_correspondence'] = corr_dic
    else:
        info[map_version+'_perturb'] = trans_dic['map_perturb']

    visual.vis_contours(trans_dic['map_dict'],
                        trans_dic['patch_box'], map_version)

    return info


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

    save_path = os.path.join(
        '/home/li/Documents/map/MapTRV2Local/tools/maptrv2/map_perturbation/visual', info['scene_token'], info['token'])
    visual = RenderMap(info, vector_map.nusc_map, vector_map.map_explorer,
                       switch=False, show=False, save=save_path)

    # the oranginal map
    map_version = 'annotation'
    trans_args = PerturbParameters()
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the first perturbed map
    map_version = 'annotation_1'
    trans_args = PerturbParameters(del_ped=[1, 1],  # delet a ped_crossing
                                   del_lan=[1, 1],
                                   shi_lan=[1, 1])  # delete a lane
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the second perturbed map
    map_version = 'annotation_2'
    trans_args = PerturbParameters(del_lan_div=[1, 1],
                                   del_lan=[1, 1])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the third perturbed map
    map_version = 'annotation_3'
    trans_args = PerturbParameters(del_lan=[1, 1],
                                   shi_map=[1, 0.2])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the fourth perturbed map
    map_version = 'annotation_4'
    trans_args = PerturbParameters(add_ped=[1, 1],
                                   del_ped=[1, 2],
                                   del_lan_div=[1, 1],
                                   del_lan=[1, 1])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the fifth perturbed map
    map_version = 'annotation_5'
    trans_args = PerturbParameters(add_ped=[1, 1],
                                   del_ped=[1, 2],
                                   del_lan_div=[1, 2],
                                   del_lan=[1, 1],
                                   shi_map=[1, 0.2])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    info.pop('annotation_perturb')

    return info
