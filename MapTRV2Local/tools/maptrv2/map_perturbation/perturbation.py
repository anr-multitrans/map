import math
import random
from matplotlib import pyplot as plt

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

    # rotate=0, scale=[1, 1], skew=[0, 0, 'center'], shift=[0, 0], transform=[1, 0, 0, 1, 0, 0])
    def transfor_patch(self, instance_list, correspondence_list, patch_box, patch_angle, args):
        for key in instance_list.keys():
            if len(instance_list[key]):
                for ind, ins in enumerate(instance_list[key]):
                    if args.aff_tra_pat[0]:
                        geom = affinity.affine_transform(
                            instance_list[key][ind], args.aff_tra_pat[2])
                    if args.rot_pat[0]:
                        geom = affinity.rotate(
                            instance_list[key][ind], args.rot_pat[2][0], args.rot_pat[2][1])
                    if args.sca_pat[0]:
                        geom = affinity.scale(
                            instance_list[key][ind], args.sca_pat[2][0], args.sca_pat[2][1])
                    if args.ske_pat[0]:
                        geom = affinity.skew(
                            instance_list[key][ind], args.sca_ske_patpat[2][0], args.ske_pat[2][1], args.ske_pat[2][2])
                    if args.ske_pat[0]:
                        geom = affinity.translate(
                            instance_list[key][ind], args.ske_pat[2][0], args.ske_pat[2][1])

                    geom = self.valid_geom(geom, patch_box, patch_angle)
                    if geom is None:
                        instance_list[key].pop(ind)
                        correspondence_list[key].pop(ind)
                    else:
                        instance_list[key][ind] = geom

        return instance_list, correspondence_list

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

    def show_img(self, X):
        plt.imshow(X, interpolation='nearest')
        plt.show()
        plt.close()

    def delete_layers(self, instance_list, correspondence_list, layer_name, del_args):
        for _ in range(del_args[1]):
            if len(instance_list[layer_name]):
                ind = random.randrange(len(instance_list[layer_name]))
                instance_list[layer_name].pop(ind)
                correspondence_list[layer_name].pop(ind)

        return instance_list, correspondence_list

    def shift_layers(self, instance_list, correspondence_list, layer_name, shi_args,  patch_box, patch_angle):
        for _ in range(shi_args[1]):
            if len(instance_list[layer_name]):
                ind = random.randrange(len(instance_list[layer_name]))

                geom = affinity.translate(
                    instance_list[layer_name][ind], shi_args[2][0], shi_args[2][1])
                geom = self.valid_geom(geom, patch_box, patch_angle)
                if geom is None:
                    instance_list['ped_crossing'].pop(ind)
                    correspondence_list['ped_crossing'].pop(ind)
                else:
                    instance_list[layer_name][ind] = geom

        return instance_list, correspondence_list

    def valid_geom(self, geom,  patch_box, patch_angle):
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        new_geom = geom.intersection(patch)
        if not new_geom.is_empty:
            new_geom = affinity.rotate(
                new_geom, patch_angle, origin=(patch_box[0], patch_box[1]), use_radians=False)
            new_geom = affinity.affine_transform(new_geom,
                                                 [1.0, 0.0, 0.0, 1.0, -patch_box[0], -patch_box[1]])

            if new_polygon.geom_type == 'Polygon':
                new_polygon = MultiPolygon([new_polygon])
        else:
            return None

        return new_polygon

    def difromate_map(self, map_ins_dict, def_args, patch_box):

        # Horizontal distortion amplitude random maximum range int
        Max_v = def_args[2][0]
        # Vertical distortion amplitude random maximum range int
        Max_h = def_args[2][1]
        Max_r = def_args[2][2]  # Inclination amplitude [-Max_r Max_r] int

        xlim = [-patch_box[3]/2, patch_box[3]/2]
        ylim = [-patch_box[2]/2, patch_box[2]/2]

        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    map_ins_dict[key][ind] = self.con(
                        ins, xlim, ylim, Max_v, Max_h, Max_r)

        return map_ins_dict

    def con(self, ins, xlim, ylim, Max_v, Max_h, Max_r):
        randx = Max_v
        randy = Max_h
        randr = Max_r

        new_point_list = []
        for point in ins:

            j = point[0]
            i = point[1]

            offset_x = randx * math.sin(2 * 3.14 * i / 150)
            offset_y = randy * math.cos(2 * 3.14 * j / 150)
            offset_x += randr * math.sin(2 * 3.14 * i / (2*(xlim[1]-xlim[0])))

            new_point = []
            if j in xlim:
                new_point.append(j)
            else:
                if xlim[0] < j+offset_x < xlim[1]:
                    new_point.append(j+offset_x)
                elif xlim[1] < j+offset_x:
                    new_point.append(xlim[1])
                elif j+offset_x < xlim[0]:
                    new_point.append(xlim[0])

            if i in ylim:
                new_point.append(i)
            else:
                if ylim[0] < i+offset_y < ylim[1]:
                    new_point.append(i+offset_y)
                elif ylim[1] < i+offset_y:
                    new_point.append(ylim[1])
                elif i+offset_y < ylim[0]:
                    new_point.append(ylim[0])

            new_point_list.append(np.array(new_point))

        new_ins = np.array(new_point_list)
        return new_ins


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

    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation, trans_args):
        '''
        get transformed gt map layers
        '''

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])

        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        map_ins_dict = {'divider': [], 'ped_crossing': [],
                        'boundary': []}

        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    map_ins_dict[vec_class] += instances
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes)
                map_ins_dict[vec_class] = self.ped_poly_geoms_to_instances(
                    ped_geom)
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes)
                map_ins_dict[vec_class] = self.poly_geoms_to_instances(
                    polygon_geom)
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        return {'map_ins_dict': map_ins_dict, 'patch_box': patch_box, 'patch_angle': patch_angle}

    def geom_to_np(self, map_ins_dict):
        map_dict = {'divider': [], 'ped_crossing': [],
                    'boundary': []}

        for vec_class in map_ins_dict.keys():
            if len(map_ins_dict[vec_class]):
                for instance in map_ins_dict[vec_class]:
                    map_dict[vec_class].append(np.array(instance.coords))

        return map_dict

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(
                    patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name)  # TODO
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle)
                map_geom[layer_name] = geoms

        return map_geom

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

    def get_divider_line(self, patch_box, patch_angle, layer_name):
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

        return line_list

    def get_contour_line(self, patch_box, patch_angle, layer_name):
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

        return polygon_list

    def get_ped_crossing_line(self, patch_box, patch_angle, trans_args=None):

        patch_coords = self.map_trans.patch_box_2_coords(patch_box)
        records = self.map_explorer.map_api.get_records_in_patch(
            patch_coords, ['ped_crossing'])['ped_crossing']

        ind_list = [i for i in range(len(records))]
        map_pt = [None]*len(records)

        polygon_list = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        for ind in ind_list:
            record = self.map_explorer.map_api.get(
                'ped_crossing', records[ind])
            polygon = self.map_explorer.map_api.extract_polygon(
                record['polygon_token'])
            polygon_list, valid = self.valid_polygon(
                polygon, patch, patch_angle, patch_x, patch_y, polygon_list)
            if valid:
                map_pt[ind] = 0

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

    def get_trans_instance(self, map_ins_dict, trans_args, patch_box, patch_angle):
        corr_dict = {'divider': [], 'ped_crossing': [],
                     'boundary': []}

        for vec_class in map_ins_dict.keys():
            if len(map_ins_dict[vec_class]):
                corr_dict[vec_class] = [
                    i for i in range(len(map_ins_dict[vec_class]))]

        if trans_args.del_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, 'ped_crossing', trans_args.del_ped)

        if trans_args.shi_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, 'ped_crossing', trans_args.shi_ped,  patch_box, patch_angle)

        if trans_args.add_ped[0]:
            pass  # TODO

        if trans_args.del_div[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, 'divider', trans_args.del_ped)

        if trans_args.shi_div[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, 'ped_crossing', trans_args.shi_ped,  patch_box, patch_angle)

        if trans_args.add_div[0]:
            pass  # TODO

        if trans_args.del_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, 'boundary', trans_args.del_ped)

        if trans_args.shi_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, 'ped_crossing', trans_args.shi_ped,  patch_box, patch_angle)

        if trans_args.add_bou[0] or trans_args.aff_tra_pat[0] or trans_args.rot_pat[0] or trans_args.sca_pat[0] or trans_args.ske_pat[0] or trans_args.shi_pat[0]:
            map_ins_dict, corr_dict = self.map_trans.transfor_patch(
                map_ins_dict, corr_dict, patch_box, patch_angle, trans_args)

        return map_ins_dict, corr_dict


class PerturbParameters():
    def __init__(self,
                 # [switch, number, args]
                 # ped_crossing perturbation
                 del_ped=[0, 1, None],  # delete ped_crossing
                 # shift ped_crossing in its road_segment, shifted by offsets along each dimension[x, y]
                 shi_ped=[0, 1, [0, 0]],
                 # add ped_crossing in a road_segment TODO
                 add_ped=[0, 1, None],
                 # dividers perturbation
                 del_div=[0, 1, None],  # delete divider
                 # shift divider, shifted by offsets along each dimension[x, y]
                 shi_div=[0, 1, [0, 0]],
                 add_div=[0, 1, None],  # add divider TODO
                 # boundray perturabtion
                 del_bou=[0, 1, None],  # delete lane
                 # shift lane, shifted by offsets along each dimension[x, y]
                 shi_bou=[0, 1, [0, 0]],
                 add_bou=[0, 1, None],  # add lane TODO
                 # patch perturbation
                 aff_tra_pat=[0, 1, [1, 0, 0, 1, 0, 0]],  # affine_transform
                 rot_pat=[0, 1, [0, [0, 0]]],  # rotate the patch
                 sca_pat=[0, 1, [1, 1]],  # scale the patch
                 ske_pat=[0, 1, [0, 0, (0, 0)]],  # skew the patch
                 shi_pat=[0, 1, [0, 0]],  # translate: shift the patch
                 # deformation the patch, Inclination amplitude [-Max_r Max_r] int
                 def_pat=[0, 1, [0, 0, 0]],
                 # visulization
                 vis_path='/home/li/Documents/map/MapTRV2Local/tools/maptrv2/map_perturbation/visual',
                 visual=True,
                 vis_show=False
                 ):

        self.del_ped = del_ped
        self.shi_ped = shi_ped
        self.add_ped = add_ped
        self.del_div = del_div
        self.shi_div = shi_div
        self.add_div = add_div
        self.del_bou = del_bou
        self.shi_bou = shi_bou
        self.add_bou = add_bou
        self.aff_tra_pat = aff_tra_pat
        self.rot_pat = rot_pat
        self.sca_pat = sca_pat
        self.ske_pat = ske_pat
        self.shi_pat = shi_pat
        self.def_pat = def_pat

        self.vis_path = vis_path
        self.visual = visual
        self.vis_show = vis_show


def perturb_map(vector_map, lidar2global_translation, lidar2global_rotation, trans_args, info, map_version, visual):

    trans_dic = vector_map.gen_vectorized_samples(
        lidar2global_translation, lidar2global_rotation, trans_args)

    if '_' not in map_version:
        trans_np_dict = vector_map.geom_to_np(trans_dic['map_ins_dict'])
    else:
        trans_ins, corr_dict = vector_map.get_trans_instance(
            trans_dic['map_ins_dict'], trans_args, trans_dic['patch_box'], trans_dic['patch_angle'])
        info[map_version+'_correspondence'] = corr_dict
        trans_np_dict = vector_map.geom_to_np(trans_ins)
        if trans_args.def_pat[0]:
            trans_np_dict = vector_map.map_trans.difromate_map(
                trans_np_dict, trans_args.def_pat, trans_dic['patch_box'])

    info[map_version] = trans_np_dict

    visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)

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
    trans_args = PerturbParameters(
        visual=True, vis_show=False, vis_path='/home/li/Documents/map/MapTRV2Local/tools/maptrv2/map_perturbation/visual')
    visual = RenderMap(info, vector_map, trans_args)

    # ------peturbation------
    # the oranginal map
    map_version = 'annotation'
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the first perturbed map
    map_version = 'annotation_1'
    trans_args = PerturbParameters(del_ped=[1, 1, None],
                                   shi_ped=[1, 1, [5, 5]])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # sys.exit()

    # the second perturbed map
    map_version = 'annotation_2'
    trans_args = PerturbParameters(del_div=[1, 1, None],
                                   shi_div=[1, 1, [5, 5]])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the third perturbed map
    map_version = 'annotation_3'
    trans_args = PerturbParameters(del_bou=[1, 1, None],
                                   shi_bou=[1, 1, [5, 5]])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the fourth perturbed map
    map_version = 'annotation_4'
    trans_args = PerturbParameters(rot_pat=[1, 1, [5, [0, 0]]],
                                   shi_pat=[1, 1, [5, 5]])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    # the fifth perturbed map
    map_version = 'annotation_5'
    trans_args = PerturbParameters(def_pat=[1, 1, [0, 0, 5]])
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)

    return info
