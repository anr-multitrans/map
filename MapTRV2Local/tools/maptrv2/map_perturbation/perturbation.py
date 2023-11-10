import asyncio
import copy
import math
import random
from matplotlib import pyplot as plt
import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import MultiLineString, MultiPolygon, Point, Polygon, box, LineString
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

    def transfor_patch(self, instance_list, correspondence_list, patch_box, args):
        for key in instance_list.keys():
            if len(instance_list[key]):
                for ind, ins in enumerate(instance_list[key]):
                    if args.aff_tra_pat[0]:
                        ins = affinity.affine_transform(
                            ins, args.aff_tra_pat[2])
                    if args.rot_pat[0]:
                        ins = affinity.rotate(
                            ins, args.rot_pat[2][0], args.rot_pat[2][1])
                    if args.sca_pat[0]:
                        ins = affinity.scale(
                            ins, args.sca_pat[2][0], args.sca_pat[2][1])
                    if args.ske_pat[0]:
                        ins = affinity.skew(
                            ins, args.ske_pat[2][0], args.ske_pat[2][1], args.ske_pat[2][2])
                    if args.shi_pat[0]:
                        ins = affinity.translate(
                            ins, args.shi_pat[2][0], args.shi_pat[2][1])

                    geom = self.valid_geom(ins, [0,0,patch_box[2],patch_box[3]], 0)
                    if geom is None:
                        instance_list[key].pop(ind)
                        correspondence_list[key].pop(ind)
                    else:
                        if geom.geom_type == 'MultiLineString':
                            instance_list[key][ind]= ops.linemerge(geom)
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

    def creat_boundray(self, boundary, patch_box):
        center_point = boundary.interpolate(boundary.lenth / 2)
        c = center_point.buffer(2.5).boundary
        bi = c.intersection(boundary)

        if len(bi.geom) < 2:
            pt_1 = boundary.coords[0]
            pt_2 = boundary.coords[-1]
        else:
            pt_1 = bi.geom[0]
            pt_2 = bi.geom[-1]

        x = center_point[0]
        y = center_point[1]
        edges_distance = [x+15, 15-x, y+30, 30-y]   # left, right, down, up
        ind = edges_distance.index(min[edges_distance])

        r_x = random.uniform(-12.5, 12.5)
        r_y = random.uniform(-27.5, 27.5)
        rand_point_on_edges = [[(-15, r_y-2.5), (-15, r_y+2.5)], [(15, r_y-2.5), (15, r_y+2.5)], [
            (r_x-2.5, -30), (r_x+2.5, -30)], [(r_x-2.5, 30), (r_x+2.5, 30)]]

        ed_pt_1 = rand_point_on_edges[ind][0]
        ed_pt_2 = rand_point_on_edges[ind][1]

        new_b_1 = LineString(ed_pt_1, pt_1)
        new_b_2 = LineString(ed_pt_2, pt_2)

        if new_b_1.intersects(new_b_2):
            new_b_1 = LineString(ed_pt_1, pt_2)
            new_b_2 = LineString(ed_pt_2, pt_1)

        n_p_1 = ops.nearest_points(boundary, pt_1)
        n_p_2 = ops.nearest_points(boundary, pt_2)

        bou_1 = []
        bou_2 = []
        for ind, coord in enumerate(boundary.coords):
            bou_1.append(coord)

            if coord == n_p_1:
                bou_1.append(pt_1)
                bou_2.append(pt_2)
                if ind+1 < len(boundary.coords):
                    bou_2.append(boundary.coords[ind+1:])
                return bou_1, bou_2, new_b_1, new_b_2

            if coord == n_p_2:
                bou_1.append(pt_2)
                bou_2.append(pt_1)
                if ind+1 < len(boundary.coords):
                    bou_2.append(boundary.coords[ind+1:])
                return bou_1, bou_2, new_b_1, new_b_2

    def show_img(self, X):
        plt.imshow(X, interpolation='nearest')
        plt.show()
        plt.close()

    def delete_layers(self, instance_list, correspondence_list, len_dict, layer_name, args):
        times = math.ceil(len_dict[layer_name] * args[1])
        for _ in range(times):
            if len(instance_list[layer_name]):
                ind = random.randrange(len(instance_list[layer_name]))
                instance_list[layer_name].pop(ind)
                correspondence_list[layer_name].pop(ind)

        return instance_list, correspondence_list

    def shift_layers(self, instance_list, correspondence_list, len_dict, layer_name, args,  patch_box):
        times = math.floor(len_dict[layer_name] * args[1])
        index_list = random.choices([i for i in range(len_dict[layer_name])], k=times)
        
        for ind in index_list:
            rx = random.uniform(-1*args[2], args[2])
            ry = math.sqrt(1 - rx**2)        
            geom = affinity.translate(
                instance_list[layer_name][ind], rx, ry)

            geom = self.valid_geom(geom, [0,0,patch_box[2],patch_box[3]], 0)
            
            if geom is None:
                rx *= -1
                ry *= -1     
                geom = affinity.translate(
                    instance_list[layer_name][ind], rx, ry)
                geom = self.valid_geom(geom, [0,0,patch_box[2],patch_box[3]], 0)
                
                if geom is None:
                    instance_list[layer_name].pop(ind)
                    correspondence_list[layer_name].pop(ind)
                    
                    continue
            
            if geom.geom_type == 'MultiLineString':
                instance_list[layer_name][ind]= ops.linemerge(geom)
            else:
                instance_list[layer_name][ind] = geom

        return instance_list, correspondence_list

    def add_layers(self, instance_list, correspondence_list, len_dict, layer_name, args, patch_box, patch_angle):
        times = math.ceil(len_dict[layer_name] * args[1])
        if times == 0:
            times = 1

        if layer_name == 'ped_crossing':
            patch_coords = self.patch_box_2_coords(patch_box)
            road_seg_records = self.map_explorer.map_api.get_records_in_patch(
                patch_coords, ['road_segment'])['road_segment']
            if len(road_seg_records):
                for _ in range(times):
                    new_geom_v = None
                    counter = 0
                    while new_geom_v is None:
                        counter += 1
                        if counter > 100:
                            print("this is going nowhere")
                            break
                        new_geom = self.creat_ped_polygon(
                            random.choice(road_seg_records))
                        new_geom_v = self.valid_geom(
                            new_geom, patch_box, patch_angle)
                        if new_geom_v is not None:
                            if new_geom_v.boundary.geom_type == 'MultiLineString':
                                instance_list[layer_name].append(
                                    ops.linemerge(new_geom_v.boundary))
                            else:
                                instance_list[layer_name].append(
                                    new_geom_v.boundary)

                            correspondence_list[layer_name].append(-1)

        if layer_name == 'boundary':
            for _ in range(times):
                ind = random.randint(0, len(instance_list[layer_name])-1)
                corr_ind = correspondence_list[layer_name][ind]
                bd1, bd2, nbd1, nbd2 = self.creat_boundray(
                    instance_list[layer_name][ind], patch_box)
                instance_list[layer_name].pop(ind)
                instance_list[layer_name].insert(ind, bd1)
                instance_list[layer_name].insert(ind+1, bd2)
                correspondence_list[layer_name].insert(ind+1, corr_ind)
                instance_list[layer_name].append(nbd1)
                correspondence_list[layer_name].append(-1)
                instance_list[layer_name].append(nbd2)
                correspondence_list[layer_name].append(-1)

        return instance_list, correspondence_list

    def valid_geom(self, geom, patch_box, patch_angle):
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        
        if geom.is_valid:
            new_geom = geom.intersection(patch)
            if not new_geom.is_empty:
                if new_geom.geom_type == 'MultiLineString':
                    inter_points = geom.intersection(patch.boundary)
                    ip_list = []
                    connect_lines = []
                    for p in inter_points.geoms:
                        ip_list.append(p)
                    connect_lines.append(LineString(ip_list))
                    for l in new_geom.geoms:
                        connect_lines.append(l)
                    # new_geom.append(connect_line)
                    new_multi_lines = MultiLineString(connect_lines)
                    new_geom = ops.linemerge(new_multi_lines)
                    
                if new_geom.geom_type == 'MultiLineString':
                    correct_coords = self.fix_corner(np.array(geom.coords), patch_box)
                    new_geom = LineString([[x[0], x[1]] for x in correct_coords])
                
                if new_geom.geom_type == 'MultiLineString':
                    return None
                
                new_geom = affinity.rotate(
                    new_geom, -patch_angle, origin=(patch_box[0], patch_box[1]))
                new_geom = affinity.affine_transform(new_geom,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_box[0], -patch_box[1]])

            else:
                return None
        else:
            print('geom is not valid.')
            return None

        return new_geom

    def difromate_map(self, map_ins_dict, def_args, patch_box):
        # Vertical distortion amplitude random maximum range int
        v = def_args[2][0]
        # Horizontal distortion amplitude random maximum range int
        h = def_args[2][1]
        i = def_args[2][2]  # Inclination amplitude [-Max_r Max_r] int

        xlim = [-patch_box[3]/2, patch_box[3]/2]
        ylim = [-patch_box[2]/2, patch_box[2]/2]

        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    map_ins_dict[key][ind] = self.con(
                        ins, xlim, ylim, v, h, i)

        return map_ins_dict

    def con(self, ins, xlim, ylim, randx, randy, randr):
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

    def guassian_warping(self, map_ins_dict, def_args, patch_box):
        g_xv, g_yv = self.gaussian_grid(patch_box, def_args)

        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    ins = self.fix_corner(ins, [0,0,patch_box[2], patch_box[3]])
                    map_ins_dict[key][ind] = self.warping(ins, g_xv, g_yv)

        return map_ins_dict

    def guassian_noise(self, map_ins_dict, def_args):
        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    g_nois = np.random.normal(
                        def_args[2][0], def_args[2][1], ins.shape)
                    map_ins_dict[key][ind] += g_nois

        return map_ins_dict

    def gaussian_grid(self, patch_box, def_args):
        nx, ny = int(patch_box[3]+1)+2, int(patch_box[2]+1)+2
        x = np.linspace(-int(patch_box[3]/2)-1, int(patch_box[3]/2)+1, nx)
        y = np.linspace(-int(patch_box[2]/2)-1, int(patch_box[2]/2)+1, ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        g_xv = xv + \
            np.random.normal(def_args[2][0], def_args[2][1], size=[nx, ny])
        g_yv = yv + \
            np.random.normal(def_args[2][0], def_args[2][1], size=[nx, ny])
        g_xv[:2, :] = xv[:2, :]
        g_xv[nx-2:, :] = xv[nx-2:, :]
        g_yv[:, :2] = yv[:, :2]
        g_yv[:, ny-2:] = yv[:, ny-2:]

        return g_xv, g_yv

    def fix_corner(self, ins, patch_box):
        x_min = patch_box[0] - patch_box[3] / 2
        x_max = patch_box[0] + patch_box[3] / 2
        y_min = patch_box[1] - patch_box[2] / 2
        y_max = patch_box[1] + patch_box[2] / 2
        xy_range = [[x_min, x_max], [y_min, y_max]]
        
        for dem in range(ins.shape[1]):
            ins_c = ins[:, dem]
            ins_c[ins_c < xy_range[dem][0]] = xy_range[dem][0]
            ins_c[ins_c > xy_range[dem][1]] = xy_range[dem][1]
            ins[:, dem] = ins_c

        return ins

    def warping(self, ins, xv, yv):
        new_point_list = []
        for point in ins:
            x = point[0]
            y = point[1]

            # canonical top left
            x_floor = math.floor(x)
            y_floor = math.floor(y)

            # Check upper or lower triangle
            x_res = x - x_floor
            y_res = y - y_floor
            upper = (x_res+y_res) <= 1.0

            # transfer x_floor coord[-15,15] to x_floor ind[0,32] fro grid
            x_floor += 16
            y_floor += 31

            if upper:
                # Get anchor
                x_anc = xv[x_floor, y_floor]
                y_anc = yv[x_floor, y_floor]

                # Get basis
                x_basis_x = xv[x_floor+1, y_floor] - x_anc
                x_basis_y = yv[x_floor+1, y_floor] - y_anc

                y_basis_x = xv[x_floor, y_floor+1] - x_anc
                y_basis_y = yv[x_floor, y_floor+1] - y_anc
            else:
                # Get anchor
                x_anc = xv[x_floor+1, y_floor+1]
                y_anc = yv[x_floor+1, y_floor+1]

                # Get basis
                x_basis_x = xv[x_floor, y_floor+1] - x_anc
                x_basis_y = yv[x_floor, y_floor+1] - y_anc

                y_basis_x = xv[x_floor+1, y_floor] - x_anc
                y_basis_y = yv[x_floor+1, y_floor] - y_anc
                x_res = 1-x_res
                y_res = 1-y_res

            # Get new coordinate in warped mesh
            x_warp = x_anc + x_basis_x * x_res + y_basis_x * y_res
            y_warp = y_anc + x_basis_y * x_res + y_basis_y * y_res

            new_point_list.append((x_warp, y_warp))

        return np.array(new_point_list)


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

    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation):
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

    def geom_to_np(self, map_ins_dict, inter_args=0, int_back=False):
        map_dict = {'divider': [], 'ped_crossing': [],
                    'boundary': []}

        for vec_class in map_ins_dict.keys():
            if len(map_ins_dict[vec_class]):
                for ind, instance in enumerate(map_ins_dict[vec_class]):
                    if not int_back:
                        if inter_args:
                            instance = self.interpolate(instance, inter_args)
                        else:
                            if instance.geom_type == 'MultiLineString':
                                instance = ops.linemerge(instance)
                            
                            if instance.geom_type == 'MultiLineString':
                                line_list = []
                                for ist in instance.geom:
                                    line_list.append(np.array(ist.coords))
                                instance = np.concatenate(line_list)
                                continue
                                
                            instance = instance.coords
                                
                    else:
                        shape = inter_args[vec_class][ind].shape
                        instance = self.interpolate(instance, shape[0])

                    map_dict[vec_class].append(np.array(instance))

        return map_dict

    def np_to_geom(self, map_ins_dict):
        map_dict = {'divider': [], 'ped_crossing': [],
                    'boundary': []}

        for vec_class in map_ins_dict.keys():
            if len(map_ins_dict[vec_class]):
                for instance in map_ins_dict[vec_class]:
                    instance = LineString([[x[0], x[1]] for x in instance])

                    map_dict[vec_class].append(instance)

        return map_dict

    def interpolate(self, instance, inter_args=0):
        distance = np.linspace(0, instance.length, inter_args)
        instance = [np.array(instance.interpolate(n).coords)[0]
                    for n in distance]

        return instance

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(
                    patch_box, patch_angle, layer_name)
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name)
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle)

            map_geom[layer_name] = geoms

        return map_geom

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
                    new_line, -patch_angle, origin=(patch_x, patch_y))
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_contour_line(self, patch_box, patch_angle, layer_name):
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
                                                      origin=(patch_x, patch_y))
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
                                                      origin=(patch_x, patch_y))
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
                                                  origin=(patch_x, patch_y))
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

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

    def get_org_info_dict(self, map_ins_dict):
        corr_dict = {'divider': [], 'ped_crossing': [], 'boundary': []}
        len_dict = {'divider': 0, 'ped_crossing': 0, 'boundary': 0}

        for vec_class in map_ins_dict.keys():
            if len(map_ins_dict[vec_class]):
                len_dict[vec_class] = len(map_ins_dict[vec_class])
                corr_dict[vec_class] = [
                    i for i in range(len(map_ins_dict[vec_class]))]

        return corr_dict, len_dict

    def get_trans_instance(self, map_ins_dict, trans_args, patch_box, patch_angle):

        corr_dict, len_dict = self.get_org_info_dict(map_ins_dict)

        if trans_args.del_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, len_dict, 'ped_crossing', trans_args.del_ped)

        if trans_args.shi_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, len_dict, 'ped_crossing', trans_args.shi_ped,  patch_box)

        if trans_args.add_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.add_layers(
                map_ins_dict, corr_dict, len_dict, 'ped_crossing', trans_args.add_ped,  patch_box, patch_angle)

        if trans_args.del_div[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, len_dict, 'divider', trans_args.del_div)

        if trans_args.shi_div[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, len_dict, 'divider', trans_args.shi_div,  patch_box)

        if trans_args.add_div[0]:
            pass  # TODO

        if trans_args.del_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.del_bou)

        if trans_args.shi_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.shi_bou,  patch_box)

        if trans_args.add_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.add_layers(
                map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.add_bou,  patch_box, patch_angle)

        if trans_args.aff_tra_pat[0] or trans_args.rot_pat[0] or trans_args.sca_pat[0] or trans_args.ske_pat[0] or trans_args.shi_pat[0]:
            map_ins_dict, corr_dict = self.map_trans.transfor_patch(
                map_ins_dict, corr_dict, patch_box, trans_args)

        return map_ins_dict, corr_dict


class PerturbParameters():
    def __init__(self,
                 # [switch, proportion, parameter]
                 # ped_crossing perturbation
                 del_ped=[0, 0, None],  # delete ped_crossing
                 # shift ped_crossing in its road_segment, shifted by offsets along each dimension[x, y]
                 shi_ped=[0, 0, [0, 0]],
                 add_ped=[0, 0, None],  # add ped_crossing in a road_segment
                 # dividers perturbation
                 del_div=[0, 0, None],  # delete divider
                 # shift divider, shifted by offsets along each dimension[x, y]
                 shi_div=[0, 0, [0, 0]],
                 add_div=[0, 0, None],  # add divider TODO
                 # boundray perturabtion
                 del_bou=[0, 0, None],  # delete lane
                 # shift lane, shifted by offsets along each dimension[x, y]
                 shi_bou=[0, 0, [0, 0]],
                 add_bou=[0, 0, None],  # add boundray TODO
                 # patch perturbation
                 aff_tra_pat=[0, None, [1, 0, 0, 1, 0, 0]],  # affine_transform
                 rot_pat=[0, None, [0, [0, 0]]],  # rotate the patch
                 sca_pat=[0, None, [1, 1]],  # scale the patch
                 ske_pat=[0, None, [0, 0, (0, 0)]],  # skew the patch
                 shi_pat=[0, None, [0, 0]],  # translate: shift the patch
                 # Horizontal, Vertical, and Inclination distortion amplitude
                 def_pat_tri=[0, None, [0, 0, 0]],
                 # gaussian mean and standard deviation
                 def_pat_gau=[0, None, [0, 1]],
                 # gaussian mean and standard deviation
                 noi_pat_gau=[0, None, [0, 1]],
                 # Interpolation
                 int_num=0,
                 int_ord='before',  # before the perturbation or after it
                 int_sav=False):  # save the interpolated instances

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
        self.def_pat_tri = def_pat_tri
        self.def_pat_gau = def_pat_gau
        self.noi_pat_gau = noi_pat_gau

        self.int_num = int_num
        self.int_ord = int_ord
        self.int_sav = int_sav

# Asynchronous execution utility from https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def perturb_map_seq(vector_map, trans_args, info, map_version, visual, trans_dic):
    trans_dic = copy.deepcopy(trans_dic)
    trans_ins, corr_dict = vector_map.get_trans_instance(
        trans_dic['map_ins_dict'], trans_args, trans_dic['patch_box'], trans_dic['patch_angle'])
    info[map_version+'_correspondence'] = corr_dict

    if trans_args.int_num and trans_args.int_ord == 'before':
        trans_np_dict = vector_map.geom_to_np(
            trans_ins, inter_args=trans_args.int_num)
    else:
        trans_np_dict = vector_map.geom_to_np(trans_ins)

    if trans_args.def_pat_tri[0]:
        trans_np_dict = vector_map.map_trans.difromate_map(
            trans_np_dict, trans_args.def_pat_tri, trans_dic['patch_box'])

    if trans_args.def_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_warping(
            trans_np_dict, trans_args.def_pat_gau, trans_dic['patch_box'])

    if trans_args.noi_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_noise(
            trans_np_dict, trans_args.noi_pat_gau)

    if (trans_args.int_num and trans_args.int_ord) == 'after' or (not trans_args.int_num and trans_args.int_sav):
        trans_np_dict = vector_map.np_to_geom(trans_np_dict)
        trans_np_dict = vector_map.geom_to_np(
            trans_np_dict, trans_args.int_num)
        visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)

    elif trans_args.int_num and not trans_args.int_sav:
        visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)
        trans_np_dict = vector_map.np_to_geom(trans_np_dict)
        trans_ins_np = vector_map.geom_to_np(trans_ins)
        trans_np_dict = vector_map.geom_to_np(
            trans_np_dict, trans_ins_np, int_back=True)

    else:
        # not trans_args.int_num and not trans_args.int_sav
        if visual.switch:
            trans_np_dict_4_vis = vector_map.np_to_geom(trans_np_dict)
            trans_np_dict_4_vis = vector_map.geom_to_np(
                trans_np_dict_4_vis, trans_args.int_num)
            visual.vis_contours(trans_np_dict_4_vis,
                                trans_dic['patch_box'], map_version)

    info[map_version] = trans_np_dict

    return info


# @background
def perturb_map(vector_map, trans_args, info, map_version, visual, trans_dic):
    trans_dic = copy.deepcopy(trans_dic)
    trans_ins, corr_dict = vector_map.get_trans_instance(
        trans_dic['map_ins_dict'], trans_args, trans_dic['patch_box'], trans_dic['patch_angle'])
    info[map_version+'_correspondence'] = corr_dict

    if trans_args.int_num and trans_args.int_ord == 'before':
        trans_np_dict = vector_map.geom_to_np(
            trans_ins, inter_args=trans_args.int_num)
    else:
        trans_np_dict = vector_map.geom_to_np(trans_ins)

    if trans_args.def_pat_tri[0]:
        trans_np_dict = vector_map.map_trans.difromate_map(
            trans_np_dict, trans_args.def_pat_tri, trans_dic['patch_box'])

    if trans_args.def_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_warping(
            trans_np_dict, trans_args.def_pat_gau, trans_dic['patch_box'])

    if trans_args.noi_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_noise(
            trans_np_dict, trans_args.noi_pat_gau)

    if (trans_args.int_num and trans_args.int_ord) == 'after' or (not trans_args.int_num and trans_args.int_sav):
        trans_np_dict = vector_map.np_to_geom(trans_np_dict)
        trans_np_dict = vector_map.geom_to_np(
            trans_np_dict, trans_args.int_num)
        visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)

    elif trans_args.int_num and not trans_args.int_sav:
        visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)
        trans_np_dict = vector_map.np_to_geom(trans_np_dict)
        trans_ins_np = vector_map.geom_to_np(trans_ins)
        trans_np_dict = vector_map.geom_to_np(
            trans_np_dict, trans_ins_np, int_back=True)

    else:  # not trans_args.int_num and not trans_args.int_sav
        if visual.switch:
            trans_np_dict_4_vis = vector_map.np_to_geom(trans_np_dict)
            trans_np_dict_4_vis = vector_map.geom_to_np(
                trans_np_dict_4_vis, 20)
            visual.vis_contours(trans_np_dict_4_vis,
                                trans_dic['patch_box'], map_version)

    info[map_version] = trans_np_dict

    return info


def obtain_perturb_vectormap(nusc_maps, map_explorer, info, point_cloud_range, sequential=False):
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

    if info['scene_token'] == '7626dde27d604ac28a0240bdd54eba7a':
        print('bug')
    
    # ------peturbation------
    # visualization setting
    vis_switch = True
    vis_path='/home/li/Documents/map/MapTRV2Local/tools/maptrv2/map_perturbation/visual/'
    visual = RenderMap(info, vector_map, vis_path, vis_switch)
    
    # the oranginal map
    trans_dic = vector_map.gen_vectorized_samples(
        lidar2global_translation, lidar2global_rotation)

    #visualization
    if vis_switch:
        trans_np_dict_4_vis = vector_map.geom_to_np(trans_dic['map_ins_dict'], 20)
        visual.vis_contours(trans_np_dict_4_vis,
                            trans_dic['patch_box'], 'annotation')

    info["annotation"] = vector_map.geom_to_np(trans_dic['map_ins_dict'])

    # the pertubated map
    # w/o loop
    # pertubation 1
    trans_args = PerturbParameters(del_ped=[1, 1, None],
                                del_div=[1, 1, None])
    info = perturb_map(vector_map, trans_args, info, 'annotation_1', visual, trans_dic)
    
    # pertubation 2
    trans_args = PerturbParameters(shi_ped=[1, 1, 1],
                                   shi_div=[1, 1, 1],
                                   shi_bou=[1, 1, 1])
    info = perturb_map(vector_map, trans_args, info, 'annotation_2', visual, trans_dic)
    
    # pertubation 3
    trans_args = PerturbParameters(del_ped=[1, 0.5, None],
                                add_ped=[1, 0.25, None],
                                del_div=[1, 0.5, None],
                                # trigonometric warping
                                def_pat_tri=[1, None, [1., 1., 3.]],
                                # Gaussian warping
                                def_pat_gau=[1, None, [0, 0.1]],
                                int_num=20)
    info = perturb_map(vector_map, trans_args, info, 'annotation_3', visual, trans_dic)

    # w loop
    # loop = asyncio.get_event_loop()                                              # Have a new event loop
    # looper = asyncio.gather(*[perturb_map(vector_map, trans_args, info, 'annotation_{}'.format(i), visual, trans_dic) for i in range(10)])         # Run the loop
    # results = loop.run_until_complete(looper)

    info['order'] = ['divider', 'ped_crossing', 'boundary']

    return info
