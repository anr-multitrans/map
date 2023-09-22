import copy
import random
import matplotlib.pyplot as plt

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.nuscenes import NuScenes

from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString, Polygon, Point


    
def get_ped_crossing_line(patch_box, patch_angle):
    patch_x = patch_box[0]
    patch_y = patch_box[1]

    patch = map_explorer.get_patch_coord(patch_box, patch_angle)
    polygon_list = []
    records = getattr(map_explorer.map_api, 'ped_crossing')
    # records = getattr(self.nusc_maps[location], 'ped_crossing')
    for record in records:
        polygon = map_explorer.map_api.extract_polygon(record['polygon_token'])
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

# polygon_list = get_ped_crossing_line(patch_box, patch_angle)



class MapTransform:
    def __init__(self, map_explorer:NuScenesMapExplorer):
        self.map_explorer = map_explorer
        self.layer_names = ['ped_crossing', 'road_segment', 'road_block']   # 'road_divider' are lines
        self.patch_angle =0
        
    
    def patch_coords_2_box(self, coords):
        
        box = ((coords[0] + coords[2])/2, (coords[1] + coords[3])/2, coords[3] - coords[1], coords[2] - coords[0])
        
        return box
        
        
    def transfor_layer(self, patch_coords, layer, token, rotate = 0, scale = [1,1], skew = [0,0], shift = [0,0]):
        
        # Convert patch_box to shapely Polygon coordinates
        patch_box = self.patch_coords_2_box(patch_coords)
        patch = self.map_explorer.get_patch_coord(patch_box, self.patch_angle)
    
        # Get neede non geometric layers records intersects a particular rectangular patch
        records = self.map_explorer.map_api.get_records_in_patch(patch_coords, [layer])
    
        polygon_list = []
        for lay_token in records[layer]:
            lay_record = self.map_explorer.map_api.get(layer, lay_token)
            polygon = self.map_explorer.map_api.extract_polygon(lay_record['polygon_token'])
            # plt.plot(*polygon.exterior.xy)
            if polygon.is_valid:
                if lay_token == token:
                    if rotate:
                        polygon = affinity.rotate(polygon, rotate)
                    if scale != [1,1]:
                        polygon = affinity.scale(polygon, scale[0], scale[1])
                    if skew != [0,0]:
                        polygon = affinity.skew(polygon, skew[0], skew[1])
                    if shift != [0,0]:
                        polygon = affinity.translate(polygon, shift[0], shift[1])

                    # plt.plot(*polygon.exterior.xy)

                    # Each pedestrian crossing record has to be on a road segment.
                    if layer == 'ped_crossing':
                        roa_seg_record = self.map_explorer.map_api.get('road_segment', lay_record['road_segment_token'])
                        road_seg_polygon = self.map_explorer.map_api.extract_polygon(roa_seg_record['polygon_token'])
                        polygon = polygon.intersection(road_seg_polygon)
                
                    polygon = polygon.intersection(patch)
                
                if polygon.is_valid:
                    if polygon.geom_type == 'Polygon':
                        polygon = MultiPolygon([polygon])
                    polygon_list.append(polygon)
                else:
                    print('The transformed layer leaves the patch range and is considered deleted.')
        
        layer_names = copy.copy(self.layer_names)
        layer_names.remove(layer)      
        patch_geo_list = self.map_explorer.map_api.get_map_geom(patch_box, self.patch_angle, layer_names)
                
        polygon_list = patch_geo_list + [(layer, polygon_list)]
                
        return polygon_list
    
    
    def add_ped_crossing_layer(self, patch_coords, road_segment_token = None): 
        
        # Convert patch_box to shapely Polygon coordinates
        patch_box = self.patch_coords_2_box(patch_coords)
        patch = self.map_explorer.get_patch_coord(patch_box, self.patch_angle)
    
        # Get all non geometric layers records intersects a particular rectangular patch. Or records that need to research 
        records = self.map_explorer.map_api.get_records_in_patch(patch_coords, self.layer_names)

        min_x, min_y, max_x, max_y = self.map_explorer.map_api.get_bounds('road_segment', road_segment_token)
        
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        if max([x_range, y_range]) <= 4:
            new_polygon = self.map_explorer.map_api.extract_polygon(self.map_explorer.map_api.get('road_segment', road_segment_token)['polygon_token'])
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
                
            new_polygon = Polygon([left_top, left_bottom, right_bottom, right_top])
                       
        new_polygon = new_polygon.intersection(patch)
        # plt.plot(*new_polygon.exterior.xy)

        if new_polygon.geom_type == 'Polygon':
            new_polygon = MultiPolygon([new_polygon])
                        
        polygon_list = self.map_explorer.map_api.get_map_geom(patch_box, self.patch_angle, self.layer_names)
        
        for ind, lay_geo in enumerate(polygon_list):
            if lay_geo[0] == 'ped_crossing':
                polygon_list[ind][1].append(new_polygon)
                return polygon_list
        
        polygon_list += ['ped_crossing', [new_polygon]]
                
        return polygon_list


    def delte_layers(self, patch_coords, layers = ['ped_crossing'], token = None):
        patch_box = self.patch_coords_2_box(patch_coords)
        polygon_list = []
        
        if token is None:
            layer_names = self.map_explorer.map_api.non_geometric_layers - layers
            polygon_list = self.map_explorer.map_api.get_map_geom(patch_box, 0, ['ped_crossing'])
            
        else:
            # Convert patch_box to shapely Polygon coordinates
            patch = self.map_explorer.get_patch_coord(patch_box, self.patch_angle)
        
            # Get all non geometric layers records intersects a particular rectangular patch. Or records that need to research 
            records = self.map_explorer.map_api.get_records_in_patch(patch_coords, layers)
        
            for lay_token in records[layers[0]]:
                if lay_token == token:
                    continue
                
                lay_record = self.map_explorer.map_api.get(layers[0], lay_token)
                polygon = self.map_explorer.map_api.extract_polygon(lay_record['polygon_token'])
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
                    
        layer_names = copy.copy(self.layer_names)  
        layer_names.remove(layers[0])    
        patch_geo_list = self.map_explorer.map_api.get_map_geom(patch_box, self.patch_angle, layer_names)
                
        polygon_list = patch_geo_list + [(layers[0], polygon_list)]
                
        return polygon_list
        


if __name__ == '__main__':

    ##- setup
    local_dataroot = '/home/li/Documents/NuScenes/data/sets/nuscenes'
    # local_dataroot = '/../../../../data/sets/nuscenes'
    data_version = 'v1.0-mini'
    map_ocation = 'singapore-onenorth'    # `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown`, `boston-seaport`

    #- init NuScenes
    # nusc = NuScenes(version='v1.0-mini', dataroot=local_dataroot, verbose=False)

    #- init NuScenes Map 
    # nusc_map = NuScenesMap(dataroot=local_dataroot, map_name=map_ocation)
    # bitmap_one = BitMap(local_dataroot, map_ocation, 'basemap')  # for render
    map_explorer = NuScenesMapExplorer(NuScenesMap(dataroot=local_dataroot, map_name=map_ocation))

    map_trans = MapTransform(map_explorer)
    # print(map_trans.layer_names)
    
    # take a patch
    # patch_box = [300, 1700, 100, 100] # x_center, y_center, height, width
    patch_coords = (250, 1650, 350, 1750) # x_min, y_min, x_max, y_max
    # patch_angle = 0
    # layer_names = ['ped_crossing', 'road_segment', 'road_block', 'road_divider']

    ped_cro_records = map_explorer.map_api.get_records_in_patch(patch_coords, ['ped_crossing']) # intersect
    
    
    # shift a ped_crossing
    ped_cro_token = ped_cro_records['ped_crossing'][0]
    
    new_polygon_list_shi = map_trans.transfor_layer(patch_coords, 'ped_crossing', ped_cro_token, shift = [4, 0])
    
    # delet a ped_crossing
    new_polygon_list_del = map_trans.delte_layers(patch_coords, ['ped_crossing'], ped_cro_token)
    
    # add a ped_crossing
    roa_seg_records = map_explorer.map_api.get_records_in_patch(patch_coords, ['road_segment'])
    roa_seg_token = roa_seg_records['road_segment'][0]
    
    new_polygon_list_add = map_trans.add_ped_crossing_layer(patch_coords, roa_seg_token)
    
    print('FINISH')
    
    # patch_coords = [patch_box[0]-patch_box[2]/2, patch_box[1]-patch_box[3]/2, patch_box[0]+patch_box[2]/2, patch_box[1]+patch_box[3]/2]

    # patch = map_explorer.map_api.get_map_geom(patch_box, patch_angle, ['ped_crossing']) # 4 ped_crossing

    # records = map_explorer.map_api.get_records_in_patch(patch_coords)
    # roa_seg_records = map_explorer.map_api.get_records_in_patch(patch_coords, ['road_segment'])
    # dri_are_records = map_explorer.map_api.get_records_in_patch(patch_coords, ['drivable_area'])

    # ped_cro_bou_sample = map_explorer.map_api.get_bounds('ped_crossing', ped_cro_records['ped_crossing'][0])
    # ped_cro_rec_sample = map_explorer.map_api.get('ped_crossing', ped_cro_records['ped_crossing'][0])
    # rod_seg_rec_sample = map_explorer.map_api.get('road_segment', ped_cro_rec_sample['road_segment_token'])
    # dri_are_rec_sample = map_explorer.map_api.get('drivable_area', ped_cro_rec_sample['road_segment_token'])
    # pol_rec_sample = map_explorer.map_api.get('polygon', ped_cro_rec_sample['polygon_token'])
    # nod_rec_sample = map_explorer.map_api.get('node', ped_cro_rec_sample['exterior_node_tokens'][0])
    # ped_cro_pol_sample = map_explorer.map_api.extract_polygon(ped_cro_rec_sample['polygon_token'])

    # map_explorer.map_api.render_map_patch(patch_coords, ['ped_crossing'], bitmap=bitmap_one)
    # map_explorer.map_api.render_map_patch(patch_coords, ['road_segment'], bitmap=bitmap_one)
    # map_explorer.map_api.render_map_patch(patch_coords, ['drivable_area'], bitmap=bitmap_one)
    # plt.show()

    # polygon = map_explorer.map_api.get_bounds('ped_crossing', records['ped_crossing'][3])
    
    
    