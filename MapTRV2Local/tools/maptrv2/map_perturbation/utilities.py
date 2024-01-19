import math
import secrets
import warnings
from shapely.geometry import Polygon, MultiPolygon
from shapely import ops


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def get_length(geom):
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError("Input must be a Shapely Polygon or MultiPolygon")

    if isinstance(geom, MultiPolygon):
        geom = ops.unary_union(geom)
    
    # Get the bounding box coordinates
    min_rect = geom.minimum_rotated_rectangle
    # rect_coords = list(min_rect.exterior.coords)
    rect_coords = list(min_rect.exterior.coords)

    # Calculate length and width
    length_1 = calculate_distance(rect_coords[0], rect_coords[1])
    length_2 = calculate_distance(rect_coords[1], rect_coords[2])
    
    if length_1 < length_2:
        return [rect_coords[1], rect_coords[2], rect_coords[3], rect_coords[0]]

    return rect_coords[:4]


class record_generator:
    def __init__(self, pertu_nusc_infos):
        self.pertu_nusc_infos = pertu_nusc_infos
    
    def token_generator(self, token_type=[4,2,2,2,8]):
        """generate a token with a type:
        Ex. '3e8ea889-540c-2113-ae03-f5873a7b6c1ea070'"""
        
        token = '-'.join([secrets.token_hex(i) for i in token_type])
        
        return token
    
    def node_record_generator(self, coord):
        """{
        "token": "8260ea97-fd4e-4702-b5e8-f1f4465f286f",
        "x": 772.859616346946,
        "y": 1867.5701729191703
        },
        """
        
        record = {}
        token = self.token_generator()
        record["token"] = token
        record["x"] = float(coord[0])
        record["x"] = float(coord[1])
        
        self.pertu_nusc_infos["node"].append(record)
        
        return record
    
    def line_record_generator(self, geom):
        """    {
        "token": "97b57a20-185c-4b0f-b405-cc34bf35b1a9",
        "node_tokens": [
        "39878431-98e2-4070-a512-d2c76dcaac4c",
        "79093cc8-e5c4-458d-8d9d-96ce83fdc4af",
        "17a1446a-7dc0-45c5-9354-515455193e5b",
        "85a3cb8b-df2c-456a-a5c8-88e3ec25b4c2",
        "ac669ce0-ced1-4989-bc24-78647f17f30a",
        "5ee9e3a6-f9ea-485c-a916-b6422102467a",
        "12f6f131-97c4-40b4-a85b-4141cc74bf1d",
        "a7278b89-eddb-4f26-89dc-f1a97dec05f8",
        "db854d3d-27e2-470f-9de7-0fa383534558",
        "468e45e8-9793-46b4-a604-6ee8433531dd",
        "9f475eba-75e8-4e9f-b1a1-ac1e1b821ffa",
        "139e0ba6-d96a-4cf9-839d-df4211b291e3",
        "8e91fba4-8573-424c-8c62-baa8acaf7d0f"
          ]
        },
        """
        
        record = {}
        
        token = self.token_generator()
        record["token"] = token
        
        record["node_tokens"] = []
        for c in geom.coords:
            n_record = self.node_record_generator(c)
            record["node_tokens"].append(n_record["token"])
            
        self.pertu_nusc_infos["line"].append(record)
            
        return record
            
    
    def polygon_record_generator(self, geom):
        """{
        "token": "02eaba43-235f-4b77-99ad-5d44591e315d",
        "exterior_node_tokens": [
            "95375fac-e47e-48e0-9bca-72ccd86efa89",
            "c1b0dd41-8131-4a53-b644-08a9f43343f6",
            "d2f3f652-1179-40fe-923e-61b824d600fe",
            "177dd72d-768e-4fde-a5dd-e950a6c47643"
        ],
        "holes": []
        }
        """
        record = {}
        
        token = self.token_generator(token_type=[4,2,2,2,8])
        record["token"] = token
        
        exterior_nodes = geom.exterior.coords
        record["exterior_node_tokens"] = []
        for n in exterior_nodes:
            n_record = self.node_record_generator(n)
            record["exterior_node_tokens"].append(n_record["token"])
        
        holes = geom.interiors.coords
        record["holes"] = []
        for h in holes:
            hole = {"node_tokens":[]}
            for n in h:
                n_record = self.node_record_generator(n)
                hole["node_tokens"].append(n_record["token"])
            
            record["holes"].append(hole)
            
        self.pertu_nusc_infos["polygon"].append(record)
        
        return record
        
    def layer_record_generator(self, layer_name, geom):
        record = {}
        
        if geom.geom_type == 'LineString':
            token = self.token_generator()
            record["token"] = token
            
            line_record = self.line_record_generator(geom)
            record["line_token"] = line_record["token"]
            
        elif geom.geom_type == 'Polygon':
            token = self.token_generator()
            record["token"] = token
            
            polygon_record = self.polygon_record_generator(geom)
            record["polygon_token"] = polygon_record["token"]
        else:
            warnings.warn("Warning...........geom type is neither LineString nor Polygon")
            return None
        
        self.pertu_nusc_infos[layer_name].append(record)
        
        return record

def vector_to_map_json(info_dic):
    r_gen = record_generator(info_dic["pertu_nusc_infos"])
    
    ins_dic = info_dic["map_ins_dict"]
    for layer_name in ['boundary', 'ped_crossing', 'divider']:
        for geom in ins_dic[layer_name]:
            r_gen.layer_record_generator(layer_name, geom)
            
    return r_gen.pertu_nusc_infos

class delet_record:
    def __init__(self, map_explorerm,
                 pertu_nusc_infos) -> None:
        self.map_explorerm = map_explorerm
        self.pertu_nusc_infos = pertu_nusc_infos
    
    def delete_node_record(self, token) -> None:
        n_ind = self.map_explorerm.map_api.getind("node", token)
        self.pertu_nusc_infos["node"][n_ind] = None
    
    def delete_line_record(self, token) -> None:
        polygon_record = self.map_explorerm.map_api.get('line', token)
        
        for n in polygon_record["node_tokens"]:
            self.delete_node_record(n)
            
        l_ind = self.map_explorerm.map_api.getind("line", token)
        self.pertu_nusc_infos["line"][l_ind] = None
    
    def delete_polygon_record(self, token) -> None:
        polygon_record = self.map_explorerm.map_api.get('polygon', token)
        
        for n in polygon_record["exterior_node_tokens"]:
            self.delete_node_record(n)
        
        for l in polygon_record["holes"]:
            self.delete_node_record(l)
                
        p_ind = self.map_explorerm.map_api.getind("polygon", token)
        self.pertu_nusc_infos["polygon"][p_ind] = None
    
    def delete_layer_record(self, layer_name, token) -> None:
        layer_record = self.map_explorerm.map_api.get(layer_name, token)
        
        if layer_name in ["road_segment", "lane", "ped_crossing"]:
            self.delete_polygon_record(layer_record["polygon_token"])
        elif layer_name in ["road_divider", "lane_divider"]:
            self.delete_line_record(layer_record["line_token"])
        else:
            warnings.warn("Warning...........layer is not ture")
            
        layer_ind = self.map_explorerm.map_api.getind(layer_name, token)
        self.pertu_nusc_infos[layer_name][layer_ind] = None
    