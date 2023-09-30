# How to run
The main running file is: custom_nusc_map_converter_copy.py. It is obtained with very few modifications to the custom_nusc_map_converter.py file.

To run it, you need to download the NuScenes V1.0-mini and can_bus data. And modify the following parameters according to local conditions:
```python
parser.add_argument(
    '--root-path',
    type=str,
    default='/home/li/Documents/map/data/sets/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='/home/li/Documents/map/data/sets/nuscenes',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/home/li/Documents/map/MapTR_local/tools/maptrv2/map_perturbation/output',
    required=False,
    help='name of info pkl')
```

The output will be two .pkl files: nuscenes_map_infos_temporal_train.pkl and nuscenes_map_infos_temporal_val.pkl

more parameters may be needed in future studies:
'--version' the version of nuscenes data. If you want to use other versions of NuScenes data, commit the version = 'v1.0-mini' block and uncommit the upper two blocks: train_version and test_version. And set the version to '--version'


# How to modify perturbation parameters
Modify it in perturbation.py file.

## Modify the number of perturbed annotation versions
In the function obtain_perturb_vectormap(), by adding and modifying trans_args(), you can create multiple annotation versions to add to info.
```python
trans_args = {'add_ped': [1, None],  # add a ped_crossing in a road_segment
                'del_ped': [1, None],  # delet a ped_crossing
                # shift a ped_crossing in its road_segment #TODO
                'shi_ped': [1, None],
                'del_roa_div': [1, None],  # delete a road_divider
                'del_lan_div': [1, None],  # delete a lane_divier
                # [shift the map, the ratio of the length that the shift range does not exceed ]
                'shi_map': [1, 0.3],
                # [rotate the map, the degree of the rotation range]
                'rot_map': [1, 54]
                }
map_anns_tran = vector_map.gen_trans_vectorized_samples(
    lidar2global_translation, lidar2global_rotation, trans_args)
info["annotation_1"] = map_anns_tran

```
The key in the trans_args() dictionary is the modification type, and the value is a list containing two elements [Boolean value: whether to perform this perturbation, variable type: additional perturbation parameters]
