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
In the function obtain_perturb_vectormap(), by setting the perturbation version name and perturbation parameters, you can create multiple annotation versions to add to info.
```python
    map_version = 'annotation_1'
    trans_args = PerturbParameters(del_ped=[1, 1],  # delet a ped_crossing
                                   del_lan=[1, 1])  # delete a lane
    info = perturb_map(vector_map, lidar2global_translation,
                       lidar2global_rotation, trans_args, info, map_version, visual)
```
The default parameters in the trans_args() class include all perturbation types, and value is a list containing two elements [Boolean value: whether to perform this perturbation, variable type: additional perturbation parameters]

# Visualization
In the perturbation.py file, before the perturbation parameter setting block.

- switch: whether to visualize
- show: whether to display
- save: save path. If not saving side set to None.

```python
save_path = os.path.join(
    '/home/li/Documents/map/MapTR_local/tools/maptrv2/map_perturbation/visual', info['scene_token'], info['token'])
visual = RenderMap(info, vector_map.nusc_map, vector_map.map_explorer,
                    switch=True, show=False, save=save_path)
```
