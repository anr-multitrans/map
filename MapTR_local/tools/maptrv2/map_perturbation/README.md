# How to run
The main running file is: custom_nusc_map_converter_copy.py. It is obtained with very few modifications to the custom_nusc_map_converter.py file.

To run it, you need to download the NuScenes V1.0-mini and can_bus data. And modify the following parameters according to local conditions:
'--root-path' this is the root path of nuscenes dataset
'--canbus' this is the root path of can_bus data
'--out-dir' this is the path for the output .pkl files

The output will be two .pkl files: nuscenes_map_infos_temporal_train.pkl and nuscenes_map_infos_temporal_val.pkl

more parameters may needs in the futur study:
'--version' the version of nuscenes data. If you want to use other versions of NuScenes data, commit the version = 'v1.0-mini' block and uncommit the upper two block: train_version and test_version. And set the version in '--version'


# How to modify perturbation parameters
Modify it in perturbation.py file.

## Modifiy the number of perturbed annotaion versions
In function obtain_perturb_vectormap(), by adding and modifying trans_args(), you can create multiple annotation versions to add to info.
```python
    trans_args = {'add_ped': [1, None],  # add a ped_crossing in a road_segment
                  'del_ped': [1, None],  # delet a ped_crossing
                  # shift a ped_crossing in its road_segment #TODO
                  'shi_ped': [1, None],
                  'del_roa_div': [1, None],  # delete a road_divider
                  'del_lan_div': [1, None],  # delete a lane_divier
                  # [shift the map, the ration of the length that the shift range does not exceed ]
                  'shi_map': [1, 0.3],
                  # [rotate the map, the degree of the rotate range]
                  'rot_map': [1, 54]
                  }
```
The key in the trans_args() dictionary is the modification type, and the value is a list containing two elements [Boolean value: whether to perform this perturbation, variable type: additional perturbation parameters]
