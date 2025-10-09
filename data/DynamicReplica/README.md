# Prepare DynamicReplica Dataset

DynamicReplica consists of 145200 stereo frames (524 videos) with humans and animals in motion.

Dataset can be downloaded at the following website: https://github.com/facebookresearch/dynamic_stereo

The directory structure should be:
```text
DynamicReplica
└───disparity
|   └───train
|   └───valid
└───real
└───test
└───train
└───valid
    
```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{karaev2023dynamicstereo,
  title={DynamicStereo: Consistent Dynamic Depth from Stereo Videos},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={CVPR},
  year={2023}
}
```



{
    "real": [
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/real/real_000.zip"
    ],
    "valid": [
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_000.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_001.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_002.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_003.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_004.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_005.zip"
    ],
    "test": [
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_000.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_001.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_002.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_003.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_004.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_005.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_006.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_007.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_008.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_009.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_010.zip"
    ],
    "train": [
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_000.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_001.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_002.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_003.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_004.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_005.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_006.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_007.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_008.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_009.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_010.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_011.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_012.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_013.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_014.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_015.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_016.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_017.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_018.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_019.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_020.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_021.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_022.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_023.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_024.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_025.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_026.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_027.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_028.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_029.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_030.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_031.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_032.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_033.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_034.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_035.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_036.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_037.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_038.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_039.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_040.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_041.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_042.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_043.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_044.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_045.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_046.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_047.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_048.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_049.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_050.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_051.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_052.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_053.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_054.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_055.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_056.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_057.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_058.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_059.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_060.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_061.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_062.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_063.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_064.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_065.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_066.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_067.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_068.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_069.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_070.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_071.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_072.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_073.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_074.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_075.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_076.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_077.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_078.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_079.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_080.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_081.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_082.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_083.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_084.zip",
      "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_085.zip"
    ]
  }