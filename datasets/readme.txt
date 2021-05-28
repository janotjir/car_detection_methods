COPY IN THIS DIRECTORY DATASETS DOWNLOADED FROM: https://www.kaggle.com/jijanota/car-detection-2d-lidar-data
-----------------------------------------------------------------------------

trn_data
- augmented training dataset
- contains fields 'data' and 'label'
- size: (7968, 1526, 3)
- first point of each frame is position of the robot
- frames contain many (0, 0, 0) points with zero coordinates - cause of need of fixed size (1526 points), those need to be discarded before use
-----------------------------------------------------------------------------

tst_data
- augmented testing dataset collected in standard weather conditions
- contains fields 'data' and 'label'
- size: (2403, 1526, 3)
- first point of each frame is position of the robot
- frames contain many (0, 0, 0) points with zero coordinates - cause of need of fixed size (1526 points), those need to be discarded before use
-----------------------------------------------------------------------------

tst_data_snow
- augmented testing dataset collected in snowfall
- contains fields 'data' and 'label'
- size: (682, 1526, 3)
- first point of each frame is position of the robot
- frames contain many (0, 0, 0) points with zero coordinates - cause of need of fixed size (1526 points), those need to be discarded before use
-----------------------------------------------------------------------------

tst_data_wheels
- non-augmented rosbag for wheel localization evaluation
- contains fields 'data' and 'label'
- size: (112, 1526, 3)
- frames contain many (0, 0, 0) points with zero coordinates - cause of need of fixed size (1526 points), those need to be discarded before use
-----------------------------------------------------------------------------

- for example of use look in one of evaluate.py files
