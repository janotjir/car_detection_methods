----------------------------------EVALUATION----------------------------------
- script: evaluate.py
- arguments: --normalization noscale/dynamic/static
	     --criteria acc/wheels/snow

Example - evaluate accuracy of PointNet model with no-scaling normalization on standard dataset:
	python3 evaluate.py --criteria acc --normalization noscale

Example - evaluate accuracy of PointNet model with static-scaling normalization on standard dataset:
	python3 evaluate.py --criteria acc --normalization static

Example - evaluate accuracy of PointNet model with no-scaling normalization on dataset collected in snowfall:
	python3 evaluate.py --criteria snow --normalization noscale

Example - evaluate accuracy of PointNet model with static-scaling normalization on dataset collected in snowfall:
	python3 evaluate.py --criteria snow --normalization static

Example - evaluate wheels localization ability of PointNet model with no-scaling normalization:
	python3 evaluate.py --criteria wheels --normalization noscale

Example - evaluate wheels localization ability of PointNet model with static-scaling normalization:
	python3 evaluate.py --criteria wheels --normalization static


- to visualize classifications, uncomment lines 176-184 in evaluate.py
- to visualize wheels localization, uncomment lines 395-407 in evaluate.py


----------------------------------TRAINING----------------------------------
1) generate training and testing dataset
- script: generate_data.py
- arguments: --type trn/tst

Example - generate training dataset:
	python3 generate_data.py --type trn

Example - generate testing dataset:
	python3 generate_data.py --type tst

2) run the training script
- script: train_segmentation.py
- arguments: --bs (int)
	     --nepoch (int)
	     --outf (str)
	     --model
	     --lr (float)
	     --feature_transform (action)
	     --optim adam/sgd/amsgrad
	     --momentum (float)
             --weight_decay (float)
	     --weight (float)
	     --numc (int)
	     --normalize noscale/dynamic/static
	     --gpu (int)

Example - run the training that led to the best model:
	python3 train_segmentation.py --bs 128 --lr 0.0001 --feature_transform --normalize noscale

-----------------------------------------------------------------------------
NOTE: the training was realized with Pytorch 1.5.0
If there are any issues, contact me: janotjir@fel.cvut.cz
