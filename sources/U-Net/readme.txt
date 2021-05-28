----------------------------------EVALUATION----------------------------------
- script: evaluate.py
- arguments: --criteria acc/wheels/snow

Example - evaluate accuracy of U-Net model on dataset with standard weather:
	python3 evaluate.py --criteria acc

Example - evaluate accuracy of U-Net model on dataset collected in snowfall:
    python3 evaluate.py --criteria snow

Example - evaluate wheels localization ability of U-Net model:
	python3 evaluate.py --criteria wheels


- to visualize preprocessed input, uncomment lines 156-159 in evaluate.py
- to visualize classifications, uncomment lines 195-198 in evaluate.py
- to visualize wheels localization, uncomment lines 413-425 in evaluate.py


----------------------------------TRAINING----------------------------------
1) generate training and testing dataset
- script: generate_data.py
- arguments: --type trn/tst

Example - generate training dataset:
	python3 generate_data.py --type trn

Example - generate testing dataset:
	python3 generate_data.py --type tst

2) run the training script
- script: train.py
- arguments: --bs (int)
	     --nepoch (int)
	     --outf (str)
	     --model
	     --lr (float)
	     --optim adam/sgd/amsgrad
	     --momentum (float)
             --weight_decay (float)
	     --weight (float)
	     --gpu (int)

Example - run the training that led to the best model:
	python3 train.py --bs 32

