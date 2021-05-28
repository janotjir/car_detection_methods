----------------------------------EVALUATION----------------------------------
- script: evaluate.py
- arguments: --criteria acc/wheels/snow

Example - evaluate accuracy of SVM model on testing dataset in standard weather:
	python3 -W ignore evaluate.py --criteria acc

Example - evaluate accuracy of SVM model on testing dataset in snowfall:
	python3 -W ignore evaluate.py --criteria snow

Example - evaluate wheels localization ability of SVM model:
	python3 -W ignore evaluate.py --criteria wheels


- to visualize classifications, uncomment lines 178-184 in evaluate.py
- to visualize wheels localization, uncomment lines 386-398 in evaluate.py


----------------------------------TRAINING----------------------------------
run the training script

Example - run the training that led to the best model:
	python3 train.py

