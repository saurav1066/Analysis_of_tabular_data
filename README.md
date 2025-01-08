# About the project and how to run it

- Data Folder contains data for both datasets in respective folders

- docs folder contains the thesis document

- models contains the offloaded mistral model

## contents of notebooks Folder:

- The requirements are all mentioned in the requirements.txt
- There are two different experimentations in notebook folder: main one is for the Heart Disease Dataset and the one inside the breast cancer folder is for the breast cancer dataset
- For both datasets the files are same.
- Initially preprocessing.ipynb has all the preprocessing for each dataset.
- traditional.ipynb contains the training and prediction from traditional models
- llm.ipynb contains training and prediction form the BERT
- mistral.ipynb contains the training and prediction from MISTRAL

Apart from these which are same for both datasets t here are other files in the notebook folder and all the notebooks contain experiments performed on the heart disesase dataset.

- smote.ipynb contains the implementation of smote on heart disease dataset and the impact on predictions for traditional and llm
- simulation.ipynb contains experimentation with multiple sample sizes for training for both all aforementioned models
- stacking.ipynb implements stacking and training a meta models


results folder contaions all generated predictions









