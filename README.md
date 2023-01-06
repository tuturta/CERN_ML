*This project was done in the framework of a semester project at CERN in the first year of the Master. More details on the background and results are available in [this paper](Docs/paper.pdf).*

# Structure


    ├── create_sample.py        # Script used to generate the model on SCITAS supercomputer  
    ├── train_model.py          # Script used to train the model on SCITAS supercomputer.
    ├── sample_methods.py       # All the methods related to the creation of the training dataset 
    ├── NN_methods.py           # All the methods related to the training of the GNN models
    ├── results_analysis.ipynb  # Analysis of predictions made by GNN models
    ├── **Docs**                    
    │   ├── requirement.txt     # Packages required to run the project
    │   └── paper.pdf           # Report of this project
    │ 
    └── **Data**
        ├── Analysis_of_results # Folder with all the dataframe generated during the analysis part in .pkl format
        └── CERN_DAs            # Folder with all the DAs data in .npz format

# Installation

> To install and run the project, you first need to download all the libraries from [requirements.txt](Docs/requirements.txt). Then you need to download the real DAs dataset to this link and put it in the folder `Data/`. The training and test sets are too heavy to be uploaded on a drive. They have to be generated from scratch with the provided script `create_sample.py`.




