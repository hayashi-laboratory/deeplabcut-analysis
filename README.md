# A package for analyzing the data from DeepLabCut


# how to use this script 
- Install python 3.9 using miniconda only for install python and pip
```
conda create -n -y your_env_name python=3.9 
conda activate your_env_name
```

- create a venv in the directory your want 
- It will create a .venv folder in your enviornment
 > https://code.visualstudio.com/docs/python/python-tutorial 
```
conda activate your_env_name
python3 -m venv .venv
```

- Deactivate the conda environment  
- Then activate the python venv enviroment  

**For windows user**  
<br>
Activate .venv
```
(.venv) C:Users\yourname\....\directory_you_want>  ./.venv/Scripts/activate
```
Clone this repository using git bash  
```
git clone https://github.com/lycantrope/deeplabcut-analysis.git && cd deeplabcut-analysis
```
Install all necessary packages 
```
python -m pip install .
```

**For Linux/MacOs user**
```
source ./.venv/bin/activate
git clone https://github.com/lycantrope/deeplabcut-analysis.git && cd deeplabcut-analysis
python -m pip install .   
```




## Add a kernel to jupyter lab 
```
pip install ipykernel
python -m ipykernel install --user --name=your_env_name
```

## Add other kernel to current kernel (Both environment need ipykernel installed)
```
/path/to/kernel/env/bin/python -m ipykernel install --prefix=/path/to/jupyter/env --name 'python-my-env'
```


---  

# YMaze_analysis.ipynb
Run this code in Jupyter.

- [x]  **dataset requirements.**
 1. csv files exported from DLC.
 2. pickle files exported from DLC.  
 
 
 
- [x] **Set the directory path `HOMEDIR` which stores above datasets.**
  
 

