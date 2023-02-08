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
Activate .venv in command prompt
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


<br> 
<br>

`pyDLCbehavior` package provides `NovelObjectRecognitionAnalysis` and `YMazeAnalysis` class objects, which can analyze the NOR test and Y-maze test.  
These class objects have the result dataframe obtained from every experimental file, so the results can be summarized in Jupyter lab or Jupyter notebook using the following ipynb files.

<br>  

 

<br>
<br>

# novel_object_analysis.ipynb
Run this code in Jupyter.  


- [x]  **dataset requirements.**
 1. csv files exported from DLC.
 2. pickle files exported from DLC.
 3. avi files you recorded in your experiment.  
 
 
- [x] **Set the directory path `HOMEDIR` that stores above datasets.**  

- [x] **Set the ROI to determine the coordinate of the center of each object**  
  e.g.)  
  zone1: right upper object  
  zone2: left lower object  
  
<br>

<br>
<br>

# YMaze_analysis.ipynb
Run this code in Jupyter.

- [x]  **dataset requirements.**
 1. csv files exported from DLC.
 2. pickle files exported from DLC.  
 
 
 
- [x] **Set the directory path `HOMEDIR` that stores above datasets.**
  
 <br>
<br>
