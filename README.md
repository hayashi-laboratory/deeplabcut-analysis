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
py -3 -m venv .venv
```

- Activate the enviroment
- - for windows
```
source ./.venv/Scripts/activate
python -m pip install -U -r requirements.txt
python -m pip install .   
python -m pip install ./deeplabcut-analysis/.
```

- - for Linux/MacOs
```
source ./.venv/bin/activate
python3.9 -m pip install -U -r requirements.txt
python3.9 -m pip install .
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
