# PFCGAN

##
WORK IN PROGRESS
##

## Preprocessing

execute `python preprocess.py` to preprocess the dataset.


```
pip install dlib==19.24.6

python preprocess.py --mode continue --model shape_predictor_68_face_landmarks.dat --folder "path/to/data" --progress_file checkpoint.txt --output_folder "path/to/output"
```

## Generating flists

execute `python flist.py` to generate flists for the dataset.

### Flist for original images

```
python flist.py --folder 'path/to/original/images' --out celeba_og_train.flist 
```

### Flist for processed images
```
python flist.py --folder 'path/to/processed/images' --out celeba_processed.flist 
```

# Running the model

## Create a environment with conda

```

conda create -n pfcgan python=3.9
conda activate pfcgan

```

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11" matplotlib numpy==1.26.4 ipython
```\
