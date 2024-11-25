# PFCGAN

##
WORK IN PROGRESS
##

## Preprocessing

execute `python preprocess.py` to preprocess the dataset.


```
pip install dlib==19.24.6

python preprocess.py --mode continue --model shape_predictor_68_face_landmarks.dat --folder 'path/to/dataset' --progress_file checkpoint.txt
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

Update your settings.yml file with the paths to the original and processed images.
