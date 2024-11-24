# PFCGAN

## Preprocessing

execute `python preprocess.py` to preprocess the dataset.

```
preprocess.py --mode continue -model shape_predictor_68_face_landmarks.dat --folder 'path/to/dataset' --progress_file checkpoint.txt
```

## Generating flists

execute `python flist.py` to generate flists for the dataset.

### Flist for original images

```
flist.py --folder 'path/to/original/images' --out celeba_og_train.flist 
```

### Flist for processed images
```
flist.py --folder 'path/to/processed/images' --out celeba_processed.flist 
```

Update your settings.yml file with the paths to the original and processed images.