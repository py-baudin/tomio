# iomed
IO utilities for medical images


## 3d volumes

Wrapper over `SimpleITK`.

```python
from iomed import volume

# load image volume (formats: .nii/.mha/.mhd/.hdr)
vol = volume.read('vol.nii.gz')

# write volume
volume.write('vol2.mha', vol)


# vol is a numpy ndarray
isinstance(vol, np.ndarray) # -> True

# volume header
vol.spacing # -> tuple[float]
vol.origin # -> tuple[float]
vol.transform # -> tuple[tuple[float]]
vol.metadata # -> dict[str, str]

# other attributes
vol.header # -> dict of spacing, origin, transform and metadata
vol.A # -> np.ndarray
vol.affine # -> affine transform
vol.anatomical_orientation # ex. RAI ("from" convention)


# methods

# get real world coordinates from pixel indices
vol.get_coords(indices)

# get pixel indices from real world coordinates
vol.get_indices(coords)


# utilities

# create volume, copy header
vol2 = volume.asvolume(array, ref=vol)
# equivalent to: vol2 = volume.Volume(array, **vol.header) 


# create volume, fill foreground values
vol2 = volume.tovolume(values, mask=mask) 

```


## Config files

```python
from iomed import configfile

# read yaml / json files
cfg = configfile.read('config.yml') # -> dict

# write 
configfile.write('config.yml', cfg)

```

## ITK-Snap label files

```python
from iomed import labelfile

# read ITK-Snap label file
labels = labelfile.read('labels.txt')

# write label file
labelfile.write('labels.txt', labels)

labels.indices # -> list of label indices
labels.description # -> list of label descriptions
labels.colors # -> list of RGBA colors

# methods

# get label description
labels[idx] 
labels.get(idx)

# set label description
labels[idx] = descr


```




