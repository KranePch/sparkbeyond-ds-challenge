# This is an example of developing a script locally with the West Nile Virus data to share on Kaggle
# Once you have a script you're ready to share, paste your code into a new script at:
#	https://www.kaggle.com/c/predict-west-nile-virus/scripts/new

# Code is borrowed from this script: https://www.kaggle.com/users/213536/vasco/predict-west-nile-virus/west-nile-heatmap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mapdata = np.loadtxt("../data/mapdata_copyright_openstreetmap_contributors.txt")
spray = pd.read_csv('../data/spray.csv')[['Longitude', 'Latitude']]
train = pd.read_csv('../data/train.csv')[['Longitude', 'Latitude', 'WnvPresent']]
train = train[train['WnvPresent'] == 1]

aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)

plt.figure(figsize=(10,14))
plt.imshow(mapdata, 
           cmap=plt.get_cmap('gray'), 
           extent=lon_lat_box, 
           aspect=aspect)

spray_loc = spray[['Longitude', 'Latitude']].drop_duplicates().values
wnv_loc = train[['Longitude', 'Latitude']].values
plt.scatter(spray_loc[:,0], spray_loc[:,1], marker='.')
plt.scatter(wnv_loc[:,0], wnv_loc[:,1], marker='x')
plt.savefig('../map_visualization/spray_wnv.png')
