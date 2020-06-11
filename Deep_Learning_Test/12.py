cd ~/Downloads
mkdir jena_climate
cd jena_climate
wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv/zip
unzip jena_climate_2009_2016.csv

#观察耶拿天气数据集的数据
import os

data_dir='/users/fchollect/Downloads/jena_climate'
fname=os.path.join(data_dir,'jena_climate_2009_2016.csv')

f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')

