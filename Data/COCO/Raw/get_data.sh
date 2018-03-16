wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d ./
mv annotations/captions*.json .
rm -r annotations*
