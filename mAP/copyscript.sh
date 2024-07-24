#/bin/bash

rm /data/catflap/mAP/input/detection-results/*
rm /data/catflap/mAP/input/ground-truth/*
rm /data/catflap/mAP/input/images-optional/*

for file in $(cat /data/catflap/catflap_pictures/yolox/voc_main/val.txt)
do
	xmlfile=`echo /data/catflap/catflap_pictures/yolox/voc_annotations/$file.xml`
	cp $xmlfile /data/catflap/mAP/input/ground-truth/
done

python /data/catflap/mAP/scripts/extra/convert_gt_xml.py
rmdir /data/catflap/mAP/outputs
python /data/catflap/mAP/calculate_map_cartucho.py --labels=/data/catflap/mAP/catlabel.txt

