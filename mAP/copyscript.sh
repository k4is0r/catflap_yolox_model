#/bin/bash

rm input/detection-results/*
rm input/ground-truth/*
rm input/images-optional/*

for file in $(cat /data/catflap/catflap_pictures/yolox/voc_main/val.txt)
do
	xmlfile=`echo /data/catflap/catflap_pictures/yolox/voc_annotations/$file.xml`
	cp $xmlfile /data/catflap/mAP/input/ground-truth/
done

python scripts/extra/convert_gt_xml.py
rmdir outputs
python calculate_map_cartucho.py --labels=catlabel.txt

