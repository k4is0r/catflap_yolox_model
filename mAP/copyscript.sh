#/bin/bash

rm -rf /data/catflap/mAP/input/ground-truth/*

for file in $(cat /data/catflap/catflap_pictures/yolox/voc_main/val.txt)
do
	xmlfile=`echo /data/catflap/catflap_pictures/yolox/voc_annotations/$file.xml`
	cp $xmlfile /data/catflap/mAP/input/ground-truth/
done

python /data/catflap/mAP/scripts/extra/convert_gt_xml.py
rm -rf /data/catflap/mAP/outputs
cd /data/catflap/mAP
python /data/catflap/mAP/calculate_map_cartucho.py --labels=/data/catflap/mAP/catlabel.txt

