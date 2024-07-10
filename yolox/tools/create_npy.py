import os
import glob, cv2
import numpy as np

array=[]
file_paths = glob.glob("/data/catflap/yolox/datasets/VOCdevkit/VOC2022/PNGImages/*.png")
end = 1500
#input_shapes = [1,3,320,320]
input_size = [1, 320, 320, 3]
#end = len(file_paths)

for i, file_path in enumerate(file_paths[:end]):
    if i/100  == int(i/100):
        print(i,"/",end)
    img = cv2.imread(file_path)
    #for input_size in input_shapes:
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[2], 3), dtype=np.float32) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    r = min(input_size[1] / img.shape[0], input_size[2] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    padded_img /= 255
    array.append (padded_img)
    #yield [np.expand_dims(padded_img, axis=0)]

print('array shape:', np.array(array).shape)
np.save("YOLOX_outputs/images.npy", array)

