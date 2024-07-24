#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite YOLOX with OpenCV.

    Copyright (c) 2021 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import colorsys
import os
import random
import time

import cv2
import glob
import numpy as np
import tflite_runtime.interpreter as tflite
from yolox.utils.demo_utils import demo_postprocess, multiclass_nms

WINDOW_NAME = "YOLOX TensorFlow lite demo"

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def make_interpreter(
    model_file, num_of_threads, delegate_library=None, delegate_option=None
):
    """make tf-lite interpreter.
    Args:
        model_file: Model file path.
        num_of_threads: Num of threads.
        delegate_library: Delegate file path.
        delegate_option: Delegate option.
    Return:
        tf-lite interpreter.
    """
    if delegate_library is not None:
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(delegate_library, options=delegate_option)
            ],
        )
    else:
        return tflite.Interpreter(model_path=model_file, num_threads=num_of_threads)


def set_input_tensor(interpreter, image):
    """Sets the input tensor.
    Args:
        interpreter: Interpreter object.
        image: a function that takes a (width, height) tuple,
        and returns an RGB image resized to those dimensions.
    """
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]

    scale, zero_point = input_details['quantization']
    print("INPUTS:", scale, zero_point)
    if input_details['dtype'] == np.uint8:
        image = image / scale + zero_point
    image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
    print(image.shape)
    input_tensor[:, :] = image.copy()

def preprocess(img, input_size, mean, std):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.float32) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.float32) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    #padded_img = padded_img.astype(np.float32)
    padded_img /= 255
    return padded_img, r

def draw_caption(image, start, caption):
    cv2.putText(image, caption, start, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(
        image, caption, start, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def read_label_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def random_colors(N):
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(
        map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
    )
    random.shuffle(colors)
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of Tflite model.", required=True)
    parser.add_argument("--label", help="File path of label file.", required=True)
    parser.add_argument(
        "--threshold", help="threshold to filter results.", default=0.5, type=float
    )
    parser.add_argument("--thread", help="Num threads.", default=2, type=int)
    parser.add_argument("--imagepath", help="File path of Images", default='')
    parser.add_argument("--voc_valpath", help="File path of voc_val.txt", default=None)
    parser.add_argument("--txt_only", help="TXT-Save-Path if no Images will be shown", default=None)
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    args = parser.parse_args()

    # Get Images
    if args.voc_valpath:
        images = []
        with open(args.voc_valpath, 'r') as f:
            for line in f.readlines():
                imagepath = args.imagepath + '/' + line[:-1] + '.png' 
                images.append(imagepath)
    else:
        images = glob.glob(args.imagepath + '/*.jpg') + glob.glob(args.imagepath + '/*.JPG') + glob.glob(args.imagepath + '/*.png') + glob.glob(args.imagepath + '/*.bmp')
    

    # Initialize TF-Lite interpreter.
    interpreter = make_interpreter(args.model, args.thread)
    interpreter.allocate_tensors()
    _, height, width, channel = interpreter.get_input_details()[0]["shape"]
    input_shape = (height, width)
    print("Interpreter(height, width, channel): ", height, width, channel)

    # Read label and generate random colors.
    labels = read_label_file(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    random.seed(42)
    colors = random_colors(last_key)

    # Video capture.
    model_name = os.path.splitext(os.path.basename(args.model))[0]

    elapsed_list = []

    for image in images:
        frame = cv2.imread(image)

        # Image preprocess.
        im, ratio = preprocess(frame, input_shape, mean, std)
        # Run inference.
        start = time.perf_counter()

        set_input_tensor(interpreter, im)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details["index"])
        print("Output dtype Vorher:", output.dtype)
        output = output.astype(np.float32)
        print("Output dtype Nachher:", output.dtype)
        scale, zero_point = output_details['quantization']
        print("OUTPUTS:", scale, zero_point)
        if scale > 0:
            output = scale * (output - zero_point)

        inference_time = (time.perf_counter() - start) * 1000

        # Detection postprocess.
        predictions = demo_postprocess(output, input_shape, p6=args.with_p6)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.1)

        detections = []

        # Display result.
        if dets is not None:
            for i in dets:
                print(i)
            final_boxes = dets[:, :4]
            final_scores = dets[:, 4]
            final_cls_inds = dets[:, 5]

            for i, box in enumerate(final_boxes):
                class_id = int(final_cls_inds[i])
                score = final_scores[i]
                if score < args.threshold:
                    continue
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                caption = "{0}({1:.2f})".format(labels[class_id], score)

                # Draw a rectangle and caption.
                cv2.rectangle(
                    frame, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=3
                )
                draw_caption(frame, (xmin, ymin), caption)

                detections.append([labels[class_id], final_scores[i], xmin, ymin, xmax, ymax])

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
        draw_caption(frame, (10, 30), display_text)

        # All the results have been drawn on the image, now display the image
        if args.txt_only is None: # "text_only" controls whether we want to display the image results or just save them in .txt files
            while True:
               cv2.imshow(WINDOW_NAME, frame)
               if cv2.waitKey(1) & 0xFF == ord("q"):
                   cv2.destroyAllWindows()

               if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) <1:
                   break
            cv2.destroyAllWindows()

        # Save detection results in .txt files (for calculating mAP)
        else:
            # Get filenames and paths
            image_fn = os.path.basename(image)
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn +'.txt'
            txt_savepath = os.path.join(args.txt_only, txt_result_fn)

            # Write results to text file
            # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            with open(txt_savepath,'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

if __name__ == "__main__":
    main()
