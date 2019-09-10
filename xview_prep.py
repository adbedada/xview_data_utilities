import aug_util as aug
import wv_util as wv
import numpy as np
import csv
from PIL import Image
import os
import numpy as np
import json

chip_dir = "/Users/eddiebedada/datasets/xview/"

output_path = './output/'
bboxes_text = 'bboxes.txt'

#Load an image
img_dir = os.path.join(chip_dir, 'planes')

src = os.path.join(chip_dir, 'planes/80.tif')
json_file = os.path.join(chip_dir,'xView_train.geojson')


def image_name(src):
    """
    :param src: path
    :return: file name
    """
    return src.split('/')[-1]


def get_labels_for_chip(src, json_file):
    """
    Get the bbox and class for a chip
    :param src: path to chip image
    :param json_file: path to json_file with labels
    :return: bbox coords and classes
    """
    coords, chips, classes = wv.get_labels(json_file)
    chip_name = image_name(src)
    coords = coords[chips==chip_name]
    classes = classes[chips==chip_name].astype(np.int64)

    return coords, classes


def get_names_for_classes(txt_file):
    """
    :param txt_file: path to class_labels file
    :return: dict of classes with labels
    """
    labels = {}

    with open(txt_file) as f:

        for row in csv.reader(f):
            labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

        return labels


def chip_image(src, json_file):

    img = Image.open(src)
    arr = np.array(img)
    coords, classes = get_labels_for_chip(src, json_file)
    c_img, c_box, c_cls = wv.chip_image(img=arr, coords=coords, classes=classes, shape=(256, 256))

    return c_img, c_box, c_cls


def chip_and_save_image(src, json_file, output_path, file_extension='.png', prefered_label=13):

    c_img, c_box, c_cls = chip_image(src, json_file)

    selected_labels = {}
    selected_chips = []
    selected_bboxes = {}

    # extract labels
    for cls in c_cls:
        for value in c_cls[cls]:
            if value == prefered_label:
                selected_labels[cls] = c_cls[cls]

    chip_name = image_name(src)
    base_name = chip_name.split('.')[0]


    # extract bbox

    for label in selected_labels:
        for box_id in c_box:
            if box_id == label:
                selected_bboxes[box_id] = c_box[box_id]

    # save bboxes
    labels_file_name = "{}{}_{}".format(output_path, base_name, bboxes_text)

    w = csv.writer(open(labels_file_name, "w"))
    for key, val in selected_bboxes.items():
        w.writerow([key, val.astype(np.uint8)])

    # save chips
    for i, array in enumerate(c_img):
        for label in selected_labels:
            if label == i:
                expand_img = np.expand_dims(array, axis=0)
                selected_chips.append(expand_img)
                output_filename_n = "{}{}_{}{}".format(output_path, base_name, i, file_extension)
                save_image = Image.fromarray(array)
                save_image.save(output_filename_n, "JPEG")

    print('done!')
    #return selected_labels



def bulk_chip_process(img_dir, json_file):

    for r, d, f in os.walk(img_dir):
        for name in f:
            chip_name = name.split('.')[0]
#            crds, cls = get_labels_for_chip(src, json_file)
            #print(src)
            src = os.path.join(img_dir, name)
            #arr = np.array(img)

            #c_img, c_box, c_cls = chip_image(src, json_file)
            #print(c_cls)
            print(chip_and_save_image(src, json_file, output_path))


bulk_chip_process(img_dir, json_file)

############ test #######

#print(get_names_for_classes('xview_class_labels.txt'))
#crds, cls = get_labels_for_chip(src, json_file)

#chip_image(src, json_file)

#print(chip_and_save_image(src, json_file, output_path))