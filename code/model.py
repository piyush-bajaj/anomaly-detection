"""
Exemplary predictive model.

You must provide at least 2 methods:
- __init__: Initialization of the class instance
- predict: Uses the model to perform predictions.

"""

import os
import json
import glob
import random
from training import train
from PIL import Image
from yolov5 import YOLOv5
from utils import general #for training to override imread method
import torch

class Model:
    def __init__(self):
        """
        Initialize the class instance

        Important: If you want to refer to relative paths, e.g., './subdir', use
        os.path.join(os.path.dirname(__file__), 'subdir')
        """

    def prediction(self, data_set_directory):
        """
        This function should provide predictions of labels on a data set.
        """

        input_files = sorted(glob.glob(os.path.join(os.path.abspath(data_set_directory), '*.jpeg')))

        yolo = YOLOv5(
            model_path=os.path.join(os.path.dirname(__file__), 'weights/best.pt')
        )
        torch.cuda.empty_cache()

        results = yolo.predict(input_files, size=256, augment=False)
        predictions = results.pred
        final_output = []

        for i, prediction in enumerate(predictions):
            output = {
                'file': input_files[i].split('/')[-1],
                'missing_pills': 0,
                'present_pills': 0,
                'coordinates': {
                    'missing': [],
                    'present': []
                }
            }

            for pred in prediction:
                center_x, center_y = (pred[0] + pred[2]) / 2 , (257 - ((pred[1] + pred[3]) / 2))
                if pred[5] == 0.0 :
                    output['missing_pills'] += 1
                    output['coordinates']['missing'].append((center_x.item(), center_y.item()))
                else :
                    output['present_pills'] += 1
                    output['coordinates']['present'].append((center_x.item(), center_y.item()))

            final_output.append(output)

        # return list of dictionaries whose length matches the number of jpeg files
        return final_output

    def training(self, data_set_directory):
        """
        This method is used for training the model
        """
        input_files = glob.glob(os.path.join(os.path.abspath(data_set_directory), '*.jpeg'))
        txt_files = glob.glob(os.path.join(os.path.abspath(data_set_directory), '*.txt'))

        for file in txt_files:
            try:
                os.remove(file)
            except OSError:
                pass

        for input_file in input_files :
            self.convert_file(input_file)

        # Percentage of images to be used for the validation set
        percentage_test = 10

        training_folder = os.path.join(os.path.abspath(data_set_directory), '../../training')
        try:
            os.system('rm -rf ' + training_folder)
        except OSError:
            print('Training folder not present')

        try:
            os.mkdir(training_folder)
        except OSError:
            print('Unable to create training folder')

        try:
            os.mkdir(training_folder + '/data')
            os.mkdir(training_folder + '/data/images')
            os.mkdir(training_folder + '/data/labels')
            os.mkdir(training_folder + '/data/images/train')
            os.mkdir(training_folder + '/data/images/valid')
            os.mkdir(training_folder + '/data/labels/train')
            os.mkdir(training_folder + '/data/labels/valid')
        except OSError:
            print("error while creating directories")

        # Populate the folders
        p = percentage_test/100
        for pathAndFilename in glob.iglob(os.path.join(data_set_directory, "*.jpeg")):
            title = os.path.splitext(os.path.basename(pathAndFilename))[0]
            if random.random() <= p :
                os.system(f"cp {data_set_directory}/{title}.jpeg " + training_folder + "/data/images/valid")
                os.system(f"cp {data_set_directory}/{title}.txt " + training_folder + "/data/labels/valid")
            else:
                os.system(f"cp {data_set_directory}/{title}.jpeg " + training_folder + "/data/images/train")
                os.system(f"cp {data_set_directory}/{title}.txt " + training_folder + "/data/labels/train")


        train.run(
            weights=os.path.join(os.path.dirname(__file__), 'weights/yolov5l.pt'),
            cfg=os.path.join(os.path.dirname(__file__), 'training/yolov5l.yaml'),
            data=os.path.join(os.path.dirname(__file__), 'training/dataset.yaml'),
            hyp=os.path.join(os.path.dirname(__file__), 'training/hyp.scratch-low.yaml'),
            epochs=5,
            imgsz=256,
            batch_size=8,
            device=0 #for cuda device
        )

    #generate txt files for all image files using .json file
    #refer example.json for input file example
    def convert_file(self, input_file):
        """
        This function is used to convert files to the model format along with normalization of center coordinates
        """
        width, height = Image.open(input_file).size
        label_filename = input_file.replace('.jpeg', '.json')

        if os.path.exists(label_filename):
            with open(label_filename, 'r') as file:
                data = json.loads(file.read())
                txt_content = list()
                for centers in data['coordinates']['missing']:
                    x,y = centers
                    x = (x) / width
                    y =(height-y) / height
                    w,h = (35 / width), (40 / height)
                    txt_content.append(f'0 {x} {y} {w} {h}')
                for centers in data['coordinates']['present']:
                    x,y = centers
                    x = (x) / width
                    y = (height-y) / height
                    w,h = (35 / width) , (40 / height)
                    txt_content.append(f'1 {x} {y} {w} {h}')

                label_filename = label_filename.replace('.json','.txt')
                with open(f'{label_filename}', 'w', encoding = 'utf-8') as f:
                    f.writelines('\n'.join(txt_content))
