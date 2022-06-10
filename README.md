This codebase is highly based on https://github.com/ultralytics/yolov5 and https://pypi.org/project/yolov5/

The objective of this code is to find the location and count of missing and present pills in a pill blister.
The code takes the help of Deep Neural Networks for training and then giving out the predictions.
Due to privacy reasons, the data-set of images cannot be provided. However, the code of training and prediction is provided. This can be used and then generalized for different applications.
Similar code was used for a competition (R & S Engineering Competition 2022) where the acheived results can be seen in the below table, where the images provided were microwave images.

+----------------------------------------------------------------------------------------------------+
| Average sample accuracy |	Anomaly detection accuracy | Localization accuracy | Prediction time (s) |
| 96.67                   |	97.63                      |	96.00                |	21.25              |
+----------------------------------------------------------------------------------------------------+

The certificate achieved as part of the above mentioned competition is present in the repository as pdf.

The YOLOv5 model version 5l is used in this project to train and detect the objects.
Refer to image.jpeg to see the kind of input (conversion to microwave not present) and example.json to see the label format provided for training.
The initial weights used for training are provided in weights folder.