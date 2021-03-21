# YOLO+MorphGAC

Pancreas segmentation by first detecting it (using YOLO or any other pancreas detection network) and then applying morphological snakes (MorphGAC) on the predicted bounding boxes to segment it.

This repo is only for the segmentation part, the pancreas detection part is available on repository “Pancreas-detection-YOLOv4”.

The whole idea is visualized on the following graph: 

![MorphGAC_algorithm_graph](https://user-images.githubusercontent.com/30274421/111902831-fa34af80-8a47-11eb-8722-5a580b53b7ea.png)
