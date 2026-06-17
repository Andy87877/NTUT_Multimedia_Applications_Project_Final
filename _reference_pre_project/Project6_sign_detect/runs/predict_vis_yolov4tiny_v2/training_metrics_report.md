# YOLOv4-tiny V2 Model Training & Test Set Metrics Report

## 1. Dataset Partition Details
* **Training Set (80%)**: 121 images (Augmented with HSV, Blur, Translation, Horizontal Flip)
* **Validation Set (10%)**: 15 images
* **Test Set (10%)**: 15 images (Completely isolated test set)

## 2. Test Set Quantitative Evaluation Results
Below is the evaluation report of the best weights model (`yolov4-tiny-custom_best.weights`) evaluated on the isolated Test Set:

| Class Name   | GT   | TP   | FP   | FN   | Precision | Recall | F1-Score |
| :---         | :---: | :---: | :---: | :---: | :---:     | :---:  | :---:    |
| stop         | 7    | 7    | 0    | 0    | 1.0000    | 1.0000 | 1.0000   |
| rail         | 6    | 6    | 2    | 0    | 0.7500    | 1.0000 | 0.8571   |
| pedestrian   | 10   | 9    | 3    | 1    | 0.7500    | 0.9000 | 0.8182   |
| blocked      | 11   | 11   | 3    | 0    | 0.7857    | 1.0000 | 0.8800   |
| **Average/mAP**| **34** | **-** | **-** | **-** | **0.8214** | **0.9750** | **0.8888** |

## 3. Training Progress Loss Trace (Representative Epochs)
The complete learning history of all 1000 epochs has been saved to [training_history.csv](file:///C:/Users/andy8/Desktop/NTUT_Media/Project6/runs/predict_vis_yolov4tiny_v2/training_history.csv) (opens in Excel).

| Epoch | Train Loss | Box Loss (CIoU) | Conf Loss | Class Loss | Val F1-Score | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Epoch 1    | 11.9264 | 6.4371 | 4.0796 | 1.4097 | 0.0020 | 0.000060 |
| Epoch 10   | 5.4495 | 3.6735 | 0.9232 | 0.8527 | 0.0230 | 0.000505 |
| Epoch 50   | 2.0161 | 1.8154 | 0.1100 | 0.0907 | 0.3990 | 0.000998 |
| Epoch 100  | 1.3671 | 1.2846 | 0.0568 | 0.0257 | 0.6726 | 0.000984 |
| Epoch 200  | 1.0665 | 1.0092 | 0.0451 | 0.0122 | 0.7298 | 0.000923 |
| Epoch 500  | 0.6096 | 0.5887 | 0.0187 | 0.0022 | 0.8787 | 0.000536 |
| Epoch 1000 | 0.1974 | 0.1822 | 0.0146 | 0.0006 | 0.9330 | 0.000011 |
