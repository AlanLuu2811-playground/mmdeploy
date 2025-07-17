from mmdeploy_runtime import Detector
import cv2

img = cv2.imread('./demo/resources/det.jpg')

# create a detector
detector = Detector(model_path='./work_dir/mmdet/rtmdet',
                    device_name='cuda', device_id=0)
# perform inference
bboxes, labels, masks = detector(img)

# visualize inference result
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
    [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
    if score < 0.3:
        continue

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imshow('Detection Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

