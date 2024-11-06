import hydra
import torch
import easyocr
import cv2
import os
import time
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# Directory to save the text file for detected plate numbers
output_dir = 'plate_texts'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'detected_plate_numbers.txt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available

# Set to track unique detections and their timestamps
detected_plates = {}
last_detection_time = time.time()


def getOCR(im, coors):
    # Expand bounding box slightly for better OCR accuracy
    x, y, w, h = int(coors[0]) - 5, int(coors[1]) - 5, int(coors[2]) + 5, int(coors[3]) + 5
    im = im[y:h, x:w]

    # Convert to grayscale and apply adaptive thresholding to improve character clarity
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform OCR on the processed image
    results = reader.readtext(binary, detail=0, paragraph=True)
    ocr_text = " ".join([res for res in results if res]) if results else ""
    return ocr_text.strip()


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255  # Normalize to 0-1
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        global last_detection_time
        current_time = time.time()

        # Check if 0.5 seconds have passed since the last detection
        if current_time - last_detection_time < 0.5:
            return ""  # Return empty string instead of None

        last_detection_time = current_time
        p, im, im0 = batch
        log_string = ""
        
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        self.all_outputs.append(det)
        
        if len(det) == 0:
            return log_string  # Return the current log_string instead of None

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
                ocr_text = getOCR(im0, xyxy)
                if ocr_text:
                    if ocr_text not in detected_plates or conf > detected_plates[ocr_text]:
                        detected_plates[ocr_text] = conf
                        with open(output_file, 'w') as text_file:
                            for plate in detected_plates:
                                text_file.write(plate + '\n')
                        label = ocr_text

                self.annotator.box_label(xyxy, label, color=colors(c, True))
        
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy, imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)
        
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
