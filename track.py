import torch as T
import os
import threading
import queue
from typing import Dict, List, Optional, Tuple, Callable
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

class VideoAnalyticsPipeline:
    def __init__(self, model_path: str, conf_thresh: float = 0.5, miss_thresh: int = 10) -> None:
        self.model: YOLO = YOLO(model_path)
        self.device: str = "cuda" if T.cuda.is_available() else "mps" if \
            hasattr(T.backends, "mps") and T.backends.mps.is_available() else "cpu"
        self.model.to(self.device).half().eval()
        self.model.fuse() # fuse Conv + Batch Norm + LeakyReLU for YOLO
        self.conf_thresh: float = conf_thresh
        self.tracker: DeepSort = DeepSort(max_age = 20, n_init = 3, nn_budget = 100, \
            max_iou_distance = 0.7, max_cosine_distance = 0.2, embedder_gpu = True)
        self.active_tracks: Dict[int, Tuple[float, float, float, float]] = {}
        self.missing_objects: Dict[int, int] = {}
        self.miss_thresh: int = miss_thresh
        self.fps: float = 0.0

    # detect & track objects in single frame, return annotated frame + events
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        init_frame: float = time.time()
        events: List[str] = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with T.no_grad(), T.autocast(device_type = self.device):
            results = self.model.predict(rgb, conf = self.conf_thresh, \
                stream = True, device = self.device)
        detections: List[Tuple[float, float, float, float, float, int]] = []
        # vectorized box extraction
        for res in results:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()  
            clses = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, clses):
                detections.append((float(x1), float(y1), float(x2), \
                    float(y2), float(conf), int(cls)))
                
        deep_sort_fmt: List[Tuple[List[float], float, int]] = []
        for x1, y1, x2, y2, conf, _cls in detections:
            w = x2 - x1; h = y2 - y1
            deep_sort_fmt.append(([x1, y1, w, h], conf, _cls))
        tracks = self.tracker.update_tracks(deep_sort_fmt, frame = rgb)
        current_tracks: Dict[int, Tuple[float, float, float, float]] = {}
        for track in tracks:
            if not track.is_confirmed(): continue
            tid = track.track_id
            # cls = getattr(track, "detection_class", -1)
            cls = track.det_class
            class_name = self.model.names[cls] 
            x, y, w, h = track.to_tlwh()
            current_tracks[tid] = (x, y, x + w, y + h) 
            if tid not in self.active_tracks: events.append(f"New object: {class_name} ID {tid}")
            p1 = (int(x), int(y)); p2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            annotation_text = f"{class_name} ID {tid}"
            cv2.putText(frame, annotation_text, (p1[0], p1[1] - 10), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # check for missing objects
        for old_id in list(self.active_tracks.keys()):
            if old_id not in current_tracks:
                self.missing_objects[old_id] = self.missing_objects.get(old_id, 0) + 1
                if self.missing_objects[old_id] >= self.miss_thresh:
                    events.append(f"Object missing: ID {old_id}")
                    del self.active_tracks[old_id]
                    del self.missing_objects[old_id]
            else: self.missing_objects[old_id] = 0
        self.active_tracks = current_tracks
        # 8) Compute FPS & annotate
        post_frame = time.time(); self.fps = 1.0 / (post_frame - init_frame)
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, events

    def run(self, source: str, out_dir: Optional[str] = "results") -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open source {source}")
        ret, frame = cap.read()
        if not ret: raise RuntimeError("Failed to read from video source.")
        # 2) Optional writer
        writer: Optional[cv2.VideoWriter] = None
        os.makedirs(out_dir, exist_ok = True)
        output_path = os.path.join(out_dir, \
            f"ann-{os.path.splitext(os.path.basename(source))[0]}.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                out_frame, events = self.process_frame(frame)
                for e in events: print(e)
                if writer: writer.write(out_frame)
        finally:
            cap.release()
            if writer: writer.release()

class AsyncVideoPipeline:
    def __init__(self, source: str, \
        frame_processor: Callable[[np.ndarray], Tuple[np.ndarray, List[str]]], \
            out_dir: Optional[str] = "results", queue_size: int = 10) -> None:
        self.source = source
        self.process_frame = frame_processor
        self.out_dir = out_dir
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open video source {source}")
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize = queue_size)
        self.result_queue: \
            "queue.Queue[Tuple[np.ndarray,List[str]]]" = queue.Queue(maxsize = queue_size)
        self.writer: Optional[cv2.VideoWriter] = None
        os.makedirs(out_dir, exist_ok = True)
        output_path = os.path.join(out_dir, \
            f"ann-{os.path.splitext(os.path.basename(source))[0]}.mp4")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
        self.stopped = False

    def capture_thread(self) -> None:
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret: self.stopped = True; break
                self.frame_queue.put(frame)
            else: time.sleep(0.005)

    def process_thread(self) -> None:
        while not self.stopped or not self.frame_queue.empty():
            try: frame = self.frame_queue.get(timeout = 0.1)
            except queue.Empty: continue
            out_frame, events = self.process_frame(frame)
            self.result_queue.put((out_frame, events))

    def write_thread(self) -> None:
        while not self.stopped or not self.result_queue.empty():
            try: frame, events = self.result_queue.get(timeout=0.1)
            except queue.Empty: continue
            if self.writer: self.writer.write(frame)
            for e in events: print(e)

    def run(self) -> None:
        threads = [
            threading.Thread(target = self.capture_thread, daemon = True),
            threading.Thread(target = self.process_thread, daemon = True),
            threading.Thread(target = self.write_thread, daemon = True),
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        self.cap.release()
        if self.writer: self.writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Video Analytics Pipeline")
    parser.add_argument("--model", type = str, default = "yolov8n.pt", \
        help = "Path to YOLO model weights")
    parser.add_argument("--source", type = str, required = True, help = "Path to video source")
    parser.add_argument("--output", type = str, default = "results", \
        help = "Output directory or file path")
    parser.add_argument("--conf", type = float, default = 0.5, \
        help = "Confidence threshold for detection")
    parser.add_argument("--miss", type = int, default = 10, \
        help = "Missing frames threshold for tracking")
    parser.add_argument("--async-mode", action = "store_true", required = False, \
        help="Enable asynchronous capture/process/write")
    args = parser.parse_args()
    print(args)
    pipeline = VideoAnalyticsPipeline( model_path = args.model, \
        conf_thresh = args.conf, miss_thresh = args.miss)
    if args.async_mode:
        async_pipe = AsyncVideoPipeline(source = args.source, \
            frame_processor = pipeline.process_frame, out_dir = args.output)
        async_pipe.run()
    else: pipeline.run(source = args.source, out_dir = args.output)
