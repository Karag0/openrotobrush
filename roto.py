import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import FastSAM
import threading
import os

class OpenRotoBrushProject:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenRotoBrush Project v1.2")
        
        # Инициализация YOLO
        self.model = YOLO('yolo11n-seg.pt')
        self.selected_object_id = None
        self.current_mask = None
        self.show_mask = False
        self.mask_history = []
        self.masks = {}
        
        # Настройка видео
        self.cap = None
        self.video_path = ""
        self.current_frame = 0
        self.total_frames = 0
        self.playing = False
        self.delay = 25
        self.original_size = (1920, 1080)
        self.fps = 30
        self.video_loaded = False
        
        # GUI элементы
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.main_frame, bg='black', cursor="arrow")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical")
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        
        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Панель управления
        self.btn_load = ttk.Button(control_frame, text="Загрузить видео", command=self.load_video)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = ttk.Button(control_frame, text="▶", width=3, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.slider = ttk.Scale(control_frame, from_=0, to=100, command=self.on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.btn_mask = ttk.Button(control_frame, text="Показать маску", command=self.toggle_mask)
        self.btn_mask.pack(side=tk.LEFT, padx=5)
        
        self.btn_process = ttk.Button(control_frame, text="Обработать все кадры", command=self.process_all_frames)
        self.btn_process.pack(side=tk.LEFT, padx=5)
        
        self.btn_export = ttk.Button(control_frame, text="Экспорт PNG", command=self.start_export)
        self.btn_export.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear = ttk.Button(control_frame, text="Сброс", command=self.clear_selection)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # Инициализация переменных
        self.processing = False
        self.start_x = self.start_y = self.end_x = self.end_y = None
        
        # Привязка событий
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = 0
            
            # Точный подсчет кадров
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, _ = self.cap.read()
                if not ret:
                    break
                self.total_frames += 1
            
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.original_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            self.canvas.config(width=self.original_size[0], height=self.original_size[1])
            self.video_loaded = True
            self.show_frame(0)
            self.slider.config(to=self.total_frames)

    def show_frame(self, frame_num):
        if self.video_loaded and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = min(frame_num, self.total_frames - 1)
                self.slider.set(self.current_frame)
                self.display_image(frame)

    def display_image(self, frame):
        display_frame = frame.copy()
        if self.show_mask and self.current_frame in self.masks:
            mask = self.masks[self.current_frame]
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_SPRING)
            display_frame = cv2.addWeighted(display_frame, 1, colored_mask, 0.3, 0)
        
        img = Image.fromarray(display_frame)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def toggle_play(self):
        if self.video_loaded:
            self.playing = not self.playing
            self.btn_play.config(text="⏸" if self.playing else "▶")
            if self.playing:
                self.play()

    def play(self):
        if self.playing and self.current_frame < self.total_frames - 1:
            self.show_frame(self.current_frame)
            self.current_frame += 1
            self.root.after(max(1, int(1000/self.fps)), self.play)

    def on_slider(self, value):
        if self.video_loaded:
            frame_num = int(float(value))
            if frame_num != self.current_frame:
                self.current_frame = frame_num
                self.show_frame(frame_num)

    def start_export(self):
        if self.video_loaded and not self.processing:
            self.processing = True
            self.btn_export.config(text="Остановить", command=self.stop_export)
            thread = threading.Thread(target=self.save_png_sequence)
            thread.start()

    def stop_export(self):
        self.processing = False
        self.btn_export.config(text="Экспорт PNG", command=self.start_export)

    def save_png_sequence(self):
        output_dir = filedialog.askdirectory()
        if not output_dir:
            self.stop_export()
            return
        
        os.makedirs(output_dir, exist_ok=True)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        
        while True:
            if not self.processing:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break

            # Создаем кадр с альфа-каналом
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            mask = self.masks.get(frame_num, np.zeros((rgba_frame.shape[0], rgba_frame.shape[1]), dtype=np.uint8))
            rgba_frame[:, :, 3] = mask  # Применяем маску к альфа-каналу

            # Сохраняем PNG
            filename = f"frame_{frame_num:05d}.png"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(
                save_path,
                rgba_frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )

            # Обновление интерфейса
            self.current_frame = frame_num
            self.slider.set(frame_num)
            self.root.update()
            frame_num += 1
        
        self.stop_export()

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.btn_mask.config(text="Скрыть маску" if self.show_mask else "Показать маску")
        self.show_frame(self.current_frame)

    def clear_selection(self):
        self.selected_object_id = None
        self.current_mask = None
        self.masks.clear()
        self.show_mask = False
        self.btn_mask.config(text="Показать маску")
        self.show_frame(self.current_frame)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_drag(self, event):
        if self.rect:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        self.init_tracking()
        self.show_mask = True
        self.btn_mask.config(text="Скрыть маску")

    def init_tracking(self):
        frame = self.get_current_frame()
        if frame is None:
            return
        
        results = self.model.track(frame, persist=True, verbose=False, classes=None)
        
        if results[0].boxes.id is not None:
            user_bbox = np.array([self.start_x, self.start_y, self.end_x, self.end_y])
            self.selected_object_id = self.find_best_match(results[0], user_bbox)
            self.update_mask(results)

    def update_mask(self, results):
        for mask, track_id in zip(results[0].masks.xy, results[0].boxes.id.cpu().numpy()):
            if int(track_id) == self.selected_object_id:
                self.current_mask = np.zeros((self.original_size[1], self.original_size[0]), dtype=np.uint8)
                cv2.fillPoly(self.current_mask, [np.int32(mask)], 255)
                self.masks[self.current_frame] = self.current_mask
                self.show_frame(self.current_frame)

    def process_all_frames(self):
        if not self.video_loaded or self.selected_object_id is None:
            return
        
        current_position = self.current_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.track(frame_rgb, persist=True, verbose=False, classes=None)
            
            current_mask = np.zeros((self.original_size[1], self.original_size[0]), dtype=np.uint8)
            if results[0].masks is not None:
                for mask, track_id in zip(results[0].masks.xy, results[0].boxes.id.cpu().numpy()):
                    if int(track_id) == self.selected_object_id:
                        cv2.fillPoly(current_mask, [np.int32(mask)], 255)
                        break
            self.masks[frame_num] = current_mask
            
            self.slider.set(frame_num)
            self.root.update()
            frame_num += 1
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        self.show_frame(current_position)

    def find_best_match(self, results, user_bbox):
        max_iou = 0
        best_id = None
        
        for box, track_id in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.id.cpu().numpy()):
            box_area = (box[2]-box[0])*(box[3]-box[1])
            intersection = max(0, min(box[2], user_bbox[2]) - max(box[0], user_bbox[0])) * \
                          max(0, min(box[3], user_bbox[3]) - max(box[1], user_bbox[1]))
            union = box_area + ((user_bbox[2]-user_bbox[0])*(user_bbox[3]-user_bbox[1])) - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > max_iou:
                max_iou = iou
                best_id = int(track_id)
        
        return best_id

    def get_current_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenRotoBrushProject(root)
    root.mainloop()
