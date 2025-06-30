from __future__ import annotations

import glob
import json
import os
import queue
import random
import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.animation import Animation # Import Animation for fading in/out


# --- Configuration for Phone APK ---
# KNOWN_FACES_DIR will be set dynamically in build() using self.user_data_dir
# This ensures it's stored in the app's private internal storage on Android.
# For desktop testing, it will fallback to a 'known_faces' folder relative to the script.

SAMPLES_PER_USER: int = 10
FRAME_REDUCE_FACTOR: float = 0.5
RECOGNITION_INTERVAL: int = 5 * 60  # seconds between repeated recognitions of same face

# Ensure these files are placed in your project's root directory for Buildozer to bundle them.
AUDIO_FILE: str = "thank_you.mp3"
TICK_ICON_PATH: str = "tick.png"
# HAAR_CASCADE_PATH points to the cascade XML file.
# We'll try loading from this relative path first (good for bundled APKs),
# then fallback to OpenCV's default data path for PC environments.
HAAR_CASCADE_PATH: str = "./haarcascade_frontalface_default.xml" 

# Google-Form configuration: View URL is used as referer header, POST goes to
# the *formResponse* endpoint.
GOOGLE_FORM_VIEW_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/viewform"
)
GOOGLE_FORM_POST_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
FORM_FIELDS: Dict[str, str] = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

# E-mail (OTP) settings. THESE SHOULD BE PROVIDED VIA ENVIRONMENT VARIABLES
# FOR SECURITY – fallback values are for offline testing only.
# On a phone, setting environment variables might require specific tools or build processes.
# For production, consider more secure methods for credentials than hardcoding defaults.
EMAIL_ADDRESS: str = os.environ.get("FACEAPP_EMAIL", "faceapp0011@gmail.com")
EMAIL_PASSWORD: str = os.environ.get("FACEAPP_PASS", "ytup bjrd pupf tuuj")
SMTP_SERVER: str = "smtp.gmail.com"
SMTP_PORT: int = 587

# NEW: Admin email address for notifications
ADMIN_EMAIL_ADDRESS: str = os.environ.get("FACEAPP_ADMIN_EMAIL", "projects@archtechautomation.com") # <--- REPLACE WITH YOUR ADMIN EMAIL

# Simple logger helper (replace with logging module for production).
Logger = print

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    """Create directory *path* (including parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def python_time_now() -> str:
    """Returns the current time formatted as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _crop_and_resize_for_passport(cv_image: np.ndarray, target_size: Tuple[int, int] = (240, 320)) -> np.ndarray:
    """
    Crops and resizes an OpenCV image (numpy array) to a specified target_size
    while maintaining a passport-like aspect ratio (3:4 portrait).
    Args:
        cv_image (np.ndarray): The input OpenCV image (BGR or grayscale).
        target_size (Tuple[int, int]): The desired output size (width, height).
    Returns:
        np.ndarray: The processed image as a NumPy array.
    """
    h, w = cv_image.shape[:2]
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height

    current_aspect_ratio = w / h

    cropped_image = cv_image
    if current_aspect_ratio > target_aspect_ratio:
        # Image is wider than target, crop width from center
        new_width = int(h * target_aspect_ratio)
        x_start = (w - new_width) // 2
        cropped_image = cv_image[:, x_start : x_start + new_width]
    elif current_aspect_ratio < target_aspect_ratio:
        # Image is taller than target, crop height from center
        new_height = int(w / target_aspect_ratio)
        y_start = (h - new_height) // 2
        cropped_image = cv2.resize(cv_image[y_start : y_start + new_height, :], target_size, interpolation=cv2.INTER_AREA)
        return cropped_image
    
    # If the aspect ratio is already close or equal, just resize
    # Use INTER_AREA for shrinking, which is generally good for image quality
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def _create_kivy_texture_from_cv_image(cv_image: np.ndarray) -> Texture:
    """
    Converts an OpenCV image (numpy array) to a Kivy Texture.
    Handles both grayscale and BGR images.
    """
    h, w = cv_image.shape[:2]
    if len(cv_image.shape) == 2: # Grayscale
        buf = cv2.flip(cv_image, 0).tobytes()
        img_texture = Texture.create(size=(w, h), colorfmt="luminance")
        img_texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
    else: # BGR (color)
        buf = cv2.flip(cv_image, 0).tobytes()
        img_texture = Texture.create(size=(w, h), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
    return img_texture


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class FaceApp(App):
    """Kivy application for face-recognition based attendance."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        # Determine the directory for known faces. On Android, this will be the app's user data directory.
        # For desktop testing, it will be a 'known_faces' folder in the current working directory.
        self.known_faces_dir: str = "" # This will be set in build() after app is ready and user_data_dir is available

        # Haar cascade for face detection.
        # Try loading from the HAAR_CASCADE_PATH first. If that fails,
        # try loading from OpenCV's default data directory (common for PC installations).
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            Logger(f"[WARN] Failed to load Haar cascade from '{HAAR_CASCADE_PATH}'. Attempting fallback to OpenCV data path.")
            fallback_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(fallback_path)
            
            if self.face_cascade.empty():
                error_msg = (f"[ERROR] Failed to load Haar cascade classifier from both "
                             f"'{HAAR_CASCADE_PATH}' and '{fallback_path}'. "
                             f"Please ensure 'haarcascade_frontalface_default.xml' is present and accessible "
                             f"in your project folder for PC, or bundled for APK.")
                Logger(error_msg)
                raise RuntimeError(error_msg) # This will cause the app to exit on startup if cascade fails
            else:
                Logger(f"[INFO] Successfully loaded Haar cascade from fallback path: '{fallback_path}'.")
        else:
            Logger(f"[INFO] Successfully loaded Haar cascade from: '{HAAR_CASCADE_PATH}'.")


        # Recognizer will be trained after known_faces_dir is set in build()
        self.recognizer = None
        self.label_map = {}

        # State dictionaries.
        self.last_seen_time: Dict[str, float] = {}
        self.otp_storage: Dict[str, str] = {}
        self.pending_names: Dict[str, Optional[str]] = {}

        # User emails will be loaded after known_faces_dir is set.
        self.user_emails: Dict[str, str] = {}

        # Frame queue between capture-thread and UI thread.
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        # Tick overlay icon (RGBA PNG) – optional.
        self.tick_icon: Optional[np.ndarray] = None # Loaded in build()

        # Optional success sound.
        self.sound = None # Loaded in build()

        # Thread/co-ordination primitives.
        self._stop_event = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None

        # Attributes for visual flash
        self.flash_event = None
        self.flash_rect = None

        # Attributes for temporary recognition display
        self.preview_layout: Optional[BoxLayout] = None
        self.preview_image: Optional[Image] = None
        self.preview_name_label: Optional[Label] = None
        self.preview_time_label: Optional[Label] = None
        self.preview_rect: Optional[Rectangle] = None


    # ---------------------------------------------------------------------
    # Kivy UI building / tearing down
    # ---------------------------------------------------------------------

    def build(self):  # noqa: D401 (Kivy signature)
        """Builds the Kivy UI layout."""
        # Set the dynamic known_faces_dir here
        self.known_faces_dir = str(Path(self.user_data_dir) / "known_faces")
        ensure_dir(self.known_faces_dir)
        Logger(f"[INFO] Known faces directory set to: {self.known_faces_dir}")

        # Load resources now that user_data_dir is available or paths are relative
        self.recognizer, self.label_map = self._train_recognizer()
        self.user_emails = self._load_emails()
        self.tick_icon = self._load_tick_icon()
        self.sound = SoundLoader.load(AUDIO_FILE) or None


        root = FloatLayout()

        # Live camera frame display.
        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        root.add_widget(self.image_widget)

        # Button bar.
        button_bar = BoxLayout(
            orientation="horizontal",
            size_hint=(1, None),
            height=dp(48),
            pos_hint={"center_x": 0.5, "y": 0.02},
            spacing=dp(10),
            padding=dp(10),
        )
        self.register_btn = Button(
            text="Register New Face", background_color=(0.13, 0.59, 0.95, 1)
        )
        self.update_btn = Button(
            text="Update Photos", background_color=(0.20, 0.80, 0.20, 1)
        )
        button_bar.add_widget(self.register_btn)
        button_bar.add_widget(self.update_btn)
        root.add_widget(button_bar)

        # Add stylish borders around buttons.
        for btn in (self.register_btn, self.update_btn):
            with btn.canvas.after:
                Color(1, 1, 1, 1)
                Line(width=1.5, rectangle=(btn.x, btn.y, btn.width, btn.height))
            btn.bind(pos=self._update_btn_border, size=self._update_btn_border)

        # Event bindings.
        self.register_btn.bind(on_press=self._register_popup)
        self.update_btn.bind(on_press=self._update_photos_popup)

        # Open webcam (index 0) – on Android, this often defaults to the front camera,
        # but exact behavior can vary by device.
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            # For a phone app, consider showing a more user-friendly error message or dialog.
            Logger("[ERROR] Cannot open webcam – please check camera permissions or availability.")
            # Show a popup message instead of crashing
            Clock.schedule_once(lambda _dt: self._show_popup("Camera Error", Label(text="Cannot open camera. Please check permissions or if another app is using it."), size=(0.8, 0.4)), 0)
            # You might want to disable camera-dependent features here
            self.capture = None # Set to None to prevent further errors if not opened
            # No need to raise RuntimeError if handling gracefully

        # Start capture/processing thread only if camera opened successfully
        if self.capture:
            self.capture_thread = threading.Thread(
                target=self._camera_loop, daemon=True, name="CameraThread"
            )
            self.capture_thread.start()

            # Schedule UI texture updates at ~30 FPS only if camera is active
            Clock.schedule_interval(self._update_texture, 1 / 30)
        else:
            Logger("[WARN] Camera not available, live feed and recognition disabled.")
            # Provide visual feedback that camera is not active
            self.image_widget.source = 'placeholder_no_camera.png' # A placeholder image (you'd need to add this file)
            self.image_widget.text = "Camera Unavailable" # Or set a text on the Image widget if no source


        # Add a label for displaying status messages (e.g., "Attendance recorded!")
        self.status_label = Label(
            text="",
            size_hint=(None, None),
            size=(dp(400), dp(50)),
            pos_hint={"center_x": 0.5, "top": 0.95},
            color=(1, 1, 0, 1), # Default color (yellow)
            font_size=dp(20),
            bold=True,
            halign='center',
            valign='middle'
        )
        root.add_widget(self.status_label)

        # --- New: Temporary recognition display (left side) ---
        self.preview_layout = BoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            # Set fixed size based on photo size + padding for text
            # Photo size (240x320) + 2*10dp padding left/right for width = 260dp
            # Height: 320 (photo) + 2*10dp (padding top/bottom) + 2*25dp (labels) = 390dp
            size=(dp(260), dp(390)),
            pos_hint={'x': 0.02, 'center_y': 0.5}, # Position on the left
            padding=dp(10),
            spacing=dp(5),
            opacity=0, # Start as invisible
        )
        # Add canvas instructions using 'with' block after initialization
        with self.preview_layout.canvas.before:
            Color(0, 0, 0, 0.7) # Background color with transparency
            self.preview_rect = Rectangle(pos=self.preview_layout.pos, size=self.preview_layout.size)
        
        self.preview_layout.bind(pos=self._update_preview_canvas, size=self._update_preview_canvas)

        # The image widget will now directly reflect the 240x320 texture in this space
        self.preview_image = Image(size_hint_y=None, height=dp(320), allow_stretch=True, keep_ratio=True) # Explicitly set height based on target_size
        self.preview_name_label = Label(text="", font_size=dp(20), color=(1, 1, 1, 1), halign='center', valign='middle', size_hint_y=None, height=dp(25)) # Increased font size, fixed height
        self.preview_time_label = Label(text="", font_size=dp(16), color=(0.8, 0.8, 0.8, 1), halign='center', valign='middle', size_hint_y=None, height=dp(25)) # Increased font size, fixed height

        self.preview_layout.add_widget(self.preview_image)
        self.preview_layout.add_widget(self.preview_name_label)
        self.preview_layout.add_widget(self.preview_time_label)
        root.add_widget(self.preview_layout)
        # --- End: New temporary recognition display ---


        return root

    def on_stop(self) -> None:  # noqa: D401 (Kivy signature)
        """Called by Kivy when the application is shutting down."""
        self._stop_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.capture:
            self.capture.release()

        Logger(f"[INFO] Application closed cleanly – {python_time_now()}")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_btn_border(instance, *_):  # noqa: ANN001 (Kivy signature)
        """Updates the border around a Kivy button."""
        instance.canvas.after.clear()
        with instance.canvas.after:
            Color(1, 1, 1, 1)
            Line(width=1.5, rectangle=(instance.x, instance.y, instance.width, instance.height))

    def _update_preview_canvas(self, instance, value):
        """Updates the background rectangle for the preview layout."""
        if hasattr(self, 'preview_rect') and self.preview_rect:
            self.preview_rect.pos = instance.pos
            self.preview_rect.size = instance.size


    def _show_popup(self, title: str, content: BoxLayout, *, size=(0.8, 0.5)) -> Popup:  # noqa: D401
        """Helper to display a Kivy popup, now with a back button."""
        # Create a new BoxLayout to hold the original content and the back button
        main_content_layout = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        
        # Add the original content to this new layout
        main_content_layout.add_widget(content)

        # Add a back button
        back_button = Button(
            text="Back to Camera",
            size_hint=(1, None),
            height=dp(40),
            background_color=(0.5, 0.5, 0.5, 1) # Grey color for back button
        )
        main_content_layout.add_widget(back_button)

        popup = Popup(title=title, content=main_content_layout, size_hint=size, auto_dismiss=False)
        
        # Bind the back button to dismiss the popup
        back_button.bind(on_press=popup.dismiss)
        
        popup.open()
        return popup

    def _show_status_message(self, message: str, duration: float = 3.0, color=(1, 1, 0, 1)):
        """
        Displays a temporary status message on the UI.
        Args:
            message (str): The message to display.
            duration (float): How long the message should be visible in seconds.
            color (tuple): RGBA color tuple for the message text.
        """
        def update_label(_dt):
            self.status_label.text = message
            self.status_label.color = color
            Clock.schedule_once(lambda __dt: self._clear_status_message(), duration)
        # Schedule the label update on the main Kivy thread
        Clock.schedule_once(update_label, 0)

    def _clear_status_message(self):
        """Clears the status message from the UI."""
        self.status_label.text = ""
        self.status_label.color = (1, 1, 0, 1) # Reset to default color

    def _flash_image_widget(self):
        """Briefly flashes a green border around the image widget to indicate a photo capture."""
        # Clear any existing flash event
        if self.flash_event:
            self.flash_event.cancel()
            if self.flash_rect:
                self.image_widget.canvas.after.remove(self.flash_rect)

        with self.image_widget.canvas.after:
            Color(0, 1, 0, 1)  # Green color for flash
            # Draw a rectangle that matches the image widget's current size and position
            self.flash_rect = Line(
                width=3,
                rectangle=(
                    self.image_widget.x,
                    self.image_widget.y,
                    self.image_widget.width,
                    self.image_widget.height
                )
            )

        # Schedule clearing the flash
        self.flash_event = Clock.schedule_once(self._clear_flash, 0.1)

    def _clear_flash(self, _dt):
        """Clears the green border flash from the image widget."""
        if self.flash_rect:
            self.image_widget.canvas.after.remove(self.flash_rect)
            self.flash_rect = None
        self.flash_event = None # Clear the event reference


    def _load_tick_icon(self) -> Optional[np.ndarray]:
        """Loads the tick icon for overlay, if available."""
        # Use a relative path assuming it's bundled with the app
        if not Path(TICK_ICON_PATH).is_file():
            Logger(f"[WARN] Tick icon '{TICK_ICON_PATH}' missing – overlay disabled.")
            return None
        return cv2.imread(TICK_ICON_PATH, cv2.IMREAD_UNCHANGED)

    # ------------------------------------------------------------------
    # Camera capture + recognition thread
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        """Runs in a background thread: capture, detect & recognise faces."""
        while not self._stop_event.is_set():
            if not self.capture or not self.capture.isOpened():
                time.sleep(0.5) # Wait if camera is not ready
                continue

            ret, frame = self.capture.read()
            if not ret:
                Logger("[WARN] Failed to grab frame from camera.")
                time.sleep(0.1) # Small delay to avoid busy-looping on capture error
                continue  # Skip invalid frame.

            # Down-scale for faster detection.
            h, w = frame.shape[:2]
            resized = cv2.resize(
                frame, (int(w * FRAME_REDUCE_FACTOR), int(h * FRAME_REDUCE_FACTOR))
            )
            gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Haar face detection.
            try:
                faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV error in detectMultiScale: {e}. This might indicate a corrupted cascade file or an issue with your OpenCV installation.")
                faces = [] # Treat as no faces detected if error occurs


            for (x, y, w_s, h_s) in faces:
                # Map coordinates back to original frame.
                x_full, y_full, w_full, h_full = [
                    int(v / FRAME_REDUCE_FACTOR) for v in (x, y, w_s, h_s)
                ]

                # --- Start of modifications for passport photo ---
                # Expand the face bounding box to include more context for a passport photo
                # A factor of 1.8 expands the width/height of the bounding box by 80%
                expansion_factor = 1.8
                
                # Calculate new dimensions for the expanded region
                exp_w = int(w_full * expansion_factor)
                exp_h = int(h_full * expansion_factor)

                # Calculate the center of the original face bounding box
                center_x = x_full + w_full // 2
                center_y = y_full + h_full // 2

                # Calculate the top-left corner of the expanded region
                exp_x = center_x - exp_w // 2
                exp_y = center_y - exp_h // 2

                # Ensure the expanded region stays within the frame boundaries
                frame_h, frame_w = frame.shape[:2]
                exp_x = max(0, min(exp_x, frame_w - exp_w))
                exp_y = max(0, min(exp_y, frame_h - exp_h))
                exp_w = min(exp_w, frame_w - exp_x)
                exp_h = min(exp_h, frame_h - exp_y)

                # Extract the expanded color ROI
                color_face_roi = frame[exp_y : exp_y + exp_h, exp_x : exp_x + exp_w].copy()
                # --- End of modifications for passport photo ---


                # Convert to grayscale for recognition
                grayscale_face_roi = cv2.cvtColor(color_face_roi, cv2.COLOR_BGR2GRAY)
                try:
                    label, conf = self.recognizer.predict(grayscale_face_roi) # Use grayscale for prediction
                except Exception as e:
                    Logger(f"[ERROR] Recognizer prediction failed: {e}")
                    label, conf = -1, 1000  # Unknown.

                name, emp_id = self.label_map.get(label, ("unknown", ""))
                now = time.time()

                if conf < 60:  # Recognised.
                    last_seen = self.last_seen_time.get(emp_id, 0)
                    if now - last_seen > RECOGNITION_INTERVAL:
                        self.last_seen_time[emp_id] = now
                        threading.Thread(
                            target=self._handle_successful_recognition,
                            args=(name, emp_id, color_face_roi), # Pass color_face_roi here
                            daemon=True,
                            name="AttendanceSubmitter",
                        ).start()
                        # Show success message on UI
                        self._show_status_message(f"Attendance recorded for {name.title()}!", 3, (0, 1, 0, 1)) # Green color
                    else:
                        # Show already done message on UI without timer
                        self._show_status_message(f"Attendance already recorded for {name.title()}.", 3, (1, 0.5, 0, 1)) # Orange color
                    
                    # Draw green rectangle & label for recognized faces (even if on cooldown)
                    cv2.rectangle(
                        frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{name.title()} ({emp_id})",
                        (x_full, y_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    # Call the new function to overlay tick next to name
                    self._overlay_tick_next_to_name(frame, x_full, y_full - 10, name.title(), emp_id, 0.7, 2)
                else:  # Unknown face.
                    cv2.rectangle(
                        frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 0, 255), 2
                    )
                    cv2.putText(
                        frame,
                        "Unknown",
                        (x_full, y_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Place latest frame into queue (discard older).
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    # ------------------------------------------------------------------
    # UI texture refresh (main thread)
    # ------------------------------------------------------------------

    def _update_texture(self, _dt) -> None:  # noqa: D401 (Kivy signature)
        """Updates the Kivy Image widget with the latest camera frame."""
        if self.frame_queue.empty():
            return
        frame = self.frame_queue.get()
        # Flip the frame vertically for Kivy texture, convert to bytes.
        buf = cv2.flip(frame, 0).tobytes()
        # Create a Kivy texture from the buffer.
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image_widget.texture = img_texture

    # ------------------------------------------------------------------
    # Training / retraining recogniser
    # ------------------------------------------------------------------

    def _train_recognizer(self):  # noqa: D401 (private helper)
        """Trains the LBPH face recognizer on known faces."""
        images: list[np.ndarray] = []
        labels: list[int] = []
        label_map: Dict[int, Tuple[str, str]] = {}
        label_id = 0

        # Use self.known_faces_dir here
        # Ensure the directory exists before listing, especially on first run.
        ensure_dir(self.known_faces_dir)
        for file in sorted(os.listdir(self.known_faces_dir)):
            if not file.lower().endswith((".jpg", ".png")):
                continue
            try:
                name, emp_id, _ = file.split("_", 2)
                name = name.lower()
                emp_id = emp_id.upper()
            except ValueError:
                Logger(f"[WARN] Skipping unrecognised filename format: {file}")
                continue

            img_path = Path(self.known_faces_dir) / file
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                Logger(f"[WARN] Could not read image: {img_path}")
                continue

            # Resize image to 200x200 during training as well, for consistency
            img_resized = cv2.resize(img_gray, (200, 200))
            images.append(img_resized)
            labels.append(label_id)
            label_map[label_id] = (name, emp_id)
            label_id += 1

        recogniser = cv2.face.LBPHFaceRecognizer_create()
        if images:
            recogniser.train(images, np.array(labels))
            Logger(
                f"[INFO] Trained recogniser on {len(images)} images across {len(label_map)} identities."
            )
        else:
            Logger("[INFO] No images found – recogniser disabled until first registration.")

        return recogniser, label_map

    # ------------------------------------------------------------------
    # Registration / update photo flows
    # ------------------------------------------------------------------

    # -------- Registration --------

    def _register_popup(self, _btn):  # noqa: ANN001 (Kivy signature)
        """Shows the popup for new face registration."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        name_input = TextInput(hint_text="Full Name", size_hint=(1, None), height=dp(40))
        id_input = TextInput(hint_text="Employee ID", size_hint=(1, None), height=dp(40))
        email_input = TextInput(hint_text="Email", size_hint=(1, None), height=dp(40))
        submit_btn = Button(text="Capture Faces", size_hint=(1, None), height=dp(40))

        for widget in (
            Label(text="Enter Details"),
            name_input,
            id_input,
            email_input,
            submit_btn,
        ):
            content.add_widget(widget)
        popup = self._show_popup("Register Face", content, size=(0.9, 0.6))

        def _submit(_):  # noqa: ANN001
            """Handles submission of registration details."""
            name = name_input.text.strip().lower().replace(" ", "_")
            emp_id = id_input.text.strip().upper()
            email = email_input.text.strip()
            if not (name and emp_id and email and "@" in email):
                Logger("[WARN] Invalid input for registration.")
                self._show_status_message("Invalid input. Please fill all fields correctly.", 3, (1, 0, 0, 1))
                return

            self._save_email(emp_id, email)
            popup.dismiss()
            threading.Thread(
                target=self._capture_samples,
                args=(name, emp_id, False), # False for new registration
                daemon=True,
                name="CaptureSamples(New)",
            ).start()

        submit_btn.bind(on_press=_submit)

    # -------- Update Existing Photos --------

    def _update_photos_popup(self, _btn):  # noqa: ANN001
        """Shows the popup for updating existing user photos."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text="Enter your Employee ID:"))
        emp_input = TextInput(hint_text="EMP ID", size_hint=(1, None), height=dp(40))
        next_btn = Button(text="Next", size_hint=(1, None), height=dp(40))
        for w in (emp_input, next_btn):
            content.add_widget(w)
        popup = self._show_popup("Update Photos", content)

        def _next(_):  # noqa: ANN001
            """Handles the next step in the update photos flow."""
            emp_id = emp_input.text.strip().upper()
            if not emp_id:
                Logger("[WARN] Employee ID cannot be empty for update.")
                self._show_status_message("Employee ID cannot be empty.", 3, (1, 0, 0, 1))
                return
            email = self.user_emails.get(emp_id)
            name_existing: Optional[str] = None
            for _lbl, (nm, eid) in self.label_map.items():
                if eid == emp_id:
                    name_existing = nm
                    break

            popup.dismiss()
            if email:
                self._send_otp_flow(emp_id, email, name_existing)
            else:
                self._email_registration_flow(emp_id, name_existing)

        next_btn.bind(on_press=_next)

    # ------------------------------------------------------------------
    # Helper flows for OTP / email registration
    # ------------------------------------------------------------------

    def _email_registration_flow(self, emp_id: str, name: Optional[str]):
        """Initiates the flow to register an email for an existing employee ID."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text="Email not found. Enter your email:"))
        email_input = TextInput(hint_text="Email", size_hint=(1, None), height=dp(40))
        submit_btn = Button(text="Submit", size_hint=(1, None), height=dp(40))
        content.add_widget(email_input)
        content.add_widget(submit_btn)
        popup = self._show_popup("Register Email", content)

        def _submit(_):  # noqa: ANN001
            """Handles email submission for registration."""
            email = email_input.text.strip()
            if email and "@" in email:
                self._save_email(emp_id, email)
                popup.dismiss()
                self._send_otp_flow(emp_id, email, name)
            else:
                Logger("[WARN] Invalid email format during registration.")
                self._show_status_message("Invalid email format.", 3, (1, 0, 0, 1))

        submit_btn.bind(on_press=_submit)

    def _send_otp_flow(self, emp_id: str, email: str, name: Optional[str] = None):
        """Manages the OTP sending process."""
        # Ensure name is not None, attempt to get it from label_map if it is.
        resolved_name = name
        if resolved_name is None:
            for _lbl, (nm, eid) in self.label_map.items():
                if eid == emp_id:
                    resolved_name = nm
                    break
        # If name is still None, use a fallback
        if resolved_name is None:
            resolved_name = "Unknown User"

        # Generate & store 6-digit OTP.
        otp = self._generate_otp()
        self.otp_storage[emp_id] = otp
        self.pending_names[emp_id] = resolved_name

        sending_popup = self._show_popup("Sending OTP", Label(text="Sending OTP email…"), size=(0.7, 0.4))

        def _send_thread():  # noqa: ANN001
            """Sends the OTP email(s) in a separate thread."""
            # Send OTP to the user
            user_mail_ok = self._send_otp_email(email, otp, resolved_name, emp_id, is_admin_email=False)
            
            # Send notification to admin
            admin_mail_ok = self._send_otp_email(ADMIN_EMAIL_ADDRESS, otp, resolved_name, emp_id, is_admin_email=True) # Pass OTP to admin email

            Clock.schedule_once(lambda _dt: sending_popup.dismiss())
            if user_mail_ok:
                Clock.schedule_once(lambda _dt: self._otp_verify_popup(emp_id, email))
            else:
                Clock.schedule_once(
                    lambda _dt: self._show_popup("Error", Label(text="Failed to send OTP email. Please check console/internet connection."), size=(0.7, 0.4))
                )
            if not admin_mail_ok:
                 Logger(f"[WARN] Failed to send admin notification email to {ADMIN_EMAIL_ADDRESS}.")


        threading.Thread(target=_send_thread, daemon=True, name="SendOTPThread").start()

    def _otp_verify_popup(self, emp_id: str, email: str):
        """Shows the popup for OTP verification."""
        content = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text=f"OTP sent to {email}"))
        otp_input = TextInput(hint_text="6-digit OTP", size_hint=(1, None), height=dp(40))
        verify_btn = Button(text="Verify", size_hint=(1, None), height=dp(40))
        resend_btn = Button(text="Resend", size_hint=(1, None), height=dp(40))
        content.add_widget(otp_input)
        content.add_widget(verify_btn)
        content.add_widget(resend_widget) # Add resend button widget
        popup = self._show_popup("Verify OTP", content)

        def _verify(_):  # noqa: ANN001
            """Verifies the entered OTP."""
            if otp_input.text.strip() == self.otp_storage.get(emp_id):
                popup.dismiss()
                name_for_capture = self.pending_names.get(emp_id)
                threading.Thread(
                    target=self._capture_samples,
                    args=(name_for_capture, emp_id, True, 5), # True for update, capture 5 samples
                    daemon=True,
                    name="CaptureSamples(Update)",
                ).start()
            else:
                otp_input.text = ""
                otp_input.hint_text = "Incorrect – try again"
                Logger("[WARN] Incorrect OTP entered.")
                self._show_status_message("Incorrect OTP. Please try again.", 3, (1, 0, 0, 1))

        def _resend(_):  # noqa: ANN001
            """Resends a new OTP."""
            new_otp = self._generate_otp()
            self.otp_storage[emp_id] = new_otp
            name_for_resend = self.pending_names.get(emp_id, "Unknown User") # Get name for resend
            self._send_otp_email(email, new_otp, name_for_resend, emp_id, is_admin_email=False) # Send to user
            self._send_otp_email(ADMIN_EMAIL_ADDRESS, new_otp, name_for_resend, emp_id, is_admin_email=True) # Send OTP to admin
            otp_input.text = ""
            otp_input.hint_text = "New OTP sent"
            Logger("[INFO] Resent OTP.")
            self._show_status_message("New OTP sent. Check your email.", 3, (0, 1, 0, 1))

        verify_btn.bind(on_press=_verify)
        resend_btn.bind(on_press=_resend)

    # ------------------------------------------------------------------
    # Core logic: capturing face samples, attendance submission
    # ------------------------------------------------------------------

    def _capture_samples(
        self,
        name: Optional[str],
        emp_id: str,
        updating: bool = False,
        sample_count: Optional[int] = None,
    ):
        """Captures face samples for a given user.
        Incorporates image resizing, specific filename format, and delay from user's reference.
        Adds a countdown before capture and a completion message.
        """
        # Resolve name for existing employee ID if not supplied.
        if name is None:
            for _lbl, (nm, eid) in self.label_map.items():
                if eid == emp_id:
                    name = nm
                    break
        if name is None:
            Logger("[ERROR] No existing face found for this ID – please register first.")
            Clock.schedule_once(lambda _dt: self._show_popup("Error", Label(text="No existing face found for this ID. Please register first."), size=(0.7, 0.4)))
            return

        count_target = sample_count if sample_count else SAMPLES_PER_USER
        # Use self.known_faces_dir here
        pattern = str(Path(self.known_faces_dir) / f"{name}_{emp_id}_*.jpg")
        existing_files = glob.glob(pattern)
        start_index = len(existing_files)
        collected = 0

        Logger(
            f"[INFO] Starting sample capture for {emp_id} – target {count_target} faces (updating={updating})."
        )

        # --- Countdown before capture ---
        for i in range(3, 0, -1):
            self._show_status_message(f"Capturing in {i}...", 1, (1, 1, 0, 1)) # Yellow color for countdown
            time.sleep(1) # Wait for 1 second for each countdown number
        self._show_status_message("Capturing now!", 1, (0, 1, 0, 1)) # Green for "Capturing now!"
        time.sleep(0.5) # Small pause before starting actual capture

        # Loop to capture the target number of samples
        while collected < count_target and not self._stop_event.is_set():
            # Get the latest frame from the queue without blocking the camera thread
            # This helps in reducing perceived lag as the UI always gets the freshest frame
            frame = None
            # Check if self.capture is valid before trying to get frames
            if not self.capture or not self.capture.isOpened():
                Logger("[WARN] Camera not available during sample capture. Stopping capture.")
                self._show_status_message("Camera not available. Cannot capture photos.", 3, (1, 0, 0, 1))
                break # Exit the loop if camera is not available

            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            
            if frame is None:
                time.sleep(0.01) # Small sleep if no frame available, to prevent busy-waiting
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use scaleFactor=1.3 as per user's reference for face detection during capture
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Iterate through all detected faces in the current frame to find one to capture
            # This loop will now continue until `count_target` images are collected
            if len(faces) > 0: # Only process if a face is detected
                # Take the first detected face (assuming only one person is registering at a time)
                x, y, w, h = faces[0] 
                face_img = gray[y : y + h, x : x + w]
                # Resize image to 200x200 as per user's reference
                face_img_resized = cv2.resize(face_img, (200, 200))
                # Use current logic for filename to ensure unique names and proper continuation for updates
                filename = f"{name}_{emp_id}_{start_index + collected:03d}.jpg"
                # Use self.known_faces_dir here
                cv2.imwrite(str(Path(self.known_faces_dir) / filename), face_img_resized) # Save resized image
                collected += 1
                Logger(f"[INFO] Captured sample {collected}/{count_target} for {emp_id}")
                
                # Visual feedback for captured photo (on main thread)
                Clock.schedule_once(lambda _dt: self._flash_image_widget(), 0)

                # Update status label with capture progress
                self._show_status_message(f"Captured {collected}/{count_target} photos...", 0.5, (1, 1, 0, 1))
                time.sleep(0.2)  # Use 0.2s delay as per user's reference to allow for head movement and varied samples
            else:
                # If no face is detected, inform the user to position correctly
                self._show_status_message("No face detected. Please position yourself.", 0.5, (1, 0, 0, 1)) # Red color for warning
                time.sleep(0.1) # Small delay to avoid hammering the loop if no face is present


        Logger("[INFO] Capture complete – retraining recogniser…")
        # Retrain recognizer after new samples are captured.
        self.recognizer, self.label_map = self._train_recognizer()
        Logger("[INFO] Update finished.")
        # Show "Registration completed" or "Face updated" message based on 'updating' flag
        if updating:
            Clock.schedule_once(lambda _dt: self._show_status_message("Face updated!", 3, (0, 1, 0, 1)), 0)
        else:
            Clock.schedule_once(lambda _dt: self._show_status_message("Registration completed!", 3, (0, 1, 0, 1)), 0)


    # ------------------------------------------------------------------
    # Successful recognition & attendance submission
    # ------------------------------------------------------------------

    def _handle_successful_recognition(self, name: str, emp_id: str, face_roi_color: np.ndarray):
        """Handles a successful face recognition event."""
        Logger(f"[INFO] Recognised {name} ({emp_id}) – submitting attendance…")
        if self.sound:
            self.sound.play()
        
        # Process the face image to passport size directly without saving to disk
        # This function takes the expanded face_roi_color and crops/resizes it
        # to the desired passport photo dimensions (240x320).
        processed_face_image = _crop_and_resize_for_passport(face_roi_color, (240, 320))
        
        # Schedule UI update for recognition display on the main thread
        current_time = datetime.now().strftime("%H:%M:%S")
        Clock.schedule_once(
            lambda _dt: self._display_recognition_info(name.title(), emp_id, current_time, processed_face_image),
            0
        )

        # Submit to Google Form in a separate thread to avoid blocking UI
        threading.Thread(
            target=self._submit_to_google_form,
            args=(name, emp_id),
            daemon=True,
            name="GoogleFormSubmitter",
        ).start()

    def _display_recognition_info(self, name: str, emp_id: str, current_time: str, processed_face_image: np.ndarray):
        """
        Displays the recognized person's photo, name, and time on the UI.
        Called on the main Kivy thread.
        Args:
            name (str): The name of the recognized person.
            emp_id (str): The employee ID of the recognized person.
            current_time (str): The current time of recognition.
            processed_face_image (np.ndarray): The NumPy array of the processed face image.
        """
        if self.preview_layout and self.preview_image and self.preview_name_label and self.preview_time_label:
            # Create Kivy texture directly from the NumPy array
            face_texture = _create_kivy_texture_from_cv_image(processed_face_image)
            self.preview_image.texture = face_texture

            self.preview_name_label.text = name
            self.preview_time_label.text = current_time

            # Fade in animation (0.2s) + Hold (2.5s)
            anim = Animation(opacity=1, duration=0.2) + Animation(opacity=1, duration=2.5) 
            anim.bind(on_complete=self._hide_recognition_info) # Schedule fade out
            anim.start(self.preview_layout)
        else:
            Logger("[WARN] Recognition preview widgets not initialized for display.")


    def _hide_recognition_info(self, animation, widget):
        """Fades out the recognition display and clears the image."""
        hide_anim = Animation(opacity=0, duration=0.3)
        hide_anim.start(widget)
        
        # Clear the image texture after animation completes
        def clear_image_texture(_dt):
            self.preview_image.texture = None # Clear the image from the widget
            self.preview_image.source = '' # Ensure source is also cleared if previously set from a file

        Clock.schedule_once(clear_image_texture, animation.duration) # Schedule after the fade-out duration


    def _submit_to_google_form(self, name: str, emp_id: str) -> None:
        """Submit attendance to Google Form with robust error handling.

        Google occasionally returns 404 if the request lacks a proper Referer
        or User-Agent header. We therefore provide both and treat non-success
        status codes as warnings (attendance is stored locally so can be
        re-submitted later if desired).
        """
        payload = {
            FORM_FIELDS["name"]: name.title(),
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: datetime.now().strftime("%d/%m/%Y"),
            FORM_FIELDS["time"]: datetime.now().strftime("%H:%M:%S"),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceApp Attendance Bot)",
            "Referer": GOOGLE_FORM_VIEW_URL,
        }
        
        Logger(f"[INFO] Attempting to submit attendance for {name} ({emp_id}) to URL: {GOOGLE_FORM_POST_URL}")
        Logger(f"[INFO] Payload: {payload}")
        try:
            # Use a session for potentially better connection management, though not strictly necessary for a single request.
            with requests.Session() as session:
                resp = session.post(
                    GOOGLE_FORM_POST_URL,
                    data=payload,
                    headers=headers,
                    timeout=10, # Set a timeout for the request
                    allow_redirects=False, # Google Forms typically redirects on success (302)
                )
            
            # Check for successful status codes. Google Forms usually returns 302 on successful submission.
            if resp.status_code in (200, 302):
                Logger("[INFO] Attendance submitted successfully to Google Form.")
                # Show success message on UI
                Clock.schedule_once(lambda _dt: self._show_status_message(f"Attendance submitted for {name.title()}!", 3, (0, 1, 0, 1)), 0)
            else:
                Logger(
                    f"[WARN] Google Form submission returned status {resp.status_code}. "
                    f"Response: {resp.text[:200]}..." # Log part of the response for debugging
                )
                # Provide a UI feedback for warning/error
                Clock.schedule_once(lambda _dt: self._show_popup("Submission Warning", Label(text=f"Form submission failed (Status: {resp.status_code}). Please check console for details and verify form configuration."), size=(0.8, 0.5)))
        except requests.exceptions.Timeout:
            Logger(f"[ERROR] Google Form submission timed out for {name} ({emp_id}).")
            Clock.schedule_once(lambda _dt: self._show_popup("Submission Error", Label(text="Form submission timed out. Check network connection."), size=(0.8, 0.5)))
        except requests.exceptions.ConnectionError as exc:
            Logger(f"[ERROR] Google Form submission connection error for {name} ({emp_id}): {exc}")
            Clock.schedule_once(lambda _dt: self._show_popup("Submission Error", Label(text="Network error during form submission. Check internet connection."), size=(0.8, 0.5)))
        except requests.RequestException as exc:
            Logger(f"[ERROR] An unexpected error occurred during form submission for {name} ({emp_id}): {exc}")
            Clock.schedule_once(lambda _dt: self._show_popup("Submission Error", Label(text=f"An error occurred during form submission: {exc}"), size=(0.8, 0.5)))


    # ------------------------------------------------------------------
    # OTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_otp() -> str:
        """Generates a 6-digit random OTP."""
        return str(random.randint(100000, 999999))

    def _send_otp_email(self, email: str, otp: str, name: str, emp_id: str, is_admin_email: bool = False) -> bool:
        """Sends an OTP email to the specified address or an admin notification.
        Args:
            email (str): The recipient email address.
            otp (str): The 6-digit OTP (only relevant for user emails).
            name (str): The name of the person.
            emp_id (str): The employee ID of the person.
            is_admin_email (bool): True if this email is for the admin, False for the user.
        """
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = email
        
        if is_admin_email:
            msg["Subject"] = f"FaceApp Notification: Person Details - {name.title()} ({emp_id})"
            body_html = (
                f"<p>Details of a person for whom an OTP process was initiated:</p>"
                f"<p><b>Name:</b> {name.title()}<br>"
                f"<b>Employee ID:</b> {emp_id}</p>"
                f"<p>Generated OTP: <b>{otp}</b></p>"
            )
        else:
            msg["Subject"] = "Your FaceApp OTP"
            body_html = (
                f"<h2>OTP Verification for {name.title()} ({emp_id})</h2><p>Your OTP is <b>{otp}</b>. "
                "It is valid for 10 minutes.</p>"
                "<p>Please use this OTP to proceed with your photo update/registration.</p>"
            )
        
        msg.attach(MIMEText(body_html, "html"))
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            Logger(f"[INFO] Sent {'admin notification' if is_admin_email else 'OTP'} to {email}")
            return True
        except Exception as exc:
            Logger(f"[ERROR] SMTP error when sending {'admin notification' if is_admin_email else 'OTP'} to {email}: {exc}")
            return False

    # ------------------------------------------------------------------
    # E-mail persistence helpers
    # ------------------------------------------------------------------

    def _load_emails(self) -> Dict[str, str]:
        """Loads stored user email addresses from a JSON file."""
        # Use self.known_faces_dir here
        emails_file = Path(self.known_faces_dir) / "user_emails.json"
        if emails_file.is_file():
            try:
                with emails_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                Logger(f"[WARN] Invalid JSON in email storage: {exc}; starting fresh.")
        return {}

    def _save_email(self, emp_id: str, email: str) -> None:
        """Saves a user's email address to the JSON file."""
        # Use self.known_faces_dir here
        self.user_emails[emp_id] = email
        with (Path(self.known_faces_dir) / "user_emails.json").open("w", encoding="utf-8") as f:
            json.dump(self.user_emails, f, indent=2)

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _overlay_tick_next_to_name(self, frame: np.ndarray, text_x: int, text_y_baseline: int, name: str, emp_id: str, font_scale: float, font_thickness: int) -> None:
        """Overlays a tick icon next to the recognized name and ID."""
        if self.tick_icon is None:
            return

        text_to_measure = f"{name} ({emp_id})"
        (text_width, text_height), _ = cv2.getTextSize(text_to_measure, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Desired size for the tick icon (e.g., 25x25 pixels)
        tick_icon_size = 25 # pixels
        icon = cv2.resize(self.tick_icon, (tick_icon_size, tick_icon_size), interpolation=cv2.INTER_AREA)

        # Calculate position for the tick mark
        # Place it slightly to the right of the text, vertically centered with the text.
        padding_x = 5 # pixels between text and tick
        
        icon_x_start = text_x + text_width + padding_x
        # Calculate y-position to center the tick vertically with the text
        # text_y_baseline is the bottom of the text. Top of text is text_y_baseline - text_height.
        # Center of text is text_y_baseline - text_height / 2.
        # Center of icon should be at text_center_y. So, icon_y_top = text_center_y - tick_icon_size / 2.
        icon_y_start = text_y_baseline - text_height + (text_height - tick_icon_size) // 2

        # Ensure the icon is within frame boundaries
        h_frame, w_frame = frame.shape[:2]
        icon_x_start = max(0, min(icon_x_start, w_frame - tick_icon_size))
        icon_y_start = max(0, min(icon_y_start, h_frame - tick_icon_size))

        # Ensure icon_x_start and icon_y_start are integers
        icon_x_start = int(icon_x_start)
        icon_y_start = int(icon_y_start)

        # Get the ROI for blending
        roi = frame[icon_y_start : icon_y_start + tick_icon_size, icon_x_start : icon_x_start + tick_icon_size]
        
        # Check if the ROI is valid (i.e., not out of bounds causing a slice of different size)
        if roi.shape[0] == tick_icon_size and roi.shape[1] == tick_icon_size:
            if icon.shape[2] == 4:  # RGBA image – use alpha channel for blending
                b, g, r, a = cv2.split(icon)
                mask = cv2.merge((a, a, a)) / 255.0
                blended = (roi * (1 - mask) + cv2.merge((b, g, r)) * mask).astype(np.uint8)
                frame[icon_y_start : icon_y_start + tick_icon_size, icon_x_start : icon_x_start + tick_icon_size] = blended
            else:
                Logger("[WARN] Tick icon is not RGBA; cannot perform alpha blending for tick next to name. Simple copy used.")
                # Fallback: simple copy if no alpha channel
                icon_to_place = cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR) if icon.shape[2] == 4 else icon
                frame[icon_y_start : icon_y_start + tick_icon_size, icon_x_start : icon_x_start + tick_icon_size] = icon_to_place
        else:
            Logger(f"[WARN] ROI for tick icon is not the expected size. Skipping overlay. ROI shape: {roi.shape}")


# ---------------------------------------------------------------------------
# Run the application
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FaceApp().run()
