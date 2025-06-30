
[app]

# (str) Title of your application
title = My Face Recognition App

# (str) Package name
package.name = facerecognitionapp

# (str) Package domain (needed for android/ios packaging)
package.domain = org.nextgenbywala

# (str) Application versioning (method 1)
version = 0.1

# (str) Application versioning (method 2)
# version.regex = __version__ = ['"](.*)['"]
# version.filename = %(source.dir)s/main.py

# (list) Requirements (python3, kivy, and your specific Python packages)
# Analyzed from your provided Python code:
# - kivy: Your app is built with Kivy.
# - opencv-python: Used extensively for camera, face detection (Haar cascade), and face recognition (LBPH).
# - numpy: Core dependency for OpenCV and general numerical operations.
# - requests: Used for submitting data to Google Forms.
# - pillow: A common image processing library often implicitly required by Kivy and OpenCV for certain image formats/operations.
# - smtplib: Used for sending emails. (Note: smtplib is part of Python's standard library, so it's not strictly a 'pip' requirement
#            for buildozer, but including it explicitly here doesn't hurt and clarifies intent for the Android toolchain.)
requirements = python3.1.1, kivy==2.3.0, opencv-python, numpy, requests, pillow, smtplib
# Note: smtplib is a built-in Python module, so it generally doesn't need to be
# listed in requirements for Python-for-Android unless there's a specific recipe for it.

# (str) Source code where the main.py lives
source.dir = .

# (list) List of exclusions from your application directory, usually things like
# .git, .buildozer, __pycache__ and other temporary files.
source.exclude_dirs = __pycache__, .buildozer, .git, .github, venv, .vscode

# (str) The entrypoint of your application, default is main.py
# Your provided code is the main application logic, assumed to be main.py.
main.py = main.py

# (list) Pattern to include additional files not included by default.
# - thank_you.mp3: Your app loads this audio file.
# - tick.png: Your app loads this image for the overlay.
# - known_faces/: Your app reads and writes to this directory for face data.
include.patterns = thank_you.mp3,tick.png,known_faces/*

# (list) Extensions to use (i.e. android, ios, desktop)
# 'android' will be automatically added when you build for Android.
extensions =

# (int) The Android SDK version to use. Setting to 33 for better stability with python-for-android.
android.api = 33

# (int) The Android NDK version to use. 25b is a stable choice compatible with many APIs.
android.ndk = 25b

# (str) The Android NDK directory (if you want to use a custom one)
# android.ndk_path =

# (str) The Android SDK directory (if you want to use a custom one)
# android.sdk_path =

# (str) The Android Java home directory (if you want to use a custom one)
# android.java_home =

# (bool) If you want to use the NDK from the Android SDK, set this to true
android.skip_ndk_setup = False

# (bool) If you want to use the SDK from the Android Studio, set this to true
android.skip_sdk_setup = False

# (bool) Whether to use the experimental new toolchain
android.new_toolchain = False

# (str) The minimum Android SDK version your app will support.
# 21 ensures compatibility with a very wide range of Android devices.
android.minapi = 21

# (str) The architecture to build for (e.g., armeabi-v7a, arm64-v8a, x86_64)
# Building for both common ARM architectures for broad device compatibility.
android.arch = arm64-v8a,armeabi-v7a

# (list) Permissions your application needs.
# - CAMERA: Explicitly required for accessing the device camera.
# - INTERNET: Required for sending data to Google Forms and sending emails (SMTP).
# - WRITE_EXTERNAL_STORAGE/READ_EXTERNAL_STORAGE: While app data goes to internal storage,
#   these can sometimes be implicitly required by libraries or for broader compatibility.
android.permissions = INTERNET,CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# (list) Optional features your application needs (e.g., usb, audio, etc.)
android.features =

# (bool) Add a notification service (experimental). Essential for larger apps.
# OpenCV can make the app large enough to require multidex.
android.enable_multidex = True

# (list) Services (experimental)
# android.services =

# (bool) Whether to use a debug build (False for release). Start with debug.
android.debug = True # Set to True for debugging!

# (str) Path to your keystore for signing the APK (for release builds)
# android.release_keystore = ~/my_release.keystore

# (str) Password to your keystore
# android.release_keystore_pass =

# (str) Alias of the key in the keystore
# android.release_keystore_alias = my_alias

# (bool) Should the app be launched after build?
# android.presplash_autoclose = False

# (str) The background color of the presplash screen
# android.presplash_color = #FFFFFF

# (str) Path to the presplash image (1280x720 recommended)
# android.presplash_img = %(source.dir)s/data/presplash.png

# (bool) Enable the loading progress bar
# android.loading_bar = True

# (str) The color of the loading progress bar
# android.loading_bar_color = #000000

# (str) AAB format output (Android App Bundle). Recommended for Play Store.
# android.aab = True

# (bool) Whether to enable hardware acceleration (usually recommended)
android.hardware_acceleration = True

# (list) Add Java extensions, if you have any.
# android.extra_java_options =

# (bool) Do not compile against Python shared library (experimental)
# android.no_shared_libs = True

# (bool) Enable the use of external storage for the app data (useful for large data)
# android.force_system_libs = True

# (str) If you are using OpenGL ES 3.0, uncomment and set this.
# android.api_level_target = 33

# (bool) Whether to add additional logging for debugging purposes (recommended for development)
# This filter directs logcat to show all Python logs at DEBUG level, and everything else at SILENT.
# This makes it easier to find your app's specific log messages and Python tracebacks.
android.logcat_filters = *:S python:D

[buildozer]

# (list) Arguments to pass to the python-for-android build process.
# Use this for specific compilation flags or if you need to pass options to recipes.
# For example, to enable a recipe's debug build:
# android.extra_args = --enable-debug --with-recipe=opencv:debug
android.extra_args =

# (int) Log level (0-5). 5 provides maximum verbose output, crucial for debugging.
log_level = 5 # Changed to 5 for maximum verbose output during build

# (str) Where Buildozer stores its internal files (builds, caches, etc.)
buildozer.dir = .buildozer

# (str) Path to the Buildozer config file (usually this file)
buildozer.config = buildozer.spec

# (str) Version of Buildozer to use (leave empty for latest)
# buildozer.version =

[app.icon]
# (str) Path to the icon file (PNG, 512x512 recommended)
# You will need to provide your app icon here, e.g., 'data/icon.png'
# android = data/icon.png
# ios = data/icon.png
# desktop = data/icon.png

[app.url]
# (str) URL to your application's homepage or repository
# homepage = https://github.com/yourusername/yourproject
