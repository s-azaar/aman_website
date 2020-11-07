
from __future__ import unicode_literals

from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect
from .models import CompanyInfo
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http.response import StreamingHttpResponse

import os
import cv2
import numpy as np
import tensorflow as tf
from  models.object_detection.utils import label_map_util
from models.object_detection.utils import visualization_utils as vis_util

def index(request):
    if request.method == 'POST':
        if request.POST.get('signup'):
            CompanyInfo.objects.create(name_of_company=request.POST.get('comp_name'),
                                       email_of_company=request.POST.get('comp_email'),
                                       password_of_company=request.POST.get('comb_pass'),
                                       activation_code=request.POST.get('activation_code'))
            return render(request, 'main/home.html', {})
        elif request.POST.get('signin'):
            print("i'm hereeeee")
            username = request.POST.get('username')
            password = request.POST.get('password')
            print(username)
            try:
                query = CompanyInfo.objects.get(email_of_company=username)
                print(username+"in try")
            except Exception as e:
                return redirect('403')
            if query.password_of_company == password:
                return redirect('home')
            else:
                print("Someone tried to login and failed.")
                return redirect('403')

    elif request.method == 'GET':
        return render(request, 'main/index.html', {})


# def home(request):
#     return render(request, "main/home.html", {})


def page403(request):
    return render(request, "main/page403.html", {})


######################################3

def dis():
    # Name of the directory containing the object detection module we're using
    print("we are here")
    print(uploaded_file_url)
    VIDEO_NAME = uploaded_file_url
    VIDEO_NAME = VIDEO_NAME[1:]
    print(VIDEO_NAME)

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    print("ffffffff")
    print(CWD_PATH)
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'models/object_detection/inference_graph/frozen_inference_graph.pb')
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'models/object_detection/inference_graph/labelmap.pbtxt')
    # Path to video
    PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)
    print("lllll")
    print(PATH_TO_VIDEO)
    # Number of classes the object detector can identify
    NUM_CLASSES = 400
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=False)
    category_index = label_map_util.create_category_index(categories)

    print("after ll")
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Open video file
    video = cv2.VideoCapture(PATH_TO_VIDEO)

    frameno = 0
    while (video.isOpened()):
        frameno += 1
        print("########################################")
        print("Frame %d" % frameno)
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frameno,
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        # print(vis_util.XDXD)
        # All the results have been drawn on the frame, so it's time to display it.
        # Press 'q' to quit
        ret, jpeg = cv2.imencode('.jpg', frame)
        jp = jpeg.tobytes()
        if cv2.waitKey(1) == ord('q'):
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jp + b'\r\n\r\n')

def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        print("wwhere is my file")
        print(myfile)
        print(myfile.content_type)
        ext = os.path.splitext(myfile.name)[1]
        print(ext)
        if ext in ['.mp4']:
            print(ext)
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            print(filename)
            global uploaded_file_url
            uploaded_file_url = fs.url(filename)
            return render(request, "main/home.html", {
                'uploaded_file_url': uploaded_file_url
            })
        else:
            flag=True
            return render(request, "main/home.html", {
                'flag': flag,'file_type':ext
            })
    return render(request, "main/home.html")

def video_result(request):
    if uploaded_file_url.find("mp4") != -1:
        return StreamingHttpResponse(dis(),
                                     content_type='multipart/x-mixed-replace; boundary=frame')


