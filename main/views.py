from __future__ import unicode_literals

from django.shortcuts import redirect
from django.views.decorators.cache import cache_control
from .models import CompanyInfo, CamerasInfo, UserInfo
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from django.contrib.auth import logout as auth_logout
from models.object_detection.utils import label_map_util
from models.object_detection.utils import visualization_utils as vis_util
from django.urls import resolve, Resolver404
import os
import cv2
import numpy as np
import tensorflow as tf
import http.client
import requests

current_user = None
url_of_cam = None
name_of_cam = None
is_company = True


def index(request):
    global current_user
    global is_company
    if request.method == 'POST':
        if request.POST.get('signup'):
            CompanyInfo.objects.create(name_of_company=request.POST.get('comp_name'),
                                       email_of_company=request.POST.get('comp_email'),
                                       password_of_company=request.POST.get('comb_pass'),
                                       activation_code=request.POST.get('activation_code'))
            current_user = request.POST.get('comp_email')
            return render(request, 'main/home.html', { 'type_user': True})
        elif request.POST.get('signin'):
            username = request.POST.get('username')
            password = request.POST.get('password')
            current_user = username
            print(username)
            try:
                query = CompanyInfo.objects.get(email_of_company=username)
                print(username + "in try")
                is_company = True
            except Exception as e:
                is_company = False
                try:
                    query = UserInfo.objects.get(email_of_user=username)
                    print(username + "in try")
                except Exception as e:
                    return redirect('401')
            if is_company:
                if query.password_of_company == password:
                    request.session['username'] = query.id
                    return redirect('home')
                else:
                    print("Someone tried to login and failed.")
                    return redirect('401')
            else:
                if query.password_of_user == password:
                    request.session['username'] = query.id
                    return redirect('home')
                else:
                    print("Someone tried to login and failed.")
                    return redirect('401')

    elif request.method == 'GET':
        return render(request, 'main/index.html', {})


def page401(request):
    return render(request, "main/page401.html", {})


def home(request):
    if not is_company:
        current_police = UserInfo.objects.get(email_of_user=current_user)
        current_company = current_police.company_id
        typeper = current_police.permission_type
        if typeper == "admin":
            type_user = True
        else:
            type_user = False
    else:
        print(is_company)
        type_user = True
        current_company = CompanyInfo.objects.get(email_of_company=current_user)

    number_of_camera = current_company.number_of_camera
    if number_of_camera >= 1:
        camera = CamerasInfo.objects.filter(company_info_id=current_company, is_running=1).first()
        global url_of_cam
        url_of_cam = camera.ip_of_camera
        global name_of_cam
        name_of_cam = camera.name_of_camera
        print(camera.name_of_camera, camera.ip_of_camera)
        check_status_of_url = is_url_image(url_of_cam)
        if check_status_of_url:
            url_is_exit = True
            return render(request, "main/home.html", {
                'url_is_exit': url_is_exit, 'type_user': type_user
            })
        else:
            flag = True
            return render(request, "main/home.html", {
                'flag': flag, 'URL': url_of_cam, 'type_user': type_user
            })

    flag = False
    return render(request, "main/home.html", {
        'flag': flag, 'type_user': type_user})


def ip_result(request):
    return StreamingHttpResponse(res_ip(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def res_ip():
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    print(CWD_PATH)
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'models/object_detection/inference_graph/frozen_inference_graph.pb')
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'models/object_detection/inference_graph/labelmap.pbtxt')
    # Path to video
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

    print("after ll in dis cam")
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

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
    # Initialize webcam feed
    ###READ FROM THE URL###

    vcap = cv2.VideoCapture(url_of_cam)
    if not vcap.isOpened():
        print("there is no cam")
    else:
        frameno = 0
        while True:
            ret, frame = vcap.read()
            if not ret:
                break
            frameno += 1
            print("########################################")
            print("Frame %d" % frameno)
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
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
            width = 900
            height = 550
            dim = (width, height)
            # resize image
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            ret, jpeg = cv2.imencode('.jpg', resized)
            jp = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jp + b'\r\n\r\n')


def addnewcamera(request):
    if request.method == 'POST':
        if request.POST.get('addnewcamera'):
            ip_camera = request.POST.get('ipofcamera')
            name_camera = request.POST.get('nameofcamera')
            
            if not is_company:
                current_police = UserInfo.objects.get(email_of_user=current_user)
                company = current_police.company_id

            else:
                company = CompanyInfo.objects.get(email_of_company=current_user)

            number_camera = company.number_of_camera

            if number_camera >= 1:
                number_camera += 1
                company.number_of_camera = number_camera
                company.save()
                CamerasInfo.objects.create(name_of_camera=name_camera,
                                           ip_of_camera=ip_camera,
                                           company_info_id=company,
                                           is_running=False,
                                           )

            else:
                number_camera += 1
                company.number_of_camera = number_camera
                company.save()
                CamerasInfo.objects.create(name_of_camera=name_camera,
                                           ip_of_camera=ip_camera,
                                           company_info_id=company,
                                           is_running=True,
                                           )

        return render(request, "main/addnewcamera.html", {'type_user': True})
    else:
        return render(request, "main/addnewcamera.html", {'type_user': True})


@cache_control(no_cache=True, must_revalidate=True)
def logout(request):
    try:
        auth_logout(request)
        del request.session['username']
    except:
        pass
    return render(request, 'main/index.html', {})


def is_url_image(image_url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    try:
        r = requests.head(image_url)
    except:
        return False

    return True


def add_policeman_account(request):
    if request.method == 'POST':
        if request.POST.get('addaccount'):
            user_name = request.POST.get('nameofuser')
            user_email = request.POST.get('email_police')
            company = CompanyInfo.objects.get(email_of_company=current_user)
            answer = request.POST['dropdown']
            password_user = request.POST.get('id_password')
            print(answer)
            UserInfo.objects.create(name_of_user=user_name,
                                    email_of_user=user_email,
                                    company_id=company,
                                    password_of_user=password_user,
                                    permission_type=answer,
                                    )

        return render(request, "main/adduseraccount.html", {'type_user': True})
    else:
        return render(request, "main/adduseraccount.html", {'type_user': True})
