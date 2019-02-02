from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
from align import detect_face
import os
import time
import pickle
import argparse
import sys
import Face
import statistics
from collections import Counter


def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list)  # use the max value here
    return max_mode


def main(args):

    videoLink = args.video_link
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = args.frame_interval
            batch_size = 1000
            image_size = 182
            input_image_size = 160
            max_age = args.max_age

            print('Loading feature extraction model')
            modeldir = args.modeldir
            debug = args.debug
            print("Debug: ", debug)
            if debug == 'True':
                debug = True
            else:
                debug = False
            if debug:
                print("videoLink: ", args.video_link)
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename = args.classifier_filename

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)

            # video_capture = cv2.VideoCapture(0) #webcam
            video_capture = cv2.VideoCapture(args.video_link)
            c = 0
            fid = 0
            faces = []
            target_distance = args.target_distance

            print('Start Recognition!')
            prevTime = 0
            while True:
                ret, frame = video_capture.read()

                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

                curTime = time.time()+1    # calc fps
                timeF = frame_interval
                new = True
                show = False
                for i in faces:
                    i.age_one()
                if (c % timeF == 0):

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(
                        frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                if debug:
                                    print('face is inner of range!')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            try:
                                cropped[i] = facenet.flip(cropped[i], False)
                            except:
                                continue
                            if debug:
                                print('Processing Status: PROCESSING FRAME')
                            scaled.append(misc.imresize(
                                cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(
                                scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {
                                images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)

                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(
                                len(best_class_indices)), best_class_indices]

                            # plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            if debug:
                                print('frame_interval: ', frame_interval)
                            # track faces
                            result_names = class_names[best_class_indices[0]]

                            for k in faces:
                                # print(best_class_probabilities[0])
                                if abs(bb[i][0]-k.getX()) <= target_distance\
                                        and abs(bb[i][1] - k.getY())\
                                        <= target_distance and k.getDone() is False:
                                    if debug:
                                        print(k.getAge(), 'X Diff: ', abs(bb[i][0]-k.getX()),
                                              'Y Diff: ', abs(bb[i][1] - k.getY()))
                                    new = False
                                    if best_class_probabilities[0] > 0.20:
                                        k.updateCoords(bb[i][0], bb[i][1])
                                        k.updateConfidence(best_class_probabilities[0])
                                        result_names = class_names[best_class_indices[0]]
                                        k.updateStaffID(result_names.split(' ')[0])
                                        k.updateName(result_names.split(' ')[1])

                                    if k.getAge() > 1:
                                        show = True

                                    color = k.getRGB()
                                    counter = Counter(k.getName())
                                    most_common = counter.most_common()
                                    if debug:
                                        print('Show: ', show)
                                        print(most_common)

                                    if show:
                                        if len(most_common) >= 2:
                                            f_n, f_v = most_common[0]
                                            s_n, s_v = most_common[1]
                                            if f_n != 'Unk':
                                                name_to_show = f_n
                                                # name_to_show = name_mode
                                            else:
                                                name_to_show = s_n
                                        if len(most_common) == 1:
                                            f_n, f_v = most_common[0]
                                            name_to_show = f_n
                                    # print(name_to_show)
                            if new:
                                f = Face.MyFace(fid, bb[i][0],
                                                bb[i][1], max_age)
                                f.updateConfidence(best_class_probabilities[0])
                                result_names = class_names[best_class_indices[0]]
                                f.updateStaffID(result_names.split(' ')[0])
                                name = result_names.split(' ')[1]
                                f.updateName(name)
                                color = f.getRGB()
                                faces.append(f)
                                fid += 1
                                name_to_show = ''

                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2],
                                                                        bb[i][3]), color, 2)  # boxing face
                            if name_to_show == 'Unk':
                                name_to_show = 'Unknown'
                            if debug:
                                print('Detected As: ', name_to_show)
                            cv2.putText(frame, name_to_show, (text_x, text_y),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color,
                                        thickness=1, lineType=2)
                    else:
                        if debug:
                            print('Unable to align')
                else:
                    if debug:
                        print('Processing Status: NOT PROCESSING FRAME')
                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(
                        frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                if debug:
                                    print('face is inner of range!')
                                continue
                            for k in faces:
                                # print(best_class_probabilities[0])
                                if abs(bb[i][0]-k.getX()) <= target_distance\
                                        and abs(bb[i][1] - k.getY())\
                                        <= target_distance and k.getDone() is False:
                                    if debug:
                                        print(k.getAge(), 'X Diff: ', abs(bb[i][0]-k.getX()),
                                              'Y Diff: ', abs(bb[i][1] - k.getY()))
                                    if k.getAge() > 1:
                                        show = True

                                    color = k.getRGB()
                                    counter = Counter(k.getName())
                                    most_common = counter.most_common()
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    if debug:
                                        print('Show: ', show)
                                        print(most_common)

                                    if show:
                                        if len(most_common) >= 2:
                                            f_n, f_v = most_common[0]
                                            s_n, s_v = most_common[1]
                                            if f_n != 'Unk':
                                                name_to_show = f_n
                                                # name_to_show = name_mode
                                            else:
                                                name_to_show = s_n
                                        elif len(most_common) == 1:
                                            f_n, f_v = most_common[0]
                                            name_to_show = f_n
                                        else:
                                            name_to_show = 'Unknown'

                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2],
                                                                                    bb[i][3]), color, 2)  # boxing face
                                        if name_to_show == 'Unk':
                                            name_to_show = 'Unknown'
                                        if debug:
                                            print('Detected As: ', name_to_show)
                                        cv2.putText(frame, name_to_show, (text_x, text_y),
                                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color,
                                                    thickness=1, lineType=2)
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                str = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, str, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                c += 1
                if frame.shape[0] < 1000:
                    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            # #video writer
            # out.release()
            cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_link', type=int,
                        help='Video info', default=0)
    parser.add_argument('--target_distance', type=int,
                        help='Video info', default=5)
    parser.add_argument('--max_age', type=int,
                        help='Max age to die', default=60)
    parser.add_argument('--frame_interval', type=int,
                        help='Frame interval to process', default=10)
    parser.add_argument('--modeldir', type=str,
                        help='Path to the model directory.',
                        default='~/Desktop/clean/trained_models/opensource/20180402-114759')
    parser.add_argument('--classifier_filename', type=str,
                        help='Path to the classifier file directory.',
                        default='~/Desktop/clean/trained_models/opensource/opensource.pkl')
    parser.add_argument('--debug',  type=str,
                        help='True or False',
                        default='False')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
