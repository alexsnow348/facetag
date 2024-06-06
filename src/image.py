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
import sys
import time
import pickle
import argparse


def main(args):
    img_path = args.img_filename

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
            frame_interval = 3
            count = 0
            batch_size = 1000
            image_size = 182
            input_image_size = 160
            recognized_pp = []

            print('Loading feature extraction model')
            # modeldir = './epcho1/20190130-091944'
            modeldir = args.modeldir
            print(modeldir)
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # classifier_filename = './epcho1/my_classifier.pkl'
            classifier_filename = args.classifier_filename
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)

            c = 0

            print('Start Recognition!')
            print("IMG PATH : " + img_path)
            prevTime = 0
            frame = cv2.imread(img_path, 0)
            frame_copy = cv2.imread(img_path)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)
            frame_copy = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(
                    frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Face Detected: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []  # to an window just select "General Appearance > Icon File". Problematic here is that Glade only shows image files locate
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
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
                        # print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)

                        best_class_probabilities = predictions[np.arange(
                            len(best_class_indices)), best_class_indices]

                        result_names = class_names[best_class_indices[0]]

                        if best_class_probabilities[0] > 0.15 and result_names != 'R99999 Unk':

                            cv2.rectangle(frame_copy, (bb[i][0], bb[i][1]), (bb[i][2],
                                                                             bb[i][3]), (0, 255, 0), 2)  # boxing face

                            # plot result idx under box DETECTED FACES
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            result_names = class_names[best_class_indices[0]]
                            # print(result_names)
                            recognized_pp.append(result_names.split(' ')[1])
                            count += 1
                            cv2.putText(frame_copy, result_names.split(' ')[1], (text_x, text_y),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 125, 255),
                                        thickness=2, lineType=2)

                else:
                    print('Unable to align')
                if frame_copy.shape[0] < 1000:
                    frame_copy = cv2.resize(frame_copy, (0, 0), fx=2.5, fy=2.5)
                people = ', '.join(recognized_pp)
                msg = str(count) + ' people(s) recognized out of ' + \
                    str(nrof_faces)+' detected people.'
                cv2.putText(frame_copy, msg, (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, 3)
                if len(recognized_pp) < 8:
                    cv2.putText(frame_copy, people, (10, 100),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, 3)
                cv2.imwrite('./result_images/'+args.save_filename+'.jpg', frame_copy)
                print("Result: " + msg)
                people_n = '\n'.join(recognized_pp)
                print("Name: " + people_n)
                cv2.imshow('Image', frame_copy)

            if cv2.waitKey(1000000) & 0xFF == ord('q'):
                sys.exit("Thanks")
            cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_filename', type=str,
                        help='File containing image', default='')
    parser.add_argument('--modeldir', type=str,
                        help='Path to the model directory.',
                        default='~/Desktop/clean/trained_models/opensource/20180402-114759')
    parser.add_argument('--classifier_filename', type=str,
                        help='Path to the classifier file directory.',
                        default='~/Desktop/clean/trained_models/opensource/opensource.pkl')
    parser.add_argument('--save_filename', type=str,
                        help='Path to the classifier file directory.',
                        default='test')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
