from __future__ import print_function
import os
import argparse
import torch
from enum import Enum
from skimage import io
from skimage import color
import numpy as np
import cv2

import os.path as osp
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from models import FAN, ResNetDepth
import models

from utils import *
#from models.tiramisu import FCDenseNet57
from models.DCN import FCDenseNet57, FCDenseNet67, FCDenseNet103
import warnings
warnings.filterwarnings('ignore')

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

class FaceAlignment:
    def __init__(self, 
                modelfilename="checkpoint.pth.tar",
                nStack=4,
                device='cuda',
                flip_input=False,
                face_detector='sfd',
                facedetectmodelfile="s3fd_convert.pth",
                verbose=False):

        self.device = device
        self.flip_input = flip_input
        self.verbose = verbose

        if not os.path.isfile(facedetectmodelfile):
            print("The face detection CNN model [%s] not exists." % facedetectmodelfile)
            return None

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, path_to_detector=facedetectmodelfile, verbose=verbose)


        # Initialise the face alignemnt networks
        #self.face_alignment_net = nn.DataParallel(model1)
        self.face_alignment_net = nn.DataParallel(FCDenseNet103(n_classes=68))
        #A = FCDenseNet57(n_classes=68)
        
        model_path = modelfilename

        if not os.path.isfile(model_path):
            print("model:%s not exists." % model_path)
            return None
        fan_weights = torch.load(
            model_path,
            map_location=lambda storage,
            loc: storage)
        #for k,v in fan_weights['state_dict'].items():
        #    print(k)

        self.face_alignment_net.load_state_dict(fan_weights['state_dict'])
        self.face_alignment_net = self.face_alignment_net.module

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces)

    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        reference_scale = 200
        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image[..., ::-1].copy())
            reference_scale = self.face_detector.reference_scale

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        torch.set_grad_enabled(False)
        landmarks = []
        landmarks_in_crops = []
        img_crops = []
        print('detected_faces num:{}'.format(len(detected_faces)))
        print('detected_faces:{}'.format(detected_faces))

        image = im_to_torch(image)
        
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] -
                 (d[3] - d[1]) / 2.0])
            center[1] = center[1] + (d[3] - d[1]) * 0.12
            hw = max(d[2] - d[0], d[3] - d[1])
            scale_x = float(hw / reference_scale)
            scale_y = float(hw / reference_scale)

            inp = crop(image, center, [scale_x, scale_y], reference_scale)

            io.imsave('crop_%s.jpg' % i,im_to_numpy(inp))
            img_crops.append(im_to_numpy(inp))

            inp = inp.to(self.device)
            inp.unsqueeze_(0)

            out = self.face_alignment_net(inp)[-1].detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp))
                            [-1].detach(), is_label=True)
            out = out.cpu()

            pts, pts_img = get_preds_fromhm(out, [center], [[scale_x, scale_y]], [reference_scale])
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            landmarks.append(pts_img.numpy())
            landmarks_in_crops.append(pts.numpy())

        return landmarks, detected_faces, landmarks_in_crops, img_crops

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True):
        detected_faces = self.face_detector.detect_from_directory(path, extensions, recursive, show_progress_bar)

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            preds, detected_faces = self.get_landmarks_from_image(image, bounding_boxes)
            predictions[image_path] = preds

        return predictions

if __name__ == '__main__':
    P = argparse.ArgumentParser(description='Predict network script')
    P.add_argument('--modelfile', type=str, default = 'checkpoint/Test_image/300W-LP/model_best.pth.tar' , help='model file path')
    P.add_argument('--detectmodelfile', type=str, default = 'detection/s3fd_convert.pth', help='face detect model file')
    mg = 'LFPW_image_test_0179_0'
    #P.add_argument('--input', type=str, default = 'test/Old/'+mg+'.jpg' , help='input image file')
    #P.add_argument('--input', type=str, default = 'data/300W/helen/testset/'+ mg + '.jpg' , help='input image file')
    P.add_argument('--input', type=str, default = 'data/300W_LP/LFPW/'+ mg + '.jpg' , help='input image file')
    args = P.parse_args()
    fa = FaceAlignment(modelfilename=args.modelfile, facedetectmodelfile=args.detectmodelfile)

    #path = 'data/300W'
    #main_pts = load(self.anno[idx][:-3] + 'pts')
    if fa:
        img_in = io.imread(args.input)
        img = img_in
        preds, detected_faces, preds_in_crops, img_crops = fa.get_landmarks(img)
        #print(preds)
        pts68_all_re = _load(osp.join('data', 'AFLW2000-3D-Reannotated.pts68.npy'))
        roi_boxs = _load(osp.join('data', 'AFLW2000-3D_crop.roi_box.npy'))
        g = 49
        sx, sy, ex,ey = roi_boxs[g]
        #print(sx)
        _pts = pts68_all_re[g]
        _pts0 = _pts[0:2, :].transpose()
        _pts1 = torch.from_numpy(_pts0)
        #_pts = to_numpy(_pts)
        _pts = [_pts1]
        pts68_fit = np.asarray(preds)
        
        dis = pts68_fit[0] - _pts0
        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        
        #print(pts68_fit, _pts1, dis)
        #print(len(pts68_fit), len(_pts1),np.mean(dis))
        #print(type(_pts))
        #print(_pts, type(preds))

        for k,d in enumerate(detected_faces):
            #cv2.rectangle(img_in,(d[0],d[1]),(d[2],d[3]),(255,255,255))
            landmark = preds[k]
            #landmark1 = _pts[k]
            for i in range(landmark.shape[0]):
                pts = landmark[i]
                #pts_1 = landmark1[i]
                #print(pts)
                cv2.circle(img_in, (pts[0], pts[1]),1,(0,255,0), 2, 8)
                #cv2.circle(img_in, (pts_1[0], pts_1[1]),1,(0,0,255), 2, 8)
                #cv2.putText(img_in,str(i),(pts[0],pts[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,2555,255))
        io.imsave('atestimage/'+mg+'8.jpg',img_in)
    else:
        print("FaceAlignment init error!")
