from skimage import io, transform
import scipy.misc
import cv2
import os

out_size = 448
def pre_process(cap, frame_dir):
    idx=0
    while True:
        flag, frame = cap.read()

        if flag:
            print 'Pre procesing frame: {}'.format(idx)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            scaling_factor = out_size / (1.0 * min(h, w))
            print '\tscaling factor: {}'.format(scaling_factor)
            cropped = transform.rescale(frame, scaling_factor)
            print '\tcropped shape: ',
            print cropped.shape

            h, w, _ = cropped.shape
            diff = max(cropped.shape[:2]) - out_size
            if w > h:
                cropped = cropped[:,diff/2:w-(diff-diff/2)]
            else:
                cropped = cropped[diff/2:h-(diff-diff/2),:]
            
            # save the frame to a directory
            print '\tFinal shape: (%d, %d)' % (cropped.shape[0], cropped.shape[1])
            scipy.misc.imsave(os.path.join(frame_dir, '{}.jpeg'.format(idx)), cropped)
            idx += 1
        else:
            print 'Frame not yet ready!'
    print 'Total frames pre-processed: {}'.format(idx)
video_dir = '/home/kv/Git/Auto_price_tag/Dataset/videos/'
frame_dir = '/home/kv/Git/Auto_price_tag/Dataset/frames/'

cap = cv2.VideoCapture(os.path.join(os.path.join(video_dir, '001'), "001.MOV"))
pre_process(cap, os.path.join(frame_dir, '001'))