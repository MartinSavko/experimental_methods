#!/usr/bin/env python

import cv2

def main():
    capture = cv2.VideoCapture('http://172.19.10.149/axis-cgi/mjpg/video.cgi')
    k=0
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #video_writer = cv2.VideoWriter("output_cam6.avi", fourcc, 20, (768, 576))
    while (capture.isOpened()):
        ret, frame = capture.read()
        k +=1
        print(k)
        cv2.imwrite('frame_%d.jpg' %k, frame)
        if ret:
            #video_writer.write(frame)
            cv2.imshow('Video Stream cam 6', frame)
        else:
            break

    capture.release()
    #video_writer.release()
    cv2.destroyAlWindows()


if __name__ == "__main__":
    main()


