from operator import le
from string import printable
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

def main():
    capture = cv.VideoCapture(0,cv.CAP_DSHOW)
    kernel_e = np.ones((4, 4), dtype=np.uint8)
    kernel_d = np.ones((4, 4), dtype=np.uint8)

    Points0 = []
    Points1 = []
    Points2 = []
    frame_num = 0
    init = True
    while True:
        frame_num = frame_num + 1
        ret, frame = capture.read()
        if frame is None:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 65, 10)
        # opening = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_e, 5)

        dilate = cv.dilate(binary, kernel_d, iterations=1)
        erosion = cv.erode(dilate, kernel_e, iterations=2)
        contours, hierarchy = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # print(type(contours))

        area = []
        coordinate = []
        for i in range(len(contours)):
            a = cv.contourArea(contours[i])
            print(a)
            if a < 50:
                continue
            area.append(abs(a))
            M = cv.moments(contours[i])
            cx = M['m10'] / (M['m00'] + 10e-5)
            cy = M['m01'] / (M['m00'] + 10e-5)
            coordinate.append([int(cx), int(cy)])
            # temp = 1.00
            # print(type(cx),type(temp))
            cv.circle(frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)
            cv.drawContours(frame, contours, i, (0, 0, 255), 3)

        # print(coordinate)
        # print(len(area))
        if len(coordinate) == 38:
            if init == True:
                Points0 = coordinate.copy()
                Points1 = coordinate.copy()
            Points0 = Points1.copy()
            Points1 = coordinate.copy()
            # print(len(Points0))
            # print(len(Points1))
            # print(Points1[0])
            # for i in range(len(coordinate)):
                # cv.line(frame, Points0[i], Points1[i], (153, 214, 188), 2, 1)
        cost = np.ones([len(Points0),len(Points1)])
        for i in range(len(Points0)):
            for j in range(len(Points1)):
                cost[i,j] = np.sqrt((Points1[j][0]-Points0[i][0])**2 +(Points1[j][1]-Points0[i][1])**2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for x,y in zip(row_ind,col_ind):
            cv.line(frame, Points0[x], Points1[y], (0, 214, 255), 2, 1)

        print([(x, y) for x, y in zip(row_ind, col_ind)])
        '''cv.imshow('FrameB', frame[:,:,0])
        cv.imshow('FrameG', fqrame[:,:,1])
        cv.imshow('FrameR', frame[:,:,2])
        cv.imshow('Frame', frame)'''
        cv.imshow('Frame', frame)
        cv.imshow('Gray', gray)
        cv.imshow('binary', erosion)
        ## [show]
        init = False
        keyboard = cv.waitKey(30)
        if keyboard == 27:
            break


if __name__ == "__main__":
    main()
