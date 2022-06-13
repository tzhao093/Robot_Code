# Dependent libraries

import Lib

# Importing utility libraries

import numpy as np
import tensorflow as tf
import cv2
import math
import time
import random
import msvcrt
import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import shutil
from utils import label_map_util

# Importing XBee libraries for wireless communication with Alphabot2 robots

from digi.xbee.devices import XBeeDevice
from digi.xbee.devices import RemoteXBeeDevice
from digi.xbee.devices import XBee64BitAddress

# Defining test configuration, if true then code is run on sample video for testing purposes

test = False

# Defining risk density option and parameters

option = 3 # Define which area coverage scenario is being tested (possible values include 1, 2 or 3)
alpha = 1
beta = 0.0001

assert option == 1 or option == 2 or option == 3

# Ensure that data collection directory exists, reset data collection for new run

if not os.path.exists("data"):
    os.mkdir("data")

if not os.path.exists("data/option{}".format(option)):
    os.mkdir("data/option{}".format(option))

if os.path.exists("data/option{}/frames".format(option)):
    shutil.rmtree('data/option{}/frames'.format(option))

if os.path.exists("data/option{}/plots".format(option)):
    shutil.rmtree('data/option{}/plots'.format(option))

os.mkdir('data/option{}/frames'.format(option))
os.mkdir('data/option{}/plots'.format(option))

# Defining paths to label map and frozen inference graph for object detection

PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'

# Variable assignment for pre-initialization phase

iterator = 0

if not test:

    # Define Controller Device

    device = XBeeDevice("COM9", 9600) # Pick your specific port
    print("Hello")

    # Open Controller Device

    device.open()
    print("Device opened")

    # Define Remote Devices

    remote_devices = []

    #remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A20040D835D1")))
    #remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200418FE7A3")))
    #remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200415E8441")))
    #remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A2004190B158")))
    remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200417CA46A")))

else:
    remote_devices = [[0], [1]]

# Variable assignment for initialization phase

positionInit = [] # Initialization position variable
changePosition = []
distanceInit = [] # Initialization distance variable
idOrder = [] # For matching XBee id to object detection id
orientationInit = [] # Initialization orientation variable

# Variable assignment for main run loop phase

iterations = 0
frame_count = 0
N = []
k= []
for agent in range(0, len(remote_devices)):
    k.append(False)
print (k)
b=True
j=0
s=0
for agent in range(0, len(remote_devices)):
    N.append(s)
    s+=6
timestamp = 0
trueCount = 0
loopMaster = True



edgeTimer = []
timer = []

pass_var = [] # Indicates if this agent should be passed this iteration
isSuccess = [] # Indicates if desired position has been achieved
isEdge = [] # Indicates if agent is near an edge
reset = []
distanceDes = [] # Real time distance between agent and desired position

currentPosition = [] # Current position of agent
prevPosition = [] # Previous position of agent
timestepDistance = [] # Distance between current position and previous position of agent

orientation = [] # Current orientation of each agent
orientationDes = [] # Current orientation of each agent
angle = [] # Angle from current orientation, to optimal orientation
messageArray = [] # The message to be sent to each agent

distanceGraph = []
timeGraph = []

def store_miniCs():
	x = 320
	y = 240
	r = 140
	angle1 = 3*math.pi/2
	w, h = 2, 60
	circle_list = [[0 for x in range(w)] for y in range(h)]
	for i in range(2):
		circle_list [i][0] = x + r*math.cos(angle1)
		circle_list [i][1] = y + r*math.sin(angle1)
		angle1 += math.pi
	return circle_list

def create_miniCs():
    x = 320
    y = 240
    r = 140
    angle1 = 3 * math.pi/2
    w, h = 2, 60
    circle_list = [[0 for x in range(w)] for y in range(h)]
    for i in range(2):
        circle_list [i][0] = x + r*math.cos(angle1)
        circle_list [i][1] = y + r*math.sin(angle1)
        angle1 += math.pi
    for i in range(2):
        cv2.circle(frame, (int(circle_list[i][0]), int(circle_list[i][1])), 4, (255, 0 , 0), -1)
        cv2.circle(frame, (int(circle_list[i][0]), int(circle_list[i][1])), 30, (255, 0 , 0), 1)

def animate(x, y):
    plt.cla()
    plt.plot(x, y, label = 'Robot 1')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

for i in range(len(remote_devices)):

    # Initialization variable assignment

    positionInit.append(None) # Initialization position variable
    changePosition.append(None)
    distanceInit.append(0.0) # Initialization distance variable
    idOrder.append(None) # For matching XBee id to object detection id
    orientationInit.append(None) # Initialization orientation variable

    # Run loop variable assignment

    edgeTimer.append(time.time())
    timer.append(time.time())

    pass_var.append(False) # Indicates if this agent should be passed this iteration
    isSuccess.append(False) # Indicates if desired position has been achieved
    isEdge.append(False) # Indicates if agent is near an edge
    reset.append(True)
    distanceDes.append(None) # Real time distance between agent and desired position

    currentPosition.append(None) # Current position of agent
    prevPosition.append(None) # Previous position of agent
    timestepDistance.append(None) # Distance between current position and previous position of agent

    orientation.append(None) # Current orientation of each agent
    orientationDes.append(None) # Current orientation of each agent
    angle.append(None) # Angle from current orientation, to optimal orientation
    messageArray.append(None) # The message to be sent to each agent

# Define centroid tracker object

tracker = Lib.Tracker()

# Define object for Voronoi diagram and Lloyds algorithm calculations

voronoi = Lib.Voronoi()

# Define object for utility functions

function = Lib.Function()

# Define frozen inference graph from path

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Define label map from path

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Start video capture from webcam, or from "Test.mp4" video is testing code

if test:
    vidcap = cv2.VideoCapture("Test.mp4")
else:
    vidcap = cv2.VideoCapture(1)

success, frame = vidcap.read()

# Define the discretized space

width_sections = 64
height_sections = 48

xdim = np.linspace(5, frame.shape[1] - 5, width_sections)
ydim = np.linspace(5, frame.shape[0] - 5, height_sections)
X, Y = np.meshgrid(xdim,ydim)

# Define map of risk density based on selected option

if option == 1:
    risk = np.ones((height_sections, width_sections))

elif option == 2:
    target = [frame.shape[1]/2, frame.shape[0]/2]
    risk = voronoi.risk_density(xdim, ydim, target, alpha=alpha, beta=beta)
    print(target)

elif option == 3:
    target = [485, 320]
    risk = voronoi.risk_density(xdim, ydim, target, alpha=alpha, beta=beta)
    print(target)

if not test:
    print("I'm here")

    # Start pre-initialization loop

    while True:
        success, frame = vidcap.read()

        if (success):
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

        if msvcrt.kbhit(): # Record user input from keyboard

            key = msvcrt.getch()
            print(key)

            if key == b'w':
                message = "P,F>" # If you press "w" the robot will go forward

            if key == b'a':
                message = "P,90>" # If you press "a" the robot will rotate 90 degrees counter-clockwise

            if key == b's':
                message = "P,B>"  # If you press "s" the robot will go backwards

            if key == b'd':
                message = "P,270>" # If you press "d" the robot will rotate 90 degrees clockwise

            if key == b'z':
                message = "P,S>" # If you press "z" the robot will stand still

            if key == b'q':
                iterator = iterator - 1 # If you press "q", you will take control of the previous robot
                message = "P,S>"

                if (iterator < 0):
                    iterator = 0

            if key == b'e':
                iterator = iterator + 1 # If you press "e", you will take control of the next robot
                message = "P,S>"

            if key == b'x': # If you press "x" the pre-initialization loop will end
                break

            if (iterator > (len(remote_devices) - 1)):
                break

            device.send_data_async(remote_devices[iterator], message)
            print(message)

# Start object detection

with detection_graph.as_default() as graph:
    with tf.compat.v1.Session() as sess:

            if not test:

    # Start initialization loop

                while True:

                    for i in range(0, len(remote_devices)):

                          success, frame = vidcap.read()

                          frame_expanded = np.expand_dims(frame, axis=0)

                          output_dict = function.run_inference_for_single_image(frame_expanded, graph, sess) # Detect Alphabot2 in the frame

                          rects = []

                          # Draw rectangles around all Alphabot2 robots detected

                          for j in range(0, len(remote_devices)):
                            if output_dict['detection_scores'][j] > 0.85:

                                  (startY, startX, endY, endX) = output_dict['detection_boxes'][j]

                                  rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                  cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                     (0, 255, 0), 1)

                          objects = tracker.update(rects) # Begin tracking all Alphabot2 robots that are detected

                          for (objectID, centroid) in objects.items():
                              text = "ID {}".format(objectID)
                              cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                              cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                              positionInit[objectID] = centroid

                          cv2.imshow("Frame", frame)
                          cv2.waitKey(1)

                          device.send_data_async(remote_devices[i], "P,F>") # Send command for robot to move forward

                          for j in range(0, 35):
                              success, frame = vidcap.read()

                              frame_expanded = np.expand_dims(frame, axis=0)

                              output_dict = function.run_inference_for_single_image(frame_expanded, graph, sess)

                              rects = []

                              for k in range(0, len(remote_devices)):
                                if output_dict['detection_scores'][k] > 0.85:

                                      (startY, startX, endY, endX) = output_dict['detection_boxes'][k]

                                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                              objects = tracker.update(rects)

                              for (objectID, centroid) in objects.items():
                                  text = "ID {}".format(objectID)
                                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                                  changePosition[objectID] = centroid
                                  distanceInit[objectID] = function.get_distance(point1=positionInit[objectID], point2=changePosition[objectID])

                              cv2.imshow("Frame", frame)
                              cv2.waitKey(1)

                          idOrder[i] = np.argmax(distanceInit) # Record ID of robot that moved forward

                          device.send_data_async(remote_devices[i], "P,B>")

                          for j in range(0, 35):
                              success, frame = vidcap.read()

                              frame_expanded = np.expand_dims(frame, axis=0)

                              output_dict = function.run_inference_for_single_image(frame_expanded, graph, sess)

                              rects = []

                              for k in range(0, len(remote_devices)):
                                if output_dict['detection_scores'][k] > 0.85:

                                      (startY, startX, endY, endX) = output_dict['detection_boxes'][k]

                                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                              objects = tracker.update(rects)

                              for (objectID, centroid) in objects.items():
                                  text = "ID {}".format(objectID)
                                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                                  positionInit[objectID] = centroid

                              cv2.imshow("Frame", frame)
                              cv2.waitKey(1)

                          orientationInit[idOrder[i]] = function.get_orientation(point1=positionInit[idOrder[i]], point2=changePosition[idOrder[i]]) # Record initial orientation of robot

                          device.send_data_async(remote_devices[i], "I," + str(orientationInit[idOrder[i]]) + ">")


    # Check user input

                    print(idOrder)

                    test1 = function.test_1(idArray=idOrder, numRobots=len(remote_devices)) # Test 1 that all robots were detected and initialized correctly

                    test2 = function.test_2(idArray=idOrder, numRobots=len(remote_devices)) # Test 2 that all robots were detected and initialized correctly

                    if test1:
                        print("Test 1 passed.")

                    if not test1:
                        print("Test 1 failed.")

                    if test2:
                        print("Test 2 passed.")

                    if  not test2:
                        print("Test 2 failed.")

                    success, frame = vidcap.read()

                    frame_expanded = np.expand_dims(frame, axis=0)

                    output_dict = function.run_inference_for_single_image(frame_expanded, graph, sess)

                    rects = []

                    for i in range(0, len(remote_devices)):
                        if output_dict['detection_scores'][i] > 0.85:

                            (startY, startX, endY, endX) = output_dict['detection_boxes'][i]

                            rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                            cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                    objects = tracker.update(rects)

                    for (objectID, centroid) in objects.items():
                            text = "ID {}".format(objectID)
                            cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, str(orientationInit[objectID]), (centroid[0] - 15, centroid[1] - 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                            cv2.arrowedLine(frame, (centroid[0], centroid[1]), (changePosition[objectID][0], changePosition[objectID][1]), (255, 0, 0), 1)
                            cv2.line(frame, (centroid[0], centroid[1]), (centroid[0] + 30, centroid[1]), (255, 0, 0), 1)

                    cv2.imshow("Frame", frame)
                    cv2.waitKey(1)

                    inputVar = input("Are you satisfied with the initialization? (y/n)") # Ask for user input that initialization results make sense

                    if (inputVar == 'y'):
                        break

                    if cv2.waitKey(1) & 0xFF == ord("x"):
                        break

# Start run loop

            idOrder_array = np.array(idOrder)

            data_txt = open("data/option{}/data.txt".format(option), "w") # Begin saving data to text file

            while (loopMaster == True):

              success, frame = vidcap.read()

              frame_expanded = np.expand_dims(frame, axis=0)

              output_dict = function.run_inference_for_single_image(frame_expanded, graph, sess)

              rects = []

              for i in range(0, len(remote_devices)):
                if output_dict['detection_scores'][i] > 0.85:

                      (startY, startX, endY, endX) = output_dict['detection_boxes'][i]

                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                         (0, 255, 0), 2)

              objects = tracker.update(rects)

              agentPosition = []

              for (objectID, centroid) in objects.items():
                  text = "ID {}".format(objectID)
                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                  agentPosition.append([centroid[0], centroid[1]])


# Get desired location of each robot
              #plt.plot(agentPosition[0][0], agentPosition[0][1])
              create_miniCs()
              #distance_update, voronoi_update, index_update = voronoi.voronoi(agentPosition=agentPosition, numAgents=len(remote_devices), xdim=xdim, ydim=ydim) # Calculate Voronoi diagram given agent positions

              if np.sum(isSuccess) == len(remote_devices) or np.sum(reset) == len(remote_devices): # If all robots have arrived at desired location, or this is the first run through of the loop

                  iterations = iterations + 1

                  desPosition = store_miniCs()




                  #for n in range(len(remote_devices)):
                      #desPosition.append(voronoi.centroid(voronoi_update[n], index_update[n], risk, agentPosition[n])) # Find centroid of each Voronoi partition
                      #isSuccess[n] = False

              coverage = 0


              for i in range(len(remote_devices)):
                  cv2.circle(frame, (int(desPosition[N[i]][0]), int(desPosition[N[i]][1])), 4, (0, 165, 255), -1)
                  cv2.circle(frame, (int(desPosition[N[i]][0]), int(desPosition[N[i]][1])), 30, (0, 165, 255), 1)
                  #coverage += voronoi.coverage_metric(agentPosition[i], voronoi_update[i], index_update[i], risk) # Calculate coverage metric at current time

              # Write data to text file

              if np.sum(reset) == len(remote_devices):
                  start_time = time.time()
                  data_txt.write("Time: {}, Coverage Metric: {}, Iteration: {}, Agent Positions: {}, Desired Positions: {}".format(0, coverage, iterations, agentPosition, desPosition))
              else:
                  data_txt.write("Time: {}, Coverage Metric: {}, Iteration: {}, Agent Positions: {}, Desired Positions: {}".format(time.time() - start_time, coverage, iterations, agentPosition, desPosition))
              data_txt.write("\n")

              for agent in range(len(remote_devices)):

    # Check distance from desired location

                  distanceDes[agent] = function.get_distance(point1=desPosition[N[agent]], point2=agentPosition[agent]) # Check distance from an agent to its desired position
                  if (trueCount==1):
                      distanceGraph.append(distanceDes[agent]/2)
                      if (timestamp==0):
                          t0 = time.time()
                          timestamp+=1

                      t1 = time.time()
                      timetotal = t1-t0
                      timeGraph.append(timetotal)

                  if (distanceDes[agent] > 20.0): # If distance is more than 30 pixels, then the agent has not arrived
                            isSuccess[agent] = False

                  if (distanceDes[agent] <= 20.0 and isSuccess[agent] == False): # If distance is less than 30 pixels, then the agent has arrived
                            #print(k)
                            #k[agent] = True
                            #print (k)
                            if (reset[agent] == True) and not test:
                                index = np.where(idOrder_array==agent)
                                device.send_data_async(remote_devices[index[0][0]], "A," + str(orientationInit[agent]) + ">")
                            if (reset[agent] == False) and not test:
                                index = np.where(idOrder_array==agent)
                                device.send_data_async(remote_devices[index[0][0]], "A," + str(orientation[agent]) + ">")

                            isSuccess[agent] = True
                            trueCount += 1
                            pass_var[agent] = True
                            """if j <=len(remote_devices):
                                if k[0] == True and k[1] == True:
                                    j+=3
                            else:
                                N[agent]+=1

                            if N[agent]>=30:
                                N[agent]=0"""


                            """ani = FuncAnimation(plt.gcf(), animate(agentPosition[0][0], agentPosition[0][1]), interval = 1000)
                            plt.tight_layout()
                            plt.show()"""



    # Check edge conditions, if robot is at the edge of the boundary then a control action is sent to make it turn around

                  if (((time.time() - edgeTimer[agent]) >= 5) and ((agentPosition[agent][0] <= frame.shape[1]*0.01) or (agentPosition[agent][0] >= frame.shape[1]*0.99) or (agentPosition[agent][1] <= frame.shape[0]*0.01) or (agentPosition[agent][1] >= frame.shape[0]*0.99))):
                      isEdge[agent] = True
                      timer[agent] = time.time() + 5
                      edgeTimer[agent] = time.time()

                      currentPosition[agent] = agentPosition[agent]
                      orientationDes[agent] = function.get_orientation(currentPosition[agent], desPosition[agent])

                      if (agentPosition[agent][0] <= frame.shape[1]*0.01) and not test:
                          index = np.where(idOrder_array==agent)
                          device.send_data_async(remote_devices[remote_devices[index[0][0]]], "E,L," + str(orientationDes[agent]) + ">")

                      if (agentPosition[agent][0] >= frame.shape[1]*0.99) and not test:
                          index = np.where(idOrder_array==agent)
                          device.send_data_async(remote_devices[remote_devices[index[0][0]]], "E,R," + str(orientationDes[agent]) + ">")

                      if (agentPosition[agent][1] <= frame.shape[0]*0.01) and not test and not (agentPosition[agent][0] <= frame.shape[1]*0.01 or agentPosition[agent][0] >= frame.shape[1]*0.99):
                          index = np.where(idOrder_array==agent)
                          device.send_data_async(remote_devices[remote_devices[index[0][0]]], "E,U," + str(orientationDes[agent]) + ">")

                      if (agentPosition[agent][1] >= frame.shape[0]*0.99) and not test and not (agentPosition[agent][0] <= frame.shape[1]*0.01 or agentPosition[agent][0] >= frame.shape[1]*0.99):
                          index = np.where(idOrder_array==agent)
                          device.send_data_async(remote_devices[remote_devices[index[0][0]]], "E,D," + str(orientationDes[agent]) + ">")

    # Check time conditions

                  if ((((time.time() - timer[agent]) >= 5) or (reset[agent])) and not isSuccess[agent] and not isEdge[agent] and not pass_var[agent]):

    # Determine orientation and position of each robot

                        timer[agent] = time.time()

                        if (reset[agent] == False): # If not first run through of loop for this agent

                                """plt.plot(agentPosition[0][0], agentPosition[0][1])
                                plt.tight_layout()
                                plt.show()"""

                                messageArray[agent] = "F"

                                timestepDistance[agent] = function.get_distance(currentPosition[agent], agentPosition[agent])
                                if (timestepDistance[agent] > 5):
                                    prevPosition[agent] = currentPosition[agent]
                                    currentPosition[agent] = agentPosition[agent]
                                    orientation[agent] = function.get_orientation(prevPosition[agent], currentPosition[agent])
                                    orientationDes[agent] = function.get_orientation(currentPosition[agent], desPosition[N[agent]])
                                    angle[agent] = function.get_angle(orientation[agent], orientationDes[agent])
                                    messageArray[agent] = function.angle_to_message(angle[agent])

                                if not test:
                                    index = np.where(idOrder_array==agent)
                                    device.send_data_async(remote_devices[index[0][0]], "R," + messageArray[agent] + ">")
                                print("Object ID: " + str(agent) + " Orientation: " + str(orientation[agent])[:4] + " Desired Orientation: " + str(orientationDes[agent])[:4] + " Message Sent: "+ messageArray[agent][:4])

                        if (reset[agent] == True): # If first run through of loop for this agent
                            reset[agent] = False
                            currentPosition[agent] = agentPosition[agent]
                            if not test:
                                orientation[agent] = orientationInit[agent]
                            else:
                                orientation[agent] = 0
                            orientationDes[agent] = function.get_orientation(currentPosition[agent], desPosition[N[agent]])
                            angle[agent] = function.get_angle(orientation[agent], orientationDes[agent])
                            messageArray[agent] = function.angle_to_message(angle[agent])
                            if not test:
                                index = np.where(idOrder_array==agent)
                                device.send_data_async(remote_devices[index[0][0]], "R," + messageArray[agent] + ">")
                            print("Object ID: " + str(agent) + " Orientation: " + str(orientation[agent])[:4] + " Desired Orientation: " + str(orientationDes[agent])[:4] + " Message Sent: "+ messageArray[agent][:4])

                  isEdge[agent] = False
                  pass_var[agent] = False

# Annotate image

              cv2.line(frame, (agentPosition[agent][0], agentPosition[agent][1]), (int(desPosition[N[agent]][0]), int(desPosition[N[agent]][1])), (255, 0, 0), 1)

# Display image, and store plots for this frame

              """fig, axs = plt.subplots(1, 1, figsize=(15, 10))

              for i in range(len(remote_devices)):
                  x, y = zip(*voronoi_update[i])
                  plt.scatter(x, y, s=80)

              x, y = zip(*desPosition)
              plt.scatter(x, y, color='b', s = 150)

              x, y = zip(*agentPosition)
              plt.scatter(x, y, color='k', s = 150)

              if option == 2 or option == 3:
                  plt.scatter(target[0], target[1], color='r')

              plt.axis('off')
              plt.gca().invert_yaxis()
              plt.savefig("Data/option{}/plots/plot{}".format(option, frame_count))
              plt.close('all')

              if option == 2 or option == 3:
                  cv2.circle(frame, (int(target[0]), int(target[1])), 4, (0, 0, 255), -1)"""
              cv2.imwrite("Data/option{}/frames/frame{}.jpg".format(option, frame_count), frame)
              cv2.imshow("Frame", frame)
              cv2.waitKey(1)
              frame_count = frame_count + 1

              Number = 0
              for agent in range(len(remote_devices)):
                  if isSuccess[agent]==True:
                      Number = Number + 1
              if Number == len(remote_devices):
                  for agent in range(len(remote_devices)):
                      if (N[agent]==0):
                          N[agent]+=1
                      else:
                          loopMaster = False

              """if N[agent]>=2:
                  N[agent]=0"""



# End loop if "x" key is pressed

              if cv2.waitKey(1) & 0xFF == ord("x"):
                  for i in range(0, len(remote_devices)):
                      if not test:
                          device.send_data_async(remote_devices[i], "P,S>")
                  break

plt.plot(timeGraph, distanceGraph, "r-")

plt.xlabel('Time (s)')
plt.ylabel('Distance (cm)')

plt.title('Distance of Robot from Target vs. Time')

plt.show()
# Close video capture and windows

#cv2.destroyAllWindows()
vidcap.release()
data_txt.close()
if not test:
    device.close()
