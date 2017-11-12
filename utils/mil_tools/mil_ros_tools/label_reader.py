#!/usr/bin/env python
import rosbag
import argparse
import os
from cv_bridge import CvBridge
import cv2
from mil_tools import rosmsg_to_numpy
import numpy as np
import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


class LabelBagReader(object):
    bridge = CvBridge()
    def __init__(self, topic):
        assert topic[0] == '/', 'Topic must be global (start with /)'
        self.topic = topic
        self.label_topic = os.path.join(topic, 'labels')
        print (self.topic, self.label_topic)

    def read_bag(self, bag, cb):
        labels = []
        bag = rosbag.Bag(bag, mode='r')
        for _, msg, _ in bag.read_messages(topics=[self.label_topic]):
            labels.append(msg)
        assert len(labels), 'No labels in bag'
        for _, msg, _ in bag.read_messages(topics=[self.topic]):
            #print msg.encoding
            for i,  label in enumerate(labels[:]):
                if msg.header.stamp == label.header.stamp:
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    cb(img, label, dtype=msg.encoding)
                    del labels[i]
                    break

    def read_bags(self, bags, cb):
        for bag in bags:
            self.read_bag(bag, cb)


class Counter(object):
    classes = ['off', 'red', 'green', 'blue', 'yellow']
    columns = ['Class', 'R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B']

    def __init__(self):
        self.count = 0
        self.features = []

    def cb(self, image, label, dtype=None):
        self.count += 1
        for o in label.objects:
            if o.name == 'stcin':
                try:
                    o_class = self.classes.index(o.attributes.strip())
                except ValueError:
                    print 'attributes nah bro', o.attributes
                    continue
                points = rosmsg_to_numpy(o.polygon)
                points = np.array(points[:, :2], dtype=np.int32)
                size = points.shape[0]
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                points = points.reshape((size, 1, 2))
                mask = cv2.drawContours(mask, [points], -1, 255, -1)
                show = cv2.bitwise_or(image, image, mask=mask)
                (text_width, text_height), _ = cv2.getTextSize(o.attributes, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
                show = cv2.putText(show, o.attributes, (0, text_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                mean = cv2.mean(image, mask)
                mean = np.array([[mean[:3]]], dtype=np.uint8)
                mean_hsv = cv2.cvtColor(mean, cv2.COLOR_BGR2HSV)
                mean_lab = cv2.cvtColor(mean, cv2.COLOR_BGR2LAB)
                attributes = np.zeros(10)
                attributes[0] = o_class
                attributes[1:4] = mean.flatten()
                attributes[4:7] = mean_hsv.flatten()
                attributes[7:] = mean_lab.flatten()
                self.features.append(attributes)

                blured = cv2.blur(image, (5, 5))
                gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 75, 200)
                #cv2.imshow('real', blured)
                #cv2.imshow('edges', edges)
                #cv2.waitKey()

    def __str__(self):
        return str(self.count)


def get_features():
    bags = []
    for name in os.listdir('.'):
        if name.endswith('.bag'):
            bags.append(name)
    c = Counter()
    x = LabelBagReader('/stereo/right/image_rect_color')
    x.read_bags(bags, c.cb)
    features = np.array(c.features)
    df = pandas.DataFrame(c.features, columns=c.columns)
    df.to_csv('features.csv')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def do_stuff():
    df = pandas.read_csv('features.csv', index_col=0)
    values = np.array(df.values, dtype=int) # Convert to int
    np.random.shuffle(values) # Shuffle so training / testing are evenly distributed about bags
    split = int(0.75 * values.shape[0])

    train = values[:split, :]
    test = values[split:, :]


    clf  = GaussianNB()
    clf.fit(train[:, 1:], train[:, 0])
    print clf.score(test[:, 1:], test[:, 0])

    prediction = clf.predict(test[:, 1:])
    cm = confusion_matrix(test[:, 0], prediction)
    plot_confusion_matrix(cm, Counter.classes)
    plt.show()


if __name__ == '__main__':
    get_features()
    # do_stuff()
