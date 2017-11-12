#!/usr/bin/env python
import argparse
import rosbag
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from tqdm import tqdm
import os
import threading


class ImageProcBag(object):
    '''
    Dictionary of topics to remap. If ends in /, remaps everything after
    Otherwise, topic must match exactly
    '''
    def __init__(self, bag, camera, out=None):
        self.bag = bag
        self.camera = camera
        self.out = out
        self.raw_stack = []
        self.info_stack = []
        self.raw_topic = camera + 'image_raw'
        self.info_topic = camera + 'camera_info'
        self.raw_pub = rospy.Publisher(self.raw_topic, Image, queue_size=5)
        self.info_pub = rospy.Publisher(self.info_topic, CameraInfo, queue_size=5)
        self.lock = threading.Lock()
        published_topics = rospy.get_published_topics()
        topics = ['image_color', 'image_mono', 'image_rect', 'image_rect_color']
        self.subs = {}
        self.recv = {}
        for topic in topics:
            resolved = self.camera + topic
            self.recv[topic] = None
            print resolved
            self.subs[topic] = rospy.Subscriber(resolved, Image, callback=self.cb, callback_args=resolved, queue_size=5)
            if not [topic for topic, _ in published_topics if topic == resolved]:
                raise Exception('{} not published. Is Image Proc running?'.format(resolved))

    def cb(self, msg, topic):
        self.lock.acquire()
        if not self.outbag:
            return
        self.outbag.write(topic, msg, self.last_time)
        self.lock.release()

    def do(self):
        # Have a synced frame, let's get proc
        self.lock.acquire()
        for topic in self.recv:
            self.recv[topic] = None
        self.lock.release()
        stamp = self.info_stack[0].header.stamp
        self.raw_pub.publish(self.raw_stack.pop())
        self.info_pub.publish(self.info_stack.pop())
        for topic in self.recv:
            print 'waiting on ', topic
            timeout = rospy.Time.now() + rospy.Duration(5)
            while True:
                if rospy.is_shutdown():
                    print 'shutdown'
                    return
                if rospy.Time.now() > timeout:
                    print 'timedout'
                    break
                self.lock.acquire()
                if self.recv[topic] is not None and self.recv[topic].header.stamp == stamp:
                    print 'got', topic
                    self.lock.release()
                    break
                self.lock.release()
                rospy.sleep(0.01)

    def go(self):
        if not os.path.isfile(self.bag):
            raise Exception('{} is not a file'.format(self.bag))
        split = os.path.splitext(self.bag)
        outname = split[0] + '_fixed' + split[1]
        if os.path.exists(outname):
            raise Exception('file already exists')
        inbag = rosbag.Bag(self.bag, mode='r')
        self.outbag = rosbag.Bag(outname, mode='w')
        try:
            for topic, msg, time in inbag.read_messages():
                if topic == self.raw_topic:
                    self.raw_pub.publish(msg)
                    rospy.sleep(0.01)
                if topic == self.info_topic:
                    self.info_pub.publish(msg)
                    rospy.sleep(0.01)
                self.lock.acquire()
                self.last_time = time
                self.outbag.write(topic, msg, time)
                self.lock.release()
        finally:
            inbag.close()
            self.lock.acquire()
            self.outbag.close()
            self.outbag = None
            self.lock.release()

if __name__ == "__main__":
    import sys
    print sys.argv
    parser = argparse.ArgumentParser(description='Fix bag topics/frame_ids')
    parser.add_argument('camera', type=str,
                        help='')
    parser.add_argument('bag', type=str,
                        help='')
    args = parser.parse_args()
    rospy.init_node('bag_image_proc', anonymous=True)
    x = ImageProcBag(args.bag, args.camera)
    x.go()
