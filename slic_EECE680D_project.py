# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:45:30 2018
This is slic algorithm implementation with opencv/python.
 
Input: I, K, M, iter_max, eps_
I: image file saved in directory
K: desired number of superpixels
M: Nc constant assuming Ns=S=sqrt(N/K) initial superpixel size where N number of pixel
iter_max: maximum iteration
eps_: lower error threshold

Output: K superpixel of similar size as image file saved in directory, including plot

@author: WIN7WOS

Souleymane Sow

"""

import math
from skimage import io, color
import numpy as np
from tqdm import tqdm
import cv2

class Cluster(object):
    cluster_index = 1
    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    
    def __init__(self, filename, K, M, iter_max, eps_):
        self.K = K # desired number of superpixel
        self.M = M # constant Nc
        self.iter_max = iter_max # max iterations
        self.eps_ = eps_ # convergence threshold

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]# raw axis
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width # number of pixel
        self.S = int(math.sqrt(self.N / self.K)) # step size and constant Ns

        self.clusters = [] # empty list object 
        self.label = {} # empty dictionary object with tuple keys and cluster values
        self.dis = np.full((self.image_height, self.image_width), np.inf) #creating an array filled with infinity
        
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    def init_clusters(self): # initialize seed at regular grid
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w)) 
                w += self.S
            w = self.S / 2 
            h += self.S

    def make_cluster(self, h, w):
        h=int(h)
        w=int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])
    
    def get_gradient(self, h, w):
        if w + 1 >= self.image_width: 
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self): # move each seed to the lowest gradient position in a 3x3 neighborhood
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh 
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S): # within 2Sx2S region around si
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w] 
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2)) 
                    if D < self.dis[h][w]: 
                        if (h, w) not in self.label: 
                            self.label[(h, w)] = cluster 
                            cluster.pixels.append((h, w)) 
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):      
        err_=0
        for cluster in self.clusters:         
            sum_h = sum_w = number = 0
            for p in cluster.pixels: 
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = sum_h / number
            _w = sum_w / number
            err_ += math.pow(_h - cluster.h, 2) + math.pow(_w - cluster.w, 2)
            _h=int(_h)
            _w=int(_w)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
        return err_  
    
    def displayContours(self):
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]

        isTaken = np.zeros(self.data.shape[:2], np.bool)
        contours = []

        for i in range(self.image_width):
            for j in range(self.image_height):
                nr_p = 0
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if x>=0 and x < self.image_width and y>=0 and y < self.image_height:
                        if isTaken[y, x] == False and self.label[(j, i)] != self.label[(y, x)]:
                            nr_p += 1

                if nr_p >= 2:
                    isTaken[j, i] = True
                    contours.append([j, i])
        return contours
    
    def iterate(self):
        self.init_clusters()
        self.move_clusters()
        iter_=0
        err=np.inf
        pbar=tqdm(total = self.iter_max+1)
        while (err> self.eps_ and iter_<=self.iter_max):
            self.assignment()
            err=self.update_cluster()
            iter_+=1
            pbar.update(1)
        name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=iter_, m=self.M, k=self.K) # change name
        self.save_current_image(name)
        print(np.array([iter_, err]))
        pbar.close()
        return name
    
    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)
        
    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels: 
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
            
        contours=self.displayContours()
        for i in range(len(contours)):
            image_arr[contours[i][0]][contours[i][1]][0] = 0
            image_arr[contours[i][0]][contours[i][1]][1] = 0 
            image_arr[contours[i][0]][contours[i][1]][2] = 0
            
        self.save_lab_image(name, image_arr)
            
    def plot_img(self,pic):
        img = cv2.imread(pic,0)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL) #create window size dafault cv2.WINDOW_AUTOSIZE 
        cv2.imshow('image',img) # display image
        cv2.waitKey(20000) # window still up to 20sec or until you press a key then close
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = SLICProcessor('lena.png', 200, 40,10,100) #K, M, iter_max, eps_
    name=p.iterate()
#    p = SLICProcessor('lena.png', 300, 40)
#    name=p.iterate()
#    p = SLICProcessor('lena.png', 500, 40)
#    name=p.iterate()
#    p = SLICProcessor('lena.png', 1000, 40)
#    name=p.iterate()
#    p = SLICProcessor('lena.png', 200, 5,10,100)
#    name=p.iterate()
#    p = SLICProcessor('lena.png', 300, 5)
#    name=p.iterate()
#    p = SLICProcessor('lena.png', 500, 5)
#    name=p.iterate()
#    p = SLICProcessor('lena.png', 1000, 5)
#    name=p.iterate()
    p.plot_img(name)