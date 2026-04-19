#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import numpy as np
import random

class DataLoader:
    def __init__(self, data_per_class=1, n_classes=10, train=True, n_modals=1):
        self.n_classes = n_classes
        self.data_per_class = data_per_class
        dir = 'dataset/10-5_'
        dir_train = '_training.npy' if train else '_testing.npy'

        vecs = np.load(dir + 'vecs' + dir_train)
        labels = np.load(dir + 'labels' + dir_train)
        self.vecs, self.labels = self.load_data(vecs, labels)


    def load_data(self, vecs, labels):
        shape = self.n_classes * self.data_per_class, 25, 128
        vecs_all = np.zeros(shape, dtype=int)
        labels_all = np.zeros(self.n_classes*self.data_per_class, dtype=int)
        
        for label in range(self.n_classes):
            label_idx = [i for i, v in enumerate(labels) if v == label]
            label_idx = np.random.choice(label_idx, self.data_per_class, replace=False)

            start = label * self.data_per_class
            end = (label + 1) * self.data_per_class
            vecs_all[start:end, :, :] = vecs[label_idx]
            labels_all[start:end] = label
        
        return vecs_all, labels_all
    

    def __iter__(self):
        for vecs, label in zip(self.vecs, self.labels):
            yield vecs, label


    def __len__(self):
        return len(self.labels)


class MMDataLoader:
    def __init__(self, data_per_class=1, n_classes=10, n_modals=1, train=True):
        self.n_classes = n_classes
        self.data_per_class = data_per_class
        self.n_modal = n_modals

        # dir name
        dirs = {}
        dirs[0] = 'dataset/10-5_'
        dirs[1] = 'dataset/10-5_'
        dir_train = '_training.npy' if train else '_testing.npy'

        # load data
        vecs = {}
        labels = {}
        for m in range(n_modals):
            vecs[m] = np.load(dirs[m] + 'vecs' + dir_train)
            labels[m] = np.load(dirs[m] + 'labels' + dir_train)
            
        # data loader
        self.vecs, self.labels = self.load_data(vecs, labels)


    def load_data(self, vecs, labels):
        shape = self.n_classes * self.data_per_class, 25, 128
        vecs_all = []
        labels_all = []
        
        
        for label in range(self.n_classes):
            label_idx = [i for i, v in enumerate(labels[0]) if v == label]
            label_idx = np.random.choice(label_idx, self.data_per_class, replace=False)

            for idx in label_idx:
                d = {}
                for m in range(self.n_modal):
                    d[m] = vecs[m][idx]
                vecs_all.append(d)

            labels_all += [label] * self.data_per_class
        
        return np.array(vecs_all), labels_all
    

    def __iter__(self):
        for vecs, label in zip(self.vecs, self.labels):
            yield vecs, label


    def __len__(self):
        return len(self.labels)


class PseudoDataLoader:
    def __init__(self, data_per_class=1, n_classes=10, n_modals=1, train=True):
        self.n_classes = n_classes
        self.data_per_class = data_per_class
        self.n_modal = 2

        # dir name
        dirs = {}
        dirs[0] = 'dataset/10-5_'
        dirs[1] = 'dataset/10-5_'
        dir_train = '_training.npy' if train else '_testing.npy'

        # load data
        vecs = {}
        labels = {}
        for m in range(2):
            vecs[m] = np.load(dirs[m] + 'vecs' + dir_train)
            labels[m] = np.load(dirs[m] + 'labels' + dir_train)
        
        self.vecs, self.labels = self.load_data(vecs, labels)
    

    def load_data(self, vecs, labels):
        vecs_all = []
        labels_all = []
        # 1. extract only class 0~4
        for label in range(5, 10):
            label_idx = [i for i, v in enumerate(labels[0]) if v == label]
            label_idx = np.random.choice(label_idx, self.data_per_class, replace=False)

            # 2. add pseudo data (class 5~9) by copying class 0~4
            for idx in label_idx:
                d_base = {}
                d_dash = {}
                
                d_base[0] = vecs[0][idx]
                d_base[1] = vecs[1][idx]

                if label % 2 == 0:
                    d_dash[0] = self.alter_vecs(vecs[0][idx])
                    d_dash[1] = vecs[1][idx]
                else:
                    d_dash[0] = vecs[0][idx]
                    d_dash[1] = self.alter_vecs(vecs[1][idx])


                vecs_all.append(d_base)
                vecs_all.append(d_dash)

                labels_all += [np.mod(2*label, 10), np.mod(2*label+1, 10)]
        
        return np.array(vecs_all), labels_all
    
    
    def alter_vecs(self, vecs):
        new_vecs = vecs.copy()

        # decide pixel to alter
        pixels = [i for i in range(20, 25)]

        # alter vectors
        for pixel in pixels:
            new_vecs[pixel, :] = 0
            
            idx = np.where(vecs[pixel, :] == 1)[0]
            new_idx = np.mod(idx + 1, 128)
            new_vecs[pixel, new_idx] = 1
        
        return new_vecs
    
    
    def __iter__(self):
        for vecs, label in zip(self.vecs, self.labels):
            yield vecs, label
    

    def __len__(self):
        return len(self.labels)


class PseudoDataLoader0:
    def __init__(self, data_per_class=1, n_classes=10, n_modals=1, train=True):
        self.n_classes = n_classes
        self.data_per_class = data_per_class
        self.n_modal = 2

        # dir name
        dirs = {}
        dirs[0] = 'dataset/10-5_'
        dirs[1] = 'dataset/10-5_'
        dir_train = '_training.npy' if train else '_testing.npy'
        print(dir_train)

        # load data
        vecs = {}
        labels = {}
        for m in range(2):
            vecs[m] = np.load(dirs[m] + 'vecs' + dir_train)
            labels[m] = np.load(dirs[m] + 'labels' + dir_train)
        
        self.vecs, self.labels = self.load_data(vecs, labels)
    

    def load_data(self, vecs, labels):
        vecs_all = []
        labels_all = []
        # 1. extract only class 0~4
        for label in range(5, 10):
            label_idx = [i for i, v in enumerate(labels[0]) if v == label]
            label_idx = np.random.choice(label_idx, self.data_per_class, replace=False)

            # 2. add pseudo data (class 5~9) by copying class 0~4
            for idx in label_idx:
                d_base = {}
                d_dash = {}
                
                d_base[0] = vecs[0][idx]
                d_base[1] = vecs[1][idx]

                d_dash[0] = self.alter_vecs(vecs[0][idx])#vecs[0][idx]
                d_dash[1] = vecs[1][idx]#self.alter_vecs(vecs[1][idx])

                vecs_all.append(d_base)
                vecs_all.append(d_dash)

                labels_all += [np.mod(2*label, 10), np.mod(2*label+1, 10)]
        
        return np.array(vecs_all), labels_all
    
    
    def alter_vecs(self, vecs):
        new_vecs = vecs.copy()

        # decide pixel to alter
        pixels = [i for i in range(20, 25)]

        # alter vectors
        for pixel in pixels:
            new_vecs[pixel, :] = 0
            
            idx = np.where(vecs[pixel, :] == 1)[0]
            new_idx = np.mod(idx + 1, 128)
            new_vecs[pixel, new_idx] = 1
        
        return new_vecs
    
    
    def __iter__(self):
        for vecs, label in zip(self.vecs, self.labels):
            #print('vecs[0][20]: {}'.format(vecs[0][20]))
            #print('vecs[1][20]: {}'.format(vecs[1][20]))
            yield vecs, label
    

    def __len__(self):
        return len(self.labels)



class PseudoDataLoader1:
    def __init__(self, data_per_class=1, n_classes=10, n_modals=1, train=True):
        self.n_classes = n_classes
        self.data_per_class = data_per_class
        self.n_modal = 2

        # dir name
        dirs = {}
        dirs[0] = 'dataset/10-5_'
        dirs[1] = 'dataset/10-5_'
        dir_train = '_training.npy' if train else '_testing.npy'

        # load data
        vecs = {}
        labels = {}
        for m in range(2):
            vecs[m] = np.load(dirs[m] + 'vecs' + dir_train)
            labels[m] = np.load(dirs[m] + 'labels' + dir_train)
        
        self.vecs, self.labels = self.load_data(vecs, labels)
    

    def load_data(self, vecs, labels):
        vecs_all = []
        labels_all = []
        # 1. extract only class 0~4
        for label in range(5, 10):
            label_idx = [i for i, v in enumerate(labels[0]) if v == label]
            label_idx = np.random.choice(label_idx, self.data_per_class, replace=False)

            # 2. add pseudo data (class 5~9) by copying class 0~4
            for idx in label_idx:
                d_base = {}
                d_dash = {}
                
                d_base[0] = vecs[0][idx]
                d_base[1] = vecs[1][idx]

                d_dash[0] = vecs[0][idx]
                d_dash[1] = self.alter_vecs(vecs[1][idx])


                vecs_all.append(d_base)
                vecs_all.append(d_dash)

                labels_all += [np.mod(2*label, 10), np.mod(2*label+1, 10)]
        
        return np.array(vecs_all), labels_all
    
    
    def alter_vecs(self, vecs):
        new_vecs = vecs.copy()

        # decide pixel to alter
        pixels = [i for i in range(20, 25)]

        # alter vectors
        for pixel in pixels:
            new_vecs[pixel, :] = 0
            
            idx = np.where(vecs[pixel, :] == 1)[0]
            new_idx = np.mod(idx + 1, 128)
            new_vecs[pixel, new_idx] = 1
        
        return new_vecs
    
    
    def __iter__(self):
        for vecs, label in zip(self.vecs, self.labels):
            yield vecs, label
    

    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    loader = PseudoDataLoader0(data_per_class=1, n_modals=2)
    for vecs, label in loader:
        #print(vecs.keys())
        print(label)
        print(vecs[0][22])
        print(vecs[1][22])
        if label == 1:
            break
        #break
