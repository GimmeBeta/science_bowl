import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.filters import (
        threshold_otsu,
        threshold_yen,
        threshold_isodata,
        threshold_li,
        threshold_local,
        threshold_minimum,

        )
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian

from scipy import ndimage as ndi

import statsmodels.api as sm

class DataSet:
    '''
        A dataset is a directory containing multiple subdirectories each containing images with nuclei and corresponding
        masks
    '''
    def __init__(self, dataset_path, data_set_type='train'):
        # path of the dataset diretcory
        self.path = dataset_path
        if data_set_type in ['train','test']:
            self.type = data_set_type
        else:
            raise ValueError("data_set_type can only take values 'train' or 'test'")
        subdir_names = os.listdir(self.path)
        self.N_subdir = len(subdir_names)
        self.subdir = []
        # get all the subdirectories
        for this_dir_name in subdir_names:
            if not('.' in this_dir_name):
               this_subdir = SubDir(self, this_dir_name)
               self.subdir.append(this_subdir)


class SubDir:
    '''
        A experimental subdirectory
    '''

    def __init__(self, dataset, subdir_name):
        # parent dataset
        self.dataset = dataset
        # name of the subdirectory
        self.name = subdir_name
        # subdirectory path
        self.path = os.path.join(self.dataset.path,self.name)
        # get the image
        self.image_dir_path = os.path.join(self.path,'images')
        self.image = Image(self)
        # get the nuclei
        self.nucleus = []
        self.masks_dir_path = os.path.join(self.path, 'masks')
        if self.dataset.type == 'train':
            masks_files_names = os.listdir(os.path.join(self.path, 'masks'))
            for this_file_name in masks_files_names:
                if '.png' in this_file_name:
                    this_nucleus = Nucleus(self, this_file_name)
                    self.nucleus.append(this_nucleus)
        self.refsize = None

    def show_full_mask(self, src='segmentation', method='otsu'):
        plt.imshow(self.get_full_mask(src=src,method=method))
        plt.title('full binary mask from: \n {}...{}  src={}'.format(self.name[:5], self.name[-5:],src))
        plt.xticks([])
        plt.yticks([])

    def get_full_mask(self, src='segmentation', method='otsu', min_area=5,delta=0):
        if src == 'segmentation':
            # local,
            # threshold_minimum,
            img = self.image.eq_img()
            if self.image.is_inverted:
                img = cv2.bitwise_not(img)
            if method=='otsu':
                thresh = threshold_otsu(img)
            elif method=='yen':
                thresh = threshold_yen(img)
            elif method == 'isodata':
                thresh = threshold_isodata(img)
            elif method == 'li':
                thresh = threshold_li(img)
            elif method == 'minimum':
                thresh = threshold_minimum(img)
            else:
                raise ValueError("the method 'get_full_mask' only accept methods 'otsu','yen','isodata','li','minimum'")
            bin_img =  img >= thresh + delta
            return remove_small_objects(bin_img, min_size=min_area, connectivity=1, in_place=False)
        elif src == 'masks':
            shape = self.nucleus[0].mask().shape
            this_full_mask = np.zeros((shape[0], shape[1]), np.uint8)
            for nucleus in self.nucleus:
                this_full_mask = cv2.bitwise_or(this_full_mask, nucleus.mask())
            return this_full_mask
        # block_size = 501
        # adaptive_thresh = threshold_local(dataset.subdir[n].image.eq_img(), block_size, offset=1)
        # binary_adaptive = dataset.subdir[n].image.eq_img() <= adaptive_thresh

    def get_props(self, src='segmentation', method='otsu', connectivity=1, min_area=5):
        label_img = self.get_label_img(src=src, method=method, connectivity=connectivity)
        props = regionprops(label_img.astype(int))
        props = [prop for prop in props if prop.area>min_area]
        return props, label_img

    def get_submasks(self, src='segmentation',  method='otsu', connectivity=1, min_area=5, n=None, add_watershed=True):
        if src=='segmentation':
            masks = []
            if add_watershed:
                props, label_img, _ = self.get_watershed_props(src=src, method=method, min_area=min_area, connectivity=connectivity)
            else:
                props, label_img = self.get_props(src=src, method=method, min_area=min_area, connectivity=connectivity)
            shape = label_img.shape
            for this_label in [prop.label for prop in props]:
                this_mask = np.zeros((shape[0], shape[1]), np.uint8)
                this_mask[label_img==this_label] = 1
                masks.append(this_mask)
            return masks
        elif src == 'masks':
            if n:
                return self.nucleus[n].mask()
            else:
                return [nucleus.mask() for nucleus in self.nucleus]

    def rle_str(self, src='segmentation', method='otsu', min_area=5, connectivity=1, add_watershed=True):
        masks = self.get_submasks(src=src, method=method, connectivity=connectivity, min_area=min_area, add_watershed=add_watershed, n=None)
        if src=='segmentation':
            true_val=1
        elif src=='masks':
            true_val = 255
        this_rle_str = ''
        for mask in masks:
            this_rle_str += '{}, {} \n'.format(self.image.id,rle_encoding(mask, true_val=true_val))
        return this_rle_str

    def get_label_img(self, src='segmentation', method='otsu', eq=True, connectivity=1,min_area=5):
        '''get the area of the '''
        if src=='masks':
            shape = self.nucleus[0].mask().shape
            labels = np.zeros((shape[0], shape[1]), np.uint8)
            cnt = 0
            for nucleus in self.nucleus:
                cnt += 1
                labels = cv2.bitwise_or(labels, nucleus.mask()*cnt)
            return labels
        elif src=='segmentation':
            return label(self.get_full_mask(src=src, method=method,min_area=min_area), connectivity=connectivity).astype(np.uint8)

    def get_overlay(self,  src='segmentation', method='otsu', eq=True, connectivity=1):
        labels = self.get_label_img(src=src, method=method, connectivity=connectivity, eq=eq)
        if eq:
            image_label_overlay = label2rgb(labels, image=self.image.eq_img(), bg_label=0)
        else:
            image_label_overlay = label2rgb(labels, image=self.image.img(), bg_label=0)
        return image_label_overlay

    def show_labelled_img(self, src='segmentation', method='otsu', connectivity=1, eq=True, min_area=5, add_watershed=False):
        if add_watershed:
            _, overlay_img = self.get_watershed_props(src=src, method=method, connectivity=connectivity, min_area=min_area)
        else:
            overlay_img = self.get_overlay(src=src, method=method, connectivity=connectivity, eq=eq)
        plt.imshow(overlay_img)
        plt.title('labelled from: \n {}...{}'.format(self.name[:5], self.name[-5:]))
        plt.xticks([])
        plt.yticks([])

    def get_nucleus_typical_size(self, src='segmentation', method='otsu', connectivity=1, min_area=5):
        image = self.get_full_mask(src=src, method=method)
        # get connected components
        props, label_img = self.get_props(src=src, method=method, connectivity=connectivity, min_area=min_area)
        # get kde for area
        dens = sm.nonparametric.KDEUnivariate([float(prop.minor_axis_length) for prop in props])
        dens.fit()
        self.refsize = dens.support[dens.density.argmax()]

    def get_watershed_props(self, src='segmentation', method='otsu', connectivity=1, min_area=5):
        if self.refsize is None:
            self.get_nucleus_typical_size(src=src, method=method, connectivity=connectivity, min_area=min_area)
        # use a gaussian filter with bandwidth extracted from the segmented object
        img = self.image.eq_img()
        if  self.image.is_inverted:
            img = cv2.bitwise_not(img)
        filtered = gaussian(img, sigma=self.refsize/4)
        this_mask = self.get_full_mask(src=src, method=method, min_area=self.refsize/4)
        distance = ndi.distance_transform_edt(this_mask)
        local_maxi = peak_local_max(
                filtered,
                min_distance=self.refsize/2,
                exclude_border=0,
                indices=False,
                labels=self.get_label_img(src=src, method=method, eq=True, connectivity=connectivity, min_area=self.refsize/4),
            )
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=this_mask)
        image_label_overlay = label2rgb(labels, image=self.image.eq_img(), bg_label=0)
        watershed_props = regionprops(labels.astype(int))
        watershed_props = [prop for prop in watershed_props if prop.area > self.refsize/4]
        return watershed_props, labels, image_label_overlay



class Image:
    def __init__(self, subdir):
        self.subdir = subdir
        self.dataset = subdir.dataset
        self.name = None
        self.path = None
        self.id = ''

        image_name = []
        for file in os.listdir(self.subdir.image_dir_path):
            if file.endswith(".png"):
                image_name.append(file)

        if len(image_name) == 0:
            raise ValueError('no .png file was found in {}'.format(self.subdir.path))
        elif len(image_name) > 1:
            raise ValueError('more than one .png file was found in {}'.format(self.subdir.path))
        else:
            self.name = image_name[0]
            self.id = self.name.strip('.png')
            self.path = os.path.join(self.subdir.image_dir_path, self.name)

    def show(self, cv2_read_option=cv2.IMREAD_UNCHANGED, eq=False):
        if eq:
            plt.imshow(self.eq_img())
            plt.title('histogram equalized image from:\n {}...{}'.format(self.subdir.name[:5], self.subdir.name[-5:]))
        else:
            plt.imshow(self.img(cv2_read_option=cv2_read_option))
            plt.title('image from:\n {}...{}'.format(self.subdir.name[:5], self.subdir.name[-5:]))
        plt.xticks([])
        plt.yticks([])

    def img(self, cv2_read_option=cv2.IMREAD_UNCHANGED):
        '''get the area of the '''
        return cv2.imread(self.path, cv2_read_option)

    def eq_img(self):
        '''get the area of the '''
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        return cl1

    def is_color(self):
        self.img().shape()

    def hist(self, use_eq_img=True,bins=32):
        ' simple histogram '
        if use_eq_img:
            hist, bin_edges = np.histogram(self.eq_img().ravel(), bins=bins,
                                           normed=True)
        else:
            hist, bin_edges = np.histogram(self.img(cv2_read_option=cv2.IMREAD_GRAYSCALE).ravel(), bins=bins, normed=True)
        return hist, bin_edges

    def hist_c(self, use_eq_img=True, bins=32):
        '''
            corrected histogram:
            this method determines whether a picture is
        '''
        is_inverted = False
        hist, bin_edges = self.hist(use_eq_img=use_eq_img,bins=bins)
        hist_c = hist
        half = int(np.floor(bins/2))
        if np.sum(hist[:half]) < np.sum(hist[-half:]):
            is_inverted = True
            hist_c = hist[::-1]
        return hist_c, bin_edges, is_inverted

    @property
    def is_inverted(self):
        _, _, is_inverted_val = self.hist_c()
        return is_inverted_val



class Nucleus:
    def __init__(self, subdir, mask_file_name):
        # experimental directory this nucleus belongs to
        self.subdir = subdir
        # mask file name
        self.name = mask_file_name
        # mask full path
        self.path = os.path.join(self.subdir.masks_dir_path, self.name)


    def show(self,cv2_read_option=cv2.IMREAD_UNCHANGED):
        plt.imshow(self.mask(cv2_read_option=cv2_read_option))
        plt.title('binary mask from:\n {}...{}'.format(self.subdir.name[:5], self.subdir.name[-5:]))
        plt.xticks([])
        plt.yticks([])

    def mask(self, cv2_read_option=cv2.IMREAD_UNCHANGED):
        return cv2.imread(self.path, cv2_read_option)


    @property
    def cc_props(self):
        return regionprops(self.mask(),intensity_image=self.subdir.image.eq_img())[0]



def mfind(a, func=lambda x: x != 0):
    '''
        equivalent to matlab find function:
    '''
    return [i for (i, val) in enumerate(a) if func(val)]

def rle_encoding(x,true_val=255):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == true_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(str(x) for x in run_lengths)
