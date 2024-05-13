import cv2 
import glob
import numpy as np 

image_path_list = glob.glob('./images/*.jpg')
mask_path_list = glob.glob('./images/*__fuse.png')

def rbg_to_gray(number = 0):
    pixel_other_than_color_class_mapping = False
    mask_rgb = cv2.cvtColor(cv2.imread(mask_path_list[number]), cv2.COLOR_BGR2RGB)
    color_class_mapping = {
        (137, 126, 126) : 0, # ground
        (27,  71, 151) : 1, # advertisement
        (111,  48, 253): 2, # audience
        (255,   0, 29) : 3, # football post
        (255, 160, 1) : 4, # team A # orange | goal keeper in green of MU
        (255, 159, 0) : 5, #  goal keeper in green of MU keeper
        (254, 233, 3) : 6, # team B # yellow | goal keeper in yellow of RMA
        (255, 235, 0) : 7, #  goal keeper in yellow of RMA
        (238, 171, 171) : 8, # refree pink
        (201,  19, 223) : 9, # football
    }
    mask_gray = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    

    for i in range(mask_rgb.shape[0]):
        for j in range(mask_rgb.shape[1]):
            pixel_color  = tuple(mask_rgb[i, j])
            if pixel_color in color_class_mapping:
                mask_gray[i, j] = color_class_mapping[pixel_color]
            else: 
                print(pixel_color)
                pixel_other_than_color_class_mapping = True
                mask_gray[i, j] = color_class_mapping[(111,  48, 253)]
    cv2.imwrite(mask_path_list[number].replace('images', 'masks'), mask_gray)

    if pixel_other_than_color_class_mapping:
        return mask_path_list[number]
    else : 
        return ''
