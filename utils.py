import tensorflow as tf
import numpy as np
import cv2

def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
        
def get_image(imageL,imageAB):
    zero = np.zeros((imageL.shape[0],imageL.shape[1],3), np.uint8)
#    imageL = cv2.cvtColor(imageL,cv2.COLOR_BGR2GRAY)
#    print(imageL.shape)
    zero[:,:,0] = ((imageL[:,:,0:1].reshape((224,224)))*255).astype(np.uint8)
    zero[:,:,1:3] = (imageAB*255).astype(np.uint8)
    img_lab = cv2.cvtColor(zero, cv2.COLOR_LAB2BGR)
    return img_lab
        
def save_image(imageL,imageAB, save_dir, name):
    """
    Save image by unprocessing and converting to rgb.
    :param image: iamge to save
    :param save_dir: location to save image at
    :param name: prefix to save filename
    :return:
    """
    zero = np.zeros((imageL.shape[0],imageL.shape[1],3), np.uint8)
#    imageL = cv2.cvtColor(imageL,cv2.COLOR_BGR2GRAY)
#    print(imageL.shape)
    zero[:,:,0] = ((imageL[:,:,0:1].reshape((224,224)))*255).astype(np.uint8)
    zero[:,:,1:3] = (imageAB*255).astype(np.uint8)
    img_lab = cv2.cvtColor(zero, cv2.COLOR_LAB2BGR)
#    img_lab = cv2.cvtColor(zero, cv2.COLOR_YUV2BGR)
    print(name)
    cv2.imwrite(save_dir + '/' + name,img_lab)
