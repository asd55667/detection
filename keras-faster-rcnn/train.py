from cfg import *
from keras.applications import VGG16        


vgg16 = VGG16(input_shape=(768,1024,3), include_top=False, )
backbone = models.Model(vgg16.input, vgg16.get_layer('block5_conv3').output)
# head = model.Model(vgg16.get_layer(''))

rpn = RPN()

faster_rcnn = Detector(20, backbone, rpn, head)