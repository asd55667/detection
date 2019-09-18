from faster_rcnn import FF_RCNN
from generator import OIDGen, ValidationGen

train_gen = OIDGen(json_file='split_1_annotation.json')
val_gen = ValidationGen(json_file='validation_level_1.json')

from keras.applications import VGG16        
vgg16 = VGG16(include_top=False, )

ff_rcnn = FF_RCNN(vgg16, cfg)