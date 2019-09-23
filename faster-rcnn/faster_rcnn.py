import datetime, re, os

from cfg import cfg
from rpn import *
from losses import *
from anchors import *

import tensorflow as tf
import keras
from keras import layers, models, optimizers, regularizers, initializers
import keras.backend as K



import functools

class FF_RCNN:
    def __init__(self, backbone, cfg, mode=None):
        self.cfg = cfg
        self.backbone = backbone
        self.rpn = RPN(cfg)()
        self.head = functools.partial(head, cfg=cfg)
        self.model = self.build_model(cfg, mode)
        self.mode = mode
        
    def build_model(self, cfg, mode):
        scale_inp = layers.Input((1, ))
        # (h , w)
        img_size = tf.cast(tf.shape(self.backbone.inputs[0])[1:3], tf.float32)
        map_size = tf.cast(tf.shape(self.backbone.outputs[0])[1:3], tf.float32)
        
        logits, delta, scores1 = self.rpn(self.backbone.outputs[0])
        
        proposals = Proposal_layer(img_size, map_size, scale_inp, cfg, mode)([delta, scores1])
        
        if mode == 'inference':
            rois = RoiPooling()([self.backbone.outputs[0], proposals])
            logits2, delta2, scores = self.head(rois)
            delta2, scores = FilterDetections()([delta2, scores])
            model = models.Model([self.backbone.inputs[0], scale_inp], [delta2, scores])
        else:
            gt_bbox_inp = layers.Input((None, 4))
            labels_inp = layers.Input((None, 1))            
            anchor_target_inp = layers.Input((None, 5))

            rois, delta2_, labels2, idxs_fg = Proposal_Target_layer(cfg)([proposals, gt_bbox_inp,labels_inp])
            rois = RoiPooling()([self.backbone.outputs[0], rois])  
            
            logits2, delta2, scores = self.head(rois)
                        
            rpn_cls_loss = Rpn_class_loss()([anchor_target_inp, logits])
            rpn_box_loss = Rpn_bbox_loss()([anchor_target_inp, delta])      
            roi_cls_loss = Roi_class_loss()([labels2, logits2])
            roi_box_loss = Roi_bbox_loss()([delta2_, delta2, labels2, idxs_fg])
            
            
            model = models.Model([self.backbone.inputs[0], scale_inp, gt_bbox_inp, labels_inp, anchor_target_inp], 
                                 [logits, delta, rpn_cls_loss, rpn_box_loss, logits2, delta2, ])#roi_cls_loss, roi_box_loss])
        return model


    def train(self, train_gen, val_gen, learning_rate, epochs, 
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        assert self.mode == "train", "Create model in training mode."


        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.compile(learning_rate, self.cfg.LEARNING_MOMENTUM)

        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        import multiprocessing
        workers = multiprocessing.cpu_count()

        self.model.fit_generator(
            train_gen,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.cfg.STEPS_PER_EPOCH,
            callbacks=callbacks,
            max_queue_size=100,
            # workers=workers,
            # use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)



    def load_weights(self, pth, by_name=True, exclude=None):
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True        

        f = h5py.File(pth, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # print('\n', 'Pretrained layers')
        pretrained_layers = list([n.decode('utf8') for n in f.attrs['layer_names']])
        # layers = self.model.inner_model.layers if hasattr(model, "inner_model") else self.model.layers
        
        ls = []
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                for l in layer.layers:
                    if l.name in pretrained_layers:
                        # print(l.name)
                        ls.append(l)
            else:
                if layer.name in pretrained_layers:
                    # print(layer.name)
                    ls.append(layer)    
 
        
        if exclude:
            ls = filter(lambda l: l.name not in exclude, ls)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, ls)
        else:
            saving.load_weights_from_hdf5_group(f, ls)

        for l in ls:
            l.trainable = False

        if hasattr(f, 'close'):
            f.close()        
        
        # print('Trainable layers')
        # self.get_trainable_layers()
        self.set_log_dir(pth)
    

    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]faster\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(self.cfg.MODEL_DIR, "{}{:%Y%m%dT%H%M}".format(
            self.cfg.NAME.lower(), now))

        self.checkpoint_path = os.path.join(self.log_dir, "faster_rcnn_{}_*epoch*.h5".format(
            self.cfg.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")                       


    def load_checkpoint(self):
        dir_names = next(os.walk(self.cfg.MODEL_DIR))[1]
        key = self.cfg.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.cfg.MODEL_DIR))
        
        dir_name = os.path.join(self.cfg.MODEL_DIR, dir_names[-1])
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("faster_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint


    def compile(self, learning_rate, momentum):

        optimizer = optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.cfg.GRADIENT_CLIP_NORM)

        self.model._losses = []
        self.model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",]
            # "roi_class_loss", "roi_bbox_loss"]
        for name in loss_names:
            layer = self.model.get_layer(name)
            if layer.output in self.model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.cfg.LOSS_WEIGHTS.get(name, 1.))
            self.model.add_loss(loss)


        reg_losses = [
            regularizers.l2(self.cfg.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(tf.add_n(reg_losses))


        self.model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.model.outputs))


    def predict(self, x):
        assert self.mode == 'inference'
        return self.model.predict(x) 


    def find_trainable_layer(self, layer):
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        layers = []
        for l in self.model.layers:
            l = self.find_trainable_layer(l)
            if l.get_weights():
                print(l)
                layers.append(l)
        return layers        


def head(x, cfg):
    # x (batch_size=1, rois, pool_height,pool_width, channels)            
    x = layers.TimeDistributed(layers.Conv2D(cfg.FC_LAYERS, (7,7)), name='flatten_conv1')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='faltten_bn1')(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.Conv2D(cfg.FC_LAYERS, (1,1)), name='flatten_conv2')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='faltten_bn2')(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    shared = layers.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2), name='flatten')(x)
    logits2 = layers.TimeDistributed(layers.Dense(cfg.N_CLASSES), name='fc1')(shared)
    scores2 = layers.TimeDistributed(layers.Softmax())(logits2)
    delta2 = layers.TimeDistributed(layers.Dense(4*cfg.N_CLASSES), name='fc2')(shared)
    delta2 = layers.Reshape((-1, cfg.N_CLASSES,4))(delta2)
    return logits2, delta2, scores2
    


def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):
    def _filter_detections(scores, labels):
        indices = tf.where(tf.greater(scores, score_threshold))

        filtered_boxes  = tf.gather_nd(boxes, indices)
        filtered_scores = tf.gather(scores, indices)[:, 0]

        nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

        indices = tf.gather(indices, nms_indices)

        labels = tf.gather_nd(labels, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((tf.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        indices = tf.concat(all_indices, axis=0)
    else:
        scores  = tf.maximum(classification, axis    = 1)
        labels  = tf.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=tf.minimum(max_detections, tf.shape(scores)[0]))

    indices             = tf.gather(indices[:, 0], top_indices)
    boxes               = tf.gather(boxes, indices)
    labels              = tf.gather(labels, top_indices)
    other_              = [tf.gather(o, indices) for o in other]

    pad_size = tf.maximum(0, max_detections - tf.shape(scores)[0])
    boxes    = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = tf.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(tf.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_

   
class FilterDetections(layers.Layer):


    def __init__(self, class_specific_filter = True, 
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        parallel_iterations   = 32,
        **kwargs
    ):

        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):

        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[K.floatx(), K.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs) + 1) * [None]

    def get_config(self):
        config = super(FilterDetections, self).get_config()
        config.update({
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config


