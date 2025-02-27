from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from frcnn import config, pascal_parser, roi_util, losses, data_generator
from pathlib import Path
import pickle, traceback
import random, time
import numpy as np
import keras.backend as K

import argparse

# set up command line arguments
args = argparse.ArgumentParser(description="Train a Faster RCNN model using keras and transfer learning on PASCAL format.")
args.add_argument("-d", "--data_path", help="path to train directory", default="data/pascal")
args.add_argument("-o", "--model_path", help="path to store trained model", default="models/")
args.add_argument("-b", "--base_network", help="pre trained base model layers for training", choices=['resnet', 'vgg'], default='resnet')
args.add_argument("-e", "--epochs", help="number of epochs for training", default=50)
args.add_argument("-i", "--iterations", help="number of iterations per epoch", default=1000)
args.add_argument("-c", "--config_path", help="path to save model configuration file which will be used in testing", default="models/config")
args.add_argument("-l", "--learning_rate", help="learning rate for rpn and rcnn", default=1e-5)
args.add_argument("-s", "--solver_optimitzer", help="solver to use for convergance", choices=['sgd', 'adam'], default='adam')


args = args.parse_args()

DATA_PATH = Path(args.data_path)
MODEL_PATH = Path(args.model_path)
CONFIG_PATH = Path(args.config_path)

if not DATA_PATH.exists():
    raise FileNotFoundError("Path not found {}. Please enter correct path for data.".format(DATA_PATH))

if not DATA_PATH.is_dir():
    raise FileNotFoundError("Path not found {}. Please enter valid directory for saving model.".format(MODEL_PATH))

C = config.Config()
# get configuration settings from user, for now using default.
C.base_network = args.base_network

if C.base_network == 'vgg':
    from frcnn import vgg as base_network
    C.base_net_weights = ''
elif C.base_network == 'resnet':
    from frcnn import resnet as base_network
    C.base_net_weights = 'models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
else:
    raise ValueError('Base network not registered!')

all_imgs, classes_count, class_mapping = pascal_parser.get_data(DATA_PATH)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

print('Training images per class:')
print(classes_count)
print('Number of classes: {}'.format(len(class_mapping)))

with open(CONFIG_PATH, 'wb') as config_writer:
    pickle.dump(C, config_writer)
    print("Saved config at {} for future use.".format(CONFIG_PATH))

random.shuffle(all_imgs)
num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print("Total training images: {}".format(len(train_imgs)))
print("Total number of test images: {}".format(len(test_imgs)))

data_gen_train = data_generator.get_anchor_gt(train_imgs, classes_count, C, base_network.get_img_output_length, mode='train')
data_gen_val = data_generator.get_anchor_gt(test_imgs, classes_count, C, base_network.get_img_output_length, mode='val')

input_img_shape = (None, None, 3)

img_input = Input(shape=input_img_shape)
roi_input = Input(shape=(None, 4))

shared_layers = base_network.nn_base(input_tensor=img_input, trainable=True)

num_anchors = len(C.anchor_ratios) * len(C.anchor_scales)
rpn = base_network.rpn(shared_layers, num_anchors)
classifier = base_network.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

model_all = Model([img_input, roi_input], classifier + rpn[:2])

try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
    print("--->> Weights loaded successfully!!")
except:
    print('Unable to load weights, please check this path: {}'.format(C.base_net_weights))

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(
    optimizer=optimizer,
    loss={
        'rpn_out_classifier': losses.rpn_loss_cls(num_anchors),
        'rpn_out_regressor': losses.rpn_loss_regr(num_anchors)
    }
)
model_classifier.compile(
    optimizer=optimizer_classifier,
    loss={
        'rcnn_classifier_{}'.format(len(classes_count)): losses.class_loss_cls,
        'rcnn_regressor_{}'.format(len(classes_count)): losses.class_loss_regr(len(classes_count)-1)
    },
    metrics={
        'rcnn_classifier_{}'.format(len(classes_count)): 'accuracy'
    }
)
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 1000
num_epochs = 50
iter_num = 0

all_losses = np.zeros((epoch_length, 5))

rpn_accuracy_monitor = []
rpn_accuracy_per_epoch = []

start_time = time.time()

best_loss = np.Inf

print("==> starting training... ...")
vis = True

for epoch_num in range(num_epochs):

    prog_bar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    while True:
        try:
            if len(rpn_accuracy_monitor) == epoch_length and C.verbose:
                mean_overlapping_boxes = float(sum(rpn_accuracy_monitor))/len(rpn_accuracy_monitor)
                rpn_accuracy_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_boxes, epoch_length))
                if mean_overlapping_boxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, image_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            roi = roi_util.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)

            X2, Y1, Y2, Ious = roi_util.calc_iou(roi, image_data, C, class_mapping)

            if X2 is None:
                # if no ROI generated, append zero to accuracy
                rpn_accuracy_monitor.append(0)
                rpn_accuracy_per_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_monitor.append(len(pos_samples))
            rpn_accuracy_per_epoch.append(len(pos_samples))

            if C.num_rois > 1:
                # randomly choosing positive and negative ROIs
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()

                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples

            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_classifier = model_classifier.train_on_batch(
                [X, X2[:, sel_samples, :]],
                [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
            )
            all_losses[iter_num, 0] = loss_rpn[1]
            all_losses[iter_num, 1] = loss_rpn[2]

            # print(dict(zip(model_rpn.metrics_names, loss_rpn)))
            # print(dict(zip(model_classifier.metrics_names, loss_classifier)))

            all_losses[iter_num, 2] = loss_classifier[1]
            all_losses[iter_num, 3] = loss_classifier[2]
            all_losses[iter_num, 4] = loss_classifier[3]

            iter_num += 1

            prog_bar.update(iter_num, [
                ('rpn_cls', np.mean(all_losses[:iter_num, 0])),
                ('rpn_rgr', np.mean(all_losses[:iter_num, 1])),
                ('rcnn_cls', np.mean(all_losses[:iter_num, 2])),
                ('rcnn_rgr', np.mean(all_losses[:iter_num, 3]))
            ])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(all_losses[:, 0])
                loss_rpn_regr = np.mean(all_losses[:, 1])
                loss_class_cls = np.mean(all_losses[:, 2])
                loss_class_regr = np.mean(all_losses[:, 3])
                class_acc = np.mean(all_losses[:, 4])

                mean_overlapping_boxes = float(sum(rpn_accuracy_per_epoch))/len(rpn_accuracy_per_epoch)
                rpn_accuracy_per_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_boxes))
                    print('Classifier accuracy for bounding boxes from RPN [Fast RCNN]: {}'.format(class_acc))
                    print('Loss [RPN] classifier: {}'.format(loss_rpn_cls))
                    print('Loss [RPN] regression: {}'.format(loss_rpn_regr))
                    print('Loss [Fast RCNN] classifier: {}'.format(loss_class_cls))
                    print('Loss [Fast RCNN] regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))

                    best_loss = curr_loss
                    model_all.save_weights(C.model_path.format(MODEL_PATH, epoch_num))

                break
                
        except Exception as e:
            print('Exception: {} <<<==='.format(e))
            print(traceback.format_exc())
            continue

print('Training completed! Run the test file.')
