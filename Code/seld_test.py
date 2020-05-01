#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_data_generator
import evaluation_metrics
import keras_model
import parameter
import utils
import time
import datetime
import simple_plotter
from keras.models import load_model
from IPython import embed
plot.switch_backend('agg')

from evaluation_metrics import compute_confidence, compute_doa_confidence


def collect_test_labels(_data_gen_test, _data_out, classification_mode, quick_test):
    # Collecting ground truth for test data
    params = parameter.get_params('1')
    nb_batch = params['quick_test_nb_batch'] if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[1]
        cnt = cnt + 1
        print(cnt)

        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _sed_score, _doa_score, _seld_score):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(311)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(nb_epoch), _sed_score, label='sed_score')
    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(nb_epoch), _doa_score, label='doa_score')
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()

    # New scores plot
    plot.figure()

    plot.plot(range(nb_epoch), _sed_score, label='sed_score')
    plot.plot(range(nb_epoch), _doa_score, label='doa_score')
    plot.plot(range(nb_epoch), _seld_score, label='seld_score')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name+'_scores')
    plot.close()


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
        second input: task_id - (optional) To chose the system configuration in parameters.py. 
                                (default) uses default parameters
    """
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two inputs')
        print('\t>> python seld.py <job-id> <task-id>')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')
    # use parameter set defined by user
    task_id = '1' if len(argv) < 3 else argv[-1]
    params = parameter.get_params(task_id)

    job_id = 1 if len(argv) < 2 else argv[1]

    model_dir = 'models/'
    utils.create_folder(model_dir)
    unique_name = '{}_train{}_validation{}_seq{}'.format(params['dataset'], params['train_split'], params['val_split'], params['sequence_length'])
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))

    # Cycling over overlaps
    for ov in range(1, params['overlap']+1):

        data_gen_test = cls_data_generator.DataGenerator(
            dataset=params['dataset'], ov=params['overlap'], ov_num=ov, split=params['test_split'], db=params['db'], nfft=params['nfft'],
            batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
            weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
            azi_only=params['azi_only'], shuffle=False
        )

        data_in, data_out = data_gen_test.get_data_sizes()
        n_classes = data_out[0][2]

        print(
            'FEATURES:\n'
            '\tdata_in: {}\n'
            '\tdata_out: {}\n'.format(
                data_in, data_out
            )
        )

        gt = collect_test_labels(data_gen_test, data_out, params['mode'], params['quick_test'])
        sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
        doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

        print("#### Saving DOA and SED GT Values ####")
        f = open("models/doa_gt.txt", "w+")
        for elem in doa_gt:
          f.write(str(list(elem)) + "\n")
        f.close()

        f = open("models/sed_gt.txt", "w+")
        for elem in sed_gt:
          f.write(str(elem)+"\n")
        f.close()
        print("######################################")

        print(
            'MODEL:\n'
            '\tdropout_rate: {}\n'
            '\tCNN: nb_cnn_filt: {}, pool_size{}\n'
            '\trnn_size: {}, fnn_size: {}\n'.format(
                params['dropout_rate'],
                params['nb_cnn3d_filt'] if params['cnn_3d'] else params['nb_cnn2d_filt'], params['pool_size'],
                params['rnn_size'], params['fnn_size']
            )
        )

        model = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                        nb_cnn2d_filt=params['nb_cnn2d_filt'], pool_size=params['pool_size'],
                        rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                        classification_mode=params['mode'], weights=params['loss_weights'], summary=False)

        if(os.path.exists('{}_model.ckpt'.format(unique_name))):
            print("Model found!")
            model.load_weights('{}_model.ckpt'.format(unique_name))
            for i in range(10):
                print("###")

        sed_score = np.zeros(params['nb_epochs'])
        doa_score = np.zeros(params['nb_epochs'])
        seld_score = np.zeros(params['nb_epochs'])
        tr_loss = np.zeros(params['nb_epochs'])
        val_loss = np.zeros(params['nb_epochs'])
        doa_loss = np.zeros((params['nb_epochs'], 6))
        sed_loss = np.zeros((params['nb_epochs'], 2))

        epoch_cnt = 0
        start = time.time()

        print("#### Prediction on validation split ####")
        pred = model.predict_generator(
            generator=data_gen_test.generate(),
            steps=params['quick_test_steps'] if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            use_multiprocessing=False,
            workers=1,
            verbose=1,
        )
        print("##########################")
        #print("pred:", pred[1].shape)

        if params['mode'] == 'regr':
            sed_pred = np.array(evaluation_metrics.reshape_3Dto2D(pred[0])) > .5
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1])

            print("#### Saving DOA and SED Pred Values ####")
            f = open("models/doa_pred.txt", "w+")
            for elem in doa_pred:
              f.write(str(list(elem)) + "\n")
            f.close()

            f = open("models/sed_pred.txt", "w+")
            for elem in sed_pred:
              f.write(str(elem)+"\n")
            f.close()
            print("########################################")

            # Old version of confidence intervals
            '''
            # Computing confidence intervals
            sed_err = sed_gt - sed_pred
            [sed_conf_low, sed_conf_up, sed_median] = compute_confidence(sed_err)
            # print("Condidence Interval for SED error is [" + str(sed_conf_low) + ", " + str(sed_conf_up) + "]")
            print("Confidence Interval for SED error is [ %f, %f ]" % (sed_conf_low, sed_conf_up))
            # print("\tMedian is " + str(sed_median))
            print("\tMedian is %f" % (sed_median))
            # print("\tDisplacement: +/- " + str(sed_conf_up - sed_median))
            print("\tDisplacement: +/- %f" % (sed_conf_up - sed_median))
            doa_err = doa_gt - doa_pred
            [doa_conf_low, doa_conf_up, doa_median] = compute_confidence(doa_err)
            # print("Condidence Interval for DOA is [" + str(doa_conf_low) + ", " + str(doa_conf_up) + "]")
            print("Confidence Interval for DOA is [ %f, %f ]" % (doa_conf_low, doa_conf_up))
            # print("Median is " + str(doa_median))
            print("\tMedian is %f" % (doa_median))
            # print("Displacement: +/- " + str(doa_conf_up - doa_median))
            print("\tDisplacement: +/- %f" % (doa_conf_up - doa_median))
            # ------------------------------
            '''

            sed_loss[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_test.nb_frames_1s())
            if params['azi_only']:
                doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred, doa_gt,
                                                                                                 sed_pred, sed_gt)
            else:
                doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt,
                                                                                                  sed_pred, sed_gt)

            sed_score[epoch_cnt] = np.mean([sed_loss[epoch_cnt, 0], 1 - sed_loss[epoch_cnt, 1]])
            doa_score[epoch_cnt] = np.mean([2 * np.arcsin(doa_loss[epoch_cnt, 1] / 2.0) / np.pi,
                                            1 - (doa_loss[epoch_cnt, 5] / float(doa_gt.shape[0]))])
            seld_score[epoch_cnt] = (sed_score[epoch_cnt] + doa_score[epoch_cnt]) / 2

            if os.path.isdir('./models'):
                plot.imshow(conf_mat, cmap='binary', interpolation='None')
                plot.savefig('models/confusion_matrix.jpg')

            # New confidence computation, differing doa and sed errors
            sed_err = sed_loss[epoch_cnt, 0]
            [sed_conf_low, sed_conf_up] = compute_confidence(sed_err, sed_pred.shape[0])
            print("Confidence Interval for SED error is [ %f, %f ]" % (sed_conf_low, sed_conf_up))

            doa_err = doa_gt - doa_pred
            [x_err, y_err, z_err] = compute_doa_confidence(doa_err, n_classes)


            print('epoch_cnt: %d, time: %.2fs, tr_loss: %.4f, val_loss: %.4f, '
                'F1_overall: %.2f, ER_overall: %.2f, '
                'doa_error_gt: %.2f, doa_error_pred: %.2f, good_pks_ratio:%.2f, '
                'sed_score: %.4f, doa_score: %.4f, seld_score: %.4f' %
                (
                    epoch_cnt, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    sed_loss[epoch_cnt, 1], sed_loss[epoch_cnt, 0],
                    doa_loss[epoch_cnt, 1], doa_loss[epoch_cnt, 2], doa_loss[epoch_cnt, 5] / float(sed_gt.shape[0]),
                    sed_score[epoch_cnt], doa_score[epoch_cnt], seld_score[epoch_cnt]
                )
            )

        simple_plotter.plot_3d("models/doa_gt.txt", "models/doa_pred.txt", 0, 11, 200)
        simple_plotter.plot_confidence(x_err, y_err, z_err, "ov"+str(ov))
        #plot_functions(unique_name, tr_loss, val_loss, sed_loss, doa_loss, sed_score, doa_score, epoch_cnt)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
