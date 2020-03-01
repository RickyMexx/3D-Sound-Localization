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
from IPython import embed
plot.switch_backend('agg')


# bootstrap confidence intervals
from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
from numpy import mean
me
from numpy import percentile

# seed the random number generator
seed(1)

def compute_confidence(data):
    # bootstrap
    length = len(data)
    scores = list()
    for _ in range(100):
        # bootstrap sample
        indices = randint(0, length, length)
        sample = data[indices]
        # calculate and store statistic
        statistic = mean(sample)
        scores.append(statistic)

    # calculate 95% confidence intervals (100 - alpha)
    alpha = 5.0

    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = max(0.0, percentile(scores, lower_p))

    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = min(1.0, percentile(scores, upper_p))

    #print('median=%.3f' % median(scores))
    #print('%.1fth percentile = %.3f' % (lower_p, lower))
    #print('%.1fth percentile = %.3f' % (upper_p, upper))

    return [lower, upper, median(scores)]



def collect_test_labels(_data_gen_test, _data_out, classification_mode, quick_test, quick_test_dim):
    # Collecting ground truth for test data
    nb_batch = quick_test_dim if quick_test else _data_gen_test.get_total_batches_in_data()
    
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


#def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _epoch_metric_loss):
#    plot.figure()
#    nb_epoch = len(_tr_loss)
#    plot.subplot(311)
#    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
#    plot.plot(range(nb_epoch), _val_loss, label='val loss')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(312)
#    plot.plot(range(nb_epoch), _epoch_metric_loss, label='metric')
#    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
#    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(313)
#    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
#    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
#    plot.legend()
#    plot.grid(True)
#
#    plot.savefig(fig_name)
#    plot.close()
    
#def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _sed_score, _doa_score, epoch_cnt):
#    plot.figure()
#    nb_epoch=epoch_cnt
#   # nb_epoch = len(_tr_loss)
#    plot.subplot(311)
#    #plot.plot(range(nb_epoch), _tr_loss, label='tr loss')
#    #plot.plot(range(nb_epoch), _val_loss, label='val loss')
#    plot.plot(range(nb_epoch), _tr_loss[:nb_epoch], label='tr loss')
#    plot.plot(range(nb_epoch), _val_loss[:nb_epoch], label='val loss')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(312)
#    #plot.plot(range(nb_epoch), _epoch_metric_loss, label='metric')
#    #plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
#    #plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
#    plot.plot(range(nb_epoch), _sed_score[:nb_epoch], label='sed_score')
#    plot.plot(range(nb_epoch), _sed_loss[:nb_epoch, 0], label='er')
#    plot.plot(range(nb_epoch), _sed_loss[:nb_epoch, 1], label='f1')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(313)
#    #plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
#    #plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
#    plot.plot(range(nb_epoch), _doa_score, label='doa_score')
#    plot.plot(range(nb_epoch), _doa_loss[:nb_epoch, 1], label='gt_thres')
#    plot.plot(range(nb_epoch), _doa_loss[:nb_epoch, 2], label='pred_thres')
#    plot.legend()
#    plot.grid(True)
#
#    plot.savefig(fig_name)
#    plot.close()

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
    unique_name = '{}_ov{}_split{}_{}{}_3d{}_{}'.format(
        params['dataset'], params['overlap'], params['split'], params['mode'], params['weakness'],
        int(params['cnn_3d']), job_id
    )
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))

    data_gen_train = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='train', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only']
    )

    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only'], shuffle=False
    )

    data_in, data_out = data_gen_train.get_data_sizes()
    print(
        'FEATURES:\n'
        '\tdata_in: {}\n'
        '\tdata_out: {}\n'.format(
            data_in, data_out
        )
    )

    gt = collect_test_labels(data_gen_test, data_out, params['mode'], params['quick_test'], params['quick_test_dim'])
    sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
    doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

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
                                  classification_mode=params['mode'], weights=params['loss_weights'])
    best_metric = 99999
    conf_mat = None
    best_conf_mat = None
    best_epoch = -1
    patience_cnt = 0
    epoch_metric_loss = np.zeros(params['nb_epochs'])
    sed_score=np.zeros(params['nb_epochs'])
    doa_score=np.zeros(params['nb_epochs'])
    seld_score=np.zeros(params['nb_epochs'])
    tr_loss = np.zeros(params['nb_epochs'])
    val_loss = np.zeros(params['nb_epochs'])
    doa_loss = np.zeros((params['nb_epochs'], 6))
    sed_loss = np.zeros((params['nb_epochs'], 2))
    for epoch_cnt in range(params['nb_epochs']):
        start = time.time()
        hist = model.fit_generator(
            generator=data_gen_train.generate(),
            steps_per_epoch=params['quick_test_dim'] if params['quick_test'] else data_gen_train.get_total_batches_in_data(),
            validation_data=data_gen_test.generate(),
            validation_steps=params['quick_test_dim'] if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            use_multiprocessing=False,
            epochs=1,
            verbose=1
        )
        tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
        val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

        pred = model.predict_generator(
            generator=data_gen_test.generate(),
            steps=params['quick_test_dim'] if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            use_multiprocessing=False,
            verbose=2
        )
        print("pred:",pred[1].shape)
        if params['mode'] == 'regr':
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1])

            ''' Computing confidence intervals '''
            sed_err = sed_gt - sed_pred
            [sed_conf_low, sed_conf_up, sed_median] = compute_confidence(sed_err)
            print("Condidence Interval for SED error is ["+str(sed_conf_low)+", "+str(sed_conf_up)+"]")
            print("Median is "+str(sed_median))
            print("Displacement: +/- "+str(sed_conf_up - sed_median))
            doa_err = doa_gt - doa_pred
            [doa_conf_low, doa_conf_up, doa_median] = compute_confidence(doa_err)
            print("Condidence Interval for DOA is ["+str(doa_conf_low)+", "+str(doa_conf_up)+"]")
            print("Median is "+str(doa_median))
            print("Displacement: +/- "+str(doa_conf_up - doa_median))
            ''' ------------------------------ '''

            sed_loss[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_test.nb_frames_1s())

            if params['azi_only']:
                doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred, doa_gt,
                                                                                                 sed_pred, sed_gt)
            else:
                doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt,
                                                                                                  sed_pred, sed_gt)
            print(len(doa_loss))
#            epoch_metric_loss[epoch_cnt] = np.mean([
#                sed_loss[epoch_cnt, 0],
#                1-sed_loss[epoch_cnt, 1],
#                2*np.arcsin(doa_loss[epoch_cnt, 1]/2.0)/np.pi,
#                1 - (doa_loss[epoch_cnt, 5] / float(doa_gt.shape[0]))]
#            )
            sed_score[epoch_cnt] = np.mean([sed_loss[epoch_cnt, 0], 1-sed_loss[epoch_cnt, 1]])
            doa_score[epoch_cnt] = np.mean([2*np.arcsin(doa_loss[epoch_cnt, 1]/2.0)/np.pi, 1 - (doa_loss[epoch_cnt, 5] / float(doa_gt.shape[0]))])
            seld_score[epoch_cnt] = (sed_score[epoch_cnt] + doa_score[epoch_cnt])/2
        
        #plot_functions(unique_name, tr_loss, val_loss, sed_loss, doa_loss, epoch_metric_loss)
        plot_functions(unique_name, tr_loss, val_loss, sed_loss, doa_loss, sed_score, doa_score, seld_score)
        patience_cnt += 1
#        if epoch_metric_loss[epoch_cnt] < best_metric:
#            best_metric = epoch_metric_loss[epoch_cnt]
#            best_conf_mat = conf_mat
#            best_epoch = epoch_cnt
#            model.save('{}_model.h5'.format(unique_name))
#            patience_cnt = 0
        if sed_score[epoch_cnt] < best_metric:
            best_metric = sed_score[epoch_cnt]
            best_conf_mat = conf_mat
            best_epoch = epoch_cnt
            model.save('{}_model.h5'.format(unique_name))
            patience_cnt = 0
            
#        print(
#            'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
#            'F1_overall: %.2f, ER_overall: %.2f, '
#            'doa_error_gt: %.2f, doa_error_pred: %.2f, good_pks_ratio:%.2f, '
#            'error_metric: %.2f, best_error_metric: %.2f, best_epoch : %d' %
#            (
#                epoch_cnt, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
#                sed_loss[epoch_cnt, 1], sed_loss[epoch_cnt, 0],
#                doa_loss[epoch_cnt, 1], doa_loss[epoch_cnt, 2], doa_loss[epoch_cnt, 5] / float(sed_gt.shape[0]),
#                epoch_metric_loss[epoch_cnt], best_metric, best_epoch
#            )
#        )
        print('epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
            'F1_overall: %.2f, ER_overall: %.2f, '
            'doa_error_gt: %.2f, doa_error_pred: %.2f, good_pks_ratio:%.2f, '
            'sed_score: %.2f, doa_score: %.2f, seld_score: %.2f, best_error_metric: %.2f, best_epoch : %d' %
            (
                epoch_cnt, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                sed_loss[epoch_cnt, 1], sed_loss[epoch_cnt, 0],
                doa_loss[epoch_cnt, 1], doa_loss[epoch_cnt, 2], doa_loss[epoch_cnt, 5] / float(sed_gt.shape[0]),
                sed_score[epoch_cnt], doa_score[epoch_cnt], seld_score[epoch_cnt], best_metric, best_epoch
            )
        )
    
    #plot_functions(unique_name, tr_loss, val_loss, sed_loss, doa_loss, sed_score, doa_score, epoch_cnt)
    print('best_conf_mat : {}'.format(best_conf_mat))
    print('best_conf_mat_diag : {}'.format(np.diag(best_conf_mat)))
    print('saved model for the best_epoch: {} with best_metric: {},  '.format(best_epoch, best_metric))
    print('DOA Metrics: doa_loss_gt: {}, doa_loss_pred: {}, good_pks_ratio: {}'.format(
        doa_loss[best_epoch, 1], doa_loss[best_epoch, 2], doa_loss[best_epoch, 5] / float(sed_gt.shape[0])))
    print('SED Metrics: ER_overall: {}, F1_overall: {}'.format(sed_loss[best_epoch, 0], sed_loss[best_epoch, 1]))
    print('unique_name: {} '.format(unique_name))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
