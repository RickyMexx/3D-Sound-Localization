#
# Implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and
# the DOA metrics explained in the SELDnet paper
#

# bootstrap confidence intervals
from numpy.random import seed
from numpy.random import randint
from numpy import median, mean, percentile

# seed the random number generator
seed(1)

import numpy as np
from sklearn.metrics import confusion_matrix
from IPython import embed

eps = np.finfo(np.float).eps

###############################################################
# Scoring functions
###############################################################


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    print("precision:", prec)
    recall = float(TP) / float(Nref + eps)
    print("recall:", recall)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()  # axis=1)
    Nref, Nsys = T.sum(), O.sum()
    ER = (max(Nref, Nsys) - TP) / (Nref + 0.0)
    return ER


def f1_framewise(O, T):
    # This is wrongly calculated f1 score where per frame F1-score is
    # caluclated and later mean of these is taken. Use this only for legacy stuff
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum(axis=1)
    Nref, Nsys = T.sum(axis=1), O.sum(axis=1)

    prec = (TP + eps) / (Nsys + eps)
    recall = (TP + eps) / (Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def f1_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i,] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i,] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return f1_framewise(O_block, T_block)


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i,] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i,] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / (block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i,] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
        T_block[i,] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), ], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_sed_scores(pred, y, nb_frames_1s):
    """Compute TUT metrics

    Parameters
    ----------
    pred : matrix
        predicted matrix / system output

    y : matrix
        reference matrix

    hop_length_seconds : float
        used frame hop length

    Returns
    -------
    scores : dict
    """
    f1o = f1_overall_1sec(pred, y, nb_frames_1s)
    ero = er_overall_1sec(pred, y, nb_frames_1s)
    scores = [ero, f1o]
    return scores


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def compute_doa_scores_regr_xy(pred, gt, pred_sed, gt_sed):
    nb_src_gt_list = np.zeros(gt.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt.shape[0]).astype(int)
    good_frame_cnt = 0
    less_frame_cnt = 0
    more_frame_cnt = 0
    doa_loss_gt = 0.0
    doa_loss_gt_cnt = 0
    doa_loss_pred = 0.0
    doa_loss_pred_cnt = 0
    nb_sed = gt_sed.shape[-1]

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))
        if nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_frame_cnt = less_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_frame_cnt = more_frame_cnt + 1
        else:
            good_frame_cnt = good_frame_cnt + 1

        # DOA Loss with respect to groundtruth
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]

        for cnt in range(nb_src_gt_list[frame_cnt]):
            doa_loss_gt += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2
            )
            doa_loss_gt_cnt += 1

        # DOA Loss with respect to predicted confidence
        sed_frame_pred = pred_sed[frame_cnt]
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]

        for cnt in range(nb_src_pred_list[frame_cnt]):
            doa_loss_pred += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2
            )
            doa_loss_pred_cnt += 1

    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    if doa_loss_gt_cnt:
        doa_loss_gt /= doa_loss_gt_cnt

    max_nb_src_gt = np.max(nb_src_gt_list)
    conf_mat = confusion_matrix(nb_src_gt_list, nb_src_pred_list)
    conf_mat = conf_mat / (eps + np.sum(conf_mat, 1)[:, None].astype('float'))
    avg_accuracy = np.mean(np.diag(conf_mat[:max_nb_src_gt, :max_nb_src_gt]))  # In frames where more DOA's are
    # predicted, the conf_mat is no more square matrix, and the average skew's the results. Hence we always calculate
    # the accuracy wrt gt number of sources
    er_metric = [avg_accuracy, doa_loss_gt, doa_loss_pred, doa_loss_gt_cnt, doa_loss_pred_cnt, good_frame_cnt]
    return er_metric, conf_mat


def compute_doa_scores_regr_xyz(pred, gt, pred_sed, gt_sed):
    nb_src_gt_list = np.zeros(gt.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt.shape[0]).astype(int)
    good_frame_cnt = 0
    less_frame_cnt = 0
    more_frame_cnt = 0
    doa_loss_gt = 0.0
    doa_loss_gt_cnt = 0
    doa_loss_pred = 0.0
    doa_loss_pred_cnt = 0
    nb_sed = gt_sed.shape[-1]

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))
        if nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_frame_cnt = less_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_frame_cnt = more_frame_cnt + 1
        else:
            good_frame_cnt = good_frame_cnt + 1

        # DOA Loss with respect to groundtruth
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]
        doa_frame_gt_z = gt[frame_cnt][2*nb_sed:][sed_frame == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame == 1]
        doa_frame_pred_z = pred[frame_cnt][2*nb_sed:][sed_frame == 1]

        for cnt in range(nb_src_gt_list[frame_cnt]):
            doa_loss_gt += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2 +
                (doa_frame_gt_z[cnt] - doa_frame_pred_z[cnt]) ** 2
            )
            doa_loss_gt_cnt += 1

        # DOA Loss with respect to predicted confidence
        sed_frame_pred = pred_sed[frame_cnt]
        doa_frame_gt_x = gt[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_gt_y = gt[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]
        doa_frame_gt_z = gt[frame_cnt][2*nb_sed:][sed_frame_pred == 1]

        doa_frame_pred_x = pred[frame_cnt][:nb_sed][sed_frame_pred == 1]
        doa_frame_pred_y = pred[frame_cnt][nb_sed:2*nb_sed][sed_frame_pred == 1]
        doa_frame_pred_z = pred[frame_cnt][2*nb_sed:][sed_frame_pred == 1]

        for cnt in range(nb_src_pred_list[frame_cnt]):
            doa_loss_pred += np.sqrt(
                (doa_frame_gt_x[cnt] - doa_frame_pred_x[cnt]) ** 2 +
                (doa_frame_gt_y[cnt] - doa_frame_pred_y[cnt]) ** 2 +
                (doa_frame_gt_z[cnt] - doa_frame_pred_z[cnt]) ** 2
            )
            doa_loss_pred_cnt += 1

    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    if doa_loss_gt_cnt:
        doa_loss_gt /= doa_loss_gt_cnt

    max_nb_src_gt = np.max(nb_src_gt_list)
    conf_mat = confusion_matrix(nb_src_gt_list, nb_src_pred_list)
    conf_mat = conf_mat / (eps + np.sum(conf_mat, 1)[:, None].astype('float'))
    avg_accuracy = np.mean(np.diag(conf_mat[:max_nb_src_gt, :max_nb_src_gt]))  # In frames where more DOA's are
    # predicted, the conf_mat is no more square matrix, and the average skew's the results. Hence we always calculate
    # the accuracy wrt gt number of sources
    er_metric = [avg_accuracy, doa_loss_gt, doa_loss_pred, doa_loss_gt_cnt, doa_loss_pred_cnt, good_frame_cnt]
    return er_metric, conf_mat

# Old implemenation of confidence
''' 
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
'''

# Updated version, using classic formula
def compute_confidence(metric, size):
    displacement = 1.96 * np.sqrt(((metric) * (1 - metric)) / size)
    return [metric-displacement, metric+displacement]

# Specific confidence computation for cartesian coordinates
def compute_doa_confidence(err, n_classes):
    # Computing doa error on x axis
    doa_err_x = np.reshape(err, newshape=(err.shape[0], n_classes, 3))
    doa_err_x = np.absolute(doa_err_x[:, :, 0])
    doa_err_x = np.reshape(doa_err_x, newshape=(doa_err_x.shape[0] * doa_err_x.shape[1]))

    # Computing doa error on y axis
    doa_err_y = np.reshape(err, newshape=(err.shape[0], n_classes, 3))
    doa_err_y = np.absolute(doa_err_y[:, :, 1])
    doa_err_y = np.reshape(doa_err_y, newshape=(doa_err_y.shape[0] * doa_err_y.shape[1]))

    # Computing doa error on y axis
    doa_err_z = np.reshape(err, newshape=(err.shape[0], n_classes, 3))
    doa_err_z = np.absolute(doa_err_z[:, :, 2])
    doa_err_z = np.reshape(doa_err_z, newshape=(doa_err_z.shape[0] * doa_err_z.shape[1]))

    return [doa_err_x, doa_err_y, doa_err_z]

