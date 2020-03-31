# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
import utils
import parameter
from sklearn import preprocessing
from sklearn.externals import joblib
from IPython import embed
from numpy import linalg as LA
import matplotlib.pyplot as plot
plot.switch_backend('agg')


class FeatureClass:
    def __init__(self, dataset='ansim', ov=3, split=1, nfft=1024, db=30, wav_extra_name='', desc_extra_name=''):

        # TODO: Change the path according to your machine.
        # TODO: It should point to a folder which consists of sub-folders for audio and metada
        
        params = parameter.get_params('1')
        dataset_dir = params['dataset_dir']

        if dataset == 'ansim':
            self._base_folder = 'ansim'
        elif dataset == 'resim':
            #self._base_folder = os.path.join('/proj/asignal/TUT_SELD/', 'doa_data_echoic/')
            self._base_folder = 'resim'
        elif dataset == 'cansim':
            self._base_folder = os.path.join('/proj/asignal/TUT_SELD/', 'doa_circdata/')
        elif dataset == 'cresim':
            self._base_folder = os.path.join('/proj/asignal/TUT_SELD/', 'doa_circdata_echoic/')
        elif dataset == 'real':
            self._base_folder = 'real'
            #self._base_folder = os.path.join('/proj/asignal/TUT_SELD/', 'tut_seld_data/')
        elif dataset == 'foa':
            self._base_folder = dataset_dir

        # Input directories
        if dataset == 'foa':
            self._aud_dir = os.path.join(self._base_folder, 'foa_dev/wav_ov{}_split{}'.format(ov, split, db, wav_extra_name))
            self._desc_dir = os.path.join(self._base_folder, 'metadata_dev/desc_ov{}_split{}{}'.format(ov, split, desc_extra_name))
        else:
            self._aud_dir = os.path.join(self._base_folder, 'wav_ov{}_split{}_{}db{}'.format(ov, split, db, wav_extra_name))
            self._desc_dir = os.path.join(self._base_folder, 'desc_ov{}_split{}{}'.format(ov, split, desc_extra_name))


        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._mode = None
        self._ov = ov
        self._split = split
        self._db = db
        self._nfft = nfft
        self._win_len = self._nfft
        self._hop_len = self._nfft/2
        self._dataset = dataset
        self._eps = np.spacing(np.float(1e-16))

        # If circular-array 8 channels else 4 for Ambisonic
        if 'c' in self._dataset:
            self._nb_channels = 8
        else:
            self._nb_channels = 4

        # Sound event classes dictionary
        self._unique_classes = dict()
        if 'real' in self._dataset:
            # Urbansound8k sound events
            self._unique_classes = \
                {
                    '1': 0,
                    '3': 1,
                    '4': 2,
                    '5': 3,
                    '6': 4,
                    '7': 5,
                    '8': 6,
                    '9': 7
                }
        else:
            # DCASE 2016 Task 2 sound events
            self._unique_classes = \
                {
                    'clearthroat': 2,
                    'cough': 8,
                    'doorslam': 9,
                    'drawer': 1,
                    'keyboard': 6,
                    'keysDrop': 4,
                    'knock': 0,
                    'laughter': 10,
                    'pageturn': 7,
                    'phone': 3,
                    'speech': 5
                }

        self._fs = 48000
        
        self._hop_len_s = self._nfft/2.0/self._fs
        self._nb_frames_1s = int(1 / self._hop_len_s)
        self._frame_res = self._fs / float(self._hop_len)

        self._resolution = 10
        self._azi_list = range(-180, 180, self._resolution)
        self._length = len(self._azi_list)
        #CNG
        self._ele_list = range(-40, 50, self._resolution)
        self._height = len(self._ele_list)
        self._weakness = None

        # For regression task only
        self._default_azi = 180
        #CNG
        self._default_ele = 50

        if self._default_azi in self._azi_list:
            print('ERROR: chosen default_azi value {} should not exist in azi_list'.format(self._default_azi))
            exit()
        if self._default_ele in self._ele_list:
            print('ERROR: chosen default_ele value {} should not exist in ele_list'.format(self._default_ele))
            exit()

        self._audio_max_len_samples = 60 * self._fs  # TODO: Fix the audio synthesis code to always generate 30s of
        # audio. Currently it generates audio till the last active sound event, which is not always 30s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        self._max_frames = int(np.ceil((self._audio_max_len_samples - self._win_len) / float(self._hop_len)))

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()
    
       
    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        hann_win = np.repeat(np.hanning(self._win_len)[np.newaxis].T, _nb_ch, 1)
        nb_bins = self._nfft / 2
        spectra = np.zeros((int(self._max_frames), int(nb_bins), int(_nb_ch)), dtype=complex)
        for ind in range(self._max_frames):
            start_ind = ind * self._hop_len
            aud_frame = audio_input[int(start_ind) + np.arange(0, int(self._win_len)), :] * hann_win
            #spectra[ind] = self.Qfft(aud_frame, n=int(self._nfft), axis=0, norm='ortho')[:int(nb_bins), :]
            spectra[ind] = np.fft.fft(aud_frame, n=int(self._nfft), axis=0, norm='ortho')[:int(nb_bins), :]
        return spectra

    def _extract_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)
        active=np.zeros((audio_spec.shape[0],audio_spec.shape[1],audio_spec.shape[2]))
        reactive=np.zeros((audio_spec.shape[0],audio_spec.shape[1],audio_spec.shape[2]))
        w=audio_spec[:,:,0]
        x=audio_spec[:,:,1]
        y=audio_spec[:,:,2]
        z=audio_spec[:,:,3]
        n=np.square(np.abs(w))+(1/3)*(np.square(np.abs(x))+np.square(np.abs(y))+np.square(np.abs(z)))
        #print("print n shape:", n.shape)
        norm_matrix=np.reciprocal(n)
        norm_matrix[norm_matrix == np.inf] = 0
        #print("norm matrix:", norm_matrix)
        #print("norm matrix shape:", norm_matrix.shape)
        #norm_factor = 1.0/pow(LA.norm(w), 2)+(1/3)*(pow(LA.norm(x), 2)+pow(LA.norm(y), 2)+pow(LA.norm(z), 2))
        active[:,:,0]=np.multiply(norm_matrix, np.real(np.multiply(np.matrix.conjugate(w),x)))
        active[:,:,1]=np.multiply(norm_matrix, np.real(np.multiply(np.matrix.conjugate(w),y)))
        active[:,:,2]=np.multiply(norm_matrix, np.real(np.multiply(np.matrix.conjugate(w),z)))
        active[:,:,3]=np.multiply(norm_matrix, np.real(np.square(np.abs(np.matrix.conjugate(w)))))
        #active=np.multiplyactive
        #print("active shape:", active.shape)
        #active=active.reshape(self._max_frames,-1)
        reactive[:,:,0]=np.multiply(norm_matrix, np.imag(np.multiply(np.matrix.conjugate(w),x)))
        reactive[:,:,1]=np.multiply(norm_matrix, np.imag(np.multiply(np.matrix.conjugate(w),y)))
        reactive[:,:,2]=np.multiply(norm_matrix, np.imag(np.multiply(np.matrix.conjugate(w),z)))
        reactive[:,:,3]=np.multiply(norm_matrix, np.imag(np.square(np.abs(np.matrix.conjugate(w)))))
        #print("reactive shape:", reactive.shape)
        #reactive=norm_factor*reactive
        #reactive=reactive.reshape(self._max_frames,-1)
        feature=np.concatenate((active,reactive),axis=2)
        #feature=feature.reshape(self._max_frames, -1)
        print(feature.shape)
        print(feature.reshape(self._max_frames, -1).shape)  
        np.save(os.path.join(self._feat_dir, audio_filename), feature.reshape(self._max_frames, -1))

    # OUTPUT LABELS
    def _read_desc_file(self, desc_filename):
        desc_file = {
            'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list(), 'dist': list()
        }
        fid = open(os.path.join(self._desc_dir, desc_filename), 'r')
        #fid.next()
        fid.readline()
        for line in fid:
            split_line = line.strip().split(',')
            if 'real' in self._dataset:
                desc_file['class'].append(split_line[0].split('.')[0].split('-')[1])
            else:
                if self._dataset == 'foa':
                    desc_file['class'].append(split_line[0].split('.')[0])            
                else:
                    desc_file['class'].append(split_line[0].split('.')[0][:-3])  
            desc_file['start'].append(int(np.floor(float(split_line[1])*self._frame_res)))
            desc_file['end'].append(int(np.ceil(float(split_line[2])*self._frame_res)))
            desc_file['ele'].append(int(split_line[3]))
            desc_file['azi'].append(int(split_line[4]))
            desc_file['dist'].append(float(split_line[5]))
        fid.close()
        return desc_file

    def get_list_index(self, azi, ele):
        azi = (azi - self._azi_list[0]) // 10
        ele = (ele - self._ele_list[0]) // 10
        return azi * self._height + ele

    def _get_matrix_index(self, ind):
        azi, ele = ind // self._height, ind % self._height
        azi = (azi * 10 + self._azi_list[0])
        ele = (ele * 10 + self._ele_list[0])
        return azi, ele

    def get_vector_index(self, ind):
        azi = (ind * 10 + self._azi_list[0])
        return azi

    def _get_doa_labels_regr(self, _desc_file):
        azi_label = self._default_azi*np.ones((self._max_frames, len(self._unique_classes)))
        ele_label = self._default_ele*np.ones((self._max_frames, len(self._unique_classes)))
        for i, ele_ang in enumerate(_desc_file['ele']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            azi_ang = _desc_file['azi'][i]
            class_ind = self._unique_classes[_desc_file['class'][i]]
            if (azi_ang >= self._azi_list[0]) & (azi_ang <= self._azi_list[-1]) & \
                    (ele_ang >= self._ele_list[0]) & (ele_ang <= self._ele_list[-1]):
                azi_label[start_frame:end_frame + 1, class_ind] = azi_ang
                ele_label[start_frame:end_frame + 1, class_ind] = ele_ang
            else:
                print('bad_angle {} {}'.format(azi_ang, ele_ang))
        doa_label_regr = np.concatenate((azi_label, ele_label), axis=1)
        return doa_label_regr

    def _get_se_labels(self, _desc_file):
        se_label = np.zeros((self._max_frames, len(self._unique_classes)))
        for i, se_class in enumerate(_desc_file['class']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            se_label[start_frame:end_frame + 1, self._unique_classes[se_class]] = 1
        return se_label

    def _get_labels_for_file(self, label_filename, _desc_file):
        label_mat = None
        if self._mode is 'regr':
            se_label = self._get_se_labels(_desc_file)
            doa_label = self._get_doa_labels_regr(_desc_file)
            label_mat = np.concatenate((se_label, doa_label), axis=1)
        else:
            print("The supported modes are 'regr', you provided {}".format(self._mode))
        print(label_mat.shape)
        np.save(os.path.join(self._label_dir, label_filename), label_mat)

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self, extra=''):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir(extra)
        utils.create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            print('file_cnt {}, file_name {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            self._extract_spectrogram_for_file(wav_filename)

    def preprocess_features(self, extra=''):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir(extra)
        self._feat_dir_norm = self.get_normalized_feat_dir(extra)
        utils.create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file(extra)

        # pre-processing starts
        print('Estimating weights for normalizing feature files:')
        print('\t\tfeat_dir {}'.format(self._feat_dir))

        #CNG
        spec_scaler = preprocessing.StandardScaler()
        train_cnt = 0
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            if 'train' in file_name:
                print(file_cnt, train_cnt, file_name)
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                #spec_scaler.partial_fit(feat_file)
                #del feat_file
                train_cnt += 1
#        joblib.dump(
#            spec_scaler,
#            normalized_features_wts_file
#        )
        joblib.dump(
            feat_file,
            normalized_features_wts_file
        )

        print('Normalizing feature files:')
        # spec_scaler = joblib.load(normalized_features_wts_file) #load weights again using this command
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print(file_cnt, file_name)
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                #feat_file = spec_scaler.transform(feat_file)
                np.save(
                    os.path.join(self._feat_dir_norm, file_name),
                    feat_file
                )
                #del feat_file
        print('normalized files written to {} folder and the scaler to {}'.format(
            self._feat_dir_norm, normalized_features_wts_file))

    def normalize_features(self, extraname=''):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir(extraname)
        self._feat_dir_norm = self.get_normalized_feat_dir(extraname)
        utils.create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()

        # pre-processing starts
        print('Estimating weights for normalizing feature files:')
        print('\t\tfeat_dir {}'.format(self._feat_dir))

        #spec_scaler = joblib.load(normalized_features_wts_file)
        print('Normalizing feature files:')
        # spec_scaler = joblib.load(normalized_features_wts_file) #load weights again using this command
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print(file_cnt, file_name)
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                #feat_file = spec_scaler.transform(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
                #feat_file = spec_scaler.transform(feat_file)
                np.save(
                    os.path.join(self._feat_dir_norm, file_name),
                    feat_file
                )
                #del feat_file
        print('normalized files written to {} folder and the scaler to {}'.format(
            self._feat_dir_norm, normalized_features_wts_file))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self, mode='regr', weakness=0, extra=''):
        self._label_dir = self.get_label_dir(mode, weakness, extra)
        self._mode = mode
        self._weakness = weakness

        print('Extracting spectrogram and labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        utils.create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            print('file_cnt {}, file_name {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file = self._read_desc_file(file_name)
            self._get_labels_for_file(wav_filename, desc_file)

    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_normalized_feat_dir(self, extra=''):
        if(self._dataset == "foa"):
            return self._base_folder + '/feat_label_tmp/foa_dev_norm'
        else:
            return os.path.join(self._base_folder, 'spec_ov{}_split{}_{}db_nfft{}{}_norm'.format(self._ov, self._split, self._db, self._nfft, extra) )
            

    def get_unnormalized_feat_dir(self, extra=''):
        if(self._dataset == "foa"):
            return self._base_folder + '/feat_label_tmp/foa_dev'
        else:
            return os.path.join(self._base_folder, 'spec_ov{}_split{}_{}db_nfft{}{}'.format(self._ov, self._split, self._db, self._nfft, extra) )
        

    def get_label_dir(self, mode, weakness, extra=''):
        if(self._dataset == "foa"):
            return self._base_folder + '/feat_label_tmp/foa_dev_label'
        else:
            return os.path.join(self._base_folder, 'label_ov{}_split{}_nfft{}_{}{}{}'.format(self._ov, self._split, self._nfft, mode, 0 if mode is 'regr' else weakness, extra))

            
    def get_normalized_wts_file(self, extra=''):
        if(self._dataset == "foa"):
            return self._base_folder + '/feat_label_tmp/foa_wts'
        else:
            return os.path.join(self._base_folder, 'spec_ov{}_split{}_{}db_nfft{}{}_wts'.format(self._ov, self._split, self._db, self._nfft, extra))


    def get_default_azi_ele_regr(self):
        return self._default_azi, self._default_ele

    def get_nb_channels(self):
        return self._nb_channels

    def nb_frames_1s(self):
        return self._nb_frames_1s
