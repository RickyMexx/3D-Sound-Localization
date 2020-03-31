# Extracts the features, labels, and normalizes the training and test split features. Make sure you update the location
# of the downloaded datasets before in the cls_feature_class.py

import cls_feature_class
import cls_feature_extr
import parameter

params = parameter.get_params('1')
dataset_name = params['dataset']
dataset_dir = params['dataset_dir']
feat_label_dir = params['feat_label_dir']


if(dataset_name == "foa"):
    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_extr.FeatureClass(dataset=dataset_name, dataset_dir=dataset_dir, feat_label_dir=feat_label_dir)

    # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # Extract labels in regression mode
    dev_feat_cls.extract_all_labels()

else:
    # Extracts feature and labels for all overlap and splits
    for ovo in [2]:  # SE overlap
        for splito in [1]:    # all splits. Use [1, 8, 9] for 'real' dataset
            for nffto in [512]:
                feat_cls = cls_feature_class.FeatureClass(ov=ovo, split=splito, nfft=nffto, dataset=dataset_name)

                # Extract features and normalize them
                feat_cls.extract_all_feature()
                feat_cls.preprocess_features()

                # # Extract labels in regression mode
                feat_cls.extract_all_labels('regr', 0)
