# 3D-Sound-Localization
## Quaternion Neural Networks for 3D Sound Source Localization in Reverberant Environments: Implementation using First Order Ambisonics.

The objective of our work is to build a working SELDnet based network that works with First Order Ambisonics data sets. In particular, we are going to extend SELDnet, adding capabilities to both support pre-existing data sets (ansim, resim, etc.) and the FOA one in a smart, modular, performing way. Therefore, other metrics have been added like the SELD score, mainly used in the [2019 paper](https://github.com/RickyMexx/3D-Sound-Localization/blob/master/Papers/2019_Quaternion%20Convolutional%20Neural%20Networks%20for%20Detection%20and%20Localization%20of%203D%20Sound%20Events.pdf) outcomes evaluation, and a tiny library for a graphical representation of the results.


![doa](/Report/img/plot_doa.jpg)

![seld](/Report/img/plot_labels.jpg)

![seld3](/Report/img/plot_3labels.jpg)

## Usage
This project can be easily executed using one of the two proposed notebooks:
- [Jupyter](/Notebooks/3DSELD-Local.ipynb)
- [Google Colab](/Notebooks/3DSELD-Colab.ipynb)

The latter gives you the possibility to use a pre-loaded and pre-extracted dataset (~200GB).


## Model metrics CSV table 
A quick view of our CSV files.
<table>
<tr><td>

| - | description |
| --- | --- |
| A | training_loss |
| B | validation_loss |
| C | sed_loss_er |
| D | sed_loss_f1 |
| E | doa_loss_avg_accuracy |
| F | doa_loss_gt |
| G | doa_loss_pred |
| H | doa_loss_gt_cnt |

</td><td>
 
| - | description |
| --- | --- |
| I | doa_loss_pred_cnt |
| J | doa_loss_good_frame_cnt |
| K | sed_score |
| L | doa_score |
| M | seld_score |
| N | sed_confidence_interval_low |
| O | sed_confidence_interval__up |


</td></tr> </table>




