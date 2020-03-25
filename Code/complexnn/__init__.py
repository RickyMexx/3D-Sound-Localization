#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Contributors: Titouan Parcollet
# Authors: Olexa Bilaniuk
#
# What this module includes by default:

from   .conv  import (QuaternionConv,
                      QuaternionConv1D,
                      QuaternionConv2D,
                      QuaternionConv3D)

from .recurrent import (QuaternionGRUCell, QuaternionGRU, QuaternionLSTMCell, QuaternionLSTM)
from .tessarine_conv import (TessarineConv, TessarineConv2D) 
from .tessarine_dense import TessarineDense
from .tessarine_recurrent import (TessarineGRUCell, TessarineGRU)

from   .dense import QuaternionDense
from   .init  import (sqrt_init, qdense_init, qconv_init)
from   .utils import (GetRFirst, GetIFirst, GetJFirst,
                      GetKFirst, getpart_quaternion_output_shape_first,
                      get_rpart_first, get_ipart_first, get_jpart_first, get_kpart_first)


