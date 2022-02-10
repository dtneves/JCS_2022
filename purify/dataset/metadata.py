########################################################################################################################
# Research Centers
# ----------------
# Medical Informatics Group
# BIH - Berlin Institute of Health
# Charité - Universitätsmedizin Berlin
# https://www.bihealth.org/en/research/research-groups/fabian-prasser/
#
# Centro ALGORITMI - School of Engineering – University of Minho
# Braga - Portugal
# http://algoritmi.uminho.pt/
#
#
# Description
# -----------
# This module allows to define the metadata of the supported datasets.
# One should be aware that exception handling to take care of incorrect data types, incorrect parameters' values, and
# so forth is, typically, NOT performed, the rule is: We are all grown up (Python) programmers!
#
#
# Moto
# ----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# Authors
# -------
# diogo telmo neves -- {dneves@di.uminho.pt, diogo-telmo.neves@charite.de, tada.science@gmail.com}
#
#
# Copyright
# ---------
# Copyright (c) 2020 diogo telmo neves.
# All rights reserved.
#
#
# Conditions
# ----------
# This code is free/open source code but the following conditions must be met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#
# DISCLAIMER
# ----------
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Date
# ----
# September 2021
########################################################################################################################

import pandas as pd

from typing import Any, Dict, List, Tuple, Union


class Metadata:
    """Define the metadata of the supported datasets and
    a few class methods that are useful to extract info from the metadata of a dataset.

    Attributes
    ----------
    DATASETS : Dict[str, Union[Dict[str, Any], Dict[int, Any]]]
        The metadata of the supported datasets.
    """

    DATASETS: Dict[str, Union[Dict[str, Any], Dict[int, Any]]] = {
        'adult': {  # https://archive.ics.uci.edu/ml/datasets/Adult
            'age': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 17, 'max': 90, 'mean': 38.58164675532078, 'median': 37.0,
                    'std': 13.640432553581146, 'skewness': 0.5587433694130484, 'kurtosis': -0.16612745957143904
                }
            },
            'workclass': {  # the UCI dataset does NOT have miss. values but the CTGAN has, thus, the list is NOT empty
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': ['?'],
                'values_dist': {
                    'Private': 22696, 'Self-emp-not-inc': 2541, 'Local-gov': 2093, 'State-gov': 1298,
                    'Self-emp-inc': 1116, 'Federal-gov': 960, 'Without-pay': 14, 'Never-worked': 7, '?': 1836
                }
            },
            'fnlwgt': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 12285, 'max': 1484705, 'mean': 189778.36651208502, 'median': 178356.0,
                    'std': 105549.97769702233, 'skewness': 1.4469800945789826, 'kurtosis': 6.218810978153801
                }
            },
            'education': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Preschool': 51, '1st-4th': 168, '5th-6th': 333, '7th-8th': 646, '9th': 514, '10th': 933,
                    '11th': 1175, '12th': 433, 'HS-grad': 10501, 'Some-college': 7291, 'Assoc-voc': 1382,
                    'Assoc-acdm': 1067, 'Bachelors': 5355, 'Masters': 1723, 'Prof-school': 576, 'Doctorate': 413
                }
            },
            'education-num': {  # this is an alias of the `education` variable (i.e., feature/column)
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    1: 51, 2: 168, 3: 333, 4: 646, 5: 514, 6: 933,
                    7: 1175, 8: 433, 9: 10501, 10: 7291, 11: 1382,
                    12: 1067, 13: 5355, 14: 1723, 15: 576, 16: 413
                    # 'min': 1, 'max': 16, 'mean': 10.0806793403151, 'median': 10.0,
                    # 'std': 2.5727203320673406, 'skewness': -0.3116758679102297, 'kurtosis': 0.6234440747629248
                }
            },
            'marital-status': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Married-civ-spouse': 14976, 'Never-married': 10683, 'Divorced': 4443, 'Separated': 1025,
                    'Widowed': 993, 'Married-spouse-absent': 418, 'Married-AF-spouse': 23
                }
            },
            'occupation': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': ['?'],
                'values_dist': {
                    'Prof-specialty': 4140, 'Craft-repair': 4099, 'Exec-managerial': 4066, 'Adm-clerical': 3770,
                    'Sales': 3650, 'Other-service': 3295, 'Machine-op-inspct': 2002, 'Transport-moving': 1597,
                    'Handlers-cleaners': 1370, 'Farming-fishing': 994, 'Tech-support': 928, 'Protective-serv': 649,
                    'Priv-house-serv': 149, 'Armed-Forces': 9, '?': 1843
                }
            },
            'relationship': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Husband': 13193, 'Not-in-family': 8305, 'Own-child': 5068,
                    'Unmarried': 3446, 'Wife': 1568, 'Other-relative': 981
                }
            },
            'race': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'White': 27816, 'Black': 3124, 'Asian-Pac-Islander': 1039, 'Amer-Indian-Eskimo': 311, 'Other': 271
                }
            },
            'sex': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Male': 21790, 'Female': 10771
                }
            },
            'capital-gain': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 99999, 'mean': 1077.6488437087312, 'median': 0.0,
                    'std': 7385.292084839299, 'skewness': 11.953847687699799, 'kurtosis': 154.79943785425334
                }
            },
            'capital-loss': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 4356, 'mean': 87.303829734959, 'median': 0.0,
                    'std': 402.960218649059, 'skewness': 4.594629121679692, 'kurtosis': 20.3768017134122
                }
            },
            'hours-per-week': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1, 'max': 99, 'mean': 40.437455852092995, 'median': 40.0,
                    'std': 12.34742868173081, 'skewness': 0.22764253680450092, 'kurtosis': 2.916686796002066
                }
            },
            'native-country': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': ['?'],
                'values_dist': {
                    'United-States': 29170, 'Mexico': 643, 'Philippines': 198, 'Germany': 137, 'Canada': 121,
                    'Puerto-Rico': 114, 'El-Salvador': 106, 'India': 100, 'Cuba': 95, 'England': 90, 'Jamaica': 81,
                    'South': 80, 'China': 75, 'Italy': 73, 'Dominican-Republic': 70, 'Vietnam': 67, 'Guatemala': 64,
                    'Japan': 62, 'Poland': 60, 'Columbia': 59, 'Taiwan': 51, 'Haiti': 44, 'Iran': 43, 'Portugal': 37,
                    'Nicaragua': 34, 'Peru': 31, 'France': 29, 'Greece': 29, 'Ecuador': 28, 'Ireland': 24, 'Hong': 20,
                    'Cambodia': 19, 'Trinadad&Tobago': 19, 'Laos': 18, 'Thailand': 18, 'Yugoslavia': 16, 'Hungary': 13,
                    'Outlying-US(Guam-USVI-etc)': 14, 'Honduras': 13, 'Scotland': 12, 'Holand-Netherlands': 1, '?': 583
                }
            },
            'income': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    '<=50K': 24720, '>50K': 7841
                }
            }
        },       # ok
        'breast': {  # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
            'ID': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    883263: 1, 906564: 1, 89122: 1, 9013579: 1, 868682: 1, 859465: 1, 859464: 1, 911685: 1, 895299: 1,
                    909220: 1, 8811842: 1, 916799: 1, 901034302: 1, 901034301: 1, 8911164: 1, 869691: 1, 8812877: 1,
                    859471: 1, 911673: 1, 87281702: 1, 85638502: 1, 91762702: 1, 859487: 1, 857438: 1, 91485: 1,
                    903516: 1, 9013594: 1, 914769: 1, 8611161: 1, 9012568: 1, 874839: 1, 905557: 1, 862548: 1, 86355: 1,
                    903483: 1, 88199202: 1, 90317302: 1, 8811779: 1, 91805: 1, 8812818: 1, 8812816: 1, 86973702: 1,
                    86973701: 1, 904969: 1, 84358402: 1, 891703: 1, 89344: 1, 857343: 1, 9111805: 1, 854268: 1,
                    88119002: 1, 873885: 1, 862485: 1, 927241: 1, 892214: 1, 906539: 1, 881972: 1, 879804: 1,
                    89143602: 1, 89143601: 1, 857392: 1, 8812844: 1, 9047: 1, 857373: 1, 8710441: 1, 9011495: 1,
                    9011494: 1, 924964: 1, 9111843: 1, 857374: 1, 8510824: 1, 874858: 1, 924342: 1, 909445: 1,
                    888264: 1, 904647: 1, 90524101: 1, 901315: 1, 91544002: 1, 91544001: 1, 859575: 1, 8810955: 1,
                    911320502: 1, 911320501: 1, 89524: 1, 88249602: 1, 914101: 1, 91227: 1, 85382601: 1, 861648: 1,
                    858477: 1, 8670: 1, 904689: 1, 892399: 1, 8810987: 1, 887181: 1, 915940: 1, 9010018: 1, 926682: 1,
                    89742801: 1, 868826: 1, 864729: 1, 864726: 1, 884180: 1, 91903901: 1, 886226: 1, 864685: 1,
                    8910251: 1, 905978: 1, 894329: 1, 912193: 1, 909777: 1, 903554: 1, 905190: 1, 8610175: 1,
                    8911230: 1, 906616: 1, 9010598: 1, 8912909: 1, 898678: 1, 873843: 1, 866674: 1, 91505: 1, 91504: 1,
                    86408: 1, 86409: 1, 84348301: 1, 922840: 1, 846226: 1, 9010333: 1, 89869: 1, 8912280: 1, 853401: 1,
                    866714: 1, 867739: 1, 8811523: 1, 861597: 1, 861598: 1, 9110944: 1, 875938: 1, 877989: 1,
                    9113846: 1, 8612080: 1, 8912284: 1, 90944601: 1, 8712289: 1, 894047: 1, 855133: 1, 8610908: 1,
                    9012315: 1, 9012795: 1, 87127: 1, 8712291: 1, 909231: 1, 9010259: 1, 9010258: 1, 899147: 1,
                    87139402: 1, 857156: 1, 855138: 1, 869476: 1, 87163: 1, 897137: 1, 873592: 1, 913102: 1, 849014: 1,
                    905539: 1, 899187: 1, 873586: 1, 907367: 1, 913505: 1, 904302: 1, 897132: 1, 87556202: 1, 901011: 1,
                    913512: 1, 84300903: 1, 857155: 1, 87106: 1, 91376702: 1, 891923: 1, 8810528: 1, 911391: 1,
                    8860702: 1, 911384: 1, 854039: 1, 90439701: 1, 9112594: 1, 91376701: 1, 908445: 1, 8911670: 1,
                    843786: 1, 84667401: 1, 911366: 1, 915460: 1, 8711202: 1, 864292: 1, 863270: 1, 924934: 1,
                    856106: 1, 9111596: 1, 8610862: 1, 8711216: 1, 84862001: 1, 906290: 1, 912600: 1, 862261: 1,
                    911157302: 1, 902727: 1, 897374: 1, 867387: 1, 859196: 1, 873593: 1, 87164: 1, 923169: 1, 86211: 1,
                    878796: 1, 922576: 1, 908489: 1, 90312: 1, 893548: 1, 881861: 1, 90769602: 1, 89296: 1, 90769601: 1,
                    86208: 1, 923748: 1, 8510653: 1, 865468: 1, 90401602: 1, 8810703: 1, 853201: 1, 855167: 1,
                    91594602: 1, 854253: 1, 915691: 1, 923465: 1, 891670: 1, 905501: 1, 873701: 1, 871001502: 1,
                    884948: 1, 88649001: 1, 871642: 1, 871641: 1, 9113816: 1, 905520: 1, 879830: 1, 88995002: 1,
                    8912055: 1, 88299702: 1, 883852: 1, 859283: 1, 881094802: 1, 907409: 1, 865423: 1, 871201: 1,
                    891716: 1, 90251: 1, 844981: 1, 894090: 1, 894089: 1, 9112712: 1, 912519: 1, 893061: 1, 923780: 1,
                    8910996: 1, 8915: 1, 865432: 1, 8913049: 1, 866458: 1, 871122: 1, 88147102: 1, 872608: 1, 909411: 1,
                    904357: 1, 874662: 1, 901288: 1, 912558: 1, 8912049: 1, 9113778: 1, 90291: 1, 916221: 1, 86517: 1,
                    871001501: 1, 8910988: 1, 845636: 1, 862028: 1, 921386: 1, 87880: 1, 896839: 1, 91813702: 1,
                    8610629: 1, 898690: 1, 924632: 1, 89864002: 1, 90401601: 1, 902976: 1, 902975: 1, 90602302: 1,
                    911654: 1, 8610637: 1, 859983: 1, 862009: 1, 896864: 1, 865128: 1, 901303: 1, 858981: 1, 903011: 1,
                    869218: 1, 911201: 1, 911202: 1, 917092: 1, 8711003: 1, 858970: 1, 9110720: 1, 897880: 1, 893783: 1,
                    883539: 1, 8911163: 1, 882488: 1, 9013838: 1, 862980: 1, 842517: 1, 864018: 1, 862989: 1, 904971: 1,
                    888570: 1, 86730502: 1, 9011971: 1, 8953902: 1, 917897: 1, 875263: 1, 887549: 1, 907145: 1,
                    88203002: 1, 862965: 1, 88206102: 1, 925292: 1, 863031: 1, 921385: 1, 863030: 1, 8911834: 1,
                    9112367: 1, 911150: 1, 852781: 1, 88518501: 1, 906024: 1, 852763: 1, 85713702: 1, 866083: 1,
                    91930402: 1, 864033: 1, 9012000: 1, 91858: 1, 858986: 1, 895100: 1, 9113239: 1, 914366: 1,
                    884689: 1, 8611792: 1, 9110732: 1, 8810436: 1, 9113538: 1, 918465: 1, 877501: 1, 915276: 1,
                    877500: 1, 89813: 1, 8911800: 1, 922577: 1, 925622: 1, 91979701: 1, 84610002: 1, 898431: 1,
                    893988: 1, 88466802: 1, 915452: 1, 8810158: 1, 8712766: 1, 908469: 1, 854002: 1, 919537: 1,
                    852973: 1, 861853: 1, 881046502: 1, 905189: 1, 91550: 1, 901088: 1, 911296202: 1, 8510426: 1,
                    894326: 1, 88147202: 1, 857010: 1, 868223: 1, 894855: 1, 869254: 1, 874373: 1, 9112366: 1,
                    8910721: 1, 8712064: 1, 926125: 1, 901041: 1, 87930: 1, 889719: 1, 913535: 1, 914862: 1, 865137: 1,
                    9113455: 1, 892438: 1, 91813701: 1, 873357: 1, 921644: 1, 884626: 1, 899987: 1, 866203: 1,
                    8910748: 1, 854941: 1, 85759902: 1, 91903902: 1, 908194: 1, 879523: 1, 901028: 1, 9113514: 1,
                    877486: 1, 861103: 1, 915186: 1, 892657: 1, 869104: 1, 911408: 1, 893526: 1, 875093: 1, 899667: 1,
                    922296: 1, 89511502: 1, 89511501: 1, 8813129: 1, 911296201: 1, 855625: 1, 852552: 1, 844359: 1,
                    883270: 1, 859717: 1, 897604: 1, 917080: 1, 875099: 1, 906878: 1, 914333: 1, 90745: 1, 886776: 1,
                    898677: 1, 908916: 1, 8910720: 1, 9110127: 1, 864877: 1, 925277: 1, 853612: 1, 915664: 1, 915143: 1,
                    861799: 1, 8610404: 1, 897630: 1, 859711: 1, 842302: 1, 889403: 1, 903507: 1, 848406: 1, 9112085: 1,
                    855563: 1, 901549: 1, 84501001: 1, 868871: 1, 919812: 1, 89263202: 1, 921092: 1, 862722: 1,
                    925291: 1, 9113156: 1, 85922302: 1, 862717: 1, 88147101: 1, 8712729: 1, 84799002: 1, 919555: 1,
                    9013005: 1, 86561: 1, 857637: 1, 868202: 1, 869931: 1, 911916: 1, 846381: 1, 8612399: 1,
                    88330202: 1, 925236: 1, 851509: 1, 88411702: 1, 917062: 1, 8711803: 1, 88725602: 1, 871149: 1,
                    905502: 1, 86135501: 1, 901836: 1, 90250: 1, 89382602: 1, 89382601: 1, 914580: 1, 88350402: 1,
                    8711002: 1, 857793: 1, 9010877: 1, 892604: 1, 922297: 1, 898143: 1, 926954: 1, 86135502: 1, 8913: 1,
                    92751: 1, 892189: 1, 905686: 1, 874217: 1, 9010872: 1, 8611555: 1, 924084: 1, 884448: 1, 877159: 1,
                    857810: 1, 917896: 1, 84458202: 1, 926424: 1, 884437: 1, 89812: 1, 85715: 1, 914102: 1, 885429: 1,
                    886452: 1, 905680: 1, 895633: 1, 8712853: 1, 864496: 1, 88143502: 1, 91789: 1, 894604: 1, 907914: 1,
                    89346: 1, 8912521: 1, 868999: 1, 921362: 1, 894335: 1, 903811: 1, 8711561: 1, 925311: 1, 907915: 1,
                    869224: 1, 852631: 1, 894618: 1, 909410: 1, 8511133: 1, 916838: 1, 8910499: 1, 891936: 1, 913063: 1,
                    89827: 1, 8910506: 1, 874158: 1, 914062: 1, 918192: 1, 872113: 1, 875878: 1
                }
            },
            'Diagnosis': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'B': 357, 'M': 212
                }
            },
            'Mean Radius': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 6.981, 'max': 28.11,
                    'mean': 14.127291739894563, 'median': 13.37, 'std': 3.524048826212078,
                    'skewness': 0.9423795716730992, 'kurtosis': 0.8455216229065377
                }
            },
            'Mean Texture': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 9.71, 'max': 39.28,
                    'mean': 19.28964850615117, 'median': 18.84, 'std': 4.301035768166949,
                    'skewness': 0.6504495420828159, 'kurtosis': 0.7583189723727752
                }
            },
            'Mean Perimeter': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 43.79, 'max': 188.5,
                    'mean': 91.96903339191566, 'median': 86.24, 'std': 24.2989810387549,
                    'skewness': 0.9906504253930081, 'kurtosis': 0.9722135477110654
                }
            },
            'Mean Area': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 143.5, 'max': 2501.0,
                    'mean': 654.8891036906857, 'median': 551.1, 'std': 351.9141291816527,
                    'skewness': 1.6457321756240424, 'kurtosis': 3.6523027623507582
                }
            },
            'Mean Smoothness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.052629999999999996, 'max': 0.1634,
                    'mean': 0.096360281195079, 'median': 0.09587000000000001, 'std': 0.014064128137673618,
                    'skewness': 0.45632376481956155, 'kurtosis': 0.8559749303632262
                }
            },
            'Mean Compactness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.01938, 'max': 0.3454,
                    'mean': 0.10434098418277686, 'median': 0.09262999999999999, 'std': 0.0528127579325122,
                    'skewness': 1.1901230311980404, 'kurtosis': 1.650130467219256
                }
            },
            'Mean Concavity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.4268,
                    'mean': 0.08879931581722322, 'median': 0.06154, 'std': 0.0797198087078935,
                    'skewness': 1.4011797389486722, 'kurtosis': 1.9986375291042124
                }
            },
            'Mean Concave Points': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.2012,
                    'mean': 0.048919145869947236, 'median': 0.0335, 'std': 0.03880284485915359,
                    'skewness': 1.1711800812336282, 'kurtosis': 1.066555702965477
                }
            },
            'Mean Symmetry': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.106, 'max': 0.304,
                    'mean': 0.181161862917399, 'median': 0.1792, 'std': 0.027414281336035712,
                    'skewness': 0.7256089733642002, 'kurtosis': 1.2879329922294565
                }
            },
            'Mean Fractal Dimension': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.049960000000000004, 'max': 0.09744,
                    'mean': 0.06279760984182776, 'median': 0.06154, 'std': 0.007060362795084458,
                    'skewness': 1.3044888125755076, 'kurtosis': 3.005892120169494
                }
            },
            'Radius SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.1115, 'max': 2.873,
                    'mean': 0.4051720562390161, 'median': 0.3242, 'std': 0.2773127329861041,
                    'skewness': 3.088612166384756, 'kurtosis': 17.686725966164637
                }
            },
            'Texture SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.3602, 'max': 4.885,
                    'mean': 1.2168534270650269, 'median': 1.1079999999999999, 'std': 0.5516483926172023,
                    'skewness': 1.646443808753053, 'kurtosis': 5.349168692469973
                }
            },
            'Perimeter SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.757, 'max': 21.98,
                    'mean': 2.8660592267135288, 'median': 2.287, 'std': 2.021854554042107,
                    'skewness': 3.4436152021948976, 'kurtosis': 21.40190492588044
                }
            },
            'Area SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 6.8020000000000005, 'max': 542.2,
                    'mean': 40.33707908611603, 'median': 24.53, 'std': 45.49100551613178,
                    'skewness': 5.447186284898394, 'kurtosis': 49.20907650724119
                }
            },
            'Smoothness SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.001713, 'max': 0.03113,
                    'mean': 0.007040978910369071, 'median': 0.006379999999999999, 'std': 0.003002517943839067,
                    'skewness': 2.314450056636761, 'kurtosis': 10.469839532360393
                }
            },
            'Compactness SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.002252, 'max': 0.1354,
                    'mean': 0.025478138840070306, 'median': 0.02045, 'std': 0.017908179325677377,
                    'skewness': 1.9022207096378565, 'kurtosis': 5.10625248342338
                }
            },
            'Concavity SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.396,
                    'mean': 0.03189371634446394, 'median': 0.025889999999999996, 'std': 0.030186060322988394,
                    'skewness': 5.110463049043661, 'kurtosis': 48.8613953017919
                }
            },
            'Concave Points SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.05279,
                    'mean': 0.011796137082601056, 'median': 0.01093, 'std': 0.006170285174046866,
                    'skewness': 1.4446781446974788, 'kurtosis': 5.1263019430439565
                }
            },
            'Symmetry SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.007882, 'max': 0.07895,
                    'mean': 0.020542298769771532, 'median': 0.01873, 'std': 0.008266371528798399,
                    'skewness': 2.195132899547822, 'kurtosis': 7.896129827528971
                }
            },
            'Fractal Dimension SE': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0008948000000000001, 'max': 0.02984,
                    'mean': 0.0037949038664323374, 'median': 0.003187, 'std': 0.0026460709670891942,
                    'skewness': 3.923968620227413, 'kurtosis': 26.280847486373336
                }
            },
            'Worst Radius': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 7.93, 'max': 36.04,
                    'mean': 16.269189806678394, 'median': 14.97, 'std': 4.833241580469324,
                    'skewness': 1.1031152059604372, 'kurtosis': 0.9440895758772196
                }
            },
            'Worst Texture': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 12.02, 'max': 49.54,
                    'mean': 25.677223198594014, 'median': 25.41, 'std': 6.146257623038323,
                    'skewness': 0.49832130948716474, 'kurtosis': 0.22430186846478772
                }
            },
            'Worst Perimeter': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 50.41, 'max': 251.2,
                    'mean': 107.2612126537786, 'median': 97.66, 'std': 33.60254226903635,
                    'skewness': 1.1281638713683722, 'kurtosis': 1.070149666654432
                }
            },
            'Worst Area': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 185.2, 'max': 4254.0,
                    'mean': 880.5831282952545, 'median': 686.5, 'std': 569.3569926699492,
                    'skewness': 1.8593732724433467, 'kurtosis': 4.396394828992138
                }
            },
            'Worst Smoothness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.07117000000000001, 'max': 0.2226,
                    'mean': 0.13236859402460469, 'median': 0.1313, 'std': 0.022832429404835458,
                    'skewness': 0.4154259962824678, 'kurtosis': 0.5178251903311124
                }
            },
            'Worst Compactness': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.02729, 'max': 1.058,
                    'mean': 0.25426504393673144, 'median': 0.2119, 'std': 0.15733648891374194,
                    'skewness': 1.4735549003297963, 'kurtosis': 3.0392881719200675
                }
            },
            'Worst Concavity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.252,
                    'mean': 0.27218848330404205, 'median': 0.2267, 'std': 0.20862428060813235,
                    'skewness': 1.1502368219460262, 'kurtosis': 1.6152532975830205
                }
            },
            'Worst Concave Points': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.29100000000000004,
                    'mean': 0.11460622319859404, 'median': 0.09992999999999999, 'std': 0.0657323411959421,
                    'skewness': 0.49261552688550875, 'kurtosis': -0.5355351225188589
                }
            },
            'Worst Symmetry': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.1565, 'max': 0.6638,
                    'mean': 0.29007557117750454, 'median': 0.2822, 'std': 0.06186746753751869,
                    'skewness': 1.4339277651893279, 'kurtosis': 4.4445595178465815
                }
            },
            'Worst Fractal Dimension': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.05504, 'max': 0.2075,
                    'mean': 0.08394581722319855, 'median': 0.08004, 'std': 0.01806126734889399,
                    'skewness': 1.6625792663955172, 'kurtosis': 5.2446105558150125
                }
            }
        },      # ok
        'cover': {  # https://archive.ics.uci.edu/ml/datasets/Covertype

        },       # NOT ok
        'credit': {  # https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
            'ID': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    # for the sake of simplicity the values distribution for this variable is NOT provided here
                }
            },
            'LIMIT_BAL': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 10000, 'max': 1000000,
                    'mean': 167484.32266666667, 'median': 140000.0, 'std': 129747.66156719506,
                    'skewness': 0.992866960519544, 'kurtosis': 0.536262896398668
                }
            },
            'SEX': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    2: 18112, 1: 11888
                }
            },
            'EDUCATION': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    2: 14030, 1: 10585, 3: 4917, 5: 280, 4: 123, 6: 51, 0: 14
                }
            },
            'MARRIAGE': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    2: 15964, 1: 13659, 3: 323, 0: 54
                }
            },
            'AGE': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 21, 'max': 79,
                    'mean': 35.4855, 'median': 34.0, 'std': 9.217904068090183,
                    'skewness': 0.7322458687830562, 'kurtosis': 0.04430337823580954
                }
            },
            'PAY_0': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 14737, -1: 5686, 1: 3688, -2: 2759, 2: 2667, 3: 322, 4: 76, 5: 26, 8: 19, 6: 11, 7: 9
                }
            },
            'PAY_2': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 15730, -1: 6050, 2: 3927, -2: 3782, 3: 326, 4: 99, 1: 28, 5: 25, 7: 20, 6: 12, 8: 1
                }
            },
            'PAY_3': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 15764, -1: 5938, -2: 4085, 2: 3819, 3: 240, 4: 76, 7: 27, 6: 23, 5: 21, 1: 4, 8: 3
                }
            },
            'PAY_4': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 16455, -1: 5687, -2: 4348, 2: 3159, 3: 180, 4: 69, 7: 58, 5: 35, 6: 5, 8: 2, 1: 2
                }
            },
            'PAY_5': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 16947, -1: 5539, -2: 4546, 2: 2626, 3: 178, 4: 84, 7: 58, 5: 17, 6: 4, 8: 1
                }
            },
            'PAY_6': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 16286, -1: 5740, -2: 4895, 2: 2766, 3: 184, 4: 49, 7: 46, 6: 19, 5: 13, 8: 2
                }
            },
            'BILL_AMT1': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -165580, 'max': 964511,
                    'mean': 51223.3309, 'median': 22381.5, 'std': 73635.86057552874,
                    'skewness': 2.6638610220232612, 'kurtosis': 9.806289341330837
                }
            },
            'BILL_AMT2': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -69777, 'max': 983931,
                    'mean': 49179.07516666667, 'median': 21200.0, 'std': 71173.76878252918,
                    'skewness': 2.7052208534082856, 'kurtosis': 10.302945922629279
                }
            },
            'BILL_AMT3': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -157264, 'max': 1664089,
                    'mean': 47013.1548, 'median': 20088.5, 'std': 69349.38742703729,
                    'skewness': 3.0878300462007244, 'kurtosis': 19.783255144801103
                }
            },
            'BILL_AMT4': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -170000, 'max': 891586,
                    'mean': 43262.94896666666, 'median': 19052.0, 'std': 64332.85613391704,
                    'skewness': 2.8219652908028117, 'kurtosis': 11.309324826831903
                }
            },
            'BILL_AMT5': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -81334, 'max': 927171,
                    'mean': 40311.40096666667, 'median': 18104.5, 'std': 60797.155770264195,
                    'skewness': 2.8763798667028633, 'kurtosis': 12.30588128593057
                }
            },
            'BILL_AMT6': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -339603, 'max': 961664,
                    'mean': 38871.7604, 'median': 17071.0, 'std': 59554.10753674454,
                    'skewness': 2.8466445756603678, 'kurtosis': 12.270705286713094
                }
            },
            'PAY_AMT1': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 873552,
                    'mean': 5663.5805, 'median': 2100.0, 'std': 16563.280354026534,
                    'skewness': 14.66836433284317, 'kurtosis': 415.25474270738493
                }
            },
            'PAY_AMT2': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 1684259,
                    'mean': 5921.1635, 'median': 2009.0, 'std': 23040.870402054872,
                    'skewness': 30.45381745016943, 'kurtosis': 1641.6319110097434
                }
            },
            'PAY_AMT3': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 896040,
                    'mean': 5225.6815, 'median': 1800.0, 'std': 17606.96146980426,
                    'skewness': 17.216635435129238, 'kurtosis': 564.3112294697712
                }
            },
            'PAY_AMT4': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 621000,
                    'mean': 4826.076866666666, 'median': 1500.0, 'std': 15666.159744031342,
                    'skewness': 12.904984823542545, 'kurtosis': 277.3337677160758
                }
            },
            'PAY_AMT5': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 426529,
                    'mean': 4799.387633333334, 'median': 1500.0, 'std': 15278.30567914539,
                    'skewness': 11.127417052173817, 'kurtosis': 180.06394016204655
                }
            },
            'PAY_AMT6': {
                'var_type': 'continuous', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0, 'max': 528666,
                    'mean': 5215.502566666667, 'median': 1500.0, 'std': 17777.465775434066,
                    'skewness': 10.640727325044317, 'kurtosis': 167.16142958843525
                }
            },
            'default payment next month': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 23364, 1: 6636
                }
            }
        },      # ok
        'eeg': {  # https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
            'AF3': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1030.77, 'max': 309231.0, 'mean': 4321.917777036056, 'median': 4294.36,
                    'std': 2492.0721742651103, 'skewness': 122.29386525011812, 'kurtosis': 14963.840002182125
                }
            },
            'F7': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2830.77, 'max': 7804.62, 'mean': 4009.767693591461, 'median': 4005.64,
                    'std': 45.94167248479191, 'skewness': 39.046557690711396, 'kurtosis': 3210.171915006226
                }
            },
            'F3': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1040.0, 'max': 6880.51, 'mean': 4264.022432576795, 'median': 4262.56,
                    'std': 44.428051757419446, 'skewness': -13.615160740497625, 'kurtosis': 2921.967694389361
                }
            },
            'FC5': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2453.33, 'max': 642564.0, 'mean': 4164.9463264352735, 'median': 4120.51,
                    'std': 5216.40463229992, 'skewness': 122.38777688436551, 'kurtosis': 14979.17873537246
                }
            },
            'T7': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2089.74, 'max': 6474.36, 'mean': 4341.741075433922, 'median': 4338.97,
                    'std': 34.73882081848658, 'skewness': 7.561902122619299, 'kurtosis': 2578.229693199016
                }
            },
            'P7': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2768.21, 'max': 362564.0, 'mean': 4644.022379172214, 'median': 4617.95,
                    'std': 2924.7895373250954, 'skewness': 122.36281054964749, 'kurtosis': 14975.088891085208
                }
            },
            'O1': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2086.15, 'max': 567179.0, 'mean': 4110.400159546061, 'median': 4070.26,
                    'std': 4600.926542533738, 'skewness': 122.38359282836112, 'kurtosis': 14978.495281856303
                }
            },
            'O2': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4567.18, 'max': 7264.1, 'mean': 4616.056903871852, 'median': 4613.33,
                    'std': 29.292603201776014, 'skewness': 51.09721901768454, 'kurtosis': 4491.11404630705
                }
            },
            'P8': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1357.95, 'max': 265641.0, 'mean': 4218.826610146869, 'median': 4199.49,
                    'std': 2136.4085228873855, 'skewness': 122.33467120005805, 'kurtosis': 14970.509845636481
                }
            },
            'T8': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1816.41, 'max': 6674.36, 'mean': 4231.316199599468, 'median': 4229.23,
                    'std': 38.05090262121652, 'skewness': 10.23070102200852, 'kurtosis': 2710.083429497155
                }
            },
            'FC6': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 3273.33, 'max': 6823.08, 'mean': 4202.456899866489, 'median': 4200.51,
                    'std': 37.78598137403701, 'skewness': 31.649004824348655, 'kurtosis': 2056.5210594418677
                }
            },
            'F4': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2257.95, 'max': 7002.56, 'mean': 4279.232774365836, 'median': 4276.92,
                    'std': 41.54431151666411, 'skewness': 26.556468850889043, 'kurtosis': 2714.7186392430435
                }
            },
            'F8': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 86.6667, 'max': 152308.0, 'mean': 4615.205335560761, 'median': 4603.08,
                    'std': 1208.3699582560462, 'skewness': 121.90727242585102, 'kurtosis': 14901.910996495131
                }
            },
            'AF4': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1366.15, 'max': 715897.0, 'mean': 4416.435832443256, 'median': 4354.87,
                    'std': 5891.2850425236575, 'skewness': 118.1250449998719, 'kurtosis': 14214.276393221251
                }
            },
            'Eye Detection': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {0: 8257, 1: 6723}
            }
        },         # ok
        'iris': {  # https://archive.ics.uci.edu/ml/datasets/Iris
            'sepal length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4.3, 'max': 7.9, 'mean': 5.843333333333335, 'median': 5.8,
                    'std': 0.8280661279778629, 'skewness': 0.3149109566369728, 'kurtosis': -0.5520640413156395
                }
            },
            'sepal width': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.0, 'max': 4.4, 'mean': 3.0540000000000007, 'median': 3.0,
                    'std': 0.4335943113621737, 'skewness': 0.3340526621720866, 'kurtosis': 0.2907810623654279
                }
            },
            'petal length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.0, 'max': 6.9, 'mean': 3.7586666666666693, 'median': 4.35,
                    'std': 1.7644204199522617, 'skewness': -0.27446425247378287, 'kurtosis': -1.4019208006454036
                }
            },
            'petal width': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.1, 'max': 2.5, 'mean': 1.1986666666666672, 'median': 1.3,
                    'std': 0.7631607417008414, 'skewness': -0.10499656214412734, 'kurtosis': -1.3397541711393433
                }
            },
            'class': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Iris-virginica': 50, 'Iris-versicolor': 50, 'Iris-setosa': 50
                }
            }
        },        # ok
        'iris_sample': {  # https://archive.ics.uci.edu/ml/datasets/Iris
            'sepal length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4.4, 'max': 7.6, 'mean': 5.846666666666666, 'median': 5.8,
                    'std': 0.9515901478702947, 'skewness': 0.13492393266856537, 'kurtosis': -0.9072164547039541
                }
            },
            'sepal width': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.4, 'max': 3.9, 'mean': 3.066666666666667, 'median': 3.0,
                    'std': 0.3848314411469495, 'skewness': 0.6057690358311313, 'kurtosis': 0.33387944547878945
                }
            },
            'petal length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.4, 'max': 6.6, 'mean': 3.8733333333333335, 'median': 4.6,
                    'std': 1.9069297187838234, 'skewness': -0.29586425993697496, 'kurtosis': -1.569734914760088
                }
            },
            'petal width': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.2, 'max': 2.2, 'mean': 1.2066666666666668, 'median': 1.3,
                    'std': 0.7666873703208654, 'skewness': -0.23840004554489685, 'kurtosis': -1.6015576473253543
                }
            },
            'class': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'Iris-setosa': 5, 'Iris-versicolor': 5, 'Iris-virginica': 5
                }
            }
        },  # ok
        'letter': {  # https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
            'letter': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'U': 813, 'D': 805, 'P': 803, 'T': 796, 'M': 792, 'A': 789, 'X': 787, 'Y': 786, 'Q': 783,
                    'N': 783, 'F': 775, 'G': 773, 'E': 768, 'B': 766, 'V': 764, 'L': 761, 'R': 758, 'I': 755,
                    'O': 753, 'W': 752, 'S': 748, 'J': 747, 'K': 739, 'C': 736, 'Z': 734, 'H': 734
                }
            },
            'x-box': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    4: 4477, 3: 4157, 5: 3169, 2: 2909, 6: 1894, 1: 1261, 7: 1006, 8: 513,
                    9: 284, 0: 132, 10: 121, 11: 48, 12: 20, 13: 4, 14: 3, 15: 2
                }
            },
            'y-box': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    9: 2702, 7: 2302, 10: 2211, 8: 2180, 6: 1705, 11: 1625, 5: 1566, 4: 1342,
                    3: 1330, 1: 778, 0: 709, 2: 530, 12: 321, 13: 271, 15: 230, 14: 198
                }
            },
            'width': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    5: 4262, 4: 3816, 6: 3641, 3: 1994, 7: 1946, 8: 1418, 2: 1285, 9: 679,
                    1: 385, 10: 237, 0: 195, 11: 91, 12: 39, 13: 6, 14: 4, 15: 2
                }
            },
            'high': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    6: 3656, 8: 3559, 4: 2718, 7: 2695, 5: 2675, 3: 1559, 2: 1304, 1: 883,
                    0: 365, 9: 347, 10: 103, 11: 76, 12: 31, 14: 15, 13: 10, 15: 4
                }
            },
            'onpix': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    2: 4153, 3: 3939, 4: 3157, 1: 2437, 5: 2153, 6: 1379, 7: 857, 0: 830,
                    8: 519, 9: 283, 10: 142, 11: 85, 12: 40, 13: 13, 14: 7, 15: 6
                }
            },
            'x-bar': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    7: 6024, 8: 4019, 6: 2717, 5: 1780, 9: 1752, 4: 1069, 10: 802, 3: 680,
                    11: 338, 12: 201, 1: 179, 2: 167, 0: 148, 13: 67, 14: 43, 15: 14
                }
            },
            'y-bar': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    7: 6010, 8: 3753, 6: 2506, 9: 1599, 10: 1252, 11: 1131, 5: 877, 4: 714,
                    12: 663, 3: 488, 2: 393, 13: 192, 14: 165, 1: 134, 15: 76, 0: 47
                }
            },
            'x2bar': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    3: 3424, 4: 3199, 5: 2982, 2: 2693, 6: 2340, 7: 1422, 1: 1084, 8: 1013,
                    9: 436, 0: 422, 10: 235, 14: 184, 12: 158, 11: 150, 15: 138, 13: 120
                }
            },
            'y2bar': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    5: 3243, 6: 3099, 4: 3011, 7: 2623, 3: 1914, 2: 1852, 8: 1688, 9: 874,
                    1: 822, 10: 318, 0: 269, 11: 128, 12: 48, 14: 46, 15: 34, 13: 31
                }
            },
            'xybar': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    7: 5578, 6: 2751, 10: 2563, 9: 1971, 11: 1799, 8: 1668, 12: 1043, 5: 819,
                    13: 772, 14: 279, 4: 267, 0: 145, 2: 97, 15: 96, 3: 90, 1: 62
                }
            },
            'x2ybr': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    6: 5666, 7: 2598, 5: 2457, 8: 1495, 4: 1495, 9: 1408, 11: 935, 10: 888,
                    2: 887, 3: 726, 12: 472, 1: 430, 13: 200, 0: 188, 14: 135, 15: 20
                }
            },
            'xy2br': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    8: 6548, 9: 3000, 7: 2807, 6: 1874, 10: 1437, 5: 1335, 11: 926, 4: 701,
                    12: 409, 13: 313, 3: 269, 14: 219, 15: 85, 2: 67, 1: 9, 0: 1
                }
            },
            'x-edge': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    3: 4779, 2: 4213, 1: 2568, 0: 2461, 4: 1500, 5: 1363, 6: 1264, 7: 722,
                    8: 587, 9: 246, 10: 154, 11: 81, 12: 29, 13: 17, 14: 12, 15: 4
                }
            },
            'xegvy': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    8: 7624, 9: 3437, 7: 2516, 10: 2394, 6: 1592, 11: 1437, 5: 416, 12: 348,
                    4: 79, 13: 72, 3: 34, 2: 17, 14: 16, 1: 13, 15: 4, 0: 1
                }
            },
            'y-edge': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    4: 3091, 3: 3078, 2: 2475, 0: 2472, 5: 2048, 1: 2040, 6: 1723, 7: 1227,
                    8: 973, 9: 613, 10: 154, 11: 64, 12: 22, 13: 13, 15: 4, 14: 3
                }
            },
            'yegvx': {
                'var_type': 'discrete', 'data_type': int, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    8: 8047, 7: 3472, 9: 2358, 6: 1827, 10: 1578, 5: 992, 11: 868, 4: 478,
                    12: 137, 3: 130, 13: 49, 2: 30, 1: 17, 14: 13, 15: 2, 0: 2
                }
            }
        },      # ok
        'mushroom': {  # https://archive.ics.uci.edu/ml/datasets/Mushroom
            'class': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'e', 4208, 'p', 3916
                }
            },
            'cap-shape': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 452, 'c', 4, 'f', 3152, 'k', 828, 's', 32, 'x', 3656
                }
            },
            'cap-surface': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f', 2320, 'g', 4, 's', 2556, 'y', 3244
                }
            },
            'cap-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 168, 'c', 44, 'e', 1500, 'g', 1840, 'n', 2284, 'p', 144, 'r', 16, 'u', 16, 'w', 1040, 'y', 1072
                }
            },
            'bruises': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f': 4748, 't': 3376
                }
            },
            'odor': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'a', 400, 'c', 192, 'f', 2160, 'l', 400, 'm', 36, 'n', 3528, 'p', 256, 's', 576, 'y', 576
                }
            },
            'gill-attachment': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'a': 210, 'f': 7914
                }
            },
            'gill-spacing': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'c': 6812, 'w': 1312
                }
            },
            'gill-size': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b': 5612, 'n': 2512
                }
            },
            'gill-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 1728, 'e', 96, 'g', 752, 'h', 732, 'k', 408, 'n', 1048,
                    'o', 64, 'p', 1492, 'r', 24, 'u', 492, 'w', 1202, 'y', 86
                }
            },
            'stalk-shape': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'e': 3516, 't': 4608
                }
            },
            'stalk-root': {  # drop feature/variable --> has missing values
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': True, 'missing_values': ['?'],
                'values_dist': {
                    'b': 3776, 'c': 556, 'e': 1120, 'r': 192, '?': 2480
                }
            },
            'stalk-surface-above-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f', 552, 'k', 2372, 's', 5176, 'y', 24
                }
            },
            'stalk-surface-below-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'f', 600, 'k', 2304, 's', 4936, 'y', 284
                }
            },
            'stalk-color-above-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 432, 'c', 36, 'e', 96, 'g', 576, 'n', 448, 'o', 192, 'p', 1872, 'w', 4464, 'y', 8
                }
            },
            'stalk-color-below-ring': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 432, 'c', 36, 'e', 96, 'g', 576, 'n', 512, 'o', 192, 'p', 1872, 'w', 4384, 'y', 24
                }
            },
            'veil-type': {  # this feature contributes with NOTHING to any ML task --> variance = 0
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    'p': 8124
                }
            },
            'veil-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'n', 96, 'o', 96, 'w', 7924, 'y', 8
                }
            },
            'ring-number': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'n', 36, 'o', 7488, 't', 600
                }
            },
            'ring-type': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'e', 2776, 'f', 48, 'l', 1296, 'n', 36, 'p', 3968
                }
            },
            'spore-print-color': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'b', 48, 'h', 1632, 'k', 1872, 'n', 1968, 'o', 48, 'r', 72, 'u', 48, 'w', 2388, 'y', 48
                }
            },
            'population': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'a', 384, 'c', 340, 'n', 400, 's', 1248, 'v', 4040, 'y', 1712
                }
            },
            'habitat': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'd', 3148, 'g', 2148, 'l', 832, 'm', 292, 'p', 1144, 'u', 368, 'w', 192
                }
            }
        },    # ok
        'news': {  # https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
            'url': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {}  # still NOT sure what to do in terms of metadata for `str` variables/features/columns
            },
            'timedelta': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    'min': 8.0, 'max': 731.0,
                    'mean': 354.53047119362327, 'median': 339.0, 'std': 214.16376716976876,
                    'skewness': 0.12050427403732356, 'kurtosis': -1.2571905462199464
                }
            },
            'n_tokens_title': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.0, 'max': 23.0,
                    'mean': 10.398748864897588, 'median': 10.0, 'std': 2.11403680830354,
                    'skewness': 0.16532037674928027, 'kurtosis': -0.0007496707024086113
                }
            },
            'n_tokens_content': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 8474.0,
                    'mean': 546.514731106851, 'median': 409.0, 'std': 471.10750794804716,
                    'skewness': 2.9454219387867084, 'kurtosis': 19.478411622411965
                }
            },
            'n_unique_tokens': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 701.0,
                    'mean': 0.5482157168486088, 'median': 0.5392255481715, 'std': 3.5207083312411243,
                    'skewness': 198.65511559825592, 'kurtosis': 39523.83200079352
                }
            },
            'n_non_stop_words': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1042.0,
                    'mean': 0.9964685654578603, 'median': 0.999999995968, 'std': 5.231230945150023,
                    'skewness': 198.7924453768874,  'kurtosis': 39560.294949690826
                }
            },
            'n_non_stop_unique_tokens': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 650.0,
                    'mean': 0.6891753940079369, 'median': 0.6904761882745, 'std': 3.2648163548136893,
                    'skewness': 198.44329440926512, 'kurtosis': 39467.693707784514
                }
            },
            'num_hrefs': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 304.0,
                    'mean': 10.883689839572192, 'median': 8.0, 'std': 11.332017376010986,
                    'skewness': 4.013494828201318, 'kurtosis': 35.5063328441666
                }
            },
            'num_self_hrefs': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 116.0,
                    'mean': 3.2936383815962063, 'median': 3.0, 'std': 3.8551411453741316,
                    'skewness': 5.172751105757634, 'kurtosis': 56.17145631073022
                }
            },
            'num_imgs': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 128.0,
                    'mean': 4.544142871556856, 'median': 1.0, 'std': 8.309433519603113,
                    'skewness': 3.9465958446535474, 'kurtosis': 24.525745897623377
                }
            },
            'num_videos': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 91.0,
                    'mean': 1.2498738775098375, 'median': 0.0, 'std': 4.107855086225022,
                    'skewness': 7.0195327862958665, 'kurtosis': 74.07541389830322
                }
            },
            'average_token_length': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 8.04153354633,
                    'mean': 4.548239318341858, 'median': 4.664082178605, 'std': 0.844405565195359,
                    'skewness': -4.57601155020474, 'kurtosis': 22.180449878813747
                }
            },
            'num_keywords': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.0, 'max': 10.0,
                    'mean': 7.223766522046211, 'median': 7.0, 'std': 1.9091303859704352,
                    'skewness': -0.14725125199950523, 'kurtosis': -0.8058971527749366
                }
            },
            'data_channel_is_lifestyle': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.05294622137019473, 'median': 0.0, 'std': 0.2239289706014597,
                    'skewness': 3.9930191433554167, 'kurtosis': 13.944905383135367}
            },
            'data_channel_is_entertainment': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.17800928261527596, 'median': 0.0, 'std': 0.3825253833532349,
                    'skewness': 1.68358480940472, 'kurtosis': 0.8344999075957689
                }
            },
            'data_channel_is_bus': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.15785490868731714, 'median': 0.0, 'std': 0.3646095032189732,
                    'skewness': 1.8768701859879158, 'kurtosis': 1.5227185121254707
                }
            },
            'data_channel_is_socmed': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.0585965089294723, 'median': 0.0, 'std': 0.234870921068934,
                    'skewness': 3.758879630973088, 'kurtosis': 12.12978801322776
                }
            },
            'data_channel_is_tech': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.1852991625466653, 'median': 0.0, 'std': 0.38854496648089887,
                    'skewness': 1.6199757646890423, 'kurtosis': 0.6243529736154327
                }
            },
            'data_channel_is_world': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.21256684491978609, 'median': 0.0, 'std': 0.409128834963538,
                    'skewness': 1.4051693841208097, 'kurtosis': -0.025500290938787806
                }
            },
            'kw_min_min': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 377.0,
                    'mean': 26.10680052466956, 'median': -1.0, 'std': 69.6332151214891,
                    'skewness': 2.3749472801825444, 'kurtosis': 3.6600028208082196
                }
            },
            'kw_max_min': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 298400.0,
                    'mean': 1153.951682221782, 'median': 660.0, 'std': 3857.9908765299933,
                    'skewness': 35.32843373115432, 'kurtosis': 2100.070757658957
                }
            },
            'kw_avg_min': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 42827.8571429,
                    'mean': 312.36696679715607, 'median': 235.5, 'std': 620.7838873138074,
                    'skewness': 31.306108102660584, 'kurtosis': 1592.2443533180226
                }
            },
            'kw_min_max': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 843300.0,
                    'mean': 13612.35410150338, 'median': 1400.0, 'std': 57986.02935737641,
                    'skewness': 10.386371634782769, 'kurtosis': 123.43210860504985
                }
            },
            'kw_max_max': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 843300.0,
                    'mean': 752324.0666935728, 'median': 843300.0, 'std': 214502.12957278264,
                    'skewness': -2.6449817621966782, 'kurtosis': 5.7238508359874025
                }
            },
            'kw_avg_max': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 843300.0,
                    'mean': 259281.93808264713, 'median': 244572.22222250002, 'std': 135102.2472847749,
                    'skewness': 0.6243096463608956, 'kurtosis': 0.8305209909510953
                }
            },
            'kw_min_avg': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 3613.0398195000002,
                    'mean': 1117.1466099698375, 'median': 1023.6356107649999, 'std': 1137.4569507729946,
                    'skewness': 0.4679758464905322, 'kurtosis': -1.1264404764395628
                }
            },
            'kw_max_avg': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 298400.0,
                    'mean': 5657.211151064998, 'median': 4355.68883632, 'std': 6098.871956751086,
                    'skewness': 16.41166955537124, 'kurtosis': 481.9265124506484
                }
            },
            'kw_avg_avg': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 43567.6599458,
                    'mean': 3135.8586389465395, 'median': 2870.07487821, 'std': 1318.1503970937204,
                    'skewness': 5.760177291618559, 'kurtosis': 100.58610802540827
                }
            },
            'self_reference_min_shares': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 843300.0,
                    'mean': 3998.75539552013, 'median': 1200.0, 'std': 19738.67051625999,
                    'skewness': 26.264364160300094, 'kurtosis': 864.8911257753615
                }
            },
            'self_reference_max_shares': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 843300.0,
                    'mean': 10329.212661941277, 'median': 2800.0, 'std': 41027.57661292357,
                    'skewness': 13.870849049433598, 'kurtosis': 224.1617475049218
                }
            },
            'self_reference_avg_sharess': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 843300.0,
                    'mean': 6401.69757982146, 'median': 2200.0, 'std': 24211.332231313183,
                    'skewness': 17.9140933776756, 'kurtosis': 428.499441269726
                }
            },
            'weekday_is_monday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.16802038139441025, 'median': 0.0, 'std': 0.3738890999216466,
                    'skewness': 1.7759082442285052, 'kurtosis': 1.1539083028891146
                }
            },
            'weekday_is_tuesday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.18640904046009485, 'median': 0.0, 'std': 0.3894413123393486,
                    'skewness': 1.6105470619092879, 'kurtosis': 0.5938917973244595
                }
            },
            'weekday_is_wednesday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.18754414287155685, 'median': 0.0, 'std': 0.39035263664233,
                    'skewness': 1.6009709768881089, 'kurtosis': 0.5631364759627973
                }
            },
            'weekday_is_thursday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.18330642720209867, 'median': 0.0, 'std': 0.3869224176522378,
                    'skewness': 1.6370700482983118, 'kurtosis': 0.6800326474546692
                }
            },
            'weekday_is_friday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.14380486328322067, 'median': 0.0, 'std': 0.3508961818323727,
                    'skewness': 2.030304835180609, 'kurtosis': 2.122244786331605
                }
            },
            'weekday_is_saturday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.06187569367369589, 'median': 0.0, 'std': 0.2409326803316144,
                    'skewness': 3.6370857599701125, 'kurtosis': 11.228959312557096
                }
            },
            'weekday_is_sunday': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.06903945111492281, 'median': 0.0, 'std': 0.25352441026427036,
                    'skewness': 3.3999273763003046, 'kurtosis': 9.559988453391597
                }
            },
            'is_weekend': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.1309151447886187, 'median': 0.0, 'std': 0.3373117840779847,
                    'skewness': 2.18850033431371, 'kurtosis': 2.7896744470214334
                }
            },
            'LDA_00': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.926994384393,
                    'mean': 0.18459904828385093, 'median': 0.03338735938015, 'std': 0.2629747091334928,
                    'skewness': 1.5674632332004765, 'kurtosis': 1.0610988132338393
                }
            },
            'LDA_01': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.9259469861380001,
                    'mean': 0.1412557731283027, 'median': 0.03334502936825, 'std': 0.21970732884667032,
                    'skewness': 2.0867218234169407, 'kurtosis': 3.3456040411400845
                }
            },
            'LDA_02': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.9199990886660001,
                    'mean': 0.2163209667730632, 'median': 0.04000393631995, 'std': 0.28214520389036624,
                    'skewness': 1.311694902028395, 'kurtosis': 0.26185891448890963
                }
            },
            'LDA_03': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.926533782685,
                    'mean': 0.22376961651356606, 'median': 0.040000715639749995, 'std': 0.2951907334780088,
                    'skewness': 1.2387159863782737, 'kurtosis': -0.029651763833265843
                }
            },
            'LDA_04': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.9271908092579999,
                    'mean': 0.23402937080318217, 'median': 0.040727352783049994, 'std': 0.28918347808148437,
                    'skewness': 1.1731294759766238, 'kurtosis': -0.08148949990956256
                }
            },
            'global_subjectivity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.44337019955073476, 'median': 0.4534573473025, 'std': 0.11668464244606234,
                    'skewness': -1.3726888305603973, 'kurtosis': 4.611593207028966
                }
            },
            'global_sentiment_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -0.39375, 'max': 0.727840909091,
                    'mean': 0.11930926928571718, 'median': 0.119116768648, 'std': 0.09693066125190916,
                    'skewness': 0.10545709665820545, 'kurtosis': 1.5099316087025527
                }
            },
            'global_rate_positive_words': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.155487804878,
                    'mean': 0.039624833009040926, 'median': 0.03902277713255, 'std': 0.01742865786264911,
                    'skewness': 0.3230466111504906, 'kurtosis': 1.019328835728062
                }
            },
            'global_rate_negative_words': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.184931506849,
                    'mean': 0.016612119534643206, 'median': 0.015337423312899999, 'std': 0.010827792187851662,
                    'skewness': 1.491917309190822, 'kurtosis': 6.975001442538169
                }
            },
            'rate_positive_words': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.68215022110031, 'median': 0.7105263157889999, 'std': 0.19020632466079915,
                    'skewness': -1.423105853002299, 'kurtosis': 3.2752393993847337
                }
            },
            'rate_negative_words': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.28793352423315954, 'median': 0.28, 'std': 0.1561558879818461,
                    'skewness': 0.4072406539941201,'kurtosis': 0.5215870502682196
                }
            },
            'avg_positive_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.35382494340707044, 'median': 0.3587551652895, 'std': 0.10454218830821896,
                    'skewness': -0.7247949503201233, 'kurtosis': 3.3808518014994378
                }
            },
            'min_positive_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.09544553807022542, 'median': 0.1, 'std': 0.07131493174568221,
                    'skewness': 3.0404677374643283, 'kurtosis': 17.417102575004463
                }
            },
            'max_positive_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.7567275436994538, 'median': 0.8, 'std': 0.24778571839481928,
                    'skewness': -0.9397564591253907, 'kurtosis': 0.662354307112476
                }
            },
            'avg_negative_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 0.0,
                    'mean': -0.25952414109091587, 'median': -0.25333333333299995, 'std': 0.12772572202940655,
                    'skewness': -0.5516440290009489, 'kurtosis': 2.369671831225205
                }
            },
            'min_negative_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 0.0,
                    'mean': -0.5219437277484072, 'median': -0.5, 'std': 0.29028950220936445,
                    'skewness': -0.07315481617331099, 'kurtosis': -0.8265223064758707
                }
            },
            'max_negative_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 0.0,
                    'mean': -0.10750024015635165, 'median': -0.1, 'std': 0.09537298483252668,
                    'skewness': -3.4597470578480207, 'kurtosis': 19.567564730812794
                }
            },
            'title_subjectivity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.2823531841135558, 'median': 0.15, 'std': 0.32424737528929615,
                    'skewness': 0.816084749635643, 'kurtosis': -0.5405391617060173
                }
            },
            'title_sentiment_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': -1.0, 'max': 1.0,
                    'mean': 0.0714254319759339, 'median': 0.0, 'std': 0.26545022913425936,
                    'skewness': 0.39610883665169594, 'kurtosis': 3.2354769696806858
                }
            },
            'abs_title_subjectivity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.5,
                    'mean': 0.34184275596034974, 'median': 0.5, 'std': 0.18879080475303386,
                    'skewness': -0.6241493828840421, 'kurtosis': -1.2787700753041948
                }
            },
            'abs_title_sentiment_polarity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.15606366285653878, 'median': 0.0, 'std': 0.22629419772609116,
                    'skewness': 1.7041934399140888, 'kurtosis': 2.664140916447295
                }
            },
            'shares': {  # target feature/variable
                'var_type': 'continuous', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1, 'max': 843300,
                    'mean': 3395.3801836343455, 'median': 1400.0, 'std': 11626.950748651716,
                    'skewness': 33.96388487571415, 'kurtosis': 1832.6726571600288
                }
            }
        },        # ok
        'spam': {  # https://archive.ics.uci.edu/ml/datasets/Spambase
            'word_freq_make': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.54,
                    'mean': 0.10455335796565962, 'median': 0.0, 'std': 0.3053575620234765,
                    'skewness': 5.675639163765851, 'kurtosis': 49.30506415870118
                }
            },
            'word_freq_address': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 14.28,
                    'mean': 0.21301456205172783, 'median': 0.0, 'std': 1.290575190945365,
                    'skewness': 10.086810872543571, 'kurtosis': 105.64747160937861
                }
            },
            'word_freq_all': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 5.1,
                    'mean': 0.2806563790480323, 'median': 0.0, 'std': 0.5041428838471751,
                    'skewness': 3.0092485272970904, 'kurtosis': 13.308743418467486
                }
            },
            'word_freq_3d': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 42.81,
                    'mean': 0.06542490762877635, 'median': 0.0, 'std': 1.3951513704927585,
                    'skewness': 26.227744470347666, 'kurtosis': 726.4515380990603
                }
            },
            'word_freq_our': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 10.0,
                    'mean': 0.3122234296891974, 'median': 0.0, 'std': 0.6725127692846654,
                    'skewness': 4.747126114133084, 'kurtosis': 37.94116889421045
                }
            },
            'word_freq_over': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 5.88,
                    'mean': 0.09590089111062813, 'median': 0.0, 'std': 0.273824083009814,
                    'skewness': 5.956952736312442, 'kurtosis': 68.44525800064235
                }
            },
            'word_freq_remove': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 7.27,
                    'mean': 0.11420778091719185, 'median': 0.0, 'std': 0.39144135475059055,
                    'skewness': 6.7655804692611206, 'kurtosis': 75.41343865108215
                }
            },
            'word_freq_internet': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 11.11,
                    'mean': 0.10529450119539223, 'median': 0.0, 'std': 0.40107145247361387,
                    'skewness': 9.724847529977309, 'kurtosis': 169.16287621284124
                }
            },
            'word_freq_order': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 5.26,
                    'mean': 0.0900673766572484, 'median': 0.0, 'std': 0.27861586424185475,
                    'skewness': 5.226066951054146, 'kurtosis': 46.94025552319656
                }
            },
            'word_freq_mail': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 18.18,
                    'mean': 0.23941317104977197, 'median': 0.0, 'std': 0.6447553994517325,
                    'skewness': 8.487809520414334, 'kurtosis': 161.21464053175148
                }
            },
            'word_freq_receive': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 2.61,
                    'mean': 0.05982395131493154, 'median': 0.0, 'std': 0.20154466405444324,
                    'skewness': 5.510250153233095, 'kurtosis': 39.65094521268408
                }
            },
            'word_freq_will': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 9.67,
                    'mean': 0.5417018039556623, 'median': 0.1, 'std': 0.8616984712807816,
                    'skewness': 2.8673535839033857, 'kurtosis': 12.550746661368029
                }
            },
            'word_freq_people': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 5.55,
                    'mean': 0.09392958052597254, 'median': 0.0, 'std': 0.3010358035849348,
                    'skewness': 6.955548226899194, 'kurtosis': 84.94182188187557
                }
            },
            'word_freq_report': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 10.0,
                    'mean': 0.058626385568354686, 'median': 0.0, 'std': 0.3351838297557633,
                    'skewness': 11.75464547524248, 'kurtosis': 229.20127117236146
                }
            },
            'word_freq_addresses': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.41,
                    'mean': 0.04920452075635739, 'median': 0.0, 'std': 0.2588434513226294,
                    'skewness': 6.971040823271989, 'kurtosis': 57.72767594499322
                }
            },
            'word_freq_free': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 20.0,
                    'mean': 0.24884807650510785, 'median': 0.0, 'std': 0.825791701128899,
                    'skewness': 10.763594029506557, 'kurtosis': 196.42497538751527
                }
            },
            'word_freq_business': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 7.14,
                    'mean': 0.14258639426211697, 'median': 0.0, 'std': 0.44405532898213196,
                    'skewness': 5.688642098501172, 'kurtosis': 45.67377542768852
                }
            },
            'word_freq_email': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 9.09,
                    'mean': 0.18474462073462303, 'median': 0.0, 'std': 0.5311224220070001,
                    'skewness': 5.4137537226351125, 'kurtosis': 47.96167443680963
                }
            },
            'word_freq_you': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 18.75,
                    'mean': 1.6620995435774817, 'median': 1.31, 'std': 1.7754806647661487,
                    'skewness': 1.5916742687064176, 'kurtosis': 5.257394367988139
                }
            },
            'word_freq_credit': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 18.18,
                    'mean': 0.08557704846772435, 'median': 0.0, 'std': 0.5097668889532889,
                    'skewness': 14.602586644736904, 'kurtosis': 383.0018817314078
                }
            },
            'word_freq_your': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 11.11,
                    'mean': 0.8097609215387966, 'median': 0.22, 'std': 1.2008098116265178,
                    'skewness': 2.435527175852193, 'kurtosis': 9.009506007508783
                }
            },
            'word_freq_font': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 17.1,
                    'mean': 0.12120191262768967, 'median': 0.0, 'std': 1.0257555914313832,
                    'skewness': 9.97544072677198, 'kurtosis': 109.14232507803064
                }
            },
            'word_freq_000': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 5.45,
                    'mean': 0.10164529450119543, 'median': 0.0, 'std': 0.35028641855801296,
                    'skewness': 5.713775497993735, 'kurtosis': 46.80785977291932
                }
            },
            'word_freq_money': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 12.5,
                    'mean': 0.0942686372527711, 'median': 0.0, 'std': 0.4426355267683524,
                    'skewness': 14.687028058263406, 'kurtosis': 302.0564085457346
                }
            },
            'word_freq_hp': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 20.83,
                    'mean': 0.5495044555531406, 'median': 0.0, 'std': 1.6713493422823613,
                    'skewness': 5.716843442866135, 'kurtosis': 43.603633704848
                }
            },
            'word_freq_hpl': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 16.66,
                    'mean': 0.2653836122582047, 'median': 0.0, 'std': 0.8869553378203145,
                    'skewness': 6.3500116031273555, 'kurtosis': 63.900394418657434
                }
            },
            'word_freq_george': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 33.33,
                    'mean': 0.767304933710063, 'median': 0.0, 'std': 3.3672918024526606,
                    'skewness': 5.744493293820434, 'kurtosis': 34.20447595624731
                }
            },
            'word_freq_650': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 9.09,
                    'mean': 0.12484459900021724, 'median': 0.0, 'std': 0.5385760440208041,
                    'skewness': 6.606533933409737, 'kurtosis': 58.37302181110971
                }
            },
            'word_freq_lab': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 14.28,
                    'mean': 0.09891545316235598, 'median': 0.0, 'std': 0.5933265997216176,
                    'skewness': 11.370231659312, 'kurtosis': 175.24825133967164
                }
            },
            'word_freq_labs': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 5.88,
                    'mean': 0.10285155400999778, 'median': 0.0, 'std': 0.4566815529283778,
                    'skewness': 6.636015224797515, 'kurtosis': 52.006797967304486
                }
            },
            'word_freq_telnet': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 12.5,
                    'mean': 0.06475331449684846, 'median': 0.0, 'std': 0.4033925009035417,
                    'skewness': 12.669081126154037,  'kurtosis': 254.23250863507016
                }
            },
            'word_freq_857': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.76,
                    'mean': 0.04704846772440773, 'median': 0.0, 'std': 0.32855888168990294,
                    'skewness': 10.549184168787116, 'kurtosis': 127.37652934848592
                }
            },
            'word_freq_data': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 18.18,
                    'mean': 0.09722886329058895, 'median': 0.0, 'std': 0.5559072037057607,
                    'skewness': 13.190055836835201, 'kurtosis': 296.0883383057452
                }
            },
            'word_freq_415': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.76,
                    'mean': 0.04783525320582481, 'median': 0.0, 'std': 0.3294453306636256,
                    'skewness': 10.475181193648435, 'kurtosis': 125.94333200357252
                }
            },
            'word_freq_85': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 20.0,
                    'mean': 0.10541186698543797, 'median': 0.0, 'std': 0.5322598753371446,
                    'skewness': 15.230811463318567, 'kurtosis': 449.3742708196391
                }
            },
            'word_freq_technology': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 7.69,
                    'mean': 0.09747663551401871, 'median': 0.0, 'std': 0.40262313907304403,
                    'skewness': 7.6734613482809495, 'kurtosis': 81.20727605761869
                }
            },
            'word_freq_1999': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 6.89,
                    'mean': 0.1369528363399261, 'median': 0.0, 'std': 0.423451367456984,
                    'skewness': 5.323491678805356, 'kurtosis': 42.621043203171965
                }
            },
            'word_freq_parts': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 8.33,
                    'mean': 0.013201477939578352, 'median': 0.0,
                    'std': 0.22065078755805823, 'skewness': 28.263215566652605, 'kurtosis': 912.0457217576984
                }
            },
            'word_freq_pm': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 11.11,
                    'mean': 0.0786285590089111, 'median': 0.0, 'std': 0.4346720518434216,
                    'skewness': 12.056911999154961, 'kurtosis': 215.71814408609066
                }
            },
            'word_freq_direct': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.76,
                    'mean': 0.06483373179743535, 'median': 0.0, 'std': 0.3499159830230198,
                    'skewness': 9.147029409140545, 'kurtosis': 99.38656121877835
                }
            },
            'word_freq_cs': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 7.14,
                    'mean': 0.04366659421864813, 'median': 0.0, 'std': 0.3612047006517741,
                    'skewness': 12.587900367947181, 'kurtosis': 193.61929393146076
                }
            },
            'word_freq_meeting': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 14.28,
                    'mean': 0.13233862203868724, 'median': 0.0, 'std': 0.766819438588455,
                    'skewness': 9.455754913724958, 'kurtosis': 115.70597364310112
                }
            },
            'word_freq_original': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 3.57,
                    'mean': 0.0460986742012606, 'median': 0.0, 'std': 0.22381177612192932,
                    'skewness': 7.62922800107737, 'kurtosis': 78.57235677617535
                }
            },
            'word_freq_project': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 20.0,
                    'mean': 0.07919582699413176, 'median': 0.0, 'std': 0.6219755735032897,
                    'skewness': 18.7715154952594, 'kurtosis': 479.83090678442505
                }
            },
            'word_freq_re': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 21.42,
                    'mean': 0.3012236470332536, 'median': 0.0, 'std': 1.011687227279597,
                    'skewness': 9.146093367333956, 'kurtosis': 128.86467236557849
                }
            },
            'word_freq_edu': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 22.05,
                    'mean': 0.1798239513149317, 'median': 0.0, 'std': 0.9111190631640783,
                    'skewness': 10.1226627199091, 'kurtosis': 150.89981668745125
                }
            },
            'word_freq_table': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 2.17,
                    'mean': 0.005444468593783961, 'median': 0.0, 'std': 0.07627427063724908,
                    'skewness': 19.867691365298015, 'kurtosis': 459.4334630845171
                }
            },
            'word_freq_conference': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 10.0,
                    'mean': 0.03186915887850466, 'median': 0.0, 'std': 0.2857346462966567,
                    'skewness': 19.720445779318982, 'kurtosis': 537.4930073835098
                }
            },
            'char_freq_;': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.385,
                    'mean': 0.038574657683112336, 'median': 0.0, 'std': 0.2434713279276333,
                    'skewness': 13.708621273870152, 'kurtosis': 213.06819364467577
                }
            },
            'char_freq_(': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 9.752,
                    'mean': 0.13903042816778938, 'median': 0.065, 'std': 0.2703553739143305,
                    'skewness': 13.583754916091074, 'kurtosis': 393.41515820946023
                }
            },
            'char_freq_[': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 4.081,
                    'mean': 0.01697587480982397, 'median': 0.0, 'std': 0.10939416398282649,
                    'skewness': 21.083544814043776, 'kurtosis': 618.475609743282
                }
            },
            'char_freq_!': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 32.478,
                    'mean': 0.26907085416213833, 'median': 0.0, 'std': 0.8156716310876558,
                    'skewness': 18.658004365776716, 'kurtosis': 607.4556853851803
                }
            },
            'char_freq_$': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 6.002999999999999,
                    'mean': 0.07581069332753756, 'median': 0.0, 'std': 0.24588201134489382,
                    'skewness': 11.163141047276826, 'kurtosis': 199.95369159087346
                }
            },
            'char_freq_#': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 19.829,
                    'mean': 0.04423820908498153, 'median': 0.0, 'std': 0.42934208792226897,
                    'skewness': 31.062064279035454, 'kurtosis': 1218.4936470320708
                }
            },
            'capital_run_length_average': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.0, 'max': 1102.5,
                    'mean': 5.19151510541188, 'median': 2.276, 'std': 31.729448740210845,
                    'skewness': 23.761922995797903, 'kurtosis': 670.3687053958087
                }
            },
            'capital_run_length_longest': {  # the `data_type` could be `int`
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1, 'max': 9989,
                    'mean': 52.17278852423386, 'median': 15.0, 'std': 194.89130952646204,
                    'skewness': 30.764992575016173, 'kurtosis': 1480.6420502862773
                }
            },
            'capital_run_length_total': {  # the `data_type` could be `int`
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1, 'max': 15841,
                    'mean': 283.28928493805694, 'median': 95.0, 'std': 606.3478507248471,
                    'skewness': 8.709850408122785, 'kurtosis': 145.82981384531186
                }
            },
            'spam': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0: 2788, 1: 1813
                }
            }
        },        # ok
        'wine-red': {  # https://archive.ics.uci.edu/ml/datasets/Wine+Quality
            'fixed acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 4.6, 'max': 15.9,
                    'mean': 8.319637273295838, 'median': 7.9, 'std': 1.7410963181277006,
                    'skewness': 0.9827514413284587, 'kurtosis': 1.1321433977276252
                }
            },
            'volatile acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.12, 'max': 1.58,
                    'mean': 0.5278205128205131, 'median': 0.52, 'std': 0.17905970415353498,
                    'skewness': 0.6715925723840199, 'kurtosis': 1.2255422501791422
                }
            },
            'citric acid': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0,
                    'mean': 0.2709756097560964, 'median': 0.26, 'std': 0.19480113740531785,
                    'skewness': 0.3183372952546368, 'kurtosis': -0.7889975153633966
                }
            },
            'residual sugar': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.9, 'max': 15.5,
                    'mean': 2.5388055034396517, 'median': 2.2, 'std': 1.4099280595072805,
                    'skewness': 4.54065542590319, 'kurtosis': 28.617595424475443
                }
            },
            'chlorides': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.012, 'max': 0.611,
                    'mean': 0.08746654158849257, 'median': 0.079, 'std': 0.047065302010090154,
                    'skewness': 5.680346571971722, 'kurtosis': 41.71578724757661
                }
            },
            'free sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 1.0, 'max': 72.0,
                    'mean': 15.874921826141339, 'median': 14.0, 'std': 10.46015696980973,
                    'skewness': 1.250567293314441, 'kurtosis': 2.023562045840575
                }
            },
            'total sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 6.0, 'max': 289.0,
                    'mean': 46.46779237023139, 'median': 38.0, 'std': 32.89532447829901,
                    'skewness': 1.515531257594554, 'kurtosis': 3.8098244878645744
                }
            },
            'density': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.9900700000000001, 'max': 1.00369,
                    'mean': 0.9967466791744833, 'median': 0.99675, 'std': 0.0018873339538425563,
                    'skewness': 0.07128766294945525, 'kurtosis': 0.9340790654648083
                }
            },
            'pH': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.74, 'max': 4.01,
                    'mean': 3.311113195747343, 'median': 3.31, 'std': 0.15438646490354266,
                    'skewness': 0.19368349811284427, 'kurtosis': 0.806942508246574
                }
            },
            'sulphates': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.33, 'max': 2.0,
                    'mean': 0.6581488430268921, 'median': 0.62, 'std': 0.16950697959010977,
                    'skewness': 2.4286723536602945, 'kurtosis': 11.720250727147674
                }
            },
            'alcohol': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 8.4, 'max': 14.9,
                    'mean': 10.422983114446502, 'median': 10.2, 'std': 1.0656675818473926,
                    'skewness': 0.8608288068888538, 'kurtosis': 0.2000293113417695
                }
            },
            'quality': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    5: 681, 6: 638, 7: 199, 4: 53, 8: 18, 3: 10
                }
            }
        },    # ok
        'wine-white': {  # https://archive.ics.uci.edu/ml/datasets/Wine+Quality
            'fixed acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 3.8, 'max': 14.2,
                    'mean': 6.854787668436075, 'median': 6.8, 'std': 0.8438682276875188,
                    'skewness': 0.6477514746297539, 'kurtosis': 2.1721784645585807
                }
            },
            'volatile acidity': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.08, 'max': 1.1,
                    'mean': 0.27824111882401087, 'median': 0.26, 'std': 0.10079454842486428,
                    'skewness': 1.5769795029952025, 'kurtosis': 5.091625816866611
                }
            },
            'citric acid': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.66,
                    'mean': 0.33419150673743736, 'median': 0.32, 'std': 0.12101980420298301,
                    'skewness': 1.2819203981671066, 'kurtosis': 6.174900656983394
                }
            },
            'residual sugar': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.6, 'max': 65.8,
                    'mean': 6.391414863209486, 'median': 5.2, 'std': 5.072057784014864,
                    'skewness': 1.0770937564240868, 'kurtosis': 3.4698201025634265
                }
            },
            'chlorides': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.009000000000000001, 'max': 0.34600000000000003,
                    'mean': 0.0457723560636995, 'median': 0.043, 'std': 0.02184796809372882,
                    'skewness': 5.023330682759707, 'kurtosis': 37.564599706679516
                }
            },
            'free sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.0, 'max': 289.0,
                    'mean': 35.30808493262556, 'median': 34.0, 'std': 17.007137325232566,
                    'skewness': 1.4067449205303078, 'kurtosis': 11.466342426607905
                }
            },
            'total sulfur dioxide': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 9.0, 'max': 440.0,
                    'mean': 138.36065741118824, 'median': 134.0, 'std': 42.49806455414294,
                    'skewness': 0.3907098416536745, 'kurtosis': 0.5718532333534614
                }
            },
            'density': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.98711, 'max': 1.03898,
                    'mean': 0.9940273764801896, 'median': 0.99374, 'std': 0.0029909069169369393,
                    'skewness': 0.9777730048689881, 'kurtosis': 9.793806910765209
                }
            },
            'pH': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 2.72, 'max': 3.82,
                    'mean': 3.1882666394446693, 'median': 3.18, 'std': 0.1510005996150667,
                    'skewness': 0.4577825459180807, 'kurtosis': 0.5307749515326159
                }
            },
            'sulphates': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.22, 'max': 1.08,
                    'mean': 0.4898468762760325, 'median': 0.47, 'std': 0.11412583394883138,
                    'skewness': 0.9771936833065663, 'kurtosis': 1.5909296303516225
                }
            },
            'alcohol': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 8.0, 'max': 14.2,
                    'mean': 10.514267047774638, 'median': 10.4, 'std': 1.2306205677573183,
                    'skewness': 0.4873419932161276, 'kurtosis': -0.6984253277895518
                }
            },
            'quality': {  # target feature/variable
                'var_type': 'discrete', 'data_type': int, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    6: 2198, 5: 1457, 7: 880, 8: 175, 4: 163, 3: 20, 9: 5
                }
            }
        },  # ok
        'yeast': {  # https://archive.ics.uci.edu/ml/datasets/Yeast
            'Sequence_Name': {
                'var_type': 'discrete', 'data_type': str, 'target': False, 'drop': True, 'missing_values': [],
                'values_dist': {
                    'RL12_YEAST': 2, 'RS28_YEAST': 2, 'MTC_YEAST ': 2, 'RL2_YEAST ': 2, 'H3_YEAST  ': 2,
                    'RLUB_YEAST': 2, 'IF4A_YEAST': 2, 'RL44_YEAST': 2, 'RL41_YEAST': 2, 'MAT2_YEAST': 2,
                    'RL19_YEAST': 2, 'RS41_YEAST': 2, 'RS22_YEAST': 2, 'H4_YEAST  ': 2, 'RS24_YEAST': 2,
                    'RL15_YEAST': 2, 'RS8_YEAST ': 2, 'RL35_YEAST': 2, 'RL1A_YEAST': 2, 'EF1A_YEAST': 2,
                    'RS4E_YEAST': 2, 'RL18_YEAST': 2, 'DHSA_YEAST': 1, 'RL3A_YEAST': 1, 'RAS1_YEAST': 1,
                    'PES4_YEAST': 1, 'RS6_YEAST ': 1, 'R14A_YEAST': 1, 'BGL2_YEAST': 1, 'YCT8_YEAST': 1,
                    'SCO1_YEAST': 1, 'TAL1_YEAST': 1, 'MEIX_YEAST': 1, 'FU34_YEAST': 1, 'CC7_YEAST ': 1,
                    'MASZ_YEAST': 1, 'DPOX_YEAST': 1, 'R26B_YEAST': 1, 'EF1H_YEAST': 1, 'PUR4_YEAST': 1,
                    'YHK5_YEAST': 1, 'PROB_YEAST': 1, 'YB00_YEAST': 1, 'UBC2_YEAST': 1, 'E2BA_YEAST': 1,
                    'CYPR_YEAST': 1, 'NAB3_YEAST': 1, 'NOT1_YEAST': 1, 'SWI3_YEAST': 1, 'RA54_YEAST': 1,
                    'ADT2_YEAST': 1, 'SC11_YEAST': 1, 'PPAC_YEAST': 1, 'KR11_YEAST': 1, 'SC17_YEAST': 1,
                    'ADA2_YEAST': 1, 'HXKA_YEAST': 1, 'DPM1_YEAST': 1, 'YN94_YEAST': 1, 'PEM2_YEAST': 1,
                    'R13A_YEAST': 1, 'ACT2_YEAST': 1, 'YHU0_YEAST': 1, 'RTG1_YEAST': 1, 'PPX1_YEAST': 1,
                    'ACT3_YEAST': 1, 'R104_YEAST': 1, 'ERS1_YEAST': 1, 'CHL4_YEAST': 1, 'ARD1_YEAST': 1,
                    'SR54_YEAST': 1, 'RT05_YEAST': 1, 'THDH_YEAST': 1, 'M101_YEAST': 1, 'CCR4_YEAST': 1,
                    'TFC3_YEAST': 1, 'R161_YEAST': 1, 'RNC1_YEAST': 1, 'LAS1_YEAST': 1, 'SC23_YEAST': 1,
                    'MLH1_YEAST': 1, 'VATL_YEAST': 1, 'SNQ2_YEAST': 1, 'KEM1_YEAST': 1, 'YHQ1_YEAST': 1,
                    'N116_YEAST': 1, 'MRS3_YEAST': 1, 'SLA1_YEAST': 1, 'YGL1_YEAST': 1, 'PNPP_YEAST': 1,
                    'GR78_YEAST': 1, 'GALY_YEAST': 1, 'TCPD_YEAST': 1, 'DHSB_YEAST': 1, 'SSN6_YEAST': 1,
                    'SST2_YEAST': 1, 'AROG_YEAST': 1, '6PGD_YEAST': 1, 'R19A_YEAST': 1, 'PT94_YEAST': 1,
                    'VATD_YEAST': 1, 'PPB_YEAST ': 1, 'YKA5_YEAST': 1, 'UBC5_YEAST': 1, 'KHS1_YEAST': 1,
                    'PR39_YEAST': 1, 'YHA9_YEAST': 1, 'SAC7_YEAST': 1, 'ODPA_YEAST': 1, 'ACE2_YEAST': 1,
                    'RM37_YEAST': 1, 'ATR1_YEAST': 1, 'H2B1_YEAST': 1, 'CC27_YEAST': 1, 'TPS3_YEAST': 1,
                    'EXG1_YEAST': 1, 'RS11_YEAST': 1, 'CC25_YEAST': 1, 'CYBM_YEAST': 1, 'S61A_YEAST': 1,
                    'NI96_YEAST': 1, 'ERG6_YEAST': 1, 'YB72_YEAST': 1, 'UBP3_YEAST': 1, 'SIR3_YEAST': 1,
                    'SKO1_YEAST': 1, 'PTP2_YEAST': 1, 'FLO1_YEAST': 1, 'SC18_YEAST': 1, 'RL2A_YEAST': 1,
                    'YE14_YEAST': 1, 'LIP5_YEAST': 1, 'KC21_YEAST': 1, 'TBF1_YEAST': 1, 'RL3P_YEAST': 1,
                    'MCM3_YEAST': 1, 'RPBY_YEAST': 1, 'RL71_YEAST': 1, 'ATP8_YEAST': 1, 'HS76_YEAST': 1,
                    'CSE1_YEAST': 1, 'MDJ1_YEAST': 1, 'YCR8_YEAST': 1, 'CBF1_YEAST': 1, 'TNT_YEAST ': 1,
                    'DPOA_YEAST': 1, 'RT17_YEAST': 1, 'MS51_YEAST': 1, 'K6P2_YEAST': 1, 'CEM1_YEAST': 1,
                    'YHI7_YEAST': 1, 'CPT1_YEAST': 1, 'RED1_YEAST': 1, 'NOP2_YEAST': 1, 'SDH4_YEAST': 1,
                    'INO2_YEAST': 1, 'YCZ6_YEAST': 1, 'R102_YEAST': 1, 'PUT2_YEAST': 1, 'ATC4_YEAST': 1,
                    'SPT2_YEAST': 1, 'RCA1_YEAST': 1, 'APE2_YEAST': 1, 'RL3B_YEAST': 1, 'MAD1_YEAST': 1,
                    'TPS1_YEAST': 1, 'RU17_YEAST': 1, 'TPM1_YEAST': 1, 'YCG9_YEAST': 1, 'COXA_YEAST': 1,
                    'ADH1_YEAST': 1, 'HO_YEAST  ': 1, 'RLA1_YEAST': 1, 'YIN0_YEAST': 1, 'E2BD_YEAST': 1,
                    'EST1_YEAST': 1, 'CATT_YEAST': 1, 'TP20_YEAST': 1, 'UCR6_YEAST': 1, 'GDA1_YEAST': 1,
                    'YM38_YEAST': 1, 'SIN3_YEAST': 1, 'GBB_YEAST ': 1, 'GDI1_YEAST': 1, 'CYC2_YEAST': 1,
                    'RSU2_YEAST': 1, 'SRS2_YEAST': 1, 'YPT7_YEAST': 1, 'RT28_YEAST': 1, 'MGM1_YEAST': 1,
                    'NAM9_YEAST': 1, 'CNA1_YEAST': 1, 'TCPE_YEAST': 1, 'UBC1_YEAST': 1, 'RT08_YEAST': 1,
                    'PAN1_YEAST': 1, 'H2A2_YEAST': 1, 'ENO2_YEAST': 1, 'G3P3_YEAST': 1, 'VRP1_YEAST': 1,
                    'FUR1_YEAST': 1, 'MDL2_YEAST': 1, 'TPM2_YEAST': 1, 'SIR1_YEAST': 1, 'RPB1_YEAST': 1,
                    'RTPT_YEAST': 1, 'YAB8_YEAST': 1, 'CDC3_YEAST': 1, 'PORI_YEAST': 1, 'ATH1_YEAST': 1,
                    'ATPY_YEAST': 1, 'EGD1_YEAST': 1, 'CC42_YEAST': 1, 'HS83_YEAST': 1, 'ARF1_YEAST': 1,
                    'FKBP_YEAST': 1, 'MGMT_YEAST': 1, 'IXR1_YEAST': 1, 'OTC_YEAST ': 1, 'GRPE_YEAST': 1,
                    'PIR1_YEAST': 1, 'SC13_YEAST': 1, 'CB32_YEAST': 1, 'RM41_YEAST': 1, 'GSP1_YEAST': 1,
                    'SAG1_YEAST': 1, 'STL1_YEAST': 1, 'MCE1_YEAST': 1, 'UBIQ_YEAST': 1, 'RS33_YEAST': 1,
                    'HAL1_YEAST': 1, 'ATE1_YEAST': 1, 'KCC1_YEAST': 1, 'VATB_YEAST': 1, 'HXT5_YEAST': 1,
                    'ZNRP_YEAST': 1, 'RPC6_YEAST': 1, 'PUR3_YEAST': 1, 'POP2_YEAST': 1, 'MNN1_YEAST': 1,
                    'HAP2_YEAST': 1, 'HS72_YEAST': 1, 'SNF3_YEAST': 1, 'COX4_YEAST': 1, 'MLP1_YEAST': 1,
                    'SOF1_YEAST': 1, 'RRN6_YEAST': 1, 'ATU1_YEAST': 1, 'RM06_YEAST': 1, 'F26_YEAST ': 1,
                    'UCRQ_YEAST': 1, 'RPB3_YEAST': 1, 'SHM1_YEAST': 1, 'RER1_YEAST': 1, 'PYR1_YEAST': 1,
                    'MSH3_YEAST': 1, 'YD66_YEAST': 1, 'NAB2_YEAST': 1, 'NHPA_YEAST': 1, 'PAS3_YEAST': 1,
                    'CTK1_YEAST': 1, 'SEC9_YEAST': 1, 'RT13_YEAST': 1, 'PGM2_YEAST': 1, 'VATA_YEAST': 1,
                    'TRPF_YEAST': 1, 'SPB4_YEAST': 1, 'SED4_YEAST': 1, 'RSD1_YEAST': 1, 'YCQ7_YEAST': 1,
                    'RS31_YEAST': 1, 'PHR_YEAST ': 1, 'SYTC_YEAST': 1, 'PGK_YEAST ': 1, 'DLDH_YEAST': 1,
                    'ANC1_YEAST': 1, 'SPT3_YEAST': 1, 'SGE1_YEAST': 1, 'LYS2_YEAST': 1, 'YKH4_YEAST': 1,
                    'MNS1_YEAST': 1, 'KRE6_YEAST': 1, 'NAM7_YEAST': 1, 'R26A_YEAST': 1, 'RM25_YEAST': 1,
                    'HS30_YEAST': 1, 'IF41_YEAST': 1, 'K6P1_YEAST': 1, 'S61G_YEAST': 1, 'ALG1_YEAST': 1,
                    'CAPB_YEAST': 1, 'STF1_YEAST': 1, 'HIR2_YEAST': 1, 'AMPM_YEAST': 1, 'IMP2_YEAST': 1,
                    'KIN1_YEAST': 1, 'PEM1_YEAST': 1, 'YB8E_YEAST': 1, 'DHE4_YEAST': 1, 'YHB7_YEAST': 1,
                    'RM20_YEAST': 1, 'MK11_YEAST': 1, 'MA3T_YEAST': 1, 'YCV1_YEAST': 1, 'ERG2_YEAST': 1,
                    'DATI_YEAST': 1, 'UCR9_YEAST': 1, 'YJ51_YEAST': 1, 'MAC1_YEAST': 1, 'NOP3_YEAST': 1,
                    'YCK1_YEAST': 1, 'YHJ2_YEAST': 1, 'GLYM_YEAST': 1, 'THIK_YEAST': 1, 'COXX_YEAST': 1,
                    'NAB1_YEAST': 1, 'YN70_YEAST': 1, 'MPCP_YEAST': 1, 'STH1_YEAST': 1, 'RA18_YEAST': 1,
                    'YJ36_YEAST': 1, 'YCS7_YEAST': 1, 'MDM1_YEAST': 1, 'G6PI_YEAST': 1, 'SOK1_YEAST': 1,
                    'COQ1_YEAST': 1, 'ACT5_YEAST': 1, 'MFA2_YEAST': 1, 'MER1_YEAST': 1, 'ILVB_YEAST': 1,
                    'ATPD_YEAST': 1, 'SUI1_YEAST': 1, 'RPM2_YEAST': 1, 'SNF4_YEAST': 1, 'COX9_YEAST': 1,
                    'MRF1_YEAST': 1, 'CB33_YEAST': 1, 'PR38_YEAST': 1, 'RPA3_YEAST': 1, 'YK44_YEAST': 1,
                    'HXTY_YEAST': 1, 'ER24_YEAST': 1, 'UNG_YEAST ': 1, 'FRE1_YEAST': 1, 'FBRL_YEAST': 1,
                    'SMY1_YEAST': 1, 'YKN4_YEAST': 1, 'MAK3_YEAST': 1, 'SUV3_YEAST': 1, 'PLSC_YEAST': 1,
                    'INV1_YEAST': 1, 'MSS1_YEAST': 1, 'YHR0_YEAST': 1, 'NHPB_YEAST': 1, 'LYP1_YEAST': 1,
                    'PRCD_YEAST': 1, 'FET3_YEAST': 1, 'GLN3_YEAST': 1, 'PRI1_YEAST': 1, 'RS3A_YEAST': 1,
                    'MFA4_YEAST': 1, 'PYRD_YEAST': 1, 'RM02_YEAST': 1, 'CHI2_YEAST': 1, 'PROC_YEAST': 1,
                    'COQ3_YEAST': 1, 'R142_YEAST': 1, 'YEU2_YEAST': 1, 'MYS4_YEAST': 1, 'SCA1_YEAST': 1,
                    'ALG8_YEAST': 1, 'GCR2_YEAST': 1, 'PCNA_YEAST': 1, 'RPC3_YEAST': 1, 'MR11_YEAST': 1,
                    'CLH_YEAST ': 1, 'HR25_YEAST': 1, 'NHP2_YEAST': 1, 'SNM1_YEAST': 1, 'CAP_YEAST ': 1,
                    'GLRX_YEAST': 1, 'CC48_YEAST': 1, 'PDC2_YEAST': 1, 'PH84_YEAST': 1, 'RFC4_YEAST': 1,
                    'HCM1_YEAST': 1, 'TEC1_YEAST': 1, 'GAL4_YEAST': 1, 'CCPR_YEAST': 1, 'SYH_YEAST ': 1,
                    'R29B_YEAST': 1, 'KICH_YEAST': 1, 'VM11_YEAST': 1, 'YHP0_YEAST': 1, 'SPT7_YEAST': 1,
                    'ATH2_YEAST': 1, 'ATPG_YEAST': 1, 'BLH1_YEAST': 1, 'RFT1_YEAST': 1, 'SX19_YEAST': 1,
                    'DBR1_YEAST': 1, 'PRCH_YEAST': 1, 'YKE6_YEAST': 1, 'ADA3_YEAST': 1, 'YBE2_YEAST': 1,
                    'RFC1_YEAST': 1, 'ADH3_YEAST': 1, 'ACOX_YEAST': 1, 'SC59_YEAST': 1, 'ST14_YEAST': 1,
                    'MRS4_YEAST': 1, 'SEC1_YEAST': 1, 'IF51_YEAST': 1, 'SNC1_YEAST': 1, 'UGS2_YEAST': 1,
                    'SLA2_YEAST': 1, 'TRPG_YEAST': 1, 'PHO2_YEAST': 1, 'SRPR_YEAST': 1, 'TCPB_YEAST': 1,
                    'CHS1_YEAST': 1, 'NSP1_YEAST': 1, 'SIC1_YEAST': 1, 'SLY1_YEAST': 1, 'CTPT_YEAST': 1,
                    'PUB1_YEAST': 1, 'UCR2_YEAST': 1, 'PRC2_YEAST': 1, 'SC15_YEAST': 1, 'COAC_YEAST': 1,
                    'R167_YEAST': 1, 'PHO4_YEAST': 1, 'SEC6_YEAST': 1, 'GLYC_YEAST': 1, 'H2B2_YEAST': 1,
                    'MSH2_YEAST': 1, 'PRC3_YEAST': 1, 'CH10_YEAST': 1, 'RN15_YEAST': 1, 'STV1_YEAST': 1,
                    'FET4_YEAST': 1, 'RM31_YEAST': 1, 'GAP1_YEAST': 1, 'YJ43_YEAST': 1, 'YUR1_YEAST': 1,
                    'EPT1_YEAST': 1, 'TSL1_YEAST': 1, 'YHF0_YEAST': 1, 'GPDA_YEAST': 1, 'G6PD_YEAST': 1,
                    'SPT5_YEAST': 1, 'TOP1_YEAST': 1, 'RAD1_YEAST': 1, 'DA81_YEAST': 1, 'TCPZ_YEAST': 1,
                    'STE3_YEAST': 1, 'SNP2_YEAST': 1, 'SHR3_YEAST': 1, 'NUF1_YEAST': 1, 'CAN1_YEAST': 1,
                    'PMS1_YEAST': 1, 'E2BE_YEAST': 1, 'STE5_YEAST': 1, 'PYC2_YEAST': 1, 'SEC8_YEAST': 1,
                    'YBG6_YEAST': 1, 'VATC_YEAST': 1, 'PHD1_YEAST': 1, 'GCN5_YEAST': 1, 'PH80_YEAST': 1,
                    'RM38_YEAST': 1, 'KRE5_YEAST': 1, 'TSM1_YEAST': 1, 'IDHP_YEAST': 1, 'YHR8_YEAST': 1,
                    'PRC6_YEAST': 1, 'NUF2_YEAST': 1, 'SWI4_YEAST': 1, 'CYPD_YEAST': 1, 'CC31_YEAST': 1,
                    'MAT1_YEAST': 1, 'CIN1_YEAST': 1, 'PTR2_YEAST': 1, 'SRB4_YEAST': 1, 'CC68_YEAST': 1,
                    'ATPS_YEAST': 1, 'SYN_YEAST ': 1, 'AP19_YEAST': 1, 'ARGD_YEAST': 1, 'SRP2_YEAST': 1,
                    'UGA4_YEAST': 1, 'R19B_YEAST': 1, 'HEM2_YEAST': 1, 'S120_YEAST': 1, 'HPR1_YEAST': 1,
                    'ATPU_YEAST': 1, 'SLY4_YEAST': 1, 'EFG1_YEAST': 1, 'IF5_YEAST ': 1, 'TYR1_YEAST': 1,
                    'MANA_YEAST': 1, 'YOX1_YEAST': 1, 'VP17_YEAST': 1, 'ADP1_YEAST': 1, 'SSD1_YEAST': 1,
                    'DUR3_YEAST': 1, 'RA25_YEAST': 1, 'GPT_YEAST ': 1, 'RAM2_YEAST': 1, 'PDI_YEAST ': 1,
                    'AMYH_YEAST': 1, 'PRC8_YEAST': 1, 'EFG2_YEAST': 1, 'RS3B_YEAST': 1, 'ODO2_YEAST': 1,
                    'ATPT_YEAST': 1, 'GEF1_YEAST': 1, 'IRA2_YEAST': 1, 'MIF2_YEAST': 1, 'R14B_YEAST': 1,
                    'CYPH_YEAST': 1, 'UGA3_YEAST': 1, 'TFS2_YEAST': 1, 'GLO3_YEAST': 1, 'ATP7_YEAST': 1,
                    'PUT4_YEAST': 1, 'YKW2_YEAST': 1, 'CARB_YEAST': 1, 'RCS1_YEAST': 1, 'GAS1_YEAST': 1,
                    'SMC1_YEAST': 1, 'SC65_YEAST': 1, 'CYC1_YEAST': 1, 'OAT_YEAST ': 1, 'MCM1_YEAST': 1,
                    'ABF2_YEAST': 1, 'COX7_YEAST': 1, 'YHY1_YEAST': 1, 'PPA1_YEAST': 1, 'ESP1_YEAST': 1,
                    'MCM2_YEAST': 1, 'ODPX_YEAST': 1, 'TOP2_YEAST': 1, 'E2BG_YEAST': 1, 'HXT1_YEAST': 1,
                    'CCHL_YEAST': 1, 'BDF1_YEAST': 1, 'HS60_YEAST': 1, 'SMI1_YEAST': 1, 'TRPD_YEAST': 1,
                    'CISZ_YEAST': 1, 'MPI2_YEAST': 1, 'AAR2_YEAST': 1, 'UBA1_YEAST': 1, 'R17A_YEAST': 1,
                    'MASY_YEAST': 1, 'DNLI_YEAST': 1, 'ENO1_YEAST': 1, 'CC6_YEAST ': 1, 'GALX_YEAST': 1,
                    'LCB2_YEAST': 1, 'NMT_YEAST ': 1, 'SS10_YEAST': 1, 'SUP2_YEAST': 1, 'R114_YEAST': 1,
                    'UBCX_YEAST': 1, 'SP21_YEAST': 1, 'GCN4_YEAST': 1, 'PT27_YEAST': 1, 'NSR1_YEAST': 1,
                    'PEP8_YEAST': 1, 'MIG1_YEAST': 1, 'SYI_YEAST ': 1, 'CBP6_YEAST': 1, 'YKA8_YEAST': 1,
                    'FKB3_YEAST': 1, 'SYV_YEAST ': 1, 'YBC6_YEAST': 1, 'S160_YEAST': 1, 'MSH1_YEAST': 1,
                    'HS82_YEAST': 1, 'MTD1_YEAST': 1, 'RIF1_YEAST': 1, 'ADT3_YEAST': 1, 'YN46_YEAST': 1,
                    'UCR8_YEAST': 1, 'ARO1_YEAST': 1, 'ERG7_YEAST': 1, 'YHC6_YEAST': 1, 'KTR1_YEAST': 1,
                    'SNF1_YEAST': 1, 'R271_YEAST': 1, 'YK68_YEAST': 1, 'RN12_YEAST': 1, 'YMC2_YEAST': 1,
                    'CBP3_YEAST': 1, 'YKY8_YEAST': 1, 'AGA2_YEAST': 1, 'CATA_YEAST': 1, 'RPBX_YEAST': 1,
                    'RFC3_YEAST': 1, 'SCJ1_YEAST': 1, 'RPCX_YEAST': 1, 'CHS2_YEAST': 1, 'TF2D_YEAST': 1,
                    'R37B_YEAST': 1, 'YIB3_YEAST': 1, 'AST1_YEAST': 1, 'YB8I_YEAST': 1, 'MYS2_YEAST': 1,
                    'REB1_YEAST': 1, 'IRE1_YEAST': 1, 'YCR3_YEAST': 1, 'SED1_YEAST': 1, 'SP11_YEAST': 1,
                    'RIM1_YEAST': 1, 'SRPI_YEAST': 1, 'YDA2_YEAST': 1, 'TPS2_YEAST': 1, 'TBA3_YEAST': 1,
                    'PYC1_YEAST': 1, 'VAC1_YEAST': 1, 'RL34_YEAST': 1, 'UBC7_YEAST': 1, 'RAD4_YEAST': 1,
                    'CHI1_YEAST': 1, 'LEU1_YEAST': 1, 'CAO_YEAST ': 1, 'TF3A_YEAST': 1, 'TF2B_YEAST': 1,
                    'ERG1_YEAST': 1, 'TF3B_YEAST': 1, 'RPB5_YEAST': 1, 'MRS1_YEAST': 1, 'SPA2_YEAST': 1,
                    'COXZ_YEAST': 1, 'PDR1_YEAST': 1, 'SMD1_YEAST': 1, 'CAP2_YEAST': 1, 'LAG1_YEAST': 1,
                    'UME5_YEAST': 1, 'SC21_YEAST': 1, 'SON1_YEAST': 1, 'OPI1_YEAST': 1, 'ACO1_YEAST': 1,
                    'SWI5_YEAST': 1, 'PRTB_YEAST': 1, 'SP14_YEAST': 1, 'SPT6_YEAST': 1, 'YJ91_YEAST': 1,
                    'ACR1_YEAST': 1, 'BCK1_YEAST': 1, 'GAL8_YEAST': 1, 'YGP1_YEAST': 1, 'YP51_YEAST': 1,
                    'TBB_YEAST ': 1, 'ATPA_YEAST': 1, 'ACH1_YEAST': 1, 'CBP4_YEAST': 1, 'CBF5_YEAST': 1,
                    'INO4_YEAST': 1, 'STE2_YEAST': 1, 'BEM1_YEAST': 1, 'NC5R_YEAST': 1, 'RF1M_YEAST': 1,
                    'KI28_YEAST': 1, 'RU1A_YEAST': 1, 'PUT3_YEAST': 1, 'YKJ5_YEAST': 1, 'GCN2_YEAST': 1,
                    'YKE9_YEAST': 1, 'HSF_YEAST ': 1, 'GCR1_YEAST': 1, 'ATM1_YEAST': 1, 'YK62_YEAST': 1,
                    'YMC1_YEAST': 1, 'MSN1_YEAST': 1, 'STP1_YEAST': 1, 'SPT4_YEAST': 1, 'SRB5_YEAST': 1,
                    'TRP_YEAST ': 1, 'ERD2_YEAST': 1, 'RM27_YEAST': 1, 'HXT2_YEAST': 1, 'MDHP_YEAST': 1,
                    'RGM1_YEAST': 1, 'RAD9_YEAST': 1, 'PEP3_YEAST': 1, 'VP16_YEAST': 1, 'RFA3_YEAST': 1,
                    'COX1_YEAST': 1, 'MDHC_YEAST': 1, 'SWI6_YEAST': 1, 'KRE2_YEAST': 1, 'CIK1_YEAST': 1,
                    'EF3_YEAST ': 1, 'PRTD_YEAST': 1, 'RPC8_YEAST': 1, 'R16A_YEAST': 1, 'LCF2_YEAST': 1,
                    'NMD2_YEAST': 1, 'ITR2_YEAST': 1, 'HBS1_YEAST': 1, 'GLNA_YEAST': 1, 'TSA_YEAST ': 1,
                    'EUG1_YEAST': 1, 'MDL1_YEAST': 1, 'SYDC_YEAST': 1, 'TCPA_YEAST': 1, 'CBP2_YEAST': 1,
                    'LYS1_YEAST': 1, 'TRM1_YEAST': 1, 'RL3E_YEAST': 1, 'SYWM_YEAST': 1, 'RPC4_YEAST': 1,
                    'SC72_YEAST': 1, 'LONM_YEAST': 1, 'ACT_YEAST ': 1, 'GAL1_YEAST': 1, 'PRC9_YEAST': 1,
                    'RL1_YEAST ': 1, 'CYT2_YEAST': 1, 'DBP2_YEAST': 1, 'KRE1_YEAST': 1, 'END3_YEAST': 1,
                    'CHMU_YEAST': 1, 'HXKG_YEAST': 1, 'CISY_YEAST': 1, 'YKM9_YEAST': 1, 'MA3R_YEAST': 1,
                    'SC66_YEAST': 1, 'CBP1_YEAST': 1, 'RA50_YEAST': 1, 'APE3_YEAST': 1, 'VAL1_YEAST': 1,
                    'SMF2_YEAST': 1, 'PDR3_YEAST': 1, 'YHA8_YEAST': 1, 'UBP2_YEAST': 1, 'SC20_YEAST': 1,
                    'NIP1_YEAST': 1, 'PT12_YEAST': 1, 'EF2_YEAST ': 1, 'MT28_YEAST': 1, 'RA10_YEAST': 1,
                    'LEU3_YEAST': 1, 'TYSY_YEAST': 1, 'KEX2_YEAST': 1, 'BET1_YEAST': 1, 'YJ13_YEAST': 1,
                    'SYQ_YEAST ': 1, 'DCP1_YEAST': 1, 'ADT1_YEAST': 1, 'MS18_YEAST': 1, 'INV4_YEAST': 1,
                    'AFG3_YEAST': 1, 'PH85_YEAST': 1, 'RM08_YEAST': 1, 'YAE2_YEAST': 1, 'NAP1_YEAST': 1,
                    'RL27_YEAST': 1, 'CCL1_YEAST': 1, 'NOT4_YEAST': 1, 'CARP_YEAST': 1, 'IMP1_YEAST': 1,
                    'DBP1_YEAST': 1, 'DCP3_YEAST': 1, 'ST12_YEAST': 1, 'HDF1_YEAST': 1, 'ODPB_YEAST': 1,
                    'ADH4_YEAST': 1, 'RAD2_YEAST': 1, 'PAS1_YEAST': 1, 'SNF6_YEAST': 1, 'YCS4_YEAST': 1,
                    'DPB3_YEAST': 1, 'NOT3_YEAST': 1, 'KAPB_YEAST': 1, 'RLA2_YEAST': 1, 'SYMM_YEAST': 1,
                    'SMY2_YEAST': 1, 'RL9_YEAST ': 1, 'GLGB_YEAST': 1, 'RPB4_YEAST': 1, 'RM33_YEAST': 1,
                    'USO1_YEAST': 1, 'MPI1_YEAST': 1, 'ACEA_YEAST': 1, 'CG11_YEAST': 1, 'RA14_YEAST': 1,
                    'PMGY_YEAST': 1, 'TREA_YEAST': 1, 'STF2_YEAST': 1, 'UBC6_YEAST': 1, 'ITR1_YEAST': 1,
                    'KAD1_YEAST': 1, 'JNM1_YEAST': 1, 'RM09_YEAST': 1, 'SYG_YEAST ': 1, 'RAS2_YEAST': 1,
                    'SEN2_YEAST': 1, 'BIK1_YEAST': 1, 'SEC4_YEAST': 1, 'CALX_YEAST': 1, 'ALP1_YEAST': 1,
                    'PR04_YEAST': 1, 'DYR_YEAST ': 1, 'MNN9_YEAST': 1, 'AROF_YEAST': 1, 'GLS1_YEAST': 1,
                    'RM36_YEAST': 1, 'KIME_YEAST': 1, 'RM04_YEAST': 1, 'COXB_YEAST': 1, 'PRC5_YEAST': 1,
                    'PHSG_YEAST': 1, 'HEM6_YEAST': 1, 'PR22_YEAST': 1, 'SKI3_YEAST': 1, 'YCY8_YEAST': 1,
                    'BUD5_YEAST': 1, 'HMD2_YEAST': 1, 'EF1B_YEAST': 1, 'CYPC_YEAST': 1, 'KAPC_YEAST': 1,
                    'AP54_YEAST': 1, 'RL46_YEAST': 1, 'CLC1_YEAST': 1, 'SYFB_YEAST': 1, 'AMYG_YEAST': 1,
                    'LGT3_YEAST': 1, 'OCH1_YEAST': 1, 'ACP_YEAST ': 1, 'RPB8_YEAST': 1, 'SODC_YEAST': 1,
                    'TOA2_YEAST': 1, 'CYB_YEAST ': 1, 'SC22_YEAST': 1, 'G3P2_YEAST': 1, 'YHE0_YEAST': 1,
                    'CNA2_YEAST': 1, 'ADB1_YEAST': 1, 'RAT1_YEAST': 1, 'ATP6_YEAST': 1, 'NUC1_YEAST': 1,
                    'OSTD_YEAST': 1, 'RPC9_YEAST': 1, 'DPB2_YEAST': 1, 'SYKM_YEAST': 1, 'SX18_YEAST': 1,
                    'ATPB_YEAST': 1, 'SRB6_YEAST': 1, 'IF2B_YEAST': 1, 'R29A_YEAST': 1, 'SIP2_YEAST': 1,
                    'RL3_YEAST ': 1, 'SUG1_YEAST': 1, 'CYS3_YEAST': 1, 'GPDM_YEAST': 1, 'ZUO1_YEAST': 1,
                    'YEP0_YEAST': 1, 'FOX2_YEAST': 1, 'NIN1_YEAST': 1, 'RPB6_YEAST': 1, 'PMP1_YEAST': 1,
                    'PGM1_YEAST': 1, 'YIR3_YEAST': 1, 'G3P1_YEAST': 1, 'FUMH_YEAST': 1, 'SPK1_YEAST': 1,
                    'YB30_YEAST': 1, 'CACP_YEAST': 1, 'ARF2_YEAST': 1, 'RPC5_YEAST': 1, 'BOS1_YEAST': 1,
                    'PPR1_YEAST': 1, 'SNC2_YEAST': 1, 'RM32_YEAST': 1, 'HXT6_YEAST': 1, 'YAF3_YEAST': 1,
                    'ATC3_YEAST': 1, 'MTF1_YEAST': 1, 'GBA1_YEAST': 1, 'ABC1_YEAST': 1, 'DPOG_YEAST': 1,
                    'SC25_YEAST': 1, 'SYLC_YEAST': 1, 'ROX1_YEAST': 1, 'DHE2_YEAST': 1, 'YB32_YEAST': 1,
                    'LY14_YEAST': 1, 'OSTB_YEAST': 1, 'YJ49_YEAST': 1, 'SIS2_YEAST': 1, 'OM70_YEAST': 1,
                    'TRPE_YEAST': 1, 'PRCZ_YEAST': 1, 'NCPR_YEAST': 1, 'SIN4_YEAST': 1, 'CYC7_YEAST': 1,
                    'PAS7_YEAST': 1, 'SSO1_YEAST': 1, 'PIK1_YEAST': 1, 'YMX1_YEAST': 1, 'GCR3_YEAST': 1,
                    'ARG2_YEAST': 1, 'ODO1_YEAST': 1, 'KRE9_YEAST': 1, 'RHO1_YEAST': 1, 'UME6_YEAST': 1,
                    'RT02_YEAST': 1, 'RMAR_YEAST': 1, 'RNA1_YEAST': 1, 'COX3_YEAST': 1, 'MYS1_YEAST': 1,
                    'ATC1_YEAST': 1, 'DCP2_YEAST': 1, 'IF4E_YEAST': 1, 'RS37_YEAST': 1, 'RAD5_YEAST': 1,
                    'ILV5_YEAST': 1, 'SYG1_YEAST': 1, 'YHY0_YEAST': 1, 'VP34_YEAST': 1, 'CYPB_YEAST': 1,
                    'UBC3_YEAST': 1, 'HOP1_YEAST': 1, 'C1TM_YEAST': 1, 'YEA6_YEAST': 1, 'GAR1_YEAST': 1,
                    'AGA1_YEAST': 1, 'COXE_YEAST': 1, 'DA80_YEAST': 1, 'MET2_YEAST': 1, 'COQ2_YEAST': 1,
                    'RPB2_YEAST': 1, 'YHX8_YEAST': 1, 'MTA1_YEAST': 1, 'TFC1_YEAST': 1, 'ADR1_YEAST': 1,
                    'ST20_YEAST': 1, 'CALM_YEAST': 1, 'YAC2_YEAST': 1, 'CC23_YEAST': 1, 'PEP1_YEAST': 1,
                    'GBP2_YEAST': 1, 'NAM1_YEAST': 1, 'ZIP1_YEAST': 1, 'LEUR_YEAST': 1, 'YP53_YEAST': 1,
                    'COX6_YEAST': 1, 'RAD7_YEAST': 1, 'SIR4_YEAST': 1, 'SP10_YEAST': 1, 'SNF2_YEAST': 1,
                    'COFI_YEAST': 1, 'NPL1_YEAST': 1, 'KTR4_YEAST': 1, 'ATPO_YEAST': 1, 'LEU2_YEAST': 1,
                    'HXT3_YEAST': 1, 'MS16_YEAST': 1, 'END2_YEAST': 1, 'THRC_YEAST': 1, 'AROC_YEAST': 1,
                    'DPOD_YEAST': 1, 'CBS1_YEAST': 1, 'N100_YEAST': 1, 'YKS8_YEAST': 1, 'CBS2_YEAST': 1,
                    'RM16_YEAST': 1, 'YIG3_YEAST': 1, 'UBC4_YEAST': 1, 'AATC_YEAST': 1, 'YHB9_YEAST': 1,
                    'HS26_YEAST': 1, 'KCC2_YEAST': 1, 'YAB1_YEAST': 1, 'RPB9_YEAST': 1, 'PMT_YEAST ': 1,
                    'CYB2_YEAST': 1, 'SUL1_YEAST': 1, 'SYFM_YEAST': 1, 'SLN1_YEAST': 1, 'RFC2_YEAST': 1,
                    'SRB2_YEAST': 1, 'SNF5_YEAST': 1, 'RHO2_YEAST': 1, 'FIMB_YEAST': 1, 'DAP1_YEAST': 1,
                    'YHT3_YEAST': 1, 'RA51_YEAST': 1, 'RCC_YEAST ': 1, 'MOT1_YEAST': 1, 'FUS3_YEAST': 1,
                    'KAD2_YEAST': 1, 'ERG8_YEAST': 1, 'YN19_YEAST': 1, 'AATM_YEAST': 1, 'RLA3_YEAST': 1,
                    'RL16_YEAST': 1, 'PIF1_YEAST': 1, 'PT17_YEAST': 1, 'KHR1_YEAST': 1, 'PT09_YEAST': 1,
                    'C1TC_YEAST': 1, 'ADH2_YEAST': 1, 'VP15_YEAST': 1, 'MK16_YEAST': 1, 'DHA1_YEAST': 1,
                    'HIS7_YEAST': 1, 'IF2M_YEAST': 1, 'ATC2_YEAST': 1, 'YHG2_YEAST': 1, 'VATE_YEAST': 1,
                    'PR19_YEAST': 1, 'RIM2_YEAST': 1, 'YJ16_YEAST': 1, 'RA55_YEAST': 1, 'RME1_YEAST': 1,
                    'VATX_YEAST': 1, 'YCA9_YEAST': 1, 'YIN4_YEAST': 1, 'RT01_YEAST': 1, 'OM45_YEAST': 1,
                    'KAR3_YEAST': 1, 'YHN8_YEAST': 1, 'CTR1_YEAST': 1, 'YCD8_YEAST': 1, 'PT11_YEAST': 1,
                    'END1_YEAST': 1, 'IPPI_YEAST': 1, 'RL13_YEAST': 1, 'CC12_YEAST': 1, 'RA52_YEAST': 1,
                    'RPA9_YEAST': 1, 'ATN1_YEAST': 1, 'RPD3_YEAST': 1, 'R141_YEAST': 1, 'INV2_YEAST': 1,
                    'THIL_YEAST': 1, 'KIP1_YEAST': 1, 'MA6T_YEAST': 1, 'VATF_YEAST': 1, 'FUR4_YEAST': 1,
                    'TRK1_YEAST': 1, 'KC22_YEAST': 1, 'RS3_YEAST ': 1, 'PTSR_YEAST': 1, 'CAT8_YEAST': 1,
                    'TKT2_YEAST': 1, 'ERG4_YEAST': 1, 'CALB_YEAST': 1, 'ACON_YEAST': 1, 'RN14_YEAST': 1,
                    'IDH2_YEAST': 1, 'UBP1_YEAST': 1, 'MPP2_YEAST': 1, 'IATP_YEAST': 1, 'IF42_YEAST': 1,
                    'HXKB_YEAST': 1, 'SIR2_YEAST': 1, 'PRI2_YEAST': 1, 'AP17_YEAST': 1, 'MDHM_YEAST': 1,
                    'CBPS_YEAST': 1, 'PUT1_YEAST': 1, 'SODM_YEAST': 1, 'CSG2_YEAST': 1, 'IF1A_YEAST': 1,
                    'IRA1_YEAST': 1, 'NDC1_YEAST': 1, 'NUP2_YEAST': 1, 'NUP1_YEAST': 1, 'PPA5_YEAST': 1,
                    'IF2A_YEAST': 1, 'CC24_YEAST': 1, 'KAPR_YEAST': 1, 'PPA3_YEAST': 1, 'YCFI_YEAST': 1,
                    'EFTU_YEAST': 1, 'OM20_YEAST': 1, 'PABP_YEAST': 1, 'SYKC_YEAST': 1, 'MSN4_YEAST': 1,
                    'TRAM_YEAST': 1, 'PIS_YEAST ': 1, 'MAS6_YEAST': 1, 'KC2C_YEAST': 1, 'PTM1_YEAST': 1,
                    'IPYR_YEAST': 1, 'SSO2_YEAST': 1, 'PLB1_YEAST': 1, 'KTHY_YEAST': 1, 'RL6_YEAST ': 1,
                    'ALF_YEAST ': 1, 'HXT4_YEAST': 1, 'CHS3_YEAST': 1, 'DPOE_YEAST': 1, 'PRCF_YEAST': 1,
                    'ERG3_YEAST': 1, 'SFL1_YEAST': 1, 'MPP1_YEAST': 1, 'RAM1_YEAST': 1, 'RFA1_YEAST': 1,
                    'YBI8_YEAST': 1, 'P2B1_YEAST': 1, 'ASG2_YEAST': 1, 'FPS1_YEAST': 1, 'RM49_YEAST': 1,
                    'TIF3_YEAST': 1, 'SDH3_YEAST': 1, 'COX2_YEAST': 1, 'H150_YEAST': 1, 'GOG5_YEAST': 1,
                    'YPT1_YEAST': 1, 'MEP1_YEAST': 1, 'SRP1_YEAST': 1, 'MD10_YEAST': 1, 'DLD1_YEAST': 1,
                    'HXT7_YEAST': 1, 'RRN7_YEAST': 1, 'SLP1_YEAST': 1, 'RL4A_YEAST': 1, 'PDR4_YEAST': 1,
                    'RA26_YEAST': 1, 'FPPS_YEAST': 1, 'KSS1_YEAST': 1, 'RAD3_YEAST': 1, 'AGAL_YEAST': 1,
                    'MER2_YEAST': 1, 'YHD5_YEAST': 1, 'RPA1_YEAST': 1, 'PPAB_YEAST': 1, 'TOA1_YEAST': 1,
                    'KIP2_YEAST': 1, 'SC62_YEAST': 1, 'TAT2_YEAST': 1, 'SMF1_YEAST': 1, 'UCRI_YEAST': 1,
                    'FHL1_YEAST': 1, 'ATPE_YEAST': 1, 'POB1_YEAST': 1, 'MSF1_YEAST': 1, 'MSB2_YEAST': 1,
                    'UCRX_YEAST': 1, 'YCQ0_YEAST': 1, 'RS15_YEAST': 1, 'YB8G_YEAST': 1, 'AFR1_YEAST': 1,
                    'VP45_YEAST': 1, 'PR28_YEAST': 1, 'KAR1_YEAST': 1, 'CY1_YEAST ': 1, 'YB91_YEAST': 1,
                    'PR16_YEAST': 1, 'NU49_YEAST': 1, '6P2K_YEAST': 1, 'BAR1_YEAST': 1, 'PR02_YEAST': 1,
                    'PMT1_YEAST': 1, 'DCUP_YEAST': 1, 'ASSY_YEAST': 1, 'RL4B_YEAST': 1, 'IF52_YEAST': 1,
                    'INO1_YEAST': 1, 'SEC2_YEAST': 1, 'ADB2_YEAST': 1, 'YHK8_YEAST': 1, 'HAP3_YEAST': 1,
                    'HMD1_YEAST': 1, 'RFA2_YEAST': 1, 'PMP2_YEAST': 1, 'KTR2_YEAST': 1, 'ROX3_YEAST': 1,
                    'COX8_YEAST': 1, 'RA23_YEAST': 1, 'SYTM_YEAST': 1, 'SYDM_YEAST': 1, 'HS73_YEAST': 1,
                    'LOS1_YEAST': 1, 'UCR1_YEAST': 1, 'SDHL_YEAST': 1, 'YHU2_YEAST': 1, 'YCK2_YEAST': 1,
                    'FAS1_YEAST': 1, 'MBP1_YEAST': 1, 'SYR_YEAST ': 1, 'FAD1_YEAST': 1, 'PR06_YEAST': 1,
                    'FKB2_YEAST': 1, 'PR05_YEAST': 1, 'CARA_YEAST': 1, 'VPH1_YEAST': 1, 'DRS1_YEAST': 1,
                    'CG13_YEAST': 1, 'DAL4_YEAST': 1, 'PDR5_YEAST': 1, 'ARG1_YEAST': 1, 'MRS2_YEAST': 1,
                    'YCX9_YEAST': 1, 'CACM_YEAST': 1, 'GTS1_YEAST': 1, 'H104_YEAST': 1, 'CC10_YEAST': 1,
                    'KEX1_YEAST': 1, 'PMM_YEAST ': 1, 'YKD8_YEAST': 1, 'PR21_YEAST': 1, 'RPB7_YEAST': 1,
                    'ARG3_YEAST': 1, 'ARLY_YEAST': 1, 'YKW0_YEAST': 1, 'TKT1_YEAST': 1, 'YAG7_YEAST': 1,
                    'GFA1_YEAST': 1, 'NDI1_YEAST': 1, 'SYLM_YEAST': 1, 'SYMC_YEAST': 1, 'T2EA_YEAST': 1,
                    'DPSD_YEAST': 1, 'COXG_YEAST': 1, 'YJ12_YEAST': 1, 'DAL5_YEAST': 1, 'YBR8_YEAST': 1,
                    'HEM1_YEAST': 1, 'ACE1_YEAST': 1, 'AR56_YEAST': 1, 'ALG5_YEAST': 1, 'TRK2_YEAST': 1,
                    'FZF1_YEAST': 1, 'RM13_YEAST': 1, 'SPR1_YEAST': 1, 'SP23_YEAST': 1, 'SC14_YEAST': 1,
                    'ABP1_YEAST': 1, 'HS71_YEAST': 1, 'PRCI_YEAST': 1, 'DMC1_YEAST': 1, 'GAL7_YEAST': 1,
                    'CHL1_YEAST': 1, 'MA6R_YEAST': 1, 'RPC1_YEAST': 1, 'SR72_YEAST': 1, 'TCTP_YEAST': 1,
                    'ATP9_YEAST': 1, 'RS21_YEAST': 1, 'HIR1_YEAST': 1, 'TYE7_YEAST': 1, 'YMC4_YEAST': 1,
                    'SEN1_YEAST': 1, 'DPO2_YEAST': 1, 'RIMI_YEAST': 1, 'SC12_YEAST': 1, 'MT17_YEAST': 1,
                    'CP51_YEAST': 1, 'SYFA_YEAST': 1, 'KAPA_YEAST': 1, 'IDH1_YEAST': 1, 'YAP3_YEAST': 1,
                    'CC16_YEAST': 1, 'CRM1_YEAST': 1, 'BCS1_YEAST': 1, 'UGS1_YEAST': 1, 'RL21_YEAST': 1,
                    'CIN4_YEAST': 1, 'NFS1_YEAST': 1, 'RPOM_YEAST': 1, 'PT91_YEAST': 1, 'GCS1_YEAST': 1,
                    'YAG3_YEAST': 1, 'PRT1_YEAST': 1, 'HS74_YEAST': 1, 'R61A_YEAST': 1, 'ISP6_YEAST': 1,
                    'RLA4_YEAST': 1, 'UMPK_YEAST': 1, 'NOT2_YEAST': 1, 'PTP1_YEAST': 1, 'CAPA_YEAST': 1,
                    'PIR3_YEAST': 1, 'DKA1_YEAST': 1, 'MAG_YEAST ': 1, 'YAB9_YEAST': 1, 'PR09_YEAST': 1,
                    'CCE1_YEAST': 1, 'YEX1_YEAST': 1, 'XRS2_YEAST': 1, 'DBF4_YEAST': 1, 'CC46_YEAST': 1,
                    'CYP1_YEAST': 1, 'GCN1_YEAST': 1, 'KIN2_YEAST': 1, 'MSP1_YEAST': 1, 'FUS1_YEAST': 1,
                    'VM12_YEAST': 1, 'R272_YEAST': 1, 'ATN2_YEAST': 1, 'ORC2_YEAST': 1, 'YJ90_YEAST': 1,
                    'SFP1_YEAST': 1, 'YBC4_YEAST': 1, 'CC4_YEAST ': 1, 'SRD1_YEAST': 1, 'YCH0_YEAST': 1,
                    'ANP1_YEAST': 1, 'SED5_YEAST': 1, 'YBY2_YEAST': 1, 'HIP1_YEAST': 1, 'YME1_YEAST': 1,
                    'IPY2_YEAST': 1, 'UBR1_YEAST': 1, 'BAS1_YEAST': 1, 'MAOX_YEAST': 1, 'P152_YEAST': 1,
                    'KPYK_YEAST': 1, 'YJ64_YEAST': 1, 'PSS_YEAST ': 1, 'SR68_YEAST': 1, 'HS77_YEAST': 1,
                    'TOR2_YEAST': 1, 'IPB2_YEAST': 1, 'PRCE_YEAST': 1, 'STE6_YEAST': 1, 'RA57_YEAST': 1,
                    'CC11_YEAST': 1, 'DHH1_YEAST': 1, 'VPS1_YEAST': 1, 'FCY2_YEAST': 1, 'APN1_YEAST': 1,
                    'PR08_YEAST': 1, 'FMT_YEAST ': 1, 'RLA0_YEAST': 1, 'TFB1_YEAST': 1, 'RPA2_YEAST': 1,
                    'DA82_YEAST': 1, 'TBA1_YEAST': 1, 'MET4_YEAST': 1, 'HSP7_YEAST': 1, 'SR14_YEAST': 1,
                    'IME1_YEAST': 1, 'TUP1_YEAST': 1, 'TOP3_YEAST': 1, 'TTP1_YEAST': 1, 'GAL2_YEAST': 1,
                    'RM44_YEAST': 1, 'DED1_YEAST': 1, 'H2A1_YEAST': 1, 'SLU7_YEAST': 1, 'YCW2_YEAST': 1,
                    'SUP1_YEAST': 1, 'TCPG_YEAST': 1, 'YNE2_YEAST': 1, 'CB31_YEAST': 1, 'PRCG_YEAST': 1,
                    'YB52_YEAST': 1, 'CSE2_YEAST': 1, 'CC40_YEAST': 1, 'LCB1_YEAST': 1, 'CAD1_YEAST': 1,
                    'DEP1_YEAST': 1, 'PT22_YEAST': 1, 'DSS4_YEAST': 1, 'R37A_YEAST': 1, 'STU1_YEAST': 1,
                    'YB54_YEAST': 1, 'PE12_YEAST': 1, 'HAP4_YEAST': 1, 'DAP2_YEAST': 1, 'MDS1_YEAST': 1,
                    'YP52_YEAST': 1, 'CBPY_YEAST': 1, 'TPIS_YEAST': 1, 'MSH4_YEAST': 1, 'BSD2_YEAST': 1,
                    'COT1_YEAST': 1, 'HS75_YEAST': 1, 'NOP4_YEAST': 1, 'DUN1_YEAST': 1, 'YIA6_YEAST': 1,
                    'YINO_YEAST': 1, 'MOD5_YEAST': 1, 'KTR3_YEAST': 1, 'YB48_YEAST': 1, 'CYAA_YEAST': 1,
                    'SPT8_YEAST': 1, 'FDFT_YEAST': 1, 'IF2G_YEAST': 1, 'HEX2_YEAST': 1, 'P2B2_YEAST': 1,
                    'F16P_YEAST': 1, 'SKN7_YEAST': 1, 'FAS2_YEAST': 1, 'RMR3_YEAST': 1, 'PROF_YEAST': 1,
                    'PDS1_YEAST': 1, 'PGD1_YEAST': 1, 'YK42_YEAST': 1, 'GAC1_YEAST': 1, 'BAF1_YEAST': 1,
                    'RL25_YEAST': 1, 'YIF2_YEAST': 1, 'RPC2_YEAST': 1, 'TIP1_YEAST': 1, 'PAP_YEAST ': 1,
                    'HEMZ_YEAST': 1, 'YAA7_YEAST': 1, 'ARGI_YEAST': 1, 'SSB1_YEAST': 1, 'ADR6_YEAST': 1,
                    'PT54_YEAST': 1, 'SAR1_YEAST': 1, 'DYHC_YEAST': 1, 'RT04_YEAST': 1, 'SMC2_YEAST': 1,
                    'AMPL_YEAST': 1, 'R17B_YEAST': 1, 'MA3S_YEAST': 1, 'SYAC_YEAST': 1, 'YB33_YEAST': 1,
                    'ODP2_YEAST': 1, 'EF1G_YEAST': 1, 'RA16_YEAST': 1, 'NAT1_YEAST': 1, 'PR18_YEAST': 1,
                    'CG12_YEAST': 1, 'IS42_YEAST': 1, 'HEM3_YEAST': 1, 'CAL1_YEAST': 1, 'SEC7_YEAST': 1,
                    'E2BB_YEAST': 1, 'GSP2_YEAST': 1, 'RT09_YEAST': 1, 'NAT2_YEAST': 1, 'YAF1_YEAST': 1,
                    'HXT8_YEAST': 1, 'SKN1_YEAST': 1, 'ERD1_YEAST': 1, 'RL17_YEAST': 1, 'YB37_YEAST': 1,
                    'DHA2_YEAST': 1, 'SYSC_YEAST': 1, 'PPCK_YEAST': 1, 'SIS1_YEAST': 1, 'MAN1_YEAST': 1,
                    'DPO4_YEAST': 1, 'MFA3_YEAST': 1, 'YBT6_YEAST': 1, 'GBG_YEAST ': 1, 'R16B_YEAST': 1,
                    'BCK2_YEAST': 1, 'METE_YEAST': 1, 'RS4_YEAST ': 1, 'MAS5_YEAST': 1, 'TRNL_YEAST': 1,
                    'PMT2_YEAST': 1, 'GCY_YEAST ': 1, 'CIN8_YEAST': 1, 'PP12_YEAST': 1, 'TFC4_YEAST': 1,
                    'MSN2_YEAST': 1, 'RAP1_YEAST': 1
                }
            },
            'mcg': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.11, 'max': 1.0, 'mean': 0.500121293800539, 'median': 0.49,
                    'std': 0.1372993003895818, 'skewness': 0.604291161361435, 'kurtosis': 0.4590599900673058
                }
            },
            'gvh': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.13, 'max': 1.0, 'mean': 0.4999326145552553, 'median': 0.49,
                    'std': 0.12392434900413846, 'skewness': 0.4166394134928298, 'kurtosis': 0.5561109616052562
                }
            },
            'alm': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.21, 'max': 1.0, 'mean': 0.500033692722372, 'median': 0.51,
                    'std': 0.08667024770783191, 'skewness': -0.22099540133881013, 'kurtosis': 1.60931957594235
                }
            },
            'mit': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0, 'mean': 0.26118598382749364, 'median': 0.22,
                    'std': 0.13709763089421498, 'skewness': 1.444776165204374, 'kurtosis': 2.2899705478348666
                }
            },
            'erl': {
                'var_type': 'discrete', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0.5: 1470, 1.0: 14
                    # 'min': 0.5, 'max': 1.0, 'mean': 0.5047169811320755, 'median': 0.5,
                    # 'std': 0.04835096692671292, 'skewness': 10.159632813962212, 'kurtosis': 101.35473389753923
                }
            },
            'pox': {
                'var_type': 'discrete', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    0.00: 1469, 0.83: 11, 0.50: 4
                    # 'min': 0.0, 'max': 0.83, 'mean': 0.007500000000000001, 'median': 0.0,
                    # 'std': 0.0756826652050668, 'skewness': 10.276883707764107, 'kurtosis': 105.738712367983
                }
            },
            'vac': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 0.73, 'mean': 0.4998854447439337, 'median': 0.51,
                    'std': 0.05779658638925979, 'skewness': -1.7916406496353354, 'kurtosis': 9.501358534499472
                }
            },
            'nuc': {
                'var_type': 'continuous', 'data_type': float, 'target': False, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'min': 0.0, 'max': 1.0, 'mean': 0.27619946091644665, 'median': 0.22,
                    'std': 0.10649052826089482, 'skewness': 2.41303121845126, 'kurtosis': 7.777761531779612
                }
            },
            'Localization_Site': {  # target feature/variable
                'var_type': 'discrete', 'data_type': str, 'target': True, 'drop': False, 'missing_values': [],
                'values_dist': {
                    'CYT': 463, 'NUC': 429, 'MIT': 244, 'ME3': 163, 'ME2': 51,
                    'ME1': 44, 'EXC': 35, 'VAC': 30, 'POX': 20, 'ERL': 5
                }
            }
        }        # ok
    }
    """The metadata of the supported datasets.
    A dataset is identified by its (short) name (e.g., 'adult') which maps to the respective metadata of each column. 
    For the moment, the collected metadata of each column is relatively shallow and is composed by:
        - 'var_type': Either 'discrete' (aka categorical) or 'continuous' (aka numerical).
        - 'data_type': Such as int, float, str, bool, and so forth.
        - 'target': If True the column (i.e., the feature/variable) is the target variable. 
        - 'drop': If True the column (i.e., the feature/variable) is marked to be dropped.
        - 'missing_values': A list of symbols (e.g., [' ', 0, '?']) to be handled as missing values.
        - 'values_dist': If the column (i.e., the feature/variable) is 'discrete' then 
          the collected metadata is a dict of the values' occurrences. If the column is 'continuous' then 
          the collected metadata is a dict that stores the 'min', the 'max', the 'mean', the 'median', and 
          the 'std' (i.e., the standard deviation), the 'skewness', the 'kurtosis' values. 
          For other data types (e.g., str) it is still NOT defined the metadata that this dict (i.e., the 'values_dist') 
          should store.
    """

    @classmethod
    def continuous_vars(cls,
                        dataset: str = 'adult',
                        df: pd.DataFrame = None,
                        verbose: bool = False) -> Union[List[str], List[int]]:
        all_cont_vars: Union[List[int], List[str]] = [
            var for var in Metadata.DATASETS[dataset] if Metadata.DATASETS[dataset][var]['var_type'] == 'continuous']
        cont_vars: Union[List[str], List[int]] = all_cont_vars

        if df is not None:
            # TODO: encoder_type = 'label' --> plots.py --> boxes_plots() --> df_melt = ... --> ERROR HERE!
            #       PROBABLY THE SAME HAPPENS FOR encoder_type = 'pne-hot'
            #       FIX THIS TO TURN POSSIBLE TO UNCOMMENT THE if AND THE raise STATEMENTS!
            # if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
            #     raise ValueError("purify.dataset.metadata.Metadata :: continuous_vars()\n"
            #                      "The set of variables (i.e., features/columns) "
            #                      "of the given `df` (i.e., pandas DataFrame) "
            #                      "is NOT a subset of the one of the given `dataset`.")
            cont_vars = [cont_var for cont_var in cont_vars if cont_var in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: continuous_vars()")
            print(f"Continuous variables of the `{dataset}` dataset:\n{all_cont_vars}")
            print(f"Continuous variables to be returned:\n{cont_vars}")
        return cont_vars

    @classmethod
    def discrete_vars(cls,
                      dataset: str = 'adult',
                      df: pd.DataFrame = None,
                      verbose: bool = False) -> Union[List[str], List[int]]:
        """Return a list of the discrete (aka categorical) variables (i.e., features/columns) of
        the given `dataset`. If the given pandas DataFrame (i.e., `df`) is NOT None then the list of its columns
        has to be a subset of the columns of the given `dataset` and
        the list to be returned has only the discrete columns that are present in the given `df`.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported by this class.
        df : DataFrame
            A pandas DataFrame (only) with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output,
            which is useful, for instance, to debug and to trace the execution.

        Returns
        -------
        disc_vars : Union[List[str], List[int]]
            A list with the discrete (aka categorical) variables (i.e., features/columns) of the given `dataset` or
            a subset of those that are present in the given `df` (i.e., in the given pandas DataFrame).

        Raises
        ------
        ValueError
            If the given `df` (i.e., the given pandas DataFrame) is NOT None and
            the columns of the given `df` is NOT a subset of the ones of the given `dataset`.
        """
        all_disc_vars: Union[List[str], List[int]] = [variable for variable, meta in Metadata.DATASETS[dataset].items()
                                                      if meta['var_type'] == 'discrete']
        disc_vars: Union[List[str], List[int]] = all_disc_vars

        if df is not None:
            if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
                raise ValueError("purify.dataset.metadata.Metadata :: discrete_vars()\n"
                                 "The set of variables (i.e., features/columns) "
                                 "of the given `df` (i.e., pandas DataFrame) "
                                 "is NOT a subset of the one of the given `dataset`.")
            disc_vars = [variable for variable in all_disc_vars if variable in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: discrete_vars()")
            print(f"Discrete variables of the `{dataset}` dataset:\n{all_disc_vars}")
            print(f"Discrete variables to be returned:\n{disc_vars}")
        return disc_vars

    @classmethod
    def discrete_vars_and_dtypes(cls,
                                 dataset: str = 'adult',
                                 df: pd.DataFrame = None,
                                 verbose: bool = False) -> Union[List[Tuple[str, Any]], List[Tuple[int, Any]]]:
        """Return a list of the discrete (aka categorical) variables (i.e., features/columns) and
        their data types of the given `dataset`. If the given pandas DataFrame (i.e., `df`) is NOT None then
        the list of its columns has to be a subset of the columns of the given `dataset` and
        the list to be returned has only the discrete columns that are present in the given `df`.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported by this class.
        df : DataFrame
            A pandas DataFrame (only) with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output,
            which is useful, for instance, to debug and to trace the execution.

        Returns
        -------
        disc_vars_and_dtypes : Union[List[Tuple[str, Any]], List[Tuple[int, Any]]]
            A list with the discrete (aka categorical) variables (i.e., features/columns) and
            their data types of the given `dataset` or
            a subset of those that are present in the given `df` (i.e., in the given pandas DataFrame).

        Raises
        ------
        ValueError
            If the given `df` (i.e., the given pandas DataFrame) is NOT None and
            the columns of the given `df` is NOT a subset of the ones of the given `dataset`.
        """
        all_disc_vars_and_dtypes: Union[List[Tuple[str, Any]], List[Tuple[int, Any]]] = [
            (variable, meta['data_type']) for variable, meta in Metadata.DATASETS[dataset].items()
            if meta['var_type'] == 'discrete']
        disc_vars_and_dtypes: Union[List[Tuple[str, Any]], List[Tuple[int, Any]]] = all_disc_vars_and_dtypes

        if df is not None:
            if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
                raise ValueError("purify.dataset.metadata.Metadata :: discrete_vars_and_dtypes()\n"
                                 "The set of variables (i.e., features/columns) "
                                 "of the given `df` (i.e., pandas DataFrame) "
                                 "is NOT a subset of the one of the given `dataset`.")
            disc_vars_and_dtypes = [(variable, data_type) for variable, data_type in all_disc_vars_and_dtypes
                                    if variable in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: discrete_vars_and_dtypes()")
            print(f"Discrete variables and data types of the `{dataset}` dataset:\n{all_disc_vars_and_dtypes}")
            print(f"Discrete variables and data types to be returned:\n{disc_vars_and_dtypes}")
        return disc_vars_and_dtypes

    @classmethod
    def target_var(cls,
                   dataset: str = 'adult',
                   df: pd.DataFrame = None,
                   verbose: bool = False) -> Union[str, int]:
        target: Union[str, int] = [
            var for var in Metadata.DATASETS[dataset] if Metadata.DATASETS[dataset][var]['target']][0]

        if df is not None and target not in df.columns:
            raise ValueError("purify.dataset.metadata.Metadata :: target_var()\n"
                             "The set of variables (i.e., features/columns) "
                             "of the given `df` (i.e., pandas DataFrame) "
                             f"does NOT have the (target) variable `{target}`.")
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: target_var()")
            print(f"Target variable of the `{dataset}` dataset: {target}")
        return target

    @classmethod
    def vars_to_drop(cls,
                     dataset: str = 'adult',
                     df: pd.DataFrame = None,
                     verbose: bool = False) -> Union[List[str], List[int]]:
        """Return a list of the variables (i.e., features/columns) to be dropped of the given `dataset`.
        If the given pandas DataFrame (i.e., `df`) is NOT None then the list of its columns has to be
        a subset of the columns of the given `dataset` and
        the list to be returned has only variables that are present in the given `df`.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported by this class.
        df : DataFrame
            A pandas DataFrame (only) with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output,
            which is useful, for instance, to debug and to trace the execution.

        Returns
        -------
        disc_vars_and_dtypes : Union[List[Tuple[str, Any]], List[Tuple[int, Any]]]
            A list with the discrete (aka categorical) variables (i.e., features/columns) and
            their data types of the given `dataset` or
            a subset of those that are present in the given `df` (i.e., in the given pandas DataFrame).

        Raises
        ------
        ValueError
            If the given `df` (i.e., the given pandas DataFrame) is NOT None and
            the columns of the given `df` is NOT a subset of the ones of the given `dataset`.
        """
        all_vars_to_drop: Union[List[str], List[int]] = [
            variable for variable, meta in Metadata.DATASETS[dataset].items() if meta['drop']]
        vars_to_drop: Union[List[str], List[int]] = all_vars_to_drop

        if df is not None:
            if not set(df.columns).issubset(set(Metadata.DATASETS[dataset].keys())):
                raise ValueError("purify.dataset.metadata.Metadata :: vars_to_drop()\n"
                                 "The set of variables (i.e., features/columns) "
                                 "of the given `df` (i.e., pandas DataFrame) "
                                 "is NOT a subset of the one of the given `dataset`.")
            vars_to_drop = [variable for variable in all_vars_to_drop if variable in df.columns]
        if verbose:
            print()
            print("purify.dataset.metadata.Metadata :: vars_to_drop()")
            print(f"Variables to drop of the {dataset} dataset:\n{all_vars_to_drop}")
            print(f"Variables to drop to be returned:\n{vars_to_drop}")
        return vars_to_drop

