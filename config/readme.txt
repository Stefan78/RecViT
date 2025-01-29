# Create a "definitions.py" file using the template bellow:
# Alternatively, copy and rename this file to "definitions.py"

'''
Directory specification for data & root
Set DATA_DIR according to your specifications
'''

import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = ROOT_DIR + '/data'
NETWORKS_DIR = DATA_DIR + '/networks'
DATASET_DIR = DATA_DIR + '/datasets'
AES_DIR = DATA_DIR + '/aes'

