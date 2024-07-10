# Extra-tiny network - very small, non-pretrained network. We ended up not using this in analysis
type_1 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4],
    'model_name': 'extra_tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': False,
    'on_off': False,
    'reg_1000': False,
    'method2': False,
    'idx': 1
}

# Simple, pretrained network M1 on CIFAR-10
type_2 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': False,
    'reg_1000': False,
    'method2': False,
    'idx': 2
}

# Not used for the final paper
type_3 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': True,
    'reg_1000': False,
    'method2': False,
    'idx': 3
}

# Not used for the final paper
type_4 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': False,
    'reg_1000': True,
    'method2': False,
    'idx': 4
}

# Not used for the final paper
type_5 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': True,
    'reg_1000': True,
    'method2': False,
    'idx': 5
}

# starting with method2
type_6 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': False,
    'reg_1000': False,
    'method2': True,
    'idx': 6
}

# Not used for the final paper
type_7 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': True,
    'reg_1000': False,
    'method2': True,
    'idx': 7
}

# Not used for the final paper
type_8 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 3, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2],
    'pretrained': True,
    'on_off': False,
    'reg_1000': True,
    'method2': True,
    'idx': 8
}

# Not used for the final paper
type_9 = {
    'dataset': 'CIFAR10',
    'n_loops': [1, 2, 3, 4, 5],
    'model_name': 'tiny',
    'run_no': [0, 1, 2, 3, 4],
    'pretrained': True,
    'on_off': True,
    'reg_1000': True,
    'method2': True,
    'idx': 9
}


# RViTs trained on PET dataset. The error is propagated only from the last output - baseline.
# No suffix!
type_pet_m1_baseline = {
        'dataset': 'PET',
        'n_loops': [1],
        'model_name': 'tiny',
        'run_no': [0, 1, 2, 3, 4],
        'pretrained': True,
        'on_off': False,
        'reg_1000': False,
        'method2': False,
        'suffix': ['']
}


# RViTs trained on PET dataset. The error is propagated from all the output - baseline.
# No suffix!
type_pet_m2_baseline = {
        'dataset': 'PET',
        'n_loops': [1],
        'model_name': 'tiny',
        'run_no': [0, 1, 2, 3, 4],
        'pretrained': True,
        'on_off': False,
        'reg_1000': False,
        'method2': True,
        'suffix': ['']
}


# RViTs trained on PET dataset. The error is propagated only from the last output - baseline. (BASELINE RVIT METHOD 1)
type_pet_m1_all = {
        'dataset': 'PET',
        'n_loops': [2, 3, 4],
        'model_name': 'tiny',
        'run_no': [0, 1, 2, 3, 4],
        'pretrained': True,
        'on_off': False,
        'reg_1000': False,
        'method2': False,
        'suffix': ['', '_blur', '_blurinv', '_transform']
}


# RViTs trained on PET dataset. The error is propagated only from the last output - baseline. (BASELINE RVIT METHOD 2)
type_pet_m2_all = {
        'dataset': 'PET',
        'n_loops': [2, 3, 4],
        'model_name': 'tiny',
        'run_no': [0, 1, 2, 3, 4],
        'pretrained': True,
        'on_off': False,
        'reg_1000': False,
        'method2': True,
        'suffix': ['', '_blur', '_blurinv', '_transform']
}
