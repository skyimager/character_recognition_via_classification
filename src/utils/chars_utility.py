#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import string

def get_chars74k_label_map():
    """
    This function generates a label map for the chars74K datasetselfself.
    This will help to display the true label from the predicted  class.
    """
    # samples 1 through 10 are numbers '0' - '9'
    # samples 11 through 36 are uppercase letters
    # samples 37 through 62 are lowercase letters
    num_start = 0
    upper_start = 10
    lower_start = 36

    label_map = dict()
    for label, char in enumerate(string.digits, start = num_start):
        label_map[label] = char

    for label, char in enumerate(string.ascii_uppercase, start = upper_start):
        label_map[label] = char

    for label, char in enumerate(string.ascii_lowercase, start = lower_start):
        label_map[label] = char

    return label_map

