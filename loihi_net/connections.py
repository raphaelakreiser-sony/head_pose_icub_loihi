# Copyright(c) 2020 University of Zurich. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Intel Corporation nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Functions to connect populations of neurons in an all:all, 1:1 or shifted 1:1 manner
Author: Raphaela Kreiser (rakrei@ini.uzh.ch)
"""

import numpy as np


def all2all(pre_num_neurons, post_num_neurons, weight):
    """
     Give matrix of all2all communication
    """

    return weight * np.ones((post_num_neurons, pre_num_neurons))


def connect_populations_1to1(p1, p2, shift_offset=0):
    """
    :param p1: size of pre-synaptic neuron group
    :param p2: size of post-synaptic neuron group
    :param shift_offset: if asymmetric connectivity, positive values shift right, negative left
    :return: lists of pre- and post-synaptic indices
    """

    list_exc_pre = []
    list_exc_post = []

    # Sweep over all neurons of both populations (p2 receive list_exc_pre, list_exc_post)
    for srcNeuron in range(p1):
        for destNeuron in range(p2):

            # Check if shift or not
            if (shift_offset == 0):
                if (srcNeuron == destNeuron):
                    list_exc_pre.append(srcNeuron)
                    list_exc_post.append(destNeuron)
            elif (shift_offset > 0):
                if ((destNeuron + shift_offset <= p2)
                        & (destNeuron == srcNeuron + shift_offset)):
                    list_exc_pre.append(srcNeuron)
                    list_exc_post.append(destNeuron)
                elif ((srcNeuron + shift_offset >= p2)
                      & (destNeuron == -1 + shift_offset)):
                    list_exc_pre.append(srcNeuron)
                    list_exc_post.append(-1 + 2 * shift_offset + srcNeuron - p2)
            elif (shift_offset < 0):
                if ((srcNeuron + shift_offset >= 0)
                        & (srcNeuron == destNeuron - shift_offset)):
                    list_exc_pre.append(srcNeuron)
                    list_exc_post.append(destNeuron)
                if ((srcNeuron + shift_offset < 0)
                        & (destNeuron == p2 + shift_offset + srcNeuron)):
                    list_exc_pre.append(srcNeuron)
                    list_exc_post.append(destNeuron)
            destNeuron = destNeuron + 1
        srcNeuron = srcNeuron + 1
    return list_exc_pre, list_exc_post


def connect_populations_inh_shift(p1, p2):
    """
    :param p1: size of pre-synaptic neuron group
    :param p2: size of post-synaptic neuron group
    :return: lists of pre- and post-synaptic indices
             where the pre-synaptic neurons p1 inhibit
             all neurons of p2 except for the one with same index
    """
    list_inh_pre = []
    list_inh_post = []
    # Sweep over all neurons of both populations (p2 receive list_exc_pre, list_exc_post)
    for srcNeuron in range(p1):
        for destNeuron in range(p2):
            # inhibit all but neuron with same index
            if (srcNeuron != destNeuron):
                list_inh_pre.append(srcNeuron)
                list_inh_post.append(destNeuron)
            destNeuron = destNeuron + 1
        srcNeuron = srcNeuron + 1
    return list_inh_pre, list_inh_post
