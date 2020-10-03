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
Contains the parameters used for
Kreiser, R., Renner, A., Leite, V. R., Serhan, B., Bartolozzi, C., Glover, A., & Sandamirskaya, Y. (2020).
An On-chip Spiking Neural Network for Estimation of the Head Pose of the iCub Robot. Frontiers in Neuroscience, 14.

"""

# threshold of the neurons
th = 100
with_reset = True

params = {}

# how many timesteps should the simulation have
params['sim_time'] = 100000
params['speed_integration_thr'] = 0.5
# at which timestep should a drift be introduced
params['start_bias'] = 30000
# the factor of the drift, 1.0 is zero drift
params['bias_factor'] = 1.0
params['with_vision'] = True

# how many axis are estimated
params['num_rings'] = 2
# how many neurons on a single ring
params['num_neurons'] = 100
# how many objects/features
params['feature_size'] = 1

params['threshold0'] = int(th * 0.9)
params['tauv0'] = 1
params['taui0'] = 1

params['threshold_RHD'] = int(th * 0.9)
params['tauv_RHD'] = 1

params['threshold1'] = int(th * 0.9)
params['tauv1'] = 1
params['taui1'] = 1

params['threshold_vis'] = int(th * 0.9)  # 2.5* int(th * 0.9)
params['tauv_vis'] = 1
params['taui_vis'] = 1

params['threshold_in'] = 1
params['tauv_in'] = 1
params['taui_in'] = 1

# weights for HD net path integration
params['w_North_HD'] = 1.5 * th
params['w_in'] = 1 * th
params['w_in_inh'] = -1 * th
params['w_HD_HD_e'] = 1.2 * th
params['w_HD_HD_i'] = -0  # -0.5 * th
params['w_HD_S'] = -0.5 * th
params['w_V_S'] = 1 * th
params['w_S_IHD'] = 1 * th
params['w_IHD_HD_e'] = 2 * th
params['w_IHD_HD_i'] = -1 * th

# weights for visual reset
params['w_init_plast'] = 0.8 * th

params['w_in_vis'] = 1 * th

# params['w_in_goal'] = 2.1 * th

# params['w_vis_RHD'] = 0.8 * th
params['w_HD_RHD'] = 0.2 * th

if with_reset:
    params['w_RHD_IHD_e'] = 1 * th
    params['w_RHD_IHD_i'] = -0.7 * th
else:
    params['w_RHD_IHD_e'] = 0
    params['w_RHD_IHD_i'] = 0

params['w_vis_IHD_i'] = -1 * th
