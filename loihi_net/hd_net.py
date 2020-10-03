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
Contains functions to create the heading direction net

For questions, please contact: Raphaela Kreiser (rakrei@ini.uzh.ch) or Alpha Renner (alpren@ini.uzh.ch)
"""

import numpy as np
from scipy import sparse
from collections import OrderedDict
import nxsdk.api.n2a as nx
import hd_net_icub.loihi_net.connections as con
from nxsdk.graph.nxprobes import SpikeProbeCondition, IntervalProbeCondition
from loihi_tools.compartment_tools import create_group_core_zero
from loihi_tools.compartment_tools import create_distributed_group_over_cores as create_group


def decay(tau):
    """
    converts time constant to decay for Loihi neurons
    :param tau: time constant
    :return: converts time constant tau to decay
    """
    return int(4096 / tau)


def setup_loihi_network(params):
    """
    sets up the HD network
    :param num_rings: determines the dimensionality, in how many axes do we estimate
    :param num_neurons: number of neurons on one ring network
    :param feature_size: number of pixels, determines the size of the visual receptive field
    :param params: dictionary defined in the main with all parameters (e.g. threshold, tau, and weight values)
    :param with_vision: True if vision data is used
    :return: net, InputPort, spike_probes, state_probes, port_cons, HD
    """

    num_rings = params['num_rings']
    num_hd = params['num_neurons']
    num_vis = params['feature_size']
    # num_src determines the number of neurons in spike generators and "helper" neurons
    num_src = 1

    # start position is set to the center neuron
    start_pos = [int(num_hd / 2), int(num_hd / 2)]

    # include GHD to enable gazing at learned positions
    include_GHD = True

    # Create Loihi network
    net = nx.NxNet()

    #  Create Prototypes
    cxProto0 = nx.CompartmentPrototype(vThMant=params['threshold0'],
                                       compartmentCurrentDecay=decay(params['taui0']),
                                       compartmentVoltageDecay=decay(params['tauv0']))

    cxProto1 = nx.CompartmentPrototype(vThMant=params['threshold1'],
                                       compartmentCurrentDecay=decay(params['taui1']),
                                       compartmentVoltageDecay=decay(params['tauv1']))

    cxProto_learn = nx.CompartmentPrototype(vThMant=params['threshold_RHD'],
                                            compartmentCurrentDecay=decay(params['taui0']),
                                            compartmentVoltageDecay=decay(params['tauv_RHD']),
                                            enableSpikeBackprop=1, enableSpikeBackpropFromSelf=1)

    cxProto_vis = nx.CompartmentPrototype(vThMant=params['threshold_vis'],
                                          # refractoryDelay=63,
                                          compartmentCurrentDecay=decay(params['taui_vis']),
                                          compartmentVoltageDecay=decay(params['tauv_vis']))

    cxProto_in = nx.CompartmentPrototype(vThMant=params['threshold_in'],
                                         compartmentCurrentDecay=decay(params['taui_in']),
                                         compartmentVoltageDecay=decay(params['tauv_in']))

    #  Create Compartments
    HD, IHD, SL, SR, VR, VL, North = [], [], [], [], [], [], []
    In, RHD, GHD, vision, goal, Keep_gate, HD_gate = [], [], [], [], [], [], []

    # In will receive external stimulation from the InputPort
    In.append(create_group_core_zero(net, name="In1", prototype=cxProto_in, size=5 + num_vis + num_vis))
    sc = 0

    '''
    Names of neuron groups:
    
    VR, VL: Velocity Right, Velocity Left, receives encoder information
    North: Used to set the initial activity at a specific place on the ring
    HD: Head Direction, outputs the estimated orientation as neuron index which is converted to degrees in /data_visualization
    IHD: Integrated Head Direction
    SL, SR: Shift Left, Shift Right
    RHD: Reset Heading Direction, is connected to the visual input neuron and learns object associations
    GHD: Goal Heading Direction, is connected to the visual input neuron and learns object associations
    see https://doi.org/10.3389/fnins.2020.00551 for more information
    '''
    # specify how many 1D rings you want to have, here we have one for pitch and one for yaw, so range(2)
    # the number of rings determines the dimensions, i.e. how many axis are estimated
    # the parameters are set in main_hd_net.py
    for ii in range(num_rings):
        # create groups
        VR.append(create_group_core_zero(net, name="VR" + str(ii), prototype=cxProto1, size=num_src))
        VL.append(create_group_core_zero(net, name="VL" + str(ii), prototype=cxProto1, size=num_src))
        # This group simply initilizes the peak at a given position on the ring
        North.append(create_group_core_zero(net, name="N" + str(ii), prototype=cxProto0, size=1))
        
        # Groups are distributed over cores on Loihi
        HD.append(create_group(net, 1 + sc, 5 + sc, name="HD" + str(ii), prototype=cxProto0, size=num_hd))
        IHD.append(create_group(net, 5 + sc, 10 + sc, name="IHD" + str(ii), prototype=cxProto0, size=num_hd))
        SL.append(create_group(net, 10 + sc, 15 + sc, name="SL" + str(ii), prototype=cxProto0, size=num_hd))
        SR.append(create_group(net, 15 + sc, 20 + sc, name="SR" + str(ii), prototype=cxProto0, size=num_hd))
        RHD.append(create_group(net, 20 + sc, 25 + sc, name="RHD" + str(ii), prototype=cxProto_learn, size=num_hd))
        if include_GHD:
            GHD.append(create_group(net, 25 + sc, 30 + sc, name="GHD" + str(ii), prototype=cxProto_learn, size=num_hd))
            sc += 30
        else:
            sc += 30

    vision.append(
        create_group(net, start_core=61, end_core=62, name="vision", prototype=cxProto_vis, size=num_vis))
    if include_GHD:
        goal.append(
            create_group(net, start_core=62, end_core=63, name="goal", prototype=cxProto_vis, size=num_vis))
    # Create spike input port
    InputPort = net.createSpikeInputPortGroup(size=In[0].numNodes)

    # Connection prototypes
    connProtoEx = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)
    connProtoInh = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY)
    connProtoMix = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
    connProtoEx_in = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY, weight=params['w_in'])
    connProtoInh_V = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY, weight=-1)
    connProtoEx_vis = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY, weight=params['w_in_vis'])
    connProto_iport = nx.ConnectionPrototype(weight=2)

    # Create learning rule used by the learning-enabled synapse
    lr = net.createLearningRule(dw='2*y0*x1-32*x0',
                                x1Impulse=125, x1TimeConstant=2, tEpoch=20)
    connProtoLrn = nx.ConnectionPrototype(enableLearning=True, learningRule=lr, numWeightBits=8,
                                          signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)

    # connect initialization input
    In[0][0].connect(North[0], prototype=connProtoEx_in)
    In[0][0].connect(North[1], prototype=connProtoEx_in)
    # connect input to velocity neurons
    In[0][1].connect(VR[0], prototype=connProtoEx_in)
    In[0][2].connect(VL[0], prototype=connProtoEx_in)
    In[0][3].connect(VR[1], prototype=connProtoEx_in)
    In[0][4].connect(VL[1], prototype=connProtoEx_in)

    if params['with_vision']:
        # connect vision input
        for i in range(num_vis):
            In[0][5 + i].connect(vision[0][i], prototype=connProtoEx_vis)

        # connect goal input
        if include_GHD:
            for j in range(num_vis):
                In[0][5 + j].connect(goal[0][j], prototype=connProtoEx_vis)
                In[0][5 + j + num_vis].connect(goal[0][j], prototype=connProtoEx_vis)

    for i in range(num_rings):
    # Set all connections inside each ring network

        # Vision inhibits IHD and Velocities
        w_matrix = con.all2all(num_vis, num_hd, params['w_vis_IHD_i'])
        vision[0].connect(IHD[i], prototype=connProtoInh, weight=w_matrix)

        if params['w_RHD_IHD_e'] > 0.0:
            vision[0].connect(VL[i], prototype=connProtoInh_V)
            vision[0].connect(VR[i], prototype=connProtoInh_V)

        # HD WTA
        w_matrix = np.ones((num_hd, num_hd)) * params['w_HD_HD_i']
        pre_e, post_e = con.connect_populations_1to1(num_hd, num_hd)
        for j in range(len(pre_e)):
            w_matrix[post_e[j], pre_e[j]] = params['w_HD_HD_e']
        HD[i].connect(HD[i], prototype=connProtoMix, weight=w_matrix)

        # Connect HD to Shifts via inhibition
        w_matrix = np.zeros((num_hd, num_hd))
        pre, post = con.connect_populations_inh_shift(num_hd, num_hd)
        for j in range(len(pre)):
            w_matrix[post[j], pre[j]] = params['w_HD_S']
        HD[i].connect(SR[i], prototype=connProtoInh, weight=w_matrix, connectionMask=w_matrix != 0)
        HD[i].connect(SL[i], prototype=connProtoInh, weight=w_matrix, connectionMask=w_matrix != 0)

        # Shifts asymmetric connection to IHD
        w_matrix = np.zeros((num_hd, num_hd))
        pre, post = con.connect_populations_1to1(num_hd, num_hd, shift_offset=1)
        for j in range(len(pre) - 1):  # in order not to connect the last one -1
            w_matrix[pre[j], post[j]] = params['w_S_IHD']
        SR[i].connect(IHD[i], prototype=connProtoEx, weight=w_matrix, connectionMask=(w_matrix != 0))

        w_matrix = np.zeros((num_hd, num_hd))
        pre, post = con.connect_populations_1to1(num_hd, num_hd, shift_offset=-1)
        for j in range(len(pre) - 1):  # in order not to connect the last one -1
            w_matrix[pre[j], post[j]] = params['w_S_IHD']
        SL[i].connect(IHD[i], prototype=connProtoEx, weight=w_matrix, connectionMask=(w_matrix != 0))

        # IHD to HD, here one2one excitatory
        w_matrix = np.zeros((num_hd, num_hd))
        pre, post = con.connect_populations_1to1(num_hd, num_hd)
        for j in range(len(pre)):
            w_matrix[pre[j], post[j]] = params['w_IHD_HD_e']
        IHD[i].connect(HD[i], prototype=connProtoEx, weight=w_matrix, connectionMask=(w_matrix != 0))

        # IHD to HD, here inhibitory to all others
        w_matrix = np.zeros((num_hd, num_hd))
        pre, post = con.connect_populations_inh_shift(num_hd, num_hd)
        for j in range(len(pre)):
            w_matrix[post[j], pre[j]] = params['w_IHD_HD_i']
        IHD[i].connect(HD[i], prototype=connProtoMix, weight=w_matrix, connectionMask=(w_matrix != 0))

        # RHD to IHD for reset, one-to-one excitatory, inhibitory to all others
        if params['w_RHD_IHD_e'] > 0.0:
            w_matrix = np.zeros((num_hd, num_hd))
            pre, post = con.connect_populations_1to1(num_hd, num_hd)
            for j in range(len(pre)):
                w_matrix[pre[j], post[j]] = params['w_RHD_IHD_e']
            RHD[i].connect(IHD[i], prototype=connProtoEx, weight=w_matrix, connectionMask=(w_matrix != 0))

        if params['w_RHD_IHD_i'] > 0.0:
            w_matrix = np.zeros((num_hd, num_hd))
            pre, post = con.connect_populations_inh_shift(num_hd, num_hd)
            for j in range(len(pre)):
                w_matrix[post[j], pre[j]] = params['w_RHD_IHD_i']
            RHD[i].connect(IHD[i], prototype=connProtoMix, weight=w_matrix, connectionMask=(w_matrix != 0))

        # HD to RHD subthreshold
        w_matrix = np.zeros((num_hd, num_hd))
        pre, post = con.connect_populations_1to1(num_hd, num_hd)
        for j in range(len(pre)):
            w_matrix[pre[j], post[j]] = params['w_HD_RHD']
        HD[i].connect(RHD[i], prototype=connProtoEx, weight=w_matrix, connectionMask=(w_matrix != 0))

        if include_GHD:
            # HD to GHD subthreshold
            w_matrix = np.zeros((num_hd, num_hd))
            pre, post = con.connect_populations_1to1(num_hd, num_hd)
            for j in range(len(pre)):
                w_matrix[pre[j], post[j]] = params['w_HD_RHD']  # uses the same weight
            HD[i].connect(GHD[i], prototype=connProtoEx, weight=w_matrix, connectionMask=(w_matrix != 0))

        # plastic synapse to learn landmark position (RHD/GHD) association
        w_matrix = con.all2all(num_vis, num_hd, params['w_init_plast'])

        if params['w_RHD_IHD_e'] > 0.0:
            syn_plast = vision[0].connect(RHD[i], prototype=connProtoLrn, weight=w_matrix)

        if include_GHD:
            syn_plast_GHD = goal[0].connect(GHD[i], prototype=connProtoLrn, weight=w_matrix)

        # Velocities excite shifts
        w_matrix = con.all2all(num_vis, num_hd, params['w_V_S'])
        VR[i].connect(SR[i], prototype=connProtoEx, weight=w_matrix)
        VL[i].connect(SL[i], prototype=connProtoEx, weight=w_matrix)

        # Initial Heading
        w_matrix = np.zeros((num_hd, num_src))
        w_matrix[start_pos[i], :] = params['w_North_HD']
        North[i].connect(HD[i], prototype=connProtoEx, weight=w_matrix)  # , connectionMask=(w_matrix != 0))

    # input port to input group
    port_cons = InputPort.connect(In[0], prototype=connProto_iport, connectionMask=sparse.eye(In[0].numNodes))
    logicalAxonIds = []
    for port_c in port_cons:
        logicalAxonIds.append(port_c.nodeIds)

    # Hacking spike probes to create spike counter and defer probing
    # Check tutorial on lakemont spike counters
    # lmt counters created will be read from Embedded snip
    probeParameters = [nx.ProbeParameter.SPIKE]

    pc_yarp = SpikeProbeCondition(dt=1, tStart=100000000)
    pc_probe = SpikeProbeCondition(dt=1, tStart=1)

    # if real-time recording using yarp: set to pc_yarp, for offline probing set to pc_probe
    pc = pc_yarp  # pc_probe
    spike_probes = OrderedDict()
    spike_probes['HD1'] = HD[0].probe(probeParameters, pc)
    spike_probes['HD2'] = HD[1].probe(probeParameters, pc)
    if include_GHD:
        # plot GHD instead of RHD
        spike_probes['RHD1'] = GHD[0].probe(probeParameters, pc)
        spike_probes['RHD2'] = GHD[1].probe(probeParameters, pc)
        spike_probes['vision'] = goal[0].probe(probeParameters, pc)
    else:
        spike_probes['RHD1'] = RHD[0].probe(probeParameters, pc)
        spike_probes['RHD2'] = RHD[1].probe(probeParameters, pc)
        spike_probes['vision'] = vision[0].probe(probeParameters, pc)

    # Create state probes
    # voltageProbe1 = HD[1].probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE])
    state_probes = OrderedDict()
    # state_probes['weight'] = syn_plast.probe([nx.ProbeParameter.SYNAPSE_WEIGHT], IntervalProbeCondition(dt=50))

    return net, InputPort, spike_probes, state_probes, port_cons, HD
