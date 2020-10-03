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
Main file for the iCub heading direction project from the following publication:
Kreiser, R., Renner, A., Leite, V. R., Serhan, B., Bartolozzi, C., Glover, A., & Sandamirskaya, Y. (2020).
An On-chip Spiking Neural Network for Estimation of the Head Pose of the iCub Robot. Frontiers in Neuroscience, 14.

For questions, please contact: Raphaela Kreiser (rakrei@ini.uzh.ch) or Alpha Renner (alpren@ini.uzh.ch)
"""

import os
import time
import matplotlib as mpl
import json

print(mpl.get_backend())
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
else:
    mpl.use('Qt4Agg')
print(mpl.get_backend())
import matplotlib.pyplot as plt
from nxsdk.graph.processes.phase_enums import Phase
import hd_net_icub.loihi_net.hd_net as hd
from hd_net_icub.loihi_net.parameters import params
from loihi_tools.snip_tools import get_map_from_iport, get_map_from_oport, define_input_indices, \
    define_output_indices, build_shared_libary, overwrite_definitions, write_output_map_to_file

os.environ['KAPOHOBAY'] = '1'

try:
    __file__
except:
    __file__ = os.path.join(os.getcwd(), 'main_HDnet.py')


def setup_network(params):
    """
    setup the network on Loihi using the parameters defined in parameters.py
    :params: parameters as given in hd_net_icub.loihi_net.parameters
    :return: board, spike_probes, state_probes, port_cons, HD
    """
    #  create network using connections as given in hd_net_icub.loihi_net.hd_net
    net, InputPort, spike_probes, state_probes, port_cons, HD = hd.setup_loihi_network(params=params)

    snip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'snips')
    embedded_snip_path = os.path.join(snip_path, "embedded_snip.c")

    # Write SIM_TIME to embedded SNIP
    definitions = {"SIM_TIME": params['sim_time']}
    overwrite_definitions(embedded_snip_path, definitions)

    print('compiling...')
    start_compile_time = time.time()
    board = net.compiler.compile(net)
    print('compile took:', time.time() - start_compile_time)

    # Extract i/o axon indices needed for spike readout and input stimulation
    input_map = get_map_from_iport(InputPort)
    print("Input axon map \n" + str(input_map))
    define_input_indices(embedded_snip_path, "input_con", input_map)

    # Path to dump the recorded data
    dump_dir = os.path.join(os.path.expanduser('~'), 'yarp_dump')
    indices_fiename = os.path.join(dump_dir, "indices.txt")
    try:
        os.remove(indices_fiename)
    except Exception as e:
        print(e)
    output_maps = {}
    # use spike_probes as defined in the Loihi network to get i/o addresses
    for sp in spike_probes:
        output_map = get_map_from_oport(spike_probes[sp][0])
        output_maps[sp] = output_map
        print("Output axon map " + str(sp) + "\n" + str(output_map))
        # Define i/o axon indices in embedded snip
        define_output_indices(embedded_snip_path, "output_con_" + str(sp), output_map)

        write_output_map_to_file(indices_fiename, sp, output_maps[sp])

    # write parameter settings to file
    with open(indices_fiename, 'a') as f:
        f.write('SPEED_INTEGRATION_THR: ' + str(params['speed_integration_thr']) + '\n')
        f.write('SIM_TIME: ' + str(params['sim_time']) + '\n')
        f.write('WITH_VISION: ' + str(params['with_vision']) + '\n')
        f.write('START_BIAS: ' + str(params['start_bias']) + '\n')
        f.write('BIAS_FACTOR: ' + str(params['bias_factor']) + '\n')

        f.write(json.dumps(params))

    # replace defines in host snip before it is compiled
    definitions_hostSNIP = {"SPEED_INTEGRATION_THR": params['speed_integration_thr'],
                            "BIAS_FACTOR": params['bias_factor'],
                            "START_BIAS": params['start_bias']}
    overwrite_definitions(os.path.join(snip_path, "yarp_host_snip.cc"), definitions_hostSNIP)
    # Build the shared library for the host architecture
    shared_library_path = build_shared_libary(build_script_path=snip_path)

    # Create the host snip and assign it to run in Concurrent Execution phase
    hostProcess = board.createSnip(phase=Phase.HOST_CONCURRENT_EXECUTION,
                                   library=shared_library_path)

    # Creating snip to run in spiking phase
    # This snip inserts spike in axonId received via the channel, which is sent from the host.
    cFilePath = embedded_snip_path
    includeDir = snip_path
    funcName = "runSpiking"
    guardName = "doSpiking"
    embeddedProcess = board.createSnip(
        phase=Phase.EMBEDDED_SPIKING,
        cFilePath=cFilePath,
        includeDir=includeDir,
        funcName=funcName,
        guardName=guardName)

    # Creating a channel named input for sending data from host snip to embedded snip (spiking phase)
    inChannel = board.createChannel('in', messageSize=4, numElements=100)
    # Connecting input channel from inputProcess to embeddedProcess making it send channel
    inChannel.connect(hostProcess, embeddedProcess)

    # Create a channel named feedback for getting the feedback value from the embedded snip
    outChannel = board.createChannel('out', messageSize=4, numElements=100)
    # Connecting feedback channel from embeddedProcess to outputProcess making it receive channel
    outChannel.connect(embeddedProcess, hostProcess)

    return board, spike_probes, state_probes, port_cons, HD


def plot_probes(spike_probes, state_probes, sim_time):
    """
    Plot spikes and neuron states if "offline" probes were recorded
    :spike_probes: created in hd_net.py, defines from which neuron groups spikes were recorded
    :state_probes: created in hd_net.py, defines from which neuron groups the membrane potential was recorded
    :sim_time: given in parameters.py, how many timesteps are in the simulation
    """
    
    if len(spike_probes) > 0:
        print('create figure')
        fig = plt.figure(1, figsize=(18, 18))
        for i, sp in enumerate(spike_probes):
            plt.subplot(len(spike_probes), 1, i + 1)
            spike_probes[sp][0].plot()
            plt.xlim(0, sim_time)
            plt.title(sp)
        fileName1 = "pi_icub_raster.png"
        if haveDisplay:
            fig.savefig(fileName1)
            plt.show()
        else:
            print("No display available, saving to file " + fileName1)
            fig.savefig(fileName1)

    if len(state_probes) > 0:
        fig = plt.figure(1, figsize=(18, 18))
        for i, sp in enumerate(state_probes):
            plt.subplot(len(state_probes), 1, i + 1)
            for pl in state_probes[sp]:
                pl[0].plot()
                plt.xlim(0, sim_time)
                plt.title(sp)
        fileName1 = "pi_icub_weights.png"
        if haveDisplay:
            fig.savefig(fileName1)
            plt.show()
        else:
            print("No display available, saving to file " + fileName1)
            fig.savefig(fileName1)


if __name__ == "__main__":
    # Setup the network, host snips and embedded snip
    board, spike_probes, state_probes, iport, oport = setup_network(params)

    try:
        board.run(params['sim_time'])
    except:
        print('run failed or stopped')
    finally:
        board.disconnect()
        print('run stopped, board disconnected')

    plot_probes(spike_probes, state_probes, sim_time=params['sim_time'])
