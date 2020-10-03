/*
Copyright(c) 2020 University of Zurich. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Intel Corporation nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Adapted from nxnet tutorial 26 (concurrent host snip with yarp integration)

#include <stdio.h>
#include <stdlib.h>
#include <yarp/os/all.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
#include "nxsdkhost.h"
#include "boost/date_time.hpp"

#define SPEED_INTEGRATION_THR 0.5
//has to match head_control such that integration speed matches movement speed of the robot
#define YAW_INDEX_YARP 2
#define PITCH_INDEX_YARP 0
#define PREC_SCALER 100000.0 // so that rounding doesn't become zero
#define MAX_YAW_SPEED 2*SPEED_INTEGRATION_THR * PREC_SCALER
#define MAX_PITCH_SPEED SPEED_INTEGRATION_THR* PREC_SCALER

#define START_BIAS 30000
#define BIAS_FACTOR 1.0

// has to match spiking.c
// Correspond to input ports on Loihi
#define INIT_INDEX_LOIHI 0
#define YAW_POS_INDEX_LOIHI 1
#define YAW_NEG_INDEX_LOIHI 2
#define PITCH_POS_INDEX_LOIHI 3
#define PITCH_NEG_INDEX_LOIHI 4
#define VISION_INDEX_LOIHI 5 // vision has to be the last, as there can be several vision neurons

// determines the size of the receptive field of the vision neuron (in pixels)
#define RADIUS_VIS 5 


using yarp::os::Bottle;
using yarp::os::BufferedPort;
using yarp::os::Network;
using std::vector;
using BP = BufferedPort<Bottle>;


// has to match ATIS features, center of ATIS which is presumably 304x240
// should be the center pixel
std::vector<int> VIS_ID = {151, 119};

namespace yarp_globals {
// Static globals (Construct on First Use Idiom)
// Avoids static initialization fiasco and is thread-safe
Network& g_yarp_network() {
  // global yarp network
  static Network* yarp = new Network();
  return *yarp;
  }
}  // namespace yarp_globals

namespace yarp_demo {
// Instantiate the globals in this namespace
auto& yarp = yarp_globals::g_yarp_network();
const char yarp_in_vision[] = "/loihi/vision:i";
const char yarp_in_cmd[] = "/loihi/vel_cmd:i";
const char yarp_out_raster[] = "/loihi/spike:o";
const char yarp_out_cmd[] = "/loihi/state:o";


void check_success(bool result, const std::string& name) {
  // Checks if the result is true while opening a port with name
  if (!result) {
    std::cerr << "Failed to create port on name: " << name << std::endl;
    std::cerr << "Maybe you need to start a nameserver (run 'yarpserver')"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// InputProcess which writes to EmbeddedSnip via input channel and reads from
// YARP port /cx/in
class HostProcess : public ConcurrentHostSnip {
  const std::string inputChannelLoihi = "in";
  const std::string outputChannelLoihi = "out";
  BP yarpInputPortVision;
  BP yarpInputPortCmd;
  BP yarpOutputPortCmd;
  BP yarpOutputPortRaster;

  int initial_idx = INIT_INDEX_LOIHI;
  int t_step=0;

  int32_t timeStamp;

  int32_t spike_buffer[30] = {0};
  double pitch;
  double yaw;
  double vtsScaler = 0.0002 / (0.000000001 * 80);

  // detune factor should be different from 1.0 if artificial drift is induced
  double detune_factor = 1.0;
  double detune_factor_later = BIAS_FACTOR;
  // time when the drift is induced
  int detune_time = START_BIAS; 
  
  // time when to stimulate a goal neuron to initiate the gaze action
  int goal_time = 400000;
  int goal_neuron =0;

  std::vector<double> thresholds = {MAX_YAW_SPEED,MAX_YAW_SPEED,MAX_PITCH_SPEED,MAX_PITCH_SPEED};
  std::vector<int> accumulator_indices = {YAW_POS_INDEX_LOIHI,YAW_NEG_INDEX_LOIHI,PITCH_POS_INDEX_LOIHI,PITCH_NEG_INDEX_LOIHI};
  int ref_period = 3;


 public:
  HostProcess() {
    // Opening the vision input port
    bool result_vision = yarpInputPortVision.open(yarp_in_vision);
    check_success(result_vision, yarp_in_vision);

    // Opening the other command input port
    bool result_cmd = yarpInputPortCmd.open(yarp_in_cmd);
    check_success(result_cmd, yarp_in_cmd);

    // Opening the output port for visualization on a ring
    bool result_o = yarpOutputPortCmd.open(yarp_out_cmd);
    check_success(result_o, yarp_out_cmd);

    // Opening the other output port for rasterplot visualization
    bool result_r = yarpOutputPortRaster.open(yarp_out_raster);
    check_success(result_o, yarp_out_raster);

    if(!yarp.connect("/head-control/vel_cmd:o", yarp_in_cmd, "fast_tcp"))
      std::cerr << "could not connect" << yarp_in_cmd <<std::endl;

    if(!yarp.connect("/atis_features/spike:o", yarp_in_vision, "fast_tcp"))
      std::cerr << "could not connect " << yarp_in_vision <<std::endl;

    if(!yarp.connect(yarp_out_cmd, "/head-dir-vis/head_direction:i", "fast_tcp"))
      std::cerr << "could not connect " << yarp_out_cmd <<std::endl;

    if(!yarp.connect(yarp_out_raster, "/vFramer/raster/AE:i", "fast_tcp"))
      std::cerr << "could not connect" << yarp_out_raster <<std::endl;

  }

public:

  // update speed values according to read movement command
  void update_speeds(Bottle &input, std::vector<int> &speeds)
  {
    double yaw_speed = input.get(YAW_INDEX_YARP).asDouble();
    if(yaw_speed > 0) {
      speeds[0] =  PREC_SCALER * yaw_speed * detune_factor;     //YAW_POS
      speeds[1] = 0;                                            // YAW_NEG
    }
    else if (yaw_speed < 0) {
      speeds[0] = 0;                            //YAW_POS
      speeds[1] = - PREC_SCALER * yaw_speed;    // YAW_NEG
    }
    else {
      speeds[0] = 0;//YAW_POS
      speeds[1] = 0;//YAW_NEG
    }
    double pitch_speed = input.get(PITCH_INDEX_YARP).asDouble();
    if(pitch_speed > 0) {
      speeds[2] =  PREC_SCALER * pitch_speed * detune_factor; //PITCH_POS
      speeds[3] = 0;//PITCH_NEG
    }
    else if(pitch_speed < 0) {
      speeds[2] = 0;//PITCH_POS
      speeds[3] = - PREC_SCALER * pitch_speed;//PITCH_NEG
    }
    else {
      speeds[2] = 0;//PITCH_POS
      speeds[3] = 0;//PITCH_NEG
    }
  }

 // check if a spike should be sent to Loihi because the input neuron crossed the threshold
 // accumulator is updated with current movement speeds and determines the threshold crossing
  vector<int> check_if_fired(std::vector<int> &speeds, std::vector<int> &accumulator,
                             std::vector<int> vision_spike,
                             std::vector<int> goal_spike, std::vector<int> &refractory, double dt)
  {
    vector<int> firing_neurons;
    
    for(int i = 0; i< accumulator.size(); i++) {
        refractory[i] -= 1;
        accumulator[i] += speeds[i] * dt;
        if (accumulator[i] > thresholds[i] && refractory[i]<1) {
            firing_neurons.push_back(accumulator_indices[i]);
            accumulator[i] -= thresholds[i];
            refractory[i] = ref_period;
            if (i%2==0){
                refractory[i+1] = ref_period;
            }
            if (accumulator[i] > thresholds[i]){
                accumulator[i] = thresholds[i];
            }
        }
    }
    
    // if the object was detected using software algorithm, send a vision spike
    for(int i = 0; i< vision_spike.size(); i++) {
      // std::cout << "push back vision spike" << VISION_INDEX_LOIHI+vision_spike[i] << std::endl;
      firing_neurons.push_back(VISION_INDEX_LOIHI+vision_spike[i]);
    }
    // if a gaze command is send to the icub, send it to the goal neuron on Loihi
    for(int i = 0; i< goal_spike.size(); i++) {
      std::cout << "send goal spike" << VISION_INDEX_LOIHI+goal_spike[i] << std::endl;
      firing_neurons.push_back(VISION_INDEX_LOIHI + goal_spike[i]);
    }
    return firing_neurons;
  }

  void run(std::atomic_bool& endOfExecution) override {
    // initialize vectors to zero
    std::vector<int> accumulator = {0, 0, 0, 0}; // yaw+, yaw-, pitch+, pitch-
    std::vector<int> speeds = {0, 0, 0, 0}; // yaw+, yaw-, pitch+, pitch-
    std::vector<int> refractory = {0, 0, 0, 0}; // yaw+, yaw-, pitch+, pitch-

    std::cout << "waiting for yarp to send input..." << std::endl;

    //check if there is an input (velocity)
    Bottle* inCmd = yarpInputPortCmd.read(true);
    if(!inCmd) {
      //ERROR QUIT
      std::cerr << "command read returned false before starting" << std::endl;
      return;
    }

    // change the input neuron according to movement speeds
    update_speeds(*inCmd, speeds);

    // initialize the network
    int one = 1;
    writeChannel(inputChannelLoihi.c_str(), &one, 1);
    writeChannel(inputChannelLoihi.c_str(), &initial_idx, 1);
    std::cout << "Neuron to start HD network is : " << initial_idx << std::endl;

    // set initialization flag
    //bool init_flag = true;
    auto prev_time = std::chrono::system_clock::now();

    while (endOfExecution == false) {

      //check vision
      Bottle* inVis = yarpInputPortVision.read(false);
      std::vector<int> send_vis_spike;
      std::vector<int> send_goal_spike;

      // if the object was detected, send a vision spike to Loihi
      if(inVis) {
        for(int i = 0; i< inVis->size(); i+=2) {
          double x_pixel = inVis->get(i).asDouble();
          double y_pixel = inVis->get(i+1).asDouble();
          if(( x_pixel >= VIS_ID[0] - RADIUS_VIS) && (x_pixel <= VIS_ID[0] + RADIUS_VIS)
          && (y_pixel >= VIS_ID[1] - RADIUS_VIS ) && (y_pixel <= VIS_ID[1] + RADIUS_VIS )) {
            send_vis_spike.push_back(i/2);
            }
        }
      }

     // at a predefined time, gaze at the object and activate the goal neuron, activating the learned pose
     if (t_step > goal_time && goal_neuron <=4) {
        send_goal_spike.push_back(goal_neuron);
        goal_time += 10;
        goal_neuron++;
        //std::cout << "Sending goal spike from host" << goal_neuron->toString() << std::endl;
      }

      //check cmds to the robot and update speeds accordingly
      Bottle* inCmd = yarpInputPortCmd.read(false);
      if(inCmd) {
          update_speeds(*inCmd, speeds);
        //std::cout << "Recieved " << inCmd->toString() << " speeds" << std::endl;
      }

      // to determine how long a timestep takes
      auto current_time = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_sec = current_time - prev_time;
      double dt = elapsed_sec.count();

      // check if input neurons crossed the threshold and determine if a spike should be send
      vector<int> neurons_fired = check_if_fired(speeds, accumulator, send_vis_spike,send_goal_spike, refractory, dt);
      prev_time = current_time;

      //write the neurons that fired, send to the embedded process on Loihi
      int amount_neurons = neurons_fired.size();
      writeChannel(inputChannelLoihi.c_str(), &amount_neurons, 1);

      for (int i=0; i < amount_neurons; i++) {
        writeChannel(inputChannelLoihi.c_str(), &neurons_fired[i], 1);
      }

      //read the number of neurons that spiked in the Loihi
      int numSpikes;
      readChannel(outputChannelLoihi.c_str(), &numSpikes, 1);

      //read the neuron ids that spiked
      readChannel(outputChannelLoihi.c_str(), spike_buffer, numSpikes);

      int sentinel;  // Flag to tell us if its end of execution
      readChannel(outputChannelLoihi.c_str(), &sentinel, 1);
      if (sentinel == -1) {
        std::cout << "Execution Over, Stop execution of Host Snip "
                    << std::endl;
        break;
      }

      // if neurons spiked on Loihi, send to rasterplot
      if (numSpikes) {

        Bottle& raster_out = yarpOutputPortRaster.prepare();
        raster_out.clear();
        raster_out.addString("AE");
        Bottle &spike_list = raster_out.addList();
        for(int i = 0; i < numSpikes; i++) {
            spike_list.addInt32(t_step * vtsScaler);
            uint16_t spike = spike_buffer[i];
            spike_list.addInt32(spike);
        }
        yarpOutputPortRaster.write(true);

      }
      // keep track of timesteps
      t_step++;
      if (t_step == detune_time) { 
        detune_factor = detune_factor_later;
      }

    }
  }

};

}  // namespace yarp_demo

using yarp_demo::HostProcess;
REGISTER_SNIP(HostProcess, ConcurrentHostSnip);
