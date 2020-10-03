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

#include "embedded_snip.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>

// SIM_TIME and neuron indices are overwritten using loihi_tools

#define SIM_TIME 100000
#define CORE 0

// input // pairs of core and neuron
uint16_t input_con[][2] = {{0, 6}, {0, 7}, {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}};

// output con (replaced by python script! Do not delete the following line line)
// replace here
uint16_t output_con_vision[] = {211};
uint16_t output_con_RHD2[] = {250, 219, 262, 425, 274, 280, 286, 292, 298, 304, 310, 430, 432, 325, 216, 222, 429, 171, 355, 212, 214, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 427, 162, 163, 164, 165, 166, 167, 168, 169, 170, 428, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210};
uint16_t output_con_RHD1[] = {225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 161, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 213, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 215, 275, 276, 277, 278, 279, 217, 281, 282, 283, 284, 285, 218, 287, 288, 289, 290, 291, 220, 293, 294, 295, 296, 297, 221, 299, 300, 301, 302, 303, 223, 305, 306, 307, 308, 309, 224, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324};
uint16_t output_con_HD2[] = {431, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 426, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424};
uint16_t output_con_HD1[] = {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131};

// buffer is used to read out spikes
uint32_t buffer[20];


// Channel Id of the output channel
uint16_t outChannelId = -1;
// Channel Id of the input channel
uint16_t inChannelId = -1;

// send spikes to logical core- & axon indices
inline void send_spike(uint32_t time, uint16_t idx[2]){
    nx_send_discrete_spike(time, nx_nth_coreid(idx[0]), 1 << 14 | (idx[1] & 0x00003fff));
}

int doSpiking(runState *s) {
  if (s->time_step == 1) {
   outChannelId = getChannelID("out");
    inChannelId = getChannelID("in");
  }
  return 1;
}

void runSpiking(runState *s) {
  
  // Id of the cx in which spike needs to be inserted
  int cxId = -1;
  // How many neurons should be stimulated
  int num_neurons_read = -1;
  // Flag variable to indicate if its last time step
  uint32_t sentinel = 0;

  // ///////////////////////
  // WRITING TO LOIHI FROM EXTERNAL
  // ///////////////////////

  // Read the cxId from the input channel
  readChannel(inChannelId, &num_neurons_read, 1);
  // printf("num neurons read %d \n", num_neurons_read);

  // Then send spikes using this information to the right neurons of the input port
  while (num_neurons_read > 0) {

      //printf("num neurons read %d \n", num_neurons_read);
      readChannel(inChannelId, &cxId, 1);
      uint16_t axon_cmd = cxId;
      send_spike(s->time_step , input_con[axon_cmd]);
      num_neurons_read --;
    }

  // ///////////////////////
  // WRITING TO EXTERNAL FROM LOIHI
  // ///////////////////////

  // Reading out the id of neurons that spikes and writing them in the buffer
  // Then resetting numSpikes to 0
  uint32_t numSpikes = 0;
  int k0 = (s->time_step - 1) & 3;
  for(uint16_t *p=output_con_HD1; *p; p++) {
	if (SPIKE_COUNT[k0][*p]) {
	   buffer[numSpikes++] = *p;
       SPIKE_COUNT[k0][*p] = 0;
    }}
  for(uint16_t *p=output_con_HD2;  *p; p++) {
    if (SPIKE_COUNT[k0][*p]) {
	   buffer[numSpikes++] = *p;
       SPIKE_COUNT[k0][*p] = 0;
    }}
  for(uint16_t *p=output_con_RHD1;  *p; p++) {
	if (SPIKE_COUNT[k0][*p]) {
	   buffer[numSpikes++] =*p;
       SPIKE_COUNT[k0][*p] = 0;
    }}
  for(uint16_t *p=output_con_RHD2;  *p; p++) {
    if (SPIKE_COUNT[k0][*p]) {
       buffer[numSpikes++] = *p;
       SPIKE_COUNT[k0][*p] = 0;
    }}
  for(uint16_t *p=output_con_vision;  *p; p++) {
	if (SPIKE_COUNT[k0][*p]) {
	   buffer[numSpikes++] = *p;
       SPIKE_COUNT[k0][*p] = 0;
    }}

  // First, determine how many spikes we read out
  // Then, send those spikes (stored in buffer) to the host
  writeChannel(outChannelId, &numSpikes, 1);
  writeChannel(outChannelId, buffer, numSpikes);

  // Checking if the sentinel needs to be changed (if simulation time is over)
  // sentinel is being used to tell concurrent host snip to quit
  if (s->time_step == SIM_TIME) {
    sentinel = -1;
  }
  // Writing the value of sentinel
  writeChannel(outChannelId, &sentinel, 1);

}
