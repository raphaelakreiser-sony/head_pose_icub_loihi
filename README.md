# cc19-headdirnet

Spiking neural network (SNN) for 2D head pose estimation of the iCub robot.
The project was initiated at the Capocaccia Neuromorphic Engineering workshop in 2019 and was published in "Frontiers in Neurosience 2020":
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325709/

The SNN is run on Intel's neuromorphic research chip Loihi and communication between the robot and the Loihi is established over yarp.
The project consists of the following folders:

- yarp_apps: Includes the xml files required for connecting ports in yarp.

- data_rand_mov: Includes one example dataset of a random movement of the iCub's head.

- loihi_net&interface: 
                     - Includes the "main_HDnet.py" which sets up the neural network on Loihi, runs it, and records spike probes.
                     - "HD_net_2rings.py" is called in the main and defines all connections using connectivity matrices defined in "connections.py".
                     - The communication between host and embedded process is established in "main_HDnet.py".
                       "yarp_host_snip.cc" determines the usage of the iCub data and sends signals to the embedded process (which runs the "spiking.c").
                       It also receives spike information from the embedded process ("spiking.c"), stores them and sends them, such that it can be plotted with yarp.
                       The embedded "spiking.c" snips receive signals from the host and send spikes to Loihi accordingly.
                       Here, the spikes are also readout from Loihi and send to the host process.
                      
![alt text](https://github.com/intel-nrc-ecosystem/robotic-demos/blob/master/hd_net_icub/system.png)

### Instructions to run the code

Yarp (https://www.yarp.it) and Event-driven (https://github.com/robotology/event-driven) need to be installed.

Move the yarp applications in the right location:
- cd cc19-headdirnet/yarp_apps/
- Move the .xml files into your /usr/share/yarp/applications folder.

To install all dependencies and build the project: Download the cc19-headdirnet repository and run "cmake" and "make" (in a seperate build folder)

To get started:

- Terminal 1: 
yarpserver --write

- Terminal 2:
yarpdataplayer

         in the yarpdataplayer open:
         --> File --> Open --> cc19-headdirnet/final_frontiers/data_rand_move
         
- Terminal 3:
yarpmanager

         in the yarpmanager open:
         --> Applications --> head_direction_net
         select the following, then click on the "run" button: - hd-visualization
                                                               - vFramerLite
                                                               - yarpview (#7, #10, #11)
         click on the "connect all" button

- Terminal 4:

         cd cc19-headdirnet/final_frontiers/loihi_net&interface/
         sudo rmmod ftdi_sio (if needed)
         source activate py352 (if needed)
         export KAPOHOBAY=1
         python main_HDnet.py


In order to save data:
- in the yarpmanager --> click on the dumper_app, run all and connect all


Final step:
- in the yarpdataplayer --> click on "run"


You should now see the neurons firing in the rasterplot.

Data should be saved in a folder called yarp_dump.
