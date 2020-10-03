# INTEL CONFIDENTIAL
#
# Copyright © 2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

# Adapted from http://www.yarp.it/yarp_cmake_hello.html

cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)

# find YARP
find_package(YARP REQUIRED)

# set up our program
add_library(yarp_host_snip SHARED yarp_host_snip.cc)

# add nxsdkhost.h include
target_include_directories(yarp_host_snip
    PUBLIC ${PROJECT_SOURCE_DIR}/build/includes/nxsdk/include
)

# link with YARP libraries
target_link_libraries(yarp_host_snip ${YARP_LIBRARIES})
