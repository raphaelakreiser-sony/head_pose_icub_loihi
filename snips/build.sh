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

#!/usr/bin/env bash

# Get the directory this script resides in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# nxsdk should be availabe in python path
nxsdk_path=`python3 -c "import nxsdk; print(nxsdk.__path__)"`

pushd $DIR

# Wipe out and re-create the build directory
rm -rf build
mkdir build
cd build

# Copy headers (nxsdkhost.h)
mkdir -p includes/nxsdk
cp -v -R ${nxsdk_path:2:-2}/include includes/nxsdk

# Run CMake/Make
echo "Building Shared Library.............."

cmake ..
make

echo "Shared Library Built................."

popd
