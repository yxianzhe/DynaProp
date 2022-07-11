echo "Building ROS nodes"

cd Examples/ROS/DynaProp_ROS
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release -DPYTHON_EXUTABLE=/usr/bin/python3
make -j
