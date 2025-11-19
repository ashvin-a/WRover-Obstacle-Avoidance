Must have OpenCV version 4.6.0 and NumPy version 1.26.4

Make sure to replace "" with whichever version of ros2 is in use (ex: humble). Install the DepthAI library using the command:
```
sudo apt install ros-[distro]-depthai-ros
```
First, connect to OAK-D W using ROS with the following command:
```
ros2 launch depthai_ros_driver camera.launch.py
```
Then build and run the package.

To run via pipeline, run the following:
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
And then u must install depthai:
to install the dependencies:
```
sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash
```
to install actual package:
```
python3 -m pip install depthai
```
checklist for progress:

Phase 1: basic prototype(vfh algorithm)

pixel location to angle ✔️
find min value of sectors ✔️
Gap detection ✔️
rudementary ground removal without plane detection (no ransac) ✔️
choosing paths and moving to gnss location
Phase 2:

better ground detection using ransac
cost function / danger function for concave rock problem
terrain classifications
and lots and lots of testing :>
