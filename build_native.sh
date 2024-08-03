module load dealii/9.4.0-gcc-9.3.0-openmpi

cd ~/lxl/pfm-dealii
rm -rf build-native
mkdir build-native
cd build-native
cmake ..
make release
make

