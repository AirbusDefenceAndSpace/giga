#Setup.sh for the Giga API Project
set -e # Stop the script if an error occurs

echo "--- 1. System Preparation ---"
sudo apt update

sudo apt install -y git cmake build-essential g++

echo "--- Switch to the right folder ---"
cd ..

echo "--- 3. Compile GIGA API (Stub) ---"
cd giga
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..

echo "--- 4. Compile GIGA Soft Backend (CPU) ---"
cd giga_soft
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Tell the system where the new libraries are located
echo "--- 5. Update Library Cache ---"
sudo ldconfig

echo "--- DONE! GIGA API dependencies installed ---"