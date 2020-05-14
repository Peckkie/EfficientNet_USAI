# ขั้นตอนการสร้าง Ubuntu
## 1. ลง miniconda (Link : https://www.tomordonez.com/install-miniconda-linux.html)
- Create the enviroment (link : https://janakiev.com/blog/jupyter-virtual-envs/) -Python 3.6.9
## 2. ลง jupyter notebook on enviroment by thise code $conda install jupyter

- $conda install numpy == 1.18.4

- $conda install pandas

- Install scikit-image(0.16.2)
     
     $pip install -U scikit-image
     
     $pip install --upgrade scikit-image(version 0.16.2)

- Install tensorflow 
     
     $pip install --upgrade tensorflow(version 2.2.0)
     
- Install keras_efficientnets
     
     $pip install keras_efficientnets(version 2.3.1)
     
## 3. ลบ รูปที่ใช้ไม่ได้ของจาก dataset cat&dog on pwd Linux
     
     $ls-laS 
     
     Dog image  11702.jpg ใช้ไม่ได้ (rm 11702.jpg)
    
## 4. กรณีมี GPU support (Link : https://www.tensorflow.org/install/gpu)
     $pip install tensorflow-gpu==2.2.0
     
### Install CUDA with apt
     
     # Add NVIDIA package repositories
     
     $wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
     
     $sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
     
     $sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
     
     $sudo apt-get update
     
     $wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
     
     $sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    
     $sudo apt-get update

     # Install NVIDIA driver
    
     $sudo apt-get install --no-install-recommends nvidia-driver-430
     
     # Reboot. Check that GPUs are visible using the command: nvidia-smi

     # Install development and runtime libraries (~4GB)
     
     $sudo apt-get install --no-install-recommends \
      cuda-10-1 \
      libcudnn7=7.6.4.38-1+cuda10.1  \
      libcudnn7-dev=7.6.4.38-1+cuda10.1

     # Install TensorRT. Requires that libcudnn7 is installed above.
     
     $sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
      libnvinfer-dev=6.0.1-1+cuda10.1 \
      libnvinfer-plugin6=6.0.1-1+cuda10.1


****test tansorflow GPU
    
    
# Kernel Dead Problem 

## Solving by Uninstall ipykernel and install jupyter agian
   
   $conda remove jpykernel
   
   $conda clean -tipsy
   
   $conda install jupyter

    
    
    
   
