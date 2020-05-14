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
    
    
****test tansorflow GPU
    
    
# Kernel Dead Problem 

## Solving by Uninstall jpykernel and install jupyter agian
   
   $conda remove jpykernel
   
   $conda clean -tipsy
   
   $conda install jupyter

    
    
    
   
