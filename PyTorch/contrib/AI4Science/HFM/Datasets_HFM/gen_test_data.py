# Adapted to tecorigin hardware
import numpy as np


def gen_test_data(save_path, dt=0.001,dx=0.01,dy=0.001,xmax=3000,ymax=600,z=3.01):
    all_data = []
    xx = np.arange(0.01, 29.92,dt)
    yy = np.arange(0.01, 5.92, dy)
    for x in xx:
        for y in yy:
            data = [0, x,y,z,0,0,0,0,0]
            all_data.append(data)
    all_data = np.array(all_data)
    np.save(save_path, all_data)
    print(f'save shape:{all_data.shape},{save_path}')
    
#176768100

if __name__ == '__main__':
    save_path = '/home/hpc/cfd/Datasets_HFM/gen_data_predict_t1.npy'
    gen_test_data(save_path)