import numpy as np
from sample_methods import *
from multiprocessing import Process
import multiprocessing


n_samples_per_cat= 50000 # 5000 for testset
n_unique_ellipse = n_samples_per_cat
n_two_circles    = n_samples_per_cat
n_three_circles  = n_samples_per_cat
n_trunc_once     = n_samples_per_cat
n_trunc_twice    = n_samples_per_cat
ratio_cartesian  = 0.0
n_neighbours     = 4
max_radius       = 15


n_uni_ellipse_cartesian    = round(n_unique_ellipse*ratio_cartesian)
n_uni_ellipse_polar        = round(n_unique_ellipse*(1-ratio_cartesian))
n_1extra_circle_cartesian  = round(n_two_circles*ratio_cartesian)
n_1extra_circle_polar      = round(n_two_circles*(1-ratio_cartesian))
n_2extra_circles_cartesian = round(n_three_circles*ratio_cartesian)
n_2extra_circles_polar     = round(n_three_circles*(1-ratio_cartesian))
n_trunc_once_cartesian     = round(n_trunc_once*ratio_cartesian)
n_trunc_once_polar         = round(n_trunc_once*(1-ratio_cartesian))
n_trunc_twice_cartesian    = round(n_trunc_twice*ratio_cartesian)
n_trunc_twice_polar        = round(n_trunc_twice*(1-ratio_cartesian))

root = 'Data'
seed = 4 # 5 for testset
suff = ''
processes = []


if __name__ == '__main__':
    # put the generation of numpy seed in this if statement to avoid restarting the seed for every processes 
    np.random.seed(seed)

    print("You have ",multiprocessing.cpu_count(), " CPUs.")

    cartesian = False
    processes.append(Process(target=generate_random_samples, args = ( n_uni_ellipse_polar,    max_radius, n_neighbours, cartesian, 0, False, ['data'+ suff + f'{idx}.npz' for idx in range(0, n_samples_per_cat)])))
    processes.append(Process(target=generate_random_samples, args = ( n_1extra_circle_polar,  max_radius, n_neighbours, cartesian, 1, False, ['data'+ suff + f'{idx}.npz' for idx in range(n_samples_per_cat, n_samples_per_cat*2)],)))
    processes.append(Process(target=generate_random_samples, args = ( n_2extra_circles_polar, max_radius, n_neighbours, cartesian, 2, False, ['data'+ suff + f'{idx}.npz' for idx in range(n_samples_per_cat*2, n_samples_per_cat*3)],)))
    processes.append(Process(target=generate_random_samples, args = ( n_trunc_once_polar,     max_radius, n_neighbours, cartesian, 1, True,  ['data'+ suff + f'{idx}.npz' for idx in range(n_samples_per_cat*3, n_samples_per_cat*4)],)))
    processes.append(Process(target=generate_random_samples, args = ( n_trunc_twice_polar ,   max_radius, n_neighbours, cartesian, 2, True,  ['data'+ suff + f'{idx}.npz' for idx in range(n_samples_per_cat*4, n_samples_per_cat*5)],)))

    # Starting the processes
    for proc in processes:
        proc.start()

    # Complete the processes
    for proc in processes:
        proc.join()



#generate_random_samples(n_1extra_circle_cartesian,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_cartesian'+suff+'.npy'    )

'''
seed = 0
np.random.seed(seed)
suff = str(seed)

cartesian = True 

generate_random_samples( n_uni_ellipse_cartesian,    max_radius, n_neighbours, cartesian, 0, False, 'Unique_Ellipse_cartesian'+suff+'.npz' )
generate_random_samples( n_1extra_circle_cartesian,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_cartesian'+suff+'.npz'    )
generate_random_samples( n_2extra_circles_cartesian, max_radius, n_neighbours, cartesian, 2, False, 'Three_Circles_cartesian'+suff+'.npz'  )
generate_random_samples( n_trunc_once_cartesian,     max_radius, n_neighbours, cartesian, 1, True,  'Truncated_Once_cartesian'+suff+'.npz' )
generate_random_samples( n_trunc_twice_cartesian ,   max_radius, n_neighbours, cartesian, 2, True,  'Truncated_Twice_cartesian'+suff+'.npz')

cartesian = False
generate_random_samples( n_uni_ellipse_polar,    max_radius, n_neighbours, cartesian, 0, False, 'Unique_Ellipse_polar'+suff+'.npz' )
generate_random_samples( n_1extra_circle_polar,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_polar'+suff+'.npz'    )
generate_random_samples( n_2extra_circles_polar, max_radius, n_neighbours, cartesian, 2, False, 'Three_Circles_polar'+suff+'.npz'  )
generate_random_samples( n_trunc_once_polar,     max_radius, n_neighbours, cartesian, 1, True,  'Truncated_Once_polar'+suff+'.npz' )
generate_random_samples( n_trunc_twice_polar ,   max_radius, n_neighbours, cartesian, 2, True,  'Truncated_Twice_polar'+suff+'.npz') 


cartesian = False
generate_random_samples( n_uni_ellipse_polar,    max_radius, n_neighbours, cartesian, 0, False, filename=[f'data{idx}.npy' for idx in range(0, n_uni_ellipse_polar)])
generate_random_samples( n_1extra_circle_polar,  max_radius, n_neighbours, cartesian, 1, False, filename=[f'data{idx}.npy' for idx in range(n_uni_ellipse_polar, n_uni_ellipse_polar + n_1extra_circle_polar)])
generate_random_samples( n_2extra_circles_polar, max_radius, n_neighbours, cartesian, 2, False, filename=[f'data{idx}.npy' for idx in range(n_uni_ellipse_polar, n_1extra_circle_polar + n_2extra_circle_polar)])
generate_random_samples( n_trunc_once_polar,     max_radius, n_neighbours, cartesian, 1, True,  filename=[f'data{idx}.npy' for idx in range(0, n_uni_ellipse_polar)])
generate_random_samples( n_trunc_twice_polar ,   max_radius, n_neighbours, cartesian, 2, True,  filename=[f'data{idx}.npy' for idx in range(0, n_uni_ellipse_polar)]) 
'''