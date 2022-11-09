import numpy as np
from dataset import *
from multiprocessing import Process
import multiprocessing

seed = 0
np.random.seed(seed)

n_unique_ellipse = 20
n_two_circles    = 20
n_three_circles  = 20
n_trunc_once     = 20
n_trunc_twice    = 20
ratio_cartesian  = 0.5
n_neighbours     = 10
max_radius       = 10


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
suff = 'test'
processes = []


if __name__ == '__main__':
    print("You have ",multiprocessing.cpu_count(), " CPUs.")

    cartesian = True 
    processes.append(Process(target=EllipseDataset, args = (root, n_uni_ellipse_cartesian,    max_radius, n_neighbours, cartesian, 0, False, 'Unique_Ellipse_cartesian'+suff+'.dt' ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_1extra_circle_cartesian,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_cartesian'+suff+'.dt'    ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_2extra_circles_cartesian, max_radius, n_neighbours, cartesian, 2, False, 'Three_Circles_cartesian'+suff+'.dt'  ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_trunc_once_cartesian,     max_radius, n_neighbours, cartesian, 1, True,  'Truncated_Once_cartesian'+suff+'.dt' ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_trunc_twice_cartesian ,   max_radius, n_neighbours, cartesian, 2, True,  'Truncated_Twice_cartesian'+suff+'.dt',)))

    cartesian = False
    processes.append(Process(target=EllipseDataset, args = (root, n_uni_ellipse_polar,    max_radius, n_neighbours, cartesian, 0, False, 'Unique_Ellipse_polar'+suff+'.dt' ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_1extra_circle_polar,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_polar'+suff+'.dt'    ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_2extra_circles_polar, max_radius, n_neighbours, cartesian, 2, False, 'Three_Circles_polar'+suff+'.dt'  ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_trunc_once_polar,     max_radius, n_neighbours, cartesian, 1, True,  'Truncated_Once_polar'+suff+'.dt' ,)))
    processes.append(Process(target=EllipseDataset, args = (root, n_trunc_twice_polar ,   max_radius, n_neighbours, cartesian, 2, True,  'Truncated_Twice_polar'+suff+'.dt',)))

    # Starting the processes
    for proc in processes:
        proc.start()

    # Complete the processes
    for proc in processes:
        proc.join()


cartesian = True

cartesian = True 
'''EllipseDataset(root, n_uni_ellipse_cartesian,    max_radius, n_neighbours, cartesian, 0, False, 'Unique_Ellipse_cartesian'+suff+'.dt' )
EllipseDataset(root, n_1extra_circle_cartesian,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_cartesian'+suff+'.dt'    )
EllipseDataset(root, n_2extra_circles_cartesian, max_radius, n_neighbours, cartesian, 2, False, 'Three_Circles_cartesian'+suff+'.dt'  )
EllipseDataset(root, n_trunc_once_cartesian,     max_radius, n_neighbours, cartesian, 1, True,  'Truncated_Once_cartesian'+suff+'.dt' )
EllipseDataset(root, n_trunc_twice_cartesian ,   max_radius, n_neighbours, cartesian, 2, True,  'Truncated_Twice_cartesian'+suff+'.dt')

cartesian = False
EllipseDataset(root, n_uni_ellipse_polar,    max_radius, n_neighbours, cartesian, 0, False, 'Unique_Ellipse_polar'+suff+'.dt' )
EllipseDataset(root, n_1extra_circle_polar,  max_radius, n_neighbours, cartesian, 1, False, 'Two_Circles_polar'+suff+'.dt'    )
EllipseDataset(root, n_2extra_circles_polar, max_radius, n_neighbours, cartesian, 2, False, 'Three_Circles_polar'+suff+'.dt'  )
EllipseDataset(root, n_trunc_once_polar,     max_radius, n_neighbours, cartesian, 1, True,  'Truncated_Once_polar'+suff+'.dt' )
EllipseDataset(root, n_trunc_twice_polar ,   max_radius, n_neighbours, cartesian, 2, True,  'Truncated_Twice_polar'+suff+'.dt') '''