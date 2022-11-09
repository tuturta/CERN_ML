# Import modules
import random
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import scipy.spatial
from multiprocessing import Process


# create 11 equaly space angles between 0 et pi/2
angles = np.linspace(1e-3, np.pi/2-1e-3, 11)

def get_ellipse(density, length, angle = 0, cartesian = True):
    '''Create a quarter 2D ellipse in cartesian coordinates with width 2a and height 2b an a angle (in rad) '''
    a,b = length 
    # Uniform distribution over all the grid
    if cartesian:
        xx = np.linspace(0, 10, round(10*np.sqrt(density)))
        yy = np.linspace(0, 10, round(10*np.sqrt(density)))
        x, y = np.meshgrid(xx,yy)
        x = x.flatten()
        y = y.flatten()
    
    # Polar distribution according to 11 angles between 0 and pi/2 
    else:
        radius = np.linspace(1e-3, 10, round(10*density))
        r, theta = np.meshgrid(radius, angles)
        x = (r*np.cos(theta)).flatten()
        y = (r*np.sin(theta)).flatten()

    
    # Check if points are within the ellipse
    points = np.stack((x,y), axis=1)
    points = points[(x*np.cos(angle) - y*np.sin(angle))**2/a**2 + (y*np.cos(angle) + x*np.sin(angle))**2/b**2 <= 1]
    
    # Compute the area 
    area = get_ellipse_area(a,b)

    return points, area


def get_multiple_circles(density, radius_main, center_extra1, radius_extra1, center_extra2 = (0,0), radius_extra2 = 0, cartesian = True, truncated = False):
    
    R      = radius_main
    x1, y1 = center_extra1
    r1     = max(radius_extra1, radius_extra2)
    x2, y2 = center_extra2
    r2     = min(radius_extra1, radius_extra2)
    assert np.sqrt((x1-x2)**2 + (y1-y2)**2) >= r1 + r2, "The extra circles are overlapping!"
    assert (r1 <= 0.5*R) and (r2 <= 0.5*R) , "The extra circles radius have to be smaller radius than half of the main circle radius."

    # Uniform distribution over all the grid
    if cartesian:
        xx = np.linspace(0, 10, round(10*np.sqrt(density)))
        yy = np.linspace(0, 10, round(10*np.sqrt(density)))
        x, y = np.meshgrid(xx,yy)
        x = x.flatten()
        y = y.flatten()
    
    # Polar distribution according to 11 angles between 0 and pi/2 
    else:
        radius = np.linspace(1e-3, 10, 10*round(density))
        r, theta = np.meshgrid(radius, angles)
        x = (r*np.cos(theta)).flatten()
        y = (r*np.sin(theta)).flatten()

    # Create masks 
    mask_main_ellipse = (x**2 + y**2 <= R**2)
    mask_islands      = ((x-x1)**2 + (y-y1)**2 <= r1**2)
    if r2 > 0:
        mask_islands = (mask_islands) | ((x-x2)**2 + (y-y2)**2 <= r2**2)
    if truncated:
        mask = (mask_main_ellipse) & (np.logical_not(mask_islands))
    else:
        mask = (mask_main_ellipse | mask_islands)

    # Check if points are within the mask
    points = np.stack((x,y), axis=1)
    points = points[mask]
    
    # Compute the area of the circles
    main_area    = get_ellipse_area(R,R)
    circle1_area = get_ellipse_area(r1,r1)
    circle2_area = get_ellipse_area(r2,r2)

    # Compute the overlapping areas between the circles
    area_m1      = get_overlapping_circles_area(np.sqrt(x1**2 + y1**2), R, r1)
    if r2>0:
        area_m2  = get_overlapping_circles_area(np.sqrt(x2**2 + y2**2), R, r2)
    else:
        area_m2  = 0
    
    # Remove all the intersections of the 
    if truncated:
        area = main_area - area_m1 - area_m2
    else:
        area = main_area + circle1_area + circle2_area - area_m1 - area_m2

        # if the overlapping area is zero that means that the extra circle is an island 
        # and then its area has to be remove from the total area sum
        if area_m1 == 0:
            area -= circle1_area
        if area_m2 == 0:
            area -= circle2_area
    return points, area


def get_ellipse_area(a,b):
    '''Return the area of an ellipse with width 2a and height 2b'''
    return np.pi*a*b

def get_overlapping_circles_area(d, r1, r2):
    '''Input: 
    d: distance between centers (positive float)
    r1: radius of the main circle (positive float )
    r2: radius of the small circle (positive float)

    Return: the overlapping area of two circles (float)'''
    assert r1 >= r2, "r1 can't be lower than r2"
    assert d > 0,  "the distance between of the two circles has to be stricly positive"

    # if the circles intersect at most up to a point
    if d >= r1 + r2:
        return 0
    
    # if circle 2 is entirely contained in circle 1, for r1>r2
    if r1 >= d + r2:
        return get_ellipse_area(r2,r2)
    
    # for all the other cases
    return r1**2*np.arccos((d**2 + r1**2 - r2**2)/(2*d*r1)) + r2**2*np.arccos((d**2 + r2**2 - r1**2)/(2*d*r2)) - 0.5*np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))


def generate_random_samples(n_samples, max_length, neighbours, cartesian = True, extra_circles = 0, truncated = False):
    '''
    Inputs:
    - n_samples: number of samples to create 
    - max_length: maximum radius of the main circle
    - neighbours: number of neighbours for the edge index function
    - cartesian: Cartesian coordinates if True, polar otherwise.
    - extra_circle: number of extra circle generated. Either to remove or add.
    - truncated: truncate the main circle with extra circles if True. Otherwise, add these extra circles.
    
    Outputs:
    - list of n_samples Data objects
    '''

    # if no samples, stop the function
    if n_samples == 0:
        return []

    data_list = []
    density = np.random.choice([100, 1000, 10000], n_samples)/max_length

    if extra_circles == 0:
        length = np.random.uniform(1, max_length, (n_samples, 2))
        angles = np.random.uniform(0, np.pi, n_samples)
        # Compute a random density rmax/100, rmax/1000, rmax/10000
    
    if extra_circles >= 1:
        radius_main   = np.random.uniform(1,max_length, n_samples)

        # r1 < R/2 to avoid that r1 erase completely the main circle
        radius_extra1 = np.random.uniform(1/density, 0.5*radius_main, n_samples)
        center_extra1 = np.random.uniform((radius_extra1+1/density)[:, np.newaxis], max_length, (n_samples,2))

        # set to zero by default
        radius_extra2 = np.zeros(n_samples)
        center_extra2 = np.zeros((n_samples,2))
    
    if extra_circles == 2:
        radius_extra2 = np.random.uniform(1/density, 0.5*radius_main, n_samples)
        center_extra2 = np.random.uniform((radius_extra2+1/density)[:, np.newaxis], max_length, (n_samples,2))
        
        # To avoid triple intersection between circles
        cond = np.linalg.norm(center_extra1-center_extra2, axis =1) < (radius_extra1 + radius_extra2)
        while cond.any(): 
            center_extra2[cond] += [0.5,0.5]
            radius_extra1 *= 0.5
            radius_extra2 *= 0.5
            cond = np.linalg.norm(center_extra1-center_extra2,axis =1) < (radius_extra1 + radius_extra2)
        
    for i in tqdm(range(n_samples)):  
        if extra_circles == 0:
            points, area = get_ellipse(density[i], length[i], angles[i], cartesian)

            
            # To avoid to have less points that neighbours
            while len(points) < neighbours:
                points, area = get_ellipse(density[i], length[i], angles[i], cartesian)

        else:
            points, area = get_multiple_circles(density[i], radius_main[i], center_extra1[i], radius_extra1[i], center_extra2[i], radius_extra2[i], cartesian, truncated)
            # To avoid to have less points that neighbours
            while len(points) < neighbours:
                points, area = get_multiple_circles(density[i], radius_main[i], center_extra1[i], radius_extra1[i]*0.5, center_extra2[i], radius_extra2[i]*0.5, cartesian, truncated)

        data_list.append(create_graph(points, area, neighbours))
    
    return data_list


def get_edge_index(points, neighbours):
    '''Find the k neirest neighbours for each point using a KD-tree algorithm '''
    tree = scipy.spatial.cKDTree(points)
    _, ii = tree.query(points, neighbours)
    a = np.repeat(np.arange(len(points)),neighbours)
    edge_index = torch.tensor(np.vstack((a, ii.flatten())), dtype = torch.long)
    return edge_index


def create_graph(points, area, neighbours):
    '''Create a Data object with points coordinates and volumes'''

    # Add coordinates as features
    sample = torch.tensor(points, dtype=torch.float64)
    # Add volume as label
    label = torch.tensor([area], dtype=torch.float64)
    # Create edge_index
    edge_index = get_edge_index(points, neighbours)    
    # Create Data object
    graph = Data(x=sample, edge_index=edge_index, y=label)

    return graph


class EllipseDataset(InMemoryDataset):
    '''Definition of our class Dataset to construct datasets of ellipses and points both in radial and cartesian coordinates '''
    def __init__(self, root, n_samples, max_radius, neighbours, cartesian = True, extra_circles = 0, truncated = False, filename = 'dataset.pt', transform=None, pre_transform=None):
        '''
        Inputs:
        - n_samples: number of samples to create 
        - max_length: maximum radius of the main circle
        - neighbours: number of neighbours for the edge index function
        - cartesian: Cartesian coordinates if True, polar otherwise.
        - extra_circle: number of extra circle generated. Either to remove or add.
        - truncated: truncate the main circle with extra circles if True. Otherwise, add these extra circles.
        '''  

        self.n_samples      = n_samples
        self.max_radius     = max_radius
        self.n_neighbours   = neighbours
        self.cartesian      = cartesian
        self.extra_circles  = extra_circles
        self.truncated      = truncated
        self.filename       = filename        
        super(EllipseDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.filename]

    def download(self):
        pass

    def process(self):
        if self.n_samples >0:
            data_list = generate_random_samples(self.n_samples, self.max_radius, self.n_neighbours, self.cartesian, self.extra_circles, self.truncated) 
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])