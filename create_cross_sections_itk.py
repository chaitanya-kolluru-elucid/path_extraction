import SimpleITK as sitk
import numpy as np
import json
import os, sys
import glob
from tqdm import tqdm
import itk

def levelset_reinit(levelset):

    reinitFilter = itk.ReinitializeLevelSetImageFilter.New(levelset)
    reinitFilter.Update()
    retval = reinitFilter.GetOutput()
    retval.DisconnectPipeline()
    return retval

def create_resample_image(original_image, position, xaxis, yaxis, target_spacing_mm, target_size_mm, default_pixel_value, image_data_type = itk.SS):

    zaxis = np.cross(xaxis, yaxis)

    # Normalize the axis vectors
    xaxis = xaxis / np.linalg.norm(xaxis)
    yaxis = yaxis / np.linalg.norm(yaxis)
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Setup direction cosines array
    direction_array = np.array([[xaxis[0], xaxis[1], xaxis[2]],
                               [yaxis[0], yaxis[1], yaxis[2]],
                               [zaxis[0], zaxis[1], zaxis[2]]])
    
    # Invert the rotation matrix
    inverted_direction_array = np.transpose(direction_array)

    # Get origin of the sampling reference grid
    origin = [position[0] - ((target_size_mm/2) * xaxis[0]) - ((target_size_mm/2) *yaxis[0]),
              position[1] - ((target_size_mm/2) * xaxis[1]) - ((target_size_mm/2) * yaxis[1]), 
              position[2] - ((target_size_mm/2) * xaxis[2]) - ((target_size_mm/2) * yaxis[2])]    

    # Define pixel types and image dimension
    Dimension = 3
    InputPixelType = image_data_type
    OutputPixelType = image_data_type

    # Define input and output image types
    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Instantiate resample filter
    filter = itk.ResampleImageFilter[InputImageType, OutputImageType].New()

    # Instantiate transform
    TransformType = itk.AffineTransform[itk.D, Dimension]
    transform = TransformType.New()
    filter.SetTransform(transform)

    # Instantiate interpolator
    InterpolatorType = itk.LinearInterpolateImageFunction[InputImageType, itk.D]
    interpolator = InterpolatorType.New()
    filter.SetInterpolator(interpolator)

    # Set default pixel value
    filter.SetDefaultPixelValue(default_pixel_value)

    # Set output spacing
    filter.SetOutputSpacing(target_spacing_mm)

    # Set output origin
    filter.SetOutputOrigin(origin)

    # Set output direction
    filter.SetOutputDirection(itk.matrix_from_array(inverted_direction_array))

    # Set output size
    size = itk.Size[Dimension]()
    size[0] = int(target_size_mm/target_spacing_mm[0])
    size[1] = int(target_size_mm/target_spacing_mm[1])
    size[2] = 1

    filter.SetSize(size)

    # Connect the pipeline
    filter.SetInput(original_image)
    filter.Update()

    return filter.GetOutput() 

def parse_path_for_positions_axes(segment, positions, xaxes, yaxes):
    
    cross_sections = segment["cross_sections"]

    for k in range(len(cross_sections)):

        position = cross_sections[k]["position"]
        positions.append(position)

        xaxis = cross_sections[k]["xaxis"]
        xaxes.append(xaxis)

        yaxis = cross_sections[k]["yaxis"]
        yaxes.append(yaxis)

    for segment in segment["distal_segments"]:
        parse_path_for_positions_axes(segment, positions, xaxes, yaxes)


def resample_and_savecross_sections(case_name, path_data, cta_image, lumen_dist_map = None, target_spacing_mm = tuple([0.1, 0.1, 0.1]),
                                    target_size_mm = 12, cs_image_save_dir = './cross_section_images', cs_label_save_dir = './cross_section_labels'):

    # Counter to keep track of cross-sections for filename save
    cs_counter = 0

    # Create case specific dirs in the save folders
    if not os.path.exists(os.path.join(cs_image_save_dir, case_name)):
        os.makedirs(os.path.join(cs_image_save_dir, case_name))    
    
    if not os.path.exists(os.path.join(cs_label_save_dir, case_name)):
        os.makedirs(os.path.join(cs_label_save_dir, case_name))

    # Get the root segment in path
    root = path_data['root_segment']
    positions = []
    xaxes = []
    yaxes = []

    # Get values for all segments, every cross section
    parse_path_for_positions_axes(root, positions, xaxes, yaxes)
    
    # Go to each position, resample and save
    for k in tqdm(range(len(positions))):

        position = positions[k]
        xaxis = xaxes[k]
        yaxis = yaxes[k]

        resample_cta_image = create_resample_image(cta_image, position, xaxis, yaxis, target_spacing_mm, target_size_mm, 0, itk.SS)
        itk.imwrite(resample_cta_image, os.path.join(cs_image_save_dir, case_name,  str(k).zfill(5) + '.tiff'))

        if lumen_dist_map is not None:
            
            resample_lumen_image = create_resample_image(lumen_dist_map, position, xaxis, yaxis, target_spacing_mm, target_size_mm, 100, itk.F)            
            resample_lumen_image = levelset_reinit(resample_lumen_image)
            itk.imwrite(resample_lumen_image, os.path.join(cs_label_save_dir, case_name, str(k).zfill(5) + '.tiff'))

        cs_counter = cs_counter + 1

if __name__ == '__main__':

    # Set the pixel spacing and size of the cross-sections
    target_spacing_mm = tuple([0.1, 0.1, 0.1])
    target_physical_size_mm = 12

    # Get all the path jsons we want to analyze
    path_json_list = glob.glob(r'./paths/*.json')

    # Loop through each case
    for path_json in path_json_list:

        case_name = os.path.basename(path_json).split('.json')[0]

        print('Processing case ' + case_name)

        with open(path_json, 'rb') as f:
            path_data = json.load(f)

        cta_image = itk.imread(os.path.join('./images/', case_name + '.nii.gz'))
        lumen_distance_map = itk.imread(os.path.join('./labels/', case_name + '.nrrd'))

        resample_and_savecross_sections(case_name, path_data, cta_image, lumen_distance_map)
