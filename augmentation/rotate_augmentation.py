import cv2
import numpy as np

def rotate_image(mat, angle, points):
    'Rotates the image matrix and cue points by a given angle'
    #print(angle)
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    no_abs_cos = rotation_mat[0,0] 
    no_abs_sin = rotation_mat[0,1]

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
     
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    # updates cue points to a new state
    new_points = []
    for number in range(0, len(points)):
        if number % 2 == 0:
            x = (points[number] - width/2) * no_abs_cos + (points[number+1] - height/2) * no_abs_sin + bound_w/2
            new_points.append(int(round(x)))
        else:
            y = (points[number] - height/2) * no_abs_cos - (points[number-1] - width/2) * no_abs_sin + bound_h/2
            new_points.append(int(round(y)))
            
    border_w = (bound_h - width)/2
    border_h = (bound_w - height)/2
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    # cut image matrix to the previous shape
    cut_rotated_mat = rotated_mat[int(border_w):bound_w - int(round(border_w)):, int(border_h):bound_h - int(round(border_h)):]
    if height > cut_rotated_mat.shape[0]:
        cut_rotated_mat = rotated_mat[int(border_w):bound_w - int(round(border_w)-1):, int(border_h):bound_h - int(round(border_h)-1):]
    elif height < cut_rotated_mat.shape[0]:
        cut_rotated_mat = rotated_mat[int(border_w):bound_w - int(round(border_w)+1):, int(border_h):bound_h - int(round(border_h) + 1):]

    # updates cue points to a previous shape
    for number in range(0, len(new_points)):
        if number % 2 == 0:
            new_points[number] = new_points[number] - int(border_w)
        else:
            new_points[number] = new_points[number] - int(border_h)
    
    return cut_rotated_mat, new_points


if __name__ == "__main__":
    pass