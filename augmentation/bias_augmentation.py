import numpy as np
import cv2

def image_bias(matrix, x_bias, y_bias, points):
    # updates cue points to a new state
    new_points = []
    for number in range(0, len(points)):
        if number % 2 == 0:
            x = points[number] + x_bias
            new_points.append(int(x))
        else:
            y = points[number] + y_bias
            new_points.append(int(y))

    num_rows, num_cols = matrix.shape[:2]
    translation_matrix = np.float32([[1, 0, x_bias], [0, 1, y_bias]])
    img_translation = cv2.warpAffine(matrix, translation_matrix, (num_cols, num_rows))

    return img_translation, new_points


if __name__ == "__main__":
    pass