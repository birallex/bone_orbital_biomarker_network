import numpy as np
import random

def noise(source_matrix, noise_level):
    result = np.copy(source_matrix)
    for x in range(source_matrix.shape[0]):
        for y in range(source_matrix.shape[1]):
            if result[x][y].any() != 0:
                noise = random.randint(-noise_level, noise_level)
                result[x][y] += noise
    result = np.clip(result, 0, 255) 
    return result


if __name__ == "__main__":
    pass