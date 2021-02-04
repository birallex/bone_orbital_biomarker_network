import numpy as np

def brightness(source_matrix, brightness):
    result = np.copy(source_matrix)
    result *= brightness
    result = np.clip(result, 0, 255) 
    return result

if __name__ == "__main__":
    pass