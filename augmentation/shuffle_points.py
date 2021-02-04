import random

def shuffle_points(points):
    new_points = []
    for i in range(len(points)):
        point = points[i] + random.randrange(-3, 3)
        new_points.append(point)
    return new_points

if __name__ == "__main__":
    pass