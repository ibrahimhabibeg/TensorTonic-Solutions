import numpy as np

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    image = np.array(image)
    H, W = image.shape
    angle_radian = np.radians(angle_degrees)
    dx = np.repeat(np.expand_dims(np.arange(W) - (W-1)/2, axis=0), H, axis=0) # (H, W)
    dy = np.repeat(np.expand_dims(np.arange(H) - (H-1)/2, axis=1), W, axis=1) # (H, W)
    src_x = np.round((W-1)/2 + dx*np.cos(angle_radian) - dy*np.sin(angle_radian)).astype(int)
    src_y = np.round((H-1)/2 + dx*np.sin(angle_radian) + dy*np.cos(angle_radian)).astype(int)
    rotated_image = np.where((src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H), image[src_y, src_x], 0.0)
    return rotated_image.tolist()
