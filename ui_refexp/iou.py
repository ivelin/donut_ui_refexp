
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Modifed version from the following original on stackoverflow:
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    bb1 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x, y) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # print(f"IoU input bb1, bb2: {bb1} , {bb2}")
    # if predictions are not resulting in properly shaped bounding boxes, return no-match
    if bb1['xmin'] >= bb1['xmax']:
        return 0
    if bb1['ymin'] >= bb1['ymax']:
        return 0

    # if any of the bounding box labels are not properly shaped, return no-match
    if bb2['xmin'] >= bb2['xmax']:
        return 0
    if bb2['ymin'] >= bb2['ymax']:
        return 0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    # print(f"IoU x_left: {x_left}, y_top: {y_top}, x_right: {x_right}, y_bottom: {y_bottom}")

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # print(f"IoU intersection_area: {intersection_area}")

    # compute the area of both AABBs
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])
    # print(f"IoU bb1_area: {bb1_area}")
    # print(f"IoU bb2_area: {bb2_area}")

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # if iou > 0:
    #   print(f"IoU input bb1, bb2: {bb1} , {bb2}")
    #   print(f"IoU : {iou}")
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
