from typing import List
import pyautogui
import cv2
import numpy as np

from station import Station
from point import Point
import time

from PIL import Image
import pytesseract


blue_color_range = ((95, 20, 220), (110, 100, 255)) # Adjust these values
brown_color_range = ((0, 50, 50), (35, 100, 100))  # Adjust these values
white_color_range = ((0, 0, 180), (180, 20, 255))
yellow_color_range = ((20, 150, 220), (30, 255, 255))
red_color_range = ((0, 185, 200), (10, 200, 230))
blue_color_range = ((105, 185, 130), (125, 200, 150))


# Function to classify shapes based on their contours
def classify_shape(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.90 and ar <= 1.10 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    
    return shape

def create_mask(image, color_range):
    (lower, upper) = color_range
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


def get_new_stations(all_stations, old_stations):
    new_stations = []
    for station_1 in all_stations:
        new_station = True
        x1, y1 = station_1.position
        for station_2 in old_stations:
            x2, y2 = station_2.position
            if abs(x1 - x2) < 100 and abs(y1 - y2) < 100:
                new_station = False
        if new_station:
            new_stations.append(station_1)
    return new_stations


def get_nearest_neighbor(station, all_stations):
    nearest = None
    lowest_dist = float('inf')
    for other_station in all_stations:
        if other_station == station:
            continue
        dist = station.get_distance(other_station)
        if dist < lowest_dist:
            lowest_dist = dist
            nearest = other_station
    return nearest


def create_exclusion_mask(image, exclusion_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in exclusion_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        full_mask = cv2.bitwise_or(full_mask, mask)
    
    # Invert the mask to include only non-excluded colors
    exclusion_mask = cv2.bitwise_not(full_mask)
    return exclusion_mask


# Function to capture a region of the screen
def capture_region(region=None):
    # screenshot_path = "/Users/masonking/CompSci/Mini-Metro/screenshots/region_screenshot.png"
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot


# Find a template image on the screen
def locate_on_screen(template_path, region=None, confidence=0.8):
    screenshot = capture_region(region)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    max_loc = (max_loc[0] // 2, max_loc[1]//2)
    size = (template.shape[1] // 2, template.shape[0] // 2)
    if max_val >= confidence:
        return max_loc, size  # Return top-left corner and size (width, height)
    # print('Location not found:', template_path)
    return None, None


def is_done():
    if locate_on_screen('screenshots/game_over.png')[0]:
        return True
    else:
        return False


def get_delivered_passengers():
    region = (1235, 60, 55, 45)
    screenshot = pyautogui.screenshot(region=region)
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    text = pytesseract.image_to_string(screenshot, config='--psm 6').strip()
    return int(text) if text.isdigit() else 0


def get_white_area():
    # Returns a percentage of how much white is on the screen
    region = (0, 0, 1512, 982)
    screenshot = capture_region(region)
    white_mask = create_mask(screenshot, white_color_range)
    total_white_area = np.sum(white_mask == 255)
    total_area = np.sum(white_mask > -1)

    return total_white_area / total_area


def find_contours(image, min_area=100):
    # Find contours in the masked image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_locations = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:  # Ignore small contours
            continue
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
        else:
            cX, cY = 1512, 982
        contour_locations.append((contour, Point(cX, cY)))
    
    return contour_locations


# Function to detect and classify shapes in an image
def detect_shapes(region, min_area=100):
    screenshot = capture_region(region)
    brown_mask = create_mask(screenshot, brown_color_range)
    contour_locations = find_contours(brown_mask, min_area=min_area)
    shapes = []
    for contour, pos in contour_locations:
        shape = classify_shape(contour)
        shapes.append((shape, pos, contour))
    return shapes


def identify_stations(min_size = 2000):
    region = (0, 0, 1512, 982)
    shapes = detect_shapes(region, min_area=min_size)
    stations = []
    for shape, (x, y), contour in shapes:
        if shape.upper() == "RECTANGLE":
            shape = "SQUARE"
        stations.append(Station(shape, Point(x, y), contour))
    return stations


def identify_water():
    region = (0, 0, 1512, 982)
    screenshot = capture_region(region)
    blue_mask = create_mask(screenshot, blue_color_range)
    contour_locations = find_contours(blue_mask)

    water_zones = []

    for contour, _ in contour_locations:
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        X = topmost[0]
        Y = topmost[1]
        water_zones.append(("water", (X, Y - 10), contour))
    
    return water_zones


def get_nearest_neighbors(station, all_stations):
    """Finds the nearest stations by finding the closest
    distance and returning all stations within a certain
    range beyond that distance"""
    lowest_dist = float('inf')
    for other_station in all_stations:
        if other_station == station:
            continue
        lowest_dist = min(station.get_distance(other_station), lowest_dist)
    neighbors = []
    for other_station in all_stations:
        if other_station == station:
            continue
        if station.get_distance(other_station) < lowest_dist * 1.5:
            neighbors.append(other_station)
    return neighbors


def t_shaped_contours(contours, max_area=2000):
    t_shaped = []
    for contour, loc in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        print(cv2.contourArea(contour))
        if len(approx) >= 8 and cv2.contourArea(contour) < max_area:
            t_shaped.append((contour, loc))
    return t_shaped


def find_tab(line_num: int, station: Station | Point):
    
    # yellow, red, blue
    color_ranges = [((20, 150, 220), (30, 255, 255)), ((0, 185, 200), (10, 200, 230)), ((105, 185, 130), (125, 200, 150))]
    region = (0, 0, 1512, 900)

    screenshot = capture_region(region)
    mask = create_mask(screenshot, color_ranges[line_num])
    contours = find_contours(mask)
    end_tabs = t_shaped_contours(contours)

    closest = None
    closest_dist = float('inf')
    for tab, pos in end_tabs:
        dist = station.get_distance(pos)
        if dist < closest_dist:
            closest = pos
            closest_dist = dist
            
    if not closest:
        print(f"Line {line_num} end tabs not found")
        troubleshoot(mask)
        return
    
    return closest


def get_info():
    region = (450, 900, 690, 70)
    screenshot = capture_region(region)
    return screenshot


def troubleshoot(image, contours=None):
    if contours:
        for contour, (x, y) in contours:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


# Example usage
if __name__ == "__main__":
    
    time.sleep(4)

    region = (0, 0, 1512, 982)
    screenshot = capture_region(region)

    info = get_info()

    ### FEATURE DETECTION ###

    waters = identify_water()

    # Detect and classify shapes in the captured region
    shapes = detect_shapes(region, min_area=50)
    
    # Display the result for debugging
    features = waters + shapes

    hsv = cv2.cvtColor(info, cv2.COLOR_BGR2HSV)
    cv2.imshow("Detected Shapes", hsv)
    cv2.waitKey(0)

    mask = create_exclusion_mask(info, [blue_color_range, brown_color_range, white_color_range])

    cv2.imshow("Detected Shapes", mask)
    cv2.waitKey(0)

    for contour, _ in find_contours(mask, min_area=0):
        cv2.drawContours(info, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("Detected Shapes", info)
    cv2.waitKey(0)

    for feature, (x, y), contour in features:
        cv2.drawContours(screenshot, [contour], -1, (0, 255, 0), 2)
        cv2.putText(screenshot, feature, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    cv2.imshow("Detected Shapes", screenshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


