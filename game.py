import cv2
import numpy as np
import time
import game_detection as gd
import pytesseract
import pyautogui

from station import Station
from point import Point
from typing import List



# yellow, red, blue
color_ranges = [((20, 150, 220), (30, 255, 255)), ((0, 185, 200), (10, 200, 230)), ((105, 185, 130), (125, 200, 150))]


def move_to_center(location, size):
    if location and size:
        center_x = location[0] + size[0] // 2
        center_y = location[1] + size[1] // 2
        pyautogui.moveTo(center_x, center_y)



class MetroGame:

    def __init__(self):
        self.reset()

    def reset(self):

        loc, size = gd.locate_on_screen('screenshots/restart.png')
        if loc:
            gd.move_to_center(loc, size)
            pyautogui.click(button='left')
        time.sleep(2)

        self.delivered_passengers = 0
        self.waiting_passengers = 0

        time.sleep(3)
        self.water = gd.identify_water()
        self.stations = gd.identify_stations(min_size=1000)

        # Add identification later
        self.num_lines = 3
        self.lines = []
        self.trains = 3
        self.train_cars = 0
        self.tunnels = 3

    def update(self):
        self.stations = gd.identify_stations(min_size=500)
        self.delivered_passengers = gd.get_delivered_passengers()
        self.waiting_passengers = len(gd.detect_shapes(region=(0, 0, 1512, 982), min_area=10)) - len(self.stations)
        
        reward = self.delivered_passengers - self.waiting_passengers
        done = gd.is_done()

        return reward, done
    
    def create_line(self, stations: List[Station]):
        pos = stations[0].position
        pyautogui.moveTo(pos.x, pos.y)
        pyautogui.mouseDown()
        for station in stations[1:]:
            pos = station.position
            pyautogui.dragTo(pos.x, pos.y, 
                            duration=0.3,
                            button='left',
                            mouseDownUp=False)
        pyautogui.mouseUp()
        self.lines.append(stations)
        if self.trains > 0:
            self.trains -= 1


    def extend_line(self, line_idx: int, from_station: Station, to_station: Station):

        end_tab = gd.find_tab(line_idx, from_station)

        pyautogui.moveTo(end_tab.x, end_tab.y)
        move_to = to_station.position
        pyautogui.dragTo(move_to.x, move_to.y, button='left', duration=0.3)
        line = self.lines[line_idx]
        if line[0] == from_station:
            line = [to_station] + line
        else:
            line.append(to_station)


    def add_train(self, line):
        train_pos, train_size = gd.locate_on_screen('screenshots/locomotives.png', confidence=0.5)
        move_to_center(train_pos, train_size)
        move_to = line[0].position
        for station in line:
            if len(station.lines) == 1:
                move_to = station.position
                break
        pyautogui.dragTo(move_to.x, move_to.y, duration=0.5, button='left')
        if self.trains > 0:
            self.trains -= 1


    def add_train_car(self, line):
        pass


    def connect(self, point_1, point_2):
        (x1, y1), (x2, y2) = point_1, point_2
        pyautogui.moveTo(x1, y1)
        pyautogui.dragTo(x2, y2, duration=0.5, button='left')


def troubleshoot(image, contours=None):
    if contours:
        for contour, (x, y) in contours:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)



if __name__ == '__main__':
    game = MetroGame()
    game.reset()
    while True:
        time.sleep(5)
        if gd.get_white_area() < 0.5:
            print('Game not detected')
            continue
        game.update()
        print(game.delivered_passengers)
        print(game.waiting_passengers)
        print(game.stations[1])
        print(gd.get_nearest_neighbors(game.stations[1], game.stations))
