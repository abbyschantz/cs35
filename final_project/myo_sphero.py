
import myo as libmyo
import sphero
import cv2
from time import sleep
libmyo.init()
#libmyo.init(os.path.join(os.path.dirname(__file__), 'libs', 'Myo.framework'))
## export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/aschantz/Library/sdk/myo.framework



print("starting up")

s = sphero.Sphero()
coords = [15,15]

class Listener(libmyo.DeviceListener):
    print ("in listener class")

    def on_connect(self, myo, timestamp, firmware_version):
        print("Myo connected")
        try:
            s.connect()
            print("sphero connected")
        except:
            print("err!")
            s.close()

        print( """Bluetooth info:name: %s \nbta: %s """ %
           (s.get_bluetooth_info().name, s.get_bluetooth_info().bta))
        # print("battery level:", s.get_power_state())

    def on_disconnect(self, myo, timestamp):
        print("Myo disconnected")

    # def on_orientation_data(self, myo, timestamp, quat):
        # # print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        # if quat.z > -0.2:
        #     print("!!armed raised (>-0.2)!!, forward")
        #     # s.roll(250, 0)
        # return

    def on_pose(self, myo, timestamp, pose):
        if pose == libmyo.Pose.double_tap:
            self.last_message = "dbltap"
            print("!!DOUBLE TAP!!, shutting down")
            s.stop()
            return False  # Stops the Hub
        if pose == libmyo.Pose.fingers_spread:
            print("!!FINGERS SPREAD!!, forward")
            #print("yaw", libmyo.yaw())
            # s.set_rgb(0,252,80)
            s.roll(80, 0)
            coords[1] += 10
            print("current coords", coords)
            coords
            return
        if pose == libmyo.Pose.wave_out:
            print("!!WAVE OUT!!, right")
            # s.set_heading(90)
            # s.set_rgb(156,109,254)
            s.roll(80, 90)
            coords[0] += 10
            print("current coords", coords)
            # draw([3,3])
            return
        if pose == libmyo.Pose.wave_in:
            print("!!WAVE IN!!, left")
            # s.set_heading(270)
            # s.set_rgb(72,252,252)
            s.roll(80, 270)
            coords[0] -= 10
            print("current coords", coords)
            return
        if pose == libmyo.Pose.fist:
            print("!!FIST!!, roll back")
            # s.set_rgb(255,0,0)
            # s.stop()
            s.roll(80, 180)
            coords[1] -= 10
            print("current coords", coords)
            return   # Stops the Hub

            #another idea is to only use forward roll and use device rotation to turn??
            #also try roll, pitch, yaw for myo to get another variables -- go when arm raised? 
            #s.roll(self, spped, heading, state, response)
                # :param speed: 0-255 value representing 0-max speed of the sphero.
                # :param heading: heading in degrees from 0 to 359.
            #s.set_heading(heading, response): :param heading: heading in degrees from 0 to 359 (motion will be shortest angular distance to heading command)
            #s.set_rgb_led(self, red, green, blue, save, response):
            # :param red: red color value.
            # :param green: green color value.
            # :param blue: blue color value.

# def draw(coordinate):
#     """
#         to demo:  click to bring focus to the messi image
#         move mouse around and hit 'r' (lowercase r)
#         a cyan rectangle should appear at your mouse
#         hit spacebar to clear

#         drawing reference:
#           http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
#     """
#     # Create a black image, a window and bind the function to window
#     # this is from here:

#     # def mouse_handler(event,x,y,flags,param):
#     #     """ a function that gets called on mouse events 
#     #         reference: 
#     #     """
#     #     current_mouse_pos[0] = x
#     #     current_mouse_pos[1] = y
#     #     #print("The mouse is currently at", current_mouse_pos)
#     #     if event == cv2.EVENT_LBUTTONDOWN: print("Left button clicked!")
#     #     if event == cv2.EVENT_RBUTTONDOWN: print("Right button clicked!")

#     # cv2.namedWindow('image')
#     # cv2.setMouseCallback('image',mouse_handler)

#     while True:
#         x, y = coordinate  # adjusted by the mouse_handler!
#         # while True:
#         cv2.imshow('image',image)

#         x_change = 0
#         y_change = 0
#         """ key-press handling """
#         k = cv2.waitKey(20) & 0xFF
#         k_char = chr(k)
#         if k_char == 'f':
#             x_change = -10
#         if k_char == 't':
#             y_change = -10
#         if k_char == 'h':
#             x_change = 10
#         if k_char == 'g':
#             y_change = 10
        
#         DELTA = 1
#         UL = (x-DELTA,y-DELTA)  # Upper Left
#         LR = (x+DELTA,y+DELTA)  # Lower Right
#         CLR = (255,255,255)  # color
#         WIDTH = 5  # rectangle width]
#         THICK = 10
#         CENTER = (x,y)
#         cv2.circle(image, CENTER, WIDTH, CLR, THICK) # draw a circle
#         if k_char == ' ': image = image_orig.copy() # clear by re-copying!
#         # if k == 27: # escape key has value 27 (no string represetation...)
#         #     print("Quitting!")
#         #     break
#         # """ end of key-press handling """

#         x += x_change
#         y += y_change




#         # outside of the while True loop...
#     cv2.destroyAllWindows()

import cv2

def main():
    print ("connecting to myo")
    try:
        hub = libmyo.Hub()
    except MemoryError:
        print("could not connect")
        return
    hub.set_locking_policy(libmyo.LockingPolicy.none)
    L = Listener()
    hub.run(1000, L)

    # cam = cv2.VideoCapture(0)
    FILE_NAME = "messi5.jpg"
    image_orig = cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
    #image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image = image_orig.copy()
    # current_mouse_pos = [0,0]  # 

    try:
        while hub.running:

            #START OF CV2 PIXEL PROCESSING 
            x, y = coords  
        # while True:
            cv2.imshow('image',image)

            x_change = 0
            y_change = 0
            """ key-press handling """
            k = cv2.waitKey(20) & 0xFF
            k_char = chr(k)

            DELTA = 1
            UL = (x-DELTA,y-DELTA)  # Upper Left
            LR = (x+DELTA,y+DELTA)  # Lower Right
            CLR = (255,255,255)  # color
            WIDTH = 5  # rectangle width]
            THICK = 10
            CENTER = (x,y)
            cv2.circle(image, CENTER, WIDTH, CLR, THICK) # draw a circle
            print("i just drew a circle at", CENTER)
            # if k_char == ' ': image = image_orig.copy() # clear by re-copying!
            # if k == 27: # escape key has value 27 (no string represetation...)
            #     print("Quitting!")
            #     break
            # """ end of key-press handling """

            # x += x_change
            # y += y_change
            #END PIXEL PROCESSING 



            sleep(0.25)
    except KeyboardInterrupt:
        print("quitting")
    finally:
        print("shutting down")
        hub.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()