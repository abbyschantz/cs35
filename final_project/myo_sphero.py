
import myo as libmyo
import sphero
from time import sleep
libmyo.init()
#libmyo.init(os.path.join(os.path.dirname(__file__), 'libs', 'Myo.framework'))
## export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/aschantz/Library/sdk/myo.framework



print("starting up")

s = sphero.Sphero()

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

    def on_orientation_data(self, myo, timestamp, quat):
        # print("Orientation:", quat.x, quat.y, quat.z, quat.w)
        if quat.z > -0.2:
            print("!!armed raised (>-0.2)!!, forward")
            # s.roll(250, 0)
        return

    def on_pose(self, myo, timestamp, pose):
        if pose == libmyo.Pose.double_tap:
            print("!!DOUBLE TAP!!, shutting down")
            s.stop()
            return False  # Stops the Hub
        if pose == libmyo.Pose.fingers_spread:
            print("!!FINGERS SPREAD!!, forward")
            #print("yaw", libmyo.yaw())
            s.set_rgb(0,252,80)
            s.roll(255, 0)
            return
        if pose == libmyo.Pose.wave_out:
            print("!!WAVE OUT!!, right")
            # s.set_heading(90)
            s.set_rgb(156,109,254)
            s.roll(255, 90)
            return
        if pose == libmyo.Pose.wave_in:
            print("!!WAVE IN!!, left")
            # s.set_heading(270)
            s.set_rgb(72,252,252)
            s.roll(255, 270)
            return
        if pose == libmyo.Pose.fist:
            print("!!FIST!!, stop")
            s.set_rgb(255,0,0)
            s.stop()
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



def main():
    print ("connecting to myo")
    try:
        hub = libmyo.Hub()
    except MemoryError:
        print("could not connect")
        return
    hub.set_locking_policy(libmyo.LockingPolicy.none)
    hub.run(1000, Listener())

    try:
        while hub.running:
            sleep(0.25)
    except KeyboardInterrupt:
        print("quitting")
    finally:
        print("shutting down")
        hub.shutdown()

if __name__ == '__main__':
    main()