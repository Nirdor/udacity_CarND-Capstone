import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, decel_limit, vehicle_mass, wheel_radius):
        
        self.yaw_control = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.throttel_control = PID(0.3, 0.1, 0.001, 0., 1.)
        self.filter = LowPassFilter(0.5, 0.02)
        
        self.last_step = rospy.get_time()
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

    def control(self, dbw, velocity, linear, angular):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw:
            self.throttel_control.reset()
            return 0., 0., 0.
        
        velocity = self.filter.filt(velocity)
        
        steering = self.yaw_control.get_steering(linear, angular, velocity)
        
        error = linear - velocity
        current = rospy.get_time()
        sample_time = current - self.last_step
        self.last_step = current
        
        throttle = self.throttel_control.step(error, sample_time)
        brake = 0
        
        if linear == 0 and velocity < 1.:
            brake = 700
        elif throttle < 0.01 and error < 0.:
            throttle = 0
            brake = abs(max(error, self.decel_limit) * self.vehicle_mass * self.wheel_radius)
            
        return throttle, brake, steering
