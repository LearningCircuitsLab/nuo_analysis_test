from village.custom_classes.task import Task
import random
from sound_functions import crescendo_looming_sound
import time
import softcode_functions
from pathlib import Path
from bpod_mock import BpodMock
#from gpiozero import LED

class EscapeBehavior_1(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Escape Behavior Task
        -------------------

        Animals are placed in an open arena with a shelter located in one side.
        There is a trigger zone defined by the user. When the animal enters this zone,
        there is a chance that a looming sound is played, prompting the animal to escape
        to the shelter.
        """

        # create a list to hold the states
        self.states_visited = []
        # states are lists of [START,END,MSG]

        self.animal_in_trigger_zone = False


    def start(self):
        #self.led = LED(18)
        # Open the raw file once and keep it open
        self.output_file = Path(self.rt_session_path)
        # I thought I fixed this bug before... but apparently not. It should be a "w" instead of "a".
        self.raw_file = self.output_file.open("w")
        self.raw_file.write("TRIAL;START;END;MSG;VALUE\n")
        # create a crescendo sound
        self.crescendo_sound = crescendo_looming_sound(
            amp_start=self.settings.starting_amplitude,
            amp_end=self.settings.ending_amplitude,
            ramp_duration=self.settings.ramp_duration,
            ramp_down_duration=self.settings.ramp_down_duration,
            hold_duration=self.settings.hold_duration,
            n_repeats=self.settings.n_repeats,
        )
        # create a mock bpod object to log events
        self.bpod_mock = BpodMock()

        # sleep during exploration time
        time.sleep(self.settings.exploration_time)


    def create_trial(self):
        #self.led.off()
        self.trial_start_time = time.time()
        # record it in the mock bpod
        self.bpod_mock.reset_trial()
        self.bpod_mock.current_trial["Trial start timestamp"] = self.trial_start_time
        # stop any sound that might be playing
        softcode_functions.function1()
        # innitiate grace period state
        self.current_state = [time.time() , "" , "grace_period"]
        # reset the timer for triggering attempts
        self.triggering_time_reset = time.time() - self.settings.time_between_sound_triggering_attempts
        # load the sound
        softcode_functions.function2()
        # define the trigger zone as the second area in the cam_box
        # TODO: add a warning if this is not on
        self.trigger_zone = self.cam_box.areas[int(self.settings.trigger_area) - 1]
        # get the current camera frame
        self.last_camera_frame = self.cam_box.frame_number

        # while loop to go through states
        loop_counter = 0
        while True:
            loop_counter += 1
            if loop_counter % 1000 == 0:
               print(f"Escape Task Loop {loop_counter}, with x y positions {self.cam_box.x_position} {self.cam_box.y_position} and state {self.current_state[2]} for area {self.trigger_zone}")
            time.sleep(0.001)
            # if the frame of the camera has not changed, skip the rest of the loop
            if self.cam_box.frame_number == self.last_camera_frame:
                continue
            # if the frame has changed, update the last camera frame
            self.last_camera_frame = self.cam_box.frame_number

            match self.current_state[2]:
                case "grace_period":
                    if (time.time() - self.current_state[0]) > self.settings.grace_period:
                        if self.animal_in_trigger_zone:
                            self.change_state_to("animal_inside_trigger_zone")
                        else:
                            self.change_state_to("animal_outside_trigger_zone")

                case "animal_outside_trigger_zone":
                    # self.cam_box.log("out")
                    # check if the animal is in the trigger zone
                    if self.animal_in_trigger_zone:
                        # self.cam_box.log("in")
                        #self.led.on()
                        self.register_event("animal_entered_trigger_zone")
                        self.change_state_to("animal_inside_trigger_zone")

                case "animal_inside_trigger_zone":
                    # check if the animal is still in the trigger zone
                    if not self.animal_in_trigger_zone:
                        self.register_event("animal_exited_trigger_zone")
                        self.change_state_to("animal_outside_trigger_zone")
                        continue
                    # check if the time between triggering attempts has passed
                    if (time.time() - self.triggering_time_reset) < self.settings.time_between_sound_triggering_attempts:
                        continue

                    # Wait a random 1-5 seconds before checking the trigger zone again.
                    self.next_trigger_zone_check = time.time() + random.uniform(1, 5)
                    self.change_state_to("waiting_for_random_zone_check")

                case "waiting_for_random_zone_check":
                    # Keep processing camera frames until the random waiting time has passed.
                    if time.time() < self.next_trigger_zone_check:
                        continue

                    # If the animal is outside, choose another random waiting time and retry.
                    if not self.animal_in_trigger_zone:
                        self.next_trigger_zone_check = time.time() + random.uniform(1, 5)
                        continue

                    # The animal is still inside at the random check time: decide sound/no sound.
                    if random.random() < self.settings.looming_sound_probability:
                        # play the sound
                        softcode_functions.function3()
                        # register the event in the raw file
                        self.register_event("sound_played")
                        self.change_state_to("sound_triggered")
                    else:
                        self.register_event("sound_not_played")
                        self.change_state_to("sound_not_triggered")

                    # Whether sound was played or not, wait and then finish this trial.
                case "sound_triggered" | "sound_not_triggered":
                    time.sleep(self.settings.time_to_wait_after_sound)
                    self.register_event("trial_end")
                    return


    def after_trial(self):
        # write the end time of the ending state
        self.current_state[1] = time.time()
        # append the current state to the list of states visited
        self.states_visited.append(self.current_state)
        # write the states visited to the raw file
        for state in self.states_visited:
            self.raw_file.write(f"{self.current_trial};{state[0]};{state[1]};STATE_{state[2]};\n")
            # also register it in the mock bpod
            self.bpod_mock.record_state(state[2], [state[0], state[1]])
        # write the location of the trigger zone
        # TODO: use register_value function instead
        self.raw_file.write(f"{self.current_trial};;;trigger_zone;{self.trigger_zone}\n")
        self.bpod_mock.current_trial["trigger_zone"] = self.trigger_zone
        # reset the list of states visited
        self.states_visited = []
        # reset the current state
        self.current_state = ['', '', 'none']
        # write to file the trial start and end
        self.raw_file.write(f"{self.current_trial};{self.trial_start_time};{time.time()};TRIAL;\n")

        # reset the mock bpod
        self.bpod_mock.reset_trial()


    def close(self):
        # close the file
        if not self.raw_file.closed:
            self.raw_file.close()


    def change_state_to(self, new_state: str):
        # update the current state ending time
        self.current_state[1] = time.time()
        # Register the state change event
        self.register_event(f"_Transition_to_{new_state}")
        # append the current state to the list of states visited
        self.states_visited.append(self.current_state)
        # update the current state to the new state
        self.current_state = [time.time(), '', new_state]
        return


    def register_event(self, msg: str, value = '') -> None:
        time_of_event = time.time()
        self.raw_file.write(f"{self.current_trial};{time_of_event};{''};{msg};{value}\n")
        self.bpod_mock.record_event(msg, time_of_event)
        self.bpod_mock.record_message(msg)
        return