import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process1 = Process(target=self._plot_position)
        self.plot_process2 = Process(target=self._plot_torque)
        self.plot_process3 = Process(target=self._plot_vel)
        self.plot_process4 = Process(target=self._plot_position1)
        self.plot_process5 = Process(target=self._plot_torque1)
        self.plot_process6 = Process(target=self._plot_vel1)
        self.plot_process.start()
        self.plot_process1.start()
        self.plot_process2.start()
        self.plot_process3.start()
        self.plot_process4.start()
        self.plot_process5.start()
        self.plot_process6.start()
        '''
        self.plot_process.join()
        self.plot_process1.join()
        self.plot_process2.join()
        self.plot_process3.join()
        self.plot_process4.join()
        self.plot_process5.join()
        self.plot_process6.join()
        '''

    # 绘图信息
    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["foot_forcez_l"]: a.plot(time, log["foot_forcez_l"], label='foot_force_l')
        if log["foot_forcez_r"]: a.plot(time, log["foot_forcez_r"], label='foot_force_r')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='foot contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["base_height"] : a.plot(time, log["base_height"], label='base_height')
        a.set(xlabel='time [s]', ylabel='base_height [m]', title='base_height')
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["foot_z_l"]: a.plot(time, log["foot_z_l"], label='foot_h_l')
        if log["foot_z_r"]: a.plot(time, log["foot_z_r"], label='foot_h_r')
        a.set(xlabel='time [s]', ylabel='foot_height [cm]', title='foot_height')
        a.legend()
        plt.show()

    def _plot_position(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_pos[0]"]: a.plot(time, log["dof_pos[0]"], label='measured')
        if log["dof_pos_target[0]"]: a.plot(time, log["dof_pos_target[0]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[0]')
        a.legend()

        a = axs[0, 1]
        if log["dof_pos[1]"]: a.plot(time, log["dof_pos[1]"], label='measured')
        if log["dof_pos_target[1]"]: a.plot(time, log["dof_pos_target[1]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[1]')
        a.legend()

        a = axs[0, 2]
        if log["dof_pos[2]"]: a.plot(time, log["dof_pos[2]"], label='measured')
        if log["dof_pos_target[2]"]: a.plot(time, log["dof_pos_target[2]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[2]')
        a.legend()

        a = axs[1, 0]
        if log["dof_pos[3]"]: a.plot(time, log["dof_pos[3]"], label='measured')
        if log["dof_pos_target[3]"]: a.plot(time, log["dof_pos_target[3]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[3]')
        a.legend()

        a = axs[1, 1]
        if log["dof_pos[4]"]: a.plot(time, log["dof_pos[4]"], label='measured')
        if log["dof_pos_target[4]"]: a.plot(time, log["dof_pos_target[4]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[4]')
        a.legend()

        a = axs[1, 2]
        if log["dof_pos[5]"]: a.plot(time, log["dof_pos[5]"], label='measured')
        if log["dof_pos_target[5]"]: a.plot(time, log["dof_pos_target[5]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[5]')
        a.legend()
        plt.show()
        
    def _plot_position1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_pos[6]"]: a.plot(time, log["dof_pos[6]"], label='measured')
        if log["dof_pos_target[6]"]: a.plot(time, log["dof_pos_target[6]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[6]')
        a.legend()

        a = axs[0, 1]
        if log["dof_pos[7]"]: a.plot(time, log["dof_pos[7]"], label='measured')
        if log["dof_pos_target[7]"]: a.plot(time, log["dof_pos_target[7]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[7]')
        a.legend()

        a = axs[0, 2]
        if log["dof_pos[8]"]: a.plot(time, log["dof_pos[8]"], label='measured')
        if log["dof_pos_target[8]"]: a.plot(time, log["dof_pos_target[8]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[8]')
        a.legend()

        a = axs[1, 0]
        if log["dof_pos[9]"]: a.plot(time, log["dof_pos[9]"], label='measured')
        if log["dof_pos_target[9]"]: a.plot(time, log["dof_pos_target[9]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[9]')
        a.legend()

        a = axs[1, 1]
        if log["dof_pos[10]"]: a.plot(time, log["dof_pos[10]"], label='measured')
        if log["dof_pos_target[10]"]: a.plot(time, log["dof_pos_target[10]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[10]')
        a.legend()

        a = axs[1, 2]
        if log["dof_pos[11]"]: a.plot(time, log["dof_pos[11]"], label='measured')
        if log["dof_pos_target[11]"]: a.plot(time, log["dof_pos_target[11]"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position[11]')
        a.legend()
        plt.show()

    def _plot_torque(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque[0]"]!=[]: a.plot(time, log["dof_torque[0]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[1]"]!=[]: a.plot(time, log["dof_torque[1]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[2]"]!=[]: a.plot(time, log["dof_torque[2]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[3]"]!=[]: a.plot(time, log["dof_torque[3]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[4]"]!=[]: a.plot(time, log["dof_torque[4]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[5]"]!=[]: a.plot(time, log["dof_torque[5]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        plt.show()

    def _plot_torque1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_torque[6]"]!=[]: a.plot(time, log["dof_torque[6]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[0,1]
        if log["dof_torque[7]"]!=[]: a.plot(time, log["dof_torque[7]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[0, 2]
        if log["dof_torque[8]"]!=[]: a.plot(time, log["dof_torque[8]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 0]
        if log["dof_torque[9]"]!=[]: a.plot(time, log["dof_torque[9]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 1]
        if log["dof_torque[10]"]!=[]: a.plot(time, log["dof_torque[10]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        a = axs[1, 2]
        if log["dof_torque[11]"]!=[]: a.plot(time, log["dof_torque[11]"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        plt.show()
        
    def _plot_vel(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_vel"]: a.plot(time, log["dof_vel[0]"], label='measured')
        if log["dof_vel_target[0]"]: a.plot(time, log["dof_vel_target[0]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a = axs[0,1]
        a.legend()
        if log["dof_vel[1]"]: a.plot(time, log["dof_vel[1]"], label='measured')
        if log["dof_vel_target[1]"]: a.plot(time, log["dof_vel_target[1]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[0, 2]
        if log["dof_vel[2]"]: a.plot(time, log["dof_vel[2]"], label='measured')
        if log["dof_vel_target[2]"]: a.plot(time, log["dof_vel_target[2]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[1, 0]
        if log["dof_vel[3]"]: a.plot(time, log["dof_vel[3]"], label='measured')
        if log["dof_vel_target[3]"]: a.plot(time, log["dof_vel_target[3]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[1, 1]
        if log["dof_vel[4]"]: a.plot(time, log["dof_vel[4]"], label='measured')
        if log["dof_vel_target[4]"]: a.plot(time, log["dof_vel_target[4]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[1, 2]
        if log["dof_vel[5]"]: a.plot(time, log["dof_vel[5]"], label='measured')
        if log["dof_vel_target[5]"]: a.plot(time, log["dof_vel_target[5]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        plt.show()

    def _plot_vel1(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot position targets and measured positions
        a = axs[0, 0]
        if log["dof_vel[6]"]: a.plot(time, log["dof_vel[6]"], label='measured')
        if log["dof_vel_target[6]"]: a.plot(time, log["dof_vel_target[6]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a = axs[0,1]
        a.legend()
        if log["dof_vel[7]"]: a.plot(time, log["dof_vel[7]"], label='measured')
        if log["dof_vel_target[7]"]: a.plot(time, log["dof_vel_target[7]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[0, 2]
        if log["dof_vel[8]"]: a.plot(time, log["dof_vel[8]"], label='measured')
        if log["dof_vel_target[8]"]: a.plot(time, log["dof_vel_target[8]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[1, 0]
        if log["dof_vel[9]"]: a.plot(time, log["dof_vel[9]"], label='measured')
        if log["dof_vel_target[9]"]: a.plot(time, log["dof_vel_target[9]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[1, 1]
        if log["dof_vel[10]"]: a.plot(time, log["dof_vel[10]"], label='measured')
        if log["dof_vel_target[4]"]: a.plot(time, log["dof_vel_target[4]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        a = axs[1, 2]
        if log["dof_vel[11]"]: a.plot(time, log["dof_vel[11]"], label='measured')
        if log["dof_vel_target[5]"]: a.plot(time, log["dof_vel_target[11]"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        plt.show()
        
    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()