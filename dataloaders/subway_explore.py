# @Author: Enea Duka
# @Date: 4/21/21

import av

if __name__ == '__main__':
    root_path = '/BS/unintentional_actions/nobackup/subway/events/'
    subway_enter_path = '/BS/unintentional_actions/nobackup/subway/events/subway_entrance_turnstiles.AVI'
    subway_exit_path = '/BS/unintentional_actions/nobackup/subway/events/subway_exit_turnstiles.AVI'

    enter_container = av.open(subway_enter_path)
    frame_count = 0
    for frame in enter_container.decode(video=0):
        frame_count += 1
        print(frame_count)
