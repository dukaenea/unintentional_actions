# @Author: Enea Duka
# @Date: 7/16/21

import av
import cv2
import drawSvg as draw

video_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/Are You Serious! - Throwback Thursday (September 2017) _ FailArmy54.mp4'



def visualize_perdiction(video_name, gt_trn, p_trn):
    video_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/' + video_name + '.mp4'
    video = cv2.VideoCapture(video_path)
    save_path = '/BS/unintentional_actions/work/unintentional_actions/example.png'
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    nr_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = nr_frames / fps

    print(fps)
    print('frame dimensions (HxW):', int(frame_height), "x", int(frame_width))
    video_name = video_path.split('/')[-1]
    video_save_path = '/BS/unintentional_actions/work/unintentional_actions/' + video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_save_path, fourcc, fps, (int(frame_width), int(frame_height)))
    frame_count = 0

    gt_trn = gt_trn / duration
    p_trn = p_trn / duration

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_time = frame_count / fps
        print(frame_time / duration)
        gt_text = 'intentional' if (frame_time/duration) <= gt_trn else 'unintentional'
        p_text = 'intentional' if (frame_time/duration) <= p_trn else 'unintentional'
        create_overlay(int(frame_width), int(frame_height), frame_time/duration, gt_trn, p_trn, gt_text, p_text, save_path)
        img = cv2.imread(save_path, -1)
        img_width, img_height, _ = img.shape
        alpha_s = img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[0:img_width, 0:img_height, c] = (alpha_s * img[:, :, c] +
                                                    alpha_l * frame[0:img_width, 0:img_height, c])

        # frame[0:img_width, 0:img_height] = img
        out.write(frame)
        frame_count += 1

    video.release()
    out.release()


def create_overlay(c_width, c_height, cursor_position, gt_trn, p_trn, gt_text, p_text, save_path):
    d = draw.Drawing(c_width, c_height, origin=(0, 0))

    # Draw a cropped circle
    # c = draw.Circle(0, 0, 0.5, stroke_width='0.01', stroke='black',
    #                 fill_opacity=0.3, clip_path=clip,
    #                 id='circle')
    # d.append(draw.Text('Ground Truth', 20, -10, 35, fill='white'))  # Text with font size 8
    # r = draw.Rectangle(x=0, y=0, width=40, height=40,
    #                     stroke_width='1', stroke='white',
    #                     fill_opacity=0.3, id='rect',)
    # d.append(c)

    # d.append(r)
    bottom_bar_margin, top_bar_margin, bar_left_margin = draw_texts(c_width, c_height, d, gt_text, p_text)
    draw_bars(c_width, c_height, bottom_bar_margin, top_bar_margin, bar_left_margin, gt_trn, p_trn, cursor_position, d)
    # Display
    d.setRenderSize(c_width)
    d.rasterize()

    d.savePng(save_path)


def draw_texts(c_width, c_height, drawing, gt_action, p_action):
    # set the font size to 3% of the height
    font_size = int(c_width / 50)
    # set the left margin to 3% of the width
    left_margin = int(c_width / 33)
    # set the bottom margin of the bottom text to 4% of the height
    bottom_text_margin = int(c_height / 25)
    # bottom margin of the top text
    top_text_margin = bottom_text_margin * 1.5 + font_size

    word_bottom_text_margin = top_text_margin + font_size + bottom_text_margin
    word_top_text_margin = word_bottom_text_margin + font_size + bottom_text_margin * 0.5

    top_text = draw.Text('Ground Truth', font_size, left_margin, top_text_margin, fill='white')
    bottom_text = draw.Text('Prediction', font_size, left_margin, bottom_text_margin, fill='white')
    word_top_text = draw.Text('Ground Truth: %s' % gt_action, font_size, left_margin, word_top_text_margin,
                              fill='white')
    word_bottom_text = draw.Text('Prediction: %s' % p_action, font_size, left_margin, word_bottom_text_margin,
                                 fill='white')
    mask_rect = draw.Rectangle(x=0, y=0, width=c_width, height=word_top_text_margin + font_size * 2, fill='black', fill_opacity=0.7)

    drawing.append(mask_rect)
    drawing.append(top_text)
    drawing.append(bottom_text)
    drawing.append(word_top_text)
    drawing.append(word_bottom_text)

    return bottom_text_margin, top_text_margin, c_width * 0.18


def draw_bars(c_width, c_height, bottom_b_margin, top_b_margin, left_margin, gt_split_point, p_split_point,
              cursor_position, drawing):
    # calculate the bar width as 90% of the remaining space
    b_width = int((c_width - left_margin) * 0.95)
    b_height = int(c_width / 50)
    fill_opacity = 0.7
    cursor_width = 2
    cursor_height = 2 * b_height + bottom_b_margin
    cursor_bottom_margin = int(bottom_b_margin / 1.5)
    cursor_left_margin = left_margin + (int((c_width - left_margin) * 0.95) * cursor_position)

    top_bar_left = draw.Rectangle(x=left_margin, y=top_b_margin, width=b_width * gt_split_point,
                                  height=b_height, fill='green', fill_opacity=fill_opacity)
    top_bar_right = draw.Rectangle(x=left_margin + (b_width * gt_split_point), y=top_b_margin,
                                   width=b_width - (b_width * gt_split_point),
                                   height=b_height, fill='red', fill_opacity=fill_opacity)
    # bottom_bar = draw.Rectangle(x=left_margin, y=bottom_b_margin, width=b_width,
    #                         height=b_height, fill='white',  fill_opacity=fill_opacity)
    bottom_bar_left = draw.Rectangle(x=left_margin, y=bottom_b_margin, width=b_width * p_split_point,
                                     height=b_height, fill='green', fill_opacity=fill_opacity)
    bottom_bar_right = draw.Rectangle(x=left_margin + (b_width * p_split_point), y=bottom_b_margin,
                                      width=b_width - (b_width * p_split_point),
                                      height=b_height, fill='red', fill_opacity=fill_opacity)
    cursor = draw.Rectangle(x=cursor_left_margin, y=cursor_bottom_margin, width=cursor_width, height=cursor_height,
                            fill='white')


    drawing.append(top_bar_left)
    drawing.append(top_bar_right)
    drawing.append(bottom_bar_left)
    drawing.append(bottom_bar_right)
    drawing.append(cursor)


if __name__ == '__main__':
    visualize_perdiction(video_path, 0, 0, save_path)
