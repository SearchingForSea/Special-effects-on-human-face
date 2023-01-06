import wx
import cv2
import time
import math
import wx.media
import datetime
import threading
import mediapipe as mp


# ROI区域中心关键点
center = {"beard": 164, "eyeglass": 168, "halfmask": 8, "facemask": 164, "mouth": 13, "pignose": 1}
special_effect_type = ['beard', 'eyeglass', 'facemask', 'halfmask', 'mouth', 'pignose']
special_effect_button_list = []


def get_all_effects(window_name, panel, effect, x_offset):
    image_name = './static/' + effect + '.png'
    image = wx.Image(image_name, wx.BITMAP_TYPE_ANY)
    width, height = image.GetWidth(), image.GetHeight()
    max_len = width if width > height else height
    if max_len == width:
        multiple = max_len / 80
    else:
        multiple = max_len / 80
    image = image.Scale(int(width / multiple), int(height / multiple))
    bitmap = wx.Bitmap(image)
    tem_bitmap_button = wx.BitmapButton(panel, x_offset, bitmap, wx.DefaultPosition, wx.DefaultSize,
                                        wx.BU_AUTODRAW)
    tem_bitmap_button.SetSize((image.GetWidth(), image.GetHeight()))
    tem_bitmap_button.SetPosition((x_offset, 10))

    if window_name == "image-window":
        # threading.Timer(0.5, aa, args=[tem_bitmap_button, example.ImageWindow, effect]).start()
        tem_bitmap_button.Bind(wx.EVT_BUTTON,
                               lambda event: change_special_effect(event, example.ImageWindow, effect))
    if window_name == "video-window":
        tem_bitmap_button.Bind(wx.EVT_BUTTON,
                               lambda event: change_special_effect(event, example.VideoWindow, effect))
    if window_name == "live-window":
        tem_bitmap_button.Bind(wx.EVT_BUTTON,
                               lambda event: change_special_effect(event, example.LiveWindow, effect))

    special_effect_button_list.append(tem_bitmap_button)


def change_special_effect(event, window, effect_name):
    if effect_name in window.icon_list:
        window.icon_list.remove(effect_name)
        if not window.icon_list:
            window.special_effect_name.SetLabel("特效名: 未选择")
            return
    else:
        window.icon_list.append(effect_name)
    label = ""
    for icon_name in window.icon_list:
        label += icon_name
        if icon_name != window.icon_list[len(window.icon_list) - 1]:
            label += "、"
    window.special_effect_name.SetLabel("特效名: (" + label + ")")

# 获取人脸关键点
def get_landmarks(image, face_mesh):
    """
    :param image: ndarray图像
    :param face_mesh: 人脸检测模型
    :return:人脸关键点列表，如[{0:(x,y),1:{x,y},...},{0:(x,y),1:(x,y)}]
    """
    landmarks = []
    height, width = image.shape[0:2]
    # 人脸关键点检测
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 解释检测结果
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            i = 0
            points = {}
            # 根据图像的高度和宽度还原关键点位置
            for landmark in face_landmarks.landmark:
                x = math.floor(landmark.x * width)
                y = math.floor(landmark.y * height)
                points[i] = (x, y)
                i += 1
            landmarks.append(points)
    return landmarks


def process_effects(landmarks, icon_path, icon_name):
    """
    :param landmarks: 检测到的人脸关键点列表
    :param icon_path: 特效图像地址
    :param icon_name: 特效名称
    :return:处理好的特效图像、特效宽、特效高
    """
    # 特效关键点，用于调整特效的尺寸
    effect_landmarks = {"beard": ((landmarks[132][0], landmarks[5][1]), (landmarks[361][0], landmarks[0][1])),
                        "eyeglass": ((landmarks[127][0], landmarks[151][1]), (landmarks[356][0], landmarks[195][1])),
                        "halfmask": ((landmarks[162][0] - 50, landmarks[10][1] - 50),
                                     (landmarks[389][0] + 50, landmarks[195][1] + 50)),
                        "facemask": ((landmarks[132][0], landmarks[197][1]), (landmarks[361][0], landmarks[152][1])),
                        "mouth": ((landmarks[61][0], landmarks[0][1]), (landmarks[291][0], landmarks[17][1])),
                        "pignose": ((landmarks[129][0], landmarks[195][1]), (landmarks[358][0], landmarks[2][1]))}

    # 读取特效图像
    icon = cv2.imread(icon_path)
    # 选择特效关键点
    pt1, pt2 = effect_landmarks[icon_name]
    x, y, x_w, y_h = pt1[0], pt1[1], pt2[0], pt2[1]
    # 调整特效的尺寸
    w, h = x_w - x, y_h - y
    effect = cv2.resize(icon, (w, h))

    return effect, w, h


# 处理单帧的函数
def process_frame(img, icon_list, model):
    # 记录该帧处理的开始时间
    start_time = time.time()
    results = get_landmarks(img, model)
    # 逐个人脸开始处理
    if results:
        for landmarks in results:
            for icon_name in icon_list:
                effect, w, h = process_effects(landmarks, "static/" + icon_name + ".png", icon_name)
                # 确定ROI
                p = center[icon_name]
                roi = img[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
                      landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w]

                if effect.shape[:2] == roi.shape[:2]:
                    # 消除特效图像中的白色背景区域
                    # 第二种方法
                    effect = swap_non_effcet2(effect, roi, 240)

                    # 将处理好的特效添加到人脸图像上
                    img[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
                    landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w] = effect

    else:
        img = cv2.putText(img, 'NO FACE DELECTED', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                         (218, 112, 214), 1, 8)

    # 记录该帧处理完毕的时间
    # end_time = time.time()
    # 计算每秒处理图像的帧数FPS
    # use_time = end_time - start_time
    # if use_time != 0:
    #     FPS = 1 / (use_time)
    #     scaler = 1
    #     img = cv.putText(img, 'FPS' + str(int(FPS)), (25 * scaler, 100 * scaler), cv.FONT_HERSHEY_SIMPLEX,
    #                      1.25 * scaler,
    #                      (0, 0, 255), 1, 8)
    # else:
    #     pass
    return img


def swap_non_effcet2(effect, roi, threshold=240):
    """
    :param effect: 特效图像
    :param roi: ROI区域
    :param threshold: 阈值
    :return: 消除背景后的特效图像
    """

    # （1）特效图像灰度化
    effect2gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
    # （2）特效图像二值化
    ret, effect2wb = cv2.threshold(effect2gray, threshold, 255, cv2.THRESH_BINARY)
    # （3）消除特效的白色背景
    effectwb = cv2.bitwise_and(roi, roi, mask=effect2wb)

    # （4）反转二值化后的特效
    effect2wb_ne = cv2.bitwise_not(effect2wb)
    # （5）处理彩色特效
    effectcolor = cv2.bitwise_and(effect, effect, mask=effect2wb_ne)
    # (6) 组合彩色特效与黑色特效
    effect_final = cv2.add(effectcolor, effectwb)

    return effect_final

def first_page_enter(event):
    example.home.Show()
    example.login.Show(False)


def image_window_return(event):
    example.ImageWindow.input.SetValue("")
    example.ImageWindow.special_effect_name.SetLabel("特效名: 未选择")
    example.ImageWindow.icon_list = []

    #  清空背景bitmap
    if example.ImageWindow.image is not None:
        example.ImageWindow.image.Destroy()
        example.ImageWindow.image_show_panel.Layout()
        example.ImageWindow.image = None

    example.ImageWindow.Show(False)
    example.home.Show()


def video_window_return(event):
    example.VideoWindow.input.SetValue("")
    example.VideoWindow.icon_list = []
    example.VideoWindow.special_effect_name.SetLabel("特效名: 未选择")
    example.VideoWindow.Show(False)
    example.home.Show()


def live_window_return(event):
    example.LiveWindow.special_effect_name.SetLabel("特效名: 未选择")
    example.LiveWindow.icon_list = []
    example.LiveWindow.Show(False)
    example.home.Show()


def open_window_close_home(event, window):
    if window == 'image-window':
        example.ImageWindow.Show()
    if window == 'video-window':
        example.VideoWindow.Show()
    if window == 'live-window':
        example.LiveWindow.Show()
    example.home.Show(False)


def display_picture(event, icon_list, is_download):
    url = example.ImageWindow.input.GetValue()
    url = url.replace("\\", "//")
    url = url.replace("\"", "")

    image = cv2.imread(url)

    height, width = image.shape[:2]

    c = width if width > height else height
    if c == width:
        multiple = c / 500
    else:
        multiple = c / 500

    width1 = int(width / multiple)
    height1 = int(height / multiple)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,  # 静态图片设置为False,视频设置为True
                                      max_num_faces=3,  # 能检测的最大人脸数
                                      refine_landmarks=True,  # 是否需要对嘴唇、眼睛、瞳孔的关键点进行定位，
                                      min_detection_confidence=0.5,  # 人脸检测的置信度
                                      min_tracking_confidence=0.5)  # 人脸追踪的置信度（检测图像时可以忽略）

    # 获取关键点
    face_landmarks = get_landmarks(image, face_mesh)

    # ROI区域中心关键点
    center = {"beard": 164, "eyeglass": 168, "halfmask": 9, "facemask": 164, "mouth": 13, "pignose": 1}

    # 处理特效
    if len(face_landmarks) <= 5 and icon_list != []:
        for landmarks in face_landmarks:
            for icon_name in icon_list:
                effect, w, h = process_effects(landmarks, "static/" + icon_name + ".png", icon_name)
                # 确定ROI
                p = center[icon_name]
                roi = image[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
                      landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w]

                if effect.shape[:2] == roi.shape[:2]:
                    # 消除特效图像中的白色背景区域
                    s = time.time()
                    # 第二种方法
                    effect = swap_non_effcet2(effect, roi, 240)

                    # 将处理好的特效添加到人脸图像上
                    image[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
                    landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w] = effect
                    # 创建窗口并设置大小

        while is_download is not True:
            cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN | cv2.WND_PROP_VISIBLE | cv2.WINDOW_FREERATIO)
            cv2.resizeWindow("Video", width1, height1)
            cv2.imshow('Video', image)

            key = cv2.waitKey(10)
            # print(key)
            if key == 27:
                break

        if is_download:
            download_image_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.png'
            cv2.imwrite('./download/' + download_image_name, image)

            # 创建消息框
            dlg = wx.MessageDialog(None, "保存成功", "提示", wx.OK)
            # 显示消息框并等待用户响应
            dlg.ShowModal()
            # 关闭消息框
            dlg.Destroy()

        cv2.destroyAllWindows()


# 新分支播放视频
def create_thread_video(event, icon_list, url, is_download):

    url = url.replace("\\", "//")
    url = url.replace("\"", "")

    thread1 = threading.Thread(name='t1', target=lambda: play_video(icon_list, url, is_download))
    thread1.start()
    thread1.join()


# 播放视频
def play_video(icon_list, url, is_download):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,  # 静态图片设置为False,视频设置为True
                                      max_num_faces=3,  # 能检测的最大人脸数
                                      refine_landmarks=True,  # 是否需要对嘴唇、眼睛、瞳孔的关键点进行定位，
                                      min_detection_confidence=0.5,  # 人脸检测的置信度
                                      min_tracking_confidence=0.5)  # 人脸追踪的置信度（检测图像时可以忽略）
    cap = cv2.VideoCapture(url)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    c = width if width > height else height
    if c == width:
        multiple = c / 750
    else:
        multiple = c / 750

    width1 = int(width / multiple)
    height1 = int(height / multiple)

    if is_download:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编码器
        download_video_name = './download/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.mp4'
        video_writer = cv2.VideoWriter(download_video_name, fourcc, 30, (width, height))

    while True:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            # 获取关键点
            face_landmarks = get_landmarks(image, face_mesh)

            # ROI区域中心关键点
            center = {"beard": 164, "eyeglass": 168, "halfmask": 9, "facemask": 164, "mouth": 13, "pignose": 1}

            # 处理特效
            if len(face_landmarks) <= 5 and icon_list != []:
                for landmarks in face_landmarks:
                    for icon_name in icon_list:
                        effect, w, h = process_effects(landmarks, "static/" + icon_name + ".png", icon_name)
                        # 确定ROI
                        p = center[icon_name]
                        roi = image[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
                              landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w]

                        if effect.shape[:2] == roi.shape[:2]:
                            # 消除特效图像中的白色背景区域
                            s = time.time()
                            # 第二种方法
                            effect = swap_non_effcet2(effect, roi, 240)

                            # 将处理好的特效添加到人脸图像上
                            image[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h,
                            landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w] = effect

            if is_download:
                video_writer.write(image)
            else:
                # 创建窗口并设置大小
                cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN | cv2.WND_PROP_VISIBLE | cv2.WINDOW_FREERATIO)
                cv2.resizeWindow("Video", width1, height1)
                cv2.imshow('Video', cv2.flip(image, 1))

                key = cv2.waitKey(10)
                # print(key)
                if key == 27:
                    break

        if is_download:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        break


def run_dynamic_detect(event, icon_list):
    # ROI区域中心关键点
    center = {"beard": 164, "eyeglass": 168, "halfmask": 8, "facemask": 164, "mouth": 13, "pignose": 1}
    # 导入三维人脸关键点检测模型
    mp_face_mesh = mp.solutions.face_mesh

    model = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # TRUE:静态图片/False:摄像头实时读取
        refine_landmarks=True,  # 使用Attention Mesh模型
        max_num_faces=5,  # 最多检测几张人脸
        min_detection_confidence=0.5,  # 置信度阈值，越接近1越准
        min_tracking_confidence=0.5,  # 追踪阈值
    )

    # 导入可视化函数和可视化样式
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 调用摄像头
    cap = cv2.VideoCapture(0)
    cap.open(0)
    # 无限循环，直到break被触发
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('ERROR')
            break
        frame = process_frame(frame, icon_list, model)
        # 展示处理后的三通道图像
        cv2.namedWindow("my_window", cv2.WND_PROP_FULLSCREEN | cv2.WND_PROP_VISIBLE | cv2.WINDOW_FREERATIO)
        cv2.imshow('my_window', cv2.flip(frame, 1))
        key = cv2.waitKey(10)
        # print(key)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


class Login(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="登录页", size=(400, 350),
                          style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN)
        panel = wx.Panel(self)
        title = wx.StaticText(panel, label="欢迎来到人脸特效项目", pos=(95, 70))
        title_font = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(title_font)

        bt_enter = wx.Button(panel, label="进入", pos=(145, 200))
        bt_enter.SetSize(90, 35)
        bt_font = wx.Font(10, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        bt_enter.SetFont(bt_font)

        bt_enter.Bind(wx.EVT_BUTTON, first_page_enter)

        wx.Frame.Center(self)


class Home(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="主页", size=(500, 450),
                          style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN)
        wx.Frame.Center(self)

        panel = wx.Panel(self)  # 创建画板
        title = wx.StaticText(panel, label='请选择使用类型', pos=(175, 80))

        bitmap = wx.ArtProvider.GetBitmap(wx.ART_GO_UP, size=(50, 50))
        icon1 = wx.StaticBitmap(panel, -1, bitmap)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_FOLDER, size=(50, 50))
        icon2 = wx.StaticBitmap(panel, -1, bitmap)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_FIND, size=(50, 50))
        icon3 = wx.StaticBitmap(panel, -1, bitmap)
        bt_up_image = wx.Button(panel, label="上传图片生成特效", size=(200, 50), pos=(180, 130))
        bt_up_video = wx.Button(panel, label="上传视频生成特效", size=(200, 50), pos=(180, 200))
        bt_on_live = wx.Button(panel, label="开启摄像头生成特效", size=(200, 50), pos=(180, 270))

        bt_up_image.Bind(wx.EVT_BUTTON, lambda event: open_window_close_home(event, 'image-window'))
        bt_up_video.Bind(wx.EVT_BUTTON, lambda event: open_window_close_home(event, 'video-window'))
        bt_on_live.Bind(wx.EVT_BUTTON, lambda event: open_window_close_home(event, 'live-window'))

        title_font = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        bt_font = wx.Font(11, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(title_font)
        bt_up_image.SetFont(bt_font)
        bt_up_video.SetFont(bt_font)
        bt_on_live.SetFont(bt_font)

        icon1.SetPosition((100, 130))
        icon2.SetPosition((100, 200))
        icon3.SetPosition((100, 270))


class ImageWindow(wx.Frame):
    image_show_panel = None
    input = None
    image = None
    special_effect_name = None
    icon_list = []

    def __init__(self, parent, id):

        wx.Frame.__init__(self, parent, id, title="主页", size=(700, 640),
                          style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN)
        wx.Frame.Center(self)

        image_window_panel = wx.Panel(self)
        self.image_show_panel = wx.Panel(image_window_panel, pos=(60, 105), size=(565, 340))
        self.image_show_panel.SetBackgroundColour("#D8D8DE")

        # 创建 ScrolledWindow 控件
        self.scrolled_window = wx.ScrolledWindow(image_window_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                                 wx.HSCROLL)
        # 设置 ScrolledWindow 的滚动范围
        self.scrolled_window.SetScrollbars(1, 1, 800, 80)
        self.scrolled_window.SetPosition((50, 450))
        self.scrolled_window.SetSize(520, 100)

        # 创建多个内容
        x_offset = 10
        for effect in special_effect_type:
            get_all_effects('image-window', self.scrolled_window, effect, x_offset)
            x_offset += 100

        title = wx.StaticText(image_window_panel, label='输入图片路径添加特效', pos=(260, 30))
        input_title = wx.StaticText(image_window_panel, label="输入图片绝对路径", pos=(70, 68))
        self.special_effect_name = wx.StaticText(self.image_show_panel, label="特效名: 未选择", pos=(40, 50))
        self.input = wx.TextCtrl(image_window_panel, value="", pos=(200, 66), size=(320, 20))
        bt_confirm = wx.Button(image_window_panel, label="OK", pos=(540, 64), size=(50, 25))
        save_download = wx.Button(image_window_panel, label="保存/下载", pos=(500, 555), size=(100, 30))
        bt_return = wx.Button(image_window_panel, label="返回", pos=(87, 555), size=(80, 30))

        title_font = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(title_font)
        self.special_effect_name.SetFont(title_font)
        input_font = wx.Font(10, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        input_title.SetFont(input_font)

        bt_confirm.Bind(wx.EVT_BUTTON, lambda event: display_picture(event, self.icon_list, False))
        save_download.Bind(wx.EVT_BUTTON, lambda event: display_picture(event, self.icon_list, True))
        bt_return.Bind(wx.EVT_BUTTON, image_window_return)
        # save_download.Bind(wx.EVT_BUTTON, self.download_picture)

    #  图片下载函数
    # def download_picture(self):


class VideoWindow(wx.Frame):
    video_show_panel = None
    input = None
    video = None
    special_effect_name = None
    icon_list = []

    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="主页", size=(700, 640),
                          style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN)
        wx.Frame.Center(self)

        image_window_panel = wx.Panel(self)
        self.video_show_panel = wx.Panel(image_window_panel, pos=(60, 105), size=(565, 340))
        self.video_show_panel.SetBackgroundColour("#D8D8DE")

        # 创建 ScrolledWindow 控件
        self.scrolled_window = wx.ScrolledWindow(image_window_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                                 wx.HSCROLL)
        # 设置 ScrolledWindow 的滚动范围
        self.scrolled_window.SetScrollbars(1, 1, 800, 80)
        self.scrolled_window.SetPosition((50, 450))
        self.scrolled_window.SetSize(520, 100)
        # 创建多个内容
        x_offset = 10
        for effect in special_effect_type:
            get_all_effects('video-window', self.scrolled_window, effect, x_offset)
            x_offset += 100

        title = wx.StaticText(image_window_panel, label='输入视频路径添加特效', pos=(260, 30))
        input_title = wx.StaticText(image_window_panel, label="输入视频绝对路径", pos=(70, 68))
        upload_tips = wx.StaticText(self.video_show_panel, label="Esc键可关闭视频", pos=(200, 160))
        self.special_effect_name = wx.StaticText(self.video_show_panel, label="特效名: 未选择", pos=(40, 50))
        self.input = wx.TextCtrl(image_window_panel, value="", pos=(200, 66), size=(320, 20))
        bt_confirm = wx.Button(image_window_panel, label="确认/重播", pos=(535, 64), size=(70, 25))
        save_download = wx.Button(image_window_panel, label="保存/下载", pos=(500, 555), size=(100, 30))
        bt_return = wx.Button(image_window_panel, label="返回", pos=(87, 555), size=(80, 30))

        title_font = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(title_font)
        upload_tips.SetFont(title_font)
        self.special_effect_name.SetFont(title_font)
        input_font = wx.Font(10, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        input_title.SetFont(input_font)

        bt_confirm.Bind(wx.EVT_BUTTON, lambda event: create_thread_video(event, self.icon_list,
                                                                         self.input.GetValue(), False))
        save_download.Bind(wx.EVT_BUTTON, lambda event: create_thread_video(event, self.icon_list,
                                                                            self.input.GetValue(), True))
        bt_return.Bind(wx.EVT_BUTTON, video_window_return)


class LiveWindow(wx.Frame):
    live_show_panel = None
    video = None
    special_effect_name = None
    icon_list = []

    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="主页", size=(700, 640),
                          style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN)
        wx.Frame.Center(self)

        live_window_panel = wx.Panel(self)
        self.live_show_panel = wx.Panel(live_window_panel, pos=(60, 65), size=(565, 370))
        self.live_show_panel.SetBackgroundColour("#D8D8DE")

        # 创建 ScrolledWindow 控件
        self.scrolled_window = wx.ScrolledWindow(live_window_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                                 wx.HSCROLL)
        # 设置 ScrolledWindow 的滚动范围
        self.scrolled_window.SetScrollbars(1, 1, 800, 80)
        self.scrolled_window.SetPosition((50, 450))
        self.scrolled_window.SetSize(520, 100)

        # 创建多个内容
        x_offset = 10
        for effect in special_effect_type:
            get_all_effects('live-window', self.scrolled_window, effect, x_offset)
            x_offset += 100

        title = wx.StaticText(live_window_panel, label='打开摄像头添加特效', pos=(260, 30))
        live_tips = wx.StaticText(self.live_show_panel, label='点击下方开始触发摄像头（Esc键退出）', pos=(120, 150))
        live_tips2 = wx.StaticText(self.live_show_panel, label='可能卡顿，稍等~~~', pos=(230, 200))
        self.special_effect_name = wx.StaticText(self.live_show_panel, label="特效名: 未选择", pos=(40, 50))
        # save_download = wx.Button(live_window_panel, label="录制", pos=(500, 555), size=(100, 30))
        open = wx.Button(live_window_panel, label="开始", pos=(590, 470), size=(80, 40))
        bt_return = wx.Button(live_window_panel, label="返回", pos=(87, 555), size=(80, 30))

        title_font = wx.Font(14, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(title_font)
        live_tips.SetFont(title_font)
        self.special_effect_name.SetFont(title_font)

        bt_return.Bind(wx.EVT_BUTTON, live_window_return)
        open.Bind(wx.EVT_BUTTON, lambda event: run_dynamic_detect(event, self.icon_list))


class Example:
    def __init__(self):
        self.login = Login(parent=None, id=1)
        self.home = Home(parent=None, id=2)
        self.ImageWindow = ImageWindow(parent=None, id=3)
        self.VideoWindow = VideoWindow(parent=None, id=4)
        self.LiveWindow = LiveWindow(parent=None, id=5)


if __name__ == "__main__":
    app = wx.App()
    example = Example()
    example.login.Show()
    app.MainLoop()
