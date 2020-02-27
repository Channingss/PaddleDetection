# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import ast
import time
import json

import gflags
import yaml
import cv2

import numpy as np
import paddle.fluid as fluid

from concurrent.futures import ThreadPoolExecutor, as_completed

gflags.DEFINE_string("conf", default="", help="Configuration File Path")
gflags.DEFINE_string("input_dir", default="", help="Directory of Input Images")
gflags.DEFINE_string("trt_mode", default="", help="Use optimized model")
gflags.DEFINE_string("ext",
                     default=".jpeg|.jpg",
                     help="Input Image File Extensions")
gflags.DEFINE_float('threshold', 0.0, 'threshold of score')
gflags.DEFINE_string('c2l_path', 'ghk', 'class to label path')
Flags = gflags.FLAGS


# Generate ColorMap for visualization
def colormap(rgb=False):
    """
    Get colormap
    """
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


# Paddle-TRT Precision Map
trt_precision_map = {
    "int8": fluid.core.AnalysisConfig.Precision.Int8,
    "fp32": fluid.core.AnalysisConfig.Precision.Float32,
    "fp16": fluid.core.AnalysisConfig.Precision.Half
}


# scan a directory and get all images with support extensions
def get_images_from_dir(img_dir, support_ext=".jpg|.jpeg"):
    if (not os.path.exists(img_dir) or not os.path.isdir(img_dir)):
        raise Exception("Image Directory [%s] invalid" % img_dir)
    imgs = []
    for item in os.listdir(img_dir):
        ext = os.path.splitext(item)[1][1:].strip().lower()
        if (len(ext) > 0 and ext in support_ext):
            item_path = os.path.join(img_dir, item)
            imgs.append(item_path)
    return imgs


# Deploy Configuration File Parser
class DeployConfig:
    def __init__(self, conf_file):
        if not os.path.exists(conf_file):
            raise Exception('Config file path [%s] invalid!' % conf_file)

        with open(conf_file) as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)

            deploy_conf = configs["DEPLOY"]
            # 1. get eval_crop_size
            self.eval_crop_size = ast.literal_eval(
                deploy_conf["EVAL_CROP_SIZE"])
            # 2. get mean
            self.mean = deploy_conf["MEAN"]
            # 3. get std
            self.std = deploy_conf["STD"]
            # 4. get class_num
            self.class_num = deploy_conf["NUM_CLASSES"]
            # 5. get paddle model and params file path
            self.model_file = os.path.join(deploy_conf["MODEL_PATH"],
                                           deploy_conf["MODEL_FILENAME"])
            self.param_file = os.path.join(deploy_conf["MODEL_PATH"],
                                           deploy_conf["PARAMS_FILENAME"])
            # 6. use_gpu
            self.use_gpu = deploy_conf["USE_GPU"]
            # 7. predictor_mode
            self.predictor_mode = deploy_conf["PREDICTOR_MODE"]
            # 8. batch_size
            self.batch_size = deploy_conf["BATCH_SIZE"]
            # 9. channels
            self.channels = deploy_conf["CHANNELS"]
            # 10. resize_type
            self.resize_type = deploy_conf["RESIZE_TYPE"]
            # 11. target_short_size
            self.target_short_size = deploy_conf["TARGET_SHORT_SIZE"]
            # 12. resize_max_size
            self.resize_max_size = deploy_conf["RESIZE_MAX_SIZE"]
            # 13. coarsest_stride
            if "COARSEST_STRIDE" in deploy_conf:
                self.coarsest_stride = deploy_conf["COARSEST_STRIDE"]
            # 14. feeds_size
            self.feeds_size = deploy_conf["FEEDS_SIZE"]


class ImageReader:
    def __init__(self, configs):
        self.config = configs
        self.threads_pool = ThreadPoolExecutor(configs.batch_size)

    # image processing thread worker
    def process_worker(self, imgs, idx):
        image_path = imgs[idx]
        im = cv2.imread(image_path, -1)
        im = im[:, :, :].astype('float32') / 255.0
        channels = im.shape[2]
        ori_h = im.shape[0]
        ori_w = im.shape[1]
        if channels == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            channels = im.shape[2]
        if channels != 3 and channels != 4:
            print("Only support rgb(gray) or rgba image.")
            return -1
        scale_ratio = 0
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.config.resize_type == 'UNPADDING' or gflags.FLAGS.trt_mode == 'trt_mode':
            # resize to eval_crop_size
            eval_crop_size = self.config.eval_crop_size
            if (ori_h != eval_crop_size[0] or ori_w != eval_crop_size[1]):
                im = cv2.resize(im,
                                eval_crop_size,
                                fx=0,
                                fy=0,
                                interpolation=cv2.INTER_LINEAR)
        else:
            scale_ratio = self.scaling(ori_w, ori_h,
                                       self.config.target_short_size,
                                       self.config.resize_max_size)
            im = cv2.resize(im,
                            None,
                            fx=scale_ratio,
                            fy=scale_ratio,
                            interpolation=cv2.INTER_LINEAR) 
        # if use models with no pre-processing/post-processing op optimizations
        new_h, new_w = im.shape[0], im.shape[1]
        im_mean = np.array(self.config.mean).reshape((3, 1, 1))
        im_std = np.array(self.config.std).reshape((3, 1, 1))
        # HWC -> CHW, don't use transpose((2, 0, 1))
        im = im.swapaxes(1, 2)
        im = im.swapaxes(0, 1)
        im -= im_mean
        im /= im_std
        im = im[np.newaxis, :, :, :]
        info = [image_path, im, [ori_h, ori_w], [new_h, new_w], scale_ratio]
        return info

    # process multiple images with multithreading
    def process(self, imgs):
        img_datas = []
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as exe_pool:
            tasks = [
                exe_pool.submit(self.process_worker, imgs, idx)
                for idx in range(len(imgs))
            ]
        for task in as_completed(tasks):
            img_datas.append(task.result())

        if self.config.resize_type == 'RANGE_SCALING':
            img_datas = self.padding(img_datas)

        return img_datas

    def scaling(self, w, h, target_size, max_size):
        im_max_size = max(w, h)
        im_min_size = min(w, h)
        scale_ratio = target_size / im_min_size
        if max_size > 0:
            if round(scale_ratio * im_max_size) > max_size:
                scale_ratio = max_size / im_max_size
        return scale_ratio

    def padding(self, img_infos):
        max_h = 0
        max_w = 0
        for img_info in img_infos:
            max_h = max(max_h, img_info[3][0])
            max_w = max(max_w, img_info[3][1])
#        max_h = int(max_h / coarsest_stride - 1) * coarsest_stride
#        max_w = int(max_w / coarsest_stride - 1) * coarsest_stride
        for img_info in img_infos:
            h = img_info[3][0]
            w = img_info[3][1]
            pad_width = ((0, 0), (0, 0), (0, max_h - h), (0, max_w - w))
            img_info[1] = np.pad(img_info[1],
                                 pad_width,
                                 mode='constant',
                                 constant_values=0)
            img_info[3][0] = max_h
            img_info[3][1] = max_w
        return img_infos


class Predictor:
    def __init__(self, conf_file):
        self.config = DeployConfig(conf_file)
        self.image_reader = ImageReader(self.config)
        if self.config.predictor_mode == "NATIVE":
            predictor_config = fluid.core.NativeConfig()
            predictor_config.prog_file = self.config.model_file
            predictor_config.param_file = self.config.param_file
            predictor_config.use_gpu = self.config.use_gpu
            predictor_config.device = 0
            predictor_config.fraction_of_gpu_memory = 0
        elif self.config.predictor_mode == "ANALYSIS":
            predictor_config = fluid.core.AnalysisConfig(
                self.config.model_file, self.config.param_file)
            if self.config.use_gpu:
                predictor_config.enable_use_gpu(100, 0)
                predictor_config.switch_ir_optim(True)
                if gflags.FLAGS.trt_mode != "":
                    precision_type = trt_precision_map[gflags.FLAGS.trt_mode]
                    use_calib = (gflags.FLAGS.trt_mode == "int8")
                    predictor_config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        max_batch_size=self.config.batch_size,
                        min_subgraph_size=40,
                        precision_mode=precision_type,
                        use_static=False,
                        use_calib_mode=use_calib)
            else:
                predictor_config.disable_gpu()
            predictor_config.switch_specify_input_names(True)
            predictor_config.enable_memory_optim()
        self.predictor = fluid.core.create_paddle_predictor(predictor_config)

    def create_data_tensor(self, inputs, batch_size, shape):
        im_tensor = fluid.core.PaddleTensor()
        im_tensor.name = "image"
        im_tensor.shape = [batch_size, self.config.channels] + shape
        im_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        im_tensor.data = fluid.core.PaddleBuf(inputs.ravel().astype("float32"))
        return im_tensor

    def create_size_tensor(self, inputs, batch_size, feeds_size):
        im_tensor = fluid.core.PaddleTensor()
        im_tensor.name = "im_shape"
        im_tensor.shape = [batch_size, feeds_size]
        if feeds_size == 2:
            im_tensor.dtype = fluid.core.PaddleDType.INT32
            im_tensor.data = fluid.core.PaddleBuf(inputs.astype("int32"))
        else:
            im_tensor.dtype = fluid.core.PaddleDType.FLOAT32
            im_tensor.data = fluid.core.PaddleBuf(inputs.ravel().astype("float32"))
        return im_tensor

    def create_info_tensor(self, inputs, batch_size):
        im_tensor = fluid.core.PaddleTensor()
        im_tensor.name = "im_info"
        im_tensor.shape = [batch_size, 3]
        im_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        im_tensor.data = fluid.core.PaddleBuf(inputs.ravel().astype("float32"))
        return im_tensor

    # save prediction results and visualization them
    def output_result(self, imgs_data, infer_out):
        color_list = colormap(rgb=True)
        text_thickness = 1
        text_scale = 0.3
        batch_boxes = infer_out.as_ndarray()
        lod = infer_out.lod[0]
        with open(Flags.c2l_path, "r", encoding="utf-8") as json_f:
            class2LabelMap = json.load(json_f)
            for idx in range(len(lod) - 1):
                boxes = batch_boxes[lod[idx]:lod[idx + 1], :]
                img_name = imgs_data[idx][0]
                ori_shape = imgs_data[idx][2]
                img = cv2.imread(img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for box in boxes:
                    box_class = int(box[0])
                    box_score = box[1]
                    if box[1] >= Flags.threshold:
                        left_top_x, left_top_y, right_bottom_x, right_bottom_y = box[
                            2:]
                        text_class_score_str = "%s %.2f" % (class2LabelMap.get(
                            str(box_class)), box_score)
                        text_point = (int(left_top_x), int(left_top_y))
                        ptLeftTop = (int(left_top_x), int(left_top_y))
                        ptRightBottom = (int(right_bottom_x),
                                         int(right_bottom_y))
                        box_thickness = 1
                        color = tuple([int(c) for c in color_list[box_class]])
                        cv2.rectangle(img, ptLeftTop, ptRightBottom, color,
                                      box_thickness, 8)
                        if text_point[1] < 0:
                            text_point = (int(left_top_x), int(right_bottom_y))
                        WHITE = (255, 255, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(text_class_score_str, font,
                                                    text_scale, text_thickness)

                        text_box_left_top = (text_point[0],
                                             text_point[1] - text_size[0][1])
                        text_box_right_bottom = (text_point[0] +
                                                 text_size[0][0], text_point[1])

                        cv2.rectangle(img, text_box_left_top,
                                      text_box_right_bottom, color, -1, 8)
                        cv2.putText(img, text_class_score_str, text_point, font,
                                    text_scale, WHITE, text_thickness)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_name + '_result.jpg', img)
                # visualization score png
                print("save result of [" + img_name + "] done.")

    def predict(self, images):
        # image reader preprocessing time cost
        reader_time = 0
        # inference time cost
        infer_time = 0
        # post_processing: generate mask and visualize it
        post_time = 0
        # total time cost: preprocessing + inference + postprocessing
        total_runtime = 0

        # record starting time point
        total_start = time.time()
        feeds_size = self.config.feeds_size
        batch_size = self.config.batch_size
        for i in range(0, len(images), batch_size):
            real_batch_size = batch_size
            if i + batch_size >= len(images):
                real_batch_size = len(images) - i
            reader_start = time.time()
            img_datas = self.image_reader.process(images[i:i + real_batch_size])
            feeds = []
            input_data = np.concatenate([item[1] for item in img_datas])
            input_data = self.create_data_tensor(input_data, real_batch_size, img_datas[0][3])
            feeds.append(input_data)
            if feeds_size == 2:
                input_size = np.concatenate([item[2] for item in img_datas])
                input_size = self.create_size_tensor(input_size,
                                                     real_batch_size,
                                                     feeds_size)
                feeds.append(input_size)
            if feeds_size == 3:
                input_size = np.concatenate([[item[2][0], item[2][1], 1.0]
                                             for item in img_datas])
                input_size = self.create_size_tensor(input_size,
                                                     real_batch_size,
                                                     feeds_size)
                feeds.append(input_size)
                input_info = np.concatenate([[item[3][0], item[3][1], item[4]]
                                             for item in img_datas])
                input_info = self.create_info_tensor(input_info,
                                                     real_batch_size)
                feeds.append(input_info)

            reader_end = time.time()
            infer_start = time.time()
            output_data = self.predictor.run(feeds)[0]
            infer_end = time.time()
            post_start = time.time()
            self.output_result(img_datas, output_data)
            post_end = time.time()

            reader_time += (reader_end - reader_start)
            infer_time += (infer_end - infer_start)
            post_time += (post_end - post_start)

        # finishing process all images
        total_end = time.time()
        # compute whole processing time
        total_runtime = (total_end - total_start)
        print(
            "images_num=[%d],preprocessing_time=[%f],infer_time=[%f],postprocessing_time=[%f],total_runtime=[%f]"
            % (len(images), reader_time, infer_time, post_time, total_runtime))


def run(deploy_conf, imgs_dir, support_extensions=".jpg|.jpeg"):
    # 1. scan and get all images with valid extensions in directory imgs_dir
    imgs = get_images_from_dir(imgs_dir)
    if len(imgs) == 0:
        print("No Image (with extensions : %s) found in [%s]" %
              (support_extensions, imgs_dir))
        return -1
    # 2. create a predictor
    seg_predictor = Predictor(deploy_conf)
    # 3. do a inference on images
    seg_predictor.predict(imgs)
    return 0


if __name__ == "__main__":
    # 0. parse the arguments
    gflags.FLAGS(sys.argv)
    if (gflags.FLAGS.conf == "" or gflags.FLAGS.input_dir == ""):
        print("Usage: python infer.py --conf=/config/path/to/your/model " +
              "--input_dir=/directory/of/your/input/images")
        exit(-1)
    # set empty to turn off as default
    trt_mode = gflags.FLAGS.trt_mode
    if (trt_mode != "" and trt_mode not in trt_precision_map):
        print("Invalid trt_mode [%s], only support[int8, fp16, fp32]" %
              trt_mode)
        exit(-1)
    # run inference
    run(gflags.FLAGS.conf, gflags.FLAGS.input_dir, gflags.FLAGS.ext)
