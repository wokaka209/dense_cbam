"""
Author: wokaka209 1325536985@qq.com
Date: 2026-02-11 11:46:34
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-10 13:30:34
FilePath: \TarDAL-main\iqa_my\metric calculation\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import cv2
import os
from metrics import *
from tqdm import tqdm
import time

eval_funcs = {
    "AG": ag,
    # "CE": cross_entropy,
    # "EI": edge_intensity,
    "EN": entropy,
    "MI": mutinf,
    # "MSE": mse,
    # "PSNR": psnr,
    "SD": sd,
    # "SF": sf,
    "SSIM": ssim,
    "Qabf": qabf,
    # "Qcb": qcb,
    # "Qcv": qcv,
    # "VIF": vif,
}


def main():
    path_fusimgs, path_irimgs, path_viimgs = [], [], []
    fus_root = "E:/whx_Graduation project/baseline_project/DenseFuse_2019/data_result/batch_fusion_mean"
    vi_root = "E:/whx_Graduation project/baseline_project/dataset/vi"
    ir_root = "E:/whx_Graduation project/baseline_project/dataset/ir"
    print("Reading data...")
    output_root = "./iqa_results"
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    metrics = eval_funcs.keys()

    img_list = os.listdir(fus_root)
    for img in img_list:
        if "\n" in img:
            img = img[:-1]
        if ".jpg" not in img and ".png" not in img:
            continue
        path_fusimgs.append(os.path.join(fus_root, img))
        path_irimgs.append(os.path.join(ir_root, img))
        path_viimgs.append(os.path.join(vi_root, img))

    if len(path_viimgs) != len(path_irimgs):
        print("The number of vi_imgs and ir_imgs are different!")

    res = {}
    for key in metrics:
        res[key] = [None] * len(path_fusimgs)

    pbar = iter(tqdm(range(len(path_fusimgs))))
    for i in range(len(path_fusimgs)):
        next(pbar)
        print("Now caculate the {}th img".format(i + 1))

        img_fus = cv2.imread(path_fusimgs[i], 0)
        img_vi = cv2.imread(path_viimgs[i], 0)
        img_ir = cv2.imread(path_irimgs[i], 0)
        max_h = img_fus.shape[0]
        max_w = img_fus.shape[1]
        img_fus = cv2.resize(img_fus, (max_w, max_h))
        img_vi = cv2.resize(img_vi, (max_w, max_h))
        img_ir = cv2.resize(img_ir, (max_w, max_h))

        for metric in metrics:
            try:
                res[metric][i] = eval_funcs[metric](img_fus)
            except:
                res[metric][i] = eval_funcs[metric](img_fus, img_vi, img_ir)

    N = len(path_fusimgs)
    timstamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    txt_file = os.path.join(output_root, "result_" + timstamp + ".txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("图像质量评估 (IQA) 结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总图像数量：{N}\n")
        f.write("=" * 60 + "\n")
        for k, v in res.items():
            mean_val = sum(v) / N
            f.write(f"{k}：{mean_val:.4f}\n")



if __name__ == "__main__":
    main()
