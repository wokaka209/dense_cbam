'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-14 14:33:12
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-14 14:33:16
FilePath: \my_densefuse_advantive\verify_output_sizes.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-
"""
@file name:verify_output_sizes.py
@desc: 验证输出图像尺寸与原始尺寸是否一致
@Author: wokaka209
"""
from torchvision.io import read_image, ImageReadMode
import os

class OutputSizeVerifier:
    def verify_sizes(self, ir_dir, output_dir):
        """验证输出尺寸"""
        print("="*60)
        print("验证输出图像尺寸")
        print("="*60)
        
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        output_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        success_count = 0
        fail_count = 0
        size_mismatches = []
        
        for i, (ir_file, output_file) in enumerate(zip(ir_files[:10], output_files[:10])):
            ir_path = os.path.join(ir_dir, ir_file)
            output_path = os.path.join(output_dir, output_file)
            
            ir_image = read_image(ir_path, mode=ImageReadMode.RGB)
            output_image = read_image(output_path, mode=ImageReadMode.RGB)
            
            ir_size = ir_image.shape[1:]
            output_size = output_image.shape[1:]
            
            print(f"\n验证 {i}: {ir_file}")
            print(f"  原始尺寸: {ir_size}")
            print(f"  输出尺寸: {output_size}")
            
            if ir_size == output_size:
                print(f"  ✓ 尺寸一致！")
                success_count += 1
            else:
                print(f"  ❌ 尺寸不匹配！")
                size_mismatches.append((ir_file, ir_size, output_size))
                fail_count += 1
        
        print("\n" + "="*60)
        print(f"验证完成！成功: {success_count}, 失败: {fail_count}")
        print(f"成功率: {success_count/(success_count+fail_count)*100:.2f}%")
        
        if size_mismatches:
            print("\n尺寸不匹配的图像:")
            for file, orig_size, out_size in size_mismatches:
                print(f"  {file}: {orig_size} -> {out_size}")
        
        print("="*60)
        
        return success_count == (success_count + fail_count)

if __name__ == "__main__":
    verifier = OutputSizeVerifier()
    
    ir_dir = "E:/whx_Graduation project/baseline_project/dataset/ir"
    output_dir = "data_result/batch_fusion_size_fixed"
    
    success = verifier.verify_sizes(ir_dir, output_dir)
    
    if success:
        print("\n✓ 所有输出图像尺寸与原始尺寸一致！")
    else:
        print("\n❌ 部分输出图像尺寸与原始尺寸不一致！")