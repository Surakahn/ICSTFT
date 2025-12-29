# 从 __future__ 模块导入 print_function，确保在 Python 2 中也能使用 Python 3 的 print 函数语法
from __future__ import print_function

# 导入必要的库
import numpy as np  # 用于科学计算的数组操作
import time  # 用于时间相关操作
import os  # 用于操作系统相关功能
from PIL import Image  # 用于图像处理
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns  # 用于美化图表

# 从 Keras 导入 ResNet50 预训练模型
from keras.applications.resnet50 import ResNet50
# 从 Keras 导入图像预处理工具
from keras.preprocessing import image
# 从 Keras 导入 ResNet50 的预处理和预测解码函数
from keras.applications.resnet50 import preprocess_input, decode_predictions

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 定义 ResNet50 模型的均值，用于图像预处理
# 这些值对应 BGR 通道的均值（注意：OpenCV 和 Keras 使用 BGR 顺序）
RESNET_MEAN = np.array([103.939, 116.779, 123.68])

def orthogonal_perturbation(delta, prev_sample, target_sample):
    """
    生成正交扰动，使样本在保持与目标样本距离不变的情况下进行扰动
    
    参数:
    delta: 扰动强度，控制扰动的大小
    prev_sample: 当前对抗样本 (1, 224, 224, 3)
    target_sample: 目标样本 (1, 224, 224, 3)
    
    返回:
    perturb: 生成的正交扰动向量 (1, 224, 224, 3)
    
    作用:
    1. 生成随机扰动并归一化
    2. 将扰动投影到与目标样本方向正交的平面上
    3. 确保扰动后不会超出图像像素的有效范围 [0, 255]
    """
    # 生成随机扰动 (1, 224, 224, 3)
    perturb = np.random.randn(1, 224, 224, 3)
    # 归一化扰动，使每个样本的 L2 范数为 1
    perturb /= np.linalg.norm(perturb, axis=(1, 2))
    # 根据目标样本和当前样本的差异调整扰动大小
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
    
    # 计算从当前样本指向目标样本的方向向量
    diff = (target_sample - prev_sample).astype(np.float32)  # 正交于球面的向量
    # 归一化方向向量
    diff /= get_diff(target_sample, prev_sample)  # 单位正交向量
    
    # 将扰动投影到球面上：减去在 diff 方向上的分量
    # 这样扰动就保持在与目标样本等距的球面上
    perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2) * diff
    
    # 检查并修正超出范围的像素值
    # 计算上溢：当前样本 + 扰动 - 255 + 均值（考虑 ResNet 预处理）
    overflow = (prev_sample + perturb) - 255 + RESNET_MEAN
    # 修正上溢：如果 overflow > 0，则减去溢出部分
    perturb -= overflow * (overflow > 0)
    
    # 计算下溢：-RESNET_MEAN（因为预处理后像素值会减去均值）
    underflow = -RESNET_MEAN
    # 修正下溢：如果 underflow > 0，则加上下溢部分
    perturb += underflow * (underflow > 0)
    
    return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
    """
    生成前向扰动，使样本向目标样本方向移动
    
    参数:
    epsilon: 步长，控制向目标样本移动的比例
    prev_sample: 当前对抗样本 (1, 224, 224, 3)
    target_sample: 目标样本 (1, 224, 224, 3)
    
    返回:
    perturb: 生成的前向扰动向量 (1, 224, 224, 3)
    
    作用:
    生成一个指向目标样本的扰动，使当前样本向目标样本方向移动
    """
    # 计算从当前样本指向目标样本的方向向量
    perturb = (target_sample - prev_sample).astype(np.float32)
    # 按比例 epsilon 缩放扰动
    perturb *= epsilon
    return perturb

def get_converted_prediction(sample, classifier):
    """
    获取转换后的预测结果
    
    参数:
    sample: 输入样本 (1, 224, 224, 3)
    classifier: 分类器模型
    
    返回:
    label: 预测的类别标签（字符串）
    
    作用:
    模拟图像保存为 uint8 格式后的预测结果，因为：
    1. 原始样本是 float32 类型
    2. 保存为图像时会转换为 uint8，导致精度损失
    3. 这种精度损失可能改变预测结果，特别是在类别边界附近
    4. 该函数通过模拟转换过程，获得更准确的预测标签
    """
    # 模拟 uint8 转换过程：
    # 1. 加上均值恢复原始像素值
    # 2. 转换为 uint8
    # 3. 再转换回 float32
    # 4. 减去均值进行预处理
    sample = (sample + RESNET_MEAN).astype(np.uint8).astype(np.float32) - RESNET_MEAN
    # 进行预测并解码，获取 top-1 预测结果
    # decode_predictions 返回格式: [[(class_id, class_name, probability)]]
    label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
    return label

def save_image(sample, classifier, folder):
    """
    保存对抗样本图像
    
    参数:
    sample: 要保存的样本 (1, 224, 224, 3)
    classifier: 分类器模型，用于获取预测标签
    folder: 保存图像的文件夹
    
    作用:
    1. 获取样本的预测标签
    2. 反向预处理恢复原始像素值
    3. 转换为 RGB 顺序（Keras 使用 BGR 顺序）
    4. 保存为 PNG 图像，文件名包含时间戳和预测标签
    """
    # 获取转换后的预测标签（考虑 uint8 转换的影响）
    label = get_converted_prediction(np.copy(sample), classifier)
    # 移除 batch 维度，获取单个样本
    sample = sample[0]
    # 反向预处理：加上 ResNet 均值
    sample += RESNET_MEAN
    # 转换通道顺序：BGR -> RGB
    sample = sample[..., ::-1].astype(np.uint8)
    # 转换为 PIL Image 对象
    sample = Image.fromarray(sample)
    # 生成时间戳作为唯一标识
    id_no = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # 保存图像，文件名包含时间戳和预测标签
    sample.save(os.path.join("images", folder, "{}_{}.png".format(id_no, label)))

def preprocess(sample_path):
    """
    加载并预处理图像文件
    
    参数:
    sample_path: 图像文件路径
    
    返回:
    x: 预处理后的图像数组 (1, 224, 224, 3)
    
    作用:
    1. 加载图像并调整大小为 224x224
    2. 转换为 numpy 数组
    3. 添加 batch 维度
    4. 应用 ResNet50 的预处理（减去均值，BGR 顺序等）
    """
    # 加载图像，调整大小为 224x224
    img = image.load_img(sample_path, target_size=(224, 224))
    # 转换为 numpy 数组
    x = image.img_to_array(img)
    # 添加 batch 维度 (224, 224, 3) -> (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    # 应用 ResNet50 预处理：减去均值，BGR 顺序等
    x = preprocess_input(x)
    return x

def get_diff(sample_1, sample_2):
    """
    计算两个样本之间的通道级 L2 范数差异
    
    参数:
    sample_1: 第一个样本 (1, 224, 224, 3)
    sample_2: 第二个样本 (1, 224, 224, 3)
    
    返回:
    diff: 通道级 L2 范数 (1,)
    
    作用:
    计算两个样本在空间维度 (224, 224) 上的 L2 范数，保留通道维度
    """
    # 计算差异的 L2 范数，沿着空间维度 (1, 2) 计算
    # 结果形状: (1, 3) -> 取平均后为标量
    return np.linalg.norm(sample_1 - sample_2, axis=(1, 2))

def plot_attack_progress(steps_history, mse_history, delta_history, epsilon_history, calls_history, folder, current_step=None):
    """
    生成攻击过程的综合可视化图表
    
    参数:
    steps_history: 步数历史记录
    mse_history: MSE历史记录（与目标样本的距离）
    delta_history: delta参数历史记录（正交步长）
    epsilon_history: epsilon参数历史记录（前向步长）
    calls_history: 模型调用次数历史记录
    folder: 保存图表的文件夹
    current_step: 当前步数，用于文件命名
    
    作用:
    创建包含三个子图的综合图表：
    1. 顶部：MSE随步数的变化（对数坐标）
    2. 中间：delta和epsilon参数随步数的变化（对数坐标）
    3. 底部：模型调用次数随步数的变化
    所有图表保存为单个PNG文件
    """
    if len(steps_history) < 2:
        return  # 数据不足，不生成图表
    
    # 创建图表和子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Boundary Attack Progress Visualization', fontsize=16, fontweight='bold')
    
    # 子图1: MSE变化（对数坐标）
    ax1.semilogy(steps_history, mse_history, 'b-', linewidth=2, label='MSE to Target')
    ax1.axhline(y=1e-3, color='r', linestyle='--', alpha=0.7, label='Convergence Threshold (1e-3)')
    ax1.set_ylabel('MSE (log scale)', fontsize=12)
    ax1.set_title('Distance to Target Sample', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 子图2: 参数变化（对数坐标）
    ax2.semilogy(steps_history, delta_history, 'g-', linewidth=2, label='Delta (Orthogonal Step)')
    ax2.semilogy(steps_history, epsilon_history, 'm-', linewidth=2, label='Epsilon (Forward Step)')
    ax2.set_ylabel('Parameter Values (log scale)', fontsize=12)
    ax2.set_title('Step Size Parameters', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # 子图3: 模型调用次数
    ax3.plot(steps_history, calls_history, 'r-', linewidth=2, label='Cumulative Model Calls')
    ax3.set_xlabel('Attack Steps', fontsize=12)
    ax3.set_ylabel('Number of Calls', fontsize=12)
    ax3.set_title('Model Query Efficiency', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    # 调整布局
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # 为总标题留出空间
    
    # 生成文件名
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if current_step is not None:
        filename = f"attack_progress_step_{current_step}_{timestamp}.png"
    else:
        filename = f"attack_progress_final_{timestamp}.png"
    
    # 保存图表
    plt.savefig(os.path.join("images", folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Progress visualization saved: {filename}")

def plot_sample_comparison(initial_sample, adversarial_sample, target_sample, classifier, folder, step):
    """
    生成样本对比可视化
    
    参数:
    initial_sample: 初始样本
    adversarial_sample: 当前对抗样本
    target_sample: 目标样本
    classifier: 分类器模型
    folder: 保存图表的文件夹
    step: 当前步数
    
    作用:
    创建包含4个子图的对比图表：
    1. 初始样本及其预测
    2. 当前对抗样本及其预测
    3. 目标样本及其预测
    4. 像素差异热力图
    """
    # 获取预测标签
    initial_label = get_converted_prediction(np.copy(initial_sample), classifier)
    adversarial_label = get_converted_prediction(np.copy(adversarial_sample), classifier)
    target_label = get_converted_prediction(np.copy(target_sample), classifier)
    
    # 计算像素差异
    diff = np.abs(adversarial_sample[0] - target_sample[0])
    max_diff = np.max(diff)
    
    # 创建图表
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Sample Comparison at Step {step}', fontsize=16, fontweight='bold')
    
    # 初始样本
    initial_img = (initial_sample[0] + RESNET_MEAN)[..., ::-1].astype(np.uint8)
    axes[0].imshow(initial_img)
    axes[0].set_title(f'Initial Sample\nPredicted: {initial_label}', fontsize=12)
    axes[0].axis('off')
    
    # 对抗样本
    adversarial_img = (adversarial_sample[0] + RESNET_MEAN)[..., ::-1].astype(np.uint8)
    axes[1].imshow(adversarial_img)
    axes[1].set_title(f'Adversarial Sample\nPredicted: {adversarial_label}', fontsize=12)
    axes[1].axis('off')
    
    # 目标样本
    target_img = (target_sample[0] + RESNET_MEAN)[..., ::-1].astype(np.uint8)
    axes[2].imshow(target_img)
    axes[2].set_title(f'Target Sample\nPredicted: {target_label}', fontsize=12)
    axes[2].axis('off')
    
    # 像素差异热力图
    diff_img = np.max(diff, axis=2)  # 取最大通道差异
    im = axes[3].imshow(diff_img, cmap='hot', vmin=0, vmax=max_diff)
    axes[3].set_title('Pixel Difference\n(Adversarial vs Target)', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # 调整布局
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    # 保存图表
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    filename = f"sample_comparison_step_{step}_{timestamp}.png"
    plt.savefig(os.path.join("images", folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Sample comparison saved: {filename}")

def boundary_attack():
    """
    实现边界攻击（Boundary Attack）算法
    
    作用:
    1. 加载预训练的 ResNet50 模型
    2. 加载初始样本（海豹图片）和目标样本（鳗鱼图片）
    3. 创建保存结果的文件夹
    4. 执行边界攻击，生成对抗样本
    5. 保存中间结果和最终结果
    6. 生成攻击过程的可视化图表
    
    边界攻击的核心思想:
    - 从初始样本（源类别）开始
    - 首先移动到决策边界
    - 然后迭代执行两个步骤：
      a) 正交步：在保持与目标样本距离不变的情况下探索边界
      b) 前向步：向目标样本方向移动，减小与目标样本的距离
    - 目标是找到一个与目标样本非常接近但仍被分类为源类别的样本
    
    新增功能:
    - 记录攻击过程的历史数据
    - 生成综合的攻击过程可视化图表
    - 生成样本对比可视化图表
    """
    # 加载预训练的 ResNet50 模型（ImageNet 权重）
    classifier = ResNet50(weights='imagenet')
    
    # 预处理初始样本（海豹图片）和目标样本（鳗鱼图片）
    initial_sample = preprocess('images/original/awkward_moment_seal.png')
    target_sample = preprocess('images/original/bad_joke_eel.png')
    
    # 创建以当前时间命名的文件夹，用于保存结果
    folder = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(os.path.join("images", folder), exist_ok=True)
    
    # 保存初始样本
    save_image(np.copy(np.array(initial_sample)), classifier, folder)
    
    # 获取初始样本和目标样本的类别
    attack_class = np.argmax(classifier.predict(initial_sample))  # 源类别（海豹）
    target_class = np.argmax(classifier.predict(target_sample))    # 目标类别（鳗鱼）
    
    # 打印类别信息
    initial_label = decode_predictions(classifier.predict(initial_sample), top=1)[0][0][1]
    target_label = decode_predictions(classifier.predict(target_sample), top=1)[0][0][1]
    print(f"Initial sample class: {initial_label} (ID: {attack_class})")
    print(f"Target sample class: {target_label} (ID: {target_class})")
    
    # 初始化对抗样本为初始样本
    adversarial_sample = np.copy(np.array(initial_sample))
    
    # 初始化攻击参数
    n_steps = 0      # 攻击步数
    n_calls = 0      # 模型调用次数
    epsilon = 1.     # 前向步长，初始值较大
    delta = 0.1      # 正交步长
    
    # 初始化历史记录
    steps_history = []
    mse_history = []
    delta_history = []
    epsilon_history = []
    calls_history = []
    
    # 第一阶段：移动到决策边界
    # 从初始样本开始，逐步向目标样本方向移动，直到到达决策边界
    print("Phase 1: Moving to decision boundary...")
    while True:
        # 生成前向扰动，向目标样本移动
        trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
        # 预测扰动后样本的类别
        prediction = classifier.predict(trial_sample)
        n_calls += 1  # 增加模型调用计数
        
        # 计算当前MSE
        current_mse = np.mean(get_diff(trial_sample, target_sample))
        
        print(f"Boundary search: epsilon={epsilon:.4f}, MSE={current_mse:.6f}, "
              f"Predicted class: {np.argmax(prediction)}")
        
        # 如果预测仍然是源类别，说明还未到达边界
        if np.argmax(prediction) == attack_class:
            adversarial_sample = trial_sample  # 接受这个样本
            print("✓ Reached decision boundary!")
            break  # 已经到达边界，退出循环
        else:
            # 如果预测变为其他类别，说明步长太大，需要减小步长
            epsilon *= 0.9
            if epsilon < 1e-6:
                print("✗ Could not reach decision boundary. Using initial sample.")
                break
    
    # 保存边界样本
    save_image(np.copy(adversarial_sample), classifier, folder)
    print("Boundary sample saved.")
    
    # 第二阶段：迭代执行边界攻击
    print("\nPhase 2: Iterative boundary attack...")
    while True:
        print(f"\nStep #{n_steps} | Total calls: {n_calls}")
        
        # 1. 正交步（Delta step）
        # 在保持与目标样本距离不变的情况下，探索决策边界
        print("\tDelta step...")
        d_step = 0  # 正交步尝试次数
        d_score = 0.0  # 正交步成功率
        while True:
            d_step += 1
            print(f"\t#Delta attempt {d_step}, delta={delta:.6f}")
            
            # 生成 10 个候选样本（并行尝试）
            trial_samples = []
            for i in range(10):  # 使用range而不是np.arange提高效率
                # 生成正交扰动
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
                trial_samples.append(trial_sample)
            
            # 预测所有候选样本
            predictions = classifier.predict(np.vstack(trial_samples))
            n_calls += 10  # 增加 10 次模型调用
            
            # 获取预测类别
            predicted_classes = np.argmax(predictions, axis=1)
            
            # 计算成功率：仍然被分类为源类别的比例
            d_score = np.mean(predicted_classes == attack_class)
            
            print(f"\tDelta success rate: {d_score:.2f} (delta={delta:.6f})")
            
            # 根据成功率调整步长
            if d_score > 0.0:  # 至少有一个样本成功
                # 调整 delta：如果成功率太低则减小，太高则增大
                if d_score < 0.3:
                    delta *= 0.9  # 减小步长
                elif d_score > 0.7:
                    delta /= 0.9  # 增大步长
                
                # 选择第一个成功的样本
                successful_indices = np.where(predicted_classes == attack_class)[0]
                adversarial_sample = trial_samples[successful_indices[0]]
                break
            else:
                # 所有样本都失败，减小步长重新尝试
                delta *= 0.9
                if delta < 1e-8:
                    print("\t✗ Delta step failed after maximum reduction. Continuing with current sample.")
                    break
        
        # 2. 前向步（Epsilon step）
        # 向目标样本方向移动，减小与目标样本的距离
        print("\tEpsilon step...")
        e_step = 0  # 前向步尝试次数
        e_success = False
        while True:
            e_step += 1
            print(f"\t#Epsilon attempt {e_step}, epsilon={epsilon:.6f}")
            
            # 生成前向扰动
            trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
            # 预测扰动后样本的类别
            prediction = classifier.predict(trial_sample)
            n_calls += 1  # 增加模型调用计数
            
            # 计算当前MSE
            current_mse = np.mean(get_diff(trial_sample, target_sample))
            
            # 如果预测仍然是源类别，说明可以向目标样本移动
            if np.argmax(prediction) == attack_class:
                adversarial_sample = trial_sample  # 接受这个样本
                epsilon /= 0.5  # 增大步长（除以 0.5 相当于乘以 2）
                e_success = True
                print(f"\t✓ Epsilon step succeeded! New MSE: {current_mse:.6f}")
                break
            # 如果尝试次数过多，可能是达到了极限
            elif e_step > 500:
                print("\t✗ Epsilon step failed after 500 attempts. Continuing with current sample.")
                break
            else:
                # 预测变为其他类别，说明步长太大，需要减小
                epsilon *= 0.5
                print(f"\t✗ Epsilon step failed. Reducing epsilon to {epsilon:.6f}")
        
        # 增加总步数
        n_steps += 1
        
        # 计算与目标样本的平均差异
        diff = np.mean(get_diff(adversarial_sample, target_sample))
        current_prediction = classifier.predict(adversarial_sample)
        current_class = np.argmax(current_prediction)
        current_label = decode_predictions(current_prediction, top=1)[0][0][1]
        
        # 记录历史数据
        steps_history.append(n_steps)
        mse_history.append(diff)
        delta_history.append(delta)
        epsilon_history.append(epsilon)
        calls_history.append(n_calls)
        
        # 打印当前状态
        print(f"Step {n_steps} completed:")
        print(f"  • MSE to target: {diff:.6f}")
        print(f"  • Current class: {current_label} (ID: {current_class})")
        print(f"  • Delta: {delta:.6f}, Epsilon: {epsilon:.6f}")
        print(f"  • Total model calls: {n_calls}")
        
        # 定期保存中间结果和生成可视化
        chkpts = [1, 5, 10, 50, 100, 500]  # 重要的检查点
        if (n_steps in chkpts) or (n_steps % 100 == 0) or (n_steps <= 10):
            print(f"\n{'='*50}")
            print(f"CHECKPOINT at step {n_steps}")
            print(f"{'='*50}")
            
            # 保存对抗样本
            save_image(np.copy(adversarial_sample), classifier, folder)
            
            # 生成攻击过程可视化
            plot_attack_progress(steps_history, mse_history, delta_history, epsilon_history, calls_history, folder, n_steps)
            
            # 生成样本对比可视化
            plot_sample_comparison(initial_sample, adversarial_sample, target_sample, classifier, folder, n_steps)
        
        # 检查是否达到终止条件
        if diff <= 1e-3 or e_step > 500:
            print(f"\n{'='*60}")
            print(f"ATTACK TERMINATED at step {n_steps}")
            print(f"{'='*60}")
            print(f"Final MSE: {diff:.6f}")
            print(f"Total model calls: {n_calls}")
            
            # 保存最终对抗样本
            save_image(np.copy(adversarial_sample), classifier, folder)
            
            # 生成最终的攻击过程可视化
            plot_attack_progress(steps_history, mse_history, delta_history, epsilon_history, calls_history, folder)
            
            # 生成最终的样本对比可视化
            plot_sample_comparison(initial_sample, adversarial_sample, target_sample, classifier, folder, n_steps)
            
            # 生成最终统计信息图表
            plot_final_statistics(mse_history, delta_history, epsilon_history, calls_history, n_steps, folder)
            
            break
    
    return n_steps, n_calls, diff, folder

def plot_final_statistics(mse_history, delta_history, epsilon_history, calls_history, total_steps, folder):
    """
    生成最终统计信息图表
    
    参数:
    mse_history: MSE历史记录
    delta_history: delta历史记录
    epsilon_history: epsilon历史记录
    calls_history: 模型调用次数历史记录
    total_steps: 总步数
    folder: 保存图表的文件夹
    
    作用:
    创建一个包含攻击统计信息的综合图表，展示效率和收敛性
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Boundary Attack Final Statistics', fontsize=16, fontweight='bold')
    
    # 图表1: MSE收敛曲线
    ax1.semilogy(range(1, len(mse_history)+1), mse_history, 'b-', linewidth=2)
    ax1.axhline(y=1e-3, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('MSE (log scale)')
    ax1.set_title('Convergence to Target Sample')
    ax1.grid(True, alpha=0.3)
    
    # 图表2: 参数变化
    steps = range(1, len(delta_history)+1)
    ax2.semilogy(steps, delta_history, 'g-', linewidth=2, label='Delta (Orthogonal)')
    ax2.semilogy(steps, epsilon_history, 'm-', linewidth=2, label='Epsilon (Forward)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Parameter Values (log scale)')
    ax2.set_title('Parameter Adaptation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图表3: 模型调用效率
    ax3.plot(range(1, len(calls_history)+1), np.diff([0] + calls_history), 'r-', linewidth=2)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Calls per Step')
    ax3.set_title('Model Query Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # 图表4: 统计摘要
    final_mse = mse_history[-1]
    total_calls = calls_history[-1]
    avg_calls_per_step = total_calls / total_steps if total_steps > 0 else 0
    
    stats_text = (
        f"Total Steps: {total_steps}\n"
        f"Final MSE: {final_mse:.6f}\n"
        f"Total Model Calls: {total_calls}\n"
        f"Avg Calls/Step: {avg_calls_per_step:.1f}\n"
        f"Initial Delta: {delta_history[0]:.6f}\n"
        f"Final Delta: {delta_history[-1]:.6f}\n"
        f"Initial Epsilon: {epsilon_history[0]:.6f}\n"
        f"Final Epsilon: {epsilon_history[-1]:.6f}"
    )
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.set_title('Attack Summary Statistics')
    ax4.axis('off')
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    filename = f"attack_statistics_final_{timestamp}.png"
    plt.savefig(os.path.join("images", folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Final statistics visualization saved: {filename}")

if __name__ == "__main__":
    """
    程序主入口
    
    作用:
    当脚本直接运行时，执行边界攻击并生成详细的可视化
    """
    print("Starting Boundary Attack...")
    print("=" * 50)
    
    # 确保images目录存在
    os.makedirs("images", exist_ok=True)
    
    # 执行边界攻击
    try:
        n_steps, n_calls, final_mse, result_folder = boundary_attack()
        
        print("\n" + "=" * 60)
        print("ATTACK COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total steps: {n_steps}")
        print(f"Total model calls: {n_calls}")
        print(f"Final MSE to target: {final_mse:.6f}")
        print(f"Results saved in: images/{result_folder}")
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"ERROR OCCURRED: {str(e)}")
        print(f"{'!'*60}")
        import traceback
        traceback.print_exc()