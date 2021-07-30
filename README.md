# 代码配置与运行

## SETR和DeepLabv3plus (3090服务器)

1. 代码及配置好环境的路径：`3090服务器 /mnt/projects/luoyt/mmsegmentation`
2. 代码执行需要的conda虚拟环境：`conda activate luoyt`
   * pytorch 1.9.0+cuda11.1
   * mmcv 1.3.9
   * mmsegmentation 0.15.0 (不用专门安装，git clone他给的仓库，然后安装即可)
3. 将数据准备成Pascal Voc 2012的格式，并且软连接到`data/`文件夹下
4. 主要修改的文件：
   *  `configs/_base_/datasets/pascal_voc12_water.py`：修改训练，验证以及测试文件的路径
   *  `configs/_base_/models/setr_pup_single.py`：定义SETR网络结构的配置，此文件相比于原始文件`configs/_base_/models/setr_pup_single.py`修改了第5行，将`SyncBN`改为了`BN`用于单卡训练
   *  `configs/_base_/models/deeplabv3plus_r50-d8_single.py`：修改类似于SETR的网络结构
   *  `configs/setr/setr_pup_512x512_40k_b2_water.py`：SETR训练以及测试的配置文件，修改分割类别数量，优化器，batchsize等
   *  `configs/deeplabv3plus/deeplabv3plus_r18-d8_512x512_40k_water.py`：Deeplabv3plus训练以及测试的配置文件，修改分割类别数量，优化器，batchsize等
   *  `tools/test.py`：修改了143、146和166、167行，添加了统计时间的代码
   *  `train.sh, train_deeplabv3plus_r18.sh`训练的脚本
   *  `test.sh, test_deeplabv3plus_r18.sh`测试的脚本
   *  `workdirs/`训练得到的模型、训练的log、测试时可视化的图像等

## SegFormer (3090服务器)

1. 代码及配置好环境的路径：`3090服务器 /mnt/projects/luoyt/SegFormer`
2. 代码执行需要的conda虚拟环境：`conda activate segformer`
   * pytorch 1.8.0+cuda11.1 (1.9会报错)
   * mmcv 1.2.7 (1.3.0以上版本会报错)
   * mmsegmentation 0.11.0 (不用专门安装，git clone他给的仓库，然后安装即可)
3. 将数据准备成Pascal Voc 2012的格式，并且软连接到`data/`文件夹下
4. 主要修改的文件：
   *  `local_configs/_base_/datasets/pascal_voc12_water.py`：修改训练，验证以及测试文件的路径
   *  `local_configs/_base_/models/segformer_single.py`：修改类似于SETR的网络结构
   *  `local_configs/segformer/B5/segformer.b5.512x512.pascalwater.40k.py`: SegFormer训练及测试的配置文件，修改分类类别数量，优化器，batchsize等
   *  `tools/test.py`：修改149、152和172、173行，增加了统计时间的代码
   *  `train.sh`：训练的脚本
   *  `test.sh`：测试的脚本。对于有label的数据评测mIoU，需要指定`--eval mIoU`，对于没有label的数据无法评测，需要指定`--eval None`。保存可视化结果的路径通过`--show-dir`指定

## Ultra Fast Lane Detection (1070服务器)

1. 代码及配置好环境的路径：`/home/klfy/projects/lyt/water_lane_detection_90`
2. 代码执行需要的conda虚拟环境：`conda activate deeplab`
3. 主要修改文件：
   * `configs/water.py`：配置文件，数据路径、优化器、batchsize、分类的列数等
   * `data/constant.py`：设置筛选的行
   * `data/dataset.py`：设置图像和label旋转操作，取消原本的延长车道线操作
   * `data/dataloader.py`：增加water数据集的dataloader
   * `model/model.py`：调整分类器线性层的大小来适应water数据集
   * 训练日志、模型保存在`../logs`里，按训练时间命名的目录
   * 训练：`python train.py configs/water.py`
   * 生成测试集可视化图像：`python demo.py configs/water.py --test_model logs/20210716_154909_lr_4e-04_b_32/ep029.pth`

## 逆投影算法

1. 包含两种方法，第一种论文Adaptive Inverse Perspective Mapping for Lane Map Generation with SLAM中使用的方法，在`inverse_perspective_mapping/Adaptive_IPM`路径下；第二种为基于单应性矩阵的方法，在`inverse_perspective_mapping/Homography_IPM`路径下
2. 第二种方法鲁棒性更好，具体可见该文件夹下的说明文档

