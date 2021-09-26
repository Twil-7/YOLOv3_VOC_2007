# YOLOv3_VOC_2007

# 环境配置

python == 3.8

tensorflow == 2.4.1

keras == 2.4.3

# 运行

先下载权重文件，如果是python3.6版本，加载yolo3_py3.6_weights.h5；如果是python3.8版本，加载yolo3_py3.8_weights.h5。将这两个权重文件放入model_data/文件夹下。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

Annotations、JPEGImages数据集下载：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

第1步：运行annotation.py，划分数据集写入model_data/txt；

第2步：运行train.py，搭建YOLOv3模型，进行训练；

第3步：yolo_predict.py，调用训练好的权重文件，进行目标检测。

该代码稍做修改也可在python 3.6版本下运行，如果要切换到python 3.6版本进行训练，需要修改该句语法：

 _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
 
 _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])


# 实验效果

先采用1e-3的学习率训练50轮，待到模型参数较佳时，再解开所有冰冻层，将学习率改为1e-4，达到最佳权重效果，val loss = 19.7083。

对测试集的图片进行目标检测，效果十分惊艳，不光分类定位准确，而且轮廓信息也非常精准。

神奇的是，只有上述训练方式才能取得这种最佳权重效果。如果最开始就解冻所有层，将训练得到NAN。如果不解冻，仅仅使用1e-3不断训练， val loss达到瓶颈时的效果很差，之后也无法再继续使loss下降。
