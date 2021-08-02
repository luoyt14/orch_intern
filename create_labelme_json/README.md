# 生成labelme格式的json

1. 使用训练好的模型生成分割图，见Picture_test_segformer文件夹
2. 使用create_label.py生成json格式的文件，json保存在与图片相同的文件夹下，见Picture_test文件夹
3. 命令行输入：`labelme Picture_test --labels labels.txt --nodata --validatelabel exact --config '{shift_auto_shape_color: -2}'` 即可打开labelme查看对应的标记并进行微调