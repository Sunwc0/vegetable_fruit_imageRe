# vegetable_fruit_imageRe

利用Google开源的Inception_resnet_v2模型实现了['土豆','圣女果','芒果','韭菜','大葱','大白菜','香蕉','胡萝卜','梨','黄瓜','西红柿','苹果']12个类别的水果蔬菜识别问题，模型精度达到85%以上<br>

## 图片预处理：<br>
    >>trannsform.py 将图片处理成299X299的分辨率<br>
    >>tfrecord.py 将图片样本先随机，然后生成模型的标准输入文件Tfrecords形式<br>

## 训练：<br>
    >> new_train.py 作为训练入口<br>
    >> aliyun_train.py 阿里云训练文件，具体阿里云训练参考阿里云教程<br>
