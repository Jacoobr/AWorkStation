# coding:utf-8

'''
@author: xiaojianli
@sotfware: PyCharm Community Edition
@explanation: 
@time
'''
import collections
import tensorflow as tf

slim = tf.contrib.slim      #图像分类目录
'''
Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of *****batch normalization before every weight layer.*****
'v1'： batch  normalization *after* every weight layer.


reference: 
1. https://github.com/Jacoobr/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py 
2. https://github.com/Jacoobr/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py
'''
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    '''
    重命名的tuple来描述ResNet block
    Block, 一个重命名的tuple对象，是tuple的一种子类；#namedtuple('名称'， [属性list]);
    '''


def subsample(inputs, factor, scope=None):
    '''
    图像变小，通道数增加, pool_size: 1*1
    :param inputs: A tensor of size [batch, height_in, width_in, channels]
    :param factor: 降采样因子，factor==1 则对原图像不做任何操作，factor>1 则使用slim.max_pool2d最大池化; pooling stride=factor
    :param scope:
    :return:
    '''
    if factor == 1:
        return inputs;
    else:
        # 卷积核 ksize=[1,1], stride = factor, padding = scope, None即为padding=0
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    '''
    卷积函数定义: 先判断stride是否为1，如果是1则直接用slim.conv2d(), padding模式为SAME；

    inputs为一个四维的输入数据，shape:[batch, in_height, in_width, in_channels]; batch:训练时一个batch的图片数量； in_height: 特征图像高度；in_width:特征图像宽度；输入特征图像通道数
    filter也应为和inputs同类型的四维数据，shape:[filter_height, filter_width, in_channels, out_channels]
    padding=SAME 卷积前后特征图像大小不变
    padding=VALID 卷积前后特征图像变小
    :param inputs: 输入的特征图像
    :param num_outputs:  An integer, the number of output filters. **?
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param scope: padding大小
    :return:
    '''
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size -1  # pad zero的总数
        pad_beg = pad_total // 2    # pad//2
        pad_end = pad_total - pad_beg   # 其余部分为pad_end
        #pading的具体过程， 当padding!="SAME"时，需要先进性padding,再卷积
        #tf.pad()对输入进行补零操作
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])   #pad(tensor, paddings, mode, name) tensor: A `Tensor`. paddings: A `Tensor` of type `int32`. mode, name
        #padding = 'VALIAD' 实现卷积
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='VALID', scope=scope)

'''
堆叠Block函数:
'''
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    '''
    使用两个for循环逐个Residual Unit堆叠，拿到每个Block中每个Residual Unit. eg:[(256,64,1)]*2+[(256,64,2)]的args,
    并展开为：输出depth, 前两层输出depth_bottleneck和 中间层stride 的形式
    :param net: [batch, height, width, channels]
    :param blocks: Resnet中的残差单元;
    :param outputs_collections: 用来收集各个end_points的collections
    :return:

    variable_scope将bottleneck残差单元名称为:block/unit_%d
    unit_fn: 残差单元生成函数，并顺序创建和连接所有残差单元，
    slim.utils.collect_named_outputs函数将输出net添加到collection
    '''
    '''
        """Stacks ResNet `Blocks` and controls output feature density.

        First, this function creates scopes for the ResNet in the form of
        'block_name/unit_1', 'block_name/unit_2', etc.


        Args:
          net: A `Tensor` of size [batch, height, width, channels].
          blocks: A list of length equal to the number of ResNet `Blocks`. Each
            element is a ResNet `Block` object describing the units in the `Block`.
          outputs_collections: Collection to add the ResNet block outputs.

        Returns:
          net: Output tensor

        """
    '''
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc: #tf.variable_scope: 1.创建变量 2.为变量指定命名空间
            for i, unit in enumerate(block.args):   #enumerate返回枚举类型:i: index; unit: value
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_strid = unit;   #unit_depth_bottleneck: 残差单元深度/个数
                    net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck, stride=unit_strid) #残差单元生成函数，并顺序连接
            # 将输出添加到collection
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    #将所有Redidual Unit都堆叠完之后，返回net
    return net

'''
ResNet通用的arg_scope，定义某些函数的参数默认值
先设置好BN的各项参数,然后通过slim.arg_scope将slim.conv2d的几个默认参数设置好:
'''
def resnet_arg_scope(is_training=True, weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''
    :param is_training: 在模型的BN层是否要训练参数
    :param weight_decay: 权重衰减速率 0.001 ；The weight decay to use for regularizing the model**?.
    :param batch_norm_decay: BN衰减率默认为0.997  ；The moving average decay when estimating layer activation statistics in batch normalization.
    :param batch_norm_epsilon: BN espilon 常量0.997, 见 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_utils.py
    :param batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the activations in the batch normalization layer.
    :return:
    '''
    #配置批标准化参数
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    '''
    通过slim.arg_scope将slim.conv2d默认参数 权重设置为L2正则式处理
    '''
    with slim.arg_scope(
        [slim.conv2d],
        weight_regularizer = slim.l2_regularizer(weight_decay), #l2正则化项
        weight_initializer = slim.variance_scaling_initializer(),   # 权重初始化器
        activation_fn = tf.nn.relu, #激活函数使用relu
        normalizer_fn = slim.batch_norm,    #批标准化器
        normalizer_params = batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            #对于pool1使用的是padding='SAME'即tf自动对原图像补零，使得池化前和池化后图片大小相同,对于密集型任务使得feature alignment特征定位更容易
            #但在'Deep Residual Learning for Image Recognition' kaiming He's paper中用的padding= 'VALID'
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for max_pool1. 最大池化padding模式设为SAME, You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    ''' bottleneck残差学习单元， 是Resnet v2中提到的Full Preactivation Residual Unit的一个变种，和v1中的残差学习单元主要区别如下：
    1.  use BN before each weights layers
    2.  对输入进行preactivation，而不是在卷积进行激活函数处理
    :param inputs: A tensor of size [batch, in_height, in_width, in_channels]
    :param depth: The depth of ResNet unit output.
    :param depth_bottleneck:    The depth of the bottleneck layers.(Residual Unit前两层的输出depth)
    :param stride:  Residual Unit中间层的卷积步长； the stride of ResNet unit, Determines the amount of downsampling of the units outputs
    :param rate:
    :param outputs_collections: 残差单元输出集合
    :param scope: unit名称
    :return:
    '''
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # 获取输出通道数out_channels, 即输入的最后一个维度
        # last_dimension(shape, min_rank=1)
        # parame shape: A 'TensorShape'
        # parame min_rank: Integer, minimum rank of shape
        # returns: the value of last dimension

        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)    #最少维度为4
        # do BP before nonlinearity(Relu) preactivation
        #使用ReLU预激活Preactivate
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='scope')
        #定义shorcut,如果残差单元的in_channels(或depth_in)和输出通道数out_channels一致，那么使用subsample
        #按stride对inputs进行空间上的降采样(确保输入空间和残差空间空间尺寸一致，因为残差中间那层的卷积步长为stride))
        #如果输入通道数和输出通道数不一致，则用strid的1*1卷积改变其通道数，使得输入和输入通道数保持一致

        if depth == depth_in:
            # 同等映射（shortcut）
            shortcut = subsample(inputs, stride, 'shorcut')
        else:
            # shortcut直连的x
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')
            # residual, 这里residual有3层，第一层 ksize:1*1, stride=1, 输出通道数为depth_bottleneck的卷积
            #                              第二层 ksize:3*3, stride=1
            #                              第三层 ksize:1*1, stride=1（没BN，没nonlinearity激活函数）
            # 将residual与shorcut相加， 将output添加到collection
            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
            residual = slim.con2d_same(residual, depth_bottleneck, [3, 3], stride=1, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fu=None, scope='conv3')
            output = residual+shortcut
            return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None):
    '''
    v2 preactivation ResNet model, See the resnet_v2_*() methods for specific model instantiations
     obtained by selecting different block instantiations that produce ResNets of various depths.
     不同的block对应不同的网络深度
    :param inputs:
    :param blocks:
    :param num_classes:分类任务的类数，如果值为None则在logit layer之前返回特征
    :param global_pool: 指明网络最后是否需要添加全局平均池化层
    :param include_root_block: 值为True，则说明需要在initial convolution之后加上7*7 max_pooling layer;值为False, 则忽略该max_pooling layer, 且 'inputs'
                                should be the results of an activation-less convolution
    :param reuse:指明网络或者its variables 是否被重用，如果重用，则必须给出scope参数的值
    :param scope: 整个网络的名称
    :return:
     net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, //*then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,*?//
            else both height_out and width_out equal one. **If num_classes is None, then
            net is the output of the last ResNet block, potentially after global
            average pooling.** If num_classes is not None, net contains the pre-softmax
            activations.（当目标空间的待分类类数目给出时，网络包含一个pre-softmax的一个分类器）
          end_points: A dictionary from components of the network to the corresponding
            activation.
    Raises:
          ValueError: If the //*target output_stride*// is not valid.
    '''
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
            # We do not include batch normalization or activation functions in conv1
            # because the first ResNet unit will perform these. Cf. Appendix of [2].
            # 创建ResNet最前面的64输出通道的步长为2的7*7conv, 然后再接一个步长为2的3*3 max_pooling, 两个步长为2的层，特征图尺寸
            #已经被缩小为1/4
                with slim.arg_scope([slim.conv2d], activation_fu=None, normalizer=None):
                    net=conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net=stack_blocks_dense(net, blocks)
            # This is needed because the pre-activation variant does not have batch
            # normalization or activation functions in the residual unit output. See
            # Appendix of [2].
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                #Global avarge pooling
                net=tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                #输出通道为num_classes的1*1卷积（无激活函数，无正则项）
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            #将end_points_collection转化为 dictionary of end_points
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                #Softmax分类函数输出预测结果
                end_points['predictions']   = slim.softmax(net, scope='predictions')
            return net, end_points

##设计层数分别为50 101 152 200 的ResNet

def resnet_v2_50(inputs,
                     num_classes=None,
                     global_pool=True,
                     reuse=None,
                     scope='resnet_v2_50'):
        """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
        ## 4个 Residul unit Blocks, units分量分别是3 4 6 和3， 总层数为：（3+4+6+3）*3+2=50
        ## 网络总共使得input_size: 224*224 缩小4*2^3=32陪， 即最后out_size: 224/32 * 224/32
        ## 不断使用stride = 2的层来缩减尺寸， 同时out_channels在不断增加，最终为1*1, 2048
        blocks = [
            Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
        return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_101(inputs,
                      num_classes=None,
                      global_pool=True,
                      reuse=None,
                      scope='resnet_v2_101'):
        """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),  # the key change
            Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
        return resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),  # the key change
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                      include_root_block=True, reuse=reuse, scope=scope)

##使用time_tensor_run来测试152层的ResNet的forward的性能
##图片回归到AlexNet, VGGNet的 224*224， batch_size设为32， is_training设为False
##使用resnet_v2_152创建网络
##后续工作可以测评求解ResNet全部参数梯度所需时间
from datetime import datetime
import math
import time
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    num_batches = 100
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

    batch_size = 32
    height, width = 224, 224
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net, end_points = resnet_v2_152(inputs, 1000)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    time_tensorflow_run(sess, net, "Forward")