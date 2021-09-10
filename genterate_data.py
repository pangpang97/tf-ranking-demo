import tensorflow as tf
from tensorflow_serving.apis import input_pb2

import numpy as np

def gen_rawdata():
    #user_age, user_gender,user_pay, package_type, package_price,item_type, item_cnt ,label
    #user and package feature:
    #user info
    n_user = 20 # n个用户
    user_age = np.arange(1,n_user+1)
    user_gender = np.random.randint(0,3,n_user)
    user_pay = np.random.random(n_user)

    user = np.c_[user_age,user_gender,user_pay]

    #package info
    package_type_set = np.array([1,3,5])
    package_price_set = np.array([5,20,50,99])
    package_type = np.repeat(package_type_set,len(package_price_set))
    package_price = np.tile(package_price_set, len(package_type_set))

    package_type_c = np.tile(package_type,n_user)
    package_price_c = np.tile(package_price,n_user)

    package_cnt = len(package_type_set)*len(package_price_set)
    user_c = np.repeat(user, package_cnt, axis=0)

    #combine
    context = np.c_[user_c, package_type_c,package_price_c]
    context = np.repeat(context, 3, axis=0)
    #item feature
    item_type = np.random.randint(0,10,context.shape[0])
    item_cnt = np.random.randint(0,500, context.shape[0])
    #label
    label =  np.random.randint(0,2,context.shape[0])
    all_data = np.c_[context,item_type, item_cnt,label]
    print(all_data.shape)
    feature_list = []
    for i in range(all_data.shape[1]):
        if i != 2:
            arr = all_data[:,i].astype(np.int32)
            feature_list.append(arr)
        else:
            arr = all_data[:,i].astype(np.float32)
            feature_list.append(arr)
    return feature_list

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def gen_ELWCsample(user_age, user_gender,user_pay, package_type, package_price,item_type, item_cnt ,label):

    # context feature: user_age, user_gender,user_pay, package_type, package_price,
    # item feature: item_type, item_cnt ,label
    context_dict = {}
    context_dict['user_age'] = _int64_feature(user_age)
    context_dict['user_gender'] =_int64_feature(user_gender)
    context_dict['user_pay'] = _float_feature(user_pay)
    context_dict['package_type'] = _int64_feature(package_type)
    context_dict['package_price'] = _int64_feature(package_price)
    context_proto = tf.train.Example(features=tf.train.Features(feature=context_dict))

    ELWC = input_pb2.ExampleListWithContext()
    ELWC.context.CopyFrom(context_proto)

    example_features = ELWC.examples.add()
    example_dict = {}
    example_dict['item_type'] = _int64_feature(item_type)
    example_dict['item_cnt'] = _int64_feature(item_cnt)
    example_dict['relevance'] = _int64_feature(label)
    exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
    example_features.CopyFrom(exampe_proto)
    return ELWC



if __name__ == '__main__':
    raw_data = gen_rawdata()
    with tf.io.TFRecordWriter('train.tfrecords') as writer:
        for i in range(len(raw_data[0])):
            user_age, user_gender, user_pay, package_type, package_price, item_type, item_cnt, label = raw_data[0][i],\
                                                                                                       raw_data[1][i],\
                                                                                                       raw_data[2][i],\
                                                                                                       raw_data[3][i],\
                                                                                                       raw_data[4][i],\
                                                                                                       raw_data[5][i],\
                                                                                                       raw_data[6][i],\
                                                                                                       raw_data[7][i]
            elwc_sample = gen_ELWCsample(user_age, user_gender, user_pay, package_type, package_price, item_type, item_cnt, label )
            if i == 0:
                elwc_sample_0 = elwc_sample
            writer.write(elwc_sample.SerializeToString())
    print(elwc_sample_0)