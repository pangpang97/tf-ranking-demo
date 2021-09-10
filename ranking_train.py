import tensorflow as tf
import tensorflow_ranking as tfr
import sys

_SIZE = 'example_list_size'  # Name of feature of example list sizes.
_LABEL_FEATURE = 'relevance'
_PADDING_LABEL = -1


def create_context_feature_columns():
    context_feature_columns = {}
    #all context feature embedding

    user_age_embedding = tf.feature_column.numeric_column('user_age', dtype=tf.int64, default_value=0)
    user_gender_embedding = tf.feature_column.numeric_column('user_gender', dtype=tf.int64, default_value=0)
    user_pay_embedding = tf.feature_column.numeric_column('user_pay', dtype=tf.float32, default_value=0)
    package_type_embedding = tf.feature_column.numeric_column('package_typer', dtype=tf.int64, default_value=0)
    package_price_embedding = tf.feature_column.numeric_column('package_price', dtype=tf.int64, default_value=0)

    context_feature_columns['user_age'] = user_age_embedding
    context_feature_columns['user_gender'] = user_gender_embedding
    context_feature_columns['user_pay'] = user_pay_embedding
    context_feature_columns['package_type'] = package_type_embedding
    context_feature_columns['package_price'] = package_price_embedding

    return context_feature_columns



def create_example_feature_columns():
    example_feature_columns = {}
    item_type_embedding = tf.feature_column.numeric_column('item_type', dtype=tf.int64, default_value=0)
    item_cnt_embedding = tf.feature_column.numeric_column('item_cnt', dtype=tf.int64, default_value=0)

    example_feature_columns['item_type'] = item_type_embedding
    example_feature_columns['item_cnt'] = item_cnt_embedding

    return example_feature_columns

def make_dataset(file, batch_size, randomize_input=True, num_epochs=None):
    context_feature_columns = create_context_feature_columns()
    example_feature_columns = create_example_feature_columns()
    label_embedding = tf.feature_column.numeric_column(_LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)

    context_feature_spec = tf.feature_column.make_parse_example_spec(context_feature_columns.values())
    print('context_feature_spec',context_feature_spec)
    example_feature_spec = tf.feature_column.make_parse_example_spec(list(example_feature_columns.values()) + [
        label_embedding])
    print('example_feature_spec',example_feature_spec)
    dataset = tfr.data.build_ranking_dataset(file_pattern=file,
                                            data_format=tfr.data.ELWC,
                                            batch_size=batch_size,
                                            context_feature_spec=context_feature_spec,
                                            example_feature_spec=example_feature_spec,
                                            reader=tf.data.TFRecordDataset,
                                            shuffle=randomize_input,
                                            num_epochs=num_epochs,
                                            size_feature_name=_SIZE)

    def _separate_features_and_label(features):
        #print(features)
        #tf.print('feature',features,output_stream=sys.stdout)
        label = tf.compat.v1.squeeze(features.pop(_LABEL_FEATURE), axis=2)
        label = tf.cast(label, tf.int64)
        #tf.print('label',label,output_stream=sys.stdout)
        return features, label

    dataset = dataset.map(_separate_features_and_label)
    return dataset


context_feature_columns = create_context_feature_columns()
example_feature_columns = create_example_feature_columns()
#Build a network it is a pointwise model
network = tfr.keras.canned.DNNRankingNetwork(context_feature_columns=context_feature_columns,
                                             example_feature_columns=example_feature_columns,
                                             hidden_layer_dims=[32, 16],
                                             activation=tf.nn.relu,
                                             dropout=0.5)
#loss
softmax_loss_obj = tfr.keras.losses.get(tfr.losses.RankingLossKey.SOFTMAX_LOSS)
#metric
default_metrics = tfr.keras.metrics.default_keras_metrics()
#optimizer
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
#build ranker by Keras
ranker = tfr.keras.model.create_keras_model(network=network,
                                            loss=softmax_loss_obj,
                                            metrics=default_metrics,
                                            optimizer=optimizer,
                                            size_feature_name=_SIZE)


train_dataset = make_dataset("train.tfrecords", 1, num_epochs=-1)
val_dataset = make_dataset("train.tfrecords", 1, num_epochs=-1)


for parsed_record in train_dataset.take(2):
  print(repr(parsed_record))


ranker.fit(train_dataset,
           validation_data=val_dataset,
           steps_per_epoch=2,
           epochs=5,
           validation_steps=5)


#predict


'''
   user_age_column = tf.feature_column.categorical_column_with_identity(key='user_age',num_buckets=20,default_value=0)
   user_age_embedding = tf.feature_column.embedding_column(categorical_column=user_age_column, dimension=4)

   user_gender_column = tf.feature_column.categorical_column_with_vocabulary_list('use_gender',[0,1,2])
   user_gender_embedding = tf.feature_column.embedding_column(categorical_column=user_gender_column, dimension=4)

   user_pay_embedding = tf.feature_column.numeric_column('user_pay', dtype=tf.float32, default_value=0)

   package_type_column = tf.feature_column.categorical_column_with_vocabulary_list('package_type',[1,3,5])
   package_type_embedding = tf.feature_column.embedding_column(categorical_column=package_type_column, dimension=4)

   package_price_column = tf.feature_column.categorical_column_with_vocabulary_list('package_price',[5,20,50,99])
   package_price_embedding = tf.feature_column.embedding_column(categorical_column=package_price_column, dimension=4)
   '''
'''
    item_type_column = tf.feature_column.categorical_column_with_identity(key='item_type',num_buckets=9,default_value=0)
    item_type_embedding = tf.feature_column.embedding_column(categorical_column=item_type_column, dimension=16)
'''