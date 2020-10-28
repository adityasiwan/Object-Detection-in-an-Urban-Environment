import tensorflow as tf



raw_dataset = tf.data.TFRecordDataset("/app/project/backups/raw/segment-11355519273066561009_5323_000_5343_000_with_camera_labels.tfrecord")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)