import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


def main():
    for datum in tf.data.TFRecordDataset(
        tf.io.matching_files(
            "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/train*.tfrecord*"
        ),
        compression_type="",
    ).as_numpy_iterator():
        frame = wod_e2ed_pb2.E2EDFrame()
        frame.ParseFromString(datum)
        print(frame)
        break


if __name__ == "__main__":
    main()
