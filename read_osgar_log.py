import sys
import cv2
import numpy as np
from osgar.logger import LogReader, lookup_stream_id
from osgar.lib.serialize import deserialize

log_file = "test-ip-camera-220519_162450.log"
# stream_name_color = "D455.color"
# stream_name_gps = "gps.position"
# stream_name_depth = "D455.depth"
#
# only_stream_color = lookup_stream_id(log_file, stream_name_color)
# only_stream_gps = lookup_stream_id(log_file, stream_name_gps)
# only_stream_depth = lookup_stream_id(log_file, stream_name_depth)

log = LogReader(log_file)

idx = 0
with LogReader(log_file) as log:
    for timestamp, stream_id, data in log:
        if stream_id == 1:
            buf_color = deserialize(data)
            color_im = cv2.imdecode(np.frombuffer(buf_color, dtype=np.uint8), 1)

            cv2.imwrite("img_{}.jpg".format(idx), color_im)
            idx += 1
            cv2.imshow("img", color_im)
            cv2.waitKey(5)



#         if stream_id == only_stream_color:
#             buf_color = deserialize(data)
#             color_im = cv2.imdecode(np.frombuffer(buf_color, dtype=np.uint8), 1)
#             print("color", color_im.shape)
#
#         if stream_id == only_stream_gps:
#             gps = deserialize(data)
#             print("gps", gps)
#
#         if stream_id == only_stream_depth:
#             buf_depth = deserialize(data)
#             depth_data = np.array(buf_depth, np.int16)
#             print("depth", depth_data.shape)
#
