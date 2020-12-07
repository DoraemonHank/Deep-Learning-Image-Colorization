import cv2
import numpy as np
import tensorflow as tf
import utils as utils
from vgg import vgg16
from model_vgg import ResidualDecoder

saved_ckpt_path = './checkpoint/'
BATCH_SIZE = 1
HIGHT = 224
WIDTH = 224
CHANNEL = 3
VIDEO_PATH = './video/'
VIDEO_NAME = 'videoplayback.mp4'
SAVE_VIDEO_PATH = './video/'

print("🤖 Load vgg16 model...")
vgg = vgg16.Vgg16()

# Build residual encoder model
print("🤖 Build residual encoder model...")
residual_decoder = ResidualDecoder()

x = tf.placeholder(tf.float32, [BATCH_SIZE, HIGHT, WIDTH, CHANNEL], name='x_input')
is_training = tf.placeholder(tf.bool, name="is_training")

vgg.build(x)
output = residual_decoder.build(input_data=x, vgg=vgg, is_training=is_training)

cap = cv2.VideoCapture(VIDEO_PATH + VIDEO_NAME)

fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2,
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(SAVE_VIDEO_PATH + 'output_' + VIDEO_NAME, fourcc, fps, size)

with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    while(True):
      ret, frame = cap.read()
      if(ret == False):
          break
      src = frame.copy()
      src_resize = cv2.resize(src,(WIDTH,HIGHT))
      img_gray = cv2.cvtColor(src_resize, cv2.COLOR_BGR2GRAY)
      img_gray_three_channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
      input = np.array([img_gray_three_channel.astype(np.float)/255]).astype(np.float32)

      predict = sess.run(output,feed_dict={x:input,is_training: False})
      result = utils.get_image(input[0],predict[0])
      result = cv2.resize(result,(frame.shape[1],frame.shape[0]))
      
      cv2.imshow('result', np.hstack([result,src]))
      out.write(np.hstack([result,src]))
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        
    cap.release()
    cv2.destroyAllWindows()

