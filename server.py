'''Chinese name recognition server.
The service returns a score for a word of 2 or 3 Chinese characters.
The more positive the score is (mostly in range -20 to +20), the more
likely the word is a Chinese name.
Caveats:
    * If the given word is shorter than 2 or longer than 3, a nan is returned.
    * If the given word contains non-Chinese or very rare characters, the
    score may be unreasonably large.
'''
from concurrent import futures
import math
import time
import grpc
import test
from train import Train
from eval import Eval
import cnn_model_pb2
import cnn_model_pb2_grpc
import tensorflow as tf
import sys
_ONE_DAY_IN_SECONDS = 3600 * 24

tf.flags.DEFINE_string("mode", "pred", "train, eval, pred")
tf.flags.DEFINE_string("checkpoint_dir", "", "dir of the model to be restored")
tf.flags.DEFINE_string("data_file", "query_app.txt", "the data file for model")

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


class CnnModelServicer(cnn_model_pb2_grpc.CnnModelServicer):
    def __init__(self):
        self.predictor = test.Pred(FLAGS)
        self.predictor.restore_model()

    def ScoreCnnModel(self, request, context):
        query = request.query.strip()
        score = self.predictor.pred(query)
        for app,prob in score:
            sl = cnn_model_pb2.ScoreList()
            sl.query = app
            sl.prob = prob
            yield cnn_model_pb2.QueryReply(scorelist=sl)


def serve():

    if FLAGS.mode == "train":

        if not FLAGS.data_file:
            print('train_data is required to be provided to train model')
            sys.exit(-1)
        train = Train(FLAGS)
        train.train()
    if FLAGS.mode == "eval":

        if not FLAGS.data_file:
            print('valid_data is required to be provided to valid model')
            sys.exit(-1)
        eval = Eval(FLAGS)
        eval.load_data(FLAGS.data_file)
        eval.eval()

    if FLAGS.mode == "pred":
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        cnn_model_pb2_grpc.add_CnnModelServicer_to_server(
            CnnModelServicer(), server)

        server.add_insecure_port('[::]:50051')
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    serve()