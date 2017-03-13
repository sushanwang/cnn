"""
Query classification server.
This server takes a query and returns a list of top5 app name with its probability.
"""
from concurrent import futures
import time
import grpc
import pred
import main
from train import Train
from eval import Eval
import cnn_model_pb2
import cnn_model_pb2_grpc
import sys
_ONE_DAY_IN_SECONDS = 3600 * 24
FLAGS = main.get_flags()


class CnnModelServicer(cnn_model_pb2_grpc.CnnModelServicer):
    def __init__(self):
        self.predictor = pred.Pred(FLAGS)
        self.predictor.restore_model()

    def ScoreCnnModel(self, request, context):
        query = request.query.strip()
        score = self.predictor.predict(query)
        app_list = []
        prob_list = []
        for app,prob in score:
            app_list.append(app)
            prob_list.append(prob)
        return cnn_model_pb2.QueryReply(app=app_list, prob=prob_list)


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