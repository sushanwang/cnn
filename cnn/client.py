import grpc

import cnn_model_pb2_grpc
import cnn_model_pb2
import sys


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = cnn_model_pb2_grpc.CnnModelStub(channel)
    for query in sys.stdin:
        result = cnn_model_pb2.QueryRequest(query=query)
        replies = stub.ScoreCnnModel(result)
        replies_len = len(replies.app)
        for i in range(replies_len):
            print("the app name is %s with prob %.6f" % (replies.app[i], replies.prob[i]))


if __name__ == '__main__':
    run()